import pandas as pd
import json
import numpy as np
from openai import OpenAI
import os

# Base prompt for label generation with domain context
BASE_PROMPT = """質問: AIと著作権に関する意見を分析し、各クラスタの特徴を把握する

クラスタ外部の意見：
{outside_texts}

クラスタ内部の意見：
{inside_texts}

# 指示
1. クラスタ内の意見とクラスタ外の意見を比較してください
2. クラスタ内の意見に共通する特徴を抽出してください
3. AIと著作権に関する具体的な論点を反映したラベルを生成してください
4. 一般的な表現（"意見グループ"など）は避けてください

# 出力形式
{
    "label": "具体的で意味のあるラベル（30文字以内）",
    "description": "意見の共通点や特徴の説明（200文字以内）",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"],
    "sentiment": "positive/negative/neutral"
}"""

# Validation prompt for quality check
VALIDATION_PROMPT = """生成されたラベルと意見グループを確認し、ラベルの質を評価してください：

# ラベル
{label}

# 説明
{description}

# 意見例
{examples}

# 指示
1. ラベルが意見グループの本質を捉えているか確認
2. より具体的なラベルの提案（必要な場合）
3. 代表的な意見の選択

# 出力形式
{
    "is_valid": true/false,
    "improved_label": "改善案（必要な場合）",
    "representative_ids": ["id1", "id2", "id3"]
}"""

def get_outside_cluster_samples(df, current_cluster_id, sample_size=5):
    """クラスタ外の意見をサンプリング"""
    outside_texts = df[df["cluster_id"] != current_cluster_id]["argument"].sample(
        n=min(sample_size, len(df[df["cluster_id"] != current_cluster_id]))
    ).tolist()
    return "\n".join([f"* {text}" for text in outside_texts])

def validate_label_quality(label, texts):
    """
    ラベルの質を検証し、必要に応じて改善
    
    Parameters:
    -----------
    label : str
        検証するラベル
    texts : list
        クラスタ内のテキストリスト
    
    Returns:
    --------
    tuple(bool, str)
        (検証結果, 失敗理由)
    """
    # 一般的なラベルのパターン
    generic_patterns = [
        "グループのラベル",
        "意見グループ",
        "サンプルラベル",
        "ラベル例",
        "未定義グループ"
    ]
    
    # 検証基準
    if any(pattern in label for pattern in generic_patterns):
        return False, "一般的すぎるラベル"
    
    if len(label) < 5:
        return False, "ラベルが短すぎる"
    
    if len(label) > 30:
        return False, "ラベルが長すぎる"
        
    domain_keywords = ["AI", "著作権", "権利", "創作", "生成", "モデル", "学習", "データ"]
    if not any(keyword in label for keyword in domain_keywords):
        return False, "ドメイン固有の単語が含まれていない"
    
    return True, None

def validate_label(client, label_info, examples):
    """ラベルの質を検証"""
    # まず基本的な検証を実行
    is_valid, reason = validate_label_quality(label_info["label"], examples)
    if not is_valid:
        print(f"\n警告: ラベル「{label_info['label']}」は無効です（理由: {reason}）")
    
    # LLMによる詳細な検証を実行
    validation_content = VALIDATION_PROMPT.format(
        label=label_info["label"],
        description=label_info["description"],
        examples="\n".join([f"* {ex}" for ex in examples])
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": validation_content}],
        temperature=0.4,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def preprocess_clusters(df):
    """
    クラスタデータの前処理
    - 空のクラスタを除外
    - 最小サイズ要件の確認
    - テキストの正規化
    
    Parameters:
    -----------
    df : pd.DataFrame
        クラスタリング結果のDataFrame
    
    Returns:
    --------
    pd.DataFrame
        前処理済みのDataFrame
    """
    print("\nクラスタの前処理を開始...")
    original_size = len(df)
    
    # Group by cluster_id and filter clusters with too many empty arguments
    cluster_groups = df.groupby("cluster_id")
    valid_clusters = []
    
    for cluster_id, group in cluster_groups:
        valid_args = group[group["argument"].notna() & (group["argument"] != "")]
        if len(valid_args) >= 5:  # クラスタ内の有効な意見が5件以上
            valid_clusters.append(cluster_id)
    
    df = df[df["cluster_id"].isin(valid_clusters)]
    df = df[df["argument"].notna() & (df["argument"] != "")]  # 最終的に空の意見を除外
    
    after_size = len(df)
    print(f"- 有効なクラスタ数: {len(valid_clusters)}件")
    print(f"- 最終的なデータサイズ: {after_size}件（元の{original_size}件から）")
    
    return df

def generate_cluster_labels(cluster_file="clustered_arguments.csv", output_file="cluster_labels.json"):
    """
    クラスタリング結果からクラスタラベルを生成する（2段階アプローチ）
    
    1. クラスタ内外の意見を比較してラベルを生成
    2. 代表的な意見を選択して検証
    
    Parameters:
    -----------
    cluster_file : str
        クラスタリング結果のCSVファイル
    output_file : str
        出力するJSONファイル名
    """
    # クラスタリング結果の読み込み
    df = pd.read_csv(cluster_file)
    
    # クラスタの前処理
    df = preprocess_clusters(df)
    
    # OpenAI APIクライアントの初期化
    client = OpenAI()
    
    # クラスタごとのラベル生成
    cluster_labels = []
    cluster_ids = sorted(df["cluster_id"].unique())
    total_clusters = len(cluster_ids)
    
    print(f"\n全{total_clusters}クラスタのラベル生成を開始...")
    for i, cluster_id in enumerate(cluster_ids, 1):
        print(f"\rクラスタ {i}/{total_clusters} のラベルを生成中...", end="")
        
        # クラスタ内のテキストを取得
        cluster_texts = df[df["cluster_id"] == cluster_id]["argument"]
        cluster_texts = cluster_texts.fillna("").astype(str).tolist()
        
        # クラスタ外のテキストを取得
        outside_texts = get_outside_cluster_samples(df, cluster_id)
        
        # Step 1: Generate initial label with retry logic
        max_retries = 3
        result = None
        validation_reason = None
        
        for attempt in range(max_retries):
            prompt = BASE_PROMPT.format(
                outside_texts=outside_texts,
                inside_texts="\n".join([f"* {text}" for text in cluster_texts])
            )
            
            if attempt > 0 and validation_reason:
                # 2回目以降は、より具体的な指示を追加
                prompt += f"\n\n# 注意\n前回の生成結果が無効でした。より具体的で、AIと著作権に関する論点を反映したラベルを生成してください。\n失敗理由: {validation_reason}"
            
            # GPT-4oによる分析（低いtemperatureで一貫性を重視）
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,  # 一貫性を重視
                response_format={"type": "json_object"}
            )
            
            # 結果の解析
            result = json.loads(response.choices[0].message.content)
            
            # Step 2: Validate and improve label
            validation_result = validate_label(client, result, cluster_texts[:5])
            
            # 検証結果の確認
            if validation_result["is_valid"]:
                break
            
            validation_reason = validation_result.get("improved_label", "ラベルが要件を満たしていません")
            print(f"\n警告: 試行 {attempt + 1}/{max_retries} が失敗しました。理由: {validation_reason}")
            
            if attempt == max_retries - 1 and validation_result.get("improved_label"):
                # 最終試行でも失敗した場合は改善案を採用
                result["label"] = validation_result["improved_label"]
                print(f"\n注意: 最終的に改善案を採用: {validation_result['improved_label']}")
        
        # メタデータの追加
        result["cluster_id"] = int(cluster_id)
        result["size"] = int(len(cluster_texts))
        result["texts"] = [str(text) for text in cluster_texts]
        if validation_result.get("representative_ids"):
            result["representative_ids"] = validation_result["representative_ids"]
        
        cluster_labels.append(result)
    
    # 結果をJSONファイルに保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cluster_labels, f, ensure_ascii=False, indent=2)
    
    return cluster_labels

def format_labels_report(labels, output_file="cluster_labels_report.md"):
    """
    クラスタラベルの結果をMarkdownレポートとして整形
    
    Parameters:
    -----------
    labels : list
        generate_cluster_labelsの出力結果
    output_file : str
        出力するMarkdownファイル名
    """
    report = ["# クラスタラベル生成結果\n"]
    
    # センチメント別の集計
    sentiment_counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    
    for label in labels:
        sentiment_counts[label["sentiment"]] += 1
        
        # クラスタ情報の追加
        report.append(f"## クラスタ {label['cluster_id']} (サイズ: {label['size']})")
        report.append(f"- ラベル: {label['label']}")
        report.append(f"- 説明: {label['description']}")
        report.append(f"- キーワード: {', '.join(label['keywords'])}")
        report.append(f"- センチメント: {label['sentiment']}\n")
        report.append("### 代表的な意見:")
        for text in label["texts"][:3]:  # 最初の3つの意見のみ表示
            report.append(f"- {text}")
        report.append("\n")
    
    # 集計情報の追加
    report.insert(1, "## 概要")
    report.insert(2, f"- 総クラスタ数: {len(labels)}")
    report.insert(3, "- センチメント分布:")
    for sentiment, count in sentiment_counts.items():
        report.insert(4, f"  * {sentiment}: {count}")
    report.insert(5, "\n")
    
    # レポートの保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    # クラスタラベルの生成
    labels = generate_cluster_labels()
    
    # レポートの生成
    format_labels_report(labels)
    
    print("クラスタラベルの生成が完了しました。")
    print("- cluster_labels.json: 生成されたラベル（JSON形式）")
    print("- cluster_labels_report.md: 人間が読みやすい形式のレポート")
