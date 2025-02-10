import pandas as pd
import json
from openai import OpenAI
import os

def generate_cluster_labels(cluster_file="clustered_arguments.csv", output_file="cluster_labels.json"):
    """
    クラスタリング結果からクラスタラベルを生成する
    
    Parameters:
    -----------
    cluster_file : str
        クラスタリング結果のCSVファイル
    output_file : str
        出力するJSONファイル名
    """
    # クラスタリング結果の読み込み
    df = pd.read_csv(cluster_file)
    
    # OpenAI APIクライアントの初期化
    client = OpenAI()
    
    # クラスタごとのラベル生成
    cluster_labels = []
    
    for cluster_id in sorted(df["cluster-id"].unique()):
        # クラスタ内のテキストを取得
        cluster_texts = df[df["cluster-id"] == cluster_id]["argument"].tolist()
        
        # プロンプトの作成
        prompt = f"""以下の意見グループを分析し、JSONフォーマットで回答してください：

意見グループ：
{chr(10).join(cluster_texts)}

必要な情報：
1. label: このグループを代表する短いラベル（30文字以内）
2. description: このグループの意見の共通点や特徴の説明（200文字以内）
3. keywords: 主要なキーワード（3-5個）
4. sentiment: 意見の全体的なトーン（"positive", "negative", "neutral"のいずれか）

回答フォーマット：
{{
    "label": "グループのラベル",
    "description": "グループの説明",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"],
    "sentiment": "意見のトーン"
}}"""
        
        # GPT-4による分析
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # 結果の解析
        result = json.loads(response.choices[0].message.content)
        result["cluster_id"] = cluster_id
        result["size"] = len(cluster_texts)
        result["texts"] = cluster_texts
        
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
