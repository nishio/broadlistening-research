import json
from openai import OpenAI
import pandas as pd

def evaluate_cluster_labels(labels_file="cluster_labels.json", output_file="label_evaluation.json"):
    """
    クラスタラベルの質を評価する
    
    Parameters:
    -----------
    labels_file : str
        評価対象のクラスタラベルJSONファイル
    output_file : str
        評価結果の出力ファイル
    """
    # クラスタラベルの読み込み
    with open(labels_file, "r", encoding="utf-8") as f:
        cluster_labels = json.load(f)
    
    # OpenAI APIクライアントの初期化
    client = OpenAI()
    
    evaluations = []
    
    for label in cluster_labels:
        # 評価用プロンプトの作成
        prompt = f"""以下のクラスタラベルの質を100点満点で評価し、JSONフォーマットで回答してください。

評価対象：
1. ラベル: {label['label']}
2. 説明: {label['description']}
3. キーワード: {', '.join(label['keywords'])}
4. センチメント: {label['sentiment']}

クラスタ内の意見例（最初の3件）：
{chr(10).join(label['texts'][:3])}

評価基準：
1. 一貫性（30点）: ラベルと説明が意見群の内容を一貫して表現しているか
2. 具体性（25点）: ラベルと説明が具体的で明確か
3. 網羅性（25点）: 意見群の主要な論点を捉えているか
4. キーワードの適切性（20点）: キーワードが内容を適切に表現しているか

回答フォーマット：
{{
    "total_score": 評価の合計点（0-100の整数）,
    "consistency_score": 一貫性の点数（0-30の整数）,
    "specificity_score": 具体性の点数（0-25の整数）,
    "coverage_score": 網羅性の点数（0-25の整数）,
    "keyword_score": キーワードの適切性の点数（0-20の整数）,
    "feedback": "改善点や評価理由の説明（200文字以内）"
}}"""
        
        # GPT-4による評価
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # 評価の一貫性のため低めの温度を設定
        )
        
        # 評価結果の解析
        evaluation = json.loads(response.choices[0].message.content)
        evaluation["cluster_id"] = label["cluster_id"]
        evaluation["label"] = label["label"]
        evaluations.append(evaluation)
    
    # 評価結果の保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    
    return evaluations

def format_evaluation_report(evaluations, output_file="label_evaluation_report.md"):
    """
    評価結果をMarkdownレポートとして整形
    
    Parameters:
    -----------
    evaluations : list
        evaluate_cluster_labelsの出力結果
    output_file : str
        出力するMarkdownファイル名
    """
    report = ["# クラスタラベル評価結果\n"]
    
    # 全体の統計
    scores = [e["total_score"] for e in evaluations]
    avg_score = sum(scores) / len(scores)
    
    report.append("## 評価概要")
    report.append(f"- 評価対象クラスタ数: {len(evaluations)}")
    report.append(f"- 平均スコア: {avg_score:.1f}")
    report.append(f"- 最高スコア: {max(scores)}")
    report.append(f"- 最低スコア: {min(scores)}\n")
    
    # スコア分布
    score_ranges = {
        "90-100": 0,
        "80-89": 0,
        "70-79": 0,
        "60-69": 0,
        "0-59": 0
    }
    
    for score in scores:
        if score >= 90:
            score_ranges["90-100"] += 1
        elif score >= 80:
            score_ranges["80-89"] += 1
        elif score >= 70:
            score_ranges["70-79"] += 1
        elif score >= 60:
            score_ranges["60-69"] += 1
        else:
            score_ranges["0-59"] += 1
    
    report.append("### スコア分布")
    for range_name, count in score_ranges.items():
        report.append(f"- {range_name}点: {count}件")
    report.append("")
    
    # 各クラスタの評価詳細
    report.append("## 個別評価")
    for eval in sorted(evaluations, key=lambda x: x["total_score"], reverse=True):
        report.append(f"\n### クラスタ {eval['cluster_id']} ({eval['label']})")
        report.append(f"- 総合スコア: {eval['total_score']}")
        report.append("- 項目別スコア:")
        report.append(f"  * 一貫性: {eval['consistency_score']}/30")
        report.append(f"  * 具体性: {eval['specificity_score']}/25")
        report.append(f"  * 網羅性: {eval['coverage_score']}/25")
        report.append(f"  * キーワード: {eval['keyword_score']}/20")
        report.append(f"- フィードバック: {eval['feedback']}")
    
    # レポートの保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    # クラスタラベルの評価
    evaluations = evaluate_cluster_labels()
    
    # レポートの生成
    format_evaluation_report(evaluations)
    
    print("クラスタラベルの評価が完了しました。")
    print("- label_evaluation.json: 評価結果（JSON形式）")
    print("- label_evaluation_report.md: 人間が読みやすい形式のレポート")
