import pandas as pd
import json

def generate_comparison_report():
    """評価結果の比較レポートを生成"""
    try:
        # 評価結果の読み込み
        with open("experiments/results/hdbscan_evaluation.json", "r", encoding="utf-8") as f:
            hdbscan_eval = json.load(f)
        with open("experiments/results/dataset_x_evaluation.json", "r", encoding="utf-8") as f:
            dataset_x_eval = json.load(f)
    except FileNotFoundError as e:
        print(f"評価結果のJSONファイルが見つかりません: {e}")
        return
    
    report = ["# クラスタラベル評価比較\n"]
    
    # 基本統計量の追加
    for dataset, name in [(hdbscan_eval, "HDBSCAN"), (dataset_x_eval, "Dataset X")]:
        df = pd.DataFrame(dataset)
        report.append(f"## {name}の評価結果")
        
        # 総合評価の統計
        scores = df["total_score"]
        report.append("### 総合評価")
        report.append(f"- 平均スコア: {scores.mean():.2f}")
        report.append(f"- 中央値: {scores.median():.2f}")
        report.append(f"- 標準偏差: {scores.std():.2f}")
        report.append(f"- 最高スコア: {scores.max():.2f}")
        report.append(f"- 最低スコア: {scores.min():.2f}\n")
        
        # 項目別の統計
        report.append("### 項目別評価")
        for metric in ["consistency_score", "specificity_score", "coverage_score", "keyword_score"]:
            scores = df[metric]
            name_map = {
                "consistency_score": "一貫性",
                "specificity_score": "具体性",
                "coverage_score": "網羅性",
                "keyword_score": "キーワード"
            }
            report.append(f"#### {name_map[metric]}")
            report.append(f"- 平均スコア: {scores.mean():.2f}")
            report.append(f"- 中央値: {scores.median():.2f}")
            report.append(f"- 標準偏差: {scores.std():.2f}")
            report.append(f"- 最高スコア: {scores.max():.2f}")
            report.append(f"- 最低スコア: {scores.min():.2f}\n")
    
    # レポートの保存
    with open("experiments/results/comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("比較レポートを生成しました: experiments/results/comparison_report.md")

if __name__ == "__main__":
    generate_comparison_report()
