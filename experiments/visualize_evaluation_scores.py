import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_evaluation_score_distributions():
    """HDBSCANとk-meansのLLM評価スコアの分布を可視化"""
    # データの読み込み
    try:
        with open("hdbscan_label_evaluation.json", "r", encoding="utf-8") as f:
            hdbscan_evals = json.load(f)
        with open("kmeans_label_evaluation.json", "r", encoding="utf-8") as f:
            kmeans_evals = json.load(f)
    except FileNotFoundError:
        print("評価結果のJSONファイルが見つかりません。")
        return

    # データフレームの作成
    def create_df(evals, method):
        records = []
        for eval in evals:
            records.extend([
                {"method": method, "score_type": "総合スコア", "score": eval["total_score"]},
                {"method": method, "score_type": "一貫性", "score": eval["consistency_score"]},
                {"method": method, "score_type": "具体性", "score": eval["specificity_score"]},
                {"method": method, "score_type": "網羅性", "score": eval["coverage_score"]},
                {"method": method, "score_type": "キーワード", "score": eval["keyword_score"]}
            ])
        return pd.DataFrame(records)

    hdbscan_df = create_df(hdbscan_evals, "HDBSCAN")
    kmeans_df = create_df(kmeans_evals, "k-means")
    df = pd.concat([hdbscan_df, kmeans_df])

    # 総合スコアの分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df[df["score_type"] == "総合スコア"], 
                x="score", hue="method", multiple="layer", bins=10)
    plt.title("クラスタラベルの総合評価スコア分布")
    plt.xlabel("評価スコア")
    plt.ylabel("頻度")
    plt.savefig("evaluation_total_scores.png")
    plt.close()

    # 項目別スコアの分布
    plt.figure(figsize=(15, 10))
    for i, score_type in enumerate(["一貫性", "具体性", "網羅性", "キーワード"], 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df[df["score_type"] == score_type], 
                    x="score", hue="method", multiple="layer", bins=10)
        plt.title(f"{score_type}スコアの分布")
        plt.xlabel("評価スコア")
        plt.ylabel("頻度")
    plt.tight_layout()
    plt.savefig("evaluation_detailed_scores.png")
    plt.close()

if __name__ == "__main__":
    plot_evaluation_score_distributions()
