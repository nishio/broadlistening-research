import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset_x_results():
    """Dataset Xの実験結果を分析"""
    try:
        # 評価結果の読み込み
        with open("experiments/results/dataset_x_evaluation.json", "r", encoding="utf-8") as f:
            evaluations = json.load(f)
        
        # DataFrameに変換
        df = pd.DataFrame(evaluations)
        
        print("=== Dataset X評価結果の分析 ===")
        print(f"\n総クラスタ数: {len(df)}")
        
        # スコアの基本統計量
        print("\n基本統計量:")
        for col in ['total_score', 'consistency_score', 'specificity_score', 'coverage_score', 'keyword_score']:
            scores = df[col]
            print(f"\n{col}:")
            print(f"- 平均値: {scores.mean():.2f}")
            print(f"- 中央値: {scores.median():.2f}")
            print(f"- 標準偏差: {scores.std():.2f}")
            print(f"- 最小値: {scores.min():.2f}")
            print(f"- 最大値: {scores.max():.2f}")
        
        # スコア分布の可視化
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(['total_score', 'consistency_score', 'specificity_score', 'coverage_score', 'keyword_score'], 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=df, x=col, bins=10)
            plt.title(f'{col}の分布')
        plt.tight_layout()
        plt.savefig('experiments/results/dataset_x_score_distribution.png')
        plt.close()
        
        # 高評価クラスタの分析
        high_scores = df[df['total_score'] >= 80].sort_values('total_score', ascending=False)
        print("\n高評価クラスタ（80点以上）:")
        for _, row in high_scores.iterrows():
            print(f"\nクラスタID: {row['cluster_id']}")
            print(f"ラベル: {row['label']}")
            print(f"総合スコア: {row['total_score']}")
            print(f"フィードバック: {row['feedback']}")
        
        # 低評価クラスタの分析
        low_scores = df[df['total_score'] < 50].sort_values('total_score')
        print("\n低評価クラスタ（50点未満）:")
        for _, row in low_scores.iterrows():
            print(f"\nクラスタID: {row['cluster_id']}")
            print(f"ラベル: {row['label']}")
            print(f"総合スコア: {row['total_score']}")
            print(f"フィードバック: {row['feedback']}")
    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")

if __name__ == "__main__":
    analyze_dataset_x_results()
