import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
hdbscan_df = pd.read_csv('experiments/results/hdbscan_detailed_metrics.csv')

# サイズ5未満のクラスタを除外し、密度の高い順に上位66件を選択
kmeans_filtered = kmeans_df[kmeans_df['size'] >= 5]
dataset_x = kmeans_filtered.nlargest(66, 'density')

# 結果の情報を表示
print("データセット情報:")
print(f"Dataset X (filtered k=1226) クラスタ数: {len(dataset_x)}")
print(f"HDBSCAN クラスタ数: {len(hdbscan_df)}")

# ヒストグラムの作成
metrics = ['size', 'avg_distance', 'max_distance', 'density']
titles = ['クラスタサイズの分布', 'クラスタ内の平均距離', 'クラスタ内の最大距離', 'クラスタの密度']

# プロットのスタイル設定
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    sns.histplot(data=dataset_x, x=metric, color='blue', alpha=0.5, label='Dataset X', ax=ax)
    sns.histplot(data=hdbscan_df, x=metric, color='red', alpha=0.5, label='HDBSCAN', ax=ax)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('experiments/results/comparison_histograms.png', dpi=300, bbox_inches='tight')
print("\nヒストグラムを保存しました: experiments/results/comparison_histograms.png")
