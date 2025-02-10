import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# k-means（同じクラスタ数）のデータを読み込む
kmeans_df = pd.read_csv('experiments/results/kmeans_same_n_metrics.csv')

# サイズの統計情報を表示
print('k-means（同じクラスタ数）のサイズ分布:')
print('\n基本統計量:')
print(kmeans_df['size'].describe())
print('\n標準偏差/平均（変動係数）:', kmeans_df['size'].std() / kmeans_df['size'].mean())

# サイズ分布のヒストグラムを作成
plt.figure(figsize=(10, 6))
sns.histplot(data=kmeans_df, x='size', bins=20)
plt.title('k-means（同じクラスタ数）のクラスタサイズ分布')
plt.xlabel('クラスタサイズ')
plt.ylabel('頻度')
plt.savefig('experiments/results/kmeans_same_n_size_distribution.png', dpi=300, bbox_inches='tight')
