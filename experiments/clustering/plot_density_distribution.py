import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ベースディレクトリの設定
base_dir = os.path.dirname(os.path.abspath(__file__))

# HDBSCANの結果読み込み
with open(os.path.join(base_dir, 'experiments/experiments/results/hdbscan_detailed_results.json'), 'r') as f:
    hdbscan_results = json.load(f)
hdbscan_df = pd.DataFrame(hdbscan_results['cluster_metrics'])

# k-meansの結果読み込み
with open(os.path.join(base_dir, 'experiments/results/kmeans_detailed_results.json'), 'r') as f:
    kmeans_results = json.load(f)
kmeans_df = pd.concat([
    pd.DataFrame(k_results['cluster_metrics'])
    for k, k_results in kmeans_results['results'].items()
])

# プロットの設定
plt.figure(figsize=(12, 8))

# HDBSCANのヒストグラム
plt.hist(hdbscan_df['density'], bins=15, alpha=0.6, label='HDBSCAN',
         color='lightcoral', density=True, edgecolor='black')

# k-meansのヒストグラム
plt.hist(kmeans_df['density'], bins=15, alpha=0.6, label='k-means',
         color='lightblue', density=True, edgecolor='black')

# グラフの装飾
plt.xlabel('Density (クラスタの密度)', fontsize=12)
plt.ylabel('Frequency (頻度)', fontsize=12)
plt.title('Comparison of Clustering Density Distribution\nクラスタリング手法による密度分布の比較', 
          fontsize=14, pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)

# x軸の範囲を自動調整（両方のデータを考慮）
min_density = min(hdbscan_df['density'].min(), kmeans_df['density'].min()) * 0.9
max_density = max(hdbscan_df['density'].max(), kmeans_df['density'].max()) * 1.1
plt.xlim(min_density, max_density)

# グラフの保存
plt.savefig(os.path.join(base_dir, 'density_distribution_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
