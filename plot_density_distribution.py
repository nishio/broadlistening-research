import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# クラスタリング結果の読み込み
with open('experiments/experiments/results/hdbscan_detailed_results.json', 'r') as f:
    hdbscan_results = json.load(f)

# クラスタごとのメトリクスを取得
hdbscan_df = pd.DataFrame(hdbscan_results['cluster_metrics'])

# プロットの設定
plt.figure(figsize=(12, 8))

# ヒストグラムの作成
plt.hist(hdbscan_df['density'], bins=10, alpha=0.6, label='HDBSCAN',
         color='lightcoral', density=True, edgecolor='black')

# グラフの装飾
plt.xlabel('Density (クラスタの密度)', fontsize=12)
plt.ylabel('Frequency (頻度)', fontsize=12)
plt.title('HDBSCAN Clustering Density Distribution\nHDBSCANクラスタリングの密度分布', 
          fontsize=14, pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

# x軸の範囲を自動調整
min_density = hdbscan_df['density'].min() * 0.9
max_density = hdbscan_df['density'].max() * 1.1
plt.xlim(min_density, max_density)

# グラフの保存
plt.savefig('density_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
