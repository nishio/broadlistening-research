import os
import pandas as pd
import matplotlib.pyplot as plt

# データディレクトリの設定
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'experiments/results')

# メトリクスの読み込み
print("Loading metrics...")
hdbscan_metrics = pd.read_csv(os.path.join(results_dir, 'hdbscan_detailed_metrics_2d.csv'))
kmeans_same_n_metrics = pd.read_csv(os.path.join(results_dir, 'kmeans_same_n_metrics_2d.csv'))
kmeans_same_size_metrics = pd.read_csv(os.path.join(results_dir, 'kmeans_same_size_metrics_2d.csv'))

# プロット用の設定
metrics = ['size', 'avg_distance', 'max_distance', 'density']
titles = ['クラスタサイズの分布', 'クラスタ内の平均距離', 'クラスタ内の最大距離', 'クラスタの密度']
filenames = ['size_histogram_2d.png', 'avg_distance_histogram_2d.png', 
             'max_distance_histogram_2d.png', 'density_histogram_2d.png']

for metric, title, filename in zip(metrics, titles, filenames):
    plt.figure(figsize=(10, 6))
    plt.hist([hdbscan_metrics[metric], kmeans_same_n_metrics[metric], kmeans_same_size_metrics[metric]], 
             label=['HDBSCAN', 'k-means (同じクラスタ数)', 'k-means (同じ平均サイズ)'],
             bins=30, alpha=0.6)
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel('頻度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

print("All histograms saved successfully!")
