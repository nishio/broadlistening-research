import os
import json
import pandas as pd

# データディレクトリの設定
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'experiments/results')

# JSONデータの読み込み
print("Loading clustering results...")
with open(os.path.join(results_dir, 'clustering_results_2d.json'), 'r') as f:
    results = json.load(f)

# HDBSCANメトリクスの保存
print("Saving HDBSCAN metrics...")
hdbscan_metrics = pd.DataFrame(results['hdbscan']['cluster_metrics'])
hdbscan_metrics.to_csv(os.path.join(results_dir, 'hdbscan_detailed_metrics_2d.csv'), index=False)

# k-means（同じクラスタ数）メトリクスの保存
print("Saving k-means (same n) metrics...")
kmeans_same_n_metrics = pd.DataFrame(results['kmeans_same_n']['cluster_metrics'])
kmeans_same_n_metrics.to_csv(os.path.join(results_dir, 'kmeans_same_n_metrics_2d.csv'), index=False)

# k-means（同じ平均サイズ）メトリクスの保存
print("Saving k-means (same size) metrics...")
kmeans_same_size_metrics = pd.DataFrame(results['kmeans_same_size']['cluster_metrics'])
kmeans_same_size_metrics.to_csv(os.path.join(results_dir, 'kmeans_same_size_metrics_2d.csv'), index=False)

print("All metrics saved successfully!")
