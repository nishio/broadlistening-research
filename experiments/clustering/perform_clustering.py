import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# データディレクトリの設定
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'dataset/aipubcom')
results_dir = os.path.join(base_dir, 'experiments/results')
os.makedirs(results_dir, exist_ok=True)

# 2次元データの読み込み
print("Loading 2D embeddings...")
embeddings_2d = pd.read_pickle(os.path.join(data_dir, 'embeddings_2d.pkl'))
embeddings_array = embeddings_2d.values
print(f"Loaded embeddings with shape: {embeddings_array.shape}")

def calculate_cluster_metrics(data, labels):
    """クラスタリングメトリクスの計算"""
    cluster_metrics = []
    distances = squareform(pdist(data))
    
    for label in sorted(set(labels[labels != -1])):
        cluster_points = data[labels == label]
        cluster_distances = distances[labels == label][:, labels == label]
        
        avg_distance = np.mean(cluster_distances[cluster_distances > 0])
        max_distance = np.max(cluster_distances)
        density = 1.0 / avg_distance if avg_distance > 0 else 0
        
        cluster_metrics.append({
            'cluster_id': label,
            'size': len(cluster_points),
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'density': density
        })
    return cluster_metrics

# 1. HDBSCANクラスタリング
print("\nPerforming HDBSCAN clustering...")
hdb = HDBSCAN(min_cluster_size=5, max_cluster_size=30, min_samples=2)
hdb.fit(embeddings_array)
hdb_metrics = calculate_cluster_metrics(embeddings_array, hdb.labels_)

# HDBSCANの結果を保存
hdb_results = {
    'timestamp': datetime.now().isoformat(),
    'parameters': {
        'min_cluster_size': 5,
        'max_cluster_size': 30,
        'min_samples': 2
    },
    'data_shape': list(embeddings_array.shape),
    'labels': [int(x) for x in hdb.labels_],
    'probabilities': [float(x) for x in hdb.probabilities_],
    'outlier_scores': [float(x) for x in hdb.outlier_scores_],
    'cluster_metrics': [{
        'cluster_id': int(m['cluster_id']),
        'size': int(m['size']),
        'avg_distance': float(m['avg_distance']),
        'max_distance': float(m['max_distance']),
        'density': float(m['density'])
    } for m in hdb_metrics]
}

# 2. 同じクラスタ数のk-means
n_clusters = len(set(hdb.labels_[hdb.labels_ != -1]))
print(f"\nPerforming k-means clustering with {n_clusters} clusters...")
kmeans_same_n = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_same_n_labels = kmeans_same_n.fit_predict(embeddings_array)
kmeans_same_n_metrics = calculate_cluster_metrics(embeddings_array, kmeans_same_n_labels)

# 3. 同じ平均サイズのk-means
valid_points = len(hdb.labels_[hdb.labels_ != -1])
avg_cluster_size = valid_points / n_clusters if n_clusters > 0 else 0
n_clusters_same_size = int(len(embeddings_array) / avg_cluster_size) if avg_cluster_size > 0 else 0
print(f"\nPerforming k-means clustering with {n_clusters_same_size} clusters (same average size)...")
kmeans_same_size = KMeans(n_clusters=n_clusters_same_size, random_state=42)
kmeans_same_size_labels = kmeans_same_size.fit_predict(embeddings_array)
kmeans_same_size_metrics = calculate_cluster_metrics(embeddings_array, kmeans_same_size_labels)

# 結果の保存
results = {
    'hdbscan': hdb_results,
    'kmeans_same_n': {
        'n_clusters': int(n_clusters),
        'labels': [int(x) for x in kmeans_same_n_labels],
        'cluster_metrics': [{
            'cluster_id': int(m['cluster_id']),
            'size': int(m['size']),
            'avg_distance': float(m['avg_distance']),
            'max_distance': float(m['max_distance']),
            'density': float(m['density'])
        } for m in kmeans_same_n_metrics]
    },
    'kmeans_same_size': {
        'n_clusters': int(n_clusters_same_size),
        'labels': [int(x) for x in kmeans_same_size_labels],
        'cluster_metrics': [{
            'cluster_id': int(m['cluster_id']),
            'size': int(m['size']),
            'avg_distance': float(m['avg_distance']),
            'max_distance': float(m['max_distance']),
            'density': float(m['density'])
        } for m in kmeans_same_size_metrics]
    }
}

# 結果をJSONとして保存
results_file = os.path.join(results_dir, 'clustering_results_2d.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to:", results_file)

# クラスタリング結果の概要を表示
print("\n=== Clustering Results Summary ===")
print("HDBSCAN:")
print(f"- Total points: {len(hdb.labels_)}")
print(f"- Noise points: {sum(hdb.labels_ == -1)}")
print(f"- Number of clusters: {n_clusters}")

print("\nk-means (same number of clusters):")
print(f"- Number of clusters: {n_clusters}")
print(f"- Average cluster size: {len(embeddings_array) / n_clusters:.2f}")

print("\nk-means (same average size):")
print(f"- Number of clusters: {n_clusters_same_size}")
print(f"- Average cluster size: {len(embeddings_array) / n_clusters_same_size:.2f}")
