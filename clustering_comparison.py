import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from hdbscan import HDBSCAN

# データの読み込み
embeddings_df = pd.read_pickle("outputs/anno-broadlistening/embeddings.pkl")
arguments_df = pd.read_csv("outputs/anno-broadlistening/args.csv")

# embeddings_arrayの作成（引数のIDでフィルタリング）
embeddings_array = np.array([
    embedding for idx, embedding in enumerate(embeddings_df["embedding"].values.tolist())
    if str(idx) in arguments_df["comment-id"].astype(str).values
])

# HDBSCANクラスタリング（既存の設定を再現）
hdbscan = HDBSCAN(min_cluster_size=5, max_cluster_size=30, min_samples=2)
hdbscan_labels = hdbscan.fit_predict(embeddings_array)

# k-meansクラスタリング（HDBSCANと同じ48クラスタ）
n_clusters = len(set(hdbscan_labels[hdbscan_labels != -1]))  # ノイズを除外したクラスタ数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(embeddings_array)

# 密度指標の計算関数
def calculate_density_metrics(embeddings, labels, exclude_noise=False):
    metrics = []
    unique_labels = np.unique(labels)
    if exclude_noise:
        unique_labels = unique_labels[unique_labels != -1]
    
    for cluster_id in unique_labels:
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            # クラスタ内の全ペア間の距離を計算
            distances = pdist(cluster_points)
            avg_distance = np.mean(distances)  # 平均距離（密度の逆指標）
            max_distance = np.max(distances)   # 最大距離
            metrics.append({
                'cluster_id': cluster_id,
                'size': len(cluster_points),
                'avg_distance': avg_distance,
                'max_distance': max_distance,
                'density': 1.0 / (avg_distance + 1e-10)  # 密度指標（平均距離の逆数）
            })
    return pd.DataFrame(metrics)

# 両アルゴリズムの密度指標を計算
kmeans_metrics = calculate_density_metrics(embeddings_array, kmeans_labels)
hdbscan_metrics = calculate_density_metrics(embeddings_array, hdbscan_labels, exclude_noise=True)

# シルエット係数の計算
kmeans_silhouette = silhouette_score(embeddings_array, kmeans_labels)
hdbscan_silhouette = silhouette_score(embeddings_array[hdbscan_labels != -1], 
                                     hdbscan_labels[hdbscan_labels != -1])

# 結果の可視化
plt.figure(figsize=(12, 6))

# 平均距離の分布を比較
plt.subplot(1, 2, 1)
plt.hist(kmeans_metrics['avg_distance'], alpha=0.5, label='K-means', bins=20)
plt.hist(hdbscan_metrics['avg_distance'], alpha=0.5, label='HDBSCAN', bins=20)
plt.xlabel('Average Distance within Cluster')
plt.ylabel('Number of Clusters')
plt.title('Distribution of Cluster Density')
plt.legend()

# クラスタサイズの分布を比較
plt.subplot(1, 2, 2)
plt.hist(kmeans_metrics['size'], alpha=0.5, label='K-means', bins=20)
plt.hist(hdbscan_metrics['size'], alpha=0.5, label='HDBSCAN', bins=20)
plt.xlabel('Cluster Size')
plt.ylabel('Number of Clusters')
plt.title('Distribution of Cluster Sizes')
plt.legend()

plt.tight_layout()
plt.savefig('cluster_comparison.png')

# 結果の保存と出力
kmeans_metrics.to_csv("kmeans_cluster_metrics.csv", index=False)
hdbscan_metrics.to_csv("hdbscan_cluster_metrics.csv", index=False)

print("Silhouette Scores:")
print(f"K-means: {kmeans_silhouette:.3f}")
print(f"HDBSCAN: {hdbscan_silhouette:.3f}")

print("\nCluster Metrics Summary:")
print("\nK-means:")
print(kmeans_metrics.describe())
print("\nHDBSCAN:")
print(hdbscan_metrics.describe())
