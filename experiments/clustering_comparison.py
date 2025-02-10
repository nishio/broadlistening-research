import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from hdbscan import HDBSCAN

# データの読み込み
def load_data(data_dir="dataset/aipubcom"):
    """データの読み込みと前処理を行う関数"""
    embeddings_df = pd.read_pickle(f"{data_dir}/embeddings.pkl")
    arguments_df = pd.read_csv(f"{data_dir}/args.csv")
    
    # embeddings_arrayの作成（引数のIDでフィルタリング）
    valid_indices = set(arguments_df["comment-id"].astype(str).values)
    embeddings_array = np.array([
        embedding for idx, embedding in enumerate(embeddings_df["embedding"].values.tolist())
        if str(idx) in valid_indices
    ])
    return embeddings_array, arguments_df

# データの読み込み
embeddings_array, arguments_df = load_data()

def perform_clustering(embeddings_array):
    """クラスタリングを実行する関数"""
    # HDBSCANクラスタリング（既存の設定を再現）
    hdbscan = HDBSCAN(
        min_cluster_size=3,  # より小さいクラスタを許容
        max_cluster_size=50,  # より大きいクラスタも許容
        min_samples=1,  # より柔軟なクラスタリング
        cluster_selection_epsilon=0.2,  # クラスタの密度閾値
        cluster_selection_method='leaf',  # より細かいクラスタを許容
        metric='euclidean'  # 距離の計算方法を明示的に指定
    )
    hdbscan_labels = hdbscan.fit_predict(embeddings_array)
    
    # k-meansクラスタリング（HDBSCANと同じクラスタ数）
    n_clusters = len(set(hdbscan_labels[hdbscan_labels != -1]))  # ノイズを除外したクラスタ数
    if n_clusters < 10:  # クラスタ数が少なすぎる場合は調整
        n_clusters = min(48, len(embeddings_array) // 10)  # データサイズの1/10か48のいずれか小さい方
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings_array)
    
    return hdbscan_labels, kmeans_labels

# クラスタリングの実行
hdbscan_labels, kmeans_labels = perform_clustering(embeddings_array)

# 密度指標の計算関数
def calculate_density_metrics(embeddings, labels, exclude_noise=False):
    """クラスタの密度指標を計算する関数
    
    Args:
        embeddings: 埋め込みベクトルの配列
        labels: クラスタラベルの配列
        exclude_noise: ノイズクラスタ（-1）を除外するかどうか
    
    Returns:
        pd.DataFrame: クラスタごとの密度指標
    """
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
    
    # メトリクスをDataFrameに変換し、cluster_idでソート
    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values('cluster_id').reset_index(drop=True)
    return metrics_df

# 両アルゴリズムの密度指標を計算
kmeans_metrics = calculate_density_metrics(embeddings_array, kmeans_labels)
hdbscan_metrics = calculate_density_metrics(embeddings_array, hdbscan_labels, exclude_noise=True)

# シルエット係数の計算
kmeans_silhouette = silhouette_score(embeddings_array, kmeans_labels)
hdbscan_silhouette = silhouette_score(embeddings_array[hdbscan_labels != -1], 
                                     hdbscan_labels[hdbscan_labels != -1])

def visualize_results(kmeans_metrics, hdbscan_metrics, output_dir='.'):
    """クラスタリング結果を可視化する関数"""
    plt.figure(figsize=(15, 10))
    
    # 平均距離の分布を比較
    plt.subplot(2, 1, 1)
    bins = np.linspace(
        min(kmeans_metrics['avg_distance'].min(), hdbscan_metrics['avg_distance'].min()),
        max(kmeans_metrics['avg_distance'].max(), hdbscan_metrics['avg_distance'].max()),
        10
    )
    plt.hist(kmeans_metrics['avg_distance'], alpha=0.5, label='K-means', bins=bins, color='blue')
    plt.hist(hdbscan_metrics['avg_distance'], alpha=0.5, label='HDBSCAN', bins=bins, color='red')
    plt.xlabel('Average Distance within Cluster')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # クラスタサイズの分布を比較
    plt.subplot(2, 1, 2)
    bins = np.linspace(
        min(kmeans_metrics['size'].min(), hdbscan_metrics['size'].min()),
        max(kmeans_metrics['size'].max(), hdbscan_metrics['size'].max()),
        10
    )
    plt.hist(kmeans_metrics['size'], alpha=0.5, label='K-means', bins=bins, color='blue')
    plt.hist(hdbscan_metrics['size'], alpha=0.5, label='HDBSCAN', bins=bins, color='red')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Sizes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 結果の可視化
visualize_results(kmeans_metrics, hdbscan_metrics, '.')

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
