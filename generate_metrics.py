import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def generate_metrics():
    print("データを読み込み中...")
    embeddings_df = pd.read_pickle("dataset/aipubcom/embeddings.pkl")
    labels_df = pd.read_csv("experiments/results/kmeans_cluster_labels.csv")
    embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
    labels = labels_df["cluster-id"].values
    
    print("クラスタメトリクスを計算中...")
    cluster_metrics = []
    
    for label in range(1226):
        if label % 100 == 0:
            print(f"処理中: クラスタ {label}/1226")
        
        cluster_mask = labels == label
        cluster_points = embeddings_array[cluster_mask]
        
        if len(cluster_points) > 1:
            # クラスタ内の点同士の距離を計算
            distances = euclidean_distances(cluster_points)
            # 対角要素（自分自身との距離=0）を除外
            distances = distances[~np.eye(distances.shape[0], dtype=bool)]
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            density = 1.0 / (avg_distance + 1e-10)
        else:
            avg_distance = max_distance = density = 0
        
        cluster_metrics.append({
            'cluster_id': label,
            'size': len(cluster_points),
            'avg_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'density': float(density)
        })
    
    metrics_df = pd.DataFrame(cluster_metrics)
    metrics_df.to_csv('experiments/results/kmeans_same_size_metrics.csv', index=False)
    print("\nメトリクスを保存しました")
    
    print("\nメトリクスの確認:")
    print(metrics_df.head())
    print(f"\n総行数: {len(metrics_df)}")

if __name__ == "__main__":
    generate_metrics()
