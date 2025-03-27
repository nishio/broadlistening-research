import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime
from scipy.spatial.distance import pdist, squareform

def load_hdbscan_results():
    """HDBSCANの結果を読み込む"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments/results')
    with open(os.path.join(results_dir, 'hdbscan_detailed_results.json'), 'r') as f:
        return json.load(f)

def calculate_metrics(embeddings, labels):
    """クラスタリングメトリクスを計算"""
    cluster_metrics = []
    distances = squareform(pdist(embeddings))
    
    for label in sorted(set(labels)):
        cluster_mask = labels == label
        cluster_points = embeddings[cluster_mask]
        if len(cluster_points) > 1:
            cluster_distances = distances[cluster_mask][:, cluster_mask]
            avg_distance = np.mean(cluster_distances[cluster_distances > 0])
            max_distance = np.max(cluster_distances)
            density = 1.0 / avg_distance if avg_distance > 0 else 0
        else:
            avg_distance = max_distance = density = 0
        
        cluster_metrics.append({
            'cluster_id': label,
            'size': len(cluster_points),
            'avg_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'density': float(density)
        })
    return cluster_metrics

def run_comparative_kmeans(embeddings, hdbscan_results):
    """HDBSCANの結果に基づくk-means実験"""
    # HDBSCANのクラスタ数とサイズを計算
    hdbscan_labels = np.array(hdbscan_results['labels'])
    valid_clusters = hdbscan_labels[hdbscan_labels != -1]
    n_clusters = len(set(valid_clusters))
    avg_cluster_size = len(valid_clusters) / n_clusters
    
    print(f"\n=== k-means実験パラメータ ===")
    print(f"HDBSCANのクラスタ数: {n_clusters}")
    print(f"HDBSCANの平均クラスタサイズ: {avg_cluster_size:.2f}")
    
    # 同じクラスタ数のk-means
    print(f"\n=== 同じクラスタ数でのk-means（k={n_clusters}）===")
    kmeans_same_n = KMeans(n_clusters=n_clusters, random_state=42)
    labels_same_n = kmeans_same_n.fit_predict(embeddings)
    metrics_same_n = calculate_metrics(embeddings, labels_same_n)
    
    # 同じ平均クラスタサイズのk-means
    n_clusters_size = int(len(embeddings) / avg_cluster_size)
    print(f"\n=== 同じ平均サイズでのk-means（k={n_clusters_size}）===")
    kmeans_same_size = KMeans(n_clusters=n_clusters_size, random_state=42)
    labels_same_size = kmeans_same_size.fit_predict(embeddings)
    metrics_same_size = calculate_metrics(embeddings, labels_same_size)
    
    return {
        'same_n_clusters': {
            'n_clusters': n_clusters,
            'labels': labels_same_n.tolist(),
            'metrics': metrics_same_n
        },
        'same_avg_size': {
            'n_clusters': n_clusters_size,
            'labels': labels_same_size.tolist(),
            'metrics': metrics_same_size
        }
    }

if __name__ == "__main__":
    # データの読み込み
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'dataset/aipubcom')
    
    print("=== データの読み込み ===")
    embeddings_df = pd.read_pickle(os.path.join(data_dir, 'embeddings.pkl'))
    embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
    print(f"Loaded embeddings with shape: {embeddings_array.shape}")
    
    # HDBSCANの結果を読み込み
    print("\n=== HDBSCANの結果を読み込み ===")
    hdbscan_results = load_hdbscan_results()
    
    # k-means実験の実行
    results = run_comparative_kmeans(embeddings_array, hdbscan_results)
    
    # 結果の保存
    print("\n=== 結果の保存 ===")
    results_dir = os.path.join(base_dir, 'experiments/results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # 同じクラスタ数の結果を保存
    metrics_same_n = pd.DataFrame(results['same_n_clusters']['metrics'])
    metrics_same_n.to_csv(os.path.join(results_dir, 'kmeans_same_n_metrics.csv'), index=False)
    
    # 同じ平均サイズの結果を保存
    metrics_same_size = pd.DataFrame(results['same_avg_size']['metrics'])
    metrics_same_size.to_csv(os.path.join(results_dir, 'kmeans_same_size_metrics.csv'), index=False)
    
    # NumPy型をPython標準の型に変換
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # 詳細な結果をJSONとして保存
    with open(os.path.join(results_dir, 'comparative_kmeans_results.json'), 'w') as f:
        json_data = {
            'timestamp': timestamp,
            'hdbscan_info': {
                'n_clusters': int(results['same_n_clusters']['n_clusters']),
                'avg_cluster_size': float(len(embeddings_array) / results['same_n_clusters']['n_clusters'])
            },
            'results': {
                'same_n_clusters': {
                    'n_clusters': int(results['same_n_clusters']['n_clusters']),
                    'labels': [int(x) for x in results['same_n_clusters']['labels']],
                    'metrics': [{
                        'cluster_id': int(m['cluster_id']),
                        'size': int(m['size']),
                        'avg_distance': float(m['avg_distance']),
                        'max_distance': float(m['max_distance']),
                        'density': float(m['density'])
                    } for m in results['same_n_clusters']['metrics']]
                },
                'same_avg_size': {
                    'n_clusters': int(results['same_avg_size']['n_clusters']),
                    'labels': [int(x) for x in results['same_avg_size']['labels']],
                    'metrics': [{
                        'cluster_id': int(m['cluster_id']),
                        'size': int(m['size']),
                        'avg_distance': float(m['avg_distance']),
                        'max_distance': float(m['max_distance']),
                        'density': float(m['density'])
                    } for m in results['same_avg_size']['metrics']]
                }
            }
        }
        json.dump(json_data, f, indent=2)
    
    print("\n詳細な結果は以下のファイルに保存されました:")
    print(f"- {os.path.join(results_dir, 'kmeans_same_n_metrics.csv')}")
    print(f"- {os.path.join(results_dir, 'kmeans_same_size_metrics.csv')}")
    print(f"- {os.path.join(results_dir, 'comparative_kmeans_results.json')}")
