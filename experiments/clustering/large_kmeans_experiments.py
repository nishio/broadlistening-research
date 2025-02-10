import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import time
import datetime
import json
import psutil

def get_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB単位

def run_large_k_means(embeddings, k_values=[5, 10, 20]):
    """k-meansクラスタリング実験"""
    print(f"\n=== k-meansクラスタリング（k={k_values}） ===")
    print(f"開始時刻: {datetime.datetime.now()}")
    print(f"開始時メモリ使用量: {get_memory_usage():.1f} MB")
    
    start_time = time.time()
    results = {}
    
    for k in k_values:
        print(f"\nk={k}でクラスタリング実行中...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # クラスタごとの詳細な分析
        from scipy.spatial.distance import pdist, squareform
        cluster_metrics = []
        distances = squareform(pdist(embeddings))
        
        for label in range(k):
            cluster_mask = labels == label
            cluster_points = embeddings[cluster_mask]
            if len(cluster_points) > 1:
                cluster_distances = distances[cluster_mask][:, cluster_mask]
                avg_distance = np.mean(cluster_distances[cluster_distances > 0])
                max_distance = np.max(cluster_distances)
                density = 1.0 / avg_distance if avg_distance > 0 else 0
            else:
                avg_distance = 0
                max_distance = 0
                density = 0
            
            cluster_metrics.append({
                'cluster_id': label,
                'size': len(cluster_points),
                'avg_distance': float(avg_distance),
                'max_distance': float(max_distance),
                'density': float(density)
            })
        
        print(f"\n=== k={k}の結果 ===")
        print(f"データ点の総数: {len(embeddings)}")
        print("\nクラスタサイズの分布:")
        for m in cluster_metrics:
            print(f"クラスタ {m['cluster_id']}: {m['size']} ポイント")
        
        results[k] = {
            'labels': labels.tolist(),
            'cluster_metrics': cluster_metrics
        }
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n=== k-means実験の総括 ===")
    print(f"実行時間: {execution_time/60:.1f}分")
    print(f"終了時刻: {datetime.datetime.now()}")
    print(f"終了時メモリ使用量: {get_memory_usage():.1f} MB")
    
    return results

if __name__ == "__main__":
    # データの読み込み
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'dataset/aipubcom')
    
    embeddings_df = pd.read_pickle(os.path.join(data_dir, 'embeddings.pkl'))
    arguments_df = pd.read_csv(os.path.join(data_dir, 'args.csv'))
    
    # embeddings_arrayの作成
    valid_indices = set(arguments_df['comment-id'].astype(str).values)
    embeddings_array = np.array([
        embedding for idx, embedding in enumerate(embeddings_df['embedding'].values.tolist())
        if str(idx) in valid_indices
    ])
    
    # k-meansクラスタリングの実行
    results = run_large_k_means(embeddings_array)
    
    # 結果の保存
    results_dir = os.path.join(base_dir, 'experiments/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # JSON形式で保存
    results_file = os.path.join(results_dir, 'kmeans_detailed_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'data_shape': list(embeddings_array.shape),
            'results': results
        }, f, indent=2)
    
    # メトリクスをCSVとしても保存（可視化用）
    for k, k_results in results.items():
        metrics_df = pd.DataFrame(k_results['cluster_metrics'])
        metrics_file = os.path.join(results_dir, f'kmeans_k{k}_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
    
    print("\n詳細な結果は以下のファイルに保存されました:")
    print(f"- {results_file}")
    print("- " + "\n- ".join([os.path.join(results_dir, f'kmeans_k{k}_metrics.csv') for k in results.keys()]))
