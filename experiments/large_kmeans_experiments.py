import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import time
import datetime
import psutil
from data_processing import load_embeddings

def get_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB単位

def run_large_k_means(embeddings, k_values=[50, 100, 200]):
    """大きなkでのk-means実験"""
    print(f"\n=== k-meansクラスタリング（k={k_values}） ===")
    print(f"開始時刻: {datetime.datetime.now()}")
    print(f"開始時メモリ使用量: {get_memory_usage():.1f} MB")
    
    start_time = time.time()
    results = {}
    
    for k in k_values:
        print(f"\nk={k}でクラスタリング実行中...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # クラスタリング結果の分析
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        print(f"\n=== k={k}の結果 ===")
        print(f"データ点の総数: {len(embeddings)}")
        print("\nクラスタサイズの分布:")
        for label, size in cluster_sizes.items():
            print(f"クラスタ {label}: {size} ポイント")
        
        results[k] = {
            'labels': labels.tolist(),
            'cluster_sizes': cluster_sizes.to_dict()
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
    embeddings_array, _ = load_embeddings()
    
    # k-meansクラスタリングの実行
    results = run_large_k_means(embeddings_array)
    
    # 結果の保存
    import json
    with open('results/large_kmeans_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'data_shape': list(embeddings_array.shape),
            'results': results
        }, f, indent=2)
    print("\n結果をresults/large_kmeans_results.jsonに保存しました。")
