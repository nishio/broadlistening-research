import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import time
import psutil
import datetime

def get_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB単位

def run_original_hdbscan(embeddings):
    """オリジナルのHDBSCANパラメータでの実験"""
    print("=== HDBSCANクラスタリング（オリジナルパラメータ） ===")
    print(f"開始時刻: {datetime.datetime.now()}")
    print(f"開始時メモリ使用量: {get_memory_usage():.1f} MB")
    
    start_time = time.time()
    
    # HDBSCANの設定（オリジナルのパラメータ）
    hdb = HDBSCAN(
        min_cluster_size=5,
        max_cluster_size=30,
        min_samples=2,
        core_dist_n_jobs=-1  # 並列処理を有効化
    )
    
    print("クラスタリング実行中...")
    labels = hdb.fit_predict(embeddings)
    
    # クラスタリング結果の分析
    n_clusters = len(set(labels[labels != -1]))
    n_noise = sum(labels == -1)
    cluster_sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n=== クラスタリング結果 ===")
    print(f"実行時間: {execution_time/60:.1f}分")
    print(f"終了時刻: {datetime.datetime.now()}")
    print(f"終了時メモリ使用量: {get_memory_usage():.1f} MB")
    print(f"\nデータ点の総数: {len(embeddings)}")
    print(f"クラスタ数: {n_clusters}")
    print(f"ノイズポイント数: {n_noise} ({n_noise/len(embeddings)*100:.1f}%)")
    print("\nクラスタサイズの分布:")
    for label, size in cluster_sizes.items():
        print(f"クラスタ {label}: {size} ポイント")
    
    return labels, {
        'execution_time': execution_time,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise/len(embeddings),
        'cluster_sizes': cluster_sizes.to_dict()
    }

def run_subset_test(embeddings, subset_size=1000):
    """サブセットでテスト実行"""
    print(f"\n=== サブセット（{subset_size}件）でのテスト実行 ===")
    np.random.seed(42)
    indices = np.random.choice(len(embeddings), subset_size, replace=False)
    subset = embeddings[indices]
    return run_original_hdbscan(subset)

def save_results(results, filename, embeddings_shape):
    """結果をJSONファイルとして保存"""
    import json
    from datetime import datetime
    
    # NumPy型をPythonネイティブ型に変換
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(x) for x in obj]
        return obj
    
    # 結果をネイティブ型に変換
    results = convert_to_native(results)
    
    # メタデータを追加
    results['timestamp'] = datetime.now().isoformat()
    results['data_shape'] = [int(x) for x in embeddings_shape]
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"結果を{filename}に保存しました。")

if __name__ == "__main__":
    # データの読み込み
    from data_processing import load_embeddings
    embeddings_array, _ = load_embeddings()
    
    # サブセットでテスト実行
    test_labels, test_results = run_subset_test(embeddings_array)
    save_results(test_results, 'results/subset_test_results.json', embeddings_array.shape)
    
    print("\nテスト実行が完了しました。全データでの実行を開始しますか？[y/N]")
    response = input()
    
    if response.lower() == 'y':
        print("\n=== 全データでの実行 ===")
        labels, results = run_original_hdbscan(embeddings_array)
        save_results(results, 'results/full_clustering_results.json', embeddings_array.shape)
    else:
        print("実行を中止しました。")
