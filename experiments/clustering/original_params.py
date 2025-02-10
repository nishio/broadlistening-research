import os
import sys
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist, squareform

def check_data_files():
    """必要なデータファイルの存在を確認"""
    required_files = ['embeddings.pkl', 'args.csv']
    for file in required_files:
        path = os.path.join('../dataset/aipubcom', file)
        if not os.path.exists(path):
            print(f"Error: Required file {file} not found in dataset/aipubcom/")
            sys.exit(1)

# データファイルの確認
check_data_files()

# データの読み込み
try:
    embeddings_df = pd.read_pickle('../dataset/aipubcom/embeddings.pkl')
    arguments_df = pd.read_csv('../dataset/aipubcom/args.csv')
except Exception as e:
    print(f"Error loading data files: {str(e)}")
    sys.exit(1)

# embeddings_arrayの作成
valid_indices = set(arguments_df['comment-id'].astype(str).values)
embeddings_array = np.array([
    embedding for idx, embedding in enumerate(embeddings_df['embedding'].values.tolist())
    if str(idx) in valid_indices
])

# オリジナルのパラメータでクラスタリング
hdb = HDBSCAN(min_cluster_size=5, max_cluster_size=30, min_samples=2)
hdb.fit(embeddings_array)

# クラスタごとの詳細な分析
cluster_metrics = []
distances = squareform(pdist(embeddings_array))

for label in sorted(set(hdb.labels_[hdb.labels_ != -1])):
    cluster_points = embeddings_array[hdb.labels_ == label]
    cluster_distances = distances[hdb.labels_ == label][:, hdb.labels_ == label]
    
    # クラスタ内の平均距離と最大距離を計算
    avg_distance = np.mean(cluster_distances[cluster_distances > 0])
    max_distance = np.max(cluster_distances)
    
    # 密度を計算（平均距離の逆数として定義）
    density = 1.0 / avg_distance if avg_distance > 0 else 0
    
    cluster_metrics.append({
        'cluster_id': label,
        'size': len(cluster_points),
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'density': density
    })

# メトリクスをDataFrameに変換して保存
results_dir = '../experiments/results'
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, 'hdbscan_detailed_metrics.csv')
metrics_df = pd.DataFrame(cluster_metrics)
metrics_df.to_csv(metrics_file, index=False)

# 結果の確認と出力
print("=== オリジナルパラメータでの結果 ===")
print(f"データ点の総数: {len(embeddings_array)}")
print(f"ノイズポイントの数: {sum(hdb.labels_ == -1)}")
print(f"クラスタ数: {len(set(hdb.labels_[hdb.labels_ != -1]))}")
print("\nクラスタサイズの分布:")
for label in sorted(set(hdb.labels_[hdb.labels_ != -1])):
    print(f"クラスタ {label}: {sum(hdb.labels_ == label)} ポイント")

print("\n詳細なメトリクスはhdbscan_cluster_metrics.csvに保存されました")
