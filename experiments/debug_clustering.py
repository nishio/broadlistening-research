import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

# データの読み込み
embeddings_df = pd.read_pickle('../dataset/aipubcom/embeddings.pkl')
arguments_df = pd.read_csv('../dataset/aipubcom/args.csv')

# embeddings_arrayの作成
valid_indices = set(arguments_df['comment-id'].astype(str).values)
embeddings_array = np.array([
    embedding for idx, embedding in enumerate(embeddings_df['embedding'].values.tolist())
    if str(idx) in valid_indices
])

# クラスタリングの実行
hdbscan = HDBSCAN(
    min_cluster_size=3,
    max_cluster_size=50,
    min_samples=1,
    cluster_selection_epsilon=0.2,
    cluster_selection_method='leaf',
    metric='euclidean'
)
hdbscan_labels = hdbscan.fit_predict(embeddings_array)

# 結果の確認
print('=== クラスタリング結果の診断 ===')
print('総データ点数:', len(embeddings_array))
print('HDBSCANラベルの種類:', np.unique(hdbscan_labels))
print('ノイズポイント数:', sum(hdbscan_labels == -1))
print('クラスタ数:', len(set(hdbscan_labels[hdbscan_labels != -1])))
print('\nクラスタサイズの分布:')
for label in sorted(set(hdbscan_labels[hdbscan_labels != -1])):
    print(f'クラスタ {label}: {sum(hdbscan_labels == label)} ポイント')

# メトリクス計算のデバッグ
from scipy.spatial.distance import pdist, squareform

def debug_metrics_calculation(embeddings, labels, exclude_noise=False):
    metrics = []
    unique_labels = np.unique(labels)
    if exclude_noise:
        unique_labels = unique_labels[unique_labels != -1]
    
    print(f'\n=== メトリクス計算のデバッグ ===')
    print(f'ユニークなラベル: {unique_labels}')
    print(f'ラベルの数: {len(unique_labels)}')
    
    for cluster_id in unique_labels:
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            metrics.append({
                'cluster_id': cluster_id,
                'size': len(cluster_points),
                'avg_distance': avg_distance,
                'max_distance': max_distance,
                'density': 1.0 / (avg_distance + 1e-10)
            })
            print(f'\nクラスタ {cluster_id}:')
            print(f'  サイズ: {len(cluster_points)}')
            print(f'  平均距離: {avg_distance:.3f}')
            print(f'  最大距離: {max_distance:.3f}')
            print(f'  密度: {1.0 / (avg_distance + 1e-10):.3f}')
    
    return pd.DataFrame(metrics)

# メトリクスの計算と確認
metrics_df = debug_metrics_calculation(embeddings_array, hdbscan_labels, exclude_noise=True)
print('\n=== メトリクスの要約統計 ===')
print(metrics_df.describe())
