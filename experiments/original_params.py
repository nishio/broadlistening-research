import pandas as pd
import numpy as np
from hdbscan import HDBSCAN

# データの読み込み
embeddings_df = pd.read_pickle('../dataset/aipubcom/embeddings.pkl')
arguments_df = pd.read_csv('../dataset/aipubcom/args.csv')

# embeddings_arrayの作成
valid_indices = set(arguments_df['comment-id'].astype(str).values)
embeddings_array = np.array([
    embedding for idx, embedding in enumerate(embeddings_df['embedding'].values.tolist())
    if str(idx) in valid_indices
])

# オリジナルのパラメータでクラスタリング
hdb = HDBSCAN(min_cluster_size=5, max_cluster_size=30, min_samples=2)
hdb.fit(embeddings_array)

# 結果の確認
print("=== オリジナルパラメータでの結果 ===")
print(f"データ点の総数: {len(embeddings_array)}")
print(f"ノイズポイントの数: {sum(hdb.labels_ == -1)}")
print(f"クラスタ数: {len(set(hdb.labels_[hdb.labels_ != -1]))}")
print("\nクラスタサイズの分布:")
for label in sorted(set(hdb.labels_[hdb.labels_ != -1])):
    print(f"クラスタ {label}: {sum(hdb.labels_ == label)} ポイント")
