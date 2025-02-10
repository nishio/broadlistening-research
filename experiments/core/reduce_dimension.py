import os
import pandas as pd
import numpy as np
import umap
from datetime import datetime

# データディレクトリの設定
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'dataset/aipubcom')

# データの読み込み
print("Loading embeddings...")
embeddings_df = pd.read_pickle(os.path.join(data_dir, 'embeddings.pkl'))
embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
print(f"Loaded embeddings with shape: {embeddings_array.shape}")

# UMAPの設定とフィッティング
print("Performing UMAP dimension reduction...")
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings_array)
print(f"Reduced dimensions shape: {embeddings_2d.shape}")

# 2次元データの保存
print("Saving 2D embeddings...")
embeddings_2d_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
embeddings_2d_df.to_pickle(os.path.join(data_dir, 'embeddings_2d.pkl'))
print("Done! Saved to embeddings_2d.pkl")
