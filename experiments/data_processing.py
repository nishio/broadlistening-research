import pandas as pd
import numpy as np

def load_embeddings(data_dir="../dataset/aipubcom"):
    """全データを使用してembeddings_arrayを作成"""
    print("=== データの読み込み ===")
    embeddings_df = pd.read_pickle(f"{data_dir}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    print(f"データの形状: {embeddings_array.shape}")
    return embeddings_array, embeddings_df

if __name__ == "__main__":
    # データの読み込みテスト
    embeddings_array, embeddings_df = load_embeddings()
    print("\nデータの確認:")
    print(f"embeddings_array.dtype: {embeddings_array.dtype}")
    print(f"embeddings_array.min(): {embeddings_array.min()}")
    print(f"embeddings_array.max(): {embeddings_array.max()}")
