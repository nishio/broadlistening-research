import pandas as pd
import numpy as np

def load_embeddings(data_dir=None):
    """全データを使用してembeddings_arrayを作成"""
    print("=== データの読み込み ===")
    
    # データディレクトリの探索
    possible_paths = [
        "../dataset/aipubcom/embeddings.pkl",
        "../../dataset/aipubcom/embeddings.pkl",
        "../../../dataset/aipubcom/embeddings.pkl",
        "dataset/aipubcom/embeddings.pkl"
    ]
    
    if data_dir:
        possible_paths.insert(0, f"{data_dir}/embeddings.pkl")
    
    for path in possible_paths:
        try:
            embeddings_df = pd.read_pickle(path)
            print(f"データを{path}から読み込みました。")
            embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
            print(f"データの形状: {embeddings_array.shape}")
            return embeddings_array, embeddings_df
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("embeddings.pklが見つかりませんでした。")

if __name__ == "__main__":
    # データの読み込みテスト
    embeddings_array, embeddings_df = load_embeddings()
    print("\nデータの確認:")
    print(f"embeddings_array.dtype: {embeddings_array.dtype}")
    print(f"embeddings_array.min(): {embeddings_array.min()}")
    print(f"embeddings_array.max(): {embeddings_array.max()}")
