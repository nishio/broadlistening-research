import pandas as pd
import pickle
import numpy as np

# CSVファイルの確認
print("Checking args.csv...")
args_df = pd.read_csv('dataset/aipubcom/args.csv')
print("args.csv shape:", args_df.shape)
print("\nFirst few rows of args.csv:")
print(args_df.head())
print("\nargs.csv columns:", args_df.columns.tolist())

# Pickleファイルの詳細確認
print("\nChecking embeddings.pkl in detail...")
try:
    with open('dataset/aipubcom/embeddings.pkl', 'rb') as f:
        embeddings_df = pickle.load(f)
    print("embeddings_df type:", type(embeddings_df))
    
    if isinstance(embeddings_df, pd.DataFrame):
        print("DataFrame columns:", embeddings_df.columns.tolist())
        if "embedding" in embeddings_df.columns:
            # 最初の要素のembeddingを取得して次元数を確認
            first_embedding = np.array(embeddings_df["embedding"].iloc[0])
            print("\nFirst embedding shape:", first_embedding.shape)
            print("Total number of embeddings:", len(embeddings_df))
            
            # すべてのembeddingを配列に変換
            embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
            print("\nFull embeddings array shape:", embeddings_array.shape)
            print("embeddings array dtype:", embeddings_array.dtype)
            print("embeddings min value:", embeddings_array.min())
            print("embeddings max value:", embeddings_array.max())
    else:
        print("Unexpected type for embeddings.pkl")
except Exception as e:
    print("Error loading embeddings.pkl:", str(e))
