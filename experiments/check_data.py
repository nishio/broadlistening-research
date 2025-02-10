import pandas as pd
import numpy as np

def check_embeddings(filepath):
    """Check the shape and content of embeddings file"""
    print(f"Checking embeddings from: {filepath}")
    embeddings_df = pd.read_pickle(filepath)
    embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
    print(f"Shape of embeddings: {embeddings_array.shape}")
    return embeddings_array.shape

if __name__ == "__main__":
    # Check current data
    print("\n=== Current Dataset ===")
    check_embeddings("../dataset/aipubcom/embeddings.pkl")
    
    # Check original data
    print("\n=== Original Dataset ===")
    check_embeddings("../../anno-broadlistening/scatter/pipeline/outputs/aipubcom/embeddings.pkl")
