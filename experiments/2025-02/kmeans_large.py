import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(dataset_dir="dataset/aipubcom"):
    """k-meansクラスタリングを実行（1226クラスタ）"""
    # データの読み込み
    print("データを読み込み中...")
    arguments_df = pd.read_csv(f"{dataset_dir}/args.csv")
    embeddings_df = pd.read_pickle(f"{dataset_dir}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    
    # k-meansクラスタリング
    print("k-meansクラスタリングを実行中（n_clusters=1226）...")
    kmeans = KMeans(
        n_clusters=1226,
        random_state=42
    )
    kmeans_labels = kmeans.fit_predict(embeddings_array)
    
    # 結果をDataFrameに保存
    result = pd.DataFrame({
        "arg-id": arguments_df["arg-id"],
        "comment-id": arguments_df["comment-id"],
        "argument": arguments_df["argument"],
        "cluster-id": kmeans_labels
    })
    
    # 結果をCSVに保存
    output_file = "experiments/results/kmeans_large_cluster_labels.csv"
    result.to_csv(output_file, index=False)
    print(f"k-meansの結果を{output_file}に保存しました")

if __name__ == "__main__":
    run_kmeans()
