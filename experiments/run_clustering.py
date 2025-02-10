import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

def run_clustering(dataset_dir="dataset/aipubcom"):
    """HDBSCANとk-meansのクラスタリングを実行"""
    # データの読み込み
    arguments_df = pd.read_csv(f"{dataset_dir}/args.csv")
    embeddings_df = pd.read_pickle(f"{dataset_dir}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    
    # HDBSCANクラスタリング
    print("HDBSCANクラスタリングを実行中...")
    hdb = HDBSCAN(
        min_cluster_size=5,
        max_cluster_size=30,
        min_samples=2,
        core_dist_n_jobs=-1
    )
    hdb_labels = hdb.fit_predict(embeddings_array)
    
    # k-meansクラスタリング
    print("k-meansクラスタリングを実行中...")
    kmeans = KMeans(
        n_clusters=11,
        random_state=42
    )
    kmeans_labels = kmeans.fit_predict(embeddings_array)
    
    # 結果をDataFrameに保存
    for method, labels in [("hdbscan", hdb_labels), ("kmeans", kmeans_labels)]:
        result = pd.DataFrame({
            "arg-id": arguments_df["arg-id"],
            "comment-id": arguments_df["comment-id"],
            "argument": arguments_df["argument"],
            "cluster-id": labels
        })
        
        # ノイズクラスタ（-1）を除外（HDBSCANのみ）
        if method == "hdbscan":
            result = result[result["cluster-id"] != -1]
        
        # 結果をCSVに保存
        output_file = f"{method}_clustered_arguments.csv"
        result.to_csv(output_file, index=False)
        print(f"{method.upper()}の結果を{output_file}に保存しました")

if __name__ == "__main__":
    run_clustering()
