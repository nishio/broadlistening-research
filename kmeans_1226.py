import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans():
    """k-meansクラスタリングを実行（k=1226）"""
    print("データを読み込み中...")
    embeddings_df = pd.read_pickle("dataset/aipubcom/embeddings.pkl")
    arguments_df = pd.read_csv("dataset/aipubcom/args.csv")
    embeddings_array = np.array(embeddings_df["embedding"].values.tolist())
    
    # データの標準化（収束を早める）
    print("データを標準化中...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    print("k-meansクラスタリングを実行中（k=1226）...")
    kmeans = KMeans(
        n_clusters=1226,
        random_state=42,
        n_init=10,  # 初期化回数を制限
        max_iter=300,  # 最大イテレーション数
        tol=1e-4  # 収束判定の閾値
    )
    labels = kmeans.fit_predict(embeddings_scaled)
    
    # 結果をDataFrameに保存
    result = pd.DataFrame({
        "arg-id": arguments_df["arg-id"],
        "comment-id": arguments_df["comment-id"],
        "argument": arguments_df["argument"],
        "cluster-id": labels
    })
    
    # 結果をCSVに保存
    output_file = "experiments/results/kmeans_cluster_labels.csv"
    result.to_csv(output_file, index=False)
    print(f"k-meansの結果を{output_file}に保存しました")

if __name__ == "__main__":
    run_kmeans()
