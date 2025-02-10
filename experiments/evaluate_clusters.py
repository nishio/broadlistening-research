import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import json
from scipy.spatial.distance import pdist, squareform

def evaluate_clusters(dataset_dir="dataset/aipubcom", cluster_descriptions_file="cluster_descriptions.json"):
    # データの読み込み
    embeddings_df = pd.read_pickle(f"{dataset_dir}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())
    
    # クラスタリング実行
    hdb = HDBSCAN(
        min_cluster_size=5,
        max_cluster_size=30,
        min_samples=2,
        core_dist_n_jobs=-1
    )
    cluster_labels = hdb.fit_predict(embeddings_array)
    
    # クラスタリング結果の基本統計
    n_clusters = len(set(cluster_labels)) - 1  # -1（ノイズ）を除外
    n_noise = list(cluster_labels).count(-1)
    noise_ratio = n_noise / len(cluster_labels)
    
    # 非ノイズデータのみを抽出
    valid_indices = cluster_labels != -1
    valid_embeddings = embeddings_array[valid_indices]
    valid_labels = cluster_labels[valid_indices]
    
    # 評価指標の計算
    metrics = {}
    
    # 1. シルエット係数
    if len(set(valid_labels)) > 1:  # クラスタが2つ以上ある場合のみ計算可能
        silhouette_avg = silhouette_score(valid_embeddings, valid_labels)
        metrics["silhouette_score"] = silhouette_avg
    
    # 2. クラスタごとの密度評価
    cluster_metrics = []
    for cluster_id in set(valid_labels):
        cluster_points = valid_embeddings[valid_labels == cluster_id]
        
        # クラスタ内の平均距離を計算
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            avg_distance = np.mean(distances)
            
            cluster_metrics.append({
                "cluster_id": cluster_id,
                "size": len(cluster_points),
                "avg_distance": avg_distance,
                "density": 1 / (avg_distance + 1e-10)  # 距離の逆数を密度として使用
            })
    
    # 3. クラスタラベルの質的評価
    with open(cluster_descriptions_file, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    
    avg_interestingness = np.mean([d["興味深さ"] for d in descriptions])
    
    # 総合評価の出力
    evaluation_results = {
        "basic_stats": {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio
        },
        "silhouette_score": metrics.get("silhouette_score", None),
        "cluster_metrics": cluster_metrics,
        "label_quality": {
            "avg_interestingness": avg_interestingness,
            "descriptions": descriptions
        }
    }
    
    # 結果をJSONファイルに保存
    with open("cluster_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    return evaluation_results

if __name__ == "__main__":
    results = evaluate_clusters()
    
    # 結果の要約を表示
    print("\nクラスタリング評価結果:")
    print(f"クラスタ数: {results['basic_stats']['n_clusters']}")
    print(f"ノイズ比率: {results['basic_stats']['noise_ratio']:.2%}")
    print(f"シルエット係数: {results['silhouette_score']:.3f}")
    print(f"平均興味深さ: {results['label_quality']['avg_interestingness']:.1f}")
    
    print("\nクラスタごとの評価:")
    for cluster in results['cluster_metrics']:
        print(f"クラスタ {cluster['cluster_id']}:")
        print(f"  サイズ: {cluster['size']}")
        print(f"  平均距離: {cluster['avg_distance']:.3f}")
        print(f"  密度: {cluster['density']:.3f}")
