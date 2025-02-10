import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import json

def calculate_additional_metrics(metrics_file, output_prefix):
    """既存のクラスタリング結果から追加の評価指標を計算"""
    # メトリクスファイルの読み込み
    df = pd.read_csv(metrics_file)
    
    # 基本的な統計量
    metrics = {
        "cluster_count": int(len(df)),
        "avg_cluster_size": float(df["size"].mean()),
        "std_cluster_size": float(df["size"].std()),
        "size_range": [float(df["size"].min()), float(df["size"].max())],
        
        # 密度に関する指標
        "avg_density": float(df["density"].mean()),
        "density_std": float(df["density"].std()),
        "density_range": [float(df["density"].min()), float(df["density"].max())],
        
        # 距離に関する指標
        "avg_distance": float(df["avg_distance"].mean()),
        "distance_std": float(df["avg_distance"].std()),
        "max_distance": float(df["max_distance"].max())
    }
    
    # 結果の保存
    output_file = f"{output_prefix}_additional_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics

def explain_metrics():
    """各評価指標の意味を説明"""
    explanations = {
        "cluster_count": "クラスタの総数。データの分割粒度を示す。",
        "avg_cluster_size": "クラスタの平均サイズ。クラスタの典型的な大きさを示す。",
        "std_cluster_size": "クラスタサイズの標準偏差。サイズのばらつきを示す。",
        "size_range": "クラスタサイズの範囲（最小値、最大値）。サイズの分布を示す。",
        "avg_density": "平均密度。クラスタ内の点の密集度を示す。高いほど凝集性が高い。",
        "density_std": "密度の標準偏差。密度のばらつきを示す。",
        "density_range": "密度の範囲。クラスタの密度分布を示す。",
        "avg_distance": "クラスタ内の平均距離。クラスタの広がりを示す。",
        "distance_std": "距離の標準偏差。距離のばらつきを示す。",
        "max_distance": "最大距離。クラスタの最大の広がりを示す。"
    }
    
    # 説明をファイルに保存
    with open("metric_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)
    
    return explanations

if __name__ == "__main__":
    # HDBSCANの追加メトリクス計算
    hdbscan_metrics = calculate_additional_metrics(
        "hdbscan_cluster_metrics.csv",
        "hdbscan"
    )
    
    # k-meansの追加メトリクス計算
    kmeans_metrics = calculate_additional_metrics(
        "kmeans_cluster_metrics.csv",
        "kmeans"
    )
    
    # メトリクスの説明を生成
    metric_explanations = explain_metrics()
    
    print("\n=== 追加メトリクスの説明 ===")
    for metric, explanation in metric_explanations.items():
        print(f"\n{metric}:")
        print(f"  {explanation}")
    
    print("\n=== HDBSCAN メトリクス ===")
    for k, v in hdbscan_metrics.items():
        print(f"{k}: {v}")
    
    print("\n=== k-means メトリクス ===")
    for k, v in kmeans_metrics.items():
        print(f"{k}: {v}")
