"""データ構造の分析スクリプト"""
import pandas as pd

def analyze_data_structure():
    """データ構造を分析し、各ステップでのデータの状態を確認"""
    print("=== データ構造の分析 ===")
    
    # データの読み込み
    kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
    cluster_labels_df = pd.read_csv('experiments/results/hdbscan_cluster_labels.csv')
    args_df = pd.read_csv('dataset/aipubcom/args.csv')
    
    print("\nKmeans metrics:")
    print(kmeans_df.head())
    print(f"Total rows: {len(kmeans_df)}")
    print(f"Unique clusters: {len(kmeans_df['cluster_id'].unique())}")
    
    print("\nCluster labels:")
    print(cluster_labels_df.head())
    print(f"Total rows: {len(cluster_labels_df)}")
    print(f"Unique clusters: {len(cluster_labels_df['cluster'].unique())}")
    
    print("\nArgs:")
    print(args_df.head())
    print(f"Total rows: {len(args_df)}")
    
    # Dataset X の準備
    filtered_df = kmeans_df[kmeans_df['size'] >= 5]
    dataset_x = filtered_df.nlargest(66, 'density')
    
    print("\nDataset X:")
    print(dataset_x.head())
    print(f"Dataset X size: {len(dataset_x)}")
    print(f"Unique clusters in Dataset X: {len(dataset_x['cluster_id'].unique())}")
    
    # データ型の確認
    print("\nデータ型:")
    print("Kmeans cluster_id:", kmeans_df['cluster_id'].dtype)
    print("HDBSCAN cluster:", cluster_labels_df['cluster'].dtype)
    print("Args arg-id:", args_df['arg-id'].dtype)

if __name__ == "__main__":
    analyze_data_structure()
