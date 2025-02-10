import pandas as pd
from generate_cluster_labels import generate_cluster_labels
from evaluate_cluster_labels import evaluate_cluster_labels

def process_dataset_x():
    """Dataset X（密度上位66件）のラベル生成と評価"""
    # データの読み込み
    kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
    args_df = pd.read_csv('dataset/aipubcom/args.csv')
    cluster_labels_df = pd.read_csv('experiments/results/hdbscan_cluster_labels.csv')
    
    # サイズ5以上のクラスタをフィルタリング
    filtered_df = kmeans_df[kmeans_df['size'] >= 5]
    dataset_x = filtered_df.nlargest(66, 'density')
    
    # クラスタIDとarg-idのマッピング
    cluster_mapping = cluster_labels_df[['data_index', 'cluster']]
    cluster_mapping = cluster_mapping.rename(columns={'cluster': 'cluster_id'})
    
    # クラスタ情報とテキストデータの結合
    dataset_x_with_args = pd.merge(
        dataset_x,
        cluster_mapping,
        on='cluster_id',
        how='left'
    )
    dataset_x_with_args = pd.merge(
        dataset_x_with_args,
        args_df[['arg-id', 'argument']],
        left_on='data_index',
        right_on='arg-id',
        how='left'
    )
    
    # クラスタ情報をCSVとして保存
    dataset_x_with_args.to_csv('experiments/results/dataset_x_clusters.csv', index=False)
    
    print(f"Dataset Xの準備完了（{len(dataset_x)}クラスタ）")
    print("カラム一覧:", dataset_x_with_args.columns.tolist())
    
    # クラスタラベルの生成
    labels = generate_cluster_labels(
        cluster_file="experiments/results/dataset_x_clusters.csv",
        output_file="experiments/results/dataset_x_labels.json"
    )
    
    # ラベルの評価
    evaluations = evaluate_cluster_labels(
        labels_file="experiments/results/dataset_x_labels.json",
        output_file="experiments/results/dataset_x_evaluation.json"
    )
    
    return labels, evaluations

if __name__ == "__main__":
    process_dataset_x()
