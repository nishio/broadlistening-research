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
    cluster_mapping['cluster'] = cluster_mapping['cluster'].astype(int)
    cluster_mapping = cluster_mapping.rename(columns={'cluster': 'cluster_id'})
    
    # データ型の統一
    dataset_x['cluster_id'] = dataset_x['cluster_id'].astype(int)
    cluster_mapping['data_index'] = cluster_mapping['data_index'].astype(str)
    args_df['arg-id'] = args_df['arg-id'].astype(str)
    
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
    
    print("\nクラスタラベルの生成を開始...")
    try:
        labels = generate_cluster_labels(
            cluster_file="experiments/results/dataset_x_clusters.csv",
            output_file="experiments/results/dataset_x_labels.json"
        )
        print(f"クラスタラベル生成完了（{len(labels)}件）")
    except Exception as e:
        print(f"クラスタラベル生成中にエラーが発生: {e}")
        return None, None
    
    print("\nラベルの評価を開始...")
    try:
        evaluations = evaluate_cluster_labels(
            labels_file="experiments/results/dataset_x_labels.json",
            output_file="experiments/results/dataset_x_evaluation.json"
        )
        print(f"ラベル評価完了（{len(evaluations)}件）")
    except Exception as e:
        print(f"ラベル評価中にエラーが発生: {e}")
        return labels, None
    
    return labels, evaluations

if __name__ == "__main__":
    process_dataset_x()
