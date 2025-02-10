import pandas as pd
from generate_cluster_labels import generate_cluster_labels
from evaluate_cluster_labels import evaluate_cluster_labels

def process_dataset_x():
    """Dataset X（密度上位66件）のラベル生成と評価"""
    # データの読み込み
    kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
    args_df = pd.read_csv('dataset/aipubcom/args.csv')
    
    # サイズ5以上のクラスタをフィルタリング
    filtered_df = kmeans_df[kmeans_df['size'] >= 5]
    dataset_x = filtered_df.nlargest(66, 'density')
    
    # クラスタラベルの読み込み
    cluster_labels_df = pd.read_csv('experiments/results/hdbscan_cluster_labels.csv')
    
    # データ型の統一とマッピング
    dataset_x['cluster_id'] = dataset_x['cluster_id'].astype(int)
    cluster_labels_df['data_index'] = cluster_labels_df['data_index'].astype(str)
    args_df['arg-id'] = args_df['arg-id'].astype(str)
    
    # クラスタIDとデータインデックスのマッピング
    dataset_x_with_indices = pd.merge(
        dataset_x,
        cluster_labels_df[['data_index', 'cluster']],
        left_on='cluster_id',
        right_on='cluster',
        how='inner'
    )
    
    # テキストデータの結合
    dataset_x_with_args = pd.merge(
        dataset_x_with_indices,
        args_df[['arg-id', 'argument']],
        left_on='data_index',
        right_on='arg-id',
        how='inner'  # 有効なデータのみを使用
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
