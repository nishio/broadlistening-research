import pandas as pd
from generate_cluster_labels import generate_cluster_labels
from evaluate_cluster_labels import evaluate_cluster_labels

def process_hdbscan_clusters():
    """HDBSCANクラスタのラベル生成と評価"""
    # データの読み込み
    hdbscan_df = pd.read_csv('experiments/results/hdbscan_metrics.csv')
    args_df = pd.read_csv('dataset/aipubcom/args.csv')
    cluster_labels_df = pd.read_csv('experiments/results/hdbscan_cluster_labels.csv')
    
    # サイズ5以上のクラスタをフィルタリング
    filtered_df = hdbscan_df[hdbscan_df['size'] >= 5]
    
    # クラスタIDとarg-idのマッピング
    cluster_mapping = cluster_labels_df[['data_index', 'cluster']]
    cluster_mapping['cluster'] = cluster_mapping['cluster'].astype(int)
    cluster_mapping = cluster_mapping.rename(columns={'cluster': 'cluster_id'})
    
    # データ型の統一
    filtered_df['cluster_id'] = filtered_df['cluster_id'].astype(int)
    cluster_mapping['data_index'] = cluster_mapping['data_index'].astype(str)
    args_df['arg-id'] = args_df['arg-id'].astype(str)
    
    # クラスタ情報とテキストデータの結合
    hdbscan_with_args = pd.merge(
        filtered_df,
        cluster_mapping,
        on='cluster_id',
        how='left'
    )
    hdbscan_with_args = pd.merge(
        hdbscan_with_args,
        args_df[['arg-id', 'argument']],
        left_on='data_index',
        right_on='arg-id',
        how='left'
    )
    
    # クラスタ情報をCSVとして保存
    hdbscan_with_args.to_csv('experiments/results/hdbscan_clusters.csv', index=False)
    
    print(f"HDBSCANクラスタの準備完了（{len(filtered_df)}クラスタ）")
    print("カラム一覧:", hdbscan_with_args.columns.tolist())
    
    print("\nクラスタラベルの生成を開始...")
    try:
        labels = generate_cluster_labels(
            cluster_file="experiments/results/hdbscan_clusters.csv",
            output_file="experiments/results/hdbscan_labels.json"
        )
        print(f"クラスタラベル生成完了（{len(labels)}件）")
    except Exception as e:
        print(f"クラスタラベル生成中にエラーが発生: {e}")
        return None, None
    
    print("\nラベルの評価を開始...")
    try:
        evaluations = evaluate_cluster_labels(
            labels_file="experiments/results/hdbscan_labels.json",
            output_file="experiments/results/hdbscan_evaluation.json"
        )
        print(f"ラベル評価完了（{len(evaluations)}件）")
    except Exception as e:
        print(f"ラベル評価中にエラーが発生: {e}")
        return labels, None
    
    return labels, evaluations

if __name__ == "__main__":
    process_hdbscan_clusters()
