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
    
    # データ型の統一とマッピング
    dataset_x['cluster_id'] = dataset_x['cluster_id'].astype(int)
    args_df['arg-id'] = args_df['arg-id'].astype(str)
    
    # arg-idからクラスタIDを抽出（例：A123_0 → 123）
    args_df['extracted_id'] = args_df['arg-id'].str.extract(r'A(\d+)_').astype(int)
    
    # クラスタごとの全ての意見を取得
    all_args = []
    for _, row in dataset_x.iterrows():
        cluster_id = row['cluster_id']
        cluster_args = args_df[args_df['extracted_id'] == cluster_id]
        
        if not cluster_args.empty:
            for _, arg_row in cluster_args.iterrows():
                new_row = {
                    'cluster_id': row['cluster_id'],
                    'size': row['size'],
                    'avg_distance': row['avg_distance'],
                    'max_distance': row['max_distance'],
                    'density': row['density'],
                    'data_index': arg_row['arg-id'],
                    'argument': arg_row['argument']
                }
                all_args.append(new_row)
    
    # 結果をDataFrameに変換
    if len(all_args) > 0:
        dataset_x_with_args = pd.DataFrame(all_args)
        # 空の意見を除外
        dataset_x_with_args = dataset_x_with_args[dataset_x_with_args['argument'].notna()]
        print(f"\n有効なクラスタ数: {len(dataset_x_with_args['cluster_id'].unique())}件")
        print(f"有効な意見数: {len(dataset_x_with_args)}件")
        
        # カラムの順序を整理
        columns = ['cluster_id', 'size', 'avg_distance', 'max_distance', 'density', 'data_index', 'argument']
        dataset_x_with_args = dataset_x_with_args[columns]
        
        # クラスタ情報をCSVとして保存
        dataset_x_with_args.to_csv('experiments/results/dataset_x_clusters.csv', index=False)
    else:
        print("\n警告: 有効なデータが見つかりませんでした")
    
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
