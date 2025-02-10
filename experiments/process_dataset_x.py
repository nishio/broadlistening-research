import pandas as pd
from generate_cluster_labels import generate_cluster_labels
from evaluate_cluster_labels import evaluate_cluster_labels

def process_dataset_x():
    """Dataset X（密度上位66件）のラベル生成と評価"""
    # Dataset Xの読み込み
    kmeans_df = pd.read_csv('experiments/results/kmeans_same_size_metrics.csv')
    filtered_df = kmeans_df[kmeans_df['size'] >= 5]
    dataset_x = filtered_df.nlargest(66, 'density')
    
    # クラスタ情報をCSVとして保存
    dataset_x.to_csv('experiments/results/dataset_x_clusters.csv', index=False)
    
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
