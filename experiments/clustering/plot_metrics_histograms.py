import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_histogram(hdbscan_metrics, kmeans_metrics_list, output_dir):
    """メトリクスのヒストグラムを作成"""
    metrics = ['size', 'avg_distance', 'max_distance', 'density']
    labels = ['HDBSCAN'] + [f'k-means ({desc})' for desc in kmeans_metrics_list.keys()]
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # HDBSCANのヒストグラム
        plt.hist(hdbscan_metrics[metric], alpha=0.5, label='HDBSCAN', bins=20)
        
        # k-meansのヒストグラム
        for desc, df in kmeans_metrics_list.items():
            plt.hist(df[metric], alpha=0.5, label=f'k-means ({desc})', bins=20)
        
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, f'{metric}_histogram.png'))
        plt.close()

if __name__ == "__main__":
    # ディレクトリのセットアップ
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'experiments/results')
    
    print("=== メトリクスファイルの読み込み ===")
    # メトリクスの読み込み
    hdbscan_metrics = pd.read_csv(os.path.join(results_dir, 'hdbscan_detailed_metrics.csv'))
    kmeans_metrics = {
        'same clusters': pd.read_csv(os.path.join(results_dir, 'kmeans_same_n_metrics.csv')),
        'same avg size': pd.read_csv(os.path.join(results_dir, 'kmeans_same_size_metrics.csv'))
    }
    
    print(f"\n=== データの概要 ===")
    print(f"HDBSCAN clusters: {len(hdbscan_metrics)}")
    print(f"k-means (same clusters): {len(kmeans_metrics['same clusters'])}")
    print(f"k-means (same avg size): {len(kmeans_metrics['same avg size'])}")
    
    print("\n=== ヒストグラムの生成 ===")
    # ヒストグラムの作成
    plot_metrics_histogram(hdbscan_metrics, kmeans_metrics, results_dir)
    
    print("\nヒストグラムは以下のファイルに保存されました:")
    for metric in ['size', 'avg_distance', 'max_distance', 'density']:
        print(f"- {os.path.join(results_dir, f'{metric}_histogram.png')}")
