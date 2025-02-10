import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_histogram(hdbscan_metrics, kmeans_metrics_list, output_dir):
    """メトリクスのヒストグラムを作成"""
    metrics = ['size', 'avg_distance', 'max_distance', 'density']
    
    for metric in metrics:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()  # 2つ目の縦軸を作成
        
        # HDBSCANのヒストグラム（左軸）
        ax1.hist(hdbscan_metrics[metric], alpha=0.5, label='HDBSCAN', bins=20, color='blue')
        
        # k-means（同じクラスタ数）のヒストグラム（左軸）
        ax1.hist(kmeans_metrics_list['same clusters'][metric], alpha=0.5, 
                label='k-means (same clusters)', bins=20, color='orange')
        
        # k-means（同じ平均サイズ）のヒストグラム（右軸）
        ax2.hist(kmeans_metrics_list['same avg size'][metric], alpha=0.5,
                label='k-means (same avg size)', bins=20, color='green')
        
        # タイトルと軸ラベル
        plt.title(f'Distribution of {metric}')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Frequency (HDBSCAN & k-means same clusters)')
        ax2.set_ylabel('Frequency (k-means same avg size)')
        
        # 凡例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # グラフの保存
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
