import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_cluster_size_distributions():
    """HDBSCANとk-meansのクラスタサイズ分布を可視化"""
    # データの読み込み
    hdbscan_df = pd.read_csv("hdbscan_cluster_metrics.csv")
    kmeans_df = pd.read_csv("kmeans_cluster_metrics.csv")
    
    # プロットの設定
    plt.figure(figsize=(12, 6))
    
    # HDBSCANのヒストグラム
    plt.subplot(1, 2, 1)
    sns.histplot(data=hdbscan_df, x="size", bins=10)
    plt.title("HDBSCANのクラスタサイズ分布")
    plt.xlabel("クラスタサイズ")
    plt.ylabel("頻度")
    
    # k-meansのヒストグラム
    plt.subplot(1, 2, 2)
    sns.histplot(data=kmeans_df, x="size", bins=10)
    plt.title("k-meansのクラスタサイズ分布")
    plt.xlabel("クラスタサイズ")
    plt.ylabel("頻度")
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存
    plt.savefig("cluster_size_distributions.png")
    plt.close()

def plot_density_distributions():
    """HDBSCANとk-meansのクラスタ密度分布を可視化"""
    # データの読み込み
    hdbscan_df = pd.read_csv("hdbscan_cluster_metrics.csv")
    kmeans_df = pd.read_csv("kmeans_cluster_metrics.csv")
    
    # プロットの設定
    plt.figure(figsize=(12, 6))
    
    # HDBSCANのヒストグラム
    plt.subplot(1, 2, 1)
    sns.histplot(data=hdbscan_df, x="density", bins=10)
    plt.title("HDBSCANのクラスタ密度分布")
    plt.xlabel("密度")
    plt.ylabel("頻度")
    
    # k-meansのヒストグラム
    plt.subplot(1, 2, 2)
    sns.histplot(data=kmeans_df, x="density", bins=10)
    plt.title("k-meansのクラスタ密度分布")
    plt.xlabel("密度")
    plt.ylabel("頻度")
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存
    plt.savefig("cluster_density_distributions.png")
    plt.close()

if __name__ == "__main__":
    plot_cluster_size_distributions()
    plot_density_distributions()
