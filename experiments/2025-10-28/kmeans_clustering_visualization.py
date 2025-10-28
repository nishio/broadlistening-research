#!/usr/bin/env python3
"""
高次元空間でk-meansクラスタリングを実行し、UMAPで可視化
topicベースの可視化と比較するため、5クラスタを使用
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """
    embeddingデータを読み込み（analyze_neighborhood.pyと同じ方法）
    """
    print("=== データの読み込み ===")

    # Embeddingの読み込み（team-miraiデータセット）
    with open('../../dataset/team-mirai/embeddings.pkl', 'rb') as f:
        df_embeddings = pickle.load(f)
        embeddings_array = np.array(df_embeddings['embedding'].tolist())
    print(f"Embeddings shape: {embeddings_array.shape}")

    return embeddings_array


def perform_kmeans(embeddings, n_clusters, random_state=42):
    """
    高次元空間でk-meansクラスタリングを実行
    """
    print(f"\n=== k-means クラスタリング (k={n_clusters}) ===")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    print("クラスタサイズ:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  クラスタ {cluster_id}: {count}点")

    return cluster_labels


def reduce_dimensions_umap(embeddings, n_components=2, random_state=42):
    """
    UMAPで次元削減
    """
    print(f"\n=== UMAP次元削減: {embeddings.shape[1]}次元 → {n_components}次元 ===")
    umap_model = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=15,
        min_dist=0.1
    )
    umap_embeddings = umap_model.fit_transform(embeddings)
    print(f"UMAP完了: shape = {umap_embeddings.shape}")

    return umap_embeddings


def visualize_kmeans_clusters(umap_embeddings, cluster_labels, output_path):
    """
    k-meansクラスタごとに色付け＋convex hullで可視化
    """
    print(f"\n=== k-meansクラスタの可視化: {output_path} ===")

    n_clusters = len(np.unique(cluster_labels))

    # データフレーム作成
    df = pd.DataFrame({
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'cluster': cluster_labels
    })

    # カラーマップ
    if n_clusters <= 10:
        colors = cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
    elif n_clusters <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))

    # プロット
    fig, ax = plt.subplots(figsize=(20, 16), dpi=100)

    for cluster_id in range(n_clusters):
        cluster_mask = df['cluster'] == cluster_id
        cluster_points = df[cluster_mask][['x', 'y']].values

        # 散布図
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[cluster_id]],
            s=10,
            alpha=0.5,
            label=f'クラスタ {cluster_id} ({len(cluster_points)})',
            zorder=2
        )

        # convex hull（3点以上ある場合）
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    ax.plot(
                        cluster_points[simplex, 0],
                        cluster_points[simplex, 1],
                        color=colors[cluster_id],
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3
                    )
            except Exception as e:
                # ConvexHullが失敗する場合（共線点など）
                pass

    ax.set_title(f'YouTube Comments UMAP (k-means, k={n_clusters})', fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"図を保存: {output_path}")


def visualize_comparison(umap_embeddings, cluster_labels_kmeans, comments, output_path):
    """
    k-meansクラスタとtopicを並べて比較
    """
    print(f"\n=== 比較可視化: {output_path} ===")

    df = pd.DataFrame(comments)
    df['x'] = umap_embeddings[:, 0]
    df['y'] = umap_embeddings[:, 1]
    df['kmeans_cluster'] = cluster_labels_kmeans

    # 2つのサブプロット
    fig, axes = plt.subplots(1, 2, figsize=(40, 16), dpi=100)

    # 左: topicベース
    ax = axes[0]
    unique_topics = df['topic'].unique()
    n_topics = len(unique_topics)

    if n_topics <= 10:
        colors_topic = cm.tab10(np.linspace(0, 1, 10))[:n_topics]
    elif n_topics <= 20:
        colors_topic = cm.tab20(np.linspace(0, 1, 20))[:n_topics]
    else:
        colors_topic = cm.rainbow(np.linspace(0, 1, n_topics))

    for idx, topic_name in enumerate(unique_topics):
        topic_mask = df['topic'] == topic_name
        topic_points = df[topic_mask][['x', 'y']].values

        ax.scatter(
            topic_points[:, 0],
            topic_points[:, 1],
            c=[colors_topic[idx]],
            s=10,
            alpha=0.5,
            label=f'{topic_name} ({len(topic_points)})',
            zorder=2
        )

        if len(topic_points) >= 3:
            try:
                hull = ConvexHull(topic_points)
                for simplex in hull.simplices:
                    ax.plot(
                        topic_points[simplex, 0],
                        topic_points[simplex, 1],
                        color=colors_topic[idx],
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3
                    )
            except:
                pass

    ax.set_title('YouTube Comments UMAP (colored by topic)', fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 右: k-meansベース
    ax = axes[1]
    n_clusters = len(np.unique(cluster_labels_kmeans))

    if n_clusters <= 10:
        colors_kmeans = cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
    elif n_clusters <= 20:
        colors_kmeans = cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        colors_kmeans = cm.rainbow(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        cluster_mask = df['kmeans_cluster'] == cluster_id
        cluster_points = df[cluster_mask][['x', 'y']].values

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors_kmeans[cluster_id]],
            s=10,
            alpha=0.5,
            label=f'クラスタ {cluster_id} ({len(cluster_points)})',
            zorder=2
        )

        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    ax.plot(
                        cluster_points[simplex, 0],
                        cluster_points[simplex, 1],
                        color=colors_kmeans[cluster_id],
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3
                    )
            except:
                pass

    ax.set_title(f'YouTube Comments UMAP (k-means, k={n_clusters})', fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"図を保存: {output_path}")


def main():
    # データ読み込み
    embeddings_array = load_data()

    # k-meansクラスタリング（k=5）
    cluster_labels = perform_kmeans(embeddings_array, n_clusters=5)

    # UMAP次元削減
    umap_embeddings = reduce_dimensions_umap(embeddings_array)

    # k-meansクラスタの可視化
    visualize_kmeans_clusters(
        umap_embeddings,
        cluster_labels,
        'team_mirai_umap_kmeans.png'
    )

    print("\n=== すべての処理が完了しました ===")
    print("\n生成されたファイル:")
    print("  - team_mirai_umap_kmeans.png (k-means単体)")


if __name__ == "__main__":
    main()
