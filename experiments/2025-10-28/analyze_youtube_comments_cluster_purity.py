#!/usr/bin/env python3
"""
YouTube commentsのクラスタ純度分析

analyze_youtube_comments.pyの派生実験：
- topic/subtopicを「クラスタ」と見なす
- 高次元空間でのk近傍にクラスタ外の点がどの程度含まれるかを分析
- 高次元空間上で分離しているのか混在しているのかを検証
"""

import json
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_comments_and_embeddings():
    """
    コメントデータとembeddingsを読み込み
    """
    json_path = "youtube-no-votes-summary.json"
    embeddings_pkl_path = "youtube_comments_embeddings.pkl"

    print(f"=== JSONファイルの読み込み: {json_path} ===")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # コメント抽出（analyze_youtube_comments.pyと同じロジック）
    comments = []

    def extract_comments_recursive(obj, comments_list):
        """再帰的にコメントを探索"""
        if isinstance(obj, dict):
            if 'text' in obj and 'id' in obj and 'topics' in obj:
                topics_info = obj.get('topics', [])
                if topics_info:
                    for topic_info in topics_info:
                        topic_name = topic_info.get('name', 'Unknown')
                        subtopics = topic_info.get('subtopics', [])
                        for subtopic in subtopics:
                            subtopic_name = subtopic.get('name', 'Unknown')
                            comments_list.append({
                                'text': obj['text'],
                                'id': obj['id'],
                                'topic': topic_name,
                                'subtopic': subtopic_name
                            })
                else:
                    comments_list.append({
                        'text': obj['text'],
                        'id': obj['id'],
                        'topic': 'Unknown',
                        'subtopic': 'Unknown'
                    })

            for value in obj.values():
                extract_comments_recursive(value, comments_list)

        elif isinstance(obj, list):
            for item in obj:
                extract_comments_recursive(item, comments_list)

    extract_comments_recursive(data, comments)
    print(f"抽出されたコメント数: {len(comments)}")

    # Embeddingsの読み込み
    print(f"\n=== Embeddingsの読み込み: {embeddings_pkl_path} ===")
    if not os.path.exists(embeddings_pkl_path):
        raise FileNotFoundError(
            f"{embeddings_pkl_path} が見つかりません。\n"
            f"先に analyze_youtube_comments.py を実行してください。"
        )

    with open(embeddings_pkl_path, 'rb') as f:
        embeddings_array = pickle.load(f)
    print(f"Embeddings shape: {embeddings_array.shape}")

    return comments, embeddings_array


def analyze_cluster_purity(embeddings, labels, label_name, k_values=[5, 10, 15, 30, 50]):
    """
    複数のスケールでクラスタ純度を分析

    Args:
        embeddings: 高次元embedding (n_samples, n_dims)
        labels: クラスタラベル (n_samples,)
        label_name: ラベルの名前（"topic" or "subtopic"）
        k_values: 分析する近傍サイズのリスト

    Returns:
        dict: 分析結果
    """
    print(f"\n=== {label_name}ごとのクラスタ純度分析 ===")

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    print(f"{label_name}の数: {n_clusters}")

    # クラスタサイズの確認
    label_counts = pd.Series(labels).value_counts()
    print(f"\nクラスタサイズ:")
    print(f"  最小: {label_counts.min()}")
    print(f"  最大: {label_counts.max()}")
    print(f"  平均: {label_counts.mean():.1f}")
    print(f"  中央値: {label_counts.median():.1f}")

    # 複数のkで分析
    results = []

    for k in k_values:
        if k > len(embeddings) - 1:
            continue

        print(f"\nk={k}近傍を分析中...")
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        # 各点の同一クラスタ割合を計算
        same_cluster_ratios = []

        for i in range(len(labels)):
            my_label = labels[i]
            neighbor_labels = labels[indices[i, 1:]]  # 自分自身を除く
            same_ratio = np.sum(neighbor_labels == my_label) / k
            same_cluster_ratios.append(same_ratio)

        same_cluster_ratios = np.array(same_cluster_ratios)

        results.append({
            'k': k,
            'mean': np.mean(same_cluster_ratios),
            'median': np.median(same_cluster_ratios),
            'min': np.min(same_cluster_ratios),
            'max': np.max(same_cluster_ratios),
            'boundary_pct': np.sum(same_cluster_ratios < 0.9) / len(same_cluster_ratios) * 100
        })

        print(f"  平均同一{label_name}割合: {np.mean(same_cluster_ratios):.3f}")
        print(f"  境界点割合: {np.sum(same_cluster_ratios < 0.9) / len(same_cluster_ratios) * 100:.1f}%")

    return pd.DataFrame(results)


def analyze_per_cluster_purity(embeddings, labels, label_name, k=15):
    """
    クラスタごとの近傍純度を分析

    Args:
        embeddings: 高次元embedding
        labels: クラスタラベル
        label_name: ラベルの名前
        k: 近傍サイズ

    Returns:
        DataFrame: クラスタごとの統計
    """
    print(f"\n=== クラスタごとの{k}近傍純度分析 ===")

    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    unique_labels = np.unique(labels)
    cluster_stats = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_size = np.sum(cluster_mask)

        # このクラスタの点の近傍純度
        cluster_ratios = []
        for i in np.where(cluster_mask)[0]:
            neighbor_labels = labels[indices[i, 1:]]
            same_ratio = np.sum(neighbor_labels == label) / k
            cluster_ratios.append(same_ratio)

        cluster_ratios = np.array(cluster_ratios)

        cluster_stats.append({
            label_name: label,
            'size': cluster_size,
            'avg_purity': np.mean(cluster_ratios),
            'median_purity': np.median(cluster_ratios),
            'boundary_pct': np.sum(cluster_ratios < 0.9) / len(cluster_ratios) * 100
        })

    df_stats = pd.DataFrame(cluster_stats).sort_values('boundary_pct', ascending=False)

    print(f"\n{label_name}ごとの純度統計 (k={k}):")
    print(df_stats.to_string(index=False))

    return df_stats


def visualize_purity_analysis(df_results_topic, df_results_subtopic, df_cluster_stats_topic, df_cluster_stats_subtopic):
    """
    純度分析の可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # === Topicの分析 ===

    # 1. スケールごとの純度（Topic）
    ax = axes[0, 0]
    ax.plot(df_results_topic['k'], df_results_topic['mean'], 'o-', linewidth=2, markersize=8, label='平均', color='blue')
    ax.plot(df_results_topic['k'], df_results_topic['median'], 's-', linewidth=2, markersize=8, label='中央値', color='orange')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='UMAP n_neighbors=15')
    ax.set_xlabel('近傍点数 k', fontsize=11)
    ax.set_ylabel('同一Topic内の割合', fontsize=11)
    ax.set_title('Topicごとのスケール別純度', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. 境界点割合（Topic）
    ax = axes[0, 1]
    ax.plot(df_results_topic['k'], df_results_topic['boundary_pct'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='UMAP n_neighbors=15')
    ax.set_xlabel('近傍点数 k', fontsize=11)
    ax.set_ylabel('境界点の割合 (%)', fontsize=11)
    ax.set_title('Topicごとの境界点割合\n(k近傍の10%以上が他Topic)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Topicごとの境界点割合（サイズ vs 境界点）
    ax = axes[0, 2]
    x = df_cluster_stats_topic['size']
    y = df_cluster_stats_topic['boundary_pct']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df_cluster_stats_topic)))
    ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    for i, row in df_cluster_stats_topic.iterrows():
        ax.annotate(row['topic'][:15], (row['size'], row['boundary_pct']),
                    fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('Topicサイズ', fontsize=11)
    ax.set_ylabel('境界点の割合 (%)', fontsize=11)
    ax.set_title('Topicサイズと境界点の関係 (k=15)', fontsize=12)
    ax.grid(alpha=0.3)
    if x.max() / x.min() > 10:
        ax.set_xscale('log')

    # === Subtopicの分析 ===

    # 4. スケールごとの純度（Subtopic）
    ax = axes[1, 0]
    ax.plot(df_results_subtopic['k'], df_results_subtopic['mean'], 'o-', linewidth=2, markersize=8, label='平均', color='blue')
    ax.plot(df_results_subtopic['k'], df_results_subtopic['median'], 's-', linewidth=2, markersize=8, label='中央値', color='orange')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='UMAP n_neighbors=15')
    ax.set_xlabel('近傍点数 k', fontsize=11)
    ax.set_ylabel('同一Subtopic内の割合', fontsize=11)
    ax.set_title('Subtopicごとのスケール別純度', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 5. 境界点割合（Subtopic）
    ax = axes[1, 1]
    ax.plot(df_results_subtopic['k'], df_results_subtopic['boundary_pct'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='UMAP n_neighbors=15')
    ax.set_xlabel('近傍点数 k', fontsize=11)
    ax.set_ylabel('境界点の割合 (%)', fontsize=11)
    ax.set_title('Subtopicごとの境界点割合\n(k近傍の10%以上が他Subtopic)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. Subtopicごとの境界点割合（上位20のみ）
    ax = axes[1, 2]
    df_top20 = df_cluster_stats_subtopic.nlargest(20, 'size')
    x = df_top20['size']
    y = df_top20['boundary_pct']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df_top20)))
    ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    for i, row in df_top20.iterrows():
        ax.annotate(row['subtopic'][:10], (row['size'], row['boundary_pct']),
                    fontsize=6, ha='center', va='bottom')
    ax.set_xlabel('Subtopicサイズ', fontsize=11)
    ax.set_ylabel('境界点の割合 (%)', fontsize=11)
    ax.set_title('Subtopicサイズと境界点の関係 (k=15, 上位20)', fontsize=12)
    ax.grid(alpha=0.3)
    if x.max() / x.min() > 10:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('youtube_comments_cluster_purity_analysis.png', dpi=150, bbox_inches='tight')
    print('\n図を youtube_comments_cluster_purity_analysis.png に保存しました')


def main():
    # データ読み込み
    comments, embeddings = load_comments_and_embeddings()

    # DataFrameに変換
    df = pd.DataFrame(comments)

    # Topic分布
    print("\n=== Topic分布 ===")
    print(df['topic'].value_counts())

    # Subtopic分布
    print("\n=== Subtopic分布（上位20） ===")
    print(df['subtopic'].value_counts().head(20))

    # === Topic分析 ===
    topic_labels = df['topic'].values
    df_results_topic = analyze_cluster_purity(
        embeddings, topic_labels, 'topic',
        k_values=[5, 10, 15, 30, 50]
    )

    df_cluster_stats_topic = analyze_per_cluster_purity(
        embeddings, topic_labels, 'topic', k=15
    )

    # === Subtopic分析 ===
    subtopic_labels = df['subtopic'].values
    df_results_subtopic = analyze_cluster_purity(
        embeddings, subtopic_labels, 'subtopic',
        k_values=[5, 10, 15, 30, 50]
    )

    df_cluster_stats_subtopic = analyze_per_cluster_purity(
        embeddings, subtopic_labels, 'subtopic', k=15
    )

    # 可視化
    visualize_purity_analysis(
        df_results_topic, df_results_subtopic,
        df_cluster_stats_topic, df_cluster_stats_subtopic
    )

    # 結論
    print("\n=== 結論 ===")

    # Topic
    topic_k15 = df_results_topic[df_results_topic['k'] == 15].iloc[0]
    print(f"\nTopic (k=15):")
    print(f"  平均純度: {topic_k15['mean']:.3f}")
    print(f"  境界点割合: {topic_k15['boundary_pct']:.1f}%")

    if topic_k15['mean'] > 0.90:
        print("  ✓ Topicは高次元空間で明確に分離されている")
    elif topic_k15['mean'] > 0.70:
        print("  △ Topicは部分的に混在している")
    else:
        print("  ✗ Topicは大きく混在している")

    # Subtopic
    subtopic_k15 = df_results_subtopic[df_results_subtopic['k'] == 15].iloc[0]
    print(f"\nSubtopic (k=15):")
    print(f"  平均純度: {subtopic_k15['mean']:.3f}")
    print(f"  境界点割合: {subtopic_k15['boundary_pct']:.1f}%")

    if subtopic_k15['mean'] > 0.90:
        print("  ✓ Subtopicは高次元空間で明確に分離されている")
    elif subtopic_k15['mean'] > 0.70:
        print("  △ Subtopicは部分的に混在している")
    else:
        print("  ✗ Subtopicは大きく混在している")

    print("\n=== すべての処理が完了しました ===")


if __name__ == "__main__":
    main()
