#!/usr/bin/env python3
"""
Supervised UMAPでtopicラベルを教師信号にした可視化
通常のUMAPとSupervised UMAPを比較
"""

import pickle
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定（macOS）
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化け対策
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data():
    """
    embeddingとコメントデータを読み込み
    """
    print("=== データの読み込み ===")

    # Embeddingの読み込み
    with open('youtube_comments_embeddings.pkl', 'rb') as f:
        embeddings_array = pickle.load(f)
    print(f"Embeddings shape: {embeddings_array.shape}")

    # コメントデータの読み込み（JSONから再抽出）
    import json
    with open('youtube-no-votes-summary.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    comments = []
    def extract_comments_recursive(obj, comments_list):
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
    print(f"コメント数: {len(comments)}")

    return embeddings_array, comments


def create_topic_labels(comments):
    """
    topicラベルを数値に変換
    """
    df = pd.DataFrame(comments)
    unique_topics = df['topic'].unique()
    topic_to_id = {topic: idx for idx, topic in enumerate(unique_topics)}
    topic_labels = df['topic'].map(topic_to_id).values

    print(f"\n=== Topic情報 ===")
    print(f"Unique topics: {len(unique_topics)}")
    for topic, idx in topic_to_id.items():
        count = (topic_labels == idx).sum()
        print(f"  {idx}: {topic} ({count})")

    return topic_labels, unique_topics


def visualize_comparison(umap_normal, umap_supervised, comments, unique_topics, output_path):
    """
    通常のUMAPとSupervised UMAPを並べて比較
    """
    print(f"\n=== 比較可視化: {output_path} ===")

    df = pd.DataFrame(comments)
    n_topics = len(unique_topics)

    # カラーマップ
    if n_topics <= 10:
        colors = cm.tab10(np.linspace(0, 1, 10))[:n_topics]
    elif n_topics <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))[:n_topics]
    else:
        colors = cm.rainbow(np.linspace(0, 1, n_topics))

    # 2つのサブプロット
    fig, axes = plt.subplots(1, 2, figsize=(32, 16), dpi=100)

    for ax, umap_data, title in zip(
        axes,
        [umap_normal, umap_supervised],
        ['Normal UMAP', 'Supervised UMAP (target_weight=0.9)']
    ):
        df_plot = df.copy()
        df_plot['x'] = umap_data[:, 0]
        df_plot['y'] = umap_data[:, 1]

        for idx, topic_name in enumerate(unique_topics):
            topic_mask = df_plot['topic'] == topic_name
            topic_points = df_plot[topic_mask][['x', 'y']].values

            # 散布図
            ax.scatter(
                topic_points[:, 0],
                topic_points[:, 1],
                c=[colors[idx]],
                s=15,
                alpha=0.6,
                label=f'{topic_name} ({len(topic_points)})',
                zorder=2
            )

            # convex hull
            if len(topic_points) >= 3:
                try:
                    hull = ConvexHull(topic_points)
                    for simplex in hull.simplices:
                        ax.plot(
                            topic_points[simplex, 0],
                            topic_points[simplex, 1],
                            color=colors[idx],
                            linewidth=2.0,
                            alpha=0.8,
                            zorder=3
                        )
                except:
                    pass

        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図を保存: {output_path}")


def visualize_single(umap_data, comments, unique_topics, output_path, title):
    """
    単体の可視化
    """
    print(f"\n=== 単体可視化: {output_path} ===")

    df = pd.DataFrame(comments)
    df['x'] = umap_data[:, 0]
    df['y'] = umap_data[:, 1]

    n_topics = len(unique_topics)

    # カラーマップ
    if n_topics <= 10:
        colors = cm.tab10(np.linspace(0, 1, 10))[:n_topics]
    elif n_topics <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))[:n_topics]
    else:
        colors = cm.rainbow(np.linspace(0, 1, n_topics))

    fig, ax = plt.subplots(figsize=(20, 16), dpi=100)

    for idx, topic_name in enumerate(unique_topics):
        topic_mask = df['topic'] == topic_name
        topic_points = df[topic_mask][['x', 'y']].values

        # 散布図
        ax.scatter(
            topic_points[:, 0],
            topic_points[:, 1],
            c=[colors[idx]],
            s=15,
            alpha=0.6,
            label=f'{topic_name} ({len(topic_points)})',
            zorder=2
        )

        # convex hull
        if len(topic_points) >= 3:
            try:
                hull = ConvexHull(topic_points)
                for simplex in hull.simplices:
                    ax.plot(
                        topic_points[simplex, 0],
                        topic_points[simplex, 1],
                        color=colors[idx],
                        linewidth=2.0,
                        alpha=0.8,
                        zorder=3
                    )
            except:
                pass

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図を保存: {output_path}")


def main():
    # データ読み込み
    embeddings_array, comments = load_data()

    # Topicラベルの作成
    topic_labels, unique_topics = create_topic_labels(comments)

    # 通常のUMAP
    print("\n=== 通常のUMAP ===")
    umap_normal = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    umap_embeddings_normal = umap_normal.fit_transform(embeddings_array)
    print(f"UMAP完了: shape = {umap_embeddings_normal.shape}")

    # Supervised UMAP
    print("\n=== Supervised UMAP ===")
    target_weight = 0.9
    umap_supervised = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.01,  # 小さくしてクラスタ内の凝集性を高める
        target_weight=target_weight,  # topicラベル情報の影響度を高める
        metric='euclidean'
    )
    umap_embeddings_supervised = umap_supervised.fit_transform(embeddings_array, y=topic_labels)
    print(f"Supervised UMAP完了 (target_weight={target_weight}): shape = {umap_embeddings_supervised.shape}")

    # 比較可視化
    visualize_comparison(
        umap_embeddings_normal,
        umap_embeddings_supervised,
        comments,
        unique_topics,
        'youtube_comments_umap_comparison.png'
    )

    # Supervised UMAPの単体可視化
    visualize_single(
        umap_embeddings_supervised,
        comments,
        unique_topics,
        'youtube_comments_supervised_umap_by_topic.png',
        f'Supervised UMAP (target_weight={target_weight})'
    )

    print("\n=== すべての処理が完了しました ===")
    print("\n生成されたファイル:")
    print("  - youtube_comments_umap_comparison.png (Normal vs Supervised)")
    print("  - youtube_comments_supervised_umap_by_topic.png (Supervised単体)")


if __name__ == "__main__":
    main()
