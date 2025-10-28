#!/usr/bin/env python3
"""
YouTube comments from youtube-no-votes-summary.json の分析スクリプト
- 個別コメントのembedding生成（OpenAI API）
- UMAPで次元削減
- topicごとに色付け＋convex hullで可視化
"""

import json
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
import openai
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# .envファイルから環境変数を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")

# OpenAI APIキーの設定
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 日本語フォントの設定（macOS）
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化け対策
matplotlib.rcParams['axes.unicode_minus'] = False

def load_comments(json_path):
    """
    JSONファイルから個別コメントとtopic情報を抽出

    Returns:
        list of dict: [{"text": "...", "id": "...", "topic": "...", "subtopic": "..."}, ...]
    """
    print(f"=== JSONファイルの読み込み: {json_path} ===")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    comments = []
    # contentsの最後の要素にtopicsセクションがあると仮定
    # 実際のJSONの構造に応じて調整が必要

    # データ構造を確認
    if 'contents' in data:
        # contentsの中から個別コメントを探す（citationsなど）
        # ここでは、JSONの構造を確認してから実装
        pass

    # まずはJSONの末尾付近に個別コメントがあると仮定
    # JSONをフラットに探索
    def extract_comments_recursive(obj, comments_list):
        """再帰的にコメントを探索"""
        if isinstance(obj, dict):
            # コメントオブジェクトの判定: text, id, topics を持つ
            if 'text' in obj and 'id' in obj and 'topics' in obj:
                # topicとsubtopicを取得
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
                    # topicがない場合
                    comments_list.append({
                        'text': obj['text'],
                        'id': obj['id'],
                        'topic': 'Unknown',
                        'subtopic': 'Unknown'
                    })

            # 再帰的に探索
            for value in obj.values():
                extract_comments_recursive(value, comments_list)

        elif isinstance(obj, list):
            for item in obj:
                extract_comments_recursive(item, comments_list)

    extract_comments_recursive(data, comments)

    print(f"抽出されたコメント数: {len(comments)}")

    # topic/subtopicの分布を確認
    df = pd.DataFrame(comments)
    print("\n=== Topic分布 ===")
    print(df['topic'].value_counts())
    print("\n=== Subtopic分布 ===")
    print(df['subtopic'].value_counts().head(10))

    return comments


def generate_embeddings(comments, model="text-embedding-3-small", batch_size=100):
    """
    OpenAI APIを使用してembeddingを生成

    Args:
        comments: list of dict
        model: OpenAIのembeddingモデル名
        batch_size: バッチサイズ

    Returns:
        np.array: (n_comments, embedding_dim)
    """
    print(f"\n=== Embedding生成: {model} ===")

    texts = [c['text'] for c in comments]
    embeddings_list = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding generation"):
        batch_texts = texts[i:i+batch_size]
        response = openai.Embedding.create(
            model=model,
            input=batch_texts
        )
        batch_embeddings = [item['embedding'] for item in response['data']]
        embeddings_list.extend(batch_embeddings)

    embeddings_array = np.array(embeddings_list)
    print(f"Embeddings shape: {embeddings_array.shape}")

    return embeddings_array


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


def visualize_by_topic(umap_embeddings, comments, output_path, grouping='topic'):
    """
    topicまたはsubtopicごとに色付け＋convex hullで可視化

    Args:
        umap_embeddings: UMAP後の2次元座標
        comments: コメントリスト（topic/subtopic情報を含む）
        output_path: 出力先のパス
        grouping: 'topic' or 'subtopic'
    """
    print(f"\n=== {grouping}ごとの可視化 ===")

    df = pd.DataFrame(comments)
    df['x'] = umap_embeddings[:, 0]
    df['y'] = umap_embeddings[:, 1]

    # グルーピングするカラムを選択
    group_col = grouping
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)

    print(f"{group_col}の数: {n_groups}")

    # カラーマップの選択
    if n_groups <= 10:
        colors = cm.tab10(np.linspace(0, 1, 10))[:n_groups]
    elif n_groups <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))[:n_groups]
    else:
        colors = cm.rainbow(np.linspace(0, 1, n_groups))

    # プロット
    fig, ax = plt.subplots(figsize=(20, 16), dpi=100)

    for idx, group_name in enumerate(unique_groups):
        group_mask = df[group_col] == group_name
        group_points = df[group_mask][['x', 'y']].values

        # 散布図
        ax.scatter(
            group_points[:, 0],
            group_points[:, 1],
            c=[colors[idx]],
            s=10,
            alpha=0.5,
            label=f'{group_name} ({len(group_points)})',
            zorder=2
        )

        # convex hull（3点以上ある場合）
        if len(group_points) >= 3:
            try:
                hull = ConvexHull(group_points)
                for simplex in hull.simplices:
                    ax.plot(
                        group_points[simplex, 0],
                        group_points[simplex, 1],
                        color=colors[idx],
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=3
                    )
            except Exception as e:
                # ConvexHullが失敗する場合（共線点など）
                pass

    ax.set_title(f'YouTube Comments UMAP (colored by {group_col})', fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # 凡例の配置
    if n_groups <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"図を保存: {output_path}")


def main():
    # パスの設定
    json_path = "youtube-no-votes-summary.json"
    embeddings_pkl_path = "youtube_comments_embeddings.pkl"
    output_dir = Path(".")

    # ステップ1: コメント抽出
    comments = load_comments(json_path)

    if not comments:
        print("エラー: コメントが抽出できませんでした")
        return

    # ステップ2: Embedding生成（or キャッシュから読み込み）
    if os.path.exists(embeddings_pkl_path):
        print(f"\n=== キャッシュからembeddingを読み込み: {embeddings_pkl_path} ===")
        with open(embeddings_pkl_path, 'rb') as f:
            embeddings_array = pickle.load(f)
        print(f"Embeddings shape: {embeddings_array.shape}")
    else:
        embeddings_array = generate_embeddings(comments, model="text-embedding-3-small")
        # 保存
        with open(embeddings_pkl_path, 'wb') as f:
            pickle.dump(embeddings_array, f)
        print(f"Embeddingsを保存: {embeddings_pkl_path}")

    # ステップ3: UMAP次元削減
    umap_embeddings = reduce_dimensions_umap(embeddings_array)

    # ステップ4: 可視化
    # 4-1: topicごと
    visualize_by_topic(
        umap_embeddings,
        comments,
        output_dir / "youtube_comments_umap_by_topic.png",
        grouping='topic'
    )

    # 4-2: subtopicごと
    visualize_by_topic(
        umap_embeddings,
        comments,
        output_dir / "youtube_comments_umap_by_subtopic.png",
        grouping='subtopic'
    )

    print("\n=== すべての処理が完了しました ===")


if __name__ == "__main__":
    main()
