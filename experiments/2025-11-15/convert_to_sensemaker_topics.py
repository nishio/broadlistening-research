#!/usr/bin/env python3
"""
分類体系をJigsaw Sensemaker形式に変換するスクリプト

入力: classification_tree.json（東京都の政策分類体系）
出力: topics_sensemaker.json（Sensemaker Topic型形式）
"""

import json

def convert_to_sensemaker_topics(classification_tree):
    """
    classification_tree.jsonをSensemaker Topic型形式に変換

    Sensemaker Topic型:
    - FlatTopic: { name: string }
    - NestedTopic: { name: string, subtopics: FlatTopic[] }
    """
    topics = []

    for category in classification_tree['categories']:
        topic = {
            'name': category['name']
        }

        # subcategoriesがある場合はsubtopicsとして追加
        if 'subcategories' in category and category['subcategories']:
            topic['subtopics'] = [
                {'name': subcat['name']}
                for subcat in category['subcategories']
            ]

        topics.append(topic)

    return topics

def main():
    # classification_tree.jsonを読み込み
    with open('classification_tree.json', 'r', encoding='utf-8') as f:
        classification_tree = json.load(f)

    # Sensemaker形式に変換
    topics = convert_to_sensemaker_topics(classification_tree)

    # topics_sensemaker.jsonに保存
    output_data = {
        'topics': topics,
        'metadata': {
            'source': 'classification_tree.json',
            'format': 'Jigsaw Sensemaker Topic type',
            'total_topics': len(topics),
            'total_subtopics': sum(
                len(topic.get('subtopics', [])) for topic in topics
            ),
            'description': '東京都「未来の東京」の実現に向けた重点政策方針2024の分類体系'
        }
    }

    with open('topics_sensemaker.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ topics_sensemaker.json を生成しました")
    print(f"  - トピック数: {output_data['metadata']['total_topics']}")
    print(f"  - サブトピック数: {output_data['metadata']['total_subtopics']}")
    print()
    print("生成されたトピック:")
    for topic in topics:
        subtopic_count = len(topic.get('subtopics', []))
        print(f"  - {topic['name']} ({subtopic_count}個のサブトピック)")

if __name__ == '__main__':
    main()
