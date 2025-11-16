#!/usr/bin/env python3
"""
分類結果の詳細分析
- 複数トピックに分類されたもの
- Otherに分類されたもの
- トピックなしのもの
"""

import csv
from collections import defaultdict

def analyze_results(csv_file):
    multiple_topics = []
    other_topics = []
    no_topics = []
    topic_count_distribution = defaultdict(int)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comment_id = row['comment-id']
            text = row['comment_text']
            topics_str = row['topics']

            # トピックの数を数える
            if not topics_str or topics_str.strip() == '':
                no_topics.append({
                    'id': comment_id,
                    'text': text
                })
                topic_count_distribution[0] += 1
            else:
                # セミコロンで分割
                topics = [t.strip() for t in topics_str.split(';') if t.strip()]
                topic_count = len(topics)
                topic_count_distribution[topic_count] += 1

                # 複数トピック
                if topic_count > 1:
                    multiple_topics.append({
                        'id': comment_id,
                        'text': text,
                        'topics': topics,
                        'count': topic_count
                    })

                # Otherを含むか
                if any('Other' in t or 'other' in t for t in topics):
                    other_topics.append({
                        'id': comment_id,
                        'text': text,
                        'topics': topics
                    })

    # レポート出力
    print("=" * 80)
    print("分類結果の詳細分析")
    print("=" * 80)

    print("\n【トピック数の分布】")
    for count in sorted(topic_count_distribution.keys()):
        print(f"  {count}個のトピック: {topic_count_distribution[count]}件")

    print(f"\n【複数トピックに分類されたコメント】")
    print(f"合計: {len(multiple_topics)}件\n")

    # トピック数でソート
    multiple_topics.sort(key=lambda x: x['count'], reverse=True)

    for i, item in enumerate(multiple_topics[:20], 1):
        print(f"{i}. ID: {item['id']} ({item['count']}トピック)")
        print(f"   テキスト: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
        print(f"   トピック:")
        for topic in item['topics']:
            print(f"     - {topic}")
        print()

    if len(multiple_topics) > 20:
        print(f"   ...他 {len(multiple_topics) - 20}件\n")

    print(f"\n【'Other'に分類されたコメント】")
    print(f"合計: {len(other_topics)}件\n")

    for i, item in enumerate(other_topics[:20], 1):
        print(f"{i}. ID: {item['id']}")
        print(f"   テキスト: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
        print(f"   トピック:")
        for topic in item['topics']:
            print(f"     - {topic}")
        print()

    if len(other_topics) > 20:
        print(f"   ...他 {len(other_topics) - 20}件\n")

    print(f"\n【トピックなしのコメント】")
    print(f"合計: {len(no_topics)}件\n")

    for i, item in enumerate(no_topics[:10], 1):
        print(f"{i}. ID: {item['id']}")
        print(f"   テキスト: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
        print()

    if len(no_topics) > 10:
        print(f"   ...他 {len(no_topics) - 10}件\n")

    # サマリー
    print("\n" + "=" * 80)
    print("サマリー")
    print("=" * 80)
    print(f"総コメント数: {sum(topic_count_distribution.values())}件")
    print(f"  - 1トピックのみ: {topic_count_distribution[1]}件 ({topic_count_distribution[1]/sum(topic_count_distribution.values())*100:.1f}%)")
    print(f"  - 複数トピック: {len(multiple_topics)}件 ({len(multiple_topics)/sum(topic_count_distribution.values())*100:.1f}%)")
    print(f"  - Otherあり: {len(other_topics)}件 ({len(other_topics)/sum(topic_count_distribution.values())*100:.1f}%)")
    print(f"  - トピックなし: {len(no_topics)}件 ({len(no_topics)/sum(topic_count_distribution.values())*100:.1f}%)")
    print()

if __name__ == '__main__':
    analyze_results('categorized_results.csv')
