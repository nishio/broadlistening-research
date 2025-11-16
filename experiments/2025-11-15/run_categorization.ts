#!/usr/bin/env ts-node
/**
 * 東京都ブロードリスニングデータをSensemakerで分類するスクリプト
 *
 * 使用方法:
 *   export GOOGLE_CLOUD_PROJECT="your-project-id"
 *   npx ts-node run_categorization.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { parse } from 'csv-parse/sync';
import { stringify } from 'csv-stringify/sync';

// Sensemakerライブラリのimport
import { Sensemaker } from './jigsaw-sensemaker/library/src/sensemaker';
import { VertexModel } from './jigsaw-sensemaker/library/src/models/vertex_model';
import { Comment, Topic } from './jigsaw-sensemaker/library/src/types';

async function main() {
  console.log('スクリプト開始');
  console.log('=== 東京都ブロードリスニング分類実験 ===\n');

  // 1. トピックの読み込み
  console.log('1. トピック体系を読み込んでいます...');
  const topicsData = JSON.parse(
    fs.readFileSync('topics_sensemaker.json', 'utf-8')
  );
  const topics: Topic[] = topicsData.topics;
  console.log(`   ✓ ${topics.length}個のトピック、${topicsData.metadata.total_subtopics}個のサブトピック\n`);

  // 2. サンプルデータの読み込み
  console.log('2. サンプルデータを読み込んでいます...');
  const csvContent = fs.readFileSync('arguments_sample_1000.csv', 'utf-8');
  const records = parse(csvContent, {
    columns: true,
    skip_empty_lines: true
  });

  const comments: Comment[] = records.map((record: any) => ({
    id: record['comment-id'],
    text: record['comment_text']
  }));
  console.log(`   ✓ ${comments.length}件のコメントを読み込みました\n`);

  // 3. Google Cloud認証の確認
  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (!projectId) {
    console.error('エラー: GOOGLE_CLOUD_PROJECT環境変数が設定されていません');
    console.error('\n設定方法:');
    console.error('  export GOOGLE_CLOUD_PROJECT="your-project-id"');
    console.error('  gcloud auth application-default login');
    process.exit(1);
  }
  console.log(`3. Google Cloud Project: ${projectId}\n`);

  // 4. Sensemakerの初期化
  console.log('4. Sensemakerを初期化しています...');
  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(projectId, 'global')
  });
  console.log('   ✓ 初期化完了\n');

  // 5. 分類の実行
  console.log('5. 分類を実行しています（数分かかります）...');
  const startTime = Date.now();

  try {
    const categorizedComments = await sensemaker.categorizeComments(
      comments,
      true,  // includeSubtopics
      topics,
      '東京都の「未来の東京」戦略に関する市民の意見'  // additionalContext
    );

    const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`   ✓ 分類完了（所要時間: ${elapsedTime}秒）\n`);

    // 6. 結果の保存
    console.log('6. 結果を保存しています...');

    // CSV形式で保存
    const outputRecords = categorizedComments.map(comment => {
      const topicsStr = comment.topics
        ?.map(topic => {
          if ('subtopics' in topic && topic.subtopics) {
            return topic.subtopics.map(sub => `${topic.name}:${sub.name}`).join(';');
          } else {
            return topic.name;
          }
        })
        .join(';') || '';

      return {
        'comment-id': comment.id,
        'comment_text': comment.text,
        'topics': topicsStr
      };
    });

    const csvOutput = stringify(outputRecords, { header: true });
    fs.writeFileSync('categorized_results.csv', csvOutput);
    console.log('   ✓ categorized_results.csv を保存しました');

    // JSON形式でも保存（詳細な分析用）
    fs.writeFileSync(
      'categorized_results.json',
      JSON.stringify(categorizedComments, null, 2)
    );
    console.log('   ✓ categorized_results.json を保存しました\n');

    // 7. 統計情報の表示
    console.log('7. 統計情報:');
    const topicCounts: { [key: string]: number } = {};
    categorizedComments.forEach(comment => {
      comment.topics?.forEach(topic => {
        if ('subtopics' in topic && topic.subtopics) {
          topic.subtopics.forEach(sub => {
            const key = `${topic.name}:${sub.name}`;
            topicCounts[key] = (topicCounts[key] || 0) + 1;
          });
        } else {
          topicCounts[topic.name] = (topicCounts[topic.name] || 0) + 1;
        }
      });
    });

    console.log('\n   各トピックへの分類数（上位20件）:');
    const sorted = Object.entries(topicCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);
    sorted.forEach(([topic, count]) => {
      console.log(`     ${topic}: ${count}件`);
    });

    console.log('\n✓ 分類実験が完了しました！');
  } catch (error) {
    console.error('\nエラーが発生しました:');
    console.error(error);
    process.exit(1);
  }
}

main();
