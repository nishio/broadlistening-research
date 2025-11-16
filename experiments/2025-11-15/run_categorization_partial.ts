#!/usr/bin/env ts-node
/**
 * 部分データの分類（start～endのコメントを分類）
 */

import * as fs from 'fs';
import { parse } from 'csv-parse/sync';
import { stringify } from 'csv-stringify/sync';
import { Sensemaker } from './jigsaw-sensemaker/library/src/sensemaker';
import { VertexModel } from './jigsaw-sensemaker/library/src/models/vertex_model';
import { Comment, Topic } from './jigsaw-sensemaker/library/src/types';

// コマンドライン引数から範囲を取得
const start = parseInt(process.argv[2] || '0');
const end = parseInt(process.argv[3] || '5000');
const outputSuffix = process.argv[4] || `${start}_${end}`;

async function main() {
  console.log(`=== 部分データ分類実験 (${start}～${end}件目) ===\n`);

  // 1. トピックの読み込み
  const topicsData = JSON.parse(fs.readFileSync('topics_sensemaker.json', 'utf-8'));
  const topics: Topic[] = topicsData.topics;
  console.log(`トピック: ${topics.length}個、サブトピック: ${topicsData.metadata.total_subtopics}個\n`);

  // 2. データの読み込み
  const csvContent = fs.readFileSync('arguments.csv', 'utf-8');
  const records = parse(csvContent, { columns: true, skip_empty_lines: true });

  // 指定範囲のデータのみ取得
  const targetRecords = records.slice(start, end);
  const comments: Comment[] = targetRecords.map((record: any) => ({
    id: record['comment-id'],
    text: record['comment_text']
  }));

  console.log(`全データ: ${records.length}件`);
  console.log(`処理範囲: ${start}～${end}件目（${comments.length}件）\n`);

  // 3. Google Cloud認証の確認
  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (!projectId) {
    console.error('エラー: GOOGLE_CLOUD_PROJECT環境変数が設定されていません');
    process.exit(1);
  }
  console.log(`Google Cloud Project: ${projectId}\n`);

  // 4. Sensemakerの初期化
  console.log('Sensemakerを初期化しています...');
  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(projectId, 'global')
  });
  console.log('初期化完了\n');

  // 5. 分類の実行
  console.log('分類を実行しています...');
  const startTime = Date.now();

  try {
    const categorizedComments = await sensemaker.categorizeComments(
      comments,
      true,  // includeSubtopics
      topics,
      '東京都の「未来の東京」戦略に関する市民の意見'
    );

    const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
    const elapsedMinutes = ((Date.now() - startTime) / 60000).toFixed(2);
    console.log(`✓ 分類完了（所要時間: ${elapsedTime}秒 = ${elapsedMinutes}分）\n`);

    // 6. 結果の保存
    console.log('結果を保存しています...');

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
    const csvFilename = `categorized_results_${outputSuffix}.csv`;
    fs.writeFileSync(csvFilename, csvOutput);
    console.log(`✓ ${csvFilename} を保存しました`);

    // JSON形式でも保存
    const jsonFilename = `categorized_results_${outputSuffix}.json`;
    fs.writeFileSync(
      jsonFilename,
      JSON.stringify(categorizedComments, null, 2)
    );
    console.log(`✓ ${jsonFilename} を保存しました\n`);

    // 7. 統計情報
    console.log('統計情報:');
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

    console.log(`\n  トップ10トピック:`);
    const sorted = Object.entries(topicCounts).sort((a, b) => b[1] - a[1]);
    sorted.slice(0, 10).forEach(([topic, count]) => {
      console.log(`    ${topic}: ${count}件`);
    });

    console.log(`\n✓ ${start}～${end}件目の分類が完了しました！`);
  } catch (error) {
    console.error('\nエラーが発生しました:');
    console.error(error);
    process.exit(1);
  }
}

main();
