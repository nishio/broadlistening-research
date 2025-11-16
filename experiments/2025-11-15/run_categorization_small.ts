#!/usr/bin/env ts-node
/**
 * 小規模テスト: 10件のコメントを分類
 */

import * as fs from 'fs';
import { parse } from 'csv-parse/sync';
import { stringify } from 'csv-stringify/sync';
import { Sensemaker } from './jigsaw-sensemaker/library/src/sensemaker';
import { VertexModel } from './jigsaw-sensemaker/library/src/models/vertex_model';
import { Comment, Topic } from './jigsaw-sensemaker/library/src/types';

async function main() {
  console.log('=== Small Test (10 comments) ===\n');

  // Topics
  const topicsData = JSON.parse(fs.readFileSync('topics_sensemaker.json', 'utf-8'));
  const topics: Topic[] = topicsData.topics;
  console.log(`Topics: ${topics.length}\n`);

  // Comments
  const csvContent = fs.readFileSync('arguments_sample_10.csv', 'utf-8');
  const records = parse(csvContent, { columns: true, skip_empty_lines: true });
  const comments: Comment[] = records.map((record: any) => ({
    id: record['comment-id'],
    text: record['comment_text']
  }));
  console.log(`Comments: ${comments.length}\n`);

  // Initialize
  const projectId = process.env.GOOGLE_CLOUD_PROJECT || 'gen-lang-client-0654710684';
  console.log(`Project: ${projectId}\n`);

  const sensemaker = new Sensemaker({
    defaultModel: new VertexModel(projectId, 'us-central1')
  });
  console.log('Sensemaker initialized\n');

  // Categorize
  console.log('Starting categorization...\n');
  try {
    const categorizedComments = await sensemaker.categorizeComments(
      comments,
      true,  // includeSubtopics
      topics,
      '東京都の市民意見'
    );

    console.log('\n✓ Success!\n');
    console.log('=== 分類結果（全10件） ===\n');

    // 統計情報
    const topicCounts: { [key: string]: number } = {};
    categorizedComments.forEach(comment => {
      comment.topics?.forEach(topic => {
        const mainTopic = topic.name;
        topicCounts[mainTopic] = (topicCounts[mainTopic] || 0) + 1;

        if ('subtopics' in topic && topic.subtopics) {
          topic.subtopics.forEach(sub => {
            const key = `${mainTopic} > ${sub.name}`;
            topicCounts[key] = (topicCounts[key] || 0) + 1;
          });
        }
      });
    });

    console.log('【トピック分布】');
    Object.entries(topicCounts).sort((a, b) => b[1] - a[1]).forEach(([topic, count]) => {
      console.log(`  ${topic}: ${count}件`);
    });

    console.log('\n【詳細結果】\n');
    categorizedComments.forEach((comment, i) => {
      console.log(`${i + 1}. ID: ${comment.id}`);
      console.log(`   テキスト: ${comment.text}`);
      console.log(`   分類:`);
      comment.topics?.forEach(topic => {
        if ('subtopics' in topic && topic.subtopics) {
          topic.subtopics.forEach(sub => {
            console.log(`     - ${topic.name} > ${sub.name}`);
          });
        } else {
          console.log(`     - ${topic.name}`);
        }
      });
      console.log('');
    });

  } catch (error) {
    console.error('\n❌ Error:');
    console.error(error);
    process.exit(1);
  }
}

main();
