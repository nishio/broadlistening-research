#!/usr/bin/env ts-node
/**
 * Vertex AI接続テスト
 */

import { VertexModel } from './jigsaw-sensemaker/library/src/models/vertex_model';

async function testConnection() {
  console.log('Testing Vertex AI connection...\n');

  const projectId = process.env.GOOGLE_CLOUD_PROJECT;
  if (!projectId) {
    console.error('❌ GOOGLE_CLOUD_PROJECT not set');
    process.exit(1);
  }

  console.log(`Project: ${projectId}`);
  console.log(`Initializing VertexModel...\n`);

  try {
    const model = new VertexModel(projectId, 'us-central1');
    console.log('✓ VertexModel initialized');

    console.log('\nTesting simple text generation...');
    const result = await model.generateText('Say "Hello, Vertex AI!" in Japanese.');

    console.log('✓ Connection successful!');
    console.log(`\nResponse: ${result}\n`);
  } catch (error) {
    console.error('❌ Connection failed:');
    console.error(error);
    process.exit(1);
  }
}

testConnection();
