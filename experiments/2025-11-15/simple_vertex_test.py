#!/usr/bin/env python3
"""
Vertex AI接続の簡単なテスト（Python版）
"""

import os
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

def test_vertex_ai():
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'gen-lang-client-0654710684')
    location = 'us-central1'

    print(f'Project: {project_id}')
    print(f'Location: {location}')
    print('Initializing Vertex AI...\n')

    try:
        vertexai.init(project=project_id, location=location)

        model = GenerativeModel('gemini-1.5-pro')
        print('✓ Model initialized')

        print('\nSending test request...')
        response = model.generate_content('Say "Hello from Vertex AI!" in Japanese.')

        print('✓ Request successful!')
        print(f'\nResponse: {response.text}\n')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_vertex_ai()
