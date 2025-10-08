#!/bin/bash

echo "🚀 Starting LangChain Workshop Environment..."

# Create checkpoint directory
mkdir -p /root

# Check API configuration
echo "🔍 Checking API configuration..."
if [ -n "$OPENAI_API_KEY" ] && [ -n "$OPENAI_API_BASE" ] && [ "$USE_REAL_API" = "true" ]; then
    echo "🔑 API credentials detected - testing connectivity..."

    # Test API connectivity
    python3 -c "
import os
import requests
import json

try:
    api_base = os.environ.get('OPENAI_API_BASE', '').strip().strip('\"').rstrip('/')
    api_key = os.environ.get('OPENAI_API_KEY', '').strip().strip('\"')

    if not api_base or not api_key:
        print('❌ Missing API configuration')
        exit(1)

    # Test with a simple request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # Try to get models list or make a simple completion
    response = requests.get(f'{api_base}/models', headers=headers, timeout=10)

    if response.status_code == 200:
        print('✅ API connectivity successful!')
        try:
            models = response.json()
            if 'data' in models and len(models['data']) > 0:
                print(f'📋 Available models: {len(models[\"data\"])} models found')
            else:
                print('📋 API connected but no models listed')
        except:
            print('📋 API connected (models endpoint may not be available)')
    else:
        print(f'⚠️  API responded with status {response.status_code}')
        print('   Workshop will continue but API calls may fail')

except requests.exceptions.ConnectTimeout:
    print('❌ API connection timeout - check your OPENAI_API_BASE')
    print('   Workshop will continue in demo mode')
except requests.exceptions.ConnectionError:
    print('❌ Cannot connect to API - check your network and OPENAI_API_BASE')
    print('   Workshop will continue in demo mode')
except Exception as e:
    print(f'⚠️  API test failed: {str(e)}')
    print('   Workshop will continue but API calls may fail')
"
else
    echo "📚 Running in demo mode (no API credentials provided)"
    echo "   To use real AI models, provide .env file with your LiteLLM credentials"
fi

# Pre-download sentence-transformers model to speed up workshop
echo "📥 Pre-downloading embedding model..."
python3 -c "
from langchain_huggingface import HuggingFaceEmbeddings
print('Initializing embeddings...')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print('✅ Embeddings ready!')
"

# Create welcome message
echo "
🎉 Welcome to the LangChain Workshop!

📚 Available Notebooks:
   • Task1_Setup.ipynb - Environment verification & API testing
   • Task2_Prompt_Templates.ipynb - Master prompt templates
   • Task3_Multiple_LLMs.ipynb - Connect to LLMs
   • Task4_LCEL_Pipelines.ipynb - Build LCEL chains
   • Task5_Memory_Systems.ipynb - Add conversational memory
   • Task6_RAG_System.ipynb - Create RAG systems
   • Task7_AI_Assistant.ipynb - Launch complete AI assistant

🌐 Access Points:
   • Jupyter Lab: http://localhost:8888
   • Gradio App (Task 7): http://localhost:7860

📂 Workshop Files:
   • All Python files are pre-loaded in /workshop/task{1-7}/ directories
   • .env.example file available for API configuration
   • Sample documents in /workshop/data/ directory

🔧 Setup Options:
   • Demo Mode: Workshop works without API keys (current mode)
   • Production Mode: Use real AI models with your LiteLLM credentials

💡 To use real AI models:
   1. Copy .env.example to .env
   2. Update with your LiteLLM credentials
   3. Restart with: docker run --env-file .env -p 8888:8888 -p 7860:7860 langchain-workshop

Happy Learning! 🚀
"

# Start Jupyter Lab
echo "🔧 Starting Jupyter Lab on port 8888..."
exec jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/workshop