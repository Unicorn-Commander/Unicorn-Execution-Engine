#\!/bin/bash
echo "🚀 Starting Qwen2.5 NPU+iGPU Hybrid API Server"
echo "📊 OpenAI v1 Compatible API"
echo "🔧 Base URL: http://localhost:8000"
echo "🎯 Model: Qwen2.5-7B-Instruct (Hybrid NPU+iGPU)"
echo ""

# Activate environment
source gemma3n_env/bin/activate

# Start server
python openai_api_server.py
EOF < /dev/null
