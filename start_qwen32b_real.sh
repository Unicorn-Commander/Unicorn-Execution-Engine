#!/bin/bash
# Start Qwen 2.5 32B Real API Server
# With actual model inference

echo "🦄 Starting Qwen 2.5 32B Real API Server"
echo "========================================"

# Activate environment
source ~/activate-uc1-ai-py311.sh

# Navigate to project directory
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Start server with real inference
echo "🧠 Starting server with REAL model inference"
echo "🚀 Server URL: http://0.0.0.0:8000"
echo "📚 API Documentation: http://0.0.0.0:8000/docs"
echo "🔍 Health Check: http://0.0.0.0:8000/health"
echo ""
echo "For OpenWebUI:"
echo "   URL: http://192.168.1.223:8000/v1"
echo "   Model: qwen2.5-32b-instruct"
echo ""
echo "⚡ Real Qwen 2.5 32B inference starting..."
echo ""

python qwen32b_working_api_server.py --host 0.0.0.0 --port 8000