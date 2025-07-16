#!/bin/bash
# Start Simple Qwen API Server
# Uses the working Qwen 2.5 7B model

echo "🦄 Starting Simple Qwen API Server"
echo "================================="

# Activate environment
source ~/activate-uc1-ai-py311.sh

# Navigate to project directory
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Start server
echo "🧠 Starting server with Qwen 2.5 7B model"
echo "🚀 Server URL: http://0.0.0.0:8000"
echo ""
echo "For OpenWebUI:"
echo "   URL: http://192.168.1.223:8000/v1"
echo "   Model: qwen2.5-7b-instruct"
echo ""
echo "⚡ Real Qwen inference starting..."
echo ""

python qwen_simple_api_server.py --host 0.0.0.0 --port 8000