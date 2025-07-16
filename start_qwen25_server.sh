#!/bin/bash
# Start Qwen 2.5 OpenAI API Server
# For use with OpenWebUI

echo "🦄 Starting Qwen 2.5 OpenAI API Server"
echo "=================================="

# Activate environment
source ~/activate-uc1-ai-py311.sh

# Navigate to project directory
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Start server
echo "🚀 Starting server on http://0.0.0.0:8000"
echo "📚 API Documentation: http://0.0.0.0:8000/docs"
echo "🔍 Health Check: http://0.0.0.0:8000/health"
echo "📋 Models: http://0.0.0.0:8000/v1/models"
echo ""
echo "For OpenWebUI:"
echo "   URL: http://192.168.1.223:8000/v1"
echo "   Models: qwen2.5-7b-instruct, qwen2.5-32b-instruct, qwen2.5-vl-7b-instruct"
echo ""

python qwen25_openai_api_server.py --host 0.0.0.0 --port 8000