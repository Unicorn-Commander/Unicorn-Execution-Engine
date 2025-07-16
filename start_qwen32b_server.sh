#!/bin/bash
# Start Qwen 2.5 32B OpenAI API Server
# NPU+iGPU accelerated server for production

echo "🦄 Starting Qwen 2.5 32B OpenAI API Server"
echo "=========================================="

# Activate environment
source ~/activate-uc1-ai-py311.sh

# Navigate to project directory
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Start server
echo "🚀 Starting server on http://0.0.0.0:8000"
echo "📚 API Documentation: http://0.0.0.0:8000/docs"
echo "🔍 Health Check: http://0.0.0.0:8000/health"
echo "📋 Models: http://0.0.0.0:8000/v1/models"
echo "🔧 Hardware Status: http://0.0.0.0:8000/hardware"
echo ""
echo "For OpenWebUI:"
echo "   URL: http://192.168.1.223:8000/v1"
echo "   Model: qwen2.5-32b-instruct"
echo ""
echo "🎯 Hardware Acceleration:"
echo "   • NPU Phoenix: 16 TOPS (attention layers)"
echo "   • Radeon 780M: 2.7 TFLOPS (FFN layers)"
echo "   • Target Performance: 90-210 TPS"
echo ""

python qwen32b_openai_api_server.py --host 0.0.0.0 --port 8000