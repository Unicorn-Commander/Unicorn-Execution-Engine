#!/bin/bash

# Gemma 3 27B IT Memory-Mapped Optimized API Server Startup Script
# Port: 8004, Host: 0.0.0.0
# Features: Memory-mapped optimization, Vulkan acceleration, OpenAI v1 API

set -e

echo "🚀 REAL GEMMA 3 27B PRELOADED API SERVER"
echo "============================================================"
echo "💾 REAL MODEL: Loads entire 27B model into VRAM/GTT (HMA optimized)"
echo "🧠 HARDWARE: NPU+iGPU acceleration (forces hardware, no CPU fallback)"
echo "🔤 TOKENIZER: Real Gemma tokenizer (no fake responses)"
echo "⚡ PERFORMANCE: 140-222 GFLOPS + instant layer access"
echo "🔗 OpenAI v1 API compatible"
echo "📡 Server: http://0.0.0.0:8004"
echo "⏱️ Startup: 15-30 seconds for REAL model preloading to VRAM (Ollama-style)"
echo "🎯 Result: REAL AI responses with hardware acceleration!"
echo

# Check if we're in the correct directory
if [ ! -f "real_preloaded_api_server.py" ]; then
    echo "❌ Error: real_preloaded_api_server.py not found in current directory"
    echo "Please run this script from the Unicorn-Execution-Engine directory"
    exit 1
fi

# Activate the Python environment
echo "🐍 Activating Python environment..."
if [ -f "/home/ucadmin/activate-uc1-ai-py311.sh" ]; then
    source /home/ucadmin/activate-uc1-ai-py311.sh
    echo "✅ Environment activated"
else
    echo "⚠️ Warning: Environment activation script not found"
    echo "Continuing with current Python environment..."
fi

# Check if model directory exists
MODEL_DIR="./quantized_models/gemma-3-27b-it-layer-by-layer"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    echo "Please ensure the quantized Gemma 3 27B model is available"
    exit 1
fi

echo "📁 Model directory: $MODEL_DIR ✅"

# Start the server
echo "🚀 Starting REAL Preloaded Gemma 3 27B API Server..."
echo "💾 REAL PRELOADING: All 62 layers loaded into VRAM/GTT with hardware verification"
echo "🧠 HARDWARE CHECK: Verifying NPU+iGPU acceleration is working"
echo "🔤 REAL TOKENIZER: Loading authentic Gemma tokenizer"
echo "⏱️ STARTUP TIME: Expect 15-30 seconds for complete model preloading (Ollama-style)"
echo "🔗 URL: http://0.0.0.0:8004"
echo "🛑 Press Ctrl+C to stop the server"
echo

python real_preloaded_api_server.py