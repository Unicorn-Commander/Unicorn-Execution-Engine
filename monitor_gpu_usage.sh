#!/bin/bash
# Monitor GPU usage during inference

echo "🔍 GPU Usage Monitor"
echo "===================="

# Check for AMD GPU monitoring tools
if command -v rocm-smi &> /dev/null; then
    echo "✅ Using rocm-smi"
    MONITOR_CMD="rocm-smi --showgpus"
elif command -v radeontop &> /dev/null; then
    echo "✅ Using radeontop"
    MONITOR_CMD="radeontop -d - -l 1"
else
    echo "⚠️  No GPU monitoring tool found"
    echo "Installing radeontop..."
    sudo apt-get update && sudo apt-get install -y radeontop
fi

# Start monitoring in background
echo -e "\n📊 Starting GPU monitoring..."

# Run inference test
echo -e "\n🚀 Running inference test..."

# Test 1: CPU only
echo -e "\n1. CPU-only test:"
/home/ucadmin/Development/Unicorn-Execution-Engine/llama.cpp/build/bin/llama-cli \
    -m gemma-3n-E4B-it-Q8_0.gguf \
    -p "What is artificial intelligence?" \
    -n 50 \
    --n-gpu-layers 0 \
    2>&1 | tail -10 &

PID1=$!

# Monitor GPU while running
for i in {1..5}; do
    echo -e "\nGPU Status (CPU test, sample $i):"
    if command -v rocm-smi &> /dev/null; then
        rocm-smi --showuse --showmemuse | grep -E "(GPU|Memory)"
    else
        cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null || echo "N/A"
    fi
    sleep 1
done

wait $PID1

# Test 2: GPU offload
echo -e "\n2. GPU offload test:"
/home/ucadmin/Development/Unicorn-Execution-Engine/llama.cpp/build/bin/llama-cli \
    -m gemma-3n-E4B-it-Q8_0.gguf \
    -p "What is artificial intelligence?" \
    -n 50 \
    --n-gpu-layers 35 \
    2>&1 | tail -10 &

PID2=$!

# Monitor GPU while running
for i in {1..5}; do
    echo -e "\nGPU Status (GPU test, sample $i):"
    if command -v rocm-smi &> /dev/null; then
        rocm-smi --showuse --showmemuse | grep -E "(GPU|Memory)"
    else
        cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null || echo "N/A"
    fi
    sleep 1
done

wait $PID2

# Check Vulkan info
echo -e "\n📋 Vulkan Device Info:"
vulkaninfo 2>/dev/null | grep -E "(deviceName|driverVersion|GPU id)" | head -10

echo -e "\n✅ Monitoring complete"