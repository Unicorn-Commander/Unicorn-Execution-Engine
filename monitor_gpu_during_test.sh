#!/bin/bash
# Monitor GPU usage during pipeline tests

echo "🔍 GPU Monitoring During Pipeline Tests"
echo "======================================"

# Start GPU monitoring in background
monitor_gpu() {
    while true; do
        if command -v rocm-smi &> /dev/null; then
            GPU_USE=$(rocm-smi --showuse | grep "GPU use" | awk '{print $5}')
            GPU_MEM=$(rocm-smi --showmemuse | grep "GPU Memory" | awk '{print $6}')
            echo -ne "\r🎮 GPU: ${GPU_USE}% | VRAM: ${GPU_MEM}% "
        fi
        sleep 0.5
    done
}

# Start monitoring
monitor_gpu &
MONITOR_PID=$!

# Run the actual test
echo -e "\n\n🚀 Running optimized pipeline test..."
python3 test_igpu_vs_npu_hybrid.py 2>&1 | grep -E "(Performance Comparison|tokens/second|iGPU tok/s)"

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo -e "\n\n✅ Monitoring complete"