#!/bin/bash
# Monitor GPU usage during benchmark

echo "🎮 GPU Monitoring Started - Press Ctrl+C to stop"
echo "================================================"
echo "Time | GPU% | VRAM(MB) | GTT(MB) | Status"
echo "================================================"

while true; do
    # Get current time
    TIME=$(date +"%H:%M:%S")
    
    # Get GPU usage
    GPU_INFO=$(radeontop -d - -l 1 2>/dev/null | grep -E "gpu|vram|gtt" | head -1)
    
    # Parse values
    GPU_PERCENT=$(echo "$GPU_INFO" | grep -oP 'gpu \K[0-9.]+' || echo "0")
    VRAM_MB=$(cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null | awk '{print int($1/1024/1024)}' || echo "0")
    GTT_MB=$(cat /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null | awk '{print int($1/1024/1024)}' || echo "0")
    
    # Determine status
    if (( $(echo "$GPU_PERCENT > 10" | bc -l) )); then
        STATUS="🟢 GPU ACTIVE!"
    elif (( $(echo "$VRAM_MB > 5000" | bc -l) )); then
        STATUS="🟡 Model Loaded"
    else
        STATUS="🔴 CPU Mode?"
    fi
    
    # Print status
    printf "%s | %5s%% | %8s | %7s | %s\n" "$TIME" "$GPU_PERCENT" "$VRAM_MB" "$GTT_MB" "$STATUS"
    
    sleep 0.5
done