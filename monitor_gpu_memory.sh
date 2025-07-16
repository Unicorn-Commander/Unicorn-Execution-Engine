#!/bin/bash
# Monitor GPU memory usage

echo "Monitoring GPU memory..."

# Get baseline
vram_base=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
gtt_base=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)

echo "Baseline:"
echo "  VRAM: $((vram_base / 1024 / 1024))MB"
echo "  GTT: $((gtt_base / 1024 / 1024))MB"

# Monitor every 2 seconds
while true; do
    vram=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
    gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
    
    vram_mb=$((vram / 1024 / 1024))
    gtt_mb=$((gtt / 1024 / 1024))
    vram_used=$(((vram - vram_base) / 1024 / 1024))
    gtt_used=$(((gtt - gtt_base) / 1024 / 1024))
    
    echo -ne "\rVRAM: ${vram_mb}MB (+${vram_used}MB) | GTT: ${gtt_mb}MB (+${gtt_used}MB) | Total: $(((vram_used + gtt_used) / 1024))GB    "
    
    sleep 2
done