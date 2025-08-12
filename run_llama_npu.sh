#!/bin/bash
# NPU + llama.cpp Integration Wrapper

# Enable NPU offloading
export LLAMA_NPU_ENABLE=1
export LD_PRELOAD=./libnpu_attention.so

# Function to check if model uses NPU-compatible attention
check_npu_compatible() {
    local model=$1
    # Check model architecture from metadata
    # For now, assume all models are compatible
    return 0
}

# Run llama.cpp with NPU offloading
run_with_npu() {
    local model=$1
    shift
    
    echo "[NPU] Checking model compatibility..."
    if check_npu_compatible "$model"; then
        echo "[NPU] Model compatible, enabling NPU attention"
        export LLAMA_NPU_OFFLOAD=attention
    else
        echo "[NPU] Model not compatible, using GPU only"
        unset LLAMA_NPU_OFFLOAD
    fi
    
    # Run llama.cpp
    ./llama.cpp/main -m "$model" "$@"
}

# Main execution
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [llama.cpp args...]"
    exit 1
fi

run_with_npu "$@"
