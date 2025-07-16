#!/bin/bash
# NPU Kernel Build Script
# Uses MLIR-AIE2 toolchain for real compilation

set -e

echo "🧠 Building NPU Kernels for Gemma 3"
echo "================================="

# Set environment
export MLIR_AIE_ROOT="/usr/local/mlir-aie2"
export PATH="/usr/local/bin:$PATH"

# Input files
MLIR_FILE="npu_attention_kernel.mlir"
OUTPUT_DIR="compiled_kernels"

mkdir -p $OUTPUT_DIR

echo "📝 Optimizing MLIR..."
aie-opt \
    --aie-objectfifo-stateful-transform \
    --aie-localize-locks \
    --aie-normalize-address-spaces \
    $MLIR_FILE \
    -o $OUTPUT_DIR/attention_optimized.mlir

echo "🔄 Generating NPU binary..."
aie-translate \
    --aie-generate-xaie \
    $OUTPUT_DIR/attention_optimized.mlir \
    -o $OUTPUT_DIR/attention_kernel.elf

echo "✅ NPU kernel compilation complete!"
echo "📁 Output: $OUTPUT_DIR/attention_kernel.elf"
