#!/bin/bash
# Compile RDNA3 INT4 shader

echo "🔧 Compiling RDNA3 INT4 shader..."

# Check for glslangValidator
if ! command -v glslangValidator &> /dev/null; then
    echo "❌ glslangValidator not found. Install with: sudo apt install glslang-tools"
    exit 1
fi

# Compile INT4 shader
echo "Compiling rdna3_int4.comp..."
glslangValidator -V rdna3_int4.comp -o rdna3_int4.spv \
    --target-env vulkan1.2 \
    --quiet

if [ $? -eq 0 ]; then
    echo "✅ RDNA3 INT4 shader compiled successfully"
    ls -la rdna3_int4.spv
else
    echo "❌ INT4 shader compilation failed"
    exit 1
fi

echo "🎯 RDNA3 INT4 shader ready for 2x memory efficiency!"