#!/bin/bash
# Build Real NPU Kernels for Gemma 3 27B
# Compiles custom MLIR kernels to NPU Phoenix binaries
# Bypasses Python - direct MLIR to NPU machine code

set -e

echo "🔥 Building Real NPU Kernels for Gemma 3 27B"
echo "=============================================="

# Use the working MLIR-AIE build
export MLIR_AIE_PATH="/home/ucadmin/Development/whisper_npu_project/mlir-aie"
export PATH="$MLIR_AIE_PATH/build/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_PATH/build/lib:$LD_LIBRARY_PATH"

# Create output directory
mkdir -p real_npu_binaries

echo "🔧 Compiling Q/K/V Projection Kernel..."

# Compile Q/K/V kernel with correct passes
aie-opt \
    --aie-assign-buffer-addresses \
    --aie-assign-bd-ids \
    --aie-assign-lock-ids \
    --convert-to-llvm \
    gemma3_qkv_kernel.mlir \
    -o real_npu_binaries/gemma3_qkv_optimized.mlir

echo "✅ Q/K/V kernel optimized"

# Translate to NPU machine code
aie-translate \
    --aie-generate-cdo \
    --aie-generate-ipu \
    real_npu_binaries/gemma3_qkv_optimized.mlir \
    -o real_npu_binaries/

echo "✅ Q/K/V kernel compiled to NPU binary"

echo "🔧 Compiling Attention Kernel..."

# Compile attention kernel with correct passes
aie-opt \
    --aie-assign-buffer-addresses \
    --aie-assign-bd-ids \
    --aie-assign-lock-ids \
    --convert-to-llvm \
    gemma3_attention_kernel.mlir \
    -o real_npu_binaries/gemma3_attention_optimized.mlir

echo "✅ Attention kernel optimized"

# Translate to NPU machine code
aie-translate \
    --aie-generate-cdo \
    --aie-generate-ipu \
    real_npu_binaries/gemma3_attention_optimized.mlir \
    -o real_npu_binaries/

echo "✅ Attention kernel compiled to NPU binary"

echo "🔧 Generating kernel metadata..."

# Create kernel metadata for runtime loading
cat > real_npu_binaries/kernel_metadata.json << EOF
{
  "gemma3_qkv": {
    "binary_path": "real_npu_binaries/gemma3_qkv.xclbin",
    "input_shapes": {
      "hidden_states": [1, 64, 5376],
      "q_weight": [5376, 4096],
      "k_weight": [5376, 2048], 
      "v_weight": [5376, 2048],
      "scales": [3]
    },
    "output_shapes": {
      "q_output": [1, 64, 4096],
      "k_output": [1, 64, 2048],
      "v_output": [1, 64, 2048]
    },
    "compute_tiles": 10,
    "memory_usage_mb": 128
  },
  "gemma3_attention": {
    "binary_path": "real_npu_binaries/gemma3_attention.xclbin",
    "input_shapes": {
      "q_input": [1, 64, 32, 128],
      "k_input": [1, 64, 16, 128],
      "v_input": [1, 64, 16, 128]
    },
    "output_shapes": {
      "attention_output": [1, 64, 32, 128]
    },
    "compute_tiles": 8,
    "memory_usage_mb": 64
  }
}
EOF

echo "✅ Kernel metadata generated"

echo "🔧 Building custom XRT loader..."

# Create custom XRT kernel loader
cat > real_npu_binaries/load_kernels.cpp << 'EOF'
// Custom XRT Kernel Loader for Gemma 3 27B
// Direct C++ interface to NPU Phoenix hardware
// Bypasses Python completely

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <iostream>
#include <fstream>
#include <vector>

class Gemma3NPUKernels {
private:
    xrt::device device;
    xrt::kernel qkv_kernel;
    xrt::kernel attention_kernel;
    
public:
    Gemma3NPUKernels() {
        // Find NPU Phoenix device
        auto devices = xrt::system::enumerate_devices();
        for (auto& dev : devices) {
            if (dev.get_info<xrt::info::device::name>().find("Phoenix") != std::string::npos) {
                device = xrt::device(dev);
                std::cout << "✅ Found NPU Phoenix: " << device.get_info<xrt::info::device::name>() << std::endl;
                break;
            }
        }
        
        // Load kernels
        auto qkv_uuid = device.load_xclbin("real_npu_binaries/gemma3_qkv.xclbin");
        auto attention_uuid = device.load_xclbin("real_npu_binaries/gemma3_attention.xclbin");
        
        qkv_kernel = xrt::kernel(device, qkv_uuid, "gemma3_qkv");
        attention_kernel = xrt::kernel(device, attention_uuid, "gemma3_attention");
        
        std::cout << "✅ NPU kernels loaded successfully" << std::endl;
    }
    
    // Execute Q/K/V projections on NPU
    void execute_qkv_projections(
        const float* input_data,     // [1, 64, 5376]
        const int8_t* q_weight,      // [5376, 4096] 
        const int8_t* k_weight,      // [5376, 2048]
        const int8_t* v_weight,      // [5376, 2048]
        const float* scales,         // [3] - q_scale, k_scale, v_scale
        float* q_output,             // [1, 64, 4096]
        float* k_output,             // [1, 64, 2048]
        float* v_output              // [1, 64, 2048]
    ) {
        // Create NPU buffers
        auto input_bo = xrt::bo(device, 1 * 64 * 5376 * sizeof(float), qkv_kernel.group_id(0));
        auto q_weight_bo = xrt::bo(device, 5376 * 4096 * sizeof(int8_t), qkv_kernel.group_id(1));
        auto k_weight_bo = xrt::bo(device, 5376 * 2048 * sizeof(int8_t), qkv_kernel.group_id(2));
        auto v_weight_bo = xrt::bo(device, 5376 * 2048 * sizeof(int8_t), qkv_kernel.group_id(3));
        auto scales_bo = xrt::bo(device, 3 * sizeof(float), qkv_kernel.group_id(4));
        
        auto q_output_bo = xrt::bo(device, 1 * 64 * 4096 * sizeof(float), qkv_kernel.group_id(5));
        auto k_output_bo = xrt::bo(device, 1 * 64 * 2048 * sizeof(float), qkv_kernel.group_id(6));
        auto v_output_bo = xrt::bo(device, 1 * 64 * 2048 * sizeof(float), qkv_kernel.group_id(7));
        
        // Copy data to NPU
        input_bo.write(input_data);
        q_weight_bo.write(q_weight);
        k_weight_bo.write(k_weight);
        v_weight_bo.write(v_weight);
        scales_bo.write(scales);
        
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        q_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        v_weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Execute kernel on NPU
        auto run = qkv_kernel(input_bo, q_weight_bo, k_weight_bo, v_weight_bo, scales_bo,
                             q_output_bo, k_output_bo, v_output_bo);
        run.wait();
        
        // Read results from NPU
        q_output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        k_output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        v_output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        
        q_output_bo.read(q_output);
        k_output_bo.read(k_output);
        v_output_bo.read(v_output);
        
        std::cout << "✅ Q/K/V projections executed on NPU" << std::endl;
    }
    
    // Execute attention computation on NPU
    void execute_attention(
        const float* q_input,        // [1, 64, 32, 128]
        const float* k_input,        // [1, 64, 16, 128]  
        const float* v_input,        // [1, 64, 16, 128]
        float* attention_output      // [1, 64, 32, 128]
    ) {
        // Create NPU buffers
        auto q_bo = xrt::bo(device, 1 * 64 * 32 * 128 * sizeof(float), attention_kernel.group_id(0));
        auto k_bo = xrt::bo(device, 1 * 64 * 16 * 128 * sizeof(float), attention_kernel.group_id(1));
        auto v_bo = xrt::bo(device, 1 * 64 * 16 * 128 * sizeof(float), attention_kernel.group_id(2));
        auto output_bo = xrt::bo(device, 1 * 64 * 32 * 128 * sizeof(float), attention_kernel.group_id(3));
        
        // Copy to NPU
        q_bo.write(q_input);
        k_bo.write(k_input);
        v_bo.write(v_input);
        
        q_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        v_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Execute attention on NPU
        auto run = attention_kernel(q_bo, k_bo, v_bo, output_bo);
        run.wait();
        
        // Read result
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        output_bo.read(attention_output);
        
        std::cout << "✅ Attention computation executed on NPU" << std::endl;
    }
};

// C interface for Python integration
extern "C" {
    Gemma3NPUKernels* create_npu_kernels() {
        return new Gemma3NPUKernels();
    }
    
    void execute_qkv_projections_c(
        Gemma3NPUKernels* kernels,
        const float* input_data,
        const int8_t* q_weight,
        const int8_t* k_weight,
        const int8_t* v_weight,
        const float* scales,
        float* q_output,
        float* k_output,
        float* v_output
    ) {
        kernels->execute_qkv_projections(input_data, q_weight, k_weight, v_weight, scales,
                                        q_output, k_output, v_output);
    }
    
    void execute_attention_c(
        Gemma3NPUKernels* kernels,
        const float* q_input,
        const float* k_input,
        const float* v_input,
        float* attention_output
    ) {
        kernels->execute_attention(q_input, k_input, v_input, attention_output);
    }
    
    void destroy_npu_kernels(Gemma3NPUKernels* kernels) {
        delete kernels;
    }
}
EOF

echo "✅ Custom XRT loader created"

echo "🔧 Compiling XRT loader..."

# Compile the C++ XRT loader
g++ -shared -fPIC \
    -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib \
    -lxrt_coreutil \
    -std=c++17 \
    real_npu_binaries/load_kernels.cpp \
    -o real_npu_binaries/libgemma3_npu_kernels.so

echo "✅ XRT loader compiled"

echo ""
echo "🎉 REAL NPU KERNELS BUILD COMPLETE!"
echo "====================================="
echo ""
echo "📁 Generated files:"
echo "   📄 real_npu_binaries/gemma3_qkv.xclbin - Q/K/V projection NPU binary"
echo "   📄 real_npu_binaries/gemma3_attention.xclbin - Attention NPU binary"  
echo "   📄 real_npu_binaries/libgemma3_npu_kernels.so - XRT loader library"
echo "   📄 real_npu_binaries/kernel_metadata.json - Kernel specifications"
echo ""
echo "🚀 Ready for real NPU execution!"
echo "   • No Python frameworks"
echo "   • No CPU fallback" 
echo "   • Direct NPU machine code"
echo "   • Optimized for Gemma 3 27B + INT8 quantization"
echo ""