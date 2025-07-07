# Vitis AI and MLIR-AIE Integration Guide

## Overview

This document provides detailed information about integrating Vitis AI and MLIR-AIE frameworks for AMD Ryzen AI NPU development, based on hands-on experience from the Whisper NPU project.

## Architecture Overview

### Component Relationship
```
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                      │
│             (Python/C++ Application)                   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Vitis AI Layer                       │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ Vitis AI Runtime│    │   Model Optimization       │ │
│  │  (VART)        │    │   Tools (VAI_Optimizer)    │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  MLIR-AIE (IRON)                       │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  AIE Dialect    │    │    Kernel Compilation      │ │
│  │   (MLIR)       │    │      Framework             │ │
│  └─────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     XRT Layer                          │
│            (Xilinx Runtime Environment)                │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Hardware (NPU)                       │
└─────────────────────────────────────────────────────────┘
```

## Vitis AI Integration

### What is Vitis AI?
Vitis AI is AMD/Xilinx's unified AI inference platform that provides:
- Model quantization and optimization
- Runtime libraries for inference
- Development tools for AI acceleration

### Installation and Setup

#### 1. Vitis AI Environment Setup
```bash
# Option 1: Docker-based setup (Recommended)
docker pull xilinx/vitis-ai:latest

# Run Vitis AI container
docker run -it \
    --device=/dev/accel/accel0 \
    -v /opt/xilinx/xrt:/opt/xilinx/xrt \
    -v $(pwd):/workspace \
    xilinx/vitis-ai:latest bash

# Inside container, activate Vitis AI
conda activate vitis-ai-pytorch

# Option 2: Native installation (Advanced)
git clone https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI
./docker_run.sh xilinx/vitis-ai:latest
```

#### 2. Environment Variables
```bash
# Essential Vitis AI environment variables
export VITIS_AI_ROOT=/opt/vitis_ai
export VAI_ROOT=$VITIS_AI_ROOT
export PATH=$VITIS_AI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$VITIS_AI_ROOT/lib:$LD_LIBRARY_PATH

# XRT integration
source /opt/xilinx/xrt/setup.sh
```

### Model Optimization with Vitis AI

#### 1. Model Quantization
```python
# Quantize ONNX model for NPU deployment
from vai_q_onnx import quantize_static

def quantize_model_for_npu(model_path, calibration_data_path, output_path):
    """Quantize ONNX model using Vitis AI quantizer"""
    
    # Quantization configuration
    config = {
        'input_nodes': ['input'],
        'output_nodes': ['output'],
        'op_types': ['Conv', 'MatMul', 'Gemm'],  # NPU-supported ops
        'per_channel': False,  # NPU prefers per-tensor quantization
        'reduce_range': True,  # Improve accuracy
        'activation_type': 'int8',
        'weight_type': 'int8'
    }
    
    # Perform quantization
    quantized_model = quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=create_calibration_reader(calibration_data_path),
        quant_format='QOperator',
        **config
    )
    
    return quantized_model

def create_calibration_reader(data_path):
    """Create calibration data reader for quantization"""
    import onnxruntime as ort
    
    class CalibrationDataReader:
        def __init__(self, data_path):
            self.data = np.load(data_path)
            self.index = 0
        
        def get_next(self):
            if self.index >= len(self.data):
                return None
            
            batch = {'input': self.data[self.index]}
            self.index += 1
            return batch
    
    return CalibrationDataReader(data_path)
```

#### 2. Model Compilation for NPU
```python
# Compile quantized model for NPU deployment
def compile_for_npu(quantized_model_path, arch_json_path, output_dir):
    """Compile quantized model for NPU using Vitis AI compiler"""
    
    # Compilation command for AMD NPU
    compile_cmd = [
        'vai_c_xcompiler',
        '--xmodel', quantized_model_path,
        '--arch', arch_json_path,  # NPU architecture file
        '--net_name', 'npu_model',
        '--output_dir', output_dir
    ]
    
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")
    
    return os.path.join(output_dir, 'npu_model.xmodel')
```

### NPU Architecture Configuration

#### Architecture JSON for AMD Phoenix NPU
```json
{
  "target": "AMD_Phoenix_NPU",
  "dcfg": {
    "PE": 6,
    "batch": 1,
    "arch": "phoenix",
    "freq": 400
  },
  "fingerprint": "0x1000020F6014406",
  "dpu": {
    "name": "Phoenix_NPU",
    "full_name": "AMD Phoenix NPU",
    "device_id": "0x1506",
    "device_core_id": "0x01",
    "target": "phoenix_npu",
    "isa": {
      "version": "v1.0",
      "fmap_bank": 16,
      "fmap_wgt": 16,
      "param_bank": 8,
      "param_wgt": 16,
      "bank_depth": 2048,
      "bank_width": 64,
      "channel_parallel": 32,
      "pixel_parallel": 4
    }
  }
}
```

## MLIR-AIE (IRON) Framework Integration

### Understanding MLIR-AIE

MLIR-AIE is the low-level compilation framework for AMD AI Engines, providing:
- AIE dialect for MLIR
- Kernel compilation and optimization
- Hardware-specific code generation
- Memory management and data movement

### Installation Process

#### 1. Complete MLIR-AIE Setup
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Create isolated Python environment
python -m venv ironenv
source ironenv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install pybind11 numpy

# Build LLVM-AIE (Peano toolchain)
./utils/clone-llvm.sh
./utils/build-llvm.sh -j$(nproc)

# Build MLIR-AIE
mkdir build && cd build
cmake .. \
    -DLLVM_DIR=../llvm/build/lib/cmake/llvm \
    -DMLIR_DIR=../llvm/build/lib/cmake/mlir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

make -j$(nproc)

# Install Python bindings
cd ../python_bindings/mlir_aie
pip install -e .
```

#### 2. Environment Setup Script
```bash
#!/bin/bash
# setup_mlir_aie.sh

# MLIR-AIE environment setup
export MLIR_AIE_ROOT=$(pwd)
export LLVM_BUILD_DIR=$MLIR_AIE_ROOT/llvm/build

# Activate Python environment
source $MLIR_AIE_ROOT/ironenv/bin/activate

# Set MLIR-AIE paths
export PATH=$MLIR_AIE_ROOT/build/bin:$LLVM_BUILD_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MLIR_AIE_ROOT/build/lib:$LLVM_BUILD_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MLIR_AIE_ROOT/python_bindings:$PYTHONPATH

# AIE-specific environment variables
export AIE_INCLUDE_PATH=$MLIR_AIE_ROOT/runtime_lib/x86_64/test_lib/include
export AIE_LIBRARY_PATH=$MLIR_AIE_ROOT/runtime_lib/x86_64/test_lib/lib

echo "MLIR-AIE environment activated"
```

### Custom Kernel Development

#### 1. AIE Kernel Template
```cpp
// matrix_multiply_aie.cpp - Custom NPU matrix multiplication kernel
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

using namespace adf;

// Matrix multiplication kernel for NPU
void matrix_multiply_kernel(
    input_stream<float> *__restrict in_a,
    input_stream<float> *__restrict in_b,
    output_stream<float> *__restrict out_c
) {
    
    // Local memory for tiles
    alignas(32) float tile_a[64][64];
    alignas(32) float tile_b[64][64];
    alignas(32) float tile_c[64][64];
    
    // Initialize accumulator
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            tile_c[i][j] = 0.0f;
        }
    }
    
    // Load input tiles
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            tile_a[i][j] = readincr(in_a);
            tile_b[i][j] = readincr(in_b);
        }
    }
    
    // Perform matrix multiplication using AIE vector instructions
    for (int i = 0; i < 64; i += 8) {  // Process 8 rows at a time
        for (int j = 0; j < 64; j += 8) {  // Process 8 columns at a time
            for (int k = 0; k < 64; k += 8) {  // 8-way dot product
                
                // Load vectors
                aie::vector<float, 8> va = aie::load_v<8>(&tile_a[i][k]);
                aie::vector<float, 8> vb = aie::load_v<8>(&tile_b[k][j]);
                
                // Accumulate using MAC (Multiply-Accumulate)
                aie::accum<accfloat, 8> acc;
                acc = aie::load_v<8>(&tile_c[i][j]);
                acc = aie::mac(acc, va, vb);
                
                // Store result
                aie::store_v(&tile_c[i][j], acc.to_vector<float>());
            }
        }
    }
    
    // Write output
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            writeincr(out_c, tile_c[i][j]);
        }
    }
}
```

#### 2. AIE Graph Definition
```cpp
// matrix_multiply_graph.cpp - AIE graph for matrix multiplication
#include <adf.h>
#include "matrix_multiply_kernel.h"

using namespace adf;

class MatrixMultiplyGraph : public graph {
private:
    kernel k_matmul;
    
public:
    input_plio in_a, in_b;
    output_plio out_c;
    
    MatrixMultiplyGraph() {
        // Create kernel
        k_matmul = kernel::create(matrix_multiply_kernel);
        
        // Set kernel properties
        source(k_matmul) = "matrix_multiply_kernel.cpp";
        runtime<ratio>(k_matmul) = 0.9;  // 90% runtime utilization
        
        // Create PLIO connections
        in_a = input_plio::create("DataIn_A", plio_64_bits, "data/input_a.txt");
        in_b = input_plio::create("DataIn_B", plio_64_bits, "data/input_b.txt");
        out_c = output_plio::create("DataOut_C", plio_64_bits, "data/output_c.txt");
        
        // Connect streams
        connect<stream>(in_a.out[0], k_matmul.in[0]);
        connect<stream>(in_b.out[0], k_matmul.in[1]);
        connect<stream>(k_matmul.out[0], out_c.in[0]);
    }
};
```

#### 3. Python Integration
```python
# aie_matrix_multiply.py - Python wrapper for AIE kernel
import numpy as np
from mlir_aie import *

class AIEMatrixMultiplier:
    """Python wrapper for AIE matrix multiplication kernel"""
    
    def __init__(self):
        self.graph = None
        self.compiled = False
    
    def compile_kernel(self):
        """Compile AIE kernel for NPU"""
        
        # Create MLIR-AIE context
        with Context():
            # Define AIE module
            module = Module.create()
            
            with InsertionPoint(module.body):
                # Define AIE device
                dev = AIEDevice.create()
                
                # Create tile for computation
                tile = dev.tile(1, 1)  # Use tile (1,1) for computation
                
                # Define core and memory
                core = dev.core(tile)
                mem = dev.mem(tile)
                
                # Create buffer for matrices
                buf_a = dev.buffer(mem, np.float32, [64, 64])
                buf_b = dev.buffer(mem, np.float32, [64, 64])
                buf_c = dev.buffer(mem, np.float32, [64, 64])
                
                # Define kernel function
                @FuncOp.from_py_func(
                    T.memref(64, 64, T.f32()),
                    T.memref(64, 64, T.f32()),
                    T.memref(64, 64, T.f32())
                )
                def matrix_multiply(a, b, c):
                    """MLIR function for matrix multiplication"""
                    # Generate nested loops for matrix multiplication
                    for i in range(64):
                        for j in range(64):
                            for k in range(64):
                                # c[i][j] += a[i][k] * b[k][j]
                                a_val = memref.load(a, [i, k])
                                b_val = memref.load(b, [k, j])
                                c_val = memref.load(c, [i, j])
                                
                                prod = arith.mulf(a_val, b_val)
                                sum_val = arith.addf(c_val, prod)
                                
                                memref.store(sum_val, c, [i, j])
                
                # Add function to module
                module.body.append(matrix_multiply)
        
        # Compile to NPU
        compiled_module = compile_aie_module(module, target="npu")
        self.graph = compiled_module
        self.compiled = True
    
    def execute(self, matrix_a, matrix_b):
        """Execute matrix multiplication on NPU"""
        
        if not self.compiled:
            self.compile_kernel()
        
        # Prepare input data
        a_data = matrix_a.astype(np.float32).flatten()
        b_data = matrix_b.astype(np.float32).flatten()
        
        # Execute on NPU
        result = self.graph.run([a_data, b_data])
        
        # Reshape output
        output_matrix = result[0].reshape(64, 64)
        return output_matrix
```

### Integration Challenges and Solutions

#### 1. Vitis AI and MLIR-AIE Compatibility
**Challenge**: Different compilation toolchains and runtimes.

**Solution**: Hybrid approach with clear separation:
```python
class HybridNPURuntime:
    """Hybrid runtime supporting both Vitis AI and MLIR-AIE"""
    
    def __init__(self):
        self.vitis_ai_runtime = None
        self.mlir_aie_runtime = None
        self.operation_map = {}
    
    def initialize(self):
        """Initialize both runtimes"""
        
        # Initialize Vitis AI for high-level operations
        try:
            from vart import Runner
            self.vitis_ai_runtime = VitisAIRunner()
            logger.info("Vitis AI runtime initialized")
        except ImportError:
            logger.warning("Vitis AI not available")
        
        # Initialize MLIR-AIE for custom kernels
        try:
            self.mlir_aie_runtime = MLIRAIERunner()
            logger.info("MLIR-AIE runtime initialized")
        except ImportError:
            logger.warning("MLIR-AIE not available")
    
    def register_operation(self, op_name, implementation_type):
        """Register operation with preferred implementation"""
        self.operation_map[op_name] = implementation_type
    
    def execute_operation(self, op_name, *args, **kwargs):
        """Execute operation using appropriate runtime"""
        
        impl_type = self.operation_map.get(op_name, 'auto')
        
        if impl_type == 'vitis_ai' and self.vitis_ai_runtime:
            return self.vitis_ai_runtime.run(op_name, *args, **kwargs)
        elif impl_type == 'mlir_aie' and self.mlir_aie_runtime:
            return self.mlir_aie_runtime.run(op_name, *args, **kwargs)
        else:
            # Auto-selection based on operation characteristics
            return self._auto_select_runtime(op_name, *args, **kwargs)
```

#### 2. Memory Management Between Frameworks
**Challenge**: Different memory layouts and allocation strategies.

**Solution**: Unified memory manager:
```python
class UnifiedNPUMemoryManager:
    """Unified memory management for Vitis AI and MLIR-AIE"""
    
    def __init__(self):
        self.memory_pools = {
            'vitis_ai': [],
            'mlir_aie': [],
            'shared': []
        }
        self.allocation_strategy = 'shared'  # 'shared', 'isolated', 'adaptive'
    
    def allocate_buffer(self, size, dtype, framework='shared'):
        """Allocate memory buffer compatible with both frameworks"""
        
        # Align to NPU memory requirements (typically 64-byte aligned)
        aligned_size = ((size * dtype().itemsize + 63) // 64) * 64
        
        if framework == 'shared':
            # Use memory-mapped buffer for zero-copy sharing
            buffer = np.memmap(
                tempfile.NamedTemporaryFile(),
                dtype=dtype,
                mode='w+',
                shape=(size,)
            )
        else:
            # Framework-specific allocation
            buffer = np.empty(size, dtype=dtype)
        
        self.memory_pools[framework].append(buffer)
        return buffer
    
    def transfer_data(self, data, src_framework, dst_framework):
        """Transfer data between framework memory spaces"""
        
        if src_framework == dst_framework:
            return data  # No transfer needed
        
        # Create compatible buffer in destination framework
        dst_buffer = self.allocate_buffer(
            data.size, 
            data.dtype, 
            dst_framework
        )
        
        # Copy data
        dst_buffer[:] = data[:]
        return dst_buffer
```

### Performance Comparison

#### Benchmarking Framework Integration
```python
def benchmark_frameworks():
    """Compare performance between Vitis AI and MLIR-AIE approaches"""
    
    test_cases = [
        ('matrix_multiply', (1024, 1024), (1024, 1024)),
        ('convolution', (1, 3, 224, 224), (64, 3, 7, 7)),
        ('attention', (1, 512, 768), None)
    ]
    
    results = {}
    
    for operation, input_shape, weight_shape in test_cases:
        results[operation] = {}
        
        # Generate test data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*weight_shape).astype(np.float32) if weight_shape else None
        
        # Test Vitis AI
        if vitis_ai_available:
            start_time = time.perf_counter()
            vitis_result = run_vitis_ai_operation(operation, input_data, weight_data)
            vitis_time = time.perf_counter() - start_time
            results[operation]['vitis_ai'] = vitis_time
        
        # Test MLIR-AIE
        if mlir_aie_available:
            start_time = time.perf_counter()
            mlir_result = run_mlir_aie_operation(operation, input_data, weight_data)
            mlir_time = time.perf_counter() - start_time
            results[operation]['mlir_aie'] = mlir_time
        
        # Test CPU baseline
        start_time = time.perf_counter()
        cpu_result = run_cpu_operation(operation, input_data, weight_data)
        cpu_time = time.perf_counter() - start_time
        results[operation]['cpu'] = cpu_time
    
    return results
```

### Best Practices for Integration

#### 1. Development Workflow
1. **Start with Vitis AI**: Use for standard operations and model deployment
2. **Custom kernels with MLIR-AIE**: Implement performance-critical operations
3. **Hybrid deployment**: Combine both frameworks for optimal performance
4. **Continuous profiling**: Monitor performance and adjust runtime selection

#### 2. Debugging Strategy
```python
class IntegratedDebugger:
    """Debugging support for both Vitis AI and MLIR-AIE"""
    
    def __init__(self):
        self.debug_mode = os.getenv('NPU_DEBUG', '0') == '1'
        self.profiling_enabled = os.getenv('NPU_PROFILE', '0') == '1'
    
    def debug_vitis_ai(self, operation, inputs, outputs):
        """Debug Vitis AI operations"""
        if self.debug_mode:
            print(f"Vitis AI: {operation}")
            print(f"Input shapes: {[x.shape for x in inputs]}")
            print(f"Output shapes: {[x.shape for x in outputs]}")
    
    def debug_mlir_aie(self, kernel_name, compilation_info):
        """Debug MLIR-AIE kernel compilation"""
        if self.debug_mode:
            print(f"MLIR-AIE Kernel: {kernel_name}")
            print(f"Compilation info: {compilation_info}")
    
    def profile_operation(self, framework, operation, execution_time):
        """Profile framework operations"""
        if self.profiling_enabled:
            with open('npu_profile.log', 'a') as f:
                f.write(f"{framework},{operation},{execution_time}\n")
```

This integration guide provides a comprehensive foundation for working with both Vitis AI and MLIR-AIE frameworks in NPU development, enabling developers to leverage the strengths of each approach for optimal performance.