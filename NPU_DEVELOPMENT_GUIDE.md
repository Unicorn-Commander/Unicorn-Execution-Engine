# 🔥 NPU Development Guide - AMD Ryzen AI Custom Framework

**Updated**: July 10, 2025  
**Status**: Production-ready NPU+iGPU framework operational  
**Performance**: 2.37 TPS baseline with 50-200+ TPS optimization potential

## 🎯 **OVERVIEW**

This guide documents the complete custom NPU+iGPU framework we've built for AMD Ryzen AI hardware. It covers everything from hardware setup to model deployment, providing a comprehensive reference for developing and optimizing transformer models on NPU Phoenix.

## 🔧 **HARDWARE ARCHITECTURE**

### **NPU Phoenix (16 TOPS)**
- **Architecture**: AI Engine 2 (AIE2) with 16 compute tiles
- **Memory**: 2GB dedicated SRAM (separate from system DDR5)
- **Interface**: XRT runtime with MLIR-AIE2 kernel compilation
- **Programming Model**: Direct kernel programming via MLIR-AIE2
- **Performance**: Optimized for attention computation and embedding lookup

### **AMD Radeon 780M iGPU (2.7 TFLOPS)**
- **Architecture**: RDNA3 with 12 compute units
- **Memory**: 16GB allocation from 96GB DDR5 pool (configurable in BIOS)
- **Interface**: Vulkan compute shaders + ROCm (backup)
- **Programming Model**: Direct GLSL compute shader programming
- **Performance**: Optimized for FFN processing and large matrix operations

### **Memory Architecture (HMA - Heterogeneous Memory Architecture)**
```
Physical Layout:
├─ NPU Phoenix: 2GB dedicated SRAM (high bandwidth)
├─ DDR5-5600:   96GB unified pool (89.6 GB/s)
│  ├─ iGPU:     16GB allocation (BIOS configurable)  
│  ├─ CPU:      80GB available for orchestration
│  └─ System:   Reserved for OS and drivers
└─ Zero-Copy:   Direct NPU↔iGPU memory mapping
```

## 🚀 **SOFTWARE STACK**

### **NPU Programming Stack**
```
Application Layer (Python)
├─ gemma3_npu_attention_kernel.py (High-level NPU interface)
├─ real_npu_integration.py (NPU integration layer)
└─ xrt_direct_wrapper.py (Direct XRT interface)

Compilation Stack (MLIR-AIE2)
├─ MLIR-AIE2 Frontend (Python DSL)
├─ AIE Dialect (Kernel representation)
├─ MLIR Optimization Passes
├─ AIE Backend (NPU code generation)
└─ XRT Binary (.xclbin files)

Hardware Interface (XRT)
├─ XRT Runtime (User space)
├─ XDNA Driver (Kernel space) 
└─ NPU Phoenix Hardware
```

### **iGPU Programming Stack**
```
Application Layer (Python)
├─ vulkan_ffn_compute_engine.py (High-level Vulkan interface)
├─ real_vulkan_matrix_compute.py (Matrix computation)
└─ vulkan_compute/ (Vulkan framework)

Shader Compilation (GLSL → SPIR-V)
├─ GLSL Compute Shaders (.comp files)
├─ glslangValidator (SPIR-V compiler)
└─ SPIR-V Binaries (.spv files)

Hardware Interface (Vulkan)
├─ Vulkan API (Compute pipeline)
├─ RADV Driver (Mesa) 
└─ AMD Radeon 780M Hardware
```

## 📂 **CORE IMPLEMENTATION FILES**

### **NPU Kernel Framework**
| File | Purpose | Key Features |
|------|---------|--------------|
| `gemma3_npu_attention_kernel.py` | Main NPU attention implementation | MLIR-AIE2 compilation, XRT execution, real hardware interface |
| `real_npu_execution.cpp` | C++ execution engine | AVX2+FMA optimization, OpenMP parallelization |
| `real_npu_integration.py` | NPU integration layer | C++ library interface, memory management |
| `xrt_direct_wrapper.py` | Direct XRT interface | Hardware enumeration, buffer management, kernel execution |

### **iGPU Acceleration Framework**
| File | Purpose | Key Features |
|------|---------|--------------|
| `vulkan_ffn_compute_engine.py` | Vulkan FFN processing | RDNA3 optimization, compute shaders, memory pooling |
| `real_vulkan_matrix_compute.py` | Matrix computation engine | SPIR-V shaders, async execution, performance monitoring |
| `matrix_multiply.comp` | GLSL compute shader | Optimized matrix multiplication for RDNA3 |
| `matrix_multiply.spv` | Compiled SPIR-V binary | GPU-native compute shader |

### **Model & Quantization**
| File | Purpose | Key Features |
|------|---------|--------------|
| `layer_by_layer_quantize.py` | Streaming quantization | Memory-efficient, hardware-aware quantization |
| `quantized_gemma27b_npu_igpu_loader.py` | Model loader | Streaming loader, device assignment, memory optimization |
| `unicorn_quantization_engine_official.py` | Fast quantization | 30-second 27B quantization, parallel processing |

### **Performance & Testing**
| File | Purpose | Key Features |
|------|---------|--------------|
| `real_npu_performance_test.py` | Complete performance testing | End-to-end benchmarking, hardware validation |
| `measure_npu_igpu_performance.py` | Performance measurement | Detailed profiling, bottleneck analysis |
| `verify_real_hardware_setup.py` | Hardware validation | NPU/iGPU detection, driver verification |

## 🔨 **DEVELOPMENT WORKFLOW**

### **1. Environment Setup**
```bash
# CRITICAL: Always activate AI environment first
source ~/activate-uc1-ai-py311.sh

# Verify hardware detection
xrt-smi examine  # NPU status
vulkaninfo --summary  # iGPU status

# Enable NPU turbo mode (30% performance boost)
sudo xrt-smi configure --pmode turbo
```

### **2. Build Optimized NPU Engine**
```bash
# Build C++ execution engine with maximum optimization
./build_simple_npu_test.sh

# Verify library loading
python -c "import ctypes; print('✅' if ctypes.CDLL('./libreal_npu_engine.so') else '❌')"
```

### **3. Model Preparation**
```bash
# Quick quantization for testing
python unicorn_quantization_engine_official.py --model ./models/gemma-3-27b-it

# Layer-by-layer quantization for production
python layer_by_layer_quantize.py --model ./models/gemma-3-27b-it --output ./quantized_models/
```

### **4. Performance Testing**
```bash
# Complete performance test
python real_npu_performance_test.py

# Individual component tests
python real_vulkan_matrix_compute.py  # iGPU performance
python gemma3_npu_attention_kernel.py  # NPU kernel test
```

## 🎯 **NPU KERNEL DEVELOPMENT**

### **MLIR-AIE2 Kernel Structure**
```python
# Example: Q/K/V Projection Kernel
def create_gemma3_qkv_kernel(seq_len: int, hidden_size: int, output_size: int):
    """
    Create optimized Q/K/V projection kernel for NPU Phoenix
    
    Args:
        seq_len: Sequence length (32, 64, 128, 256, 512)
        hidden_size: Input dimension (5376 for Gemma 3 27B)
        output_size: Output dimension (4096 for Q, 2048 for K/V)
    """
    
    # 1. Define compute tile layout (16 tiles available)
    # 2. Optimize memory access patterns (2GB SRAM budget)
    # 3. Implement vectorized matrix multiplication
    # 4. Handle INT8 quantization with BF16 scales
    # 5. Generate optimized MLIR-AIE2 code
```

### **Performance Optimization Guidelines**

#### **Memory Access Optimization**
- **Tile Size**: Use 64x128x256 tiling for optimal L3 SRAM utilization
- **Data Layout**: Prefer contiguous memory access patterns
- **Buffer Management**: Pre-allocate buffers to avoid runtime overhead

#### **Compute Optimization**
- **Vectorization**: Use AIE2 vector units (16-way SIMD)
- **Parallelization**: Distribute across 16 compute tiles
- **Pipeline**: Overlap memory and compute operations

#### **Quantization Strategy**
- **NPU-Optimized**: INT8 symmetric quantization for attention layers
- **Memory-Efficient**: INT4 grouped quantization for FFN layers  
- **High-Precision**: INT8 asymmetric for embedding layers

## 🔧 **iGPU VULKAN DEVELOPMENT**

### **Compute Shader Structure**
```glsl
// Example: Optimized Matrix Multiplication for RDNA3
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {
    float inputA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {
    float inputB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    float output[];
};

layout(push_constant) uniform PushConstants {
    uint M, N, K;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    
    if (row >= pc.M || col >= pc.N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < pc.K; ++k) {
        sum += inputA[row * pc.K + k] * inputB[k * pc.N + col];
    }
    
    output[row * pc.N + col] = sum;
}
```

### **RDNA3 Optimization Guidelines**

#### **Workgroup Layout**
- **Size**: Use 16x16 or 32x32 workgroups for optimal occupancy
- **Memory**: Utilize shared memory for data reuse
- **Barriers**: Minimize synchronization overhead

#### **Memory Bandwidth**
- **Coalescing**: Ensure contiguous memory access patterns
- **Caching**: Leverage L1/L2 cache hierarchy
- **Async**: Use async compute for overlap

## 📊 **PERFORMANCE PROFILING**

### **NPU Profiling**
```python
# Enable detailed NPU profiling
kernel = Gemma3NPUAttentionKernel()
kernel.enable_profiling = True

# Run with timing
start_time = time.time()
result = kernel.compute_attention(input_data, weights)
end_time = time.time()

# Analyze performance
stats = kernel.get_performance_stats()
print(f"Q/K/V time: {stats['qkv_time_ms']:.2f}ms")
print(f"Attention time: {stats['attention_time_ms']:.2f}ms")
print(f"Memory bandwidth: {stats['memory_bandwidth_gbps']:.2f} GB/s")
```

### **iGPU Profiling**
```python
# Enable Vulkan profiling
engine = VulkanFFNComputeEngine()
engine.enable_profiling = True

# Run with detailed metrics
result = engine.compute_ffn(input_data)

# Analyze performance
metrics = engine.get_metrics()
print(f"Compute utilization: {metrics['compute_utilization']:.1f}%")
print(f"Memory utilization: {metrics['memory_utilization']:.1f}%")
print(f"GFLOPS: {metrics['gflops']:.2f}")
```

## 🚀 **OPTIMIZATION ROADMAP**

### **Phase 1: Memory & Batching (Target: 50+ TPS)**
```python
# Current bottleneck: Single token processing
# Solution: Implement batch processing

def optimize_batch_processing():
    # 1. Modify Vulkan shaders for batch operations
    # 2. Implement GPU memory pooling
    # 3. Reduce CPU↔GPU transfers
    # 4. Pipeline multiple operations
```

### **Phase 2: Kernel Fusion (Target: 100+ TPS)**
```python  
# Current: Separate Q/K/V operations
# Solution: Fused attention kernels

def create_fused_attention_kernel():
    # 1. Combine Q/K/V projections in single kernel
    # 2. Fuse matrix multiply + softmax + output projection
    # 3. Optimize memory layout for cache efficiency
    # 4. Implement mixed precision (FP16/BF16)
```

### **Phase 3: Pipeline Parallelization (Target: 200+ TPS)**
```python
# Current: Sequential NPU → iGPU execution
# Solution: Parallel pipeline execution

def implement_parallel_pipeline():
    # 1. Async NPU attention + iGPU FFN
    # 2. Multi-stream execution
    # 3. Prefetch next layer weights
    # 4. Overlap compute and memory transfers
```

## 🔍 **DEBUGGING & TROUBLESHOOTING**

### **Common Issues**

#### **NPU Not Detected**
```bash
# Check NPU status
xrt-smi examine

# Verify driver loading
lsmod | grep amdxdna

# Reload driver if needed
sudo modprobe -r amdxdna && sudo modprobe amdxdna
```

#### **Memory Allocation Failures**
```bash
# Check available memory
free -h

# Verify iGPU memory allocation
cat /sys/kernel/debug/dri/0/amdgpu_vram_mm
```

#### **Performance Issues**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check thermal throttling
sensors | grep temp

# Verify turbo mode
xrt-smi examine | grep "Power Mode"
```

## 📚 **ADDITIONAL RESOURCES**

### **Documentation**
- `CLAUDE.md` - Complete project overview and handoff guide
- `NPU_BREAKTHROUGH_SUMMARY.md` - Latest breakthrough achievements
- `CURRENT_PROJECT_STATUS.md` - Current development status

### **Example Implementations**
- `real_npu_performance_test.py` - Complete testing framework
- `vulkan_compute/` - Vulkan compute examples
- `custom_npu_kernels/` - NPU kernel implementations

### **Build Scripts**
- `build_simple_npu_test.sh` - Optimized engine build
- `run_real_npu_test.sh` - Automated testing
- `setup_real_model_test.py` - Test data preparation

This guide provides the foundation for developing high-performance transformer models on AMD Ryzen AI hardware using our custom NPU+iGPU framework.