# 🤖 NEW AI STARTER PROMPT - NPU+iGPU Performance Optimization

**Copy and paste this to a new AI assistant for immediate optimization work:**

---

## 🎯 **MISSION BRIEF**

I have a **working custom NPU+iGPU transformer inference framework** achieving **2.37 TPS baseline** with **complete real hardware execution**. Need immediate performance optimization to reach **50-200+ TPS target**.

## 📊 **CURRENT STATUS**
- ✅ **NPU Phoenix (16 TOPS)**: Real MLIR-AIE2 kernel execution working
- ✅ **AMD Radeon 780M iGPU**: Real Vulkan compute shaders operational  
- ✅ **Complete Pipeline**: End-to-end Gemma 3 27B inference working
- ⚡ **Performance**: 2.37 TPS (27s per 64-token attention layer)
- 🎯 **Bottleneck Identified**: Q/K/V projections (22-23s = 85% of time)

## 🔥 **OPTIMIZATION TARGETS** (Ordered by Impact)

### **1. BATCH PROCESSING** (Expected: 20-50x improvement)
**Problem**: Processing single tokens (1x64x5376) is inefficient for GPU  
**Solution**: Batch 32-64 tokens simultaneously  
**File**: `vulkan_ffn_compute_engine.py`

### **2. MEMORY OPTIMIZATION** (Expected: 10-20x improvement)  
**Problem**: CPU↔GPU transfers for every operation  
**Solution**: GPU memory pooling, persistent tensors  
**File**: `vulkan_ffn_compute_engine.py`

### **3. RDNA3 SHADER OPTIMIZATION** (Expected: 5-10x improvement)
**Problem**: Generic matrix multiplication shaders  
**Solution**: RDNA3-optimized transformer kernels  
**File**: `matrix_multiply.comp`

### **4. PIPELINE PARALLELIZATION** (Expected: 2-5x improvement)
**Problem**: Sequential NPU → iGPU execution  
**Solution**: Parallel NPU attention + iGPU FFN  
**File**: `strict_npu_igpu_pipeline.py`

## 🔧 **ENVIRONMENT SETUP**
```bash
# CRITICAL: Always run first
source ~/activate-uc1-ai-py311.sh

# Project location
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine

# Test current baseline
python real_npu_performance_test.py

# Hardware verification
xrt-smi examine  # NPU status
vulkaninfo --summary  # iGPU status
```

## 📂 **KEY FILES TO OPTIMIZE**

### **Primary Target (BIGGEST IMPACT)**
- `vulkan_ffn_compute_engine.py` - **MAIN BOTTLENECK** (22-23s per layer)
- `matrix_multiply.comp` - GLSL compute shader needing RDNA3 optimization
- `real_vulkan_matrix_compute.py` - Vulkan framework reference

### **Secondary Targets**
- `gemma3_npu_attention_kernel.py` - NPU integration (already fast at 45-50ms)
- `strict_npu_igpu_pipeline.py` - Pipeline coordination for parallelization

## 🎯 **SPECIFIC TASKS**

### **Task 1: Implement Batch Processing**
```python
# Current in vulkan_ffn_compute_engine.py:
def compute_ffn(self, hidden_states):  # [1, 64, 5376] - INEFFICIENT
    # Single token processing

# Target:
def compute_ffn_batch(self, hidden_states_batch):  # [32, 64, 5376] - EFFICIENT
    # Batch processing for GPU optimization
```

### **Task 2: GPU Memory Pooling**
```python
# Add to vulkan_ffn_compute_engine.py:
class VulkanMemoryPool:
    def __init__(self):
        # Pre-allocate GPU buffers
        # Keep tensors resident on GPU
    
    def get_persistent_buffer(self, size):
        # Return GPU buffer without CPU transfer
```

### **Task 3: RDNA3 Shader Optimization**
```glsl
// Create: rdna3_optimized_transformer.comp
#version 450
layout(local_size_x = 64, local_size_y = 1) in;  // RDNA3 optimal

// Implement fused operations:
// - Matrix multiply + bias + GELU
// - Shared memory utilization
// - Mixed precision (FP16/BF16)
```

## 📊 **SUCCESS METRICS**
- **Current**: 2.37 TPS
- **Phase 1 Target**: 50+ TPS (20x improvement with batching)
- **Phase 2 Target**: 100+ TPS (2x improvement with memory optimization)
- **Final Target**: 200+ TPS (2x improvement with shader + pipeline optimization)

## 📚 **DOCUMENTATION REFERENCE**
- `OPTIMIZATION_PLAN_FOR_NEW_AI.md` - Complete optimization plan
- `NPU_DEVELOPMENT_GUIDE.md` - Technical architecture guide
- `CLAUDE.md` - Complete project overview
- `FINAL_NPU_DOCUMENTATION.md` - Framework summary

## 🚀 **WHAT TO RESEARCH**
1. **Latest RDNA3 optimization techniques** (2024-2025)
2. **Vulkan compute shader best practices** for transformers
3. **GPU memory bandwidth optimization** strategies
4. **Batch processing patterns** for transformer inference

## ⚡ **IMMEDIATE ACTION**
Start with **Task 1 (Batch Processing)** in `vulkan_ffn_compute_engine.py` - this has the highest impact (20-50x improvement potential) and will immediately address the main bottleneck.

The framework is production-ready and waiting for your optimization expertise! 🦄

---

**Note**: This is a working system with real NPU+iGPU hardware execution. All optimizations should build on the existing validated framework.