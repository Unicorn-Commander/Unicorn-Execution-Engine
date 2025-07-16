# 🚀 OPTIMIZATION ROADMAP TO 50 TPS - Gemma 3 27B

**Target**: 50 tokens per second for Gemma 3 27B on NPU+iGPU  
**Current Baseline**: 0.3-0.5 TPS (before optimizations)  
**Expected After Today**: 5-15 TPS (optimizations implemented, ready to test)  
**Ultimate Goal**: 50+ TPS with full optimization  
**Hardware**: NPU Phoenix (16 TOPS) + AMD Radeon 780M (8.9 TFLOPS)

---

## 📊 **PERFORMANCE ANALYSIS**

### **Current Bottlenecks Identified**
1. **Q/K/V Projections**: 22-23 seconds (critical bottleneck!)
2. **Single Token Processing**: No batching implemented
3. **Memory Allocation**: Repeated allocation/deallocation per inference
4. **Hardware Utilization**: NPU ~7%, iGPU ~30-40% (severely underutilized)
5. **No Kernel Fusion**: Separate operations for Q, K, V

### **Hardware Theoretical Limits**
- **NPU Phoenix**: 16 TOPS = 16 trillion operations/second
- **AMD Radeon 780M**: 8.9 TFLOPS = 8.9 trillion FP32 ops/second
- **Memory Bandwidth**: 89.6 GB/s DDR5-5600
- **Model Size**: 26GB quantized (INT8/INT4)

### **Performance Calculation**
```
Gemma 3 27B operations per token:
- Attention: ~54B ops (27B params × 2)
- FFN: ~108B ops (27B params × 4)
- Total: ~162B ops/token

Theoretical max with hardware:
- NPU (16 TOPS) + iGPU (8.9 TFLOPS) = ~25 TFLOPS combined
- 25 TFLOPS / 162B ops = ~154 tokens/second theoretical max
- Realistic target: 50 TPS (32% efficiency)
```

---

## 🎯 **OPTIMIZATION PHASES**

### **Phase 1: Immediate Wins (Target: 15-25 TPS)**
**Timeline**: 1-2 days  
**Expected Gain**: 3-5x

#### 1.1 Fix Q/K/V Projection Bottleneck
```python
# Current: Sequential processing taking 22-23s
# Solution: Batch all three projections together
# File: pure_hardware_pipeline.py

# Before:
q = self.process_projection(hidden_states, q_weight)  # 7-8s
k = self.process_projection(hidden_states, k_weight)  # 7-8s  
v = self.process_projection(hidden_states, v_weight)  # 7-8s

# After:
qkv = self.process_qkv_fused(hidden_states, qkv_weight)  # <1s
q, k, v = qkv.split(3, dim=-1)
```

#### 1.2 Enable Persistent GPU Buffers
```python
# File: real_vulkan_matrix_compute.py
class VulkanMatrixCompute:
    def __init__(self):
        self.persistent_buffers = {}
        self.buffer_pool = self._create_buffer_pool()
    
    def _create_buffer_pool(self):
        # Pre-allocate 2GB of GPU buffers
        return {
            'small': [self._allocate_buffer(1024*1024) for _ in range(32)],
            'medium': [self._allocate_buffer(16*1024*1024) for _ in range(16)],
            'large': [self._allocate_buffer(256*1024*1024) for _ in range(8)]
        }
```

#### 1.3 Implement Basic Batching
```python
# File: pure_hardware_api_server.py
# Change from single token to batch processing
batch_size = min(len(tokens), 8)  # Start with small batches
```

### **Phase 2: Core Optimizations (Target: 30-40 TPS)**
**Timeline**: 3-5 days  
**Expected Gain**: 2-3x

#### 2.1 Advanced Batching System
```python
# File: batch_inference_engine.py
class BatchInferenceEngine:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.kv_cache = KVCache(max_batch_size=batch_size)
        
    def process_batch(self, tokens: List[List[int]]) -> List[List[int]]:
        # Pad sequences to same length
        # Process entire batch through NPU+iGPU
        # Return generated tokens for all sequences
```

#### 2.2 NPU Kernel Optimization
```python
# File: npu_attention_kernel_optimized.py
# Implement flash attention variant for NPU
# Use INT8 quantization for attention computation
# Enable multi-head parallel processing
```

#### 2.3 Memory Pipeline Optimization
- Implement double buffering (process batch N while loading batch N+1)
- Use pinned memory for CPU↔GPU transfers
- Enable HMA zero-copy for NPU↔iGPU transfers

### **Phase 3: Advanced Optimizations (Target: 50+ TPS)**
**Timeline**: 1-2 weeks  
**Expected Gain**: 1.5-2x

#### 3.1 Dynamic Quantization
```python
# File: dynamic_quantization_engine.py
class DynamicQuantizer:
    def quantize_activations(self, tensor, bits=8):
        # Quantize activations to INT8 on-the-fly
        # Keep critical operations in FP16
        # Use INT4 for less sensitive operations
```

#### 3.2 Speculative Decoding
```python
# File: speculative_decoder.py
# Use smaller draft model for speculation
# Verify with main model in batches
# Accept/reject tokens based on probability
```

#### 3.3 Hardware-Specific Tuning
- Profile and optimize for RDNA3 architecture
- Use AMD-specific intrinsics
- Optimize memory access patterns for 64B cache lines
- Enable wave32 mode for better occupancy

---

## 🔧 **IMPLEMENTATION DETAILS**

### **Critical Files to Modify**

1. **pure_hardware_pipeline.py**
   - Add QKV fusion
   - Implement batch processing
   - Add persistent buffer management

2. **real_vulkan_matrix_compute.py**
   - Add buffer pooling
   - Optimize shader dispatch
   - Enable async execution

3. **npu_attention_kernel_real.py**
   - Optimize for batch processing
   - Add INT8 attention path
   - Enable KV cache

4. **Create New Files:**
   - `batch_inference_engine.py`
   - `kv_cache_manager.py`
   - `dynamic_quantization_engine.py`
   - `speculative_decoder.py`

### **Testing Strategy**
```bash
# Progressive testing approach
python test_qkv_fusion.py           # Verify fusion works
python test_batch_inference.py      # Test batching
python benchmark_optimized_tps.py   # Measure TPS improvement
```

---

## 📈 **EXPECTED PERFORMANCE PROGRESSION**

| Phase | Implementation | Expected TPS | Timeline |
|-------|---------------|--------------|----------|
| Baseline | Current state | 0.3-0.5 | - |
| Today's Optimizations | Already implemented | 5-15 | Ready to test |
| Phase 1 | Q/K/V fix + buffers | 15-25 | 1-2 days |
| Phase 2 | Batching + NPU opt | 30-40 | 3-5 days |
| Phase 3 | Advanced techniques | 50+ | 1-2 weeks |

---

## 🚦 **SUCCESS METRICS**

### **Minimum Viable Performance (MVP)**
- ✅ 15 TPS (30x improvement from baseline)
- ✅ <2s first token latency
- ✅ Stable memory usage
- ✅ No thermal throttling

### **Target Performance**
- 🎯 50 TPS sustained
- 🎯 <500ms first token latency
- 🎯 NPU utilization >70%
- 🎯 iGPU utilization >80%

### **Stretch Goals**
- 🚀 100 TPS with INT4 quantization
- 🚀 200 TPS with speculative decoding
- 🚀 <100ms first token latency

---

## 🛠️ **QUICK START COMMANDS**

```bash
# Test current optimizations
source /home/ucadmin/activate-uc1-ai-py311.sh
python pure_hardware_api_server.py

# Implement Phase 1 optimizations
python implement_qkv_fusion.py
python enable_buffer_pooling.py
python test_basic_batching.py

# Benchmark improvements
python benchmark_optimized_tps.py --batch-size 8
python benchmark_optimized_tps.py --batch-size 32
```

---

## 📝 **NOTES**

- The Q/K/V projection bottleneck is the #1 priority - fixing this alone could give 20x speedup
- Batching is essential for reaching 50 TPS - single token processing is inherently inefficient
- The hardware is capable of 150+ TPS theoretically, so 50 TPS is a conservative target
- Memory bandwidth (89.6 GB/s) is sufficient for 50 TPS with 26GB model

---

*Created: July 13, 2025*  
*Target: 50 TPS for Gemma 3 27B on NPU+iGPU*