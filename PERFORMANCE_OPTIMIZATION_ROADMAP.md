# Performance Optimization Roadmap
## From 2.5 TPS to 400-800 TPS Target

### 🎯 **Current Performance Status**
- **Baseline**: ~2.5 tokens/second (13.7s for 35 tokens)
- **Target**: 400-800 TPS  
- **Gap**: 160-320x improvement needed
- **Root Cause**: CPU fallback instead of full GPU/NPU execution

## 📊 **Performance Analysis & Bottlenecks**

### **Current Layer Processing Times:**
```
NPU Sparse Layers (0-9):  ~1010ms per layer (should be ~50ms)
iGPU Dense Layers (10-29): ~1325ms per layer (should be ~100ms)
Total Model Pass:          ~36 seconds (should be ~3 seconds)
```

### **Identified Bottlenecks:**
1. **PyTorch CUDA**: Using NVIDIA-targeted PyTorch on AMD hardware
2. **CPU Matrix Operations**: Falling back to numpy instead of GPU compute
3. **Memory Transfers**: Constant CPU ↔ GPU data movement
4. **Sequential Processing**: No pipeline parallelism
5. **Sparse Optimization**: Not leveraging 95% sparsity effectively

## 🚀 **Phase 1: ROCm Integration (Target: 10-20x speedup)**

### **Expected Outcome**: 25-50 TPS
### **Timeline**: 1-2 days
### **Effort**: Medium

#### **1.1 Replace PyTorch with ROCm Version**
```bash
# Current issue: NVIDIA PyTorch on AMD hardware
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Verify ROCm detection
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

#### **1.2 Update GPU Device Detection**
```python
# In igpu_optimization_engine.py
def _check_igpu_capabilities(self):
    # Add ROCm-specific detection
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "gfx1103" in device_name or "Radeon" in device_name:
            self.gpu_device = torch.device('cuda:0')  # ROCm uses cuda interface
            self.rocm_available = True
```

#### **1.3 Enable GPU Memory Persistence**
```python
# Keep weights in GPU memory between layers
class GPUMemoryManager:
    def __init__(self):
        self.cached_weights = {}
        
    def cache_layer_weights(self, layer_idx, weights):
        if layer_idx not in self.cached_weights:
            self.cached_weights[layer_idx] = {
                k: torch.from_numpy(v).cuda() for k, v in weights.items()
            }
```

#### **Expected Results:**
- **Layer Time**: 1325ms → 100-200ms (6-13x speedup)
- **Model TPS**: 2.5 → 15-30 TPS

## 🔥 **Phase 2: Real NPU Kernel Execution (Target: 5-10x on sparse)**

### **Expected Outcome**: 100-200 TPS  
### **Timeline**: 1-2 weeks
### **Effort**: High

#### **2.1 Implement Direct XRT Kernel Calls**
```cpp
// xrt_npu_kernels.cpp
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

class NPUSparseAttention {
    xrt::device device;
    xrt::kernel attention_kernel;
    
public:
    NPUSparseAttention() {
        device = xrt::device(0);  // NPU device
        auto uuid = device.load_xclbin("/path/to/attention_kernel.xclbin");
        attention_kernel = xrt::kernel(device, uuid, "sparse_attention");
    }
    
    void compute_sparse_attention(float* q, float* k, float* v, float* output, 
                                 int* sparse_mask, int batch_size, int seq_len) {
        // Direct NPU execution
        auto q_buffer = xrt::bo(device, q, batch_size * seq_len * 2048 * sizeof(float));
        auto output_buffer = xrt::bo(device, batch_size * seq_len * 2048 * sizeof(float));
        
        auto run = attention_kernel(q_buffer, k_buffer, v_buffer, output_buffer, sparse_mask);
        run.wait();
        
        output_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
};
```

#### **2.2 Optimize Sparse Computation**
```python
def sparse_attention_optimized(self, q, k, v, sparsity_mask):
    # Only compute 5% of attention matrix (95% sparse)
    non_zero_indices = torch.nonzero(sparsity_mask, as_tuple=False)
    
    # Sparse matrix multiplication
    sparse_scores = torch.sparse.mm(
        q.to_sparse(), 
        k.transpose(-2, -1).to_sparse()
    )
    
    # 20x speedup potential from avoiding 95% of computation
    return sparse_scores
```

#### **2.3 Memory Transfer Optimization**
```python
class NPUMemoryPool:
    def __init__(self, npu_memory_budget=2048):  # 2GB NPU memory
        self.memory_pool = {}
        self.budget = npu_memory_budget * 1024 * 1024  # Convert to bytes
        
    def allocate_layer_memory(self, layer_idx, tensor_shapes):
        # Pre-allocate NPU memory for sparse layers 0-9
        if layer_idx < 10:  # Sparse layers
            return self.allocate_npu_memory(tensor_shapes)
        else:
            return self.allocate_gpu_memory(tensor_shapes)
```

#### **Expected Results:**
- **Sparse Layer Time**: 1010ms → 100-200ms (5-10x speedup)
- **Overall TPS**: 30 → 100-150 TPS

## ⚡ **Phase 3: Pipeline & Memory Optimization (Target: 2-3x)**

### **Expected Outcome**: 200-400 TPS
### **Timeline**: 3-5 days  
### **Effort**: Medium

#### **3.1 Pipeline Parallelism**
```python
class PipelinedExecution:
    def __init__(self):
        self.npu_queue = asyncio.Queue()
        self.igpu_queue = asyncio.Queue()
        
    async def process_layer_pipeline(self, layer_idx, hidden_states):
        if layer_idx < 10:  # NPU sparse layers
            return await self.process_npu_async(layer_idx, hidden_states)
        else:  # iGPU dense layers
            return await self.process_igpu_async(layer_idx, hidden_states)
            
    async def process_model_pipelined(self, input_ids):
        # Overlap NPU and iGPU computation
        tasks = []
        for layer_idx in range(30):
            task = self.process_layer_pipeline(layer_idx, hidden_states)
            tasks.append(task)
            
        # Process layers with overlap
        results = await asyncio.gather(*tasks)
```

#### **3.2 Batch Processing**
```python
def process_batch_tokens(self, input_batch, batch_size=8):
    # Process multiple tokens simultaneously
    # Better GPU utilization
    batch_results = []
    
    for i in range(0, len(input_batch), batch_size):
        batch = input_batch[i:i+batch_size]
        batch_tensor = torch.stack(batch).cuda()
        
        # Process entire batch through model
        batch_output = self.forward_pass_batch(batch_tensor)
        batch_results.extend(batch_output)
        
    return batch_results
```

#### **Expected Results:**
- **Pipeline Speedup**: 2-3x from overlapped execution
- **Batch Speedup**: 1.5-2x from better GPU utilization
- **Combined TPS**: 150 → 300-400 TPS

## 🧠 **Phase 4: Advanced Algorithm Optimization (Target: 2-5x)**

### **Expected Outcome**: 400-800 TPS (TARGET ACHIEVED)
### **Timeline**: 1-2 weeks
### **Effort**: High

#### **4.1 Flash Attention Implementation**
```python
def flash_attention_igpu(self, q, k, v):
    # Memory-efficient attention for dense layers
    # Reduces memory bandwidth by 4x
    
    B, H, N, D = q.shape
    block_size = 64  # Fit in GPU cache
    
    output = torch.zeros_like(q)
    
    for i in range(0, N, block_size):
        q_block = q[:, :, i:i+block_size, :]
        
        # Compute attention for this block
        scores = torch.matmul(q_block, k.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = torch.softmax(scores, dim=-1)
        block_output = torch.matmul(attn_weights, v)
        
        output[:, :, i:i+block_size, :] = block_output
        
    return output
```

#### **4.2 KV-Cache Optimization**
```python
class KVCache:
    def __init__(self, max_seq_len=2048, num_layers=30):
        self.k_cache = {}
        self.v_cache = {}
        
    def update_cache(self, layer_idx, k, v, position):
        # Only compute new K,V for new tokens
        if layer_idx not in self.k_cache:
            self.k_cache[layer_idx] = torch.zeros(1, max_seq_len, 512).cuda()
            self.v_cache[layer_idx] = torch.zeros(1, max_seq_len, 512).cuda()
            
        # Update cache at current position
        self.k_cache[layer_idx][:, position] = k
        self.v_cache[layer_idx][:, position] = v
        
        return self.k_cache[layer_idx][:, :position+1], self.v_cache[layer_idx][:, :position+1]
```

#### **4.3 Speculative Decoding**
```python
class SpeculativeDecoding:
    def __init__(self, draft_model, target_model):
        self.draft_model = draft_model  # Smaller, faster model
        self.target_model = target_model  # Our optimized Gemma3n
        
    def generate_speculative(self, input_ids, num_speculative=4):
        # Generate multiple tokens with draft model
        draft_tokens = self.draft_model.generate(input_ids, max_new_tokens=num_speculative)
        
        # Verify with target model (parallel verification)
        verification = self.target_model.verify_tokens(input_ids, draft_tokens)
        
        # Accept verified tokens, reject rest
        accepted_tokens = verification.accepted_tokens
        return accepted_tokens  # 2-5x speedup for long sequences
```

#### **Expected Results:**
- **Flash Attention**: 1.5-2x speedup on dense layers
- **KV-Cache**: 3-5x speedup for generation
- **Speculative Decoding**: 2-4x speedup for long sequences
- **Combined TPS**: 400-800 TPS (TARGET ACHIEVED)

## 📋 **Implementation Priority Order**

### **Week 1: ROCm Integration**
1. Replace PyTorch with ROCm version
2. Fix GPU device detection
3. Enable GPU memory persistence
4. **Target**: 25-50 TPS

### **Week 2-3: NPU Kernel Optimization**  
1. Implement direct XRT calls
2. Optimize sparse computation
3. Memory transfer optimization
4. **Target**: 100-200 TPS

### **Week 4: Pipeline & Memory**
1. Pipeline parallelism
2. Batch processing
3. Memory pool optimization
4. **Target**: 200-400 TPS

### **Week 5-6: Advanced Algorithms**
1. Flash Attention
2. KV-Cache optimization
3. Speculative decoding
4. **Target**: 400-800 TPS ✅

## 🎯 **Success Metrics**

### **Performance Targets:**
- **Phase 1**: 25-50 TPS (10-20x improvement)
- **Phase 2**: 100-200 TPS (additional 2-4x)  
- **Phase 3**: 200-400 TPS (additional 2x)
- **Phase 4**: 400-800 TPS (additional 2x) **TARGET ACHIEVED**

### **Quality Metrics:**
- **API Response Time**: <1 second
- **Memory Usage**: Stay within NPU 2GB + iGPU 16GB budgets
- **Accuracy**: Maintain model output quality
- **Stability**: 99.9% uptime for production API

## 🏆 **Expected Final Performance**

With all optimizations:
- **Tokens Per Second**: 400-800 TPS
- **Response Latency**: 50-100ms per token
- **Throughput**: 20-40x faster than baseline
- **Hardware Utilization**: 
  - NPU: 80%+ for sparse layers
  - iGPU: 90%+ for dense layers
  - CPU: <20% (orchestration only)

**🦄 PERFORMANCE TARGET: ACHIEVABLE WITH SYSTEMATIC OPTIMIZATION ⚡**