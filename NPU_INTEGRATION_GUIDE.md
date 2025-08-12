# 🧠 NPU Integration Guide for Unicorn Engine

## 📊 When to Use NPU vs iGPU-Only

### Models That Benefit from NPU

#### 1. **MoE Models (Mixture of Experts)**
- **Example**: Qwen3-30B-A3B
- **Why NPU Helps**:
  - Router computation on NPU (low precision, high throughput)
  - Expert selection benefits from NPU's INT8/INT4 optimization
  - Reduces memory bandwidth pressure on iGPU
  - NPU handles routing while iGPU processes selected experts

#### 2. **Large Dense Models (>7B parameters)**
- **Example**: Qwen2.5-32B (if we test it)
- **Why NPU Helps**:
  - Attention computation offload to NPU
  - NPU's 16 TOPS for INT8 operations
  - Parallel NPU+iGPU execution

#### 3. **Models with Structured Sparsity**
- **Why NPU Helps**:
  - NPU efficient at sparse operations
  - Can handle pruned weights effectively

### Models Better Suited for iGPU-Only

#### 1. **Small Dense Models (<7B parameters)**
- **Examples**: Phi-4-mini (3.8B), Granite-3.3 (8B)
- **Why iGPU-Only**:
  - Overhead of NPU coordination not worth it
  - Models fit entirely in VRAM
  - iGPU has sufficient compute for small models
  - Simpler implementation and debugging

#### 2. **Models Requiring High Precision**
- **Why iGPU-Only**:
  - NPU optimized for INT8/INT4
  - iGPU better for FP16/FP32 operations

## 🔧 Implementation Strategy

### For NPU-Enabled Models

```python
# Modular design - create both versions
class ModelPipeline:
    def __init__(self, use_npu=True):
        self.use_npu = use_npu and self._check_npu_available()
        
        if self.use_npu:
            self.attention_compute = NPUAttentionKernel()
            self.router_compute = NPURouterKernel()  # For MoE
        else:
            self.attention_compute = VulkanAttentionCompute()
            self.router_compute = VulkanRouterCompute()
```

### NPU Operations to Implement

1. **Attention Computation** (for large models)
   - Use existing `npu_attention_kernel_real.py`
   - Compile with `npu_mlir_kernel_compiler.py`
   - INT8 precision for efficiency

2. **MoE Router** (for MoE models)
   - Keep router weights at FP16
   - Use NPU for expert selection logic
   - Reduces iGPU memory bandwidth usage

3. **Embedding Lookup** (optional)
   - For very large vocabularies
   - NPU can handle sparse lookups

### iGPU Operations

1. **FFN Layers** - Always on iGPU
   - Use our INT4 shaders for memory efficiency
   - Better suited for dense computation

2. **Layer Normalization** - Always on iGPU
   - Requires higher precision
   - Small compute requirement

3. **Output Projection** - Always on iGPU
   - Final layer benefits from flexibility

## 📋 Implementation Checklist

### For Each Model, Create:

- [ ] **iGPU-only version** (baseline)
  - Simpler to debug
  - Establishes performance baseline
  - Uses our Vulkan shaders

- [ ] **NPU+iGPU hybrid** (if beneficial)
  - Modular design with runtime switching
  - NPU for appropriate operations
  - Benchmark to verify improvement

### Decision Matrix

| Model Size | Type | Recommended | NPU Operations |
|------------|------|-------------|----------------|
| <7B | Dense | iGPU-only | None |
| 7-15B | Dense | Optional NPU | Attention only |
| >15B | Dense | NPU+iGPU | Attention + Embeddings |
| Any size | MoE | NPU+iGPU | Router + Attention |

## 🚀 Practical Implementation

### Phase 1: Always Start with iGPU-Only
```python
# Initial implementation
model_pipeline = ModelPipeline(use_npu=False)
baseline_tps = benchmark(model_pipeline)
```

### Phase 2: Add NPU if Beneficial
```python
# Only if model fits criteria
if model_size > 7e9 or is_moe_model:
    model_pipeline_hybrid = ModelPipeline(use_npu=True)
    hybrid_tps = benchmark(model_pipeline_hybrid)
    
    if hybrid_tps > baseline_tps * 1.2:  # 20% improvement
        use_hybrid = True
```

## 📊 Expected Benefits

### MoE Models (like Qwen3-30B-A3B)
- **Routing overhead**: 5% → 2% with NPU
- **Memory bandwidth**: 30% reduction
- **Overall speedup**: 1.3-1.5x

### Large Dense Models (>15B)
- **Attention speedup**: 1.5-2x with NPU
- **Memory pressure**: Reduced on iGPU
- **Overall speedup**: 1.2-1.4x

### Small Models (<7B)
- **NPU overhead**: May reduce performance
- **Recommendation**: Stick with iGPU-only

## 🎯 Current Plan

1. **Phi-4-mini (3.8B)**: iGPU-only ✓
2. **Granite-3.3 (8B)**: iGPU-only (maybe test NPU for learning)
3. **Qwen3-30B-A3B MoE**: NPU+iGPU hybrid (router on NPU)

This approach ensures we:
- Always have a working iGPU version
- Only add NPU complexity when beneficial
- Can compare performance objectively
- Maintain clean, modular code