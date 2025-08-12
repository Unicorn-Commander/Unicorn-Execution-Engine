# 🚀 Phi-4-mini Implementation Plan for Unicorn Engine

## 📋 Implementation Checklist

### Phase 1: Model Download & Setup ⏱️ (30 mins)
- [ ] Download Phi-4-mini-instruct (3.8B) in safetensors format
- [ ] Verify download integrity and file structure
- [ ] Create model-specific configuration file
- [ ] Set up logging and monitoring

### Phase 2: Baseline Testing ⏱️ (1 hour)
- [ ] Load model with existing Gemma loader (adapt for Phi-4)
- [ ] Run inference without quantization (FP16/BF16)
- [ ] Measure baseline TPS on iGPU
- [ ] Record baseline memory usage (VRAM + GTT)
- [ ] Save baseline perplexity score

### Phase 3: Custom Quantization Implementation ⏱️ (3 hours)
- [ ] Implement INT8 quantization with our shaders
  - [ ] Quantize weights using `dynamic_quantization_engine.py`
  - [ ] Load INT8 weights into Vulkan buffers
  - [ ] Use `matrix_multiply_int8.spv` for computation
  - [ ] Verify shader execution on iGPU
- [ ] Implement INT4 quantization for FFN
  - [ ] Apply INT4 to memory-intensive FFN layers
  - [ ] Use `rdna3_int4.spv` for INT4 operations
  - [ ] Keep attention at INT8 for quality
  - [ ] Implement proper dequantization in shaders
- [ ] Implement mixed precision pipeline
  - [ ] Create layer-wise quantization config
  - [ ] Use `transformer_optimized.spv` for fused ops
  - [ ] Optimize memory transfers between precisions

### Phase 4: iGPU Optimization ⏱️ (2 hours)
- [ ] Optimize memory distribution
  - [ ] Map critical layers to VRAM (16GB available)
  - [ ] Use GTT for less frequently accessed weights
- [ ] Test with Vulkan compute workaround
  - [ ] Verify Vulkan shaders work with INT4/INT8
  - [ ] Use existing RDNA3 optimized shaders
- [ ] Implement batching
  - [ ] Test batch sizes: 1, 2, 4, 8
  - [ ] Find optimal batch size for memory/speed

### Phase 5: Performance Tuning ⏱️ (1 hour)
- [ ] Profile inference bottlenecks
- [ ] Optimize layer fusion opportunities
- [ ] Implement KV cache for multi-turn
- [ ] Fine-tune workgroup sizes for RDNA3

### Phase 6: Benchmarking & Validation ⏱️ (1 hour)
- [ ] Run comprehensive benchmark suite
  - [ ] Short, medium, long generation tests
  - [ ] Measure TPS at different sequence lengths
- [ ] Validate quality metrics
  - [ ] Perplexity on validation set
  - [ ] Sample outputs for coherence
- [ ] Document optimal configuration
- [ ] Update MODEL_PERFORMANCE_TRACKER.md

## 🎯 Success Criteria
- **Minimum**: 50 TPS with INT8 quantization
- **Target**: 80-100 TPS with INT4 quantization
- **Stretch**: 120+ TPS with optimized configuration
- **Quality**: Perplexity within 5% of baseline

## 📁 Key Files & Paths

### Model Location (after download):
```
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/models/phi-4-mini-instruct/
```

### Existing Infrastructure to Use:
```python
# Quantization Engine
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/dynamic_quantization_engine.py

# Vulkan Compute (with fallback)
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/vulkan_compute_workaround.py

# Model Loader (adapt from Gemma)
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/gemma_27b_loader_v2.py

# Performance Pipeline
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/gemma_27b_working_pipeline.py
```

### Create New Files:
```python
# Phi-4 specific loader
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/phi4_mini_loader.py

# Phi-4 optimized pipeline  
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/phi4_mini_pipeline.py

# Benchmark results
/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/phi4_mini_benchmark_results.json
```

## 🔧 Technical Notes

### Model Architecture:
- **Parameters**: 3.8B
- **Architecture**: Dense transformer (no MoE)
- **Hidden Size**: TBD (check config.json after download)
- **Layers**: TBD
- **Attention Heads**: TBD

### Quantization Strategy:
```python
# Conservative (start here)
config_conservative = {
    'attention': 'INT8',
    'ffn': 'INT8', 
    'embeddings': 'INT8',
    'layer_norm': 'FP16'
}

# Aggressive (target)
config_aggressive = {
    'attention': 'INT8',
    'ffn': 'INT4_group128',
    'embeddings': 'INT8',
    'layer_norm': 'INT8'
}

# Extreme (test limits)
config_extreme = {
    'attention': 'INT4_group128',
    'ffn': 'INT4_group64',
    'embeddings': 'INT8',
    'layer_norm': 'INT8'
}
```

## 📊 Expected Results

With 3.8B parameters on our hardware:
- **FP16 Baseline**: 20-30 TPS
- **INT8 Quantized**: 50-70 TPS
- **INT4 Quantized**: 80-120 TPS
- **Memory Usage**: 2-4GB active

## 🚨 Common Issues & Solutions

1. **BFloat16 tensors**: Use existing `bfloat16_converter.py`
2. **Layer name mismatches**: Check model architecture first
3. **Memory allocation**: Start with smaller batch sizes
4. **Vulkan errors**: Falls back to optimized NumPy automatically

## ✅ Definition of Done

- [ ] Model loads and runs on iGPU
- [ ] Achieves >50 TPS with acceptable quality
- [ ] All results documented in tracker
- [ ] Optimal configuration saved
- [ ] Ready for production use