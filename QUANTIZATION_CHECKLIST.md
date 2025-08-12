# 🔧 Unicorn Engine Quantization & Customization Checklist

## 📋 Per-Model Quantization Tasks

### **Phase 1: Model Preparation**
- [ ] Download model in safetensors format (not GGUF/ONNX)
- [ ] Verify model architecture and layer names
- [ ] Identify model-specific requirements (MoE routing, special layers)
- [ ] Calculate theoretical memory requirements

### **Phase 2: Baseline Testing**
- [ ] Load model without quantization (FP16/BF16)
- [ ] Measure baseline TPS
- [ ] Record memory usage (VRAM, GTT, System RAM)
- [ ] Identify performance bottlenecks
- [ ] Save baseline quality metrics

### **Phase 3: Quantization Strategy**
- [ ] **Attention Layers**
  - [ ] Test INT8 symmetric quantization
  - [ ] Test INT4 with group size 128
  - [ ] Measure quality impact
  - [ ] Choose optimal configuration
  
- [ ] **FFN Layers**  
  - [ ] Test INT4 quantization (memory critical)
  - [ ] Test INT8 fallback if quality drops
  - [ ] Implement grouped quantization
  - [ ] Verify activation functions work
  
- [ ] **Embeddings & Special Layers**
  - [ ] Keep embeddings at INT8 minimum
  - [ ] Layer norms at FP16/INT8
  - [ ] Position encodings preservation
  
- [ ] **MoE-Specific (if applicable)**
  - [ ] Router weights at FP16 (critical)
  - [ ] Expert FFN at INT4
  - [ ] Load balancing preservation

### **Phase 4: Hardware Optimization**
- [ ] **Memory Distribution**
  - [ ] Map critical layers to VRAM
  - [ ] Bulk weights to GTT
  - [ ] Streaming strategy for large models
  
- [ ] **NPU Integration**
  - [ ] Identify NPU-suitable operations
  - [ ] Compile NPU kernels if beneficial
  - [ ] Test NPU vs GPU performance
  
- [ ] **Vulkan Shader Optimization**
  - [ ] Use pre-compiled INT4/INT8 shaders
  - [ ] Test RDNA3-specific optimizations
  - [ ] Measure shader performance

### **Phase 5: Performance Tuning**
- [ ] **Batch Size Optimization**
  - [ ] Test batch sizes: 1, 2, 4, 8, 16
  - [ ] Find memory vs speed sweet spot
  - [ ] Implement dynamic batching
  
- [ ] **Layer Fusion**
  - [ ] Fuse attention operations
  - [ ] Combine layer norm + linear
  - [ ] Reduce memory transfers
  
- [ ] **Cache Optimization**
  - [ ] Implement KV cache
  - [ ] Optimize cache size
  - [ ] Test cache eviction strategies

### **Phase 6: Quality Validation**
- [ ] **Perplexity Testing**
  - [ ] Measure on standard datasets
  - [ ] Compare to baseline
  - [ ] Set acceptable degradation threshold
  
- [ ] **Task-Specific Evaluation**
  - [ ] Run MMLU benchmark
  - [ ] Test on downstream tasks
  - [ ] Human evaluation sampling
  
- [ ] **Quantization Artifacts**
  - [ ] Check for repetition issues
  - [ ] Verify coherence maintained
  - [ ] Test edge cases

### **Phase 7: Production Readiness**
- [ ] Save quantized model weights
- [ ] Document optimal configuration
- [ ] Create loading script
- [ ] Implement health checks
- [ ] Add performance monitoring

---

## 🎯 Quantization Configurations to Test

### **Configuration A: Maximum Speed**
```python
config_max_speed = {
    'attention': 'INT4_group128',
    'ffn': 'INT4_group64',
    'embeddings': 'INT8',
    'layer_norm': 'INT8',
    'router': 'FP16'  # MoE only
}
```

### **Configuration B: Balanced**
```python
config_balanced = {
    'attention': 'INT8',
    'ffn': 'INT4_group128',
    'embeddings': 'INT8',
    'layer_norm': 'FP16',
    'router': 'FP16'
}
```

### **Configuration C: Quality Focused**
```python
config_quality = {
    'attention': 'INT8',
    'ffn': 'INT8',
    'embeddings': 'FP16',
    'layer_norm': 'FP16',
    'router': 'FP16'
}
```

---

## 📊 Metrics to Track

### **Performance Metrics**
- [ ] Tokens per second (TPS)
- [ ] Time to first token (TTFT)
- [ ] Batch processing throughput
- [ ] Memory bandwidth utilization
- [ ] GPU/NPU utilization percentage

### **Quality Metrics**
- [ ] Perplexity change from baseline
- [ ] BLEU/ROUGE scores (if applicable)
- [ ] Task-specific accuracy
- [ ] Human preference scores

### **Resource Metrics**
- [ ] Peak memory usage
- [ ] Average memory usage
- [ ] Memory allocation pattern
- [ ] Cache hit rates
- [ ] Power consumption

---

## 🔄 Optimization Iteration Cycle

1. **Measure Baseline** → Record current performance
2. **Apply Change** → Implement one optimization
3. **Test Impact** → Measure performance & quality
4. **Decision Point**:
   - ✅ Improvement? → Keep change, continue
   - ❌ Regression? → Revert, try alternative
5. **Document** → Record what worked/failed
6. **Repeat** → Next optimization

---

## 📝 Model-Specific Notes

### **Phi-4-mini (3.8B)**
- Dense model, no MoE considerations
- Perfect for iGPU-only testing
- Smallest model - ideal for rapid iteration
- Target: 50+ TPS with INT8, possibly 100+ TPS with INT4

### **Granite-3.3 (8B)**  
- Smallest model, should be fastest
- Target: 50+ TPS possible
- Good for extreme quantization tests

### **Qwen3-30B-A3B (MoE)**
- Only 3.3B active parameters
- Router precision critical
- Target: 40-50 TPS with INT4 experts

---

## ✅ Success Criteria

A model configuration is considered successful when:
1. **TPS Target Met**: Achieves minimum required TPS
2. **Quality Preserved**: Perplexity within 5% of baseline
3. **Memory Efficient**: Fits within available VRAM+GTT
4. **Stable**: No crashes or memory leaks
5. **Reproducible**: Consistent performance across runs

---

Last Updated: [Date]
Next Model to Test: [Model Name]