# ✅ OPTIMIZATION CHECKLIST - Path to 50 TPS

**Project**: Gemma 3 27B Performance Optimization on NPU+iGPU  
**Current Status**: Pure hardware pipeline needs fixing (loader compatibility issue)  
**Target**: 50 tokens per second  
**Date**: July 13, 2025

---

## 🔧 Phase 0: Fix Current Implementation Issues

### ⚠️ CRITICAL: Fix Pure Hardware Pipeline
- [ ] **Fix loader compatibility issue**
  - File: `pure_hardware_pipeline.py`
  - Problem: Using `PureMemoryMappedLoader` but calling `load_model()` which doesn't exist
  - Solution: Check `pure_mmap_loader.py` for correct method name (might be `load()` or similar)
  - Alternative: Ensure the loader interface matches what pipeline expects

- [ ] **Verify environment setup**
  - Run: `source /home/ucadmin/activate-pure-hardware-env.sh`
  - Confirm: No PyTorch dependencies loaded
  - Check: `python -c "import torch"` should fail in pure hardware env

- [ ] **Test basic server startup**
  - Command: `python pure_hardware_api_server.py`
  - Expected: Server starts on port 8006 without crashes
  - Verify: Model loads successfully (watch for "Model loaded" messages)

### 📊 Baseline Performance Testing
- [ ] **Measure current TPS**
  - Run: `python benchmark_pure_hardware_tps.py`
  - Document: Actual TPS achieved (expected 5-15 TPS)
  - Monitor: Hardware utilization during test

- [ ] **Check hardware utilization**
  - Tool: `radeontop` (GPU monitoring)
  - Tool: `xrt-smi examine` (NPU status)
  - Document: NPU % and iGPU % utilization

- [ ] **Verify memory distribution**
  - Check VRAM usage: Should show ~6.2GB
  - Check GTT usage: Should show ~17.9GB
  - Command: `radeontop -d - -l 1`

---

## 🚀 Phase 1: Immediate Optimizations (Target: 15-25 TPS)

### 1.1 Fix Q/K/V Projection Bottleneck (CRITICAL - 20x potential speedup)

- [ ] **Analyze current implementation**
  - File: `pure_hardware_pipeline.py`
  - Find: Look for Q, K, V projection code (likely in attention layer)
  - Document: Current time for each projection (should be ~7-8s each)

- [ ] **Create fused QKV implementation**
  - [ ] Create combined QKV weight matrix
  - [ ] Implement `process_qkv_fused()` method
  - [ ] Replace three separate projections with one fused operation
  - [ ] Test correctness with small examples

- [ ] **Integration steps**
  ```python
  # Before (3 operations, ~22-23s total):
  q = self.process_projection(hidden_states, q_weight)  # 7-8s
  k = self.process_projection(hidden_states, k_weight)  # 7-8s  
  v = self.process_projection(hidden_states, v_weight)  # 7-8s
  
  # After (1 operation, <1s):
  qkv = self.process_qkv_fused(hidden_states, qkv_weight)  # <1s
  q, k, v = qkv.split(3, dim=-1)
  ```

- [ ] **Test and benchmark**
  - Verify output correctness
  - Measure time reduction
  - Expected: 22-23s → <1s

### 1.2 Enable Persistent GPU Buffers

- [ ] **Implement buffer pool manager**
  - File: `real_vulkan_matrix_compute.py`
  - Add `_create_buffer_pool()` method
  - Pre-allocate 2GB of GPU buffers

- [ ] **Buffer pool design**
  - [ ] Small buffers: 32 x 1MB
  - [ ] Medium buffers: 16 x 16MB
  - [ ] Large buffers: 8 x 256MB
  - [ ] Total: ~2GB pre-allocated

- [ ] **Implement buffer reuse**
  - [ ] Track buffer usage
  - [ ] Implement get/release buffer methods
  - [ ] Add buffer lifecycle management
  - [ ] Prevent memory leaks

### 1.3 Implement Basic Batching

- [ ] **Modify token processing**
  - File: `pure_hardware_api_server.py`
  - Change single token to batch processing
  - Start with batch_size=8

- [ ] **Implement batch utilities**
  - [ ] Batch padding logic
  - [ ] Attention masking for varying lengths
  - [ ] Batch collation function

- [ ] **Update matrix operations**
  - [ ] Add batch dimension to all operations
  - [ ] Update Vulkan compute dispatch
  - [ ] Test with batch sizes: 1, 4, 8

---

## 🔍 Phase 1 Validation

- [ ] **Performance testing**
  - Run: `python test_qkv_fusion.py`
  - Run: `python test_buffer_pooling.py`
  - Run: `python test_batch_inference.py`
  - Run: `python benchmark_optimized_tps.py --batch-size 8`

- [ ] **Expected results**
  - TPS: 15-25 (3-5x improvement)
  - First token latency: <2s
  - Memory usage: Stable
  - No thermal throttling

---

## 📋 Phase 2: Core Optimizations (Target: 30-40 TPS)

### 2.1 Advanced Batching System

- [ ] **Create batch inference engine**
  - New file: `batch_inference_engine.py`
  - Implement dynamic batch scheduler
  - Support batch_size=32

- [x] **Implement KV cache**
  - New file: `kv_cache_manager.py`
  - Cache attention keys/values
  - Implement eviction policies
  - Memory-efficient storage

### 2.2 NPU Kernel Optimization

- [x] **Create optimized NPU kernels**
  - New file: `npu_attention_kernel_optimized.py`
  - Implement flash attention variant
  - Add INT8 attention path
  - Enable multi-head parallelism

### 2.3 Memory Pipeline Optimization

- [x] **Implement double buffering**
  - Process batch N while loading batch N+1
  - Use pinned memory for transfers
  - Enable async execution

- [x] **Zero-copy optimization**
  - Use HMA bridge for NPU↔iGPU
  - Eliminate unnecessary copies
  - Direct memory mapping

---

## 🚀 Phase 3: Advanced Optimizations (Target: 50+ TPS)

### 3.1 Dynamic Quantization

- [x] **Create quantization engine**
  - New file: `dynamic_quantization_engine.py`
  - On-the-fly INT8/INT4 quantization
  - Mixed precision logic

### 3.2 Speculative Decoding

- [x] **Implement draft model**
  - New file: `speculative_decoder.py`
  - Use smaller model for speculation
  - Verification with main model

### 3.3 Hardware-Specific Tuning

- [ ] **RDNA3 optimizations**
  - Profile shader performance
  - Optimize for wave32
  - Tune memory patterns
  - Use AMD intrinsics

---

## 📊 Testing & Monitoring Commands

### Performance Testing
```bash
# Test current performance
python benchmark_pure_hardware_tps.py

# Test with optimizations
python benchmark_optimized_tps.py --batch-size 8
python benchmark_optimized_tps.py --batch-size 32

# Test individual components
python test_qkv_fusion.py
python test_buffer_pooling.py
python test_batch_inference.py
```

### Hardware Monitoring
```bash
# GPU monitoring (run in separate terminal)
watch -n 0.5 radeontop

# NPU status
xrt-smi examine

# System resources
htop

# Thermal monitoring
watch -n 1 sensors
```

### API Testing
```bash
# Test inference
curl -X POST http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-27b-pure-hardware","messages":[{"role":"user","content":"Hello"}]}'

# Check health
curl http://localhost:8006/health
```

---

## 🎯 Success Criteria

### Phase 1 Complete
- ✅ Q/K/V fusion working (<1s vs 22-23s)
- ✅ Buffer pooling implemented
- ✅ Basic batching (8 tokens)
- ✅ 15-25 TPS achieved
- ✅ Stable operation

### Phase 2 Complete
- ✅ Advanced batching (32 tokens)
- ✅ KV cache operational
- ✅ NPU optimized kernels
- ✅ 30-40 TPS achieved
- ✅ <1s first token latency

### Phase 3 Complete
- ✅ Dynamic quantization
- ✅ Speculative decoding
- ✅ RDNA3 optimizations
- ✅ 50+ TPS achieved
- ✅ Production ready

---

## 🔧 Key Files Reference

### Core Files to Modify
- `pure_hardware_pipeline.py` - Main inference pipeline
- `real_vulkan_matrix_compute.py` - Vulkan GPU operations
- `npu_attention_kernel_real.py` - NPU operations
- `pure_hardware_api_server.py` - API server

### New Files to Create
- `batch_inference_engine.py` - Batching system
- `kv_cache_manager.py` - KV cache
- `npu_attention_kernel_optimized.py` - Optimized NPU
- `dynamic_quantization_engine.py` - Dynamic quant
- `speculative_decoder.py` - Speculative decoding

### Configuration Files
- `transformer_optimized.comp` - Optimized shaders
- `lightning_fast_loader.py` - Fast loading (if PyTorch-free version exists)
- `advanced_hardware_tuner.py` - Hardware optimization

---

## 📝 Notes for Implementation

1. **Start with Phase 0** - Fix the current loader issue first
2. **Q/K/V fusion is CRITICAL** - This alone could give 20x speedup
3. **Test after each change** - Ensure correctness before optimizing further
4. **Monitor hardware** - Keep radeontop and sensors running
5. **Document findings** - Update this checklist with actual results

---

## 🚨 Common Issues & Solutions

### Server Won't Start
- Check environment: Use pure hardware env, not regular AI env
- Check loader compatibility: Ensure method names match
- Check model path: Verify quantized model exists

### Low Performance
- Check Q/K/V timing: Should be <1s after fusion
- Check GPU utilization: Should be >80% with optimizations
- Check batching: Single token processing is inefficient

### Memory Issues
- Monitor with htop and radeontop
- Check for memory leaks in buffer pool
- Ensure proper cleanup in shutdown

---

*Remember: The #1 priority is fixing Q/K/V projections - this is where 95% of the time is being spent!*