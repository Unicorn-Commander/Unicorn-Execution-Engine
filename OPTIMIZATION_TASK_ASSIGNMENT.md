# 🚀 Optimization Task Assignment Checklist

**Date**: July 16, 2025  
**Goal**: Achieve 81+ TPS performance target  
**Current Status**: 0.04 TPS (persistent buffers implemented but setup overhead issue)

---

## 📋 Task Assignments

### 🔴 **CRITICAL - Fix Extreme Setup Overhead**
**Responsible**: Gemini-CLI  
**Priority**: CRITICAL - Blocking all other optimizations  
**Current Issue**: 860-880ms setup per operation (should be 50ms)  
**Expected Outcome**: Reduce setup to <50ms, unlock 17x speedup  

- [ ] Profile `compute_matrix_multiply` setup phase in `real_vulkan_matrix_compute.py`
- [ ] Identify why buffer allocation takes ~1 second
- [ ] Check if buffers are being recreated instead of reused from pool
- [ ] Verify pre-allocated buffer pool is actually being used
- [ ] Document findings and implement fix

**Instructions for Gemini-CLI**:
```
The Vulkan compute_matrix_multiply function in real_vulkan_matrix_compute.py has 860ms setup overhead instead of expected 50ms. This is visible in test_persistent_buffer_simple.py output. Please investigate why buffer allocation/setup takes so long. Look for repeated allocations, inefficient descriptor set creation, or buffer pool not being used properly.
```

---

### 🔴 **HIGH PRIORITY - Fix Model Loading Timeout**
**Responsible**: Gemini-CLI  
**Priority**: HIGH - Prevents full pipeline testing  
**Current Issue**: Model loading times out (should be 10-15s)  
**Expected Outcome**: Model loads in <15 seconds  

- [ ] Debug `LightningFastLoader` in `lightning_fast_loader.py`
- [ ] Verify parallel worker pool is actually running in parallel
- [ ] Check if memory mapping is efficient
- [ ] Profile file I/O operations
- [ ] Consider implementing progress logging to identify bottleneck

**Instructions for Gemini-CLI**:
```
The LightningFastLoader in lightning_fast_loader.py times out when loading the 26GB model from ./quantized_models/gemma-3-27b-it-layer-by-layer/. It should load in 10-15s but takes forever. The loader uses 16 parallel workers. Debug why parallel loading isn't working efficiently and add progress logging to identify bottlenecks.
```

---

### 🟢 **QUICK WIN - Integrate RDNA3 Optimized Shaders**
**Responsible**: Gemini-CLI  
**Priority**: MEDIUM - Easy 2.4x speedup  
**Current Issue**: Using generic shaders instead of RDNA3-optimized  
**Expected Outcome**: 2.4x performance improvement  

- [ ] Replace `matrix_multiply.spv` with `rdna3_optimized.spv`
- [ ] Update shader loading in `real_vulkan_matrix_compute.py`
- [ ] Enable Wave32 mode for AMD RDNA3
- [ ] Verify shader compatibility
- [ ] Benchmark performance improvement

**Instructions for Gemini-CLI**:
```
Replace the generic Vulkan shaders with RDNA3-optimized versions. Shader files already exist: rdna3_optimized.comp/.spv, rdna3_attention.comp/.spv, rdna3_int4.comp/.spv. Update real_vulkan_matrix_compute.py to load these optimized shaders instead of the generic matrix_multiply.spv. The RDNA3 shaders use Wave32 mode for better performance on AMD Radeon 780M.
```

---

### 🟡 **MEDIUM PRIORITY - Implement INT4 Quantization**
**Responsible**: Claude  
**Priority**: MEDIUM - 2x memory and compute efficiency  
**Current Issue**: Using INT8 (26GB), could use INT4 (13GB)  
**Expected Outcome**: Model size 26GB→13GB, 2x compute speedup  

- [ ] Integrate existing `rdna3_int4.comp/.spv` shaders
- [ ] Implement INT4 weight packing (2 weights per byte)
- [ ] Update pipeline to detect and use INT4 weights
- [ ] Modify buffer allocation for INT4 sizes
- [ ] Test native INT4 performance on RDNA3
- [ ] Verify accuracy with INT4 quantization

---

### 🟡 **MEDIUM PRIORITY - Layer Fusion Optimization**
**Responsible**: Claude  
**Priority**: MEDIUM - Major architectural optimization  
**Current Issue**: 7 operations per layer (Q/K/V/O + gate/up/down)  
**Expected Outcome**: Reduce to 3-4 operations per layer  

- [ ] Design fused transformer block architecture
- [ ] Create combined attention+FFN kernel
- [ ] Implement single-kernel Q/K/V projection
- [ ] Add residual connections in shader
- [ ] Eliminate intermediate memory transfers
- [ ] Benchmark fused vs separate operations

---

### 🟢 **LOW PRIORITY - NPU Integration**
**Responsible**: Claude  
**Priority**: LOW - Hardware currently blocked  
**Current Issue**: SMU errors preventing NPU execution  
**Expected Outcome**: 2.5x speedup for attention when fixed  

- [ ] Monitor hardware status for SMU fix
- [ ] Test with compiled kernels (`attention_256_int8.bin`)
- [ ] Implement XRT buffer management
- [ ] Create hybrid NPU+GPU execution path
- [ ] Benchmark NPU vs GPU attention performance

---

## 📊 Performance Impact Summary

| Task | Responsible | Expected Speedup | Complexity |
|------|-------------|------------------|------------|
| Fix Setup Overhead | Gemini-CLI | 17x | Medium |
| Fix Model Loading | Gemini-CLI | N/A (enabler) | Medium |
| RDNA3 Shaders | Gemini-CLI | 2.4x | Easy |
| INT4 Quantization | Claude | 2x | High |
| Layer Fusion | Claude | 2x | Very High |
| NPU Integration | Claude | 2.5x | High |

**Combined Potential**: 17 × 2.4 × 2 × 2 × 2.5 = **408x improvement**  
**Current**: 0.04 TPS → **Target**: 81+ TPS ✅

---

## 🎯 Success Metrics

- [ ] Model loads in <15 seconds
- [ ] Setup overhead <50ms per operation
- [ ] GPU utilization >80% during inference
- [ ] Achieve 81+ TPS on benchmark
- [ ] No CPU fallbacks during inference

---

## 📝 Notes

- Tasks assigned to Gemini-CLI are well-defined with clear problems
- Tasks assigned to Claude require deep architectural understanding
- Start with setup overhead fix - it's blocking everything else
- RDNA3 shader integration is a quick win while investigating overhead