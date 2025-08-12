# 🎯 OPTIMIZATION PROJECT PLAN - Path to 50 TPS

**Project**: Gemma 3 27B Performance Optimization  
**Target**: 50 tokens per second  
**Current Status**: 5-15 TPS expected (optimizations implemented, ready to test)  
**Timeline**: 2-3 weeks total  
**Created**: July 13, 2025

---

## 📊 Executive Summary

### Current State
- **Baseline Performance**: 0.3-0.5 TPS (before optimizations)
- **Expected with Today's Optimizations**: 5-15 TPS (10-30x improvement)
- **Critical Bottleneck**: Q/K/V projections taking 22-23 seconds
- **Hardware Utilization**: NPU ~7%, iGPU ~30-40% (severely underutilized)

### Target State
- **Performance Goal**: 50 TPS sustained
- **Hardware Utilization**: NPU >70%, iGPU >80%
- **First Token Latency**: <500ms
- **Memory Efficiency**: Optimized with persistent buffers and pooling

---

## 🚀 Phase 0: Test Current Optimizations (Immediate)

**Timeline**: Today  
**Target**: Validate 5-15 TPS improvement

### Checklist:
- [ ] Activate environment: `source /home/ucadmin/activate-uc1-ai-py311.sh`
- [ ] Start optimized server: `python pure_hardware_api_server.py`
- [ ] Measure loading time (expected: 10-15s vs 120s baseline)
- [ ] Run performance benchmark
- [ ] Document actual TPS achieved
- [ ] Monitor hardware utilization (radeontop, htop)
- [ ] Check memory distribution (VRAM vs GTT)
- [ ] Verify no thermal throttling

### Success Criteria:
- ✅ Loading time under 20 seconds
- ✅ 5-15 TPS achieved
- ✅ 8.9 TFLOPS iGPU utilization confirmed
- ✅ Stable operation without crashes

---

## 🔧 Phase 1: Immediate Wins (15-25 TPS)

**Timeline**: 1-2 days  
**Expected Gain**: 3-5x from current

### 1.1 Fix Q/K/V Projection Bottleneck ⚡ CRITICAL

**Problem**: Sequential Q, K, V projections taking 22-23 seconds total  
**Solution**: Fuse into single operation

#### Implementation Tasks:
- [ ] Analyze current projection code in `pure_hardware_pipeline.py`
- [ ] Create fused QKV weight matrix combining all three projections
- [ ] Implement `process_qkv_fused()` method
- [ ] Modify attention layer to use fused output
- [ ] Test correctness with small examples
- [ ] Benchmark performance improvement
- [ ] Update all transformer layers

#### Code Changes:
```python
# File: pure_hardware_pipeline.py
# Location: AttentionLayer.forward() method
# Current: 3 separate projections (7-8s each)
# Target: 1 fused projection (<1s total)
```

### 1.2 Enable Persistent GPU Buffers

**Problem**: Repeated GPU buffer allocation/deallocation  
**Solution**: Pre-allocate buffer pool

#### Implementation Tasks:
- [ ] Create buffer pool manager in `real_vulkan_matrix_compute.py`
- [ ] Pre-allocate 2GB of GPU buffers in various sizes
- [ ] Implement buffer reuse logic
- [ ] Add buffer lifecycle management
- [ ] Test memory leak prevention
- [ ] Benchmark allocation overhead reduction

#### Buffer Pool Design:
- Small buffers: 32 x 1MB (frequent operations)
- Medium buffers: 16 x 16MB (layer operations)
- Large buffers: 8 x 256MB (model weights)

### 1.3 Implement Basic Batching

**Problem**: Single token processing inefficient for GPU  
**Solution**: Process 8 tokens in parallel

#### Implementation Tasks:
- [ ] Modify token processing loop in `pure_hardware_api_server.py`
- [ ] Implement batch padding/masking
- [ ] Update matrix operations for batch dimension
- [ ] Test with various batch sizes (1, 4, 8)
- [ ] Handle edge cases (sequence end, varying lengths)
- [ ] Measure throughput improvement

---

## 🚀 Phase 2: Core Optimizations (30-40 TPS)

**Timeline**: 3-5 days  
**Expected Gain**: 2-3x from Phase 1

### 2.1 Advanced Batching System

**Target**: Process 32 tokens in parallel with KV caching

#### Implementation Tasks:
- [ ] Create `batch_inference_engine.py`
- [ ] Implement dynamic batch scheduler
- [ ] Add KV cache for attention optimization
- [ ] Create `kv_cache_manager.py`
- [ ] Implement cache eviction policies
- [ ] Add batch pipelining
- [ ] Test with production workloads

#### Key Components:
```python
# New file: batch_inference_engine.py
class BatchInferenceEngine:
    - Dynamic batch formation
    - Sequence padding/masking
    - KV cache integration
    - Memory-efficient processing
```

### 2.2 NPU Kernel Optimization

**Target**: Optimized NPU kernels for batch attention

#### Implementation Tasks:
- [ ] Create `npu_attention_kernel_optimized.py`
- [ ] Implement flash attention variant
- [ ] Add INT8 quantized attention path
- [ ] Enable multi-head parallelism
- [ ] Optimize for 16 TOPS Phoenix NPU
- [ ] Integrate with batch processing
- [ ] Benchmark vs baseline

### 2.3 Memory Pipeline Optimization

**Target**: Zero-copy transfers and double buffering

#### Implementation Tasks:
- [ ] Implement double buffering system
- [ ] Add pinned memory for CPU↔GPU
- [ ] Enable HMA zero-copy for NPU↔iGPU
- [ ] Create async execution pipeline
- [ ] Add memory usage monitoring
- [ ] Test with concurrent operations

---

## 🚀 Phase 3: Advanced Optimizations (50+ TPS)

**Timeline**: 1-2 weeks  
**Expected Gain**: 1.5-2x from Phase 2

### 3.1 Dynamic Quantization

**Target**: On-the-fly INT8/INT4 quantization

#### Implementation Tasks:
- [ ] Create `dynamic_quantization_engine.py`
- [ ] Implement activation quantization
- [ ] Add mixed precision logic
- [ ] Create quantization calibration
- [ ] Test quality vs performance tradeoff
- [ ] Integrate with inference pipeline

### 3.2 Speculative Decoding

**Target**: 2-4x speedup with draft model

#### Implementation Tasks:
- [ ] Create `speculative_decoder.py`
- [ ] Implement draft model (smaller Gemma variant)
- [ ] Add verification logic
- [ ] Implement rollback mechanism
- [ ] Tune acceptance thresholds
- [ ] Benchmark speedup vs quality

### 3.3 Hardware-Specific RDNA3 Tuning

**Target**: Maximize AMD Radeon 780M performance

#### Implementation Tasks:
- [ ] Profile current shader performance
- [ ] Optimize for wave32 execution
- [ ] Tune memory access patterns
- [ ] Add AMD-specific intrinsics
- [ ] Optimize cache utilization
- [ ] Final performance validation

---

## 📋 Testing & Validation Plan

### Performance Testing
```bash
# Progressive benchmarking
python test_qkv_fusion.py              # Phase 1.1
python test_buffer_pooling.py          # Phase 1.2
python test_batch_inference.py         # Phase 1.3
python benchmark_optimized_tps.py      # Overall
```

### Quality Validation
- [ ] Perplexity comparison vs baseline
- [ ] Generation quality spot checks
- [ ] Long context handling
- [ ] Memory stability tests
- [ ] Thermal throttling tests

### Integration Testing
- [ ] OpenWebUI compatibility
- [ ] API response times
- [ ] Concurrent request handling
- [ ] Error recovery
- [ ] Resource cleanup

---

## 🎯 Success Metrics & Milestones

### Phase 1 Complete (Days 1-2)
- ✅ Q/K/V fusion implemented
- ✅ 15-25 TPS achieved
- ✅ <2s first token latency
- ✅ Stable memory usage

### Phase 2 Complete (Days 3-7)
- ✅ Batch size 32 working
- ✅ 30-40 TPS achieved
- ✅ KV cache operational
- ✅ NPU utilization >50%

### Phase 3 Complete (Days 8-14)
- ✅ 50+ TPS achieved
- ✅ <500ms first token latency
- ✅ NPU utilization >70%
- ✅ iGPU utilization >80%

### Stretch Goals
- 🎯 100 TPS with INT4
- 🎯 200 TPS with speculative decoding
- 🎯 <100ms first token latency

---

## 🛠️ Development Workflow

### Daily Tasks
1. Morning: Review overnight test results
2. Implementation: Focus on one optimization at a time
3. Testing: Validate each change immediately
4. Benchmarking: Measure improvement
5. Documentation: Update progress in CLAUDE.md

### Code Review Checklist
- [ ] Performance improvement measured
- [ ] No quality regression
- [ ] Memory usage stable
- [ ] Error handling robust
- [ ] Code documented

### Rollback Plan
- Git branches for each phase
- Performance regression tests
- Quick revert capability
- Baseline comparison always available

---

## 📊 Risk Management

### Technical Risks
1. **Q/K/V Fusion Complexity**: May require significant refactoring
   - Mitigation: Incremental implementation with fallback
   
2. **Memory Fragmentation**: Buffer pooling could fragment memory
   - Mitigation: Careful pool design and monitoring
   
3. **Quality Degradation**: Aggressive optimization might hurt quality
   - Mitigation: Continuous perplexity monitoring

### Schedule Risks
1. **MLIR-AIE2 Build Issues**: NPU kernel compilation blocked
   - Mitigation: Continue with iGPU optimizations first
   
2. **Unexpected Bottlenecks**: New bottlenecks may emerge
   - Mitigation: Continuous profiling and adaptive planning

---

## 🚦 Go/No-Go Decision Points

### After Phase 0 (Today)
- If <5 TPS achieved: Debug current optimizations first
- If 5-15 TPS achieved: Proceed to Phase 1

### After Phase 1 (Day 2)
- If <15 TPS achieved: Focus on Q/K/V bottleneck
- If 15-25 TPS achieved: Proceed to Phase 2

### After Phase 2 (Day 7)
- If <30 TPS achieved: Revisit batching strategy
- If 30-40 TPS achieved: Proceed to Phase 3

### Final Decision (Day 14)
- If <50 TPS: Identify remaining bottlenecks
- If 50+ TPS: Success! Deploy to production

---

## 📝 Notes & Resources

### Key Files to Monitor
- `pure_hardware_pipeline.py` - Core inference logic
- `real_vulkan_matrix_compute.py` - iGPU operations
- `npu_attention_kernel_real.py` - NPU operations
- `pure_hardware_api_server.py` - API server

### Performance Monitoring Commands
```bash
# GPU utilization
radeontop

# NPU status
xrt-smi examine

# System resources
htop

# Thermal monitoring
watch -n 1 sensors
```

### Documentation to Update
- CLAUDE.md - Progress updates
- OPTIMIZATION_ROADMAP_50TPS.md - Detailed findings
- API documentation - New features

---

*This project plan is a living document. Update regularly with progress and findings.*