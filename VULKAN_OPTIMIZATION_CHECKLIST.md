# VULKAN OPTIMIZATION CHECKLIST - REAL GPU USAGE

## 🎯 GOAL: Fix Vulkan to achieve REAL GPU compute with 16GB VRAM + 10GB GTT usage

---

## ✅ Phase 1: Fix Memory Mapping Error

### 1.1 Fix VkErrorMemoryMapFailed
- [ ] Check memory allocation flags in `_create_buffer_empty()`
- [ ] Ensure `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` is set
- [ ] Ensure `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` is set
- [ ] Verify memory type index selection in `_find_memory_type()`
- [ ] Check if memory is already mapped before mapping
- [ ] Add proper error handling for memory allocation failures

### 1.2 Fix Buffer Pool Implementation
- [ ] Verify buffer sizes match allocation sizes
- [ ] Ensure buffers are properly unmapped before reuse
- [ ] Check memory alignment requirements (typically 256 bytes)
- [ ] Implement proper buffer cleanup in `_cleanup_buffers()`
- [ ] Add memory usage tracking/logging

### 1.3 Verify GPU Memory Properties
- [ ] Query physical device memory properties
- [ ] Log available memory heaps and types
- [ ] Ensure we're using DEVICE_LOCAL memory for GPU data
- [ ] Verify HOST_VISIBLE memory for staging buffers

---

## ✅ Phase 2: Direct VRAM/GTT Loading

### 2.1 Implement GPU Memory Allocation Strategy
- [ ] Allocate 16GB in VRAM for critical tensors
- [ ] Allocate 10GB in GTT for bulk weights
- [ ] Create memory allocator that tracks VRAM vs GTT
- [ ] Implement memory pressure handling

### 2.2 Bypass CPU RAM for Model Loading
- [ ] Create direct GPU memory mapping
- [ ] Use staging buffers for host->device transfers
- [ ] Implement async memory transfers
- [ ] Add progress tracking for large transfers

### 2.3 Memory Layout Optimization
- [ ] Align tensors to GPU cache lines (64-256 bytes)
- [ ] Group related tensors for locality
- [ ] Use optimal memory access patterns
- [ ] Minimize memory fragmentation

---

## ✅ Phase 3: Shader Optimization

### 3.1 Verify Compute Shaders
- [ ] Check SPIR-V shader compilation
- [ ] Verify shader bindings match descriptor sets
- [ ] Ensure work group sizes are optimal (64-256)
- [ ] Check shared memory usage

### 3.2 Optimize Matrix Multiplication
- [ ] Use tiled matrix multiplication
- [ ] Optimize for AMD RDNA3 architecture
- [ ] Use wave-level primitives where possible
- [ ] Implement FP16 support for 2x throughput

### 3.3 Implement Q/K/V Fusion
- [ ] Create fused attention kernel
- [ ] Combine Q, K, V projections in single pass
- [ ] Use shared memory for intermediate results
- [ ] Minimize global memory accesses

---

## ✅ Phase 4: Command Buffer Optimization

### 4.1 Batch GPU Operations
- [ ] Create command buffers for entire inference pass
- [ ] Minimize CPU-GPU synchronization points
- [ ] Use multiple command buffers for pipelining
- [ ] Implement proper fence/semaphore usage

### 4.2 Async Execution
- [ ] Use multiple queues (compute + transfer)
- [ ] Overlap compute with memory transfers
- [ ] Implement double buffering
- [ ] Add GPU timeline profiling

---

## ✅ Phase 5: NPU Integration

### 5.1 NPU Memory Management
- [ ] Allocate NPU SRAM (2GB) for attention
- [ ] Implement NPU-GPU memory synchronization
- [ ] Use zero-copy where possible
- [ ] Track NPU memory usage

### 5.2 NPU Kernel Optimization
- [ ] Verify NPU kernel compilation
- [ ] Optimize for 16 TOPS throughput
- [ ] Implement NPU-specific attention
- [ ] Add NPU performance counters

---

## ✅ Phase 6: Full System Integration

### 6.1 Multi-threaded Batch Processing
- [ ] Use all 16 CPU threads for preprocessing
- [ ] Implement work queue for GPU batches
- [ ] Optimize CPU-GPU data transfer
- [ ] Add thread pool for parallel operations

### 6.2 Memory Monitoring
- [ ] Add VRAM usage tracking
- [ ] Add GTT usage tracking
- [ ] Monitor memory bandwidth
- [ ] Log memory allocation patterns

### 6.3 Performance Validation
- [ ] Verify GPU usage >50% in radeontop
- [ ] Confirm VRAM usage ~16GB
- [ ] Confirm GTT usage ~10GB
- [ ] Measure actual TPS with real model

---

## 🔧 Testing Commands

```bash
# Monitor GPU usage
watch -n 0.1 'radeontop -d -'

# Monitor memory
watch -n 1 'cat /sys/kernel/debug/dri/0/amdgpu_vram_mm'
watch -n 1 'cat /sys/kernel/debug/dri/0/amdgpu_gtt_mm'

# Profile Vulkan
VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
python test_gpu_compute.py
```

---

## 📊 Success Metrics

| Metric | Current | Target |
|--------|---------|---------|
| GPU Usage | 0% | >50% |
| VRAM Usage | 1GB | 16GB |
| GTT Usage | 0.1GB | 10GB |
| Memory Transfers | Through CPU | Direct to GPU |
| Shader Execution | Not running | Confirmed running |
| NPU Usage | Unknown | Active |
| Real Model TPS | 0 | 50+ |

---

## 🚨 Common Pitfalls to Avoid

1. **Don't use system RAM** for model weights
2. **Don't use CPU compute** fallbacks
3. **Don't simulate performance** - real only
4. **Don't ignore memory alignment** - causes crashes
5. **Don't skip validation layers** during debugging

---

*This checklist ensures REAL GPU usage with proper VRAM/GTT allocation*