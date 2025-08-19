# UC1-NPU-Production: Detailed Development Checklist

## 🎯 **Mission**: Get Qwen3-Embedding-0.6B running on NPU at 30-50 embeddings/sec

**Current CPU Baseline**: 9 embeddings/sec (110ms each)  
**Target Performance**: 30-50 embeddings/sec (3-5x speedup)  
**Timeline**: 6-8 weeks focused development

---

## 📋 **PHASE 1: FOUNDATION (Week 1-2)**

### **🔴 CRITICAL PATH - Week 1**

#### **1.1 Download Real Qwen3-Embedding-0.6B Model** ⚔️
- [ ] **Task**: Download official Qwen3-Embedding-0.6B from Hugging Face
- [ ] **Commands**:
  ```bash
  cd models/
  huggingface-cli download Alibaba-NLP/Qwen3-Embedding-0.6B --cache-dir ./qwen3-0.6b-original/
  ```
- [ ] **Verify**: Model files total ~1.2GB (600M parameters × 2 bytes for FP16)
- [ ] **Expected**: `config.json`, `pytorch_model.bin`, `tokenizer.json`
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🔴 HIGHEST

#### **1.2 Copy Working NPU Components** ⚔️
- [ ] **Task**: Copy proven components from UC1-Embedding-NPU project
- [ ] **Files to Copy**:
  ```bash
  # From UC1-Embedding-NPU/
  cp -r quantization/ ../UC1-NPU-Production/models/quantization/
  cp -r npu_kernels/ ../UC1-NPU-Production/kernels/
  cp npu_kernels/test_npu_basic.py ../UC1-NPU-Production/src/
  cp quantization/uc1_emb_format.py ../UC1-NPU-Production/models/quantization/
  ```
- [ ] **Components**:
  - [ ] NPU hardware detection (`test_npu_basic.py`)
  - [ ] XRT integration utilities
  - [ ] UC1-EMB quantization format
  - [ ] Basic MLIR kernel framework
  - [ ] Memory management utilities
- [ ] **Verify**: All imports work, NPU detection successful
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🔴 HIGHEST

#### **1.3 Test NPU Hardware Detection** ⚔️
- [ ] **Task**: Verify NPU is detected and accessible
- [ ] **Test Script**:
  ```bash
  cd src/
  python test_npu_basic.py
  ```
- [ ] **Expected Output**:
  ```
  ✅ NPU devices found: 1
  ✅ NPU Phoenix device accessible
  ✅ XRT runtime version: 2.20.x
  ✅ NPU ready for kernel execution
  ```
- [ ] **Timeline**: 0.5 days
- [ ] **Priority**: 🔴 HIGHEST

### **🟡 FOUNDATION SETUP - Week 1-2**

#### **1.4 Create Project Structure** 📁
- [ ] **Task**: Set up complete project directory structure
- [ ] **Directories to Create**:
  ```bash
  mkdir -p {src,kernels,models,benchmarks,tests,docs}
  mkdir -p kernels/{mlir,xclbin,compiled}
  mkdir -p models/{qwen3-0.6b-original,qwen3-0.6b-uc1,quantization}
  mkdir -p benchmarks/{cpu,npu,comparisons}
  mkdir -p tests/{unit,integration,performance}
  mkdir -p docs/{development,optimization,deployment}
  ```
- [ ] **Timeline**: 0.5 days
- [ ] **Priority**: 🟡 MEDIUM

#### **1.5 Setup Development Environment** 🛠️
- [ ] **Task**: Configure development environment for NPU work
- [ ] **Requirements**:
  ```bash
  # Install MLIR-AIE development tools
  cd /opt/xilinx/mlir-aie/
  source setup.sh
  
  # Verify MLIR compiler working
  aie-opt --help
  aie-translate --help
  ```
- [ ] **Python Dependencies**:
  ```bash
  pip install torch transformers huggingface-hub pyxrt
  pip install safetensors accelerate
  ```
- [ ] **Verify**: All tools accessible and working
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🟡 MEDIUM

#### **1.6 Create Basic Model Loader** 📥
- [ ] **Task**: Create Qwen3 model loading utilities
- [ ] **File**: `src/model_loader.py`
- [ ] **Features**:
  - [ ] Load Qwen3-Embedding-0.6B config
  - [ ] Extract model weights and architecture
  - [ ] Tokenizer integration
  - [ ] Basic embedding generation (CPU baseline)
- [ ] **Test**: Load model and generate sample embedding
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🟡 MEDIUM

#### **1.7 Establish CPU Baseline** 📊
- [ ] **Task**: Measure exact CPU performance
- [ ] **File**: `benchmarks/cpu_baseline.py`
- [ ] **Measurements**:
  - [ ] Single embedding latency
  - [ ] Batch embedding throughput
  - [ ] Memory usage
  - [ ] CPU utilization
- [ ] **Target Metrics**:
  - Single: ~110ms (9 emb/sec)
  - Batch 32: ~125ms per embedding
  - Memory: <2GB
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🟡 MEDIUM

---

## 📋 **PHASE 2: CORE OPTIMIZATION (Week 3-4)**

### **🔴 CRITICAL PATH - NPU Compilation**

#### **2.1 Compile First MLIR Kernel to XCLBIN** ⚔️
- [ ] **Task**: Compile basic embedding lookup kernel
- [ ] **Input**: `kernels/embedding_lookup.mlir`
- [ ] **Process**:
  ```bash
  # Compile MLIR to AIE assembly
  aie-opt kernels/embedding_lookup.mlir -o kernels/compiled/embedding.aie
  
  # Translate to XCLBIN
  aie-translate kernels/compiled/embedding.aie -o kernels/xclbin/embedding_lookup.xclbin
  ```
- [ ] **Verify**: XCLBIN file created successfully
- [ ] **Timeline**: 2-3 days
- [ ] **Priority**: 🔴 HIGHEST

#### **2.2 Test Basic NPU Execution** ⚔️
- [ ] **Task**: Load XCLBIN and execute simple operation
- [ ] **File**: `src/npu_engine.py`
- [ ] **Test**:
  ```python
  # Load kernel
  device = pyxrt.device(0)
  xclbin = device.load_xclbin("kernels/xclbin/embedding_lookup.xclbin")
  kernel = pyxrt.kernel(device, xclbin, "embedding_lookup")
  
  # Simple test with dummy data
  result = kernel.run(test_input)
  ```
- [ ] **Expected**: Successful kernel execution
- [ ] **Timeline**: 1-2 days
- [ ] **Priority**: 🔴 HIGHEST

### **🟡 OPTIMIZATION IMPLEMENTATION - Week 3-4**

#### **2.3 Implement UC1-EMB Quantization** 📦
- [ ] **Task**: Quantize Qwen3-0.6B to UC1-EMB format
- [ ] **File**: `models/quantization/quantize_qwen3.py`
- [ ] **Process**:
  - [ ] Convert FP16 weights to UC1-Q4N format
  - [ ] Optimize for NPU memory layout
  - [ ] Create quantization scales/offsets
  - [ ] Save in NPU-friendly format
- [ ] **Target**: 1.2GB → 300MB (4x compression)
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🟡 MEDIUM

#### **2.4 Create Persistent Kernel Execution** ⚡
- [ ] **Task**: Eliminate kernel launch overhead
- [ ] **File**: `src/persistent_npu_kernel.py`
- [ ] **Features**:
  - [ ] Keep kernels loaded in NPU memory
  - [ ] Reuse buffers between requests
  - [ ] Async kernel execution
  - [ ] Request queuing system
- [ ] **Expected Speedup**: 10-20ms saved per operation
- [ ] **Timeline**: 2-3 days
- [ ] **Priority**: 🟡 MEDIUM

#### **2.5 Build Zero-Copy DMA Framework** 🚀
- [ ] **Task**: Optimize CPU-NPU data transfer
- [ ] **File**: `src/zero_copy_dma.py`
- [ ] **Optimizations**:
  - [ ] Direct DMA transfer (bypass CPU copy)
  - [ ] Persistent buffer allocation
  - [ ] Memory-mapped file access
  - [ ] Async transfer with computation
- [ ] **Expected Speedup**: 5-15ms saved per operation
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🟡 MEDIUM

#### **2.6 End-to-End Embedding Generation Test** 🧪
- [ ] **Task**: Complete NPU embedding pipeline
- [ ] **File**: `tests/test_npu_embeddings.py`
- [ ] **Pipeline**:
  1. Tokenize input text
  2. Transfer tokens to NPU
  3. Execute embedding lookup kernel
  4. Transfer results back
  5. Normalize embeddings
- [ ] **Success Criteria**: Generated embeddings match CPU output (>99% similarity)
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🔴 HIGH

#### **2.7 Performance Measurement Phase 2** 📊
- [ ] **Task**: Measure Phase 2 performance improvements
- [ ] **File**: `benchmarks/npu_comparison.py`
- [ ] **Target Performance**:
  - Single: 50-70ms (15-20 emb/sec) - **2x CPU speedup**
  - Batch 32: 60-80ms per embedding
- [ ] **Verify**: Performance improvement achieved
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🔴 HIGH

---

## 📋 **PHASE 3: ADVANCED OPTIMIZATION (Week 5-6)**

### **🔴 CRITICAL PATH - Advanced Kernels**

#### **3.1 Implement Fused Transformer Pipeline** ⚔️
- [ ] **Task**: Create 12-layer fused transformer kernel
- [ ] **File**: `kernels/transformer_fused.mlir`
- [ ] **Features**:
  - [ ] Process all 12 Qwen3 layers in single kernel
  - [ ] Fused attention + FFN operations
  - [ ] Optimized memory access patterns
  - [ ] Multi-head attention optimization
- [ ] **Expected**: Major latency reduction (50-80ms saved)
- [ ] **Timeline**: 4-5 days
- [ ] **Priority**: 🔴 HIGHEST

#### **3.2 Implement Direct INT4 Compute** 📦
- [ ] **Task**: NPU kernels compute on INT4 directly
- [ ] **File**: `kernels/int4_compute.mlir`
- [ ] **Benefits**:
  - [ ] 4x memory throughput improvement
  - [ ] 2x compute speed (INT4 vs FP16)
  - [ ] Reduced memory bandwidth
- [ ] **Implementation**:
  - [ ] INT4 weight unpacking in NPU
  - [ ] Vectorized INT4 operations
  - [ ] On-the-fly dequantization
- [ ] **Timeline**: 3-4 days
- [ ] **Priority**: 🔴 HIGHEST

### **🟡 PERFORMANCE OPTIMIZATION - Week 5-6**

#### **3.3 Add Batch Processing Optimization** 📦
- [ ] **Task**: Optimize for batch embedding generation
- [ ] **File**: `src/batch_processor.py`
- [ ] **Features**:
  - [ ] Batch size 32-64 support
  - [ ] Parallel token processing
  - [ ] Memory layout optimization
  - [ ] Load balancing across NPU tiles
- [ ] **Expected**: 2-3x throughput improvement for batches
- [ ] **Timeline**: 2-3 days
- [ ] **Priority**: 🟡 MEDIUM

#### **3.4 Optimize Memory Layout for NPU** 🧠
- [ ] **Task**: NPU-specific memory access optimization
- [ ] **File**: `src/npu_memory_manager.py`
- [ ] **Optimizations**:
  - [ ] Tile-friendly data layout
  - [ ] Bank conflict avoidance
  - [ ] Prefetching strategies
  - [ ] Memory coalescing
- [ ] **Expected**: 10-20% performance improvement
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🟡 MEDIUM

#### **3.5 Performance Tuning and Profiling** 🔧
- [ ] **Task**: Identify and eliminate bottlenecks
- [ ] **Tools**:
  ```bash
  # NPU profiling
  xrt-smi dump --device 0
  
  # Memory bandwidth analysis
  python src/memory_profiler.py
  
  # Kernel execution profiling
  python src/kernel_profiler.py
  ```
- [ ] **Activities**:
  - [ ] Profile kernel execution time
  - [ ] Memory bandwidth utilization
  - [ ] DMA transfer analysis
  - [ ] Identify optimization opportunities
- [ ] **Timeline**: 2-3 days
- [ ] **Priority**: 🟡 MEDIUM

#### **3.6 Performance Measurement Phase 3** 📊
- [ ] **Task**: Verify Phase 3 performance targets
- [ ] **File**: `benchmarks/advanced_performance.py`
- [ ] **Target Performance**:
  - Single: 25-35ms (30-40 emb/sec) - **4x CPU speedup**
  - Batch 32: 30-40ms per embedding
- [ ] **Success Criteria**: Target performance achieved
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🔴 HIGH

---

## 📋 **PHASE 4: PRODUCTION (Week 7-8)**

### **🔴 CRITICAL PATH - Production Ready**

#### **4.1 Create Production Runtime** ⚔️
- [ ] **Task**: Production-grade NPU embedding service
- [ ] **File**: `src/production_runtime.py`
- [ ] **Features**:
  - [ ] Error handling and recovery
  - [ ] Resource management
  - [ ] Load balancing
  - [ ] Health monitoring
  - [ ] Graceful degradation (NPU → CPU fallback)
- [ ] **Timeline**: 3-4 days
- [ ] **Priority**: 🔴 HIGHEST

#### **4.2 Add Monitoring and Metrics** 📊
- [ ] **Task**: Production monitoring system
- [ ] **File**: `src/monitoring.py`
- [ ] **Metrics**:
  - [ ] Embeddings per second
  - [ ] Latency percentiles (p50, p95, p99)
  - [ ] NPU utilization
  - [ ] Memory usage
  - [ ] Error rates
  - [ ] Queue depth
- [ ] **Dashboard**: Real-time performance visualization
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🔴 HIGH

### **🟡 VALIDATION AND DEPLOYMENT - Week 7-8**

#### **4.3 Real-World Validation** 🧪
- [ ] **Task**: Test with actual embedding workloads
- [ ] **File**: `tests/real_workload_test.py`
- [ ] **Tests**:
  - [ ] Document embedding for search
  - [ ] Semantic similarity tasks
  - [ ] Clustering workloads
  - [ ] Large-scale batch processing
- [ ] **Validation**:
  - [ ] Embedding quality preserved (>99.9% similarity)
  - [ ] Performance targets met
  - [ ] Stability under load
- [ ] **Timeline**: 2-3 days
- [ ] **Priority**: 🔴 HIGH

#### **4.4 Documentation and Deployment** 📚
- [ ] **Task**: Complete documentation for deployment
- [ ] **Files**:
  - [ ] `docs/DEPLOYMENT_GUIDE.md` - Production deployment
  - [ ] `docs/OPTIMIZATION_GUIDE.md` - Performance optimization
  - [ ] `docs/TROUBLESHOOTING.md` - Common issues and fixes
  - [ ] `docs/API_REFERENCE.md` - Complete API documentation
- [ ] **Deployment**:
  - [ ] Docker container creation
  - [ ] Kubernetes deployment specs
  - [ ] Configuration templates
- [ ] **Timeline**: 2 days
- [ ] **Priority**: 🟡 MEDIUM

#### **4.5 Final Performance Measurement** 📊
- [ ] **Task**: Validate final performance targets
- [ ] **File**: `benchmarks/final_validation.py`
- [ ] **Target Performance**:
  - Single: 20-25ms (40-50 emb/sec) - **5x CPU speedup**
  - Batch 32: 20-30ms per embedding
  - Throughput: 1000+ embeddings/minute
- [ ] **Success Criteria**: **40-50 embeddings/sec achieved!** ✅
- [ ] **Timeline**: 1 day
- [ ] **Priority**: 🔴 HIGHEST

---

## 🎯 **SUCCESS CRITERIA & VALIDATION**

### **📊 Performance Metrics**

#### **Phase 1 Success (Week 2)**:
- [ ] NPU hardware detected ✅
- [ ] Basic kernel compilation working ✅
- [ ] Simple NPU execution verified ✅

#### **Phase 2 Success (Week 4)**:
- [ ] **15-20 embeddings/sec** (2x CPU speedup)
- [ ] End-to-end NPU pipeline working
- [ ] Quantization reducing memory usage

#### **Phase 3 Success (Week 6)**:
- [ ] **30-40 embeddings/sec** (4x CPU speedup)  
- [ ] Fused transformer kernels working
- [ ] Batch processing optimized

#### **Phase 4 Success (Week 8)**:
- [ ] **40-50 embeddings/sec** (5x CPU speedup) 🎯
- [ ] Production-ready deployment
- [ ] Real-world validation complete

### **📈 Quality Metrics**

#### **Embedding Quality**:
- [ ] **99.9% similarity** to CPU embeddings
- [ ] Consistent results across runs
- [ ] No quality degradation from optimization

#### **System Reliability**:
- [ ] **99% uptime** during testing
- [ ] Graceful error handling
- [ ] Automatic recovery from failures

#### **Resource Utilization**:
- [ ] **<500MB memory** usage
- [ ] NPU utilization >80%
- [ ] Stable operation for 1M+ embeddings

---

## 🛠️ **DEVELOPMENT TOOLS & UTILITIES**

### **📋 Essential Scripts**

#### **Setup and Configuration**:
```bash
# Environment setup
./scripts/setup_environment.sh

# NPU detection and verification
python src/verify_npu.py

# Model download and setup
python src/download_qwen3.py
```

#### **Development Tools**:
```bash
# MLIR kernel compilation
python scripts/compile_kernels.py

# Performance benchmarking
python benchmarks/benchmark_all.py

# Memory profiling
python scripts/profile_memory.py
```

#### **Testing and Validation**:
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Performance tests
python tests/performance/test_throughput.py
```

### **📊 Monitoring Commands**

#### **NPU Status**:
```bash
# NPU device info
xrt-smi examine

# NPU utilization
xrt-smi dump --device 0

# Memory usage
cat /sys/class/accel/accel0/device/mem_info
```

#### **Performance Monitoring**:
```bash
# Real-time performance
watch -n 1 python benchmarks/quick_perf_check.py

# Detailed profiling
python src/performance_profiler.py --detailed
```

---

## 🎯 **RISK MITIGATION & CONTINGENCY PLANS**

### **🚨 High-Risk Items**

#### **MLIR Kernel Compilation Issues**:
- **Risk**: MLIR-AIE compiler bugs or compatibility issues
- **Mitigation**: Start with simple kernels, incremental complexity
- **Fallback**: Use OpenCL or Vulkan compute for GPU acceleration
- **Timeline Impact**: +1-2 weeks if major issues

#### **NPU Hardware Compatibility**:
- **Risk**: Phoenix NPU driver issues or hardware limitations
- **Mitigation**: Verify hardware early, test with simple operations
- **Fallback**: iGPU acceleration (still provides speedup)
- **Timeline Impact**: No impact if fallback used

#### **Memory Bandwidth Limitations**:
- **Risk**: Shared memory bandwidth limiting NPU effectiveness
- **Mitigation**: Profile early, optimize data transfer patterns
- **Fallback**: Hybrid NPU+CPU processing
- **Timeline Impact**: +1 week for optimization

### **🟡 Medium-Risk Items**

#### **Quantization Quality Loss**:
- **Risk**: UC1-EMB quantization affecting embedding quality
- **Mitigation**: Validate quality early, tune quantization parameters
- **Fallback**: Use FP16 instead of INT4 (less speedup)

#### **Kernel Performance Not Meeting Targets**:
- **Risk**: NPU kernels slower than expected
- **Mitigation**: Profile early, optimize incrementally
- **Fallback**: CPU optimization (still valuable)

### **🟢 Low-Risk Items**

#### **Integration Complexity**:
- **Risk**: Difficulty integrating all components
- **Mitigation**: Modular development, thorough testing

#### **Documentation and Deployment**:
- **Risk**: Incomplete documentation
- **Mitigation**: Document incrementally during development

---

## 🎉 **PROJECT SUCCESS DEFINITION**

### **🏆 PRIMARY SUCCESS**:
**Achieve 30-50 embeddings/sec with Qwen3-Embedding-0.6B on AMD Phoenix NPU**

### **🥇 STRETCH GOALS**:
- [ ] >50 embeddings/sec (beyond target)
- [ ] Production deployment ready
- [ ] Reranker acceleration working
- [ ] Multi-model support

### **📊 KEY PERFORMANCE INDICATORS**:
1. **Performance**: 3-5x speedup over CPU baseline
2. **Quality**: >99.9% embedding similarity preserved  
3. **Reliability**: >99% uptime during testing
4. **Resource Efficiency**: <500MB memory usage
5. **Deployment**: Production-ready with monitoring

---

## 🚀 **GETTING STARTED**

### **🔥 IMMEDIATE NEXT STEPS (Next 48 Hours)**:

1. **Download Qwen3 Model**:
   ```bash
   cd models/
   huggingface-cli download Alibaba-NLP/Qwen3-Embedding-0.6B
   ```

2. **Copy Working Components**:
   ```bash
   cp -r ../UC1-Embedding-NPU/quantization ./models/
   cp -r ../UC1-Embedding-NPU/npu_kernels ./kernels/
   ```

3. **Test NPU Detection**:
   ```bash
   python src/test_npu_basic.py
   ```

4. **Establish CPU Baseline**:
   ```bash
   python benchmarks/cpu_baseline.py
   ```

### **📅 WEEK 1 GOALS**:
- [ ] NPU detection working
- [ ] Model downloaded and loaded
- [ ] CPU baseline established
- [ ] First MLIR kernel compiling

**Let's build this! The foundation is solid, the plan is clear, and the target is achievable.** 🦄⚡

---

*Ready to copy/reuse existing code and achieve 30-50 embeddings/sec with Qwen3-0.6B on NPU!*