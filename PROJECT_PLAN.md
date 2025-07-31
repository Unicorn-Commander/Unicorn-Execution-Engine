# 🦄 MAGIC UNICORN PROJECT PLAN - COMPLETE TASK LIST

**Status**: Foundation Complete - Ready for Optimization  
**Target**: 50+ tokens/sec with NPU+iGPU hybrid acceleration  
**Timeline**: Phase 1 (immediate), Phase 2 (1-2 days), Phase 3 (3-5 days)

---

## 📋 **PHASE 1: MATCH OLLAMA BASELINE (21+ tok/s)** - IMMEDIATE
*Timeline: Today - High Priority*

### **Task 1.1: ROCm PyTorch Installation** ⚡
- [ ] Install PyTorch ROCm version: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1`
- [ ] Verify GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Test tensor operations on GPU: Basic matrix multiplication benchmark
- [ ] **Expected Result**: PyTorch detects AMD GPU as CUDA device

### **Task 1.2: Environment Configuration** ⚡
- [ ] Set HSA override: `export HSA_OVERRIDE_GFX_VERSION=11.0.3`
- [ ] Configure HIP visibility: `export HIP_VISIBLE_DEVICES=0`
- [ ] Test ROCm detection: `rocm-smi` should show gfx1103
- [ ] **Expected Result**: Same environment as working ollama setup

### **Task 1.3: ROCm Pipeline Integration** ⚡
- [ ] Update `magic_unicorn_rocm_speed.py` with working PyTorch GPU detection
- [ ] Replace OpenCL operations with PyTorch GPU operations
- [ ] Test GEMM operations: Move tensors to GPU, benchmark speed
- [ ] **Expected Result**: GPU-accelerated operations working

### **Task 1.4: Baseline Performance Test** ⚡
- [ ] Run full transformer layer with ROCm acceleration
- [ ] Benchmark against 21 tok/s ollama baseline
- [ ] Measure: Layer time, full model projection, tokens/sec
- [ ] **Target**: Match or exceed 21 tokens/sec

---

## 📋 **PHASE 2: CUSTOM KERNEL ADVANTAGE (40-60 tok/s)** - 1-2 DAYS
*Timeline: Next 1-2 days - High Priority*

### **Task 2.1: Custom RDNA3 Kernel Development** 🔥
- [ ] Research RDNA3 gfx1103 architecture specifications
- [ ] Write custom HIP/ROCm kernels for GEMM operations
- [ ] Optimize for gfx1103 work group sizes and memory patterns
- [ ] **Expected Result**: Custom kernels faster than PyTorch defaults

### **Task 2.2: Memory Bandwidth Optimization** 🔥
- [ ] Implement FP16 precision for 2x memory bandwidth
- [ ] Add zero-copy memory operations between operations
- [ ] Optimize memory access patterns for RDNA3 cache
- [ ] **Expected Result**: 50% memory usage reduction, faster transfers

### **Task 2.3: Kernel Fusion Implementation** 🔥
- [ ] Fuse QKV operations into single kernel call
- [ ] Fuse Gate+Up FFN operations
- [ ] Implement attention+projection fusion
- [ ] **Expected Result**: Reduced kernel launch overhead

### **Task 2.4: Performance Validation** 🔥
- [ ] Benchmark custom kernels vs PyTorch defaults
- [ ] Measure end-to-end performance improvement
- [ ] Validate numerical accuracy vs reference
- [ ] **Target**: 40-60 tokens/sec (2-3x ollama baseline)

---

## 📋 **PHASE 3: NPU TURBO MODE (80+ tok/s)** - 3-5 DAYS
*Timeline: Next 3-5 days - Medium Priority*

### **Task 3.1: NPU Attention Acceleration** 🚀
- [ ] Complete MLIR-AIE attention kernel for Phoenix 5-column topology
- [ ] Compile attention kernel to working XCLBIN
- [ ] Test NPU attention vs CPU attention performance
- [ ] **Expected Result**: NPU attention 3-5x faster than CPU

### **Task 3.2: NPU+iGPU Parallel Execution** 🚀
- [ ] Implement concurrent NPU (attention) + iGPU (GEMM) execution
- [ ] Add pipeline parallelism for overlapping operations
- [ ] Optimize memory sharing between NPU and iGPU
- [ ] **Expected Result**: True hybrid acceleration working

### **Task 3.3: Advanced Optimizations** 🚀
- [ ] Implement INT8 quantization for models
- [ ] Add batch processing for multiple sequences
- [ ] Optimize for different sequence lengths (32, 128, 512)
- [ ] **Expected Result**: Maximum hardware utilization

### **Task 3.4: Production Integration** 🚀
- [ ] Load real Gemma 4B model weights
- [ ] Implement text generation pipeline
- [ ] Add tokenization and decoding
- [ ] **Target**: 80+ tokens/sec with real models

---

## 📋 **PHASE 4: PRODUCTION DEPLOYMENT** - 1-2 WEEKS
*Timeline: Future development - Lower Priority*

### **Task 4.1: CLI Interface Development** 💻
- [ ] Create command-line interface for Magic Unicorn
- [ ] Add model selection (Gemma 4B, 27B, other models)
- [ ] Implement configuration options (precision, sequence length)
- [ ] Add performance monitoring and statistics

### **Task 4.2: Model Support Expansion** 💻
- [ ] Add support for Llama models
- [ ] Implement Qwen model support
- [ ] Add automatic model downloading and caching
- [ ] Support for custom fine-tuned models

### **Task 4.3: Distribution and Packaging** 💻
- [ ] Create installation scripts
- [ ] Docker container for easy deployment
- [ ] Documentation for end users
- [ ] Performance tuning guides

---

## 🎯 **SUCCESS CRITERIA & MILESTONES**

### **Phase 1 Success** ✅
- [ ] **Performance**: 21+ tokens/sec achieved
- [ ] **Validation**: Matches ollama baseline
- [ ] **Stability**: Reliable execution without crashes

### **Phase 2 Success** ✅
- [ ] **Performance**: 40-60 tokens/sec achieved  
- [ ] **Advantage**: 2-3x improvement over ollama
- [ ] **Quality**: Custom kernels outperform defaults

### **Phase 3 Success** ✅
- [ ] **Performance**: 80+ tokens/sec achieved
- [ ] **Innovation**: NPU+iGPU hybrid working
- [ ] **Production**: Real models running

---

## 🚨 **CRITICAL PATH PRIORITIES**

### **Immediate (Today)** 🔥
1. Install PyTorch ROCm
2. Fix GPU detection
3. Test ROCm acceleration
4. Match 21 tok/s baseline

### **High Priority (This Week)** ⚡
1. Custom RDNA3 kernels
2. Memory optimization
3. Kernel fusion
4. 40-60 tok/s target

### **Medium Priority (Next Week)** 🚀
1. NPU attention kernels
2. Hybrid execution
3. Real model integration
4. 80+ tok/s target