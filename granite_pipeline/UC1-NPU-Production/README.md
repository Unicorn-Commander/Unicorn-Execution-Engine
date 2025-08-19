# UC1-NPU-Production

## 🎯 Goal: Get Qwen3-0.6B Embedding Model Running on NPU at 30-50 embeddings/sec

### Project Scope
**Primary Target**: Qwen3-Embedding-0.6B model accelerated on AMD Phoenix NPU
**Performance Goal**: 30-50 embeddings/sec (3-5x improvement over 9 emb/sec CPU baseline)
**Timeline**: 6-8 weeks focused development

---

## 📋 Project Checklist

### Phase 1: Foundation (Week 1-2) ✅ STARTED
- [ ] **Download real Qwen3-Embedding-0.6B model**
- [ ] **Copy working NPU components** from previous projects  
- [ ] **Compile first MLIR kernel to XCLBIN**
- [ ] **Test basic NPU execution** with simple operations
- [ ] **Measure baseline NPU vs CPU** performance

### Phase 2: Core Optimization (Week 3-4)
- [ ] **Implement persistent kernel execution** (eliminate launch overhead)
- [ ] **Create direct INT4 compute kernels** (4x efficiency)
- [ ] **Build zero-copy DMA framework** (eliminate transfer overhead)
- [ ] **Test end-to-end embedding generation** on NPU
- [ ] **Target**: 15-20 embeddings/sec (2x CPU)

### Phase 3: Advanced Optimization (Week 5-6)
- [ ] **Implement fused transformer pipeline** (12 layers in one kernel)
- [ ] **Add batch processing optimization** (batch 32-64)
- [ ] **Optimize memory layout** for NPU architecture
- [ ] **Performance tuning** and bottleneck elimination
- [ ] **Target**: 30-40 embeddings/sec (4x CPU)

### Phase 4: Production (Week 7-8)
- [ ] **Create production runtime** with error handling
- [ ] **Add monitoring and metrics**
- [ ] **Real-world validation** with actual workloads
- [ ] **Documentation and deployment**
- [ ] **Target**: 40-50 embeddings/sec (5x CPU)

---

## 🏗️ Project Structure

```
UC1-NPU-Production/
├── README.md                    # This file
├── PROJECT_CHECKLIST.md         # Detailed task breakdown
├── PERFORMANCE_TARGETS.md       # Performance goals and metrics
│
├── src/                         # Source code
│   ├── npu_engine.py           # Main NPU acceleration engine
│   ├── model_loader.py         # Qwen3 model loading utilities
│   ├── memory_manager.py       # NPU memory management
│   └── benchmark.py            # Performance testing
│
├── kernels/                     # NPU kernels (MLIR → XCLBIN)
│   ├── embedding_lookup.mlir   # Embedding table lookup
│   ├── transformer_layer.mlir  # Single transformer layer
│   ├── transformer_fused.mlir  # Fused 12-layer pipeline
│   └── int4_compute.mlir       # Direct INT4 operations
│
├── models/                      # Model storage
│   ├── qwen3-0.6b-original/    # Original Qwen3 model
│   ├── qwen3-0.6b-uc1/        # UC1-optimized version
│   └── quantization/           # Quantization utilities
│
├── benchmarks/                  # Performance testing
│   ├── cpu_baseline.py         # CPU performance measurement
│   ├── npu_comparison.py       # NPU vs CPU comparison  
│   └── real_workload_test.py   # Real-world usage testing
│
├── tests/                       # Unit tests
│   ├── test_npu_kernels.py     # Kernel functionality tests
│   ├── test_model_loading.py   # Model loading tests
│   └── test_performance.py     # Performance regression tests
│
└── docs/                       # Documentation
    ├── DEVELOPMENT_LOG.md      # Daily progress log
    ├── OPTIMIZATION_GUIDE.md   # Technical optimization details
    └── DEPLOYMENT_GUIDE.md     # Production deployment
```

---

## 🎯 Performance Targets

### Current Baseline (CPU):
- **Single embedding**: 110ms (9 embeddings/sec)
- **Batch 32**: 124ms per embedding (8 embeddings/sec)
- **Model**: Qwen3-Embedding-0.6B (389MB, 102M parameters)

### Target Performance (NPU):

#### **Phase 2 Target** (Week 3-4):
- **Single**: 50-70ms (15-20 embeddings/sec) - 2x speedup
- **Batch 32**: 60-80ms per embedding (12-16 emb/sec)

#### **Phase 3 Target** (Week 5-6):
- **Single**: 25-35ms (30-40 embeddings/sec) - 4x speedup  
- **Batch 32**: 30-40ms per embedding (25-33 emb/sec)

#### **Phase 4 Target** (Week 7-8):
- **Single**: 20-25ms (40-50 embeddings/sec) - 5x speedup
- **Batch 32**: 20-30ms per embedding (33-50 emb/sec)

---

## 🔧 Technical Approach

### 1. **Copy and Reuse Working Components**
```bash
# From UC1-Embedding-NPU:
- NPU hardware detection and XRT integration
- MLIR-AIE kernel framework
- UC1-EMB quantization format
- Memory management utilities

# From Unicorn Execution Engine:
- Performance measurement methodology
- INT4/INT8 optimization techniques
- Memory bandwidth analysis
- Kernel fusion strategies
```

### 2. **Focus on Qwen3-0.6B Specifically**
- Download actual model weights (not synthetic)
- Optimize for specific architecture (768 dim, 12 layers)
- Custom quantization for this model
- Real-world text processing pipeline

### 3. **Incremental Development**
- Start with basic NPU execution
- Add optimizations one by one
- Measure performance at each step
- Validate against real workloads

### 4. **Production-Ready Code**
- Error handling and robustness
- Monitoring and logging
- Clean APIs and documentation
- Deployment automation

---

## 📊 Success Metrics

### **Technical Metrics**:
- [ ] **30+ embeddings/sec** single requests
- [ ] **40+ embeddings/sec** batch processing  
- [ ] **<50ms latency** per embedding
- [ ] **<500MB memory** usage
- [ ] **Stable operation** for 1M+ embeddings

### **Quality Metrics**:
- [ ] **99.9% similarity** to CPU embeddings
- [ ] **No quality degradation** from optimization
- [ ] **Consistent results** across runs

### **Production Metrics**:
- [ ] **99% uptime** in testing
- [ ] **Clear error messages** and logging
- [ ] **Easy deployment** and configuration
- [ ] **Performance monitoring** dashboards

---

## 🚀 Getting Started

### Prerequisites:
- AMD Phoenix NPU hardware
- XRT 2.20.0+ runtime
- MLIR-AIE development environment
- Python 3.8+ with PyXRT

### Quick Start:
```bash
cd UC1-NPU-Production

# Phase 1: Setup and baseline
python src/model_loader.py  # Download Qwen3 model
python benchmarks/cpu_baseline.py  # Measure CPU performance
python src/npu_engine.py --test  # Test basic NPU functionality

# Phase 2: First optimization
# (TBD - will be filled as we develop)
```

---

## 📈 Expected Timeline

**Week 1**: Setup, model download, basic NPU testing
**Week 2**: First XCLBIN compilation and execution
**Week 3**: Persistent kernels and DMA optimization  
**Week 4**: INT4 compute and performance tuning
**Week 5**: Kernel fusion and batch optimization
**Week 6**: Advanced optimizations and edge cases
**Week 7**: Production runtime and monitoring
**Week 8**: Real-world validation and deployment

---

## 🎯 Focus Areas

### **Primary Focus**: Qwen3-0.6B embeddings only
- No rerankers (save for later)
- No other models (focus on one)
- No theoretical work (real implementation only)

### **Success Definition**: 
**30-50 embeddings/sec with real Qwen3-0.6B model on NPU**

Let's build this! 🚀