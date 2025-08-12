# 🤖 GEMINI CLI TASK INSTRUCTIONS - MAGIC UNICORN PROJECT

## **PROJECT CONTEXT**
You are working on the **Magic Unicorn** project - a breakthrough NPU+iGPU hybrid inference engine for AMD Phoenix APU. The foundation infrastructure is complete and operational. Your role is to optimize performance through research, algorithm design, and code development.

## **CURRENT STATUS**
- ✅ **NPU Access**: Phoenix XDNA1 (16 TOPS) proven working via XRT
- ✅ **iGPU Pipeline**: AMD gfx1103 with OpenCL kernels operational
- ✅ **Hybrid Architecture**: Complete transformer layer functional
- ✅ **Baseline**: 0.44 tok/s achieved, 21 tok/s proven possible (ollama)
- 🎯 **Target**: Optimize to 50+ tokens/sec with custom kernels

## **YOUR PRIMARY TASKS**

### **TASK 1: Custom RDNA3 Kernel Development** 🔥
**Objective**: Create optimized HIP/ROCm kernels for AMD RDNA3 gfx1103

**What to Research & Develop**:
1. **RDNA3 Architecture Analysis**:
   - Research gfx1103 specifications (compute units, cache, memory)
   - Optimal work group sizes for transformer operations
   - Memory coalescing patterns and bandwidth optimization
   - SIMD utilization strategies

2. **Custom Kernel Development**:
   - Write HIP kernel for optimized GEMM (matrix multiplication)
   - Attention kernel with causal masking and softmax
   - Element-wise operations (SiLU, layer norm, bias addition)
   - FP16 precision kernels for 2x bandwidth improvement

3. **Kernel Fusion Strategy**:
   - Design QKV fusion (3 operations → 1 kernel)
   - Gate+Up FFN fusion for reduced overhead
   - Attention+projection fusion
   - Multi-operation scheduling optimization

**Expected Deliverables**:
- Complete HIP/ROCm kernel source files
- Compilation and build instructions
- Integration guide for existing codebase
- Performance optimization documentation
- Theoretical speedup analysis vs current OpenCL

### **TASK 2: Advanced Memory Optimization** ⚡
**Objective**: Design zero-copy, high-bandwidth memory management

**What to Design**:
1. **FP16 Precision System**:
   - Convert existing FP32 operations to FP16
   - Maintain numerical stability for transformer operations
   - 2x memory bandwidth improvement strategy

2. **Zero-Copy Architecture**:
   - Eliminate unnecessary CPU↔GPU transfers
   - Design in-place operations where possible
   - Memory pool management for reuse

3. **Layout Optimization**:
   - Optimal tensor layouts for RDNA3 cache
   - Memory access pattern analysis
   - Bandwidth utilization maximization

**Expected Deliverables**:
- Memory management system design
- FP16 conversion implementation
- Zero-copy operation framework
- Memory bandwidth benchmarking tools

### **TASK 3: MLIR-AIE NPU Kernel Completion** 🚀
**Objective**: Complete attention kernel for Phoenix NPU 5-column topology

**What to Develop**:
1. **Phoenix NPU Attention Kernel**:
   - MLIR-AIE code targeting 5-column (4x5) topology
   - Parallel processing across 20 AIE2 tiles
   - Memory bank optimization (131071, 65536, 65537)
   - Causal attention with softmax

2. **Compilation Pipeline**:
   - MLIR compilation to XCLBIN format
   - Integration with existing XRT infrastructure
   - Testing and validation framework

**Expected Deliverables**:
- Complete MLIR-AIE attention kernel
- Compilation scripts and documentation
- XRT integration guide
- Performance projection vs CPU attention

## **TECHNICAL SPECIFICATIONS**

### **Hardware Target**:
- **NPU**: AMD Phoenix XDNA1, 16 TOPS, 5-column topology
- **iGPU**: AMD RDNA3 gfx1103, 6 CUs, 38GB memory
- **System**: Linux 6.14, ROCm 6.1, XRT 2.20.0

### **Performance Targets**:
- **Current**: 0.44 tokens/sec with OpenCL
- **Baseline**: 21 tokens/sec (proven with ollama)
- **Target**: 50+ tokens/sec with optimizations
- **Stretch**: 100+ tokens/sec with full NPU integration

### **Key Files to Reference**:
Located in `/home/ucadmin/Development/Unicorn-Execution-Engine/`:
- `optimized_hybrid_pipeline.py` - Current working implementation
- `magic_unicorn_fp16_optimized.py` - FP16 attempt
- `magic_unicorn_ultra_speed.py` - OpenCL optimizations
- `NPU_DEVELOPMENT_GUIDE.md` - NPU hardware details
- `phoenix_npu_mlir_kernel.mlir` - Existing MLIR code

## **IMPLEMENTATION GUIDELINES**

### **Code Quality Requirements**:
- Production-ready, well-documented code
- Compatible with existing codebase architecture
- Include error handling and fallback mechanisms
- Comprehensive testing and validation

### **Performance Focus**:
- Target 2-5x improvement over current performance
- Minimize kernel launch overhead
- Maximize memory bandwidth utilization
- Optimize for transformer workload patterns

### **Documentation Standards**:
- Clear implementation instructions
- Performance analysis and projections
- Integration steps with existing code
- Troubleshooting guides

## **CONSTRAINTS**
- **Focus on algorithm/code development** (no hardware testing required)
- **Research-based optimization** using documented specifications
- **Theoretical performance modeling** based on hardware capabilities
- **Integration-ready code** that can be tested on actual hardware

## **SUCCESS METRICS**
- **Performance**: Designs targeting 40-80 tokens/sec
- **Innovation**: Novel optimization approaches beyond standard libraries
- **Quality**: Production-ready implementations
- **Integration**: Seamless compatibility with Magic Unicorn codebase
- **Documentation**: Actionable implementation guides

Your expertise in optimization research and algorithm development will accelerate Magic Unicorn to production-ready performance! Focus on creating the most optimized kernels and memory management possible for this specific hardware combination.