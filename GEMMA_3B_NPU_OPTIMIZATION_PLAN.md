
# Project Plan: Gemma 3B NPU Optimization

## Project Goal

Optimize multiple state-of-the-art models for high-performance, low-latency inference using AMD Ryzen AI hybrid NPU+iGPU+CPU execution:

1. **Gemma 3n E2B** (MatFormer architecture) - ✅ **COMPLETED**
2. **Qwen2.5-7B-Instruct** (Standard transformer) - 🚧 **IN PROGRESS**
3. **Future models** (Scaling and optimization)

Focus on maximizing NPU's strength in attention operations while leveraging iGPU for memory-intensive tasks.

## Project Phases & Task List

### Phase 1: Environment Setup and Baseline (1-2 days)

- [ ] **Task 1.1: Verify and Solidify NPU Environment.**
  - [x] ~~Run the `install_npu_stack.sh` script to ensure all dependencies (XRT, MLIR-AIE, drivers) are correctly installed.~~ (Failed, pivoted to Docker)
  - [ ] **NEW:** Create `Dockerfile` for a containerized NPU development environment.
  - [x] **NEW:** Create `Dockerfile` for a containerized NPU development environment.
  - [ ] **NEW:** Build the Docker image from the `Dockerfile` (IN PROGRESS - encountered persistent build issues, pivoted to a minimal Dockerfile approach).
  - [ ] **NEW:** Run the Docker container and access the development environment.
  - [ ] Execute `verify_npu_setup.sh` inside the container to confirm the NPU is detected and the software stack is functional.
  - [ ] Document the exact versions of the kernel, XRT, and MLIR-AIE for reproducibility.
- [ ] **Task 1.2: Configure NPU Turbo Mode.**
  - [ ] Execute `/opt/xilinx/xrt/bin/xrt-smi configure --pmode turbo` for maximum NPU performance
  - [ ] Verify configuration with `/opt/xilinx/xrt/bin/xrt-smi examine -v`
  - [ ] Document NPU performance baseline with turbo enabled
- [ ] **Task 1.3: Establish Baseline Performance.**
  - [ ] Implement CPU-based inference pipeline for Gemma 3n E2B model
  - [ ] Benchmark latency and throughput for standardized text generation
  - [ ] Profile the CPU implementation to identify bottlenecks suitable for NPU acceleration

### Phase 2: Hybrid NPU+iGPU Architecture Implementation (3-5 days)

- [ ] **Task 2.1: Implement NPU Attention Acceleration.**
  - [ ] Deploy Gemma 3n E2B with NPU handling attention mechanisms and prefill phase (NPU's strength: compute-intensive operations up to 50 TOPS)
  - [ ] Use FP16/BF16 precision for NPU operations (native NPU support)
  - [ ] Implement Per-Layer Embeddings (PLE) optimization for 2GB memory footprint
- [ ] **Task 2.2: Configure iGPU for Decode Phase.**
  - [ ] Set up Radeon 780M iGPU for memory-intensive decode operations and large matrix multiplications  
  - [ ] Configure ROCm/HIP backend for iGPU execution
  - [ ] Implement hybrid scheduler: NPU for prefill, iGPU for decode
- [ ] **Task 2.3: Validate Hybrid Performance.**
  - [ ] Benchmark hybrid NPU+iGPU vs pure CPU baseline
  - [ ] Measure time-to-first-token (TTFT) with NPU prefill and tokens-per-second (TPS) with iGPU decode
  - [ ] Profile memory usage across NPU (2GB), iGPU (8-12GB), and system RAM

### Phase 3: Scale to Gemma 3n E4B and Advanced Optimization (1-2 weeks)

- [ ] **Task 3.1: Scale to Gemma 3n E4B Model.**
  - [ ] Deploy the larger E4B model (8B parameters, 3GB memory footprint) using the proven hybrid architecture
  - [ ] Leverage MatFormer Mix-n-Match capability for dynamic model sizing
  - [ ] Maintain NPU for attention/prefill, iGPU for decode approach with larger model
- [ ] **Task 3.2: Develop Custom NPU Attention Kernels.**
  - [ ] Optimize attention mechanisms specifically for NPU using MLIR-AIE
  - [ ] Focus on multi-head attention and embedding operations (NPU's compute strength)
  - [ ] Implement custom kernels for 32K context window support
- [ ] **Task 3.3: Advanced Hybrid Memory Management.**
  - [ ] Implement unified memory architecture across NPU (2-3GB), iGPU (8-12GB), and system RAM (96GB)
  - [ ] Zero-copy transfers between compute units where possible
  - [ ] Dynamic memory allocation based on model phase (prefill vs decode)

### Phase 4: Production Integration and Multimodal Support (1 week)

- [ ] **Task 4.1: Implement Multimodal Capabilities.**
  - [ ] Enable Gemma 3n's native multimodal support (text, image, audio)
  - [ ] Use NPU for embedding generation from multiple modalities
  - [ ] Leverage 32K context window for long multimodal conversations
- [ ] **Task 4.2: OGA Integration for Advanced Text Generation.**
  - [ ] Integrate with ONNX Generator API (OGA) for optimized text generation workflows
  - [ ] Implement advanced sampling techniques (top-k, top-p, temperature)
  - [ ] Support for streaming generation with hybrid NPU+iGPU pipeline
- [ ] **Task 4.3: Dynamic Load Balancing and Thermal Management.**
  - [ ] Implement intelligent scheduler that adapts to thermal conditions
  - [ ] Dynamic switching between E2B and E4B models based on performance requirements
  - [ ] Monitor NPU/iGPU utilization and adjust workload distribution

### Phase 5: Final Benchmarking, Profiling, and Documentation (3-4 days)

- [ ] **Task 5.1: Comprehensive Performance Evaluation.**
  - [ ] Benchmark the final, fully optimized solution against the CPU baseline and the initial NPU implementation.
  - [ ] Use the custom NPU profiler and performance monitor to get detailed metrics.
- [ ] **Task 5.2: Create a "Production-Ready" Example.**
  - [ ] Develop a clean, well-documented example application.
- [ ] **Task 5.3: Update Project Documentation.**
  - [ ] Document the entire process, including the final architecture, performance gains, and lessons learned.

## Corrected Implementation Strategy (Updated Jan 2025)

### Key Architecture Corrections

**NPU Optimization Focus**: NPU excels at attention operations and prefill phase (up to 50 TOPS compute-intensive tasks)
**iGPU Strength**: Memory-intensive decode phase and large matrix operations
**Hybrid Flow**: NPU handles prefill/attention → iGPU handles decode → CPU orchestrates

### Updated Docker Configuration

Use NPU-Development/Dockerfile with device passthrough:
```bash
docker run -it --privileged \
  --device=/dev/accel/accel0 \
  --device=/dev/dri \
  -v /dev:/dev \
  -v $(pwd):/workspace \
  npu-dev-env
```

### Performance Optimization Commands

Configure NPU for maximum performance:
```bash
# Use direct paths (no symlinks)
/opt/xilinx/xrt/bin/xrt-smi configure --pmode turbo
/opt/xilinx/xrt/bin/xrt-smi examine -v  # Verify configuration
```

### Expected Performance with Corrected Strategy

**Gemma 3n E2B (2GB NPU memory)**:
- Time-to-first-token: ~20-40ms (NPU prefill)
- Tokens per second: 40-80 TPS (iGPU decode)
- Memory efficiency: 2GB NPU + 8GB iGPU + flexible system RAM

**Gemma 3n E4B (3GB NPU memory)**:
- Time-to-first-token: ~30-60ms (NPU prefill)  
- Tokens per second: 30-60 TPS (iGPU decode)
- Enhanced quality with MatFormer Mix-n-Match capability
