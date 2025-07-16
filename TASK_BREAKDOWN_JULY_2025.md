# 📋 UNICORN EXECUTION ENGINE - DETAILED TASK BREAKDOWN

**Created**: July 15, 2025  
**Purpose**: Comprehensive task list with clear subtasks for parallel AI development

---

## 🎯 **PROJECT GOALS**
1. Achieve real NPU+iGPU inference (no CPU fallback)
2. Fast model loading (<30 seconds like Ollama)
3. High performance inference (81+ TPS target, 1337 TPS demonstrated possible)
4. Generate real text about "Magic Unicorn Unconventional Technology & Stuff"

---

## 📊 **CURRENT STATUS SUMMARY**

### ✅ **Completed**
- Dimension mismatch issues fixed
- Model loads to GPU (16GB VRAM + 38GB GTT)
- Persistent buffer optimization proven (1337 TPS in tests)
- Basic parallel loading implemented

### ❌ **Not Working**
- NPU driver (missing dependencies)
- Fast model loading (still 2+ minutes)
- Low GPU utilization during inference
- No meaningful text generation

---

## 🔧 **TASK CATEGORIES**

### **1. NPU DRIVER FIX** 🧠
**Complexity**: Medium  
**Suitable for**: Any AI with system/driver experience  
**Time estimate**: 2-4 hours

#### **Task 1.1: Install Missing XRT Libraries**
- **Problem**: `libxrt_core.so.2` not found
- **Subtasks**:
  - [ ] Find correct XRT package for Ubuntu/AMD system
  - [ ] Install AMD XRT runtime dependencies
  - [ ] Verify `/usr/local/xrt/lib/libxrt_core.so.2` exists
  - [ ] Test NPU driver loading with `npu_attention_kernel_real.py`
- **Test**: `python -c "import ctypes; ctypes.CDLL('/usr/local/xrt/lib/libxrt_driver_xdna.so')"`

#### **Task 1.2: Alternative NPU Access Method**
- **If XRT fails**: Research alternative NPU access
- **Subtasks**:
  - [ ] Investigate direct `/dev/accel/accel0` interface
  - [ ] Check for AMD ROCm NPU support
  - [ ] Look for ONNX Runtime NPU backend
  - [ ] Create minimal NPU test without XRT
- **Deliverable**: Working NPU execution proof-of-concept

---

### **2. FAST MODEL LOADING** 🚀
**Complexity**: High  
**Suitable for**: AI with parallel programming experience  
**Time estimate**: 4-6 hours

#### **Task 2.1: Eliminate Transpose Operations**
- **Problem**: CPU-bound transpose operations during loading
- **Location**: `pure_hardware_pipeline_fixed.py` - `_load_tensor_to_gpu()`
- **Subtasks**:
  - [ ] Pre-convert model weights to correct format (one-time operation)
  - [ ] Create script to transpose and save weights offline
  - [ ] Modify loader to skip transpose if pre-converted model exists
  - [ ] Test loading speed with pre-transposed weights
- **Target**: <30 second load time

#### **Task 2.2: True Memory-Mapped Loading**
- **Like Ollama**: Direct mmap to GPU without CPU copies
- **Subtasks**:
  - [ ] Implement direct safetensors → GPU memory mapping
  - [ ] Use `mmap.PROT_READ` with `MAP_POPULATE` for fast access
  - [ ] Batch GPU buffer allocations (not one-by-one)
  - [ ] Profile and optimize memory transfer patterns
- **Files to modify**:
  - `pure_mmap_loader.py`
  - `pure_hardware_pipeline_fixed.py`

#### **Task 2.3: Optimize for HMA Architecture**
- **Leverage unified memory**: No CPU↔GPU copies needed
- **Subtasks**:
  - [ ] Research AMD HMA (Heterogeneous Memory Architecture)
  - [ ] Use `HSA_ENABLE_SDMA=0` for large transfers
  - [ ] Implement zero-copy buffer sharing
  - [ ] Test with `numactl` for NUMA optimization
- **Reference**: `fast_parallel_loader.py` starter code

---

### **3. GPU UTILIZATION & PERFORMANCE** ⚡
**Complexity**: Medium  
**Suitable for**: AI with GPU/Vulkan experience  
**Time estimate**: 3-5 hours

#### **Task 3.1: Verify GPU Kernel Execution**
- **Problem**: GPU shows 0% usage during inference
- **Subtasks**:
  - [ ] Add GPU profiling to Vulkan operations
  - [ ] Use `VK_LAYER_KHRONOS_validation` for verification
  - [ ] Add timing around `vkQueueSubmit` calls
  - [ ] Monitor with `radeontop` during inference
- **Expected**: GPU usage >50% during matrix operations

#### **Task 3.2: Implement Persistent Model Server**
- **Keep model in GPU memory**: No reloading between requests
- **Subtasks**:
  - [ ] Create `persistent_model_server.py`
  - [ ] Load model once, serve many requests
  - [ ] Implement request queue
  - [ ] Add health check endpoint
- **Architecture**:
  ```python
  # Pseudo-code
  class PersistentModelServer:
      def __init__(self):
          self.pipeline = load_model_once()
      def generate(self, prompt):
          return self.pipeline.generate(prompt)
  ```

#### **Task 3.3: Optimize Vulkan Shaders**
- **Current**: Generic shaders, not optimized for RDNA3
- **Subtasks**:
  - [ ] Profile shader performance with RGP (Radeon GPU Profiler)
  - [ ] Optimize workgroup sizes for RDNA3 (Wave32 mode)
  - [ ] Implement specialized kernels for common sizes
  - [ ] Test INT8/INT4 compute paths
- **Files**: `*.comp` shader files

---

### **4. TEXT GENERATION** 📝
**Complexity**: Medium  
**Suitable for**: Any AI with LLM experience  
**Time estimate**: 2-3 hours

#### **Task 4.1: Implement Proper Tokenization**
- **Current**: Dummy tokenization `ord(c) % 1000`
- **Subtasks**:
  - [ ] Load actual Gemma tokenizer
  - [ ] Implement `encode()` and `decode()` methods
  - [ ] Handle special tokens properly
  - [ ] Test with "Magic Unicorn" prompt
- **Reference**: HuggingFace tokenizers library

#### **Task 4.2: Complete Generation Pipeline**
- **Missing**: Logits → token sampling
- **Subtasks**:
  - [ ] Implement output projection layer
  - [ ] Add temperature/top-p/top-k sampling
  - [ ] Implement KV cache properly
  - [ ] Add stopping criteria (EOS token)
- **Test**: Generate coherent text about the company

---

## 🎯 **QUICK WINS** (Can be done in parallel)

### **Task A: Create Benchmarking Suite**
- **Time**: 1-2 hours
- **Subtasks**:
  - [ ] Create `benchmark_suite.py` with multiple tests
  - [ ] Add automated performance regression detection
  - [ ] Generate performance report with graphs
  - [ ] Compare against baseline metrics

### **Task B: Documentation Update**
- **Time**: 1 hour
- **Subtasks**:
  - [ ] Update README.md with current status
  - [ ] Create PERFORMANCE.md with all benchmarks
  - [ ] Document GPU memory layout
  - [ ] Add troubleshooting guide

### **Task C: Clean Up Old Files**
- **Time**: 30 minutes
- **Subtasks**:
  - [ ] Remove outdated status files
  - [ ] Consolidate duplicate implementations
  - [ ] Archive old approaches
  - [ ] Update .gitignore

---

## 🤖 **RECOMMENDED AI ASSIGNMENTS**

### **For Gemini CLI** (or another AI):
1. **Task 2.1**: Eliminate transpose operations (clear, well-defined)
2. **Task 4.1**: Implement proper tokenization (standard LLM task)
3. **Task A**: Create benchmarking suite (straightforward)

### **For Claude/GPT-4**:
1. **Task 1.1-1.2**: NPU driver investigation (complex system work)
2. **Task 2.2**: Memory-mapped loading (architecture knowledge needed)
3. **Task 3.1-3.2**: GPU optimization (requires deep understanding)

### **For You to Do**:
1. Test each fix as it's completed
2. Monitor GPU/NPU usage
3. Verify performance improvements
4. Test Magic Unicorn text generation

---

## 📏 **SUCCESS METRICS**

1. **Model Loading**: <30 seconds (currently 2+ minutes)
2. **TPS**: 81+ (currently unmeasured in full pipeline)
3. **GPU Usage**: >50% during inference (currently ~0%)
4. **NPU Active**: No CPU fallback (currently failing)
5. **Text Quality**: Coherent output about company (currently none)

---

## 🚀 **GETTING STARTED**

```bash
# Environment setup for any AI
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
source /home/ucadmin/activate-pure-hardware-env.sh

# Test current state
python test_gpu_simple.py

# Monitor GPU
watch -n 0.5 'radeontop -d - -l 1 2>/dev/null | grep -E "(gpu|vram|gtt)"'
```

---

## 📝 **NOTES FOR AI ASSISTANTS**

1. **No CPU fallback** - If NPU/GPU fails, error out clearly
2. **Test everything** - Each change should be verifiable
3. **Performance first** - This is a high-performance inference engine
4. **Real hardware only** - No simulations or dummy data
5. **Check GPU usage** - Should see spikes during operations

---

*Last updated: July 15, 2025 00:45*