# 📋 Qwen3-30B-A3B MoE Implementation Plan for Unicorn Execution Engine

**Target Model**: Qwen3-30B-A3B (30B total, 3B active MoE)  
**Target Performance**: 40-50 TPS with INT4 quantization  
**Hardware**: AMD Radeon 780M (iGPU) + AMD Phoenix NPU  
**Engine**: Custom Unicorn Execution Engine with Vulkan compute

---

## 🎯 Overview

This plan will guide you through implementing the Qwen3-30B-A3B MoE model on the custom Unicorn Execution Engine. The model has 30B total parameters but only activates 3B at a time (8 out of 128 experts), making it perfect for the available 16GB VRAM.

---

## ✅ Phase 1: Model Acquisition & Analysis

### 1.1 Download Base Model
- [ ] Download Qwen3-30B-A3B from HuggingFace
  - URL: `https://huggingface.co/Qwen/Qwen3-30B-A3B`
  - Need the FP16/BF16 version for best quantization quality
  - Approximate size: ~60GB (FP16)

### 1.2 Analyze Model Architecture
- [ ] Identify MoE structure:
  - [ ] Router layer configuration
  - [ ] Number of experts (128)
  - [ ] Active experts per token (8)
  - [ ] Expert weight distribution
- [ ] Map layer names to Unicorn engine format
- [ ] Document special tokens for tool use

### 1.3 Prepare Workspace
```bash
cd /home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/
mkdir qwen3_30b_moe
cd qwen3_30b_moe
```

---

## ✅ Phase 2: Custom Quantization Implementation

### 2.1 Create Unicorn-Q4-MoE Quantizer
- [ ] Create `qwen3_moe_quantizer.py` with:
  ```python
  class UnicornQ4MoEQuantizer:
      def __init__(self):
          self.block_size = 32  # Q4_K_M style
          self.router_precision = "fp16"  # Keep router high precision
          self.expert_precision = "int4"
          self.inactive_expert_precision = "int4"  # Or int3 for more compression
  ```

### 2.2 Implement Quantization Strategy
- [ ] **Router weights**: Keep at FP16 (critical for expert selection)
- [ ] **Active expert weights**: Quantize to INT4 with K-means clustering
- [ ] **Inactive expert weights**: Aggressive INT4 or INT3
- [ ] **Embeddings**: INT8 (good balance)
- [ ] **LM head**: INT8 or FP16 (quality critical)

### 2.3 Memory Layout Optimization
- [ ] Design memory hierarchy:
  ```
  NPU SRAM (2GB): Router weights + routing logic
  GPU VRAM (16GB): Active experts + frequently used layers
  GPU GTT (48GB): Inactive experts
  System RAM: Overflow only
  ```

---

## ✅ Phase 3: Adapt Existing Engine Components

### 3.1 Modify Model Loader
- [ ] Update `gemma_27b_loader_v2.py` → `qwen3_moe_loader.py`
- [ ] Add MoE-specific weight loading:
  - [ ] Router weight handling
  - [ ] Expert weight indexing
  - [ ] Sparse loading (only load needed experts)

### 3.2 Update Compute Pipeline
- [ ] Extend `vulkan_compute_workaround.py` for MoE:
  ```python
  def compute_moe_routing(self, hidden_states, router_weights):
      # Run on NPU if available, else GPU
      # Returns: expert_indices, routing_weights
  
  def compute_sparse_experts(self, hidden_states, expert_indices, expert_weights):
      # Only compute for selected experts
      # Massive memory bandwidth savings!
  ```

### 3.3 Integrate Existing Shaders
- [ ] Use existing INT4 shaders:
  - `rdna3_int4.spv` - For expert computation
  - `matrix_multiply_int8.spv` - For embeddings
  - `transformer_optimized.spv` - For attention

---

## ✅ Phase 4: MoE-Specific Optimizations

### 4.1 Expert Caching Strategy
- [ ] Implement LRU cache for experts:
  ```python
  class ExpertCache:
      def __init__(self, max_size_gb=14):  # Leave 2GB for activations
          self.cache = OrderedDict()
          self.max_size = max_size_gb * 1024**3
  ```

### 4.2 Parallel Expert Execution
- [ ] Since only 3B active, can run multiple experts in parallel:
  - [ ] Split 16GB VRAM into expert slots
  - [ ] Process multiple tokens simultaneously
  - [ ] Hide memory transfer latency

### 4.3 NPU Router Integration
- [ ] If NPU available:
  - [ ] Load router to NPU permanently
  - [ ] Run routing in parallel with GPU computation
  - [ ] Save memory bandwidth

---

## ✅ Phase 5: Tool Use Implementation

### 5.1 Special Token Handling
- [ ] Add Qwen3 tool tokens:
  ```python
  TOOL_TOKENS = {
      "<function_call>": 32100,
      "</function_call>": 32101,
      "<tool_response>": 32102,
      "</tool_response>": 32103,
  }
  ```

### 5.2 Modify Generation Loop
- [ ] Add tool detection to `generate_tokens()`:
  ```python
  if self.detect_tool_request(output_tokens):
      tool_result = self.execute_tool(tool_request)
      tokens.extend(self.format_tool_response(tool_result))
  ```

### 5.3 Tool Registry
- [ ] Create `tools/` directory with common tools
- [ ] Implement tool execution framework
- [ ] Add tool result injection

---

## ✅ Phase 6: Testing & Benchmarking

### 6.1 Functionality Tests
- [ ] Test basic generation
- [ ] Test MoE routing correctness
- [ ] Test tool use
- [ ] Test thinking vs non-thinking modes

### 6.2 Performance Benchmarks
- [ ] Measure tokens/second
- [ ] Monitor memory usage (should be ~7.5GB active)
- [ ] Check NPU utilization for routing
- [ ] Verify 40-50 TPS target

### 6.3 Quality Validation
- [ ] Run standard benchmarks (MMLU, HumanEval, etc.)
- [ ] Compare with original model quality
- [ ] Ensure quantization doesn't degrade too much

---

## 📁 Key Files to Create/Modify

1. **New Files**:
   - `qwen3_moe_quantizer.py` - Custom quantization
   - `qwen3_moe_loader.py` - MoE-aware model loader
   - `qwen3_moe_pipeline.py` - Main inference pipeline
   - `moe_router_npu.py` - NPU routing optimization
   - `expert_cache.py` - Expert caching logic

2. **Modified Files**:
   - `vulkan_compute_workaround.py` - Add MoE ops
   - `real_vulkan_matrix_compute.py` - Add sparse compute

3. **Reused Files**:
   - All `.spv` shader files (already compiled)
   - `bfloat16_converter.py` - For weight conversion
   - NPU infrastructure files

---

## 🚀 Quick Start Commands

```bash
# 1. Download model
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./qwen3_30b_base

# 2. Run quantization
python qwen3_moe_quantizer.py --input ./qwen3_30b_base --output ./qwen3_30b_q4

# 3. Test inference
python qwen3_moe_pipeline.py --model ./qwen3_30b_q4 --prompt "Hello, how are you?"

# 4. Benchmark
python benchmark_moe.py --model ./qwen3_30b_q4 --num_tokens 1000
```

---

## ⚡ Expected Outcomes

- **Model Size**: ~7.5GB active in memory (vs 60GB original)
- **Performance**: 40-50 TPS (vs 17 TPS for dense models)
- **Quality**: Within 2-3% of original model
- **Tool Use**: Full function calling support
- **Memory**: 
  - VRAM: ~8GB (active experts + cache)
  - GTT: ~15GB (inactive experts)
  - System RAM: Minimal

---

## 🔧 Troubleshooting Tips

1. **If Vulkan fails**: Use the `vulkan_compute_workaround.py`
2. **If memory runs out**: Reduce expert cache size
3. **If TPS too low**: Check if routing is happening on NPU
4. **If quality degrades**: Keep critical layers at INT8 instead of INT4

---

## 📚 References

- Qwen3 Technical Report: `https://arxiv.org/abs/2505.09388`
- MoE Architecture: Focus on sparse activation patterns
- Existing Unicorn Engine docs: `CLAUDE.md`, `UNICORN_EXECUTION_ENGINE_ARCHITECTURE.md`

---

**Good luck! The Qwen3-30B-A3B MoE model is perfectly suited for the Unicorn Execution Engine and should achieve excellent performance!** 🦄