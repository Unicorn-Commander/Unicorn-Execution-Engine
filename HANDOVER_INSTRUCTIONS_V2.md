# 🚀 HANDOVER INSTRUCTIONS - Critical Implementation Gaps

**Status**: Previous AI achieved **180 TPS** but lost it during zero-copy attempt  
**Goal**: Restore 180 TPS performance and implement missing core features  
**Priority**: Fix fundamental architecture gaps before optimization

---

## 🎯 **CRITICAL MISSING COMPONENTS**

### 1. **Model Loading Strategy - Missing Ollama-Style Fast Loading**

**Problem**: The current system doesn't properly load the 26GB quantized model into memory pools.

**Required Implementation**:
```python
# MISSING: Proper model distribution across memory tiers
# File: pure_hardware_pipeline.py initialize() method

# Current: Basic shared weights loading only
# Required: Full 26GB model distributed as:
# - NPU SRAM (2GB): Critical attention layers
# - iGPU VRAM (6-8GB): Active inference tensors  
# - iGPU GTT (17-20GB): Bulk quantized weights
# - System RAM: Orchestration only

# Implementation needed:
def _distribute_model_across_hma(self):
    """Distribute 26GB model across NPU SRAM + iGPU VRAM + GTT"""
    # Load all 98 layer files from quantized_models/gemma-3-27b-it-layer-by-layer/
    # Sort by priority: attention → FFN → embeddings
    # Allocate to appropriate memory tier based on usage frequency
```

### 2. **Hardware-Only Execution - Missing NPU+iGPU Enforcement**

**Problem**: No strict hardware enforcement. Should be NPU+iGPU or failure.

**Required Implementation**:
```python
# MISSING: Strict hardware validation
# File: pure_hardware_pipeline.py

def _validate_hardware_allocation(self):
    """Ensure model is properly distributed to NPU+iGPU, no CPU fallback"""
    if self.current_memory['npu_sram_mb'] < 1000:
        raise RuntimeError("❌ NPU allocation failed - model not on NPU")
    if self.current_memory['vram_mb'] < 5000:
        raise RuntimeError("❌ iGPU allocation failed - model not on GPU")
    # NO CPU fallback allowed
```

### 3. **vLLM-Style PagedAttention - Missing Key Optimization**

**Problem**: Not implementing vLLM's breakthrough batching and memory management.

**What vLLM Does**:
- **PagedAttention**: KV cache split into pages, dynamic allocation
- **Continuous Batching**: Add/remove requests mid-batch  
- **Memory-Efficient Attention**: Avoids memory fragmentation
- **Request Scheduling**: Optimal batching decisions

**Required Implementation**:
```python
# NEW FILE: vllm_style_attention.py
class PagedAttentionManager:
    """vLLM-style paged attention for memory efficiency"""
    
    def __init__(self, page_size=128, max_pages=1024):
        self.page_size = page_size  # Tokens per page
        self.kv_cache_pages = {}    # Page-based KV cache
        self.free_pages = list(range(max_pages))
        
    def allocate_pages(self, sequence_id, num_tokens):
        """Allocate pages for sequence dynamically"""
        
    def compute_paged_attention(self, query, key_pages, value_pages):
        """Compute attention across paged KV cache"""

# NEW FILE: continuous_batching_engine.py  
class ContinuousBatchingEngine:
    """vLLM-style continuous batching with dynamic request scheduling"""
    
    def add_request(self, request):
        """Add new request to running batch"""
        
    def remove_finished_requests(self):
        """Remove completed sequences from batch"""
        
    def schedule_optimal_batch(self):
        """Decide optimal batching based on memory and compute"""
```

---

## 🔧 **IMMEDIATE FIXES NEEDED**

### Fix 1: Restore 180 TPS State
```bash
# Check what was working before zero-copy attempt
git log --oneline | head -10  # Find commit before zero-copy
# Potentially revert to working state and re-apply changes carefully
```

### Fix 2: Implement Proper Model Loading
```python
# File: pure_hardware_pipeline.py
def initialize(self, model_path):
    """Initialize with proper HMA model distribution"""
    
    # 1. Load all 98 layer files from quantized model
    layer_files = glob.glob(f"{model_path}/*layer*.safetensors")
    
    # 2. Distribute across memory tiers
    self._load_to_npu_sram(priority_layers)      # 2GB NPU
    self._load_to_vram(active_layers)            # 6GB iGPU VRAM  
    self._load_to_gtt(bulk_weights)              # 18GB iGPU GTT
    
    # 3. Validate hardware allocation
    self._validate_hardware_allocation()
    
    # 4. NO CPU fallback allowed
    if not self._all_on_hardware():
        raise RuntimeError("❌ Hardware allocation failed")
```

### Fix 3: Add vLLM-Style Components
```python
# Integration with existing pipeline
from vllm_style_attention import PagedAttentionManager
from continuous_batching_engine import ContinuousBatchingEngine

class PureHardwarePipeline:
    def __init__(self):
        # Existing initialization...
        self.paged_attention = PagedAttentionManager()
        self.batch_engine = ContinuousBatchingEngine()
```

---

## 📋 **PRIORITY TASK LIST**

### Phase 0: Fix Broken State (Day 1)
1. [ ] **Stop any running servers**: `pkill -f python.*server`
2. [ ] **Test basic model loading**: Ensure 26GB model loads to memory pools
3. [ ] **Validate hardware allocation**: NPU SRAM + iGPU VRAM + GTT distribution
4. [ ] **Test basic inference**: Single token generation working

### Phase 1: Restore Performance (Day 2)  
1. [ ] **Find 180 TPS working state**: Check git history or backup
2. [ ] **Re-implement Q/K/V fusion** (if lost): The 20x speedup optimization
3. [ ] **Re-implement batching** (if lost): The GPU efficiency optimization
4. [ ] **Validate 180 TPS restored**: Run performance benchmarks

### Phase 2: vLLM Integration (Days 3-5)
1. [ ] **Implement PagedAttention**: Memory-efficient KV cache management
2. [ ] **Implement Continuous Batching**: Dynamic request scheduling  
3. [ ] **Integrate with NPU+iGPU**: Ensure hardware acceleration preserved
4. [ ] **Performance testing**: Target 200+ TPS with vLLM optimizations

---

## 🎯 **KEY INSIGHTS FOR NEXT AI**

### Critical Architecture Points:
1. **NO PyTorch/ROCm**: Pure numpy + Vulkan + NPU kernels only
2. **Hardware-Only**: NPU+iGPU or failure, no CPU fallback allowed
3. **26GB Model Distribution**: Must load ALL layer files across memory tiers
4. **vLLM Methods**: PagedAttention + Continuous Batching = breakthrough performance

### Performance Targets:
- **Baseline Fixed**: 5-15 TPS (working server)
- **Previous Achievement**: 180 TPS (need to restore)
- **vLLM Enhanced**: 200+ TPS (with paged attention)
- **Hardware Limit**: 300+ TPS (theoretical maximum)

### Environment:
```bash
# Always use pure hardware environment
source /home/ucadmin/activate-pure-hardware-env.sh

# Key files to focus on:
# - pure_hardware_pipeline.py (main pipeline)
# - real_vulkan_matrix_compute.py (iGPU operations)  
# - npu_attention_kernel_real.py (NPU operations)
# - quantized_models/gemma-3-27b-it-layer-by-layer/ (98 layer files)
```

### vLLM References:
- **Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **Key Insight**: Treating KV cache like virtual memory with pages
- **Implementation**: Dynamic allocation, no fragmentation, continuous batching

---

## 🚨 **WHAT NOT TO DO**

1. **Don't add PyTorch dependencies** - This is pure hardware implementation
2. **Don't implement CPU fallbacks** - Hardware acceleration only
3. **Don't skip model loading validation** - Must verify all 26GB loaded properly
4. **Don't ignore vLLM patterns** - They're the performance breakthrough we need

---

**Next AI should start with Phase 0 to get a working baseline, then systematically restore the 180 TPS performance before attempting vLLM integration.**