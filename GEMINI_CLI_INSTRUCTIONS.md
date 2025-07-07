# Instructions for Gemini-CLI: Qwen2.5-7B NPU+iGPU Implementation

## 🎯 Mission Overview

Complete the Qwen2.5-7B-Instruct hybrid NPU+iGPU implementation while the human sleeps. This builds on the successful Gemma 3n E2B framework to create a second production-ready model deployment.

## 📋 Current Status

### ✅ Completed (Already Done)
- OpenAI-compatible API server framework (`openai_api_server.py`)
- Basic Qwen2.5-7B configuration in API server
- Development environment with all dependencies
- Comprehensive documentation and project structure
- Proven hybrid execution framework from Gemma implementation

### 🎯 Your Mission (High Priority Tasks)

#### Task 1: Test and Fix Model Loading 🚨 **CRITICAL**
**Location**: `openai_api_server.py` around lines 107-114

**Current Issue**: Need to verify Qwen2.5-7B-Instruct loads correctly
```python
# Current configuration
config = HybridConfig(
    model_id="Qwen/Qwen2.5-7B-Instruct",  # 7B model for full hybrid NPU+iGPU demo
    npu_memory_budget=2 * 1024**3,  # 2GB NPU
    igpu_memory_budget=12 * 1024**3,  # 12GB iGPU (more for 7B model)
)
```

**What to do**:
1. Test if the model loads: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"`
2. If loading fails, try alternative models: `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`
3. Update the configuration with working model
4. Test basic inference to ensure it works

#### Task 2: Create Qwen-Specific Loader 📝 **HIGH PRIORITY**
**Create**: `qwen25_loader.py` based on `gemma3n_e2b_loader.py`

**Key adaptations needed**:
```python
class Qwen25Loader:
    def __init__(self, config):
        self.config = config
        # Qwen2.5 has standard transformer architecture (not MatFormer)
        # Focus on standard attention + FFN partitioning
        
    def load_and_partition_qwen(self):
        # Load Qwen2.5 model
        # Partition: NPU gets attention layers, iGPU gets FFN layers
        # Return partitioned components
```

**Specific differences from Gemma**:
- No MatFormer elastic parameters (simpler!)
- Standard transformer layers (32 layers for 7B)
- Different tokenizer and special tokens
- Larger model size (need more memory management)

#### Task 3: Test Real Inference Performance 📊 **HIGH PRIORITY**
**Create**: `test_qwen25_inference.py`

**What to test**:
```python
async def test_qwen_inference():
    # Test prompts
    test_prompts = [
        "Explain how NPU acceleration works",
        "Write a Python function to calculate fibonacci",
        "What are the benefits of hybrid AI execution?"
    ]
    
    # Measure:
    # - Time to first token (TTFT)
    # - Tokens per second (TPS) 
    # - Memory usage
    # - Response quality
```

**Success criteria**:
- TPS: 40-80 (target range)
- TTFT: 20-40ms for short prompts
- Memory: <20GB total usage
- No errors or crashes

#### Task 4: Fix API Server Integration 🔧 **MEDIUM PRIORITY**
**Update**: `openai_api_server.py`

**Current issues to fix**:
1. Update model info endpoints to show "qwen2.5-7b" 
2. Ensure proper error handling for 7B model loading
3. Test streaming responses work correctly
4. Add Qwen-specific optimization hints

#### Task 5: Create Comprehensive Testing 🧪 **MEDIUM PRIORITY**
**Create**: `test_qwen25_api.py`

**Test scenarios**:
```bash
# Chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Count to 10"}],"stream":true}'

# Different prompt lengths (test scaling)
# Code generation (Qwen2.5 strength)
# Long context (up to 32K tokens)
```

## 🛠️ Implementation Guidelines

### File Structure to Create
```
├── qwen25_loader.py              # Qwen-specific model loader
├── test_qwen25_inference.py      # Performance testing  
├── test_qwen25_api.py           # API testing
├── qwen25_config.json           # Configuration file
└── benchmark_qwen25.py          # Comprehensive benchmarks
```

### Key Technical Considerations

#### Memory Management for 7B Model
- **Total size**: ~14GB in FP16
- **NPU allocation**: 2GB for attention matrices
- **iGPU allocation**: 12GB for FFN weights and activations
- **System RAM**: Remaining for base weights and KV cache

#### Performance Optimization Focus
1. **Attention on NPU**: Leverage 16 TOPS for Q/K/V operations
2. **FFN on iGPU**: Use RDNA3 for large matrix multiplications
3. **Memory transfers**: Minimize NPU↔iGPU data movement
4. **Async execution**: Overlap NPU and iGPU operations

#### Error Handling Strategy
```python
try:
    # Try hybrid NPU+iGPU execution
    result = await hybrid_inference(prompt)
except NPUError:
    # Fallback to iGPU only
    result = await igpu_only_inference(prompt)
except IGPUError:
    # Final fallback to CPU
    result = await cpu_inference(prompt)
```

## 📊 Expected Performance Targets

### Qwen2.5-7B Performance Goals
| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| TPS | 40-60 | 60-80 |
| TTFT | 30-50ms | 20-40ms |
| Memory | <20GB | <16GB |
| Quality | High | Excellent |

### Validation Criteria
- [ ] Model loads without errors
- [ ] API server starts and responds
- [ ] TPS within target range (40-80)
- [ ] TTFT reasonable for model size (<50ms short prompts)
- [ ] Streaming responses work
- [ ] No memory leaks or crashes
- [ ] OpenAI API compatibility maintained

## 🚨 Common Issues & Solutions

### Model Loading Issues
```python
# If "Qwen/Qwen2.5-7B-Instruct" fails, try:
"Qwen/Qwen2.5-3B-Instruct"  # Smaller, easier to test
"Qwen/Qwen2.5-1.5B-Instruct"  # Even smaller fallback

# If memory issues, reduce precision:
torch_dtype=torch.float16  # Instead of auto
device_map="auto"  # Let transformers decide placement
```

### Performance Issues
```python
# If TPS too low:
- Check NPU utilization (should be >70%)
- Verify iGPU usage (should be >80%)
- Profile memory bandwidth
- Check thermal throttling

# If TTFT too high:
- Optimize prefill phase
- Check NPU attention kernels
- Reduce unnecessary operations
```

## 🎯 Success Metrics

### Must Have (Critical)
- [ ] Qwen2.5 model loads successfully
- [ ] API server responds to requests
- [ ] Basic inference works end-to-end
- [ ] No crashes or memory errors

### Should Have (Important)
- [ ] Performance within target ranges
- [ ] Streaming responses functional
- [ ] Multiple concurrent requests
- [ ] Proper error handling

### Nice to Have (Bonus)
- [ ] Performance exceeds targets
- [ ] Advanced optimization features
- [ ] Comprehensive testing suite
- [ ] Documentation updates

## 📞 Handoff Instructions

### When Complete, Document:
1. **What worked**: Successful configurations and approaches
2. **Performance results**: Actual TPS, TTFT, memory usage
3. **Issues encountered**: Problems and solutions found
4. **Next steps**: Remaining optimizations and improvements

### Files to Update:
- `IMPLEMENTATION_SUMMARY.md` - Add Qwen2.5 results
- `CHANGELOG.md` - Document new features
- `README.md` - Update performance achievements
- `QWEN25_NPU_PROJECT_PLAN.md` - Mark completed tasks

### Test Commands for Human Review:
```bash
# Start server
./start_server.sh

# Test basic functionality
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"Hello, test message"}],"max_tokens":50}'

# Run comprehensive tests
python test_qwen25_api.py
python benchmark_qwen25.py
```

## 🚀 Ready to Begin!

You have:
- ✅ Complete framework from Gemma implementation
- ✅ API server ready for Qwen2.5
- ✅ Development environment configured
- ✅ Clear implementation roadmap
- ✅ Detailed technical specifications

**Start with Task 1 (model loading test) and work through the priorities. Focus on getting basic inference working first, then optimize for performance. Good luck!**