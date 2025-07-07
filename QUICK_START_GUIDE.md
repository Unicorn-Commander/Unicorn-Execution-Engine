# Gemma 3n E2B Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Prerequisites Check
```bash
# Verify NPU is detected
xrt-smi examine

# Should see: NPU Phoenix detected
# If not, check BIOS IPU setting and run: sudo modprobe amdxdna
```

### 1. Environment Setup (30 seconds)
```bash
# Activate the Gemma 3n environment
source gemma3n_env/bin/activate

# Verify setup
python run_gemma3n_e2b.py --dry-run --prompt "test"
```

### 2. First Text Generation (2 minutes)
```bash
# Generate text with hybrid NPU+iGPU execution
python run_gemma3n_e2b.py --prompt "The future of artificial intelligence" --max-tokens 100

# Expected output:
# ✅ NPU Phoenix detected and available
# 🎯 Performance: ~80 TPS, ~30ms TTFT
```

### 3. Performance Validation (2 minutes)
```bash
# Run comprehensive performance tests
python validate_performance.py

# Check if targets are met:
# Target: 40-80 TPS ✅
# Target: 20-40ms TTFT (optimizing)
```

## 🎯 Performance Expectations

| Scenario | Expected TPS | Expected TTFT | Hardware Used |
|----------|-------------|---------------|---------------|
| Short prompt (10-50 tokens) | 85-95 TPS | 10-30ms | NPU+iGPU |
| Medium prompt (50-200 tokens) | 75-85 TPS | 30-100ms | NPU+iGPU |
| Long prompt (200+ tokens) | 70-80 TPS | 100-300ms | NPU+iGPU |

## ⚙️ Quick Configuration

### Adjust Memory Budgets
```bash
# More NPU memory (better for attention)
python run_gemma3n_e2b.py --npu-memory 3072 --prompt "Your prompt"

# More iGPU memory (better for long generation)
python run_gemma3n_e2b.py --igpu-memory 12288 --prompt "Your prompt"
```

### Adjust Generation Settings
```bash
# More creative (higher temperature)
python run_gemma3n_e2b.py --temperature 0.9 --prompt "Creative story:"

# More focused (lower temperature)
python run_gemma3n_e2b.py --temperature 0.3 --prompt "Technical explanation:"

# Longer generation
python run_gemma3n_e2b.py --max-tokens 500 --prompt "Long essay:"
```

## 🔧 Troubleshooting

### NPU Issues
```bash
# NPU not detected
sudo modprobe amdxdna
xrt-smi examine

# Enable turbo mode
sudo xrt-smi configure --pmode turbo
```

### Performance Issues
```bash
# Check thermal throttling
sensors

# Monitor hardware utilization
python run_gemma3n_e2b.py --verbose --prompt "test"

# Run optimization analysis
python performance_optimizer.py
```

### Memory Issues
```bash
# Reduce memory usage
python run_gemma3n_e2b.py --npu-memory 1024 --igpu-memory 4096 --prompt "test"

# Force CPU execution (fallback)
python run_gemma3n_e2b.py --force-cpu --prompt "test"
```

## 📊 Benchmark Commands

### Quick Benchmark (30 seconds)
```bash
python run_gemma3n_e2b.py --benchmark --prompt "Quick benchmark test"
```

### Comprehensive Validation (2 minutes)
```bash
python validate_performance.py
```

### Custom Performance Test
```bash
# Test specific scenario
python run_gemma3n_e2b.py --prompt "Your test prompt here" --max-tokens 200 --verbose
```

## 💡 Pro Tips

### Optimal Settings for Different Use Cases

**Chat/Conversation:**
```bash
python run_gemma3n_e2b.py --temperature 0.7 --top-p 0.9 --max-tokens 150 --prompt "Human: Hello! Assistant:"
```

**Code Generation:**
```bash
python run_gemma3n_e2b.py --temperature 0.3 --top-k 20 --max-tokens 300 --prompt "# Python function to"
```

**Creative Writing:**
```bash
python run_gemma3n_e2b.py --temperature 0.8 --top-p 0.95 --max-tokens 400 --prompt "Once upon a time"
```

### Performance Optimization
1. **Enable turbo mode**: `sudo xrt-smi configure --pmode turbo`
2. **Close other apps**: Free up system resources
3. **Check thermals**: `sensors` to ensure no throttling
4. **Adjust memory**: Balance NPU/iGPU based on workload

### Getting Help
- **Verbose mode**: Add `--verbose` to any command for detailed logging
- **Documentation**: See `IMPLEMENTATION_SUMMARY.md` for technical details
- **NPU toolkit**: Check `NPU-Development/README.md` for NPU-specific help

## 🎉 Success Indicators

### System Working Correctly
- ✅ `xrt-smi examine` shows NPU Phoenix
- ✅ `python run_gemma3n_e2b.py --dry-run --prompt "test"` completes
- ✅ TPS > 40, reasonable TTFT for prompt length
- ✅ No memory errors or thermal throttling

### Performance Targets Met
- ✅ **TPS**: 40-80 range consistently achieved
- ✅ **TTFT**: Under 40ms for short prompts
- ✅ **Memory**: Under 10GB total usage
- ✅ **Hardware**: NPU and iGPU both utilized

---

**You're ready to run Gemma 3n E2B at optimal performance on AMD NPU+iGPU hybrid architecture! 🚀**