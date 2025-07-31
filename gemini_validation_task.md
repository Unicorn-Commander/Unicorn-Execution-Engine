# Task for Gemini: Validate and Benchmark INT4 WMMA Implementation

## Current Status
You've successfully created the HIP WMMA kernel implementation. Now we need to validate it works and measure performance.

## Primary Task: Test and Benchmark INT4 WMMA

### Step 1: Build and Test the HIP Kernel
```bash
# Set up environment
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=11.0.3

# Build the kernel
cd /home/ucadmin/Development/Unicorn-Execution-Engine
hipcc -O3 -arch=gfx1103 -o test_hip_int4 test_hip_int4_wmma.cpp -I${ROCM_PATH}/include/rocwmma -L${ROCM_PATH}/lib -lrocwmma
```

### Step 2: Create Test Script
Create `test_hip_int4_performance.py`:
```python
#!/usr/bin/env python3.13
"""Test HIP INT4 WMMA performance"""

import time
import torch
import numpy as np
from magic_unicorn_ultra_speed import MagicUnicornUltraSpeed

def test_int4_performance():
    print("🦄 TESTING INT4 WMMA PERFORMANCE")
    print("=" * 60)
    
    # Initialize engine
    engine = MagicUnicornUltraSpeed()
    
    # Test configurations
    test_configs = [
        (1, 32, "Small context"),
        (1, 128, "Medium context"),
        (1, 256, "Large context"),
    ]
    
    for batch_size, seq_len, desc in test_configs:
        print(f"\n🧪 Testing {desc} (batch={batch_size}, seq_len={seq_len})")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, 2560, dtype=torch.float32)
        
        # Warmup
        for _ in range(2):
            _ = engine.transformer_layer_ultra(x, layer_idx=0)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            output = engine.transformer_layer_ultra(x, layer_idx=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        
        # Calculate tokens/sec
        tokens_per_sec = 1.0 / (min_time * 42)  # 42 layers
        
        print(f"   Average layer time: {avg_time*1000:.1f}ms")
        print(f"   Fastest layer time: {min_time*1000:.1f}ms")
        print(f"   Projected speed: {tokens_per_sec:.3f} tokens/sec")
        print(f"   vs 21 tok/s target: {tokens_per_sec/21:.3f}x")
        
        if tokens_per_sec >= 21.0:
            print(f"   🎯 TARGET ACHIEVED WITH INT4 WMMA!")

if __name__ == "__main__":
    test_int4_performance()
```

### Step 3: Verify INT4 Quantization
Create a simple test to verify INT4 packing/unpacking is working correctly:
```python
def test_int4_packing():
    # Test data
    test_tensor = torch.randint(-8, 7, (16, 16), dtype=torch.int8)
    
    # Pack to INT4
    packed = pack_to_int4(test_tensor)
    
    # Verify packing
    print(f"Original shape: {test_tensor.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Packed dtype: {packed.dtype}")
    
    # Test unpacking in kernel
    unpacked = unpack_from_int4(packed)
    
    # Verify correctness
    assert torch.allclose(test_tensor, unpacked), "INT4 packing/unpacking failed!"
    print("✅ INT4 packing/unpacking verified!")
```

### Step 4: Compare Performance
Run benchmarks comparing:
1. Original FP32 OpenCL implementation
2. New INT4 WMMA HIP implementation

Expected results:
- INT4 WMMA should be 7-8x faster than FP32
- Should achieve ~21 tokens/sec for single token generation

### Step 5: Debug if Needed
If performance is not as expected:
1. Check if WMMA instructions are actually being used (use `rocprof`)
2. Verify 16x16 tiling is correct
3. Ensure INT4 packing is efficient
4. Check memory access patterns

## Secondary Task: Document Results

Create `INT4_WMMA_RESULTS.md` with:
1. Build process and any issues encountered
2. Performance measurements
3. Comparison with FP32 baseline
4. Analysis of bottlenecks (if any)
5. Next optimization steps

## Expected Outcomes

With proper INT4 WMMA implementation:
- **Layer time**: ~3ms (down from 125ms)
- **Full model**: ~126ms (42 layers)
- **Speed**: ~7.9 tokens/sec minimum
- **With optimization**: 21+ tokens/sec achievable

## Questions to Answer

1. Does the HIP kernel compile successfully with hipcc?
2. Are WMMA instructions being generated in the assembly?
3. What is the actual measured performance vs theoretical?
4. Are there any memory bandwidth bottlenecks?
5. Does INT4 quantization maintain acceptable model quality?

Let me know the results of your testing!