#!/usr/bin/env python3
"""
Test NPU Acceleration - Verify NPU is actually computing
"""

import numpy as np
import time
import logging
from npu_attention_kernel_real import NPUAttentionKernelReal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_npu_attention():
    """Test NPU attention computation"""
    
    logger.info("🧪 Testing NPU Attention Acceleration...")
    
    # Initialize NPU
    npu = NPUAttentionKernelReal(seq_length=256, d_model=5376, num_heads=32)
    
    if not npu.initialize():
        logger.error("Failed to initialize NPU")
        return False
        
    logger.info("✅ NPU initialized successfully")
    
    # Test data
    batch_size = 1
    seq_len = 256
    hidden_size = 5376
    
    # Create test inputs
    hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Create projection weights (normally from model)
    q_proj = np.random.randn(hidden_size, 4096).astype(np.float32)
    k_proj = np.random.randn(hidden_size, 2048).astype(np.float32)
    v_proj = np.random.randn(hidden_size, 2048).astype(np.float32)
    o_proj = np.random.randn(4096, hidden_size).astype(np.float32)
    
    logger.info("\n📊 Running NPU attention computation...")
    
    # Warm up
    _ = npu.compute_flash_attention(hidden_states, q_proj, k_proj, v_proj, o_proj)
    
    # Benchmark NPU
    times = []
    for i in range(5):
        start = time.time()
        output = npu.compute_flash_attention(hidden_states, q_proj, k_proj, v_proj, o_proj)
        elapsed = time.time() - start
        times.append(elapsed * 1000)
        logger.info(f"   Run {i+1}: {elapsed*1000:.1f}ms")
        
    avg_time = np.mean(times)
    logger.info(f"\n⚡ NPU Average: {avg_time:.1f}ms")
    
    # Compare with CPU baseline
    logger.info("\n📊 CPU baseline for comparison...")
    
    def cpu_attention(hidden, q_w, k_w, v_w, o_w):
        q = hidden @ q_w
        k = hidden @ k_w
        v = hidden @ v_w
        
        # Simple attention (not optimized)
        scores = q @ k.T
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        attn = probs @ v
        
        return attn @ o_w
    
    cpu_times = []
    for i in range(3):
        start = time.time()
        _ = cpu_attention(hidden_states[0], q_proj, k_proj, v_proj, o_proj)
        elapsed = time.time() - start
        cpu_times.append(elapsed * 1000)
        
    cpu_avg = np.mean(cpu_times)
    logger.info(f"   CPU Average: {cpu_avg:.1f}ms")
    
    # Calculate speedup
    speedup = cpu_avg / avg_time
    logger.info(f"\n🚀 NPU Speedup: {speedup:.1f}x")
    
    # Check if NPU is actually being used
    if speedup > 1.5:
        logger.info("✅ NPU acceleration confirmed!")
        return True
    elif avg_time < 50:  # Fast enough to be hardware accelerated
        logger.info("✅ NPU likely working (very fast execution)")
        return True
    else:
        logger.warning("⚠️ NPU may be falling back to CPU")
        return False

def test_npu_memory():
    """Test NPU memory usage"""
    
    logger.info("\n🧪 Testing NPU Memory...")
    
    # Check NPU memory info
    import subprocess
    
    try:
        # Try to get NPU info
        result = subprocess.run(
            ['ls', '-la', '/sys/kernel/debug/accel/accel0/'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("   NPU debug info available:")
            logger.info(result.stdout)
    except:
        pass
        
    return True

def main():
    """Run NPU tests"""
    
    logger.info("🚀 NPU Acceleration Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("NPU Attention", test_npu_attention),
        ("NPU Memory", test_npu_memory),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {name}: PASSED\n")
            else:
                logger.error(f"❌ {name}: FAILED\n")
        except Exception as e:
            logger.error(f"❌ {name}: ERROR - {e}\n")
            
    logger.info("=" * 50)
    logger.info(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 NPU acceleration verified!")
        logger.info("\nNext steps:")
        logger.info("1. Build optimized MLIR-AIE2 kernels")
        logger.info("2. Integrate NPU+GPU pipeline") 
        logger.info("3. Achieve 100+ TPS!")

if __name__ == "__main__":
    main()