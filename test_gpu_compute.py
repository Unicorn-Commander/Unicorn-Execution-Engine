#!/usr/bin/env python3
"""
Test GPU Compute - Verify Vulkan is actually using the GPU
"""

import numpy as np
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_gpu():
    """Get current GPU usage"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=1)
        if result.stdout:
            import re
            match = re.search(r'gpu\s+(\d+\.\d+)%', result.stdout)
            if match:
                return float(match.group(1))
    except:
        pass
    return 0.0

def test_vulkan_gpu():
    """Test if Vulkan is actually using GPU"""
    logger.info("🧪 TESTING VULKAN GPU COMPUTE")
    
    # Import and initialize Vulkan
    from real_vulkan_matrix_compute import VulkanMatrixCompute
    
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        logger.error("❌ Failed to initialize Vulkan")
        return
    
    logger.info("✅ Vulkan initialized")
    
    # Test sizes
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        logger.info(f"\n📊 Testing {size}x{size} matrix multiplication...")
        
        # Create test matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Monitor GPU before
        gpu_before = monitor_gpu()
        logger.info(f"   GPU usage before: {gpu_before:.1f}%")
        
        # Run on GPU
        logger.info("   Running on GPU...")
        gpu_start = time.time()
        
        # Multiple iterations to see sustained GPU usage
        for i in range(5):
            gpu_result = vulkan.matrix_multiply(A, B)
            gpu_during = monitor_gpu()
            if i == 0:
                logger.info(f"   GPU usage during: {gpu_during:.1f}%")
        
        gpu_time = time.time() - gpu_start
        
        # Run on CPU for comparison
        logger.info("   Running on CPU...")
        cpu_start = time.time()
        cpu_result = np.matmul(A, B)
        cpu_time = time.time() - cpu_start
        
        # Compare results
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        error = np.mean(np.abs(gpu_result - cpu_result))
        
        logger.info(f"   ✅ Results:")
        logger.info(f"      GPU time: {gpu_time:.3f}s")
        logger.info(f"      CPU time: {cpu_time:.3f}s")
        logger.info(f"      Speedup: {speedup:.2f}x")
        logger.info(f"      Error: {error:.6f}")
        logger.info(f"      GPU peaked at: {gpu_during:.1f}%")
    
    # Test batch operations
    logger.info("\n🔥 Testing batch operations (Q/K/V fusion simulation)...")
    
    batch_size = 8
    seq_len = 256
    hidden_dim = 768
    
    # Simulate transformer attention
    Q = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    V = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    
    logger.info(f"   Batch attention: {batch_size}x{seq_len}x{hidden_dim}")
    
    # Monitor during batch
    gpu_before = monitor_gpu()
    logger.info(f"   GPU before batch: {gpu_before:.1f}%")
    
    batch_start = time.time()
    
    # Simulate attention computation
    for b in range(batch_size):
        # Q @ K^T
        scores = vulkan.matrix_multiply(Q[b], K[b].T)
        gpu_during = monitor_gpu()
        
        # Softmax would go here
        scores_soft = scores / np.sqrt(hidden_dim)
        
        # scores @ V
        output = vulkan.matrix_multiply(scores_soft, V[b])
    
    batch_time = time.time() - batch_start
    
    logger.info(f"   ✅ Batch completed in {batch_time:.3f}s")
    logger.info(f"   GPU peaked at: {gpu_during:.1f}%")
    
    # Final GPU check
    time.sleep(1)
    gpu_after = monitor_gpu()
    logger.info(f"\n📊 Final GPU usage: {gpu_after:.1f}%")
    
    vulkan.cleanup()
    logger.info("\n🎉 GPU compute test completed!")

if __name__ == "__main__":
    test_vulkan_gpu()