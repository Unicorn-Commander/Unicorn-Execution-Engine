#!/usr/bin/env python3
"""
Test GPU utilization to verify Vulkan compute is actually running on GPU
"""

import numpy as np
import logging
import time
import subprocess
from real_vulkan_matrix_compute import VulkanMatrixCompute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_gpu():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                              capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'gpu' in line:
                # Extract GPU percentage
                import re
                match = re.search(r'gpu\s+(\d+\.\d+)%', line)
                if match:
                    return float(match.group(1))
    except:
        pass
    return 0.0

def test_gpu_compute():
    """Test if Vulkan compute actually uses GPU"""
    logger.info("🔍 Testing GPU compute utilization...")
    
    # Initialize Vulkan
    vulkan = VulkanMatrixCompute()
    
    # Create test matrices
    size = 4096
    logger.info(f"Creating {size}x{size} test matrices...")
    matrix_a = np.random.randn(size, size).astype(np.float32)
    matrix_b = np.random.randn(size, size).astype(np.float32)
    
    # Allocate to GPU
    logger.info("Allocating matrices to GPU...")
    gpu_buffer_b = vulkan._allocate_gpu_memory(matrix_b)
    
    logger.info("🚀 Running matrix multiplication on GPU...")
    
    # Monitor GPU before
    gpu_before = monitor_gpu()
    logger.info(f"GPU utilization before: {gpu_before}%")
    
    # Run multiple iterations to catch GPU usage
    start_time = time.time()
    for i in range(10):
        result = vulkan.compute_matrix_multiply(matrix_a, matrix_b)
        
        # Check GPU during computation
        gpu_during = monitor_gpu()
        if gpu_during > gpu_before:
            logger.info(f"✅ GPU spike detected: {gpu_during}% (iteration {i+1})")
    
    end_time = time.time()
    
    # Monitor GPU after
    gpu_after = monitor_gpu()
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Computation time: {end_time - start_time:.2f}s")
    logger.info(f"   GPU before: {gpu_before}%")
    logger.info(f"   GPU peak: {max(gpu_before, gpu_during, gpu_after)}%")
    logger.info(f"   Matrix size: {size}x{size}")
    logger.info(f"   GFLOPS: {(2 * size**3 * 10) / (end_time - start_time) / 1e9:.1f}")
    
    if max(gpu_before, gpu_during, gpu_after) > gpu_before + 5:
        logger.info("✅ GPU compute verified - GPU utilization increased!")
    else:
        logger.warning("⚠️ GPU compute may not be working - no utilization increase detected")
        logger.info("   This could mean:")
        logger.info("   - Computation is falling back to CPU")
        logger.info("   - GPU is too fast to catch with monitoring")
        logger.info("   - Vulkan dispatch is not executing")

if __name__ == "__main__":
    test_gpu_compute()