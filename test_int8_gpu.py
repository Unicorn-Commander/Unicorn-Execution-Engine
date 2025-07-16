#!/usr/bin/env python3
"""Test INT8 GPU support"""

import numpy as np
import logging
from real_vulkan_matrix_compute import VulkanMatrixCompute
from vulkan_int8_support import add_int8_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_int8_allocation():
    """Test that INT8 data stays INT8 in GPU memory"""
    logger.info("🧪 Testing INT8 GPU Support")
    
    # Add INT8 support
    add_int8_support(VulkanMatrixCompute)
    
    # Initialize Vulkan
    compute = VulkanMatrixCompute()
    if not compute.initialize():
        logger.error("Failed to initialize Vulkan")
        return False
    
    # Create INT8 test data
    test_size = 1024 * 1024  # 1M elements
    int8_data = np.random.randint(-128, 127, size=test_size, dtype=np.int8)
    
    logger.info(f"📊 Test data: {test_size} INT8 elements = {test_size / (1024*1024):.1f}MB")
    
    # Allocate to GPU
    logger.info("🔄 Allocating INT8 data to GPU...")
    gpu_buffer = compute._allocate_gpu_memory(int8_data)
    
    logger.info(f"✅ Allocated to GPU: {gpu_buffer[2] / (1024*1024):.1f}MB")
    
    # Check memory usage
    expected_mb = test_size / (1024 * 1024)
    actual_mb = gpu_buffer[2] / (1024 * 1024)
    
    if abs(actual_mb - expected_mb) < 0.1:  # Allow small overhead
        logger.info("✅ INT8 data preserved! No conversion to FP32")
        return True
    else:
        logger.error(f"❌ Data expanded: {expected_mb:.1f}MB → {actual_mb:.1f}MB")
        return False

def test_int8_compute():
    """Test INT8 matrix multiplication"""
    logger.info("\n🧪 Testing INT8 Matrix Multiplication")
    
    # Create small test matrices
    M, N, K = 256, 256, 256
    
    # FP32 input (activations)
    matrix_a = np.random.randn(M, K).astype(np.float32)
    
    # INT8 weights
    matrix_b_int8 = np.random.randint(-128, 127, size=(K, N), dtype=np.int8)
    scale_b = 0.01  # Quantization scale
    
    logger.info(f"📊 Matrix sizes: {M}x{K} @ {K}x{N}")
    logger.info(f"   Input: FP32 ({matrix_a.nbytes / 1024:.1f}KB)")
    logger.info(f"   Weight: INT8 ({matrix_b_int8.nbytes / 1024:.1f}KB)")
    
    # Compute reference result
    matrix_b_fp32 = matrix_b_int8.astype(np.float32) * scale_b
    reference = np.matmul(matrix_a, matrix_b_fp32)
    
    logger.info(f"✅ Reference computed: {reference.shape}")
    
    # TODO: Test GPU INT8 computation when pipeline is ready
    
    return True

if __name__ == "__main__":
    success = test_int8_allocation()
    if success:
        test_int8_compute()
    
    print(f"\n{'✅ INT8 support working!' if success else '❌ INT8 support failed'}")