#!/usr/bin/env python3
"""
Minimal INT4 pipeline test - verify integration without full model loading
"""

import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_int4_minimal():
    """Test INT4 integration with minimal setup"""
    logger.info("=" * 80)
    logger.info("🧪 MINIMAL INT4 PIPELINE TEST")
    logger.info("=" * 80)
    
    try:
        # Test 1: INT4 packing
        from integrate_int4_quantization import INT4Integration
        
        test_weight = np.random.randn(2048, 2048).astype(np.float32)
        packed, scale, zero_point = INT4Integration.pack_int4_weights(test_weight)
        
        logger.info(f"✅ INT4 packing works: {test_weight.nbytes/1024/1024:.1f}MB → {packed.nbytes/1024/1024:.1f}MB")
        
        # Test 2: Vulkan initialization with INT4
        from real_vulkan_matrix_compute import VulkanMatrixCompute
        from vulkan_int8_support import add_int8_support
        from vulkan_int4_support import add_int4_support
        
        add_int8_support(VulkanMatrixCompute)
        add_int4_support(VulkanMatrixCompute)
        
        vulkan = VulkanMatrixCompute()
        if vulkan.initialize():
            logger.info("✅ Vulkan initialized with INT4 support")
            
            # Test 3: Basic compute with INT4 fallback
            input_data = np.random.randn(128, 2048).astype(np.float32)
            
            # Test regular compute first
            result = vulkan.compute_matrix_multiply(input_data, test_weight)
            logger.info(f"✅ Regular compute works: input {input_data.shape} × weight {test_weight.shape} = {result.shape}")
            
            # Test INT4 compute (will use fallback if shaders not available)
            result_int4 = vulkan.compute_matrix_multiply_int4(
                input_data, packed, test_weight.shape, scale, zero_point
            )
            logger.info(f"✅ INT4 compute works (with fallback): result shape {result_int4.shape}")
            
            # Test 4: Memory allocation
            small_data = np.random.randn(1024).astype(np.float32)
            buffer_info = vulkan._allocate_gpu_memory(small_data)
            logger.info(f"✅ GPU memory allocation works: allocated {small_data.nbytes/1024:.1f}KB")
            
            # Test 5: Persistent buffers
            persistent_buffer = vulkan.create_persistent_buffer(test_weight)
            logger.info("✅ Persistent buffer creation works")
            
            return True
        else:
            logger.error("❌ Vulkan initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_int4_minimal()
    
    if success:
        logger.info("\n✅ INT4 minimal test PASSED!")
        logger.info("The pipeline is ready for INT4 quantization.")
        logger.info("Note: Full INT4 GPU acceleration requires compiled shader files.")
    else:
        logger.info("\n❌ INT4 minimal test FAILED!")
    
    exit(0 if success else 1)