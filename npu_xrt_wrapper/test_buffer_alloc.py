#!/usr/bin/env python3
"""
Test NPU buffer allocation without XCLBIN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from npu_kernel_executor import NPUKernelExecutor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_buffer_allocation():
    """Test NPU buffer allocation"""
    
    logger.info("🧪 Testing NPU Buffer Allocation...")
    
    executor = NPUKernelExecutor()
    
    # Open device
    if not executor.open_device(0):
        logger.error("Failed to open NPU device")
        return
    
    logger.info("Testing different buffer allocation flags...")
    
    # Test different buffer sizes and flags
    test_configs = [
        (1024, "1KB", executor.XCL_BO_FLAGS_DEVICE_RAM),
        (4096, "4KB", executor.XCL_BO_FLAGS_DEVICE_RAM),
        (1024*1024, "1MB", executor.XCL_BO_FLAGS_DEVICE_RAM),
        (1024, "1KB_CACHEABLE", executor.XCL_BO_FLAGS_CACHEABLE),
        (1024, "1KB_HOST_ONLY", executor.XCL_BO_FLAGS_HOST_ONLY),
    ]
    
    successful_allocations = []
    
    for size, name, flags in test_configs:
        buffer_name = f"test_{name}"
        buffer = executor.allocate_buffer(size, buffer_name, flags)
        
        if buffer:
            logger.info(f"✅ Allocated {name} buffer with flags 0x{flags:02x}")
            successful_allocations.append((buffer_name, size, flags))
            
            # Try to write data
            test_data = np.random.randn(size // 4).astype(np.float32)
            if executor.write_buffer(buffer_name, test_data):
                logger.info(f"   ✅ Data written successfully")
                
                # Try to read back
                output_data = np.zeros_like(test_data)
                if executor.read_buffer(buffer_name, output_data):
                    logger.info(f"   ✅ Data read successfully")
                    # Verify data
                    if np.allclose(test_data, output_data):
                        logger.info(f"   ✅ Data verification passed!")
                    else:
                        logger.warning(f"   ⚠️ Data verification failed")
        else:
            logger.warning(f"❌ Failed to allocate {name} buffer with flags 0x{flags:02x}")
    
    logger.info(f"\nSummary: {len(successful_allocations)} successful allocations")
    
    # Cleanup
    executor.cleanup()
    
    logger.info("✅ Buffer allocation test complete")

if __name__ == "__main__":
    test_buffer_allocation()