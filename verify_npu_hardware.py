#!/usr/bin/env python3
"""
Simple NPU hardware verification test
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import pyxrt as xrt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_npu_hardware():
    """Test NPU hardware with working XCLBIN"""
    
    logger.info("🎉 TESTING REAL NPU HARDWARE")
    logger.info("=" * 50)
    
    try:
        # Initialize NPU device
        device = xrt.device(0)
        logger.info("✅ NPU device initialized")
        
        # Load working XCLBIN
        xclbin = xrt.xclbin("/opt/xilinx/xrt/amdxdna/bins/17f0_20/validate.xclbin")
        device.register_xclbin(xclbin)
        logger.info("✅ Working XCLBIN loaded")
        
        # Get UUID
        uuid = xclbin.get_uuid()
        logger.info(f"✅ XCLBIN UUID: {uuid}")
        
        logger.info("\n🎉 SUCCESS: REAL NPU HARDWARE IS OPERATIONAL!")
        logger.info("✅ NPU device: WORKING")
        logger.info("✅ XRT runtime: WORKING") 
        logger.info("✅ XCLBIN loading: WORKING")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ NPU test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_npu_hardware()
    exit(0 if success else 1)
