#!/usr/bin/env python3
"""Test NPU initialization"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add XRT to path
sys.path.append('/opt/xilinx/xrt/python')

try:
    import pyxrt as xrt
    logger.info("✅ XRT imported successfully")
    
    # Try to initialize device
    device = xrt.device(0)
    logger.info("✅ NPU device initialized")
    
    # Try to load xclbin
    xclbin_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels/npu_attention_kernels.xclbin"
    xclbin = xrt.xclbin(xclbin_path)
    logger.info("✅ XCLBIN loaded")
    
    # Register xclbin
    device.register_xclbin(xclbin)
    logger.info("✅ XCLBIN registered with device")
    
    # Get kernel handle
    kernel = xrt.kernel(device, xclbin.get_uuid(), "attention_256_int8")
    logger.info("✅ Kernel handle obtained")
    
    logger.info("🎉 NPU initialization successful!")
    
except Exception as e:
    logger.error(f"❌ NPU initialization failed: {e}")
    import traceback
    traceback.print_exc()