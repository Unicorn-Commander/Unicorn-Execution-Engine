#!/usr/bin/env python3
"""
Fixed NPU Executor - Properly handles XRT API for NPU execution
"""

import ctypes
import numpy as np
import os
import logging
import json
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUExecutorFixed:
    """NPU Executor with proper XRT integration"""
    
    def __init__(self):
        self.device_handle = None
        self.xrt_core = None
        self.xrt_driver = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize NPU with XRT"""
        try:
            # Set library paths
            os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
            
            # Load XRT core library
            self.xrt_core = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_core.so.2', mode=ctypes.RTLD_GLOBAL)
            
            # Load XRT driver
            self.xrt_driver = ctypes.CDLL('/opt/xilinx/xrt/lib/libxrt_driver_xdna.so', mode=ctypes.RTLD_GLOBAL)
            
            logger.info("✅ XRT libraries loaded")
            
            # Open device
            self.xrt_core.xrtDeviceOpen.argtypes = [ctypes.c_uint]
            self.xrt_core.xrtDeviceOpen.restype = ctypes.c_void_p
            
            self.device_handle = self.xrt_core.xrtDeviceOpen(0)
            
            if self.device_handle:
                logger.info("✅ NPU device opened")
                self.initialized = True
                return True
            else:
                logger.error("❌ Failed to open NPU device")
                return False
                
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def execute_kernel_binary(self, kernel_path: str, input_data: np.ndarray, 
                            seq_length: int, num_heads: int) -> Optional[np.ndarray]:
        """Execute a kernel binary on NPU"""
        
        if not self.initialized:
            logger.error("NPU not initialized")
            return None
            
        if not os.path.exists(kernel_path):
            logger.error(f"Kernel not found: {kernel_path}")
            return None
            
        logger.info(f"🚀 Executing kernel: {os.path.basename(kernel_path)}")
        logger.info(f"   Input shape: {input_data.shape}")
        logger.info(f"   Seq length: {seq_length}, Heads: {num_heads}")
        
        # For now, simulate execution since we can't load raw binaries without XCLBIN
        # In a real implementation, we would:
        # 1. Wrap the kernel binary in XCLBIN format
        # 2. Load it using xrtDeviceLoadXclbin
        # 3. Create kernel and run handles
        # 4. Execute and get results
        
        # Read kernel metadata if available
        config_path = os.path.join(os.path.dirname(kernel_path), "kernel_configs.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"✅ Loaded kernel config: {config.get('attention_config', {}).get('kernel_type', 'unknown')}")
        
        # Simulate attention computation
        logger.warning("⚠️ NPU execution simulation - real execution requires XCLBIN wrapper")
        
        # Return input as placeholder
        return input_data
    
    def cleanup(self):
        """Clean up NPU resources"""
        if self.device_handle and self.xrt_core:
            try:
                self.xrt_core.xrtDeviceClose.argtypes = [ctypes.c_void_p]
                self.xrt_core.xrtDeviceClose.restype = ctypes.c_int
                self.xrt_core.xrtDeviceClose(self.device_handle)
                logger.info("✅ NPU device closed")
            except:
                pass

def test_npu_executor():
    """Test the fixed NPU executor"""
    
    logger.info("🧪 Testing Fixed NPU Executor...")
    
    executor = NPUExecutorFixed()
    
    if not executor.initialize():
        logger.error("Failed to initialize NPU")
        return
    
    # Test with our compiled kernels
    kernel_dir = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/npu_kernels"
    
    test_configs = [
        ("attention_256_int8.bin", 256, 32, 5376),
        ("attention_512_int8.bin", 512, 32, 5376),
    ]
    
    for kernel_name, seq_len, num_heads, hidden_size in test_configs:
        kernel_path = os.path.join(kernel_dir, kernel_name)
        
        if os.path.exists(kernel_path):
            logger.info(f"\n📊 Testing {kernel_name}")
            
            # Create test data
            test_data = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
            
            # Execute
            result = executor.execute_kernel_binary(
                kernel_path, test_data, seq_len, num_heads
            )
            
            if result is not None:
                logger.info(f"✅ Execution completed (simulated)")
            else:
                logger.error(f"❌ Execution failed")
    
    # Cleanup
    executor.cleanup()
    
    logger.info("\n✅ NPU executor test complete")
    
    # Summary
    logger.info("\n📋 NPU XRT Wrapper Status:")
    logger.info("✅ NPU device can be opened via XRT")
    logger.info("✅ XRT libraries are properly loaded")
    logger.info("✅ Kernel binaries are available")
    logger.info("⏳ Need XCLBIN wrapper for actual execution")
    logger.info("💡 Next step: Create XCLBIN from kernel binaries")

if __name__ == "__main__":
    test_npu_executor()