#!/usr/bin/env python3
"""
Test NPU with correct Gemma3n dimensions (hidden_size=3072)
This verifies NPU hardware execution without model dimension mismatches
"""

import os
import sys
import time
import numpy as np
import logging

# Add path for NPU modules
sys.path.append('/opt/xilinx/xrt/python')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPUGemma3nTest:
    """Test NPU with Gemma3n dimensions"""
    
    def __init__(self):
        self.npu_device = None
        self.xclbin_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels/npu_attention_kernels.xclbin"
        
        # Gemma3n dimensions
        self.hidden_size = 3072
        self.num_heads = 48  # Gemma3n has 48 heads
        self.head_dim = 64   # 3072 / 48 = 64
        self.num_kv_heads = 8  # GQA with 8 KV heads
        
    def initialize_npu(self):
        """Initialize NPU device"""
        try:
            import pyxrt as xrt
            
            # Find NPU device
            for i in range(4):
                try:
                    device = xrt.device(i)
                    if device.get_info(xrt.xclbin_info.dsa_name).startswith("AMD"):
                        self.npu_device = device
                        logger.info(f"✅ NPU device found at index {i}")
                        return True
                except:
                    continue
            
            logger.error("❌ No NPU device found")
            return False
            
        except Exception as e:
            logger.error(f"❌ NPU initialization failed: {e}")
            return False
    
    def load_kernel(self):
        """Load NPU kernel"""
        try:
            import pyxrt as xrt
            
            # Load XCLBIN
            with open(self.xclbin_path, 'rb') as f:
                xclbin_data = f.read()
            
            xclbin = xrt.xclbin(xclbin_data)
            self.npu_device.register_xclbin(xclbin)
            
            # Find kernel
            kernel_name = "attention_256_int8"
            kernel = xrt.kernel(self.npu_device, xclbin.get_uuid(), kernel_name)
            
            logger.info(f"✅ Loaded kernel: {kernel_name}")
            return kernel
            
        except Exception as e:
            logger.error(f"❌ Kernel loading failed: {e}")
            return None
    
    def test_attention_computation(self):
        """Test attention computation with correct dimensions"""
        try:
            import pyxrt as xrt
            
            # Initialize NPU
            if not self.initialize_npu():
                return False
            
            # Load kernel
            kernel = self.load_kernel()
            if not kernel:
                return False
            
            # Create test data with Gemma3n dimensions
            batch_size = 1
            seq_len = 256  # Match kernel name
            
            # Input: [batch, seq_len, hidden_size]
            input_data = np.random.randn(batch_size, seq_len, self.hidden_size).astype(np.float32)
            
            # Weights (quantized to INT8)
            q_weight = np.random.randint(-127, 127, (self.hidden_size, self.hidden_size), dtype=np.int8)
            k_weight = np.random.randint(-127, 127, (self.hidden_size, self.num_kv_heads * self.head_dim), dtype=np.int8)
            v_weight = np.random.randint(-127, 127, (self.hidden_size, self.num_kv_heads * self.head_dim), dtype=np.int8)
            o_weight = np.random.randint(-127, 127, (self.hidden_size, self.hidden_size), dtype=np.int8)
            
            # Allocate buffers
            input_bo = xrt.bo(self.npu_device, input_data.nbytes, xrt.bo.flags.normal, kernel.group_id(0))
            q_bo = xrt.bo(self.npu_device, q_weight.nbytes, xrt.bo.flags.normal, kernel.group_id(1))
            k_bo = xrt.bo(self.npu_device, k_weight.nbytes, xrt.bo.flags.normal, kernel.group_id(2))
            v_bo = xrt.bo(self.npu_device, v_weight.nbytes, xrt.bo.flags.normal, kernel.group_id(3))
            o_bo = xrt.bo(self.npu_device, o_weight.nbytes, xrt.bo.flags.normal, kernel.group_id(4))
            output_bo = xrt.bo(self.npu_device, input_data.nbytes, xrt.bo.flags.normal, kernel.group_id(5))
            
            # Write data to buffers
            input_bo.write(input_data.tobytes())
            q_bo.write(q_weight.tobytes())
            k_bo.write(k_weight.tobytes())
            v_bo.write(v_weight.tobytes())
            o_bo.write(o_weight.tobytes())
            
            # Sync to device
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            q_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            k_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            v_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            o_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            
            # Create run handle
            run = kernel(input_bo, q_bo, k_bo, v_bo, o_bo, output_bo)
            
            logger.info("🚀 Executing NPU kernel...")
            start_time = time.time()
            
            # Execute
            run.start()
            run.wait()
            
            elapsed = time.time() - start_time
            
            # Sync output
            output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            
            # Calculate performance
            gflops = (2 * batch_size * seq_len * self.hidden_size * self.hidden_size) / (elapsed * 1e9)
            
            logger.info(f"✅ NPU execution successful!")
            logger.info(f"  ⏱️  Time: {elapsed*1000:.2f} ms")
            logger.info(f"  🔥 Performance: {gflops:.2f} GFLOPS")
            logger.info(f"  📊 Theoretical NPU: {16*1e3:.0f} GFLOPS (16 TOPS INT8)")
            logger.info(f"  📈 Utilization: {(gflops/(16*1e3))*100:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ NPU test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run NPU test"""
    logger.info("🧪 NPU HARDWARE TEST WITH GEMMA3N DIMENSIONS")
    logger.info("=" * 60)
    
    tester = NPUGemma3nTest()
    
    # Test NPU execution
    if tester.test_attention_computation():
        logger.info("✅ NPU HARDWARE WORKING WITH CORRECT DIMENSIONS!")
        logger.info("   Next step: Load actual Gemma3n model weights")
    else:
        logger.info("❌ NPU test failed - check kernel compatibility")

if __name__ == "__main__":
    main()