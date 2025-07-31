#\!/usr/bin/env python3
"""
Real NPU Attention Kernel - Direct AMD Phoenix NPU Hardware Acceleration
No simulations - real hardware or failure
"""

import numpy as np
import logging
import ctypes
import os
import time
from typing import Dict, Tuple, List, Optional, Any

# Import XRT for NPU interaction
import sys
sys.path.append('/opt/xilinx/xrt/python')
import pyxrt as xrt

logger = logging.getLogger(__name__)

class NPUAttentionKernelReal:
    """Real NPU Attention Kernel with direct hardware acceleration"""

    def __init__(self, seq_length=256, d_model=2560, num_heads=20):
        self.seq_length = seq_length
        self.d_model = d_model  # Corrected to 2560 for Gemma3 4B
        self.num_heads = num_heads  # Corrected to 20 for Gemma3 4B  
        self.head_dim = d_model // num_heads  # Should be 128
        self.initialized = False
        
        self.device = None
        self.kernel = None
        self.bo_hidden_states = None
        self.bo_q_weight = None
        self.bo_k_weight = None
        self.bo_v_weight = None
        self.bo_o_weight = None
        self.bo_output = None

        logger.info("🧠 Real NPU Attention Kernel Initialized.")
        logger.info(f"   - Sequence Length: {seq_length}")
        logger.info(f"   - Model Dimension: {d_model}")
        logger.info(f"   - Number of Heads: {num_heads}")
        logger.info(f"   - Head Dimension: {self.head_dim}")

    def initialize(self) -> bool:
        """Initialize real NPU hardware"""
        logger.info("⚡ Initializing Real NPU Hardware...")
        
        try:
            # Try versioned library first, then fallback
            try:
                ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_core.so.2")
            except:
                ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_core.so")
            self.device = xrt.device(0)
            # Load the xclbin file
            xclbin_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_real/attention_256_real.xclbin"
            self.xclbin = xrt.xclbin(xclbin_path)
            self.device.register_xclbin(self.xclbin)
            
            # Get kernel from xclbin
            kernel_name = "gemma3_attention_kernel"  # Enhanced kernel name
            self.kernel = xrt.kernel(self.device, self.xclbin.get_uuid(), kernel_name)
            self.initialized = True
            logger.info("✅ Real NPU Hardware initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ NPU initialization failed: {e}")
            return False

    def compute_flash_attention(self, hidden_states: np.ndarray, q_proj_weight: np.ndarray, 
                               k_proj_weight: np.ndarray, v_proj_weight: np.ndarray, 
                               o_proj_weight: np.ndarray, kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute Flash Attention on real NPU hardware
        """
        if not self.initialized:
            raise RuntimeError("Real NPU Kernel not initialized")

        logger.info(f"🔥 Computing Flash Attention on REAL NPU Hardware: {hidden_states.shape}")
        start_time = time.time()

        # Allocate buffers on the NPU
        self.bo_hidden_states = xrt.bo(self.device, hidden_states.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(0))
        self.bo_q_weight = xrt.bo(self.device, q_proj_weight.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(1))
        self.bo_k_weight = xrt.bo(self.device, k_proj_weight.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(2))
        self.bo_v_weight = xrt.bo(self.device, v_proj_weight.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(3))
        self.bo_o_weight = xrt.bo(self.device, o_proj_weight.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(4))
        self.bo_output = xrt.bo(self.device, hidden_states.nbytes, xrt.bo.flags.cacheable, self.kernel.group_id(5))

        # Write data to the NPU buffers
        self.bo_hidden_states.write(hidden_states, 0)
        self.bo_q_weight.write(q_proj_weight, 0)
        self.bo_k_weight.write(k_proj_weight, 0)
        self.bo_v_weight.write(v_proj_weight, 0)
        self.bo_o_weight.write(o_proj_weight, 0)

        # Synchronize the buffers
        self.bo_hidden_states.sync(xrt.bo.direction.device)
        self.bo_q_weight.sync(xrt.bo.direction.device)
        self.bo_k_weight.sync(xrt.bo.direction.device)
        self.bo_v_weight.sync(xrt.bo.direction.device)
        self.bo_o_weight.sync(xrt.bo.direction.device)

        # Execute the kernel
        run = self.kernel(self.bo_hidden_states, self.bo_q_weight, self.bo_k_weight, self.bo_v_weight, self.bo_o_weight, self.bo_output)
        run.wait()

        # Read the output back from the NPU
        output = self.bo_output.read(hidden_states.nbytes, 0).view(np.float32).reshape(hidden_states.shape)

        npu_time = time.time() - start_time
        logger.info(f"✅ Flash Attention computation complete: {output.shape}")
        return output, None, None, npu_time

    def cleanup(self):
        """Clean up NPU resources"""
        logger.info("🧹 Cleaning up Real NPU Hardware resources...")
        
        if self.npu_context:
            # Clean up NPU context
            self.npu_context = None
            
        if self.npu_device:
            # Clean up NPU device
            self.npu_device = None
            
        if self.xdna_driver:
            # Clean up driver
            self.xdna_driver = None
            
        self.initialized = False
        logger.info("✅ NPU cleanup complete")
