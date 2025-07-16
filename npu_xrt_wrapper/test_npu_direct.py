#!/usr/bin/env python3
"""
Test NPU using our compiled kernels directly
"""

import os
import sys
import json
import logging

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npu_attention_kernel_real import NPUAttentionKernelReal
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_npu():
    """Test NPU with our compiled attention kernels"""
    
    logger.info("🧪 Testing NPU with compiled attention kernels...")
    
    # Initialize NPU attention kernel
    npu_kernel = NPUAttentionKernelReal()
    
    # Test with different sequence lengths
    test_configs = [
        (256, 32, 5376),   # seq_len, num_heads, hidden_size
        (512, 32, 5376),
        (1024, 32, 5376),
    ]
    
    for seq_len, num_heads, hidden_size in test_configs:
        logger.info(f"\n📊 Testing seq_len={seq_len}, heads={num_heads}, hidden={hidden_size}")
        
        # Create test data
        hidden_states = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        
        # Execute on NPU
        try:
            output = npu_kernel.forward(hidden_states, num_heads)
            
            if output is not None:
                logger.info(f"✅ NPU execution successful!")
                logger.info(f"   Output shape: {output.shape}")
                logger.info(f"   Output stats - mean: {output.mean():.4f}, std: {output.std():.4f}")
            else:
                logger.warning("⚠️ NPU returned None - execution may have failed")
                
        except Exception as e:
            logger.error(f"❌ NPU execution failed: {e}")
            
    logger.info("\n✅ Direct NPU test complete")

if __name__ == "__main__":
    test_direct_npu()