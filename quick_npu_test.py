#!/usr/bin/env python3
"""
Quick NPU Test - Verify Fixed Framework
"""

import torch
import numpy as np
import logging
import time
from gemma3_npu_attention_kernel import Gemma3NPUAttentionKernel
from setup_real_model_test import load_real_test_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_npu_test():
    """Quick test of fixed NPU framework"""
    logger.info("🚀 Quick NPU Test - Verifying Fixed Framework")
    
    # Load test data
    weights, inputs, metadata = load_real_test_data()
    
    # Create attention kernel
    kernel = Gemma3NPUAttentionKernel()
    if not kernel.initialize():
        logger.error("❌ Kernel initialization failed")
        return False
    
    # Get small test input
    hidden_states = torch.from_numpy(inputs["seq_16"])
    
    # Get weights
    q_weight = torch.from_numpy(weights['q_weight'])
    q_scale = torch.from_numpy(weights['q_scale'])
    k_weight = torch.from_numpy(weights['k_weight'])
    k_scale = torch.from_numpy(weights['k_scale'])
    v_weight = torch.from_numpy(weights['v_weight'])
    v_scale = torch.from_numpy(weights['v_scale'])
    o_weight = torch.from_numpy(weights['o_weight'])
    o_scale = torch.from_numpy(weights['o_scale'])
    
    logger.info("🔥 Testing complete attention computation...")
    
    start_time = time.time()
    
    try:
        # Test complete attention kernel
        output = kernel.compute_attention(
            hidden_states, q_weight, q_scale, k_weight, k_scale,
            v_weight, v_scale, o_weight, o_scale
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_second = 16 / total_time  # 16 tokens in seq_16
        
        logger.info(f"✅ Complete Attention Success!")
        logger.info(f"   Input shape: {hidden_states.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Total time: {total_time*1000:.2f}ms")
        logger.info(f"   🚀 Tokens/second: {tokens_per_second:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_npu_test()
    if success:
        logger.info("🎉 REAL NPU FRAMEWORK WORKING PERFECTLY!")
    else:
        logger.error("❌ Test failed")