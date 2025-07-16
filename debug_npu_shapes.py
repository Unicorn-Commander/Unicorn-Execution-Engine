#!/usr/bin/env python3
"""
Debug NPU Attention Kernel Shapes
Quick test to identify the matrix dimension mismatch
"""

import torch
import numpy as np
import logging
from gemma3_npu_attention_kernel import Gemma3NPUAttentionKernel
from setup_real_model_test import load_real_test_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_attention_shapes():
    """Debug attention kernel shapes"""
    logger.info("🔍 Debugging NPU attention kernel shapes...")
    
    # Load test data
    weights, inputs, metadata = load_real_test_data()
    
    # Create attention kernel
    kernel = Gemma3NPUAttentionKernel()
    if not kernel.initialize():
        logger.error("❌ Kernel initialization failed")
        return
    
    # Get small test input
    hidden_states = torch.from_numpy(inputs["seq_16"])  # Small test
    
    # Get weights
    q_weight = torch.from_numpy(weights['q_weight'])
    q_scale = torch.from_numpy(weights['q_scale'])
    k_weight = torch.from_numpy(weights['k_weight'])
    k_scale = torch.from_numpy(weights['k_scale'])
    v_weight = torch.from_numpy(weights['v_weight'])
    v_scale = torch.from_numpy(weights['v_scale'])
    o_weight = torch.from_numpy(weights['o_weight'])
    o_scale = torch.from_numpy(weights['o_scale'])
    
    logger.info(f"📊 Input shapes:")
    logger.info(f"   Hidden states: {hidden_states.shape}")
    logger.info(f"   Q weight: {q_weight.shape}")
    logger.info(f"   K weight: {k_weight.shape}")
    logger.info(f"   V weight: {v_weight.shape}")
    logger.info(f"   O weight: {o_weight.shape}")
    
    try:
        # Try just the first part - Q/K/V projections
        hidden_np = hidden_states.detach().cpu().numpy()
        q_weight_np = q_weight.detach().cpu().numpy()
        q_scale_np = q_scale.detach().cpu().numpy()
        k_weight_np = k_weight.detach().cpu().numpy()
        k_scale_np = k_scale.detach().cpu().numpy()
        v_weight_np = v_weight.detach().cpu().numpy()
        v_scale_np = v_scale.detach().cpu().numpy()
        
        # Execute Q/K/V projections
        q, k, v = kernel.npu_device.execute_qkv_projections(
            hidden_np, q_weight_np, q_scale_np, k_weight_np, k_scale_np, v_weight_np, v_scale_np
        )
        
        logger.info(f"📊 Q/K/V projection results:")
        logger.info(f"   Q shape: {q.shape}")
        logger.info(f"   K shape: {k.shape}")
        logger.info(f"   V shape: {v.shape}")
        
        # Execute attention
        context = kernel.npu_device.execute_attention_compute(q, k, v)
        
        logger.info(f"📊 Attention result:")
        logger.info(f"   Context shape: {context.shape}")
        
        # Check what happens with output projection
        o_weight_np = o_weight.detach().cpu().numpy()
        o_scale_np = o_scale.detach().cpu().numpy()
        o_weight_fp = o_weight_np.astype(np.float16) * o_scale_np.astype(np.float16)
        
        logger.info(f"📊 Output projection:")
        logger.info(f"   Context shape: {context.shape}")
        logger.info(f"   O weight shape: {o_weight_fp.shape}")
        logger.info(f"   O weight.T shape: {o_weight_fp.T.shape}")
        
        # Try the multiplication
        if context.shape[-1] != o_weight_fp.shape[0]:
            logger.error(f"❌ Shape mismatch: context {context.shape} vs o_weight {o_weight_fp.shape}")
            logger.error(f"   Expected: context[..., {o_weight_fp.shape[0]}] but got context[..., {context.shape[-1]}]")
        else:
            result = np.matmul(context.astype(np.float16), o_weight_fp)
            logger.info(f"✅ Output projection works: {result.shape}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_attention_shapes()