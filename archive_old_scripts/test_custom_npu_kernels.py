#!/usr/bin/env python3
"""
Test Custom NPU Kernels Performance
Direct testing of compiled NPU kernels for Gemma 3 27B
"""

import torch
import numpy as np
import logging
import time
from pathlib import Path

# Import custom NPU kernels
from gemma3_npu_attention_kernel import Gemma3NPUAttentionKernel
from npu_qkv_projection_kernels import NPUQKVProjectionKernels
from npu_scaled_attention_kernel import NPUScaledAttentionKernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_custom_npu_kernels():
    """Test performance of custom NPU kernels with compiled binaries"""
    logger.info("🚀 Testing Custom NPU Kernels Performance")
    
    # Test dimensions based on Gemma 3 27B architecture
    batch_size = 1
    seq_len = 64  # Small sequence for quick testing
    hidden_size = 5376
    q_output_size = 4096
    kv_output_size = 2048
    
    logger.info(f"📊 Test dimensions: {batch_size}x{seq_len}x{hidden_size}")
    
    # Initialize kernels
    logger.info("🔧 Initializing custom NPU kernels...")
    gemma3_kernel = Gemma3NPUAttentionKernel()
    qkv_kernels = NPUQKVProjectionKernels() 
    scaled_attention = NPUScaledAttentionKernel()
    
    # Initialize kernels
    kernels_init_start = time.time()
    if not gemma3_kernel.initialize():
        logger.error("❌ Gemma3 kernel initialization failed")
        return False
        
    if not qkv_kernels.initialize():
        logger.error("❌ Q/K/V kernels initialization failed") 
        return False
        
    if not scaled_attention.initialize():
        logger.error("❌ Scaled attention kernel initialization failed")
        return False
    
    kernels_init_time = time.time() - kernels_init_start
    logger.info(f"✅ All kernels initialized in {kernels_init_time:.3f}s")
    
    # Create test tensors
    logger.info("📊 Creating test tensors...")
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    # Create quantized weight tensors (simulating INT8 quantized weights)
    q_weight = torch.randint(-128, 127, (hidden_size, q_output_size), dtype=torch.int8)
    k_weight = torch.randint(-128, 127, (hidden_size, kv_output_size), dtype=torch.int8) 
    v_weight = torch.randint(-128, 127, (hidden_size, kv_output_size), dtype=torch.int8)
    o_weight = torch.randint(-128, 127, (q_output_size, hidden_size), dtype=torch.int8)
    
    # Quantization scales
    q_scale = torch.tensor(0.01, dtype=torch.float16)
    k_scale = torch.tensor(0.01, dtype=torch.float16)
    v_scale = torch.tensor(0.01, dtype=torch.float16)
    o_scale = torch.tensor(0.01, dtype=torch.float16)
    
    logger.info("📊 Test tensor shapes:")
    logger.info(f"   Hidden states: {hidden_states.shape}")
    logger.info(f"   Q weight: {q_weight.shape}")
    logger.info(f"   K/V weight: {k_weight.shape} / {v_weight.shape}")
    logger.info(f"   O weight: {o_weight.shape}")
    
    # Test 1: Complete Gemma 3 NPU attention kernel
    logger.info("🔥 Test 1: Complete Gemma 3 NPU Attention Kernel")
    try:
        complete_start = time.time()
        
        attention_output = gemma3_kernel.compute_attention(
            hidden_states, q_weight, q_scale, k_weight, k_scale,
            v_weight, v_scale, o_weight, o_scale
        )
        
        complete_time = time.time() - complete_start
        
        logger.info(f"✅ Complete kernel: {complete_time*1000:.2f}ms")
        logger.info(f"   Output shape: {attention_output.shape}")
        logger.info(f"   Performance: {seq_len/complete_time:.2f} tokens/sec")
        
    except Exception as e:
        logger.error(f"❌ Complete kernel failed: {e}")
    
    # Test 2: Modular NPU kernels
    logger.info("🔥 Test 2: Modular NPU Kernels (Q/K/V + Attention)")
    try:
        modular_start = time.time()
        
        # Q/K/V projections
        q, k, v = qkv_kernels.execute_qkv_projections(
            hidden_states, q_weight, q_scale, k_weight, k_scale, v_weight, v_scale
        )
        
        # Scaled attention
        context = scaled_attention.compute_scaled_attention(q, k, v)
        
        # Output projection (simple matmul for now)
        output = torch.matmul(context, o_weight.float().T)
        
        modular_time = time.time() - modular_start
        
        logger.info(f"✅ Modular kernels: {modular_time*1000:.2f}ms")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Performance: {seq_len/modular_time:.2f} tokens/sec")
        
    except Exception as e:
        logger.error(f"❌ Modular kernels failed: {e}")
    
    # Test 3: Check compiled binaries exist
    logger.info("🔥 Test 3: Check Compiled NPU Binaries")
    
    binaries = [
        "npu_binaries/gemma3_q_projection.npu_binary",
        "npu_binaries/gemma3_k_projection.npu_binary", 
        "npu_binaries/gemma3_v_projection.npu_binary"
    ]
    
    for binary in binaries:
        if Path(binary).exists():
            size = Path(binary).stat().st_size
            logger.info(f"   ✅ {binary}: {size} bytes")
        else:
            logger.warning(f"   ⚠️ {binary}: Not found")
    
    # Test 4: Performance scaling test
    logger.info("🔥 Test 4: Performance Scaling Test")
    
    sequence_lengths = [16, 32, 64, 128]
    
    for test_seq_len in sequence_lengths:
        test_hidden = torch.randn(1, test_seq_len, hidden_size, dtype=torch.float16)
        
        try:
            scaling_start = time.time()
            
            # Use Q/K/V kernels for scaling test
            q, k, v = qkv_kernels.execute_qkv_projections(
                test_hidden, q_weight, q_scale, k_weight, k_scale, v_weight, v_scale
            )
            
            scaling_time = time.time() - scaling_start
            tps = test_seq_len / scaling_time
            
            logger.info(f"   📊 Seq len {test_seq_len}: {scaling_time*1000:.2f}ms ({tps:.2f} tokens/sec)")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Seq len {test_seq_len}: Failed ({e})")
    
    logger.info("🎉 Custom NPU kernel testing complete!")
    return True

if __name__ == "__main__":
    test_custom_npu_kernels()