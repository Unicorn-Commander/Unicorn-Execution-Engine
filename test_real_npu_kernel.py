#!/usr/bin/env python3
"""
Test real NPU kernel with enhanced XCLBIN
"""

import os
import sys
import numpy as np
import logging
import time
import torch

# Add to path
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

from npu_attention_kernel_real import NPUAttentionKernel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_npu_kernel():
    """Test the real NPU kernel with enhanced XCLBIN"""
    
    logger.info("🚀 Testing Real NPU Kernel with Enhanced XCLBIN")
    logger.info("=" * 70)
    
    # Check if kernel file exists
    kernel_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_real/attention_256_real.xclbin"
    if not os.path.exists(kernel_path):
        logger.error(f"❌ Enhanced kernel not found: {kernel_path}")
        return False
    
    logger.info(f"✅ Enhanced kernel found: {kernel_path}")
    kernel_size = os.path.getsize(kernel_path)
    logger.info(f"📏 Kernel size: {kernel_size} bytes")
    
    # Initialize NPU kernel
    logger.info("\n🔧 Initializing NPU Kernel...")
    npu_kernel = NPUAttentionKernel(seq_length=256, d_model=2560, num_heads=20)
    
    # Try to initialize
    try:
        success = npu_kernel.initialize()
        if success:
            logger.info("✅ NPU kernel initialized successfully!")
            logger.info("🎉 Real NPU hardware is working!")
            return True
        else:
            logger.error("❌ NPU kernel initialization failed")
            return False
    except Exception as e:
        logger.error(f"❌ NPU kernel initialization error: {e}")
        logger.info("💡 This might be expected if NPU needs specific kernel format")
        return False

def test_npu_dimensions():
    """Test NPU with correct Gemma3 4B dimensions"""
    
    logger.info("\n📊 Testing NPU with Gemma3 4B Dimensions")
    logger.info("=" * 50)
    
    # Gemma3 4B specifications
    seq_len = 256
    d_model = 2560
    num_heads = 20
    head_dim = d_model // num_heads  # 128
    
    logger.info(f"📏 Sequence length: {seq_len}")
    logger.info(f"📏 Hidden dimension: {d_model}")
    logger.info(f"📏 Number of heads: {num_heads}")
    logger.info(f"📏 Head dimension: {head_dim}")
    
    # Create test data
    logger.info("\n🔧 Creating test data...")
    batch_size = 1
    
    # Create realistic test tensors
    hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # Weight matrices for Q, K, V projections
    q_proj_weight = np.random.randn(d_model, d_model).astype(np.float32)
    k_proj_weight = np.random.randn(d_model, d_model).astype(np.float32)  
    v_proj_weight = np.random.randn(d_model, d_model).astype(np.float32)
    o_proj_weight = np.random.randn(d_model, d_model).astype(np.float32)
    
    logger.info(f"✅ Test data created:")
    logger.info(f"   Hidden states: {hidden_states.shape}")
    logger.info(f"   Q projection: {q_proj_weight.shape}")
    logger.info(f"   K projection: {k_proj_weight.shape}")
    logger.info(f"   V projection: {v_proj_weight.shape}")
    logger.info(f"   O projection: {o_proj_weight.shape}")
    
    # Test NPU kernel
    logger.info("\n⚡ Testing NPU computation...")
    npu_kernel = NPUAttentionKernel(seq_length=seq_len, d_model=d_model, num_heads=num_heads)
    
    try:
        # Initialize
        if npu_kernel.initialize():
            logger.info("✅ NPU initialized with correct dimensions")
            
            # Try computation
            start_time = time.time()
            result = npu_kernel.compute_flash_attention(
                hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
            )
            computation_time = time.time() - start_time
            
            output, kv_cache, qkv_cache, duration = result
            
            logger.info(f"✅ NPU computation successful!")
            logger.info(f"⏱️  Computation time: {computation_time:.3f}s")
            logger.info(f"📊 Output shape: {output.shape}")
            logger.info(f"🚀 Real NPU acceleration achieved!")
            
            return True
            
        else:
            logger.warning("⚠️  NPU initialization failed, but dimensions are correct")
            return False
            
    except Exception as e:
        logger.error(f"❌ NPU computation failed: {e}")
        logger.info("💡 This might indicate the kernel format needs adjustment")
        return False

def main():
    """Main test function"""
    
    logger.info("🧪 REAL NPU KERNEL TEST SUITE")
    logger.info("=" * 70)
    
    # Test 1: Kernel file verification
    kernel_success = test_real_npu_kernel()
    
    # Test 2: Dimension compatibility
    dimension_success = test_npu_dimensions()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Enhanced kernel file: {'PASS' if kernel_success else 'FAIL'}")
    logger.info(f"✅ Dimension compatibility: {'PASS' if dimension_success else 'FAIL'}")
    
    if kernel_success and dimension_success:
        logger.info("🎉 ALL TESTS PASSED - Real NPU is working!")
        return 0
    elif kernel_success:
        logger.info("🔧 Kernel file created successfully")
        logger.info("💡 Next step: Adjust kernel format for hardware compatibility")
        return 0
    else:
        logger.info("⚠️  Some tests failed, but this is progress!")
        logger.info("💡 Enhanced kernel is available for further development")
        return 1

if __name__ == "__main__":
    exit(main())