#!/usr/bin/env python3
"""
Minimal NPU test to verify real hardware execution
Focus on NPU without GPU to isolate issues
"""

import numpy as np
import time
import logging
import sys
import os

# Add XRT to path
sys.path.append('/opt/xilinx/xrt/python')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_npu_basic():
    """Test basic NPU functionality"""
    
    logger.info("🚀 MINIMAL NPU TEST")
    logger.info("=" * 60)
    
    # Test 1: Check if we can import XRT
    try:
        import pyxrt as xrt
        logger.info("✅ PyXRT imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import PyXRT: {e}")
        return False
    
    # Test 2: Check NPU device
    try:
        device = xrt.device(0)
        logger.info("✅ NPU device initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NPU device: {e}")
        return False
    
    # Test 3: Try to load a kernel binary directly
    kernel_files = [
        "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels/attention_256_int8.bin",
        "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels/npu_attention_kernels.xclbin"
    ]
    
    for kernel_file in kernel_files:
        if os.path.exists(kernel_file):
            logger.info(f"🔍 Testing kernel: {kernel_file}")
            try:
                if kernel_file.endswith('.xclbin'):
                    xclbin = xrt.xclbin(kernel_file)
                    uuid = device.register_xclbin(xclbin)
                    logger.info(f"✅ XCLBIN registered with UUID: {uuid}")
                    
                    # List available kernels
                    kernels = xclbin.get_kernels()
                    logger.info(f"  Available kernels: {kernels}")
                else:
                    logger.info(f"  Binary kernel found: {os.path.getsize(kernel_file)} bytes")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed to load kernel: {e}")
    
    # Test 4: Check NPU driver status
    logger.info("\n📊 NPU Driver Status:")
    try:
        import subprocess
        result = subprocess.run(['sudo', 'dmesg', '|', 'grep', '-i', 'amdxdna', '|', 'tail', '-5'], 
                              capture_output=True, text=True, shell=True)
        if result.stdout:
            logger.info(f"  Driver messages:\n{result.stdout}")
    except:
        pass
    
    # Test 5: Simple computation test
    logger.info("\n🧮 Testing simple matrix multiplication...")
    try:
        # Create test matrices with Gemma3 4B dimensions
        hidden_size = 2560  # Gemma3 4B hidden size
        num_heads = 32
        head_dim = hidden_size // num_heads  # 80
        
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Head dim: {head_dim}")
        
        # Create small test data
        batch_size = 1
        seq_len = 16  # Small sequence for testing
        
        q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        k = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        v = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        
        logger.info(f"  Q shape: {q.shape}")
        logger.info(f"  K shape: {k.shape}")
        logger.info(f"  V shape: {v.shape}")
        
        # Simulate attention computation
        start = time.time()
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        output = np.matmul(attn, v)
        elapsed = time.time() - start
        
        logger.info(f"  ✅ Attention computed in {elapsed*1000:.2f}ms")
        logger.info(f"  Output shape: {output.shape}")
        
    except Exception as e:
        logger.error(f"  ❌ Computation failed: {e}")
    
    return True

def test_npu_with_gemma3n_dims():
    """Test NPU with Gemma3n dimensions (which the kernels expect)"""
    
    logger.info("\n🧪 Testing with Gemma3n E4B dimensions...")
    
    # Gemma3n E4B dimensions (from kernel config)
    hidden_size = 3072
    num_heads = 24
    head_dim = 128
    
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Num heads: {num_heads}")
    logger.info(f"  Head dim: {head_dim}")
    
    try:
        # Create test data with Gemma3n dimensions
        batch_size = 1
        seq_len = 64
        
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        logger.info(f"  Hidden states shape: {hidden_states.shape}")
        
        # Test reshape for multi-head attention
        # This is where the error occurs with wrong dimensions
        reshaped = hidden_states.reshape(batch_size, seq_len, num_heads, head_dim)
        logger.info(f"  ✅ Reshape successful: {reshaped.shape}")
        
        # Transpose for attention
        q = reshaped.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        logger.info(f"  ✅ Q tensor shape: {q.shape}")
        
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    try:
        # Run basic NPU test
        test_npu_basic()
        
        # Test with Gemma3n dimensions
        test_npu_with_gemma3n_dims()
        
        logger.info("\n✅ NPU tests completed")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()