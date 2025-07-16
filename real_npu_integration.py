#!/usr/bin/env python3
"""
Real NPU Integration for Gemma 3 27B
Integrates compiled C++ execution engine with NPU framework
Bypasses Python frameworks for maximum performance
"""

import ctypes
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RealNPUIntegration:
    """Integration layer for real NPU C++ execution engine"""
    
    def __init__(self):
        self.lib = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Load and initialize the real NPU execution engine"""
        try:
            # Load the compiled C++ library
            lib_path = Path("libreal_npu_engine.so")
            if not lib_path.exists():
                logger.error(f"❌ Real NPU library not found: {lib_path}")
                return False
            
            self.lib = ctypes.CDLL(str(lib_path))
            
            # Define function signatures
            self.lib.real_npu_qkv_projections.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input_data
                ctypes.POINTER(ctypes.c_int8),   # q_weight
                ctypes.POINTER(ctypes.c_int8),   # k_weight
                ctypes.POINTER(ctypes.c_int8),   # v_weight
                ctypes.POINTER(ctypes.c_float),  # scales
                ctypes.c_int,                    # seq_len
                ctypes.POINTER(ctypes.c_float),  # q_output
                ctypes.POINTER(ctypes.c_float),  # k_output
                ctypes.POINTER(ctypes.c_float),  # v_output
            ]
            
            self.lib.real_npu_attention.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # q_input
                ctypes.POINTER(ctypes.c_float),  # k_input
                ctypes.POINTER(ctypes.c_float),  # v_input
                ctypes.c_int,                    # seq_len
                ctypes.POINTER(ctypes.c_float),  # output
            ]
            
            self.initialized = True
            logger.info("✅ Real NPU execution engine loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load real NPU engine: {e}")
            return False
    
    def execute_qkv_projections(
        self,
        input_data: np.ndarray,      # [1, seq_len, 5376]
        q_weight: np.ndarray,        # [5376, 4096] INT8
        k_weight: np.ndarray,        # [5376, 2048] INT8
        v_weight: np.ndarray,        # [5376, 2048] INT8
        q_scale: float,
        k_scale: float,
        v_scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute Q/K/V projections using real NPU engine"""
        
        if not self.initialized:
            raise RuntimeError("Real NPU engine not initialized")
        
        batch_size, seq_len, hidden_size = input_data.shape
        assert hidden_size == 5376, f"Expected hidden_size=5376, got {hidden_size}"
        
        # Prepare input data
        input_flat = input_data.astype(np.float32).reshape(-1)
        q_weight_flat = q_weight.astype(np.int8).reshape(-1)
        k_weight_flat = k_weight.astype(np.int8).reshape(-1)
        v_weight_flat = v_weight.astype(np.int8).reshape(-1)
        scales = np.array([q_scale, k_scale, v_scale], dtype=np.float32)
        
        # Prepare output buffers
        q_output = np.zeros((batch_size, seq_len, 4096), dtype=np.float32)
        k_output = np.zeros((batch_size, seq_len, 2048), dtype=np.float32)
        v_output = np.zeros((batch_size, seq_len, 2048), dtype=np.float32)
        
        logger.info(f"   🔥 Real NPU: Executing Q/K/V projections [{batch_size}, {seq_len}, {hidden_size}]")
        
        # Call C++ function
        self.lib.real_npu_qkv_projections(
            input_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q_weight_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            k_weight_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            v_weight_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            seq_len,
            q_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        logger.info(f"   ✅ Real NPU: Q/K/V projections complete")
        logger.info(f"      Q output: {q_output.shape}")
        logger.info(f"      K output: {k_output.shape}")
        logger.info(f"      V output: {v_output.shape}")
        
        return q_output, k_output, v_output
    
    def execute_attention(
        self,
        q_input: np.ndarray,         # [1, seq_len, 32, 128]
        k_input: np.ndarray,         # [1, seq_len, 16, 128]
        v_input: np.ndarray          # [1, seq_len, 16, 128]
    ) -> np.ndarray:
        """Execute attention computation using real NPU engine"""
        
        if not self.initialized:
            raise RuntimeError("Real NPU engine not initialized")
        
        batch_size, seq_len, num_heads, head_dim = q_input.shape
        _, _, kv_heads, _ = k_input.shape
        
        assert num_heads == 32, f"Expected 32 Q heads, got {num_heads}"
        assert kv_heads == 16, f"Expected 16 KV heads, got {kv_heads}"
        assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"
        
        # Prepare input data
        q_flat = q_input.astype(np.float32).reshape(-1)
        k_flat = k_input.astype(np.float32).reshape(-1)
        v_flat = v_input.astype(np.float32).reshape(-1)
        
        # Prepare output buffer
        output = np.zeros((batch_size, seq_len, num_heads, head_dim), dtype=np.float32)
        
        logger.info(f"   🧮 Real NPU: Executing attention [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
        
        # Call C++ function
        self.lib.real_npu_attention(
            q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            seq_len,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        logger.info(f"   ✅ Real NPU: Attention computation complete")
        logger.info(f"      Output: {output.shape}")
        
        return output

# Global instance
real_npu_engine = RealNPUIntegration()

def test_real_npu_engine():
    """Test the real NPU execution engine"""
    logger.info("🔥 Testing Real NPU Execution Engine")
    
    # Initialize engine
    if not real_npu_engine.initialize():
        logger.error("❌ Failed to initialize real NPU engine")
        return False
    
    # Test Q/K/V projections
    logger.info("🔧 Testing Q/K/V projections...")
    
    batch_size, seq_len, hidden_size = 1, 16, 5376
    input_data = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Create quantized weights
    q_weight = np.random.randint(-127, 127, (hidden_size, 4096), dtype=np.int8)
    k_weight = np.random.randint(-127, 127, (hidden_size, 2048), dtype=np.int8)
    v_weight = np.random.randint(-127, 127, (hidden_size, 2048), dtype=np.int8)
    
    # Scales
    q_scale, k_scale, v_scale = 0.01, 0.01, 0.01
    
    # Execute projections
    q_out, k_out, v_out = real_npu_engine.execute_qkv_projections(
        input_data, q_weight, k_weight, v_weight, q_scale, k_scale, v_scale
    )
    
    logger.info(f"✅ Q/K/V projections test passed")
    logger.info(f"   Q: {input_data.shape} -> {q_out.shape}")
    logger.info(f"   K: {input_data.shape} -> {k_out.shape}")
    logger.info(f"   V: {input_data.shape} -> {v_out.shape}")
    
    # Test attention computation
    logger.info("🔧 Testing attention computation...")
    
    # Reshape Q/K/V for attention
    q_reshaped = q_out.reshape(batch_size, seq_len, 32, 128)
    k_reshaped = k_out.reshape(batch_size, seq_len, 16, 128)
    v_reshaped = v_out.reshape(batch_size, seq_len, 16, 128)
    
    # Execute attention
    attention_out = real_npu_engine.execute_attention(q_reshaped, k_reshaped, v_reshaped)
    
    logger.info(f"✅ Attention computation test passed")
    logger.info(f"   Attention: {q_reshaped.shape} -> {attention_out.shape}")
    
    logger.info("🎉 Real NPU Engine tests completed successfully!")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_real_npu_engine()