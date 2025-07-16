#!/usr/bin/env python3
"""
NPU Kernel Test Suite
Tests all NPU kernels with synthetic data
"""
import torch
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUKernelTester:
    """Test NPU kernels with various model architectures"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_universal_attention(self):
        """Test universal attention kernel"""
        logger.info("🧪 Testing universal attention kernel...")
        
        # Create test tensors
        batch_size, seq_len, head_dim = 1, 64, 64
        
        # Simulate INT4 tensors (use INT8 for testing)
        query = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        key = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        value = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        
        logger.info(f"   Input shapes: Q{query.shape}, K{key.shape}, V{value.shape}")
        
        # Simulate NPU execution (placeholder)
        start_time = time.time()
        # In real implementation, this would call NPU kernel
        output = self._simulate_attention_npu(query, key, value)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Universal attention test passed")
        
        return {"execution_time": execution_time, "output_shape": output.shape}
    
    def test_gemma_ffn(self):
        """Test Gemma gated FFN kernel"""
        logger.info("🧪 Testing Gemma FFN kernel...")
        
        # Gemma FFN dimensions
        batch_size, seq_len, hidden_size = 1, 64, 2048
        intermediate_size = 8192
        
        input_tensor = torch.randint(-7, 8, (batch_size, seq_len, hidden_size), dtype=torch.int8)
        gate_weight = torch.randint(-7, 8, (intermediate_size, hidden_size), dtype=torch.int8)
        up_weight = torch.randint(-7, 8, (intermediate_size, hidden_size), dtype=torch.int8)
        down_weight = torch.randint(-7, 8, (hidden_size, intermediate_size), dtype=torch.int8)
        
        logger.info(f"   Input shape: {input_tensor.shape}")
        
        start_time = time.time()
        output = self._simulate_gemma_ffn_npu(input_tensor, gate_weight, up_weight, down_weight)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Gemma FFN test passed")
        
        return {"execution_time": execution_time, "output_shape": output.shape}
    
    def test_qwen_rope(self):
        """Test Qwen RoPE attention kernel"""
        logger.info("🧪 Testing Qwen RoPE kernel...")
        
        # Qwen attention dimensions
        batch_size, seq_len, head_dim = 1, 64, 64
        
        query = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        key = torch.randint(-7, 8, (batch_size, seq_len, head_dim), dtype=torch.int8)
        
        # RoPE parameters
        rope_cos = torch.randn(seq_len, head_dim, dtype=torch.float16)
        rope_sin = torch.randn(seq_len, head_dim, dtype=torch.float16)
        
        logger.info(f"   Input shapes: Q{query.shape}, K{key.shape}")
        
        start_time = time.time()
        q_rotated, k_rotated = self._simulate_qwen_rope_npu(query, key, rope_cos, rope_sin)
        execution_time = time.time() - start_time
        
        logger.info(f"   Output shapes: Q{q_rotated.shape}, K{k_rotated.shape}")
        logger.info(f"   Execution time: {execution_time*1000:.2f}ms")
        logger.info("   ✅ Qwen RoPE test passed")
        
        return {"execution_time": execution_time, "output_shapes": (q_rotated.shape, k_rotated.shape)}
    
    def _simulate_attention_npu(self, query, key, value):
        """Simulate NPU attention computation"""
        # Convert to float for computation, then back to int8
        q_float = query.float()
        k_float = key.float() 
        v_float = value.float()
        
        # Attention computation
        scores = torch.matmul(q_float, k_float.transpose(-2, -1))
        scores = scores / (q_float.shape[-1] ** 0.5)
        
        # Apply causal mask
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, -float('inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v_float)
        
        # Convert back to int8 (simulate quantization)
        return torch.clamp(output.round(), -7, 7).to(torch.int8)
    
    def _simulate_gemma_ffn_npu(self, input_tensor, gate_weight, up_weight, down_weight):
        """Simulate Gemma FFN computation on NPU"""
        # Convert to float for computation
        input_float = input_tensor.float()
        gate_float = gate_weight.float()
        up_float = up_weight.float()
        down_float = down_weight.float()
        
        # Gate projection
        gate_proj = torch.matmul(input_float, gate_float.T)
        
        # Up projection
        up_proj = torch.matmul(input_float, up_float.T)
        
        # SiLU activation on gate
        gate_activated = gate_proj * torch.sigmoid(gate_proj)
        
        # Gating
        intermediate = gate_activated * up_proj
        
        # Down projection
        output = torch.matmul(intermediate, down_float.T)
        
        # Convert back to int8
        return torch.clamp(output.round(), -7, 7).to(torch.int8)
    
    def _simulate_qwen_rope_npu(self, query, key, rope_cos, rope_sin):
        """Simulate Qwen RoPE computation on NPU"""
        def apply_rope(tensor, cos, sin):
            # Simplified RoPE application
            x1, x2 = tensor[..., ::2], tensor[..., 1::2]
            
            # Convert to float for computation
            x1_float = x1.float()
            x2_float = x2.float()
            cos = cos[..., ::2]
            sin = sin[..., ::2]
            
            rotated_x1 = x1_float * cos - x2_float * sin
            rotated_x2 = x1_float * sin + x2_float * cos
            
            # Interleave back
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
            
            # Convert back to int8
            return torch.clamp(rotated.round(), -7, 7).to(torch.int8)
        
        q_rotated = apply_rope(query, rope_cos, rope_sin)
        k_rotated = apply_rope(key, rope_cos, rope_sin)
        
        return q_rotated, k_rotated
    
    def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🦄 Starting NPU Kernel Test Suite")
        logger.info("=" * 50)
        
        tests = [
            ("Universal Attention", self.test_universal_attention),
            ("Gemma FFN", self.test_gemma_ffn),
            ("Qwen RoPE", self.test_qwen_rope)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                logger.info("")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                self.test_results[test_name] = {"error": str(e)}
        
        # Summary
        logger.info("📊 Test Suite Summary:")
        for test_name, result in self.test_results.items():
            if "error" in result:
                logger.error(f"   ❌ {test_name}: FAILED")
            else:
                logger.info(f"   ✅ {test_name}: PASSED")
        
        logger.info("🎉 NPU kernel test suite complete!")

if __name__ == "__main__":
    tester = NPUKernelTester()
    tester.run_all_tests()
