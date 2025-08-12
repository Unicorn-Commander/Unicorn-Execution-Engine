#!/usr/bin/env python3
"""
Qwen3-30B-A3B MoE Pipeline for Unicorn Execution Engine
Implements MoE routing and computation with Vulkan acceleration
"""

import numpy as np
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing Unicorn infrastructure
from vulkan_compute_workaround import VulkanComputeWorkaround
from dynamic_quantization_engine import DynamicQuantizationEngine
from qwen3_moe_loader import Qwen3MoELoader

logger = logging.getLogger(__name__)

class Qwen3MoEPipeline:
    """MoE inference pipeline with NPU routing and GPU expert computation"""
    
    def __init__(self, model_path: str = None):
        self.compute_engine = VulkanComputeWorkaround()
        self.quantization_engine = DynamicQuantizationEngine()
        self.loader = Qwen3MoELoader()
        
        # Model weights and configuration
        self.model_weights = {}
        self.model_config = {}
        self.router_cache = {}
        self.expert_cache = {}
        
        # MoE parameters
        self.num_experts = 128
        self.top_k = 8
        self.hidden_size = 4096
        self.intermediate_size = 22016
        
        # Performance tracking
        self.inference_stats = {
            'tokens_generated': 0,
            'total_time': 0.0,
            'routing_time': 0.0,
            'expert_time': 0.0,
            'memory_usage': 0
        }
        
        logger.info("🦄 Qwen3 MoE Pipeline initialized")
        logger.info(f"   Compute engine: {'Vulkan' if self.compute_engine.use_vulkan else 'NumPy'}")
        logger.info(f"   Target performance: 40-50 TPS")

    def initialize(self, model_path: str) -> bool:
        """Initialize the MoE pipeline with model weights"""
        try:
            logger.info(f"🚀 Initializing Qwen3 MoE pipeline from {model_path}")
            
            # Load model weights with MoE optimization
            model_data = self.loader.load_qwen3_moe_model(model_path, load_mode='sparse')
            self.model_weights = model_data['weights']
            self.model_config = model_data['config']
            
            # Update configuration
            self.num_experts = self.model_config.get('num_experts', 128)
            self.top_k = self.model_config.get('top_k', 8)
            self.hidden_size = self.model_config.get('hidden_size', 4096)
            self.intermediate_size = self.model_config.get('intermediate_size', 22016)
            
            # Pre-compile routing and expert computation
            self._initialize_routing_system()
            self._initialize_expert_computation()
            
            logger.info("✅ Qwen3 MoE pipeline initialized successfully")
            self._log_initialization_stats()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MoE pipeline: {e}")
            return False

    def generate(self, input_text: str, max_tokens: int = 100) -> str:
        """Generate text using MoE pipeline"""
        start_time = time.time()
        
        try:
            # Tokenize input (placeholder - would use actual tokenizer)
            tokens = self._tokenize_input(input_text)
            
            generated_tokens = []
            hidden_states = self._embed_tokens(tokens)
            
            for i in range(max_tokens):
                # Forward pass through MoE layers
                hidden_states = self._forward_pass_moe(hidden_states)
                
                # Generate next token
                next_token = self._sample_next_token(hidden_states)
                generated_tokens.append(next_token)
                
                # Update hidden states for next iteration
                hidden_states = self._update_hidden_states(hidden_states, next_token)
                
                # Check for EOS token
                if next_token == self._get_eos_token():
                    break
            
            # Detokenize output
            generated_text = self._detokenize_tokens(generated_tokens)
            
            # Update performance statistics
            inference_time = time.time() - start_time
            self._update_performance_stats(len(generated_tokens), inference_time)
            
            logger.info(f"🎯 Generated {len(generated_tokens)} tokens in {inference_time:.2f}s")
            logger.info(f"   Current TPS: {len(generated_tokens)/inference_time:.1f}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"

    def _forward_pass_moe(self, hidden_states: np.ndarray) -> np.ndarray:
        """Forward pass through MoE transformer layer"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply layer normalization
        normalized_states = self._apply_layer_norm(hidden_states)
        
        # MoE routing and expert selection
        routing_start = time.time()
        expert_weights, selected_experts = self._route_to_experts(normalized_states)
        self.inference_stats['routing_time'] += time.time() - routing_start
        
        # Expert computation
        expert_start = time.time()
        expert_outputs = self._compute_expert_outputs(
            normalized_states, selected_experts, expert_weights
        )
        self.inference_stats['expert_time'] += time.time() - expert_start
        
        # Combine expert outputs
        combined_output = self._combine_expert_outputs(expert_outputs, expert_weights)
        
        # Residual connection
        output = hidden_states + combined_output
        
        return output

    def _route_to_experts(self, hidden_states: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Route tokens to appropriate experts using router network"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get router weights (kept at FP16 for precision)
        router_weights = self._get_router_weights()
        
        # Compute routing logits
        # Shape: [batch_size, seq_len, num_experts]
        routing_logits = self.compute_engine.matrix_multiply(
            hidden_states.reshape(-1, hidden_dim),
            router_weights
        ).reshape(batch_size, seq_len, self.num_experts)
        
        # Apply top-k selection
        top_k_indices = np.argpartition(routing_logits, -self.top_k, axis=-1)[..., -self.top_k:]
        top_k_values = np.take_along_axis(routing_logits, top_k_indices, axis=-1)
        
        # Apply softmax to get expert weights
        expert_weights = self._softmax(top_k_values)
        
        # Get unique experts for this batch
        selected_experts = np.unique(top_k_indices.flatten()).tolist()
        
        return expert_weights, selected_experts

    def _compute_expert_outputs(self, hidden_states: np.ndarray, 
                              selected_experts: List[int], 
                              expert_weights: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute outputs from selected experts using Vulkan acceleration"""
        expert_outputs = {}
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        for expert_idx in selected_experts:
            # Get expert weights (quantized INT4)
            gate_weights = self._get_expert_weights(expert_idx, 'gate_proj')
            up_weights = self._get_expert_weights(expert_idx, 'up_proj') 
            down_weights = self._get_expert_weights(expert_idx, 'down_proj')
            
            # Dequantize weights for computation
            gate_weights = self._dequantize_int4(gate_weights)
            up_weights = self._dequantize_int4(up_weights)
            down_weights = self._dequantize_int4(down_weights)
            
            # FFN computation: gate * silu(up(x)) @ down
            input_flat = hidden_states.reshape(-1, hidden_dim)
            
            # Gate projection
            gate_output = self.compute_engine.matrix_multiply(input_flat, gate_weights)
            
            # Up projection with SiLU activation
            up_output = self.compute_engine.matrix_multiply(input_flat, up_weights)
            up_output = self._apply_silu(up_output)
            
            # Element-wise multiplication
            gated_output = gate_output * up_output
            
            # Down projection
            final_output = self.compute_engine.matrix_multiply(gated_output, down_weights)
            
            expert_outputs[expert_idx] = final_output.reshape(batch_size, seq_len, hidden_dim)
        
        return expert_outputs

    def _combine_expert_outputs(self, expert_outputs: Dict[int, np.ndarray], 
                               expert_weights: np.ndarray) -> np.ndarray:
        """Combine expert outputs using routing weights"""
        batch_size, seq_len, hidden_dim = next(iter(expert_outputs.values())).shape
        combined = np.zeros((batch_size, seq_len, hidden_dim), dtype=np.float32)
        
        # This is a simplified combination - real implementation would be more complex
        for expert_idx, output in expert_outputs.items():
            # Weight by expert importance (simplified)
            weight = np.mean(expert_weights)  # Would use actual routing weights
            combined += output * weight
            
        return combined

    def _get_router_weights(self) -> np.ndarray:
        """Get router weights for expert selection"""
        # Look for router weights in model
        for key, weight_info in self.model_weights.items():
            if 'router' in key or ('gate' in key and 'expert' not in key):
                if isinstance(weight_info, dict) and 'data' in weight_info:
                    return weight_info['data']
        
        # Fallback: create dummy router weights
        logger.warning("Router weights not found, using placeholder")
        return np.random.randn(self.hidden_size, self.num_experts).astype(np.float16)

    def _get_expert_weights(self, expert_idx: int, layer_type: str) -> Dict[str, Any]:
        """Get quantized weights for a specific expert layer"""
        key = f"expert_{expert_idx}.{layer_type}"
        if key in self.model_weights:
            return self.model_weights[key]
        
        # Fallback: create dummy expert weights
        logger.warning(f"Expert weights not found for {key}, using placeholder")
        if layer_type == 'gate_proj':
            shape = (self.hidden_size, self.intermediate_size)
        elif layer_type == 'up_proj':
            shape = (self.hidden_size, self.intermediate_size)
        else:  # down_proj
            shape = (self.intermediate_size, self.hidden_size)
            
        dummy_weights = np.random.randint(-8, 8, shape).astype(np.int8)
        dummy_scales = np.ones((shape[0] // 16,), dtype=np.float32)
        
        return {
            'data': dummy_weights,
            'scales': dummy_scales,
            'dtype': 'int4',
            'original_shape': shape
        }

    def _dequantize_int4(self, weight_info: Dict[str, Any]) -> np.ndarray:
        """Dequantize INT4 weights for computation"""
        if weight_info.get('dtype') == 'int4':
            return self.quantization_engine.dequantize_int4(
                weight_info['data'], weight_info['scales']
            )
        else:
            return weight_info['data'].astype(np.float32)

    def _initialize_routing_system(self):
        """Initialize the MoE routing system"""
        logger.info("🎯 Initializing MoE routing system")
        # Pre-compile routing computations, cache frequently used experts
        pass

    def _initialize_expert_computation(self):
        """Initialize expert computation system"""
        logger.info("⚡ Initializing expert computation")
        # Pre-allocate buffers for expert computation
        pass

    def _tokenize_input(self, text: str) -> List[int]:
        """Tokenize input text (placeholder)"""
        # This would use actual tokenizer
        return list(range(len(text.split())))

    def _detokenize_tokens(self, tokens: List[int]) -> str:
        """Detokenize tokens to text (placeholder)"""
        # This would use actual detokenizer
        return f"Generated text from {len(tokens)} tokens"

    def _embed_tokens(self, tokens: List[int]) -> np.ndarray:
        """Convert tokens to embeddings"""
        # Placeholder embedding
        return np.random.randn(1, len(tokens), self.hidden_size).astype(np.float32)

    def _sample_next_token(self, hidden_states: np.ndarray) -> int:
        """Sample next token from model output"""
        # Placeholder sampling
        return np.random.randint(0, 50000)

    def _update_hidden_states(self, hidden_states: np.ndarray, next_token: int) -> np.ndarray:
        """Update hidden states with new token"""
        # Placeholder update
        return hidden_states

    def _get_eos_token(self) -> int:
        """Get end-of-sequence token"""
        return 2  # Common EOS token

    def _apply_layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)

    def _apply_silu(self, x: np.ndarray) -> np.ndarray:
        """Apply SiLU activation function"""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _update_performance_stats(self, tokens_generated: int, inference_time: float):
        """Update performance tracking statistics"""
        self.inference_stats['tokens_generated'] += tokens_generated
        self.inference_stats['total_time'] += inference_time
        
    def _log_initialization_stats(self):
        """Log initialization statistics"""
        total_weights = len(self.model_weights)
        logger.info(f"📊 MoE Pipeline Statistics:")
        logger.info(f"   Loaded weight groups: {total_weights}")
        logger.info(f"   Active experts: {self.top_k}/{self.num_experts}")
        logger.info(f"   Hidden size: {self.hidden_size}")
        logger.info(f"   Intermediate size: {self.intermediate_size}")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        total_time = self.inference_stats['total_time']
        total_tokens = self.inference_stats['tokens_generated']
        
        return {
            'total_tokens': total_tokens,
            'total_time': total_time,
            'average_tps': total_tokens / total_time if total_time > 0 else 0,
            'routing_time_percent': (self.inference_stats['routing_time'] / total_time * 100) if total_time > 0 else 0,
            'expert_time_percent': (self.inference_stats['expert_time'] / total_time * 100) if total_time > 0 else 0
        }

def moe_routing_and_computation(model_weights, input_data, active_experts=8):
    """
    Legacy function for GPT5 compatibility
    Manage MoE routing and computation using Vulkan acceleration
    """
    compute_engine = VulkanComputeWorkaround()
    
    # Use Vulkan if available, fallback to NumPy
    if compute_engine.use_vulkan:
        result = compute_engine.matrix_multiply(model_weights.T, input_data)
    else:
        result = np.dot(model_weights.T, input_data)
    
    return result

if __name__ == "__main__":
    # Test the MoE pipeline
    pipeline = Qwen3MoEPipeline()
    
    # Test with dummy data for now
    print("🧪 Testing Qwen3 MoE Pipeline")
    
    # Test legacy function for GPT5 compatibility
    model_weights = np.random.randint(-8, 8, (512, 8)).astype(np.float32)
    input_data = np.random.randn(8, 512).astype(np.float32)
    
    result = moe_routing_and_computation(model_weights, input_data)
    print(f"✅ Legacy function test: {result.shape}")
    print(f"   Sample output: {result[:3, :3]}")
    
    # Test pipeline initialization (will fail without model)
    model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/models/qwen3-30b-a3b"
    
    if pipeline.initialize(model_path):
        # Test generation
        output = pipeline.generate("Hello, how are you?", max_tokens=10)
        print(f"✅ Generated output: {output}")
        
        # Show performance stats
        stats = pipeline.get_performance_stats()
        print(f"📊 Performance: {stats['average_tps']:.1f} TPS")
    else:
        print("ℹ️  Pipeline initialization skipped (model not available)")
        print("   This is expected until Qwen3 model is downloaded")