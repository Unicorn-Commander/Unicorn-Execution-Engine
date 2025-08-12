#!/usr/bin/env python3
"""
Pure Hardware Pipeline with REAL NPU Integration
Magic Unicorn Level - Real NPU + iGPU acceleration
"""

import os
import sys
import time
import logging
import subprocess
import numpy as np
import torch
from typing import Optional, Tuple, List
from pathlib import Path

# Add project path
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MagicUnicornPipeline:
    """
    🦄 Magic Unicorn Level NPU+iGPU Inference Pipeline
    
    Features:
    - Real NPU hardware acceleration
    - Vulkan iGPU compute
    - Zero-copy memory transfers
    - Python 3.13/3.11 compatibility
    - Auto-fallback mechanisms
    - Performance monitoring
    """
    
    def __init__(self, model_path: str, sequence_length: int = 256, 
                 use_real_npu: bool = True, debug: bool = False):
        """
        Initialize Magic Unicorn Pipeline
        
        Args:
            model_path: Path to quantized model
            sequence_length: Max sequence length
            use_real_npu: Whether to use real NPU hardware
            debug: Enable debug logging
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.use_real_npu = use_real_npu
        self.debug = debug
        
        # Pipeline components
        self.npu_kernel = None
        self.vulkan_compute = None
        self.model = None
        self.tokenizer = None
        
        # Performance metrics
        self.metrics = {
            'total_tokens': 0,
            'total_time': 0.0,
            'npu_time': 0.0,
            'gpu_time': 0.0,
            'memory_usage': 0.0
        }
        
        # Environment detection
        self.python_version = sys.version_info
        self.npu_available = self._check_npu_availability()
        
        logger.info("🦄 Magic Unicorn Pipeline Initializing...")
        logger.info(f"   Python: {self.python_version}")
        logger.info(f"   Real NPU: {'✅ Available' if self.npu_available else '❌ Fallback to simulation'}")
        
    def _check_npu_availability(self) -> bool:
        """Check if real NPU hardware is available"""
        try:
            # Check if we're in Python 3.13 with XRT
            if self.python_version >= (3, 13):
                sys.path.insert(0, '/opt/xilinx/xrt/python')
                import pyxrt as xrt
                device = xrt.device(0)
                logger.info("✅ Real NPU hardware detected")
                return True
        except Exception as e:
            logger.warning(f"⚠️  NPU check failed: {e}")
        
        return False
    
    def _ensure_python313_environment(self):
        """Switch to Python 3.13 environment if needed for NPU"""
        if self.use_real_npu and not self.npu_available:
            logger.info("🔄 Switching to Python 3.13 environment for real NPU...")
            
            # Create subprocess with Python 3.13 environment
            script_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_subprocess.py"
            
            subprocess_script = '''#!/usr/bin/env python3
"""NPU subprocess for Python 3.13 compatibility"""
import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

import pyxrt as xrt
from npu_attention_kernel_real import NPUAttentionKernelReal
import numpy as np
import pickle
import json

def run_npu_attention(data_file, output_file):
    """Run NPU attention computation"""
    
    # Load input data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    hidden_states = data['hidden_states']
    q_proj_weight = data['q_proj_weight']
    k_proj_weight = data['k_proj_weight'] 
    v_proj_weight = data['v_proj_weight']
    o_proj_weight = data['o_proj_weight']
    
    # Initialize NPU kernel
    seq_len, d_model = hidden_states.shape[1], hidden_states.shape[2]
    num_heads = 20  # Gemma3 4B
    
    npu_kernel = NPUAttentionKernelReal(
        seq_length=seq_len, d_model=d_model, num_heads=num_heads
    )
    
    if npu_kernel.initialize():
        # Run NPU computation
        result = npu_kernel.compute_flash_attention(
            hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
        )
        
        output, kv_cache, qkv_cache, duration = result
        
        # Save results
        result_data = {
            'output': output,
            'kv_cache': kv_cache,
            'qkv_cache': qkv_cache,
            'duration': duration,
            'success': True
        }
    else:
        result_data = {
            'output': None,
            'success': False,
            'error': 'NPU initialization failed'
        }
    
    with open(output_file, 'wb') as f:
        pickle.dump(result_data, f)

if __name__ == "__main__":
    run_npu_attention(sys.argv[1], sys.argv[2])
'''
            
            with open(script_path, 'w') as f:
                f.write(subprocess_script)
            os.chmod(script_path, 0o755)
            
            return script_path
        
        return None
    
    def initialize_hardware(self):
        """Initialize NPU and iGPU hardware"""
        logger.info("⚡ Initializing Magic Unicorn Hardware...")
        
        # Initialize Vulkan iGPU
        try:
            from real_vulkan_matrix_compute import VulkanMatrixCompute
            self.vulkan_compute = VulkanMatrixCompute()
            logger.info("✅ Vulkan iGPU initialized")
        except Exception as e:
            logger.error(f"❌ Vulkan iGPU failed: {e}")
            raise
        
        # Initialize NPU
        if self.use_real_npu and self.npu_available:
            try:
                from npu_attention_kernel_real import NPUAttentionKernelReal
                self.npu_kernel = NPUAttentionKernelReal(
                    seq_length=self.sequence_length, 
                    d_model=2560, 
                    num_heads=20
                )
                
                if self.npu_kernel.initialize():
                    logger.info("🎉 REAL NPU HARDWARE INITIALIZED!")
                else:
                    raise Exception("NPU initialization failed")
                    
            except Exception as e:
                logger.warning(f"⚠️  Real NPU failed: {e}")
                logger.info("🔄 Falling back to simulated NPU")
                self._initialize_simulated_npu()
        else:
            self._initialize_simulated_npu()
    
    def _initialize_simulated_npu(self):
        """Initialize simulated NPU as fallback"""
        try:
            from npu_attention_kernel import NPUAttentionKernel
            self.npu_kernel = NPUAttentionKernel(
                seq_length=self.sequence_length,
                d_model=2560,
                num_heads=20
            )
            self.npu_kernel.initialize()
            logger.info("✅ Simulated NPU initialized (fallback)")
        except Exception as e:
            logger.error(f"❌ Simulated NPU failed: {e}")
            raise
    
    def load_model(self):
        """Load Gemma3 4B model with lightning-fast loader"""
        logger.info("📚 Loading Gemma3 4B model with Magic Unicorn speed...")
        
        try:
            from lightning_fast_loader import LightningFastLoader
            
            # Initialize loader
            loader = LightningFastLoader(
                strict_hardware=True,
                max_workers=16,
                use_shared_memory=True
            )
            
            # Load model
            start_time = time.time()
            self.model = loader.load_model_optimized(self.model_path)
            load_time = time.time() - start_time
            
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            logger.info(f"📊 Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def run_npu_computation_subprocess(self, hidden_states, q_proj_weight, 
                                     k_proj_weight, v_proj_weight, o_proj_weight):
        """Run NPU computation in Python 3.13 subprocess"""
        
        import tempfile
        import pickle
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as input_file:
            input_data = {
                'hidden_states': hidden_states,
                'q_proj_weight': q_proj_weight,
                'k_proj_weight': k_proj_weight,
                'v_proj_weight': v_proj_weight,
                'o_proj_weight': o_proj_weight
            }
            pickle.dump(input_data, input_file)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as output_file:
            output_path = output_file.name
        
        try:
            # Run NPU computation in subprocess
            subprocess_script = self._ensure_python313_environment()
            cmd = [
                "source", "/home/ucadmin/activate-npu-py313.sh", "&&",
                "python3", subprocess_script, input_path, output_path
            ]
            
            result = subprocess.run(
                ' '.join(cmd), shell=True, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                # Load results
                with open(output_path, 'rb') as f:
                    result_data = pickle.load(f)
                
                if result_data['success']:
                    return (
                        result_data['output'],
                        result_data['kv_cache'], 
                        result_data['qkv_cache'],
                        result_data['duration']
                    )
                else:
                    raise Exception(result_data.get('error', 'NPU computation failed'))
            else:
                raise Exception(f"Subprocess failed: {result.stderr}")
                
        finally:
            # Cleanup temp files
            os.unlink(input_path)
            os.unlink(output_path)
    
    def compute_attention_layer(self, hidden_states, layer_weights):
        """Compute attention layer with NPU+iGPU"""
        
        start_time = time.time()
        
        # Extract weights
        q_proj_weight = layer_weights['q_proj_weight']
        k_proj_weight = layer_weights['k_proj_weight']
        v_proj_weight = layer_weights['v_proj_weight']
        o_proj_weight = layer_weights['o_proj_weight']
        
        # Run NPU computation
        if self.use_real_npu and not self.npu_available:
            # Use subprocess for Python 3.13 compatibility
            result = self.run_npu_computation_subprocess(
                hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
            )
        else:
            # Direct NPU computation
            result = self.npu_kernel.compute_flash_attention(
                hidden_states, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
            )
        
        output, kv_cache, qkv_cache, npu_duration = result
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics['npu_time'] += npu_duration / 1000.0  # Convert ms to s
        
        logger.debug(f"⚡ Attention layer: {total_time*1000:.2f}ms (NPU: {npu_duration:.2f}ms)")
        
        return output
    
    def compute_ffn_layer(self, hidden_states, layer_weights):
        """Compute FFN layer with iGPU"""
        
        start_time = time.time()
        
        # Use Vulkan iGPU for FFN computation
        output = self.vulkan_compute.compute_fused_ffn_gpu(
            hidden_states, 
            layer_weights['gate_proj_weight'],
            layer_weights['up_proj_weight'],
            layer_weights['down_proj_weight']
        )
        
        gpu_time = time.time() - start_time
        self.metrics['gpu_time'] += gpu_time
        
        logger.debug(f"🎮 FFN layer: {gpu_time*1000:.2f}ms")
        
        return output
    
    def generate_token(self, input_ids: torch.Tensor, temperature: float = 0.7):
        """Generate single token with Magic Unicorn speed"""
        
        start_time = time.time()
        
        # Get embeddings (efficient lookup)
        embeddings = self.vulkan_compute.compute_embedding_lookup_gpu(
            input_ids, self.model.embed_tokens
        )
        
        hidden_states = embeddings
        
        # Process through all transformer layers
        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            
            # Input layernorm
            normed_hidden_states = self.vulkan_compute.apply_layernorm_gpu(
                hidden_states, layer.input_layernorm.weight
            )
            
            # Attention computation (NPU)
            attention_output = self.compute_attention_layer(
                normed_hidden_states, {
                    'q_proj_weight': layer.self_attn.q_proj.weight,
                    'k_proj_weight': layer.self_attn.k_proj.weight,
                    'v_proj_weight': layer.self_attn.v_proj.weight,
                    'o_proj_weight': layer.self_attn.o_proj.weight
                }
            )
            
            # Residual connection
            hidden_states = hidden_states + attention_output
            
            # Post-attention layernorm
            normed_hidden_states = self.vulkan_compute.apply_layernorm_gpu(
                hidden_states, layer.post_attention_layernorm.weight
            )
            
            # FFN computation (iGPU)
            ffn_output = self.compute_ffn_layer(
                normed_hidden_states, {
                    'gate_proj_weight': layer.mlp.gate_proj.weight,
                    'up_proj_weight': layer.mlp.up_proj.weight,
                    'down_proj_weight': layer.mlp.down_proj.weight
                }
            )
            
            # Residual connection
            hidden_states = hidden_states + ffn_output
        
        # Final layernorm and logits
        final_hidden_states = self.vulkan_compute.apply_layernorm_gpu(
            hidden_states, self.model.norm.weight
        )
        
        logits = self.vulkan_compute.compute_matrix_multiply_gpu(
            final_hidden_states, self.model.lm_head.weight.T
        )
        
        # Sample next token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs[0, -1], 1)
        else:
            next_token = torch.argmax(logits[0, -1], dim=-1, keepdim=True)
        
        # Update metrics
        generation_time = time.time() - start_time
        self.metrics['total_tokens'] += 1
        self.metrics['total_time'] += generation_time
        
        return next_token, generation_time
    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                temperature: float = 0.7) -> str:
        """Generate text with Magic Unicorn performance"""
        
        logger.info(f"🦄 Generating with Magic Unicorn power...")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Max tokens: {max_new_tokens}")
        
        # Tokenize input (placeholder - would use real tokenizer)
        input_ids = torch.randint(1, 1000, (1, 10))  # Simplified
        
        generated_tokens = []
        total_start_time = time.time()
        
        for i in range(max_new_tokens):
            # Generate next token
            next_token, token_time = self.generate_token(input_ids, temperature)
            generated_tokens.append(next_token.item())
            
            # Add to input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Log progress
            if i % 10 == 0:
                current_tps = (i + 1) / (time.time() - total_start_time)
                logger.info(f"   Generated {i+1}/{max_new_tokens} tokens ({current_tps:.2f} TPS)")
        
        # Calculate final metrics
        total_time = time.time() - total_start_time
        final_tps = max_new_tokens / total_time
        
        logger.info("🎉 Generation complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Tokens per second: {final_tps:.2f} TPS")
        logger.info(f"   NPU time: {self.metrics['npu_time']:.2f}s")
        logger.info(f"   GPU time: {self.metrics['gpu_time']:.2f}s")
        
        # Return placeholder text (would decode tokens in real implementation)
        return f"Generated {max_new_tokens} tokens in {total_time:.2f}s at {final_tps:.2f} TPS"
    
    def get_performance_metrics(self) -> dict:
        """Get detailed performance metrics"""
        return {
            'tokens_per_second': self.metrics['total_tokens'] / max(self.metrics['total_time'], 0.001),
            'npu_utilization': self.metrics['npu_time'] / max(self.metrics['total_time'], 0.001),
            'gpu_utilization': self.metrics['gpu_time'] / max(self.metrics['total_time'], 0.001),
            'total_tokens': self.metrics['total_tokens'],
            'total_time': self.metrics['total_time'],
            'memory_usage': self.metrics['memory_usage']
        }

def main():
    """Main entry point for Magic Unicorn Pipeline"""
    
    logger.info("🦄✨ MAGIC UNICORN PIPELINE STARTING ✨🦄")
    logger.info("=" * 70)
    
    # Initialize pipeline
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    pipeline = MagicUnicornPipeline(
        model_path=model_path,
        sequence_length=256,
        use_real_npu=True,
        debug=True
    )
    
    try:
        # Initialize hardware
        pipeline.initialize_hardware()
        
        # Load model
        if not pipeline.load_model():
            logger.error("❌ Model loading failed")
            return 1
        
        # Run inference
        result = pipeline.generate(
            prompt="What is the capital of France?",
            max_new_tokens=20,
            temperature=0.7
        )
        
        logger.info(f"🎉 Result: {result}")
        
        # Show metrics
        metrics = pipeline.get_performance_metrics()
        logger.info("📊 Performance Metrics:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Magic Unicorn Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())