#!/usr/bin/env python3
"""
Gemma 3n E4B Accelerated Model - Maximum Performance
Real NPU + Vulkan iGPU + CPU orchestration
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nTextAttention, 
    Gemma3nTextMLP,
    Gemma3nAudioAttention,
    Gemma3nAudioConformerAttention
)

from gemma3n_e4b_npu_acceleration import NPUPhoenixAccelerator
from gemma3n_e4b_vulkan_acceleration import VulkanRadeonAccelerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AcceleratedGemma3nTextAttention(nn.Module):
    """Gemma 3n text attention with NPU acceleration"""
    
    def __init__(self, original_attention, npu_accelerator: NPUPhoenixAccelerator):
        super().__init__()
        self.original_attention = original_attention
        self.npu_accelerator = npu_accelerator
        self.use_npu = npu_accelerator.npu_available
        
        logger.info(f"✅ Accelerated attention initialized (NPU: {self.use_npu})")
        
    def __getattr__(self, name):
        """Proxy all attributes to the original attention layer"""
        try:
            original_attention = object.__getattribute__(self, 'original_attention')
            return getattr(original_attention, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value=None,
                output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, ...]:
        
        if self.use_npu and past_key_value is None:  # Only use NPU for prefill
            try:
                return self.forward_npu(hidden_states, attention_mask, position_ids,
                                      past_key_value, output_attentions, use_cache)
            except Exception as e:
                logger.warning(f"NPU attention failed, falling back to CPU: {e}")
                self.use_npu = False
                
        # Fallback to original attention
        return self.original_attention(hidden_states, attention_mask, position_ids,
                                     past_key_value, output_attentions, use_cache)
        
    def forward_npu(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.LongTensor] = None, past_key_value=None,
                   output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, ...]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Get Q, K, V using original projections
        query_states = self.original_attention.q_proj(hidden_states)
        key_states = self.original_attention.k_proj(hidden_states)
        value_states = self.original_attention.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.original_attention.num_heads, 
                                       self.original_attention.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.original_attention.num_key_value_heads, 
                                   self.original_attention.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.original_attention.num_key_value_heads, 
                                       self.original_attention.head_dim).transpose(1, 2)
        
        # Apply rotary position embedding
        if hasattr(self.original_attention, 'rotary_emb'):
            cos, sin = self.original_attention.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = self.original_attention.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)
        
        # Expand key/value states for group query attention
        if self.original_attention.num_key_value_heads != self.original_attention.num_heads:
            key_states = key_states.repeat_interleave(
                self.original_attention.num_heads // self.original_attention.num_key_value_heads, dim=1)
            value_states = value_states.repeat_interleave(
                self.original_attention.num_heads // self.original_attention.num_key_value_heads, dim=1)
        
        # Execute attention on NPU
        logger.debug("⚡ Executing attention on NPU...")
        attn_output = self.npu_accelerator.execute_attention_npu(
            query_states, key_states, value_states, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.original_attention.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, None  # NPU doesn't return attention weights
        else:
            return (attn_output,)

class AcceleratedGemma3nTextMLP(nn.Module):
    """Gemma 3n text MLP with Vulkan acceleration"""
    
    def __init__(self, original_mlp, vulkan_accelerator: VulkanRadeonAccelerator):
        super().__init__()
        self.original_mlp = original_mlp
        self.vulkan_accelerator = vulkan_accelerator
        self.use_vulkan = vulkan_accelerator.vulkan_available
        
        logger.info(f"✅ Accelerated MLP initialized (Vulkan: {self.use_vulkan})")
        
    def __getattr__(self, name):
        """Proxy all attributes to the original MLP layer"""
        try:
            original_mlp = object.__getattribute__(self, 'original_mlp')
            return getattr(original_mlp, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_vulkan:
            try:
                return self.forward_vulkan(x)
            except Exception as e:
                logger.warning(f"Vulkan MLP failed, falling back to CPU: {e}")
                self.use_vulkan = False
                
        # Fallback to original MLP
        return self.original_mlp(x)
        
    def forward_vulkan(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug("⚡ Executing MLP on Vulkan...")
        
        # Get weights from original MLP
        gate_weight = self.original_mlp.gate_proj.weight
        up_weight = self.original_mlp.up_proj.weight
        down_weight = self.original_mlp.down_proj.weight
        
        # Execute FFN on Vulkan
        output = self.vulkan_accelerator.execute_ffn_vulkan(
            x, gate_weight, up_weight, down_weight)
        
        return output

class Gemma3nE4BAcceleratedModel:
    """Gemma 3n E4B model with maximum NPU+Vulkan acceleration"""
    
    def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.npu_accelerator = None
        self.vulkan_accelerator = None
        
        # Performance metrics
        self.performance_metrics = {
            'npu_attention_time': 0.0,
            'vulkan_ffn_time': 0.0,
            'cpu_overhead_time': 0.0,
            'total_inference_time': 0.0,
            'tokens_per_second': 0.0,
            'npu_utilization': 0.0,
            'vulkan_utilization': 0.0
        }
        
        self.initialize_accelerators()
        self.load_model()
        
    def initialize_accelerators(self):
        """Initialize hardware accelerators"""
        logger.info("🚀 Initializing hardware accelerators...")
        
        # Initialize NPU accelerator
        self.npu_accelerator = NPUPhoenixAccelerator()
        
        # Initialize Vulkan accelerator
        self.vulkan_accelerator = VulkanRadeonAccelerator()
        
        # Log acceleration status
        npu_status = "✅ ENABLED" if self.npu_accelerator.npu_available else "❌ DISABLED"
        vulkan_status = "✅ ENABLED" if self.vulkan_accelerator.vulkan_available else "❌ DISABLED"
        
        logger.info(f"🔥 NPU Phoenix acceleration: {npu_status}")
        logger.info(f"🔥 Vulkan iGPU acceleration: {vulkan_status}")
        
    def load_model(self):
        """Load model with acceleration patches"""
        logger.info("📥 Loading Gemma 3n E4B model with acceleration...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Apply acceleration patches
        self.apply_acceleration_patches()
        
        # Optimize accelerators for this model
        self.optimize_accelerators()
        
    def apply_acceleration_patches(self):
        """Apply acceleration patches to model layers"""
        logger.info("🔧 Applying acceleration patches to model layers...")
        
        patched_layers = 0
        
        # Patch attention and MLP layers for Gemma 3n E4B
        for name, module in self.model.named_modules():
            if isinstance(module, (Gemma3nTextAttention, Gemma3nAudioAttention, Gemma3nAudioConformerAttention)):
                # Replace with accelerated attention
                parent = self.get_parent_module(name)
                layer_name = name.split('.')[-1]
                accelerated_attention = AcceleratedGemma3nTextAttention(module, self.npu_accelerator)
                setattr(parent, layer_name, accelerated_attention)
                patched_layers += 1
                logger.info(f"✅ Patched attention layer: {name} ({type(module).__name__})")
                
            elif isinstance(module, Gemma3nTextMLP):
                # Replace with accelerated MLP
                parent = self.get_parent_module(name)
                layer_name = name.split('.')[-1]
                accelerated_mlp = AcceleratedGemma3nTextMLP(module, self.vulkan_accelerator)
                setattr(parent, layer_name, accelerated_mlp)
                patched_layers += 1
                logger.info(f"✅ Patched MLP layer: {name} ({type(module).__name__})")
                
        logger.info(f"✅ Applied acceleration to {patched_layers} layers")
        
    def get_parent_module(self, module_name: str):
        """Get parent module for a given module name"""
        parts = module_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent
        
    def optimize_accelerators(self):
        """Optimize accelerators for Gemma 3n E4B"""
        logger.info("🔧 Optimizing accelerators for Gemma 3n E4B...")
        
        model_config = {
            'hidden_size': 3072,
            'intermediate_size': 8192,
            'num_attention_heads': 24,
            'num_key_value_heads': 8,
            'max_position_embeddings': 8192
        }
        
        # Optimize NPU
        npu_opt = self.npu_accelerator.optimize_for_model(model_config)
        logger.info(f"📊 NPU optimization: {npu_opt}")
        
        # Optimize Vulkan
        vulkan_opt = self.vulkan_accelerator.optimize_for_model(model_config)
        logger.info(f"📊 Vulkan optimization: {vulkan_opt}")
        
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text with maximum acceleration"""
        logger.info(f"🚀 Generating with maximum acceleration...")
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate with acceleration
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode
        generated_tokens = outputs[0][len(input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_generated = len(generated_tokens)
        tps = tokens_generated / total_time if total_time > 0 else 0
        
        # Update performance metrics
        self.performance_metrics['total_inference_time'] = total_time
        self.performance_metrics['tokens_per_second'] = tps
        
        # Get hardware status
        npu_status = self.npu_accelerator.get_npu_status()
        vulkan_status = self.vulkan_accelerator.get_vulkan_status()
        
        return {
            'generated_text': generated_text,
            'tokens_generated': tokens_generated,
            'inference_time': total_time,
            'tokens_per_second': tps,
            'npu_status': npu_status,
            'vulkan_status': vulkan_status,
            'acceleration_active': {
                'npu': self.npu_accelerator.npu_available,
                'vulkan': self.vulkan_accelerator.vulkan_available
            }
        }
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        npu_status = self.npu_accelerator.get_npu_status()
        vulkan_status = self.vulkan_accelerator.get_vulkan_status()
        
        return {
            'model': 'Gemma 3n E4B MatFormer',
            'acceleration': {
                'npu_phoenix': npu_status,
                'vulkan_radeon': vulkan_status
            },
            'performance': self.performance_metrics,
            'hardware_utilization': {
                'npu_utilization': npu_status.get('utilization_percent', 0),
                'vulkan_memory_mb': vulkan_status.get('memory_allocated_mb', 0),
                'cpu_coordinated': True
            }
        }
        
def main():
    """Test accelerated model"""
    logger.info("🦄 Testing Gemma 3n E4B Maximum Acceleration")
    logger.info("=" * 60)
    
    # Initialize accelerated model
    model = Gemma3nE4BAcceleratedModel()
    
    # Test prompts
    test_prompts = [
        "Hello, I'm Aaron. Please tell me about yourself.",
        "What are the advantages of NPU acceleration?",
        "Explain how Vulkan compute shaders work."
    ]
    
    total_tps = []
    
    for prompt in test_prompts:
        logger.info(f"\n🔍 Testing: '{prompt[:30]}...'")
        
        result = model.generate(prompt, max_tokens=50)
        
        logger.info(f"✅ Generated: {result['tokens_generated']} tokens")
        logger.info(f"⚡ Performance: {result['tokens_per_second']:.1f} TPS")
        logger.info(f"💬 Response: {result['generated_text'][:100]}...")
        
        total_tps.append(result['tokens_per_second'])
        
    # Final performance report
    avg_tps = sum(total_tps) / len(total_tps)
    report = model.get_performance_report()
    
    logger.info("\n📊 FINAL PERFORMANCE REPORT:")
    logger.info(f"   Average TPS: {avg_tps:.1f}")
    logger.info(f"   NPU acceleration: {report['acceleration']['npu_phoenix']['available']}")
    logger.info(f"   Vulkan acceleration: {report['acceleration']['vulkan_radeon']['available']}")
    logger.info(f"   Hardware coordination: NPU+Vulkan+CPU")
    
    if avg_tps > 10:
        logger.info("🎉 MAXIMUM ACCELERATION ACHIEVED!")
    else:
        logger.warning("⚠️  Performance optimization needed")
        
    logger.info("=" * 60)
    logger.info("✅ MAXIMUM ACCELERATION TEST COMPLETE")

if __name__ == "__main__":
    main()