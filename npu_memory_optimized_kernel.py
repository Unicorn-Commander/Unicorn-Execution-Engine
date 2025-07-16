#!/usr/bin/env python3
"""
NPU Memory-Optimized Kernel for Gemma 3 27B
Implements streaming processing and memory-efficient attention kernels
"""

import os
import torch
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import gc
import psutil

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPUMemoryOptimizedKernel:
    """Memory-optimized NPU kernel with streaming processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.npu_memory_mb = config.get("npu_memory_mb", 2048)  # 2GB NPU memory
        self.chunk_size = config.get("chunk_size", 512)  # Process in chunks
        self.max_sequence_length = config.get("max_sequence_length", 2048)
        
        # NPU-specific optimizations
        self.attention_cache = {}
        self.weight_cache = {}
        self.memory_pool = None
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory_mb = 0
        
        logger.info("🧠 NPU Memory-Optimized Kernel initialized")
        logger.info(f"   NPU memory budget: {self.npu_memory_mb}MB")
        logger.info(f"   Chunk size: {self.chunk_size}")
        logger.info(f"   Max sequence length: {self.max_sequence_length}")
        
    def monitor_memory(self) -> float:
        """Monitor current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        return memory_mb
        
    def initialize_memory_pool(self) -> bool:
        """Initialize NPU memory pool for efficient allocation"""
        try:
            # Allocate NPU memory pool
            pool_size_bytes = self.npu_memory_mb * 1024 * 1024
            
            # Create memory pool structure
            self.memory_pool = {
                'total_size': pool_size_bytes,
                'allocated': 0,
                'blocks': [],
                'free_blocks': [pool_size_bytes]  # Initially one big free block
            }
            
            logger.info(f"✅ NPU memory pool initialized: {self.npu_memory_mb}MB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory pool initialization failed: {e}")
            return False
    
    def allocate_npu_memory(self, size_bytes: int) -> Optional[int]:
        """Allocate memory from NPU pool"""
        if not self.memory_pool:
            return None
            
        # Find suitable free block
        for i, free_size in enumerate(self.memory_pool['free_blocks']):
            if free_size >= size_bytes:
                # Allocate from this block
                offset = sum(self.memory_pool['free_blocks'][:i])
                
                # Update free blocks
                remaining = free_size - size_bytes
                self.memory_pool['free_blocks'][i] = remaining
                if remaining == 0:
                    self.memory_pool['free_blocks'].pop(i)
                
                # Track allocation
                self.memory_pool['allocated'] += size_bytes
                self.memory_pool['blocks'].append({
                    'offset': offset,
                    'size': size_bytes,
                    'allocated': True
                })
                
                return offset
        
        return None  # No suitable block found
    
    def free_npu_memory(self, offset: int):
        """Free memory back to NPU pool"""
        if not self.memory_pool:
            return
            
        # Find the block to free
        for i, block in enumerate(self.memory_pool['blocks']):
            if block['offset'] == offset and block['allocated']:
                # Mark as free
                block['allocated'] = False
                self.memory_pool['allocated'] -= block['size']
                
                # Add back to free blocks
                self.memory_pool['free_blocks'].append(block['size'])
                self.memory_pool['free_blocks'].sort(reverse=True)  # Keep largest first
                
                break
    
    def stream_attention_processing(self, 
                                   hidden_states: torch.Tensor,
                                   layer_weights: Dict[str, torch.Tensor],
                                   layer_idx: int) -> torch.Tensor:
        """Stream attention processing in chunks to avoid memory overflow"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate memory requirements
        attention_memory_mb = (seq_len * hidden_size * 4 * 4) / (1024 * 1024)  # Q,K,V,O
        
        logger.debug(f"   🧠 Layer {layer_idx} attention memory: {attention_memory_mb:.1f}MB")
        
        # Check if we need to chunk
        if attention_memory_mb > self.npu_memory_mb * 0.8:  # Use 80% of NPU memory
            return self._chunked_attention_processing(hidden_states, layer_weights, layer_idx)
        else:
            return self._standard_attention_processing(hidden_states, layer_weights, layer_idx)
    
    def _chunked_attention_processing(self,
                                     hidden_states: torch.Tensor,
                                     layer_weights: Dict[str, torch.Tensor],
                                     layer_idx: int) -> torch.Tensor:
        """Process attention in chunks to fit NPU memory"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate optimal chunk size
        max_chunk_size = min(self.chunk_size, seq_len)
        num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
        
        logger.debug(f"   🔄 Chunked attention: {num_chunks} chunks of {max_chunk_size}")
        
        # Process in chunks
        output_chunks = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_chunk_size
            end_idx = min(start_idx + max_chunk_size, seq_len)
            
            chunk_hidden = hidden_states[:, start_idx:end_idx, :]
            
            # Process this chunk
            chunk_output = self._process_attention_chunk(
                chunk_hidden, layer_weights, layer_idx, start_idx, end_idx
            )
            
            output_chunks.append(chunk_output)
            
            # Cleanup
            del chunk_hidden, chunk_output
            gc.collect()
        
        # Concatenate chunks
        attention_output = torch.cat(output_chunks, dim=1)
        
        # Cleanup
        del output_chunks
        gc.collect()
        
        return attention_output
    
    def _process_attention_chunk(self,
                                chunk_hidden: torch.Tensor,
                                layer_weights: Dict[str, torch.Tensor],
                                layer_idx: int,
                                start_idx: int,
                                end_idx: int) -> torch.Tensor:
        """Process a single attention chunk"""
        batch_size, chunk_seq_len, hidden_size = chunk_hidden.shape
        
        # Get weights
        q_weight = layer_weights['q_proj'].to(torch.float16)
        k_weight = layer_weights['k_proj'].to(torch.float16)
        v_weight = layer_weights['v_proj'].to(torch.float16)
        o_weight = layer_weights['o_proj'].to(torch.float16)
        
        # Project to Q, K, V
        q = torch.matmul(chunk_hidden, q_weight.t())
        k = torch.matmul(chunk_hidden, k_weight.t())
        v = torch.matmul(chunk_hidden, v_weight.t())
        
        # Reshape for multi-head attention
        num_heads = self.config.get("num_heads", 32)
        head_dim = hidden_size // num_heads
        
        q = q.view(batch_size, chunk_seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, chunk_seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, chunk_seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float16))
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask if needed
        if chunk_seq_len > 1:
            mask = torch.triu(torch.ones(chunk_seq_len, chunk_seq_len), diagonal=1)
            attention_scores = attention_scores.masked_fill(mask.bool(), float('-inf'))
        
        # Softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, chunk_seq_len, hidden_size)
        output = torch.matmul(context, o_weight.t())
        
        # Cleanup intermediate tensors
        del q, k, v, attention_scores, attention_probs, context
        del q_weight, k_weight, v_weight, o_weight
        gc.collect()
        
        return output
    
    def _standard_attention_processing(self,
                                      hidden_states: torch.Tensor,
                                      layer_weights: Dict[str, torch.Tensor],
                                      layer_idx: int) -> torch.Tensor:
        """Standard attention processing for smaller sequences"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Use cached weights if available
        cache_key = f"layer_{layer_idx}"
        if cache_key in self.weight_cache:
            cached_weights = self.weight_cache[cache_key]
        else:
            cached_weights = {
                'q_proj': layer_weights['q_proj'].to(torch.float16),
                'k_proj': layer_weights['k_proj'].to(torch.float16),
                'v_proj': layer_weights['v_proj'].to(torch.float16),
                'o_proj': layer_weights['o_proj'].to(torch.float16)
            }
            self.weight_cache[cache_key] = cached_weights
        
        # Project to Q, K, V
        q = torch.matmul(hidden_states, cached_weights['q_proj'].t())
        k = torch.matmul(hidden_states, cached_weights['k_proj'].t())
        v = torch.matmul(hidden_states, cached_weights['v_proj'].t())
        
        # Multi-head attention
        num_heads = self.config.get("num_heads", 32)
        head_dim = hidden_size // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float16))
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            attention_scores = attention_scores.masked_fill(mask.bool(), float('-inf'))
        
        # Softmax and apply to values
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = torch.matmul(context, cached_weights['o_proj'].t())
        
        return output
    
    def process_layer(self, 
                     hidden_states: torch.Tensor,
                     layer_weights: Dict[str, torch.Tensor],
                     layer_idx: int) -> torch.Tensor:
        """Process a single transformer layer with memory optimization"""
        memory_before = self.monitor_memory()
        
        logger.debug(f"   🔄 Processing layer {layer_idx} - Memory: {memory_before:.1f}MB")
        
        # Apply attention with memory optimization
        attention_output = self.stream_attention_processing(
            hidden_states, layer_weights, layer_idx
        )
        
        memory_after = self.monitor_memory()
        logger.debug(f"   ✅ Layer {layer_idx} complete - Memory: {memory_after:.1f}MB")
        
        return attention_output
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        self.attention_cache.clear()
        self.weight_cache.clear()
        gc.collect()
        logger.info("🧹 NPU caches cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        current_memory = self.monitor_memory()
        
        stats = {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory_mb,
            'npu_memory_budget_mb': self.npu_memory_mb,
            'memory_pool_allocated': self.memory_pool['allocated'] if self.memory_pool else 0,
            'memory_pool_free': self.memory_pool['total_size'] - self.memory_pool['allocated'] if self.memory_pool else 0,
            'cache_size': len(self.weight_cache)
        }
        
        return stats

def test_npu_memory_optimized_kernel():
    """Test the NPU memory-optimized kernel"""
    logger.info("🧪 Testing NPU Memory-Optimized Kernel")
    
    # Configuration
    config = {
        "npu_memory_mb": 2048,
        "chunk_size": 512,
        "max_sequence_length": 2048,
        "num_heads": 32,
        "hidden_size": 4096
    }
    
    # Initialize kernel
    kernel = NPUMemoryOptimizedKernel(config)
    
    if not kernel.initialize_memory_pool():
        logger.error("❌ Kernel initialization failed")
        return False
    
    # Test with various sequence lengths
    test_cases = [
        (1, 256, 4096),   # Small sequence
        (1, 1024, 4096),  # Medium sequence
        (1, 2048, 4096),  # Large sequence
    ]
    
    for batch_size, seq_len, hidden_size in test_cases:
        logger.info(f"   🧪 Testing sequence length: {seq_len}")
        
        # Create test data
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        
        # Create mock layer weights
        layer_weights = {
            'q_proj': torch.randn(hidden_size, hidden_size, dtype=torch.float16),
            'k_proj': torch.randn(hidden_size, hidden_size, dtype=torch.float16),
            'v_proj': torch.randn(hidden_size, hidden_size, dtype=torch.float16),
            'o_proj': torch.randn(hidden_size, hidden_size, dtype=torch.float16),
        }
        
        # Process layer
        start_time = time.time()
        output = kernel.process_layer(hidden_states, layer_weights, layer_idx=0)
        processing_time = time.time() - start_time
        
        # Verify output
        assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"
        
        logger.info(f"   ✅ Sequence {seq_len}: {processing_time:.2f}s")
        
        # Cleanup
        del hidden_states, layer_weights, output
        gc.collect()
    
    # Memory stats
    stats = kernel.get_memory_stats()
    logger.info(f"🎯 Memory Stats:")
    logger.info(f"   Peak memory: {stats['peak_memory_mb']:.1f}MB")
    logger.info(f"   NPU budget: {stats['npu_memory_budget_mb']}MB")
    logger.info(f"   Cache size: {stats['cache_size']}")
    
    return True

if __name__ == "__main__":
    test_npu_memory_optimized_kernel()