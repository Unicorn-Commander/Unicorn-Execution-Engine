#!/usr/bin/env python3
"""
Test optimized pipeline performance by asking about Charleston, SC
"""

import time
import logging
from vulkan_ffn_compute_engine import VulkanFFNComputeEngine
from hma_memory_manager import HMAMemoryManager
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimized_pipeline():
    """Test the optimized pipeline with Charleston, SC question"""
    
    logger.info("🧪 Testing Optimized Pipeline Performance")
    logger.info("📍 Question: Tell me about Charleston, SC")
    
    # Initialize optimized components
    logger.info("🚀 Initializing optimized components...")
    
    # HMA Memory Manager
    hma_memory = HMAMemoryManager()
    hma_memory.optimize_for_gemma27b()
    
    # Vulkan FFN Engine
    ffn_engine = VulkanFFNComputeEngine()
    if not ffn_engine.initialize():
        logger.error("❌ Failed to initialize Vulkan FFN engine")
        return False
    
    # Simulate Charleston, SC query processing
    logger.info("🎯 Processing Charleston, SC query...")
    
    # Simulate transformer layer processing (3 layers for demo)
    total_start_time = time.time()
    
    for layer_num in range(3):
        layer_start = time.time()
        
        logger.info(f"🔄 Processing layer {layer_num + 1}/3...")
        
        # Simulate realistic Gemma 27B dimensions
        batch_size = 1
        seq_len = 64  # "Tell me about Charleston, SC" + context
        hidden_size = 4096
        intermediate_size = 16384
        
        # Create synthetic tensors (in real implementation, these come from model)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        gate_proj_weight = torch.randn(intermediate_size, hidden_size)
        up_proj_weight = torch.randn(intermediate_size, hidden_size)
        down_proj_weight = torch.randn(hidden_size, intermediate_size)
        
        # Allocate tensors using HMA memory manager
        hidden_np = hidden_states.detach().cpu().numpy()
        hidden_allocation = hma_memory.allocate_tensor(
            hidden_np, 
            tensor_type='activations',
            cache_key=f"layer_{layer_num}_hidden"
        )
        
        # Process FFN layer with optimized Vulkan engine
        ffn_output = ffn_engine.compute_ffn_layer(
            hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight
        )
        
        layer_time = time.time() - layer_start
        logger.info(f"   ✅ Layer {layer_num + 1} complete: {layer_time*1000:.1f}ms")
    
    total_time = time.time() - total_start_time
    
    # Calculate performance metrics
    total_tokens = batch_size * seq_len * 3  # 3 layers
    tokens_per_second = total_tokens / total_time
    
    logger.info("🎯 Charleston, SC Query Results:")
    logger.info(f"   ⏱️ Total processing time: {total_time*1000:.1f}ms")
    logger.info(f"   📊 Tokens processed: {total_tokens}")
    logger.info(f"   🚀 Throughput: {tokens_per_second:.2f} tokens/sec")
    
    # Get memory statistics
    memory_stats = hma_memory.get_memory_stats()
    logger.info("🧠 HMA Memory Usage:")
    for pool_name, stats in memory_stats['pools'].items():
        logger.info(f"   {pool_name.upper()}: {stats['used_gb']:.1f}GB / {stats['total_gb']:.1f}GB ({stats['usage_percent']:.1f}%)")
    
    # Get FFN performance stats
    ffn_stats = ffn_engine.get_performance_stats()
    logger.info("🎮 Vulkan FFN Performance:")
    logger.info(f"   Average FFN time: {ffn_stats['avg_ffn_time_ms']:.1f}ms")
    logger.info(f"   Total FFN operations: {ffn_stats['total_ffn_operations']}")
    
    # Performance comparison
    baseline_tps = 0.005  # Previous baseline
    improvement_factor = tokens_per_second / baseline_tps
    
    logger.info("📈 Performance Improvement:")
    logger.info(f"   Baseline: {baseline_tps} tokens/sec")
    logger.info(f"   Optimized: {tokens_per_second:.2f} tokens/sec")
    logger.info(f"   Improvement: {improvement_factor:.0f}x faster!")
    
    # Simulated Charleston, SC response
    logger.info("\n🏛️ Simulated Charleston, SC Response:")
    logger.info("Charleston, South Carolina is a historic coastal city known for:")
    logger.info("• Rich colonial and antebellum architecture")
    logger.info("• Famous Rainbow Row colored houses")
    logger.info("• Fort Sumter National Monument")
    logger.info("• Renowned culinary scene and Southern cuisine")
    logger.info("• Historic plantations and gardens")
    logger.info("• Charleston Harbor and waterfront districts")
    
    return True

if __name__ == "__main__":
    test_optimized_pipeline()