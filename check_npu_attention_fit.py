#!/usr/bin/env python3
"""
Check if attention computation fits in NPU's 2GB SRAM
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_attention_memory():
    """Calculate memory requirements for attention on NPU"""
    
    # Gemma 27B dimensions
    batch_size = 1
    seq_len = 2048  # Max sequence length we'd process
    num_heads = 32
    head_dim = 128
    
    # Attention weights (per layer)
    q_weight = 4096 * 5376  # [4096, 5376]
    k_weight = 2048 * 5376  # [2048, 5376]
    v_weight = 2048 * 5376  # [2048, 5376]
    o_weight = 5376 * 4096  # [5376, 4096]
    
    # Total weight size per layer (INT8)
    weight_bytes = (q_weight + k_weight + v_weight + o_weight)
    weight_mb = weight_bytes / (1024 * 1024)
    
    logger.info(f"🧠 NPU Attention Memory Requirements:")
    logger.info(f"   Attention weights per layer: {weight_mb:.1f} MB")
    
    # Intermediate tensors during computation
    # Q, K, V after projection
    qkv_tensors = batch_size * seq_len * (4096 + 2048 + 2048) * 4  # FP32
    qkv_mb = qkv_tensors / (1024 * 1024)
    
    # Attention scores matrix
    scores = batch_size * num_heads * seq_len * seq_len * 4  # FP32
    scores_mb = scores / (1024 * 1024)
    
    # Total per layer
    total_mb = weight_mb + qkv_mb + scores_mb
    
    logger.info(f"   QKV tensors: {qkv_mb:.1f} MB")
    logger.info(f"   Attention scores: {scores_mb:.1f} MB")
    logger.info(f"   Total per layer: {total_mb:.1f} MB")
    
    # NPU SRAM capacity
    npu_sram_mb = 2048  # 2GB
    
    logger.info(f"\n📊 NPU SRAM: {npu_sram_mb} MB")
    logger.info(f"   Can fit: {int(npu_sram_mb / total_mb)} attention layers")
    
    # For streaming one layer at a time
    logger.info(f"\n✅ Single layer attention: {total_mb:.1f} MB")
    if total_mb < npu_sram_mb:
        logger.info(f"   ✅ FITS in NPU SRAM with {npu_sram_mb - total_mb:.1f} MB spare")
        logger.info(f"   💡 Can process attention on NPU layer-by-layer")
    else:
        logger.info(f"   ❌ Does NOT fit in NPU SRAM")
        logger.info(f"   💡 Need to use GPU fallback")
    
    # Check if we can fit weights only
    logger.info(f"\n🔍 Weights-only check:")
    logger.info(f"   Attention weights: {weight_mb:.1f} MB")
    if weight_mb < npu_sram_mb:
        logger.info(f"   ✅ Weights fit with {npu_sram_mb - weight_mb:.1f} MB for activations")
        
        # How much sequence length can we handle?
        remaining_mb = npu_sram_mb - weight_mb
        bytes_per_token = (4096 + 2048 + 2048) * 4  # QKV tensors
        max_tokens = int((remaining_mb * 1024 * 1024) / bytes_per_token)
        logger.info(f"   📏 Max sequence length with weights loaded: ~{max_tokens} tokens")

if __name__ == "__main__":
    calculate_attention_memory()