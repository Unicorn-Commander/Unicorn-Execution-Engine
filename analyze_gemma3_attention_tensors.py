#!/usr/bin/env python3
"""
Analyze Gemma 3 27B Attention Tensor Shapes for Custom NPU Kernel Design
"""

import os
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_attention_tensors():
    """Analyze Gemma 3 27B attention tensor shapes and quantization requirements"""
    
    quantized_path = Path("./quantized_models/gemma-3-27b-it-layer-by-layer")
    
    if not quantized_path.exists():
        logger.error(f"❌ Quantized model not found at {quantized_path}")
        return None
    
    logger.info("🔍 Analyzing Gemma 3 27B Attention Tensor Shapes...")
    
    # Find a layer file to examine
    layer_files = list(quantized_path.glob("*layer_0.safetensors"))
    if not layer_files:
        layer_files = list(quantized_path.glob("*layer_*.safetensors"))
    
    if not layer_files:
        logger.error("❌ No layer files found")
        return None
    
    layer_file = layer_files[0]
    logger.info(f"📁 Examining layer file: {layer_file.name}")
    
    attention_analysis = {}
    
    try:
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            logger.info("\n📊 TENSOR INVENTORY:")
            
            all_keys = list(f.keys())
            attention_keys = [k for k in all_keys if any(proj in k for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]
            
            for key in attention_keys:
                tensor = f.get_tensor(key)
                
                # Try to get scale
                scale_key = f"{key}_scale"
                scale = None
                if scale_key in all_keys:
                    scale = f.get_tensor(scale_key)
                
                # Get metadata
                metadata = f.metadata()
                scheme = metadata.get(key, 'unknown')
                
                attention_analysis[key] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
                    'quantization_scheme': scheme,
                    'scale_shape': list(scale.shape) if scale is not None else None,
                    'scale_dtype': str(scale.dtype) if scale is not None else None
                }
                
                logger.info(f"  🎯 {key}:")
                logger.info(f"     Shape: {tensor.shape}")
                logger.info(f"     Dtype: {tensor.dtype}")
                logger.info(f"     Size: {tensor.numel() * tensor.element_size() / (1024 * 1024):.2f} MB")
                logger.info(f"     Quantization: {scheme}")
                if scale is not None:
                    logger.info(f"     Scale shape: {scale.shape}")
                    logger.info(f"     Scale dtype: {scale.dtype}")
                logger.info("")
        
        # Analyze architecture implications
        logger.info("🧠 ATTENTION ARCHITECTURE ANALYSIS:")
        
        # Look for Q projection to determine dimensions
        q_proj_key = None
        for key in attention_analysis:
            if 'q_proj' in key:
                q_proj_key = key
                break
        
        if q_proj_key:
            q_shape = attention_analysis[q_proj_key]['shape']
            logger.info(f"📐 Q Projection Shape: {q_shape}")
            
            if len(q_shape) == 2:
                hidden_size = q_shape[1]  # Input hidden size
                num_heads_x_head_dim = q_shape[0]  # Output size
                
                # For Gemma 3 27B: typically 32 attention heads
                num_heads = 32
                head_dim = num_heads_x_head_dim // num_heads
                
                logger.info(f"🔢 Inferred Architecture:")
                logger.info(f"   Hidden Size: {hidden_size}")
                logger.info(f"   Number of Heads: {num_heads}")
                logger.info(f"   Head Dimension: {head_dim}")
                logger.info(f"   Total Attention Output: {num_heads_x_head_dim}")
                
                # Calculate NPU kernel requirements
                logger.info("\n⚡ NPU KERNEL REQUIREMENTS:")
                logger.info(f"   Q Projection: {hidden_size} → {num_heads_x_head_dim} (matrix: {hidden_size}x{num_heads_x_head_dim})")
                logger.info(f"   K Projection: {hidden_size} → {num_heads_x_head_dim} (matrix: {hidden_size}x{num_heads_x_head_dim})")
                logger.info(f"   V Projection: {hidden_size} → {num_heads_x_head_dim} (matrix: {hidden_size}x{num_heads_x_head_dim})")
                logger.info(f"   O Projection: {num_heads_x_head_dim} → {hidden_size} (matrix: {num_heads_x_head_dim}x{hidden_size})")
                
                # Attention computation requirements
                seq_len = 128  # Example sequence length
                logger.info(f"\n🧮 ATTENTION COMPUTATION (seq_len={seq_len}):")
                logger.info(f"   Q, K, V tensors: ({seq_len}, {num_heads}, {head_dim})")
                logger.info(f"   Attention scores: ({seq_len}, {seq_len}) per head")
                logger.info(f"   Total attention matrix: ({num_heads}, {seq_len}, {seq_len})")
                
                # Memory requirements for NPU (2GB SRAM budget)
                q_memory_mb = (seq_len * num_heads * head_dim * 2) / (1024 * 1024)  # FP16
                attention_scores_mb = (num_heads * seq_len * seq_len * 2) / (1024 * 1024)  # FP16
                
                logger.info(f"\n💾 NPU MEMORY REQUIREMENTS (FP16):")
                logger.info(f"   Q/K/V tensors: {q_memory_mb:.2f} MB each")
                logger.info(f"   Attention scores: {attention_scores_mb:.2f} MB")
                logger.info(f"   Total working memory: {(3 * q_memory_mb + attention_scores_mb):.2f} MB")
                logger.info(f"   NPU SRAM budget: 2048 MB")
                logger.info(f"   Memory utilization: {((3 * q_memory_mb + attention_scores_mb) / 2048 * 100):.1f}%")
                
                # Phoenix NPU optimization recommendations
                logger.info(f"\n🔥 PHOENIX NPU OPTIMIZATION STRATEGY:")
                logger.info(f"   Tile Configuration: 16 compute tiles available")
                logger.info(f"   Matrix Tiling: Tile {hidden_size}x{num_heads_x_head_dim} matrix for parallel processing")
                logger.info(f"   Memory Hierarchy: L1 (32KB per tile) → L2 (512KB) → L3 (2GB SRAM)")
                logger.info(f"   Quantization: INT8/INT4 for weights, FP16 for activations")
                logger.info(f"   Kernel Fusion: Combine Q/K/V projections + scaled dot-product attention")
                
                return {
                    'hidden_size': hidden_size,
                    'num_heads': num_heads, 
                    'head_dim': head_dim,
                    'attention_output_size': num_heads_x_head_dim,
                    'tensor_analysis': attention_analysis,
                    'memory_requirements_mb': {
                        'qkv_tensors': q_memory_mb,
                        'attention_scores': attention_scores_mb,
                        'total_working': 3 * q_memory_mb + attention_scores_mb
                    }
                }
        
        return attention_analysis
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None

def analyze_quantization_schemes():
    """Analyze quantization schemes used for different tensor types"""
    
    quantized_path = Path("./quantized_models/gemma-3-27b-it-layer-by-layer")
    
    logger.info("\n🔬 QUANTIZATION SCHEME ANALYSIS:")
    
    scheme_analysis = {
        'attention': {},
        'ffn': {},
        'embeddings': {}
    }
    
    # Sample a few layer files
    layer_files = list(quantized_path.glob("*layer_*.safetensors"))[:3]
    
    for layer_file in layer_files:
        try:
            with safe_open(layer_file, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                
                for key, scheme in metadata.items():
                    if '_scale' in key:
                        continue
                        
                    # Categorize by tensor type
                    if any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                        scheme_analysis['attention'][key] = scheme
                    elif any(proj in key for proj in ['gate_proj', 'up_proj', 'down_proj']):
                        scheme_analysis['ffn'][key] = scheme
                    elif 'embed' in key:
                        scheme_analysis['embeddings'][key] = scheme
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze {layer_file}: {e}")
    
    # Summarize quantization schemes
    for category, schemes in scheme_analysis.items():
        if schemes:
            unique_schemes = set(schemes.values())
            logger.info(f"🎯 {category.upper()} Quantization:")
            for scheme in unique_schemes:
                count = sum(1 for s in schemes.values() if s == scheme)
                logger.info(f"   {scheme}: {count} tensors")
    
    return scheme_analysis

if __name__ == "__main__":
    attention_info = analyze_attention_tensors()
    quantization_info = analyze_quantization_schemes()
    
    if attention_info:
        logger.info("\n✅ Analysis complete! Ready for custom NPU kernel design.")
    else:
        logger.error("❌ Analysis failed. Check quantized model availability.")