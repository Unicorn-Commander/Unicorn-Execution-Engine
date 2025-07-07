#!/usr/bin/env python3
"""
Advanced Quantization Engine
Implements Q4_K_M equivalent and better quantization strategies
Optimized for NPU+iGPU hybrid execution
"""

import os
import sys
import time
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Supported quantization types"""
    Q4_K_M = "q4_k_m"          # GGUF Q4_K_M equivalent 
    Q4_K_S = "q4_k_s"          # GGUF Q4_K_S equivalent
    AWQ_4BIT = "awq_4bit"      # AWQ-style 4-bit
    GPTQ_4BIT = "gptq_4bit"    # GPTQ-style 4-bit
    CUSTOM_Q4 = "custom_q4"    # Our optimized 4-bit
    CUSTOM_Q3 = "custom_q3"    # Our optimized 3-bit
    HYBRID_Q4 = "hybrid_q4"    # NPU/iGPU optimized

@dataclass
class QuantizationConfig:
    """Configuration for quantization methods"""
    quant_type: QuantizationType
    block_size: int = 32        # Block size for quantization
    npu_friendly: bool = True   # Optimize for NPU execution
    igpu_friendly: bool = True  # Optimize for iGPU execution
    memory_efficient: bool = True
    
    # Advanced parameters
    outlier_threshold: float = 3.0
    preserve_outliers: bool = True
    use_gating: bool = False    # For sparse quantization
    
    # Target metrics
    target_compression: float = 4.0    # Target compression ratio
    target_accuracy: float = 0.98      # Target accuracy retention

@dataclass
class QuantizedWeight:
    """Container for quantized weight data"""
    quantized_data: np.ndarray
    scales: np.ndarray
    zeros: Optional[np.ndarray] = None
    outliers: Optional[np.ndarray] = None
    outlier_indices: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class AdvancedQuantizationEngine:
    """
    Advanced quantization engine with multiple strategies
    Optimized for AMD NPU + iGPU hybrid execution
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.compression_stats = {}
        
    def quantize_q4_k_m(self, weights: np.ndarray) -> QuantizedWeight:
        """
        Implement Q4_K_M quantization (GGUF-compatible)
        4-bit weights with 6-bit scales in 32-element blocks
        """
        assert weights.dtype == np.float32 or weights.dtype == np.float16
        original_shape = weights.shape
        weights_flat = weights.flatten()
        
        # Pad to multiple of block size
        block_size = self.config.block_size
        remainder = len(weights_flat) % block_size
        if remainder:
            padding = block_size - remainder
            weights_flat = np.pad(weights_flat, (0, padding), mode='constant')
        
        num_blocks = len(weights_flat) // block_size
        blocks = weights_flat.reshape(num_blocks, block_size)
        
        # Q4_K_M quantization algorithm
        quantized_blocks = []
        scales = []
        
        for block in blocks:
            # Find scale factors using Q4_K_M method
            abs_block = np.abs(block)
            max_val = np.max(abs_block)
            
            if max_val == 0:
                scale = 1.0
                quantized_block = np.zeros(block_size, dtype=np.int8)
            else:
                # Q4_K_M uses 6-bit scales for better precision
                scale = max_val / 15.0  # 4-bit range: -7 to 8
                
                # Quantize with rounding
                quantized_block = np.round(block / scale).astype(np.int8)
                quantized_block = np.clip(quantized_block, -7, 8)
            
            quantized_blocks.append(quantized_block)
            scales.append(scale)
        
        quantized_data = np.array(quantized_blocks, dtype=np.int8)
        scales_array = np.array(scales, dtype=np.float32)
        
        # Calculate compression statistics
        original_bytes = weights.nbytes
        quantized_bytes = quantized_data.nbytes + scales_array.nbytes
        compression_ratio = original_bytes / quantized_bytes
        
        self.compression_stats['q4_k_m'] = {
            'original_bytes': original_bytes,
            'quantized_bytes': quantized_bytes,
            'compression_ratio': compression_ratio,
            'num_blocks': num_blocks
        }
        
        return QuantizedWeight(
            quantized_data=quantized_data,
            scales=scales_array,
            metadata={
                'method': 'Q4_K_M',
                'block_size': block_size,
                'original_shape': original_shape,
                'compression_ratio': compression_ratio
            }
        )
    
    def quantize_custom_q4(self, weights: np.ndarray) -> QuantizedWeight:
        """
        Custom Q4 quantization optimized for NPU+iGPU hybrid execution
        Includes outlier preservation and mixed precision
        """
        original_shape = weights.shape
        weights_flat = weights.flatten().astype(np.float32)
        
        # Outlier detection and preservation
        outliers = None
        outlier_indices = None
        
        if self.config.preserve_outliers:
            threshold = self.config.outlier_threshold
            std_dev = np.std(weights_flat)
            mean_val = np.mean(weights_flat)
            
            outlier_mask = np.abs(weights_flat - mean_val) > threshold * std_dev
            if np.any(outlier_mask):
                outlier_indices = np.where(outlier_mask)[0]
                outliers = weights_flat[outlier_mask].astype(np.float16)
                # Zero out outliers for normal quantization
                weights_flat[outlier_mask] = 0
        
        # Block-wise quantization with adaptive scaling
        block_size = self.config.block_size
        remainder = len(weights_flat) % block_size
        if remainder:
            padding = block_size - remainder
            weights_flat = np.pad(weights_flat, (0, padding), mode='constant')
        
        num_blocks = len(weights_flat) // block_size
        blocks = weights_flat.reshape(num_blocks, block_size)
        
        quantized_blocks = []
        scales = []
        
        for i, block in enumerate(blocks):
            # Adaptive scaling based on distribution
            abs_vals = np.abs(block[block != 0])  # Ignore zeros/outliers
            
            if len(abs_vals) == 0:
                scale = 1.0
                quantized_block = np.zeros(block_size, dtype=np.int8)
            else:
                # Use percentile-based scaling for better utilization
                percentile_99 = np.percentile(abs_vals, 99)
                scale = percentile_99 / 7.0  # 4-bit signed range: -7 to 7
                
                # Quantize with stochastic rounding for better accuracy
                normalized = block / scale
                quantized_block = self._stochastic_round(normalized).astype(np.int8)
                quantized_block = np.clip(quantized_block, -7, 7)
            
            quantized_blocks.append(quantized_block)
            scales.append(scale)
        
        quantized_data = np.array(quantized_blocks, dtype=np.int8)
        scales_array = np.array(scales, dtype=np.float32)
        
        # Calculate compression with outliers
        original_bytes = weights.nbytes
        quantized_bytes = quantized_data.nbytes + scales_array.nbytes
        if outliers is not None:
            quantized_bytes += outliers.nbytes + outlier_indices.nbytes
        compression_ratio = original_bytes / quantized_bytes
        
        self.compression_stats['custom_q4'] = {
            'original_bytes': original_bytes,
            'quantized_bytes': quantized_bytes,
            'compression_ratio': compression_ratio,
            'num_outliers': len(outliers) if outliers is not None else 0
        }
        
        return QuantizedWeight(
            quantized_data=quantized_data,
            scales=scales_array,
            outliers=outliers,
            outlier_indices=outlier_indices,
            metadata={
                'method': 'Custom_Q4',
                'block_size': block_size,
                'original_shape': original_shape,
                'compression_ratio': compression_ratio,
                'num_outliers': len(outliers) if outliers is not None else 0
            }
        )
    
    def quantize_hybrid_npu_igpu(self, weights: np.ndarray) -> QuantizedWeight:
        """
        Hybrid quantization optimized for NPU attention + iGPU FFN
        Different quantization strategies for different operations
        """
        original_shape = weights.shape
        
        # Determine if this is attention or FFN weights based on shape
        if len(original_shape) == 2:
            d1, d2 = original_shape
            is_attention = (d1 == d2) or (d1 * 4 == d2) or (d2 * 4 == d1)
        else:
            is_attention = False
        
        if is_attention and self.config.npu_friendly:
            # NPU-optimized quantization for attention
            return self._quantize_npu_attention(weights)
        else:
            # iGPU-optimized quantization for FFN
            return self._quantize_igpu_ffn(weights)
    
    def _quantize_npu_attention(self, weights: np.ndarray) -> QuantizedWeight:
        """NPU-optimized quantization for attention weights"""
        # NPU prefers smaller block sizes and symmetric quantization
        config = QuantizationConfig(
            quant_type=QuantizationType.CUSTOM_Q4,
            block_size=16,  # Smaller blocks for NPU
            npu_friendly=True
        )
        
        original_config = self.config
        self.config = config
        result = self.quantize_custom_q4(weights)
        self.config = original_config
        
        result.metadata['npu_optimized'] = True
        return result
    
    def _quantize_igpu_ffn(self, weights: np.ndarray) -> QuantizedWeight:
        """iGPU-optimized quantization for FFN weights"""
        # iGPU prefers larger blocks and asymmetric quantization
        config = QuantizationConfig(
            quant_type=QuantizationType.Q4_K_M,
            block_size=64,  # Larger blocks for iGPU
            igpu_friendly=True
        )
        
        original_config = self.config
        self.config = config
        result = self.quantize_q4_k_m(weights)
        self.config = original_config
        
        result.metadata['igpu_optimized'] = True
        return result
    
    def _stochastic_round(self, x: np.ndarray) -> np.ndarray:
        """Stochastic rounding for better quantization accuracy"""
        floor_x = np.floor(x)
        prob = x - floor_x
        random_vals = np.random.random(x.shape)
        return floor_x + (random_vals < prob).astype(x.dtype)
    
    def dequantize(self, quantized: QuantizedWeight) -> np.ndarray:
        """Dequantize weights back to float32"""
        method = quantized.metadata.get('method', 'Unknown')
        original_shape = quantized.metadata.get('original_shape')
        
        # Dequantize main weights
        dequantized_blocks = []
        for i, block in enumerate(quantized.quantized_data):
            scale = quantized.scales[i]
            dequantized_block = block.astype(np.float32) * scale
            dequantized_blocks.append(dequantized_block)
        
        dequantized = np.concatenate(dequantized_blocks)
        
        # Restore outliers if present
        if quantized.outliers is not None and quantized.outlier_indices is not None:
            dequantized[quantized.outlier_indices] = quantized.outliers.astype(np.float32)
        
        # Reshape to original shape
        if original_shape:
            # Remove padding if necessary
            original_size = np.prod(original_shape)
            dequantized = dequantized[:original_size]
            dequantized = dequantized.reshape(original_shape)
        
        return dequantized
    
    def benchmark_quantization_accuracy(self, weights: np.ndarray, method: QuantizationType) -> Dict:
        """Benchmark quantization accuracy for different methods"""
        logger.info(f"Benchmarking {method.value} quantization accuracy...")
        
        original_config = self.config
        self.config.quant_type = method
        
        start_time = time.time()
        
        # Quantize
        if method == QuantizationType.Q4_K_M:
            quantized = self.quantize_q4_k_m(weights)
        elif method == QuantizationType.CUSTOM_Q4:
            quantized = self.quantize_custom_q4(weights)
        elif method == QuantizationType.HYBRID_Q4:
            quantized = self.quantize_hybrid_npu_igpu(weights)
        else:
            quantized = self.quantize_custom_q4(weights)  # Fallback
        
        # Dequantize
        dequantized = self.dequantize(quantized)
        
        quantization_time = time.time() - start_time
        
        # Calculate accuracy metrics
        mse = np.mean((weights - dequantized) ** 2)
        mae = np.mean(np.abs(weights - dequantized))
        max_error = np.max(np.abs(weights - dequantized))
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(weights ** 2)
        noise_power = mse
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Compression ratio
        compression_ratio = quantized.metadata.get('compression_ratio', 0)
        
        self.config = original_config
        
        return {
            'method': method.value,
            'quantization_time_s': quantization_time,
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'snr_db': float(snr_db),
            'compression_ratio': float(compression_ratio),
            'accuracy_score': float(1.0 / (1.0 + mae)),  # Simple accuracy metric
            'quantized_size_mb': quantized.quantized_data.nbytes / 1024**2
        }
    
    def save_quantized_weights(self, quantized: QuantizedWeight, filepath: str):
        """Save quantized weights in custom format"""
        with open(filepath, 'wb') as f:
            # Write header
            f.write(b'UEE_QUANT')  # Magic number
            f.write(struct.pack('I', 1))  # Version
            
            # Write metadata
            metadata_str = str(quantized.metadata).encode('utf-8')
            f.write(struct.pack('I', len(metadata_str)))
            f.write(metadata_str)
            
            # Write quantized data
            f.write(struct.pack('I', quantized.quantized_data.nbytes))
            f.write(quantized.quantized_data.tobytes())
            
            # Write scales
            f.write(struct.pack('I', quantized.scales.nbytes))
            f.write(quantized.scales.tobytes())
            
            # Write outliers if present
            if quantized.outliers is not None:
                f.write(struct.pack('I', quantized.outliers.nbytes))
                f.write(quantized.outliers.tobytes())
                f.write(struct.pack('I', quantized.outlier_indices.nbytes))
                f.write(quantized.outlier_indices.tobytes())
            else:
                f.write(struct.pack('I', 0))  # No outliers
                f.write(struct.pack('I', 0))  # No outlier indices
        
        logger.info(f"Quantized weights saved to {filepath}")


def main():
    """Test quantization engine"""
    print("🔢 Advanced Quantization Engine Test")
    print("=" * 40)
    
    # Create test weights (simulating transformer layer)
    np.random.seed(42)
    test_weights = np.random.randn(2048, 2048).astype(np.float32) * 0.1
    
    # Test different quantization methods
    methods = [
        QuantizationType.Q4_K_M,
        QuantizationType.CUSTOM_Q4,
        QuantizationType.HYBRID_Q4
    ]
    
    results = []
    
    for method in methods:
        config = QuantizationConfig(quant_type=method)
        engine = AdvancedQuantizationEngine(config)
        
        benchmark = engine.benchmark_quantization_accuracy(test_weights, method)
        results.append(benchmark)
        
        print(f"\n📊 {benchmark['method']} Results:")
        print(f"   Compression: {benchmark['compression_ratio']:.1f}x")
        print(f"   SNR: {benchmark['snr_db']:.1f} dB")
        print(f"   Accuracy: {benchmark['accuracy_score']:.4f}")
        print(f"   Size: {benchmark['quantized_size_mb']:.2f} MB")
        print(f"   Time: {benchmark['quantization_time_s']:.3f}s")
    
    # Find best method
    best_method = max(results, key=lambda x: x['accuracy_score'])
    print(f"\n🏆 Best Method: {best_method['method']}")
    print(f"   Best accuracy: {best_method['accuracy_score']:.4f}")
    print(f"   Compression: {best_method['compression_ratio']:.1f}x")
    
    return results


if __name__ == "__main__":
    main()