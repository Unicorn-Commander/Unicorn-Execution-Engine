#!/usr/bin/env python3
"""
UC1-EMB Quantization Format Specification
NPU-optimized quantization for embedding models on AMD XDNA hardware
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class UC1EMBConfig:
    """Configuration for UC1-EMB quantization format"""
    
    # Format variants
    UC1_EMB_Q4N = "UC1-EMB-Q4N"  # 4-bit NPU-optimized
    UC1_EMB_Q8N = "UC1-EMB-Q8N"  # 8-bit NPU-optimized  
    UC1_EMB_MXN = "UC1-EMB-MXN"  # Mixed precision NPU
    
    # NPU-specific parameters
    npu_tile_size: int = 256      # NPU tile dimensions
    npu_vector_width: int = 64    # NPU SIMD width
    npu_memory_banks: int = 4      # NPU memory banks
    
    # Quantization parameters
    group_size: int = 128          # Quantization group size
    scale_bits: int = 16           # Bits for scale factor
    zero_point_bits: int = 8       # Bits for zero point
    
    # Memory alignment
    alignment: int = 64            # NPU DMA alignment requirement
    interleave_factor: int = 4     # Memory interleaving for NPU

class UC1EMBQuantizer:
    """
    UC1-EMB Quantization for NPU deployment
    Optimized for AMD XDNA NPU architecture
    """
    
    def __init__(self, format_type: str = "UC1-EMB-Q4N"):
        self.format = format_type
        self.config = UC1EMBConfig()
        
    def quantize_for_npu(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Quantize weights specifically for NPU characteristics
        
        NPU Optimizations:
        1. Tile-aligned memory layout
        2. Bank-interleaved storage
        3. Vector-width aligned operations
        4. DMA-friendly data arrangement
        """
        
        if self.format == self.config.UC1_EMB_Q4N:
            return self._quantize_4bit_npu(weights)
        elif self.format == self.config.UC1_EMB_Q8N:
            return self._quantize_8bit_npu(weights)
        elif self.format == self.config.UC1_EMB_MXN:
            return self._quantize_mixed_npu(weights)
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def _quantize_4bit_npu(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        4-bit quantization optimized for NPU INT4 compute units
        
        NPU-specific layout:
        - Packed 2 values per byte
        - Tile-aligned for NPU processing
        - Interleaved for parallel NPU access
        """
        
        # Reshape for NPU tiles
        orig_shape = weights.shape
        weights_flat = weights.flatten()
        
        # Pad to tile size
        tile_size = self.config.npu_tile_size
        padded_size = ((len(weights_flat) + tile_size - 1) // tile_size) * tile_size
        weights_padded = np.pad(weights_flat, (0, padded_size - len(weights_flat)))
        
        # Reshape into tiles
        num_tiles = padded_size // tile_size
        weights_tiled = weights_padded.reshape(num_tiles, tile_size)
        
        # Quantize per tile (NPU processes tiles independently)
        quantized_tiles = []
        scales = []
        zero_points = []
        
        for tile in weights_tiled:
            # Calculate scale and zero point
            min_val, max_val = tile.min(), tile.max()
            scale = (max_val - min_val) / 15.0  # 4-bit range
            zero_point = 8 - (min_val / scale) if scale != 0 else 8
            
            # Quantize tile
            if scale != 0:
                tile_q = np.round((tile / scale) + zero_point).astype(np.uint8)
                tile_q = np.clip(tile_q, 0, 15)
            else:
                tile_q = np.zeros_like(tile, dtype=np.uint8)
            
            quantized_tiles.append(tile_q)
            scales.append(scale)
            zero_points.append(zero_point)
        
        # Pack 4-bit values (2 per byte) with NPU-friendly layout
        packed_data = self._pack_4bit_npu_optimized(np.array(quantized_tiles))
        
        return {
            'format': self.format,
            'data': packed_data,
            'scales': np.array(scales, dtype=np.float16),
            'zero_points': np.array(zero_points, dtype=np.uint8),
            'shape': orig_shape,
            'tile_size': tile_size,
            'num_tiles': num_tiles,
            'npu_optimized': True
        }
    
    def _quantize_8bit_npu(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        8-bit quantization optimized for NPU INT8 compute units
        
        NPU-specific layout:
        - Direct INT8 format
        - Bank-interleaved for parallel access
        - Vector-width aligned
        """
        
        orig_shape = weights.shape
        weights_flat = weights.flatten()
        
        # Align to vector width
        vector_width = self.config.npu_vector_width
        aligned_size = ((len(weights_flat) + vector_width - 1) // vector_width) * vector_width
        weights_aligned = np.pad(weights_flat, (0, aligned_size - len(weights_flat)))
        
        # Reshape for vectorized processing
        num_vectors = aligned_size // vector_width
        weights_vectorized = weights_aligned.reshape(num_vectors, vector_width)
        
        # Quantize per vector (NPU processes vectors)
        quantized_vectors = []
        scales = []
        zero_points = []
        
        for vector in weights_vectorized:
            min_val, max_val = vector.min(), vector.max()
            scale = (max_val - min_val) / 255.0
            zero_point = 128 - (min_val / scale) if scale != 0 else 128
            
            if scale != 0:
                vector_q = np.round((vector / scale) + zero_point).astype(np.uint8)
            else:
                vector_q = np.full_like(vector, 128, dtype=np.uint8)
            
            quantized_vectors.append(vector_q)
            scales.append(scale)
            zero_points.append(zero_point)
        
        # Interleave for NPU memory banks
        interleaved_data = self._interleave_for_npu(np.array(quantized_vectors))
        
        return {
            'format': self.format,
            'data': interleaved_data,
            'scales': np.array(scales, dtype=np.float16),
            'zero_points': np.array(zero_points, dtype=np.uint8),
            'shape': orig_shape,
            'vector_width': vector_width,
            'num_vectors': num_vectors,
            'npu_optimized': True
        }
    
    def _quantize_mixed_npu(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Mixed precision quantization for NPU
        Critical layers in INT8, others in INT4
        """
        
        # Determine layer importance (simplified heuristic)
        weight_magnitude = np.abs(weights).mean()
        
        if weight_magnitude > 0.1:  # Important layer
            return self._quantize_8bit_npu(weights)
        else:  # Less critical layer
            return self._quantize_4bit_npu(weights)
    
    def _pack_4bit_npu_optimized(self, tiles: np.ndarray) -> np.ndarray:
        """
        Pack 4-bit values with NPU-optimized memory layout
        
        NPU optimization:
        - Consecutive values packed for SIMD operations
        - Aligned to DMA transfer boundaries
        - Minimizes NPU memory stalls
        """
        
        packed = []
        for tile in tiles:
            # Pack pairs of 4-bit values
            tile_packed = np.zeros(len(tile) // 2, dtype=np.uint8)
            tile_packed = (tile[0::2] & 0x0F) | ((tile[1::2] & 0x0F) << 4)
            
            # Align to DMA boundary
            alignment = self.config.alignment
            if len(tile_packed) % alignment != 0:
                pad_size = alignment - (len(tile_packed) % alignment)
                tile_packed = np.pad(tile_packed, (0, pad_size))
            
            packed.append(tile_packed)
        
        return np.concatenate(packed)
    
    def _interleave_for_npu(self, vectors: np.ndarray) -> np.ndarray:
        """
        Interleave data for NPU memory banks
        Enables parallel access across NPU compute units
        """
        
        num_banks = self.config.npu_memory_banks
        
        # Simple bank interleaving - distribute vectors across banks
        # Each bank gets vectors[i::num_banks]
        interleaved = []
        for bank in range(num_banks):
            bank_vectors = vectors[bank::num_banks]
            interleaved.append(bank_vectors)
        
        # Concatenate all banks
        return np.concatenate(interleaved)

def compare_formats():
    """Compare different UC1-EMB formats"""
    
    print("UC1-EMB FORMAT COMPARISON")
    print("=" * 60)
    
    # Test data
    test_weights = np.random.randn(1024, 768).astype(np.float32)
    original_size = test_weights.nbytes
    
    formats = ["UC1-EMB-Q4N", "UC1-EMB-Q8N", "UC1-EMB-MXN"]
    
    for format_type in formats:
        quantizer = UC1EMBQuantizer(format_type)
        result = quantizer.quantize_for_npu(test_weights)
        
        compressed_size = result['data'].nbytes + result['scales'].nbytes + result['zero_points'].nbytes
        compression_ratio = original_size / compressed_size
        
        print(f"\n{format_type}:")
        print(f"  Original size: {original_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  NPU optimized: {result['npu_optimized']}")
        
        if 'tile_size' in result:
            print(f"  Tile size: {result['tile_size']}")
            print(f"  Num tiles: {result['num_tiles']}")
        
        if 'vector_width' in result:
            print(f"  Vector width: {result['vector_width']}")
            print(f"  Num vectors: {result['num_vectors']}")

if __name__ == "__main__":
    compare_formats()
    
    print("\n\nUC1-EMB Format Benefits:")
    print("-" * 40)
    print("✅ NPU-optimized memory layout")
    print("✅ Tile-aligned for NPU processing")
    print("✅ Bank-interleaved for parallelism")
    print("✅ DMA-friendly data arrangement")
    print("✅ Minimal NPU memory stalls")
    print("✅ Hardware-specific optimizations")