#!/usr/bin/env python3
"""
Dynamic Quantization Engine for the Unicorn Execution Engine
"""

import numpy as np
import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)

class DynamicQuantizationEngine:
    """Handles on-the-fly INT8/INT4 quantization and dequantization."""

    def __init__(self):
        logger.info("🔢 Dynamic Quantization Engine Initialized.")

    def quantize_int8(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantizes a float32 numpy array to INT8 symmetric.
        Returns the quantized tensor and the scale factor.
        """
        scale = 127.0 / np.max(np.abs(tensor))
        quantized_tensor = np.round(tensor * scale).astype(np.int8)
        logger.debug(f"Quantized to INT8: {tensor.shape} -> {quantized_tensor.shape}, scale={scale:.4f}")
        return quantized_tensor, scale

    def dequantize_int8(self, quantized_tensor: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantizes an INT8 numpy array back to float32.
        """
        dequantized_tensor = (quantized_tensor.astype(np.float32) / scale)
        logger.debug(f"Dequantized from INT8: {quantized_tensor.shape} -> {dequantized_tensor.shape}")
        return dequantized_tensor

    def quantize_int4(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes a float32 numpy array to INT4 grouped quantization.
        This is a simplified placeholder. Real INT4 would involve packing 2 INT4s into a byte.
        Returns the quantized tensor (as int8 for simplicity) and group scales.
        """
        # For simplicity, we'll simulate grouped quantization by scaling groups of elements.
        # A real implementation would involve more complex packing/unpacking.
        group_size = 16  # Example group size
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        num_groups = (flat_tensor.size + group_size - 1) // group_size
        scales = np.zeros(num_groups, dtype=np.float32)
        quantized_flat = np.zeros(flat_tensor.size, dtype=np.int8)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, flat_tensor.size)
            group = flat_tensor[start_idx:end_idx]
            
            if group.size > 0:
                group_max_abs = np.max(np.abs(group))
                scale = 7.0 / group_max_abs if group_max_abs > 0 else 1.0 # INT4 range is -8 to 7
                scales[i] = scale
                quantized_flat[start_idx:end_idx] = np.round(group * scale).astype(np.int8)
            
        quantized_tensor = quantized_flat.reshape(original_shape)
        logger.debug(f"Quantized to INT4 (grouped): {tensor.shape} -> {quantized_tensor.shape}, groups={num_groups}")
        return quantized_tensor, scales

    def dequantize_int4(self, quantized_tensor: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Dequantizes an INT4 (simulated as INT8) numpy array back to float32.
        """
        group_size = 16 # Must match quantization group size
        original_shape = quantized_tensor.shape
        flat_quantized = quantized_tensor.flatten().astype(np.float32)
        
        dequantized_flat = np.zeros_like(flat_quantized)
        num_groups = (flat_quantized.size + group_size - 1) // group_size

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, flat_quantized.size)
            group_quantized = flat_quantized[start_idx:end_idx]
            
            if group_quantized.size > 0:
                scale = scales[i]
                dequantized_flat[start_idx:end_idx] = group_quantized / scale

        dequantized_tensor = dequantized_flat.reshape(original_shape)
        logger.debug(f"Dequantized from INT4 (grouped): {quantized_tensor.shape} -> {dequantized_tensor.shape}")
        return dequantized_tensor

    def apply_mixed_precision(self, tensor: np.ndarray, target_dtype: Union[np.dtype, str]) -> np.ndarray:
        """
        Applies mixed precision conversion (e.g., float32 to float16).
        """
        if isinstance(target_dtype, str):
            target_dtype = np.dtype(target_dtype)

        if tensor.dtype == target_dtype:
            logger.debug(f"Tensor already in {target_dtype} precision.")
            return tensor
        
        converted_tensor = tensor.astype(target_dtype)
        logger.debug(f"Converted tensor from {tensor.dtype} to {target_dtype}: {tensor.shape} -> {converted_tensor.shape}")
        return converted_tensor
