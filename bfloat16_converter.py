#!/usr/bin/env python3
"""
BFloat16 conversion utilities for Gemma 3 27B model loading
"""

import numpy as np
import struct
import logging

logger = logging.getLogger(__name__)

def bfloat16_to_float32(data):
    """
    Convert bfloat16 data to float32
    BFloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
    Float32: 1 sign bit, 8 exponent bits, 23 mantissa bits
    """
    if isinstance(data, np.ndarray):
        # Handle numpy array
        if data.dtype.name == 'bfloat16' or str(data.dtype) == 'bfloat16':
            # Get raw bytes
            raw_bytes = data.tobytes()
            # Each bfloat16 is 2 bytes
            num_elements = len(raw_bytes) // 2
            
            # Convert to float32
            float32_values = []
            for i in range(num_elements):
                # Get 2 bytes for bfloat16
                bf16_bytes = raw_bytes[i*2:(i+1)*2]
                # Pad with zeros to make float32 (bfloat16 is just truncated float32)
                f32_bytes = bf16_bytes + b'\x00\x00'
                # Unpack as float32
                f32_value = struct.unpack('f', f32_bytes)[0]
                float32_values.append(f32_value)
            
            # Reshape to original shape
            result = np.array(float32_values, dtype=np.float32).reshape(data.shape)
            return result
    
    # If not bfloat16, return as-is
    return data

def bfloat16_to_float16(data):
    """
    Convert bfloat16 to float16 for GPU compatibility
    This involves precision loss but maintains GPU efficiency
    """
    if isinstance(data, np.ndarray) and (data.dtype.name == 'bfloat16' or str(data.dtype) == 'bfloat16'):
        # First convert to float32
        float32_data = bfloat16_to_float32(data)
        # Then to float16
        return float32_data.astype(np.float16)
    return data

def safe_tensor_convert(tensor, target_dtype='float16'):
    """
    Safely convert tensor from any dtype to target dtype
    Handles bfloat16 specially
    """
    try:
        # Check if it's bfloat16
        if hasattr(tensor, 'dtype'):
            dtype_str = str(tensor.dtype)
            if 'bfloat16' in dtype_str:
                logger.debug(f"Converting bfloat16 tensor to {target_dtype}")
                if target_dtype == 'float16':
                    return bfloat16_to_float16(tensor)
                elif target_dtype == 'float32':
                    return bfloat16_to_float32(tensor)
        
        # For other dtypes, use numpy conversion
        if target_dtype == 'float16':
            return tensor.astype(np.float16)
        elif target_dtype == 'float32':
            return tensor.astype(np.float32)
        else:
            return tensor
            
    except Exception as e:
        logger.error(f"Failed to convert tensor: {e}")
        # Return original tensor if conversion fails
        return tensor

def get_tensor_info(tensor):
    """Get information about a tensor including dtype"""
    info = {
        'shape': tensor.shape if hasattr(tensor, 'shape') else None,
        'dtype': str(tensor.dtype) if hasattr(tensor, 'dtype') else None,
        'size': tensor.size if hasattr(tensor, 'size') else None,
        'bytes': tensor.nbytes if hasattr(tensor, 'nbytes') else None
    }
    return info

if __name__ == "__main__":
    # Test the conversion
    logging.basicConfig(level=logging.INFO)
    
    # Create mock bfloat16 data (using float32 as placeholder)
    test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    print(f"Original: {test_data} (dtype: {test_data.dtype})")
    
    # Convert to float16
    converted = safe_tensor_convert(test_data, 'float16')
    print(f"Converted: {converted} (dtype: {converted.dtype})")