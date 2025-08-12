#!/usr/bin/env python3
"""
Optimized loader for Gemma 3 27B - handles INT8 + BF16 properly
"""

import numpy as np
import logging
import time
import struct
from pathlib import Path
from safetensors import safe_open

logger = logging.getLogger(__name__)

def convert_bf16_to_fp32_bytes(bf16_bytes):
    """Convert BF16 bytes to FP32 by padding zeros"""
    # BF16 is just FP32 with lower 16 bits truncated
    # So we pad with zeros to restore FP32
    fp32_bytes = bf16_bytes + b'\x00\x00'
    return struct.unpack('f', fp32_bytes)[0]

class Gemma27BLoaderV2:
    """Loader that handles INT8 weights + BF16 scales"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.tensors = {}
        
    def load_layer_raw(self, layer_file):
        """Load layer with raw data access to handle BF16"""
        import json
        
        tensors = {}
        
        with open(layer_file, 'rb') as f:
            # Read header size
            header_size = int.from_bytes(f.read(8), 'little')
            
            # Read and parse header
            header = json.loads(f.read(header_size))
            
            # Current position after header
            data_start = 8 + header_size
            
            # Process each tensor
            for name, info in header.items():
                if isinstance(info, dict) and 'dtype' in info:
                    dtype = info['dtype']
                    shape = info['shape']
                    offset = info['data_offsets'][0] + data_start
                    size = info['data_offsets'][1] - info['data_offsets'][0]
                    
                    # Seek to tensor data
                    f.seek(offset)
                    data = f.read(size)
                    
                    # Convert based on dtype
                    if dtype == 'I8':
                        # INT8 weights
                        tensor = np.frombuffer(data, dtype=np.int8).reshape(shape)
                    elif dtype == 'BF16':
                        # BFloat16 scales - convert to FP32
                        num_elements = size // 2  # 2 bytes per BF16
                        values = []
                        for i in range(num_elements):
                            bf16_bytes = data[i*2:(i+1)*2]
                            fp32_value = convert_bf16_to_fp32_bytes(bf16_bytes)
                            values.append(fp32_value)
                        tensor = np.array(values, dtype=np.float32).reshape(shape)
                    elif dtype == 'F32':
                        # Float32
                        tensor = np.frombuffer(data, dtype=np.float32).reshape(shape)
                    else:
                        logger.warning(f"Unknown dtype {dtype} for {name}")
                        continue
                    
                    tensors[name] = tensor
                    
        return tensors
    
    def load_layer_numpy(self, layer_file):
        """Load using numpy framework (INT8 only)"""
        tensors = {}
        
        try:
            with safe_open(layer_file, framework="numpy") as f:
                for name in f.keys():
                    try:
                        tensor = f.get_tensor(name)
                        # Only load INT8 tensors this way
                        if tensor.dtype == np.int8:
                            tensors[name] = tensor
                    except:
                        # Skip BF16 tensors
                        pass
                        
        except Exception as e:
            logger.error(f"Error in numpy loading: {e}")
            
        return tensors
    
    def load_layer(self, layer_file):
        """Load a layer combining both methods"""
        logger.info(f"Loading {layer_file.name}")
        
        # First get INT8 weights via numpy
        tensors = self.load_layer_numpy(layer_file)
        int8_count = len(tensors)
        
        # Then get BF16 scales via raw access
        all_tensors = self.load_layer_raw(layer_file)
        
        # Merge, preferring numpy-loaded INT8
        for name, tensor in all_tensors.items():
            if name not in tensors:
                tensors[name] = tensor
                
        bf16_count = len(tensors) - int8_count
        logger.info(f"   Loaded {int8_count} INT8 weights, {bf16_count} BF16 scales")
        
        return tensors
    
    def test_single_layer(self):
        """Test loading a single layer"""
        layer_files = sorted(list(self.model_path.glob("*.safetensors")))[:1]
        
        for layer_file in layer_files:
            tensors = self.load_layer(layer_file)
            
            # Show some tensors
            for name, tensor in list(tensors.items())[:5]:
                logger.info(f"   {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                
            return tensors

def main():
    """Test the V2 loader"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    loader = Gemma27BLoaderV2(model_path)
    
    logger.info("🚀 Testing Gemma 27B V2 Loader")
    tensors = loader.test_single_layer()
    
    if tensors:
        logger.info(f"✅ Successfully loaded {len(tensors)} tensors")
        
        # Calculate memory usage
        total_bytes = sum(t.nbytes for t in tensors.values())
        logger.info(f"   Total size: {total_bytes / (1024**2):.1f} MB")
    else:
        logger.error("❌ Failed to load tensors")

if __name__ == "__main__":
    main()