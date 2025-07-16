#!/usr/bin/env python3
"""
Pure Memory-Mapped Loader - No PyTorch Dependencies
Uses numpy and mmap for pure hardware acceleration
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import mmap
import json
import struct
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PureMemoryMappedLoader:
    """Pure numpy memory-mapped loader for quantized models"""
    
    def __init__(self, quantized_model_path: str):
        self.quantized_path = Path(quantized_model_path)
        self.mmap_cache = {}  # Cache for memory-mapped files
        self.tensor_cache = {}  # Cache for frequently accessed tensors
        self.file_handles = {}  # Keep file handles open for mmap
        self.metadata = {}  # Model metadata
        
        logger.info(f"🗺️ Pure memory-mapped loader initialized for {self.quantized_path}")
        
        # Pre-open all safetensor files
        self._pre_open_safetensor_files()

    @staticmethod
    def _read_safetensor_header(file_path: Path) -> Dict[str, Any]:
        """Read safetensors header to get tensor metadata"""
        try:
            with open(file_path, 'rb') as f:
                # Read header length (first 8 bytes)
                header_length_bytes = f.read(8)
                header_length = struct.unpack('<Q', header_length_bytes)[0]
                
                # Read header JSON
                header_bytes = f.read(header_length)
                header = json.loads(header_bytes.decode('utf-8'))
                
                return header, 8 + header_length  # Return header and data offset
                
        except Exception as e:
            logger.error(f"❌ Failed to read safetensor header from {file_path}: {e}")
            return {}, 0

    def _pre_open_safetensor_files(self):
        """Pre-open all .safetensors files and store their mmap handles."""
        for file_path in self.quantized_path.glob("*.safetensors"):
            try:
                f = open(file_path, 'rb')
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                header, data_offset = PureMemoryMappedLoader._read_safetensor_header(file_path)
                
                self.file_handles[file_path] = {
                    'file': f,
                    'mmap': mm,
                    'header': header,
                    'data_offset': data_offset
                }
                logger.info(f"   🗺️ Pre-mapped {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to pre-map {file_path}: {e}")

    def _get_mmap_handle(self, file_path: Path):
        """Get memory-mapped handle for file (should be pre-opened)."""
        if file_path not in self.file_handles:
            logger.error(f"❌ Error: {file_path} not pre-opened. This should not happen.")
            return None
        return self.file_handles[file_path]
    
    def _dtype_from_safetensor(self, dtype_str: str) -> np.dtype:
        """Convert safetensor dtype string to numpy dtype"""
        dtype_map = {
            'F32': np.float32,
            'F16': np.float16,
            'BF16': np.float16,  # BF16 -> float16 approximation
            'I8': np.int8,
            'I32': np.int32,
            'I64': np.int64,
            'U8': np.uint8,
            'BOOL': np.bool_
        }
        return dtype_map.get(dtype_str, np.float32)
    
    def _load_tensor_from_mmap(self, file_handle: Dict, tensor_name: str) -> np.ndarray:
        """Load tensor from memory-mapped file"""
        header = file_handle['header']
        mm = file_handle['mmap']
        data_offset = file_handle['data_offset']
        
        if tensor_name not in header:
            raise KeyError(f"Tensor {tensor_name} not found in file")
        
        tensor_info = header[tensor_name]
        dtype = self._dtype_from_safetensor(tensor_info['dtype'])
        shape = tensor_info['shape']
        data_offsets = tensor_info['data_offsets']
        
        # Calculate absolute offset in file
        start_offset = data_offset + data_offsets[0]
        end_offset = data_offset + data_offsets[1]
        
        # Read tensor data from memory-mapped file
        tensor_bytes = mm[start_offset:end_offset]
        
        # Convert to numpy array
        try:
            tensor = np.frombuffer(tensor_bytes, dtype=dtype)
        except ValueError as e:
            if "buffer size must be a multiple of element size" in str(e):
                # Handle misaligned buffer - pad to make it divisible
                element_size = np.dtype(dtype).itemsize
                buffer_size = len(tensor_bytes)
                remainder = buffer_size % element_size
                
                if remainder != 0:
                    padding_needed = element_size - remainder
                    logger.warning(f"🔧 Buffer alignment issue for {tensor_name}: padding {padding_needed} bytes")
                    tensor_bytes = tensor_bytes + b'\x00' * padding_needed
                
                tensor = np.frombuffer(tensor_bytes, dtype=dtype)
            else:
                raise
        
        # Calculate expected size based on shape
        expected_size = np.prod(shape)
        actual_size = tensor.size
        
        logger.debug(f"📊 Tensor {tensor_name}: shape={shape}, expected_size={expected_size}, actual_size={actual_size}, dtype={dtype}")
        
        # Handle size mismatch (common with quantized tensors)
        if actual_size != expected_size:
            logger.warning(f"⚠️ Size mismatch for {tensor_name}: expected {expected_size}, got {actual_size}")
            
            # Try to reshape to the closest valid shape
            if actual_size < expected_size:
                # Pad with zeros if too small
                padding = expected_size - actual_size
                tensor = np.pad(tensor, (0, padding), mode='constant', constant_values=0)
                logger.info(f"🔧 Padded tensor {tensor_name} with {padding} zeros")
            else:
                # Truncate if too large
                tensor = tensor[:expected_size]
                logger.info(f"🔧 Truncated tensor {tensor_name} to {expected_size} elements")
        
        try:
            tensor = tensor.reshape(shape)
        except ValueError as e:
            logger.error(f"❌ Failed to reshape {tensor_name} to {shape}: {e}")
            # Fallback: flatten and use first dimension only
            if len(shape) > 1:
                new_shape = (tensor.size,)
                logger.warning(f"🔧 Fallback: reshaping {tensor_name} to {new_shape}")
                tensor = tensor.reshape(new_shape)
            else:
                raise
        
        return tensor
    
    def get_tensor(self, tensor_info: Dict[str, Any]) -> np.ndarray:
        """Get tensor from memory-mapped file and convert to numpy"""
        if not tensor_info.get('lazy', False):
            # Already loaded
            if hasattr(tensor_info.get('tensor'), 'numpy'):
                return tensor_info['tensor'].numpy()
            return tensor_info['tensor']
        
        try:
            file_path = tensor_info['file_path']
            tensor_name = tensor_info['tensor_name']
            
            # Get memory-mapped file handle
            file_handle = self._get_mmap_handle(file_path)
            if file_handle is None:
                raise RuntimeError(f"Failed to get mmap handle for {file_path}")
            
            # Load tensor from memory-mapped file
            tensor = self._load_tensor_from_mmap(file_handle, tensor_name)
            
            return tensor
            
        except Exception as e:
            logger.error(f"❌ Failed to get tensor: {e}")
            raise
    
    def load_tensor_lazy(self, file_path: Path, tensor_name: str) -> Dict[str, Any]:
        """Load tensor lazily with memory mapping"""
        cache_key = f"{file_path.name}:{tensor_name}"
        
        # Check cache first
        if cache_key in self.tensor_cache:
            return self.tensor_cache[cache_key]
        
        try:
            # Get memory-mapped file handle
            file_handle = self._get_mmap_handle(file_path)
            if file_handle is None:
                raise RuntimeError(f"Failed to get mmap handle for {file_path}")
            
            header = file_handle['header']
            
            # Check if tensor exists
            if tensor_name not in header:
                raise KeyError(f"Tensor {tensor_name} not found in {file_path}")
            
            # Get tensor info from header
            tensor_header = header[tensor_name]
            
            # Look for scale tensor
            scale_name = f"{tensor_name}_scale"
            scale_exists = scale_name in header
            
            # Get quantization scheme from metadata if available
            scheme = header.get('__metadata__', {}).get(tensor_name, 'unknown')
            
            # Create lazy tensor wrapper
            tensor_info = {
                'file_path': file_path,
                'tensor_name': tensor_name,
                'scale_name': scale_name if scale_exists else None,
                'scheme': scheme,
                'shape': tensor_header['shape'],
                'dtype': tensor_header['dtype'],
                'quantized': '_scale' not in tensor_name,  # Not a scale tensor
                'lazy': True  # Mark as lazy-loaded
            }
            
            # Cache for reuse
            self.tensor_cache[cache_key] = tensor_info
            
            return tensor_info
            
        except Exception as e:
            logger.error(f"❌ Failed to lazy load {tensor_name} from {file_path}: {e}")
            raise
    
    def dequantize_numpy(self, quantized: np.ndarray, scale: np.ndarray, scheme: str) -> np.ndarray:
        """Pure numpy dequantization"""
        
        if scheme == 'int8_symmetric':
            return quantized.astype(np.float32) * scale.astype(np.float32)
        elif scheme == 'int4_grouped':
            # INT4 grouped quantization
            tensor_flat = quantized.flatten().astype(np.float32)
            scale_flat = scale.flatten().astype(np.float32)
            
            # Determine group size
            group_size = tensor_flat.size // scale_flat.size
            
            # Expand scale to match tensor dimensions
            scale_expanded = np.repeat(scale_flat, group_size)
            if scale_expanded.size < tensor_flat.size:
                # Pad with last scale value if needed
                padding_size = tensor_flat.size - scale_expanded.size
                scale_expanded = np.concatenate([scale_expanded, np.full(padding_size, scale_flat[-1])])
            elif scale_expanded.size > tensor_flat.size:
                # Truncate if too large
                scale_expanded = scale_expanded[:tensor_flat.size]
            
            dequantized_flat = tensor_flat * scale_expanded
            return dequantized_flat.reshape(quantized.shape)
        elif scheme == 'int8_asymmetric':
            # Assume scale contains [scale_value, zero_point]
            if scale.size >= 2:
                scale_val, zero_point = scale[0], scale[1]
                return (quantized.astype(np.float32) - zero_point) * scale_val
            else:
                # Single scale value, use default zero point
                return (quantized.astype(np.float32) - 128.0) * scale
        else:
            # Unknown scheme, just convert to float
            return quantized.astype(np.float32)
    
    def dequantize_on_demand(self, tensor_info: Dict[str, Any]) -> np.ndarray:
        """Dequantize tensor only when actually needed - pure numpy"""
        if not tensor_info.get('lazy', False):
            # Already loaded
            return tensor_info.get('tensor', np.array([]))
        
        try:
            # Load quantized tensor
            quantized_tensor = self.get_tensor(tensor_info)
            
            # Check if quantized
            if not tensor_info.get('quantized', False):
                # Not quantized, return as-is
                return quantized_tensor.astype(np.float32)
            
            # Load scale if exists
            scale = None
            if tensor_info.get('scale_name'):
                scale_info = {
                    'file_path': tensor_info['file_path'],
                    'tensor_name': tensor_info['scale_name'],
                    'lazy': True,
                    'quantized': False
                }
                scale = self.get_tensor(scale_info)
            
            # Dequantize if scale exists
            if scale is not None:
                scheme = tensor_info.get('scheme', 'unknown')
                logger.debug(f"🔄 Dequantizing {tensor_info['tensor_name']}: scheme={scheme}")
                return self.dequantize_numpy(quantized_tensor, scale, scheme)
            else:
                # No scale, just convert to float32
                return quantized_tensor.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Failed to dequantize tensor: {e}")
            raise
    
    def cleanup(self):
        """Cleanup memory-mapped files"""
        for file_handle in self.file_handles.values():
            try:
                file_handle['mmap'].close()
                file_handle['file'].close()
            except Exception as e:
                logger.error(f"❌ Error cleaning up mmap: {e}")
        
        self.file_handles.clear()
        self.tensor_cache.clear()
        self.mmap_cache.clear()

class MemoryMappedOptimizedLoader(PureMemoryMappedLoader):
    """Alias for compatibility with existing code"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        logger.info("🔄 Using pure numpy memory-mapped loader (no PyTorch)")
        self._pre_open_safetensor_files()
    
    def _load_shared_weights_only(self) -> Dict[str, Any]:
        """Load only shared weights for fast startup - layers loaded on demand"""
        logger.info("🚀 Progressive loading: Loading shared weights only...")
        
        # Scan for shared weights files only
        shared_patterns = [
            'model-*-of-*_shared.safetensors'
        ]
        
        shared_files = []
        for pattern in shared_patterns:
            files = list(self.quantized_path.glob(pattern))
            shared_files.extend(files)
        
        logger.info(f"📂 Found {len(shared_files)} shared weight files")
        
        # Load shared weights using parent class method
        shared_weights = {}
        for shared_file in shared_files:
            logger.info(f"🔄 Loading shared weights from: {shared_file.name}")
            weights = self._load_safetensors_file(shared_file)
            shared_weights.update(weights)
        
        logger.info(f"✅ Loaded {len(shared_weights)} shared weight tensors")
        
        # Create a simple layer loader for on-demand loading
        def layer_loader(layer_idx: int) -> Dict[str, Any]:
            """Load single layer on demand"""
            layer_files = list(self.quantized_path.glob(f'*_layer_{layer_idx}.safetensors'))
            if layer_files:
                return self._load_safetensors_file(layer_files[0])
            return {}
        
        return {
            'shared_weights': shared_weights,
            'layer_loader': layer_loader,
            'total_layers': 62,  # Gemma 3 27B has 62 layers
            'loading_mode': 'progressive'
        }
    
    def load_model(self) -> Dict[str, Any]:
        """Load model metadata and create layer loader"""
        try:
            # Scan for shared weights (embeddings, layer norms, etc.)
            shared_weights = {}
            
            # Look for actual shared weight files in quantized model format
            shared_patterns = [
                'model-*-of-*_shared.safetensors',
                'shared.safetensors' # Also look for a single shared file
            ]
            
            # Find shared weight files using glob pattern
            shared_files = []
            for pattern in shared_patterns:
                search_path = str(self.quantized_path / pattern)
                found_files = glob.glob(search_path)
                shared_files.extend(found_files)
                logger.info(f"🔍 Searching for shared weights with pattern: {search_path} -> Found {len(found_files)} files")

            logger.info(f"🔍 Found {len(shared_files)} shared files: {[Path(f).name for f in shared_files]}")
            
            for shared_file in shared_files:
                shared_path = Path(shared_file)
                logger.info(f"📂 Loading shared weights from: {shared_path.name}")
                
                # Get file handle to read available tensors
                file_handle = self._get_mmap_handle(shared_path)
                if file_handle:
                    header = file_handle['header']
                    tensor_count = 0
                    for tensor_name in header.keys():
                        if not tensor_name.startswith('__'):  # Skip metadata
                            shared_weights[tensor_name] = self.load_tensor_lazy(shared_path, tensor_name)
                            tensor_count += 1
                            logger.info(f"      📂 Loaded tensor: {tensor_name}")
                    logger.info(f"   ✅ Loaded {tensor_count} tensors from {shared_path.name}")
            
            # Create layer loader function for quantized model format
            def layer_loader(layer_idx: int) -> Dict[str, Any]:
                layer_pattern = f"model-*-of-*_layer_{layer_idx}.safetensors"
                search_path = self.quantized_path.joinpath(layer_pattern)
                layer_files = list(self.quantized_path.glob(layer_pattern))
                logger.info(f"🔍 Searching for layer {layer_idx} with pattern: {search_path} -> Found {len(layer_files)} files")

                if not layer_files:
                    # Fallback for different naming conventions
                    layer_pattern = f"*_layer_{layer_idx}.safetensors"
                    search_path = self.quantized_path.joinpath(layer_pattern)
                    layer_files = list(self.quantized_path.glob(layer_pattern))
                    logger.info(f"🔍 Retrying with pattern: {search_path} -> Found {len(layer_files)} files")

                if not layer_files:
                    raise FileNotFoundError(f"Layer file not found for layer {layer_idx} with any known pattern in {self.quantized_path}")
                
                # Load all tensors for this layer from all matching files
                layer_weights = {}
                for layer_file in layer_files:
                    layer_path = Path(layer_file)
                    logger.debug(f"📂 Loading layer {layer_idx} from: {layer_path.name}")
                    
                    file_handle = self._get_mmap_handle(layer_path)
                    if file_handle:
                        header = file_handle['header']
                        for tensor_name in header.keys():
                            if not tensor_name.startswith('__'):  # Skip metadata
                                layer_weights[tensor_name] = self.load_tensor_lazy(layer_path, tensor_name)
                
                return layer_weights
            
            logger.info(f"✅ Pure numpy model loader ready: {len(shared_weights)} shared weights")
            
            return {
                'shared_weights': shared_weights,
                'layer_loader': layer_loader,
                'model_path': self.quantized_path,
                'framework': 'pure_numpy'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise


def test_pure_mmap_loader():
    """Test the pure memory-mapped loader"""
    print("🧪 Testing Pure Memory-Mapped Loader (No PyTorch)")
    print("=" * 60)
    
    model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        return False
    
    try:
        loader = MemoryMappedOptimizedLoader(model_path)
        model_info = loader.load_model()
        
        print(f"✅ Model loaded: {len(model_info['shared_weights'])} shared weights")
        print(f"✅ Framework: {model_info['framework']}")
        
        # Test loading a layer
        layer_loader = model_info['layer_loader']
        layer_0 = layer_loader(0)
        print(f"✅ Layer 0: {len(layer_0)} tensors")
        
        # Test tensor dequantization
        for name, tensor_info in list(layer_0.items())[:3]:  # Test first 3 tensors
            if tensor_info.get('quantized', False):
                dequantized = loader.dequantize_on_demand(tensor_info)
                print(f"✅ Dequantized {name}: {dequantized.shape} {dequantized.dtype}")
            else:
                tensor = loader.get_tensor(tensor_info)
                print(f"✅ Loaded {name}: {tensor.shape} {tensor.dtype}")
        
        loader.cleanup()
        print("🎉 Pure memory-mapped loader test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_pure_mmap_loader()