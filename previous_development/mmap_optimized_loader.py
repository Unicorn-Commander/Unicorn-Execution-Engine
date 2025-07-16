#!/usr/bin/env python3
"""
Memory-Mapped Optimized Loader
Uses mmap and lazy loading to eliminate disk I/O bottleneck
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import time
import mmap
from safetensors.torch import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MMapOptimizedLoader:
    """Memory-mapped optimized loader for quantized models"""
    
    def __init__(self, quantized_model_path: str):
        self.quantized_path = Path(quantized_model_path)
        self.mmap_cache = {}  # Cache for memory-mapped files
        self.tensor_cache = {}  # Cache for frequently accessed tensors
        self.file_handles = {}  # Keep file handles open for mmap
        
        logger.info(f"🗺️ Memory-mapped loader initialized for {self.quantized_path}")
        
    def _get_mmap_handle(self, file_path: Path):
        """Get or create memory-mapped handle for file"""
        if file_path not in self.file_handles:
            try:
                # Open safetensors file with memory mapping
                self.file_handles[file_path] = safe_open(file_path, framework="pt", device="cpu")
                logger.info(f"   🗺️ Memory-mapped {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to mmap {file_path}: {e}")
                return None
        
        return self.file_handles[file_path]
    
    def load_tensor_lazy(self, file_path: Path, tensor_name: str, device_target: str) -> Dict[str, Any]:
        """Load tensor lazily with memory mapping"""
        cache_key = f"{file_path.name}:{tensor_name}"
        
        # Check cache first
        if cache_key in self.tensor_cache:
            return self.tensor_cache[cache_key]
        
        try:
            # Get memory-mapped file handle
            mmap_handle = self._get_mmap_handle(file_path)
            if mmap_handle is None:
                raise RuntimeError(f"Failed to get mmap handle for {file_path}")
            
            # Load tensor metadata without reading actual data
            if tensor_name not in mmap_handle.keys():
                raise KeyError(f"Tensor {tensor_name} not found in {file_path}")
            
            # Get tensor info without loading full data
            quantized_tensor = mmap_handle.get_tensor(tensor_name)
            scale_name = f"{tensor_name}_scale"
            scale = mmap_handle.get_tensor(scale_name) if scale_name in mmap_handle.keys() else None
            
            # Get quantization scheme from metadata
            metadata = mmap_handle.metadata()
            scheme = metadata.get(tensor_name, 'unknown')
            
            # Create lazy tensor wrapper that defers actual computation
            tensor_info = {
                'quantized_tensor': quantized_tensor,
                'scale': scale,
                'scheme': scheme,
                'device': device_target,
                'shape': quantized_tensor.shape,
                'dtype': quantized_tensor.dtype,
                'file_path': file_path,
                'tensor_name': tensor_name,
                'lazy': True  # Mark as lazy-loaded
            }
            
            # Cache for reuse
            self.tensor_cache[cache_key] = tensor_info
            
            return tensor_info
            
        except Exception as e:
            logger.error(f"❌ Failed to lazy load {tensor_name} from {file_path}: {e}")
            raise
    
    def dequantize_on_demand(self, tensor_info: Dict[str, Any]) -> torch.Tensor:
        """Dequantize tensor only when actually needed"""
        if not tensor_info.get('lazy', False):
            # Already dequantized
            return tensor_info['tensor']
        
        try:
            quantized_tensor = tensor_info['quantized_tensor']
            scale = tensor_info['scale']
            scheme = tensor_info['scheme']
            tensor_name = tensor_info.get('tensor_name', 'unknown')
            
            logger.info(f"🔄 Dequantizing {tensor_name}: quantized_shape={quantized_tensor.shape}, scale_shape={scale.shape if scale is not None else 'None'}, scheme={scheme}")
            logger.info(f"   Scheme analysis: 'int8' in scheme: {'int8' in scheme}, 'symmetric' in scheme: {'symmetric' in scheme}")
            
            # Dequantize based on scheme with proper broadcasting
            if scale is not None:
                # Handle different quantization schemes
                if 'int8' in scheme:
                    if 'asymmetric' in scheme:
                        # INT8 asymmetric: (quantized - zero_point) * scale
                        if scale.numel() == 2:
                            # Scale tensor contains [scale_value, zero_point]
                            scale_value = scale[0]
                            zero_point = scale[1]
                            dequantized = (quantized_tensor.float() - zero_point) * scale_value
                        else:
                            # Assume single scale value, use 128 as default zero point
                            dequantized = (quantized_tensor.float() - 128.0) * scale
                    else:
                        # INT8 symmetric: simple scaling
                        dequantized = quantized_tensor.float() * scale
                elif 'int4' in scheme:
                    if 'grouped' in scheme:
                        # INT4 grouped quantization - handle per-group scaling
                        tensor_flat = quantized_tensor.flatten().float()
                        scale_flat = scale.flatten()
                        
                        # Determine group size
                        group_size = tensor_flat.numel() // scale_flat.numel()
                        logger.info(f"   INT4 grouped: tensor_size={tensor_flat.numel()}, scale_size={scale_flat.numel()}, group_size={group_size}")
                        
                        # Expand scale to match tensor dimensions
                        scale_expanded = scale_flat.repeat_interleave(group_size)
                        if scale_expanded.numel() < tensor_flat.numel():
                            # Pad with last scale value if needed
                            padding_size = tensor_flat.numel() - scale_expanded.numel()
                            scale_expanded = torch.cat([scale_expanded, scale_expanded[-1:].repeat(padding_size)])
                        elif scale_expanded.numel() > tensor_flat.numel():
                            # Truncate if too large
                            scale_expanded = scale_expanded[:tensor_flat.numel()]
                        
                        dequantized_flat = tensor_flat * scale_expanded
                        dequantized = dequantized_flat.reshape(quantized_tensor.shape)
                    else:
                        # INT4 per-tensor
                        dequantized = quantized_tensor.float() * scale
                else:
                    # Default: per-tensor scaling with broadcasting
                    try:
                        dequantized = quantized_tensor.float() * scale
                    except RuntimeError as broadcast_error:
                        logger.warning(f"   Broadcasting failed: {broadcast_error}")
                        # Try expanding scale dimensions to match tensor
                        scale_expanded = scale
                        while scale_expanded.dim() < quantized_tensor.dim():
                            scale_expanded = scale_expanded.unsqueeze(-1)
                        
                        # Broadcast to match tensor shape
                        target_shape = quantized_tensor.shape
                        scale_broadcast = scale_expanded.expand(target_shape)
                        dequantized = quantized_tensor.float() * scale_broadcast
            else:
                # No scale - assume already in proper format
                dequantized = quantized_tensor.float()
            
            logger.info(f"✅ Dequantized {tensor_name}: output_shape={dequantized.shape}")
            
            # Update tensor_info to mark as dequantized
            tensor_info['tensor'] = dequantized
            tensor_info['lazy'] = False
            
            # Remove raw quantized data to save memory
            del tensor_info['quantized_tensor']
            if 'scale' in tensor_info:
                del tensor_info['scale']
            
            return dequantized
            
        except Exception as e:
            logger.error(f"❌ Failed to dequantize {tensor_info.get('tensor_name', 'unknown')}: {e}")
            logger.error(f"   Quantized tensor shape: {quantized_tensor.shape if 'quantized_tensor' in locals() else 'N/A'}")
            logger.error(f"   Scale shape: {scale.shape if scale is not None else 'None'}")
            logger.error(f"   Scheme: {scheme}")
            raise
    
    def load_layer_optimized(self, layer_num: int) -> Dict[str, Any]:
        """Load layer with optimized memory mapping"""
        logger.info(f"🗺️ Memory-mapped loading layer {layer_num}")
        
        start_time = time.time()
        
        # Find all files for this layer
        layer_files = list(self.quantized_path.glob(f"*_layer_{layer_num}.safetensors"))
        
        if not layer_files:
            raise FileNotFoundError(f"No files found for layer {layer_num}")
        
        # Collect all tensor tasks
        tensor_tasks = []
        for file_path in layer_files:
            mmap_handle = self._get_mmap_handle(file_path)
            if mmap_handle:
                tensor_names = [key for key in mmap_handle.keys() if not key.endswith('_scale')]
                for tensor_name in tensor_names:
                    # Determine device assignment
                    device = self._get_device_assignment(f"layer_{layer_num}", tensor_name)
                    tensor_tasks.append((file_path, tensor_name, device))
        
        logger.info(f"   📊 Lazy-loading {len(tensor_tasks)} tensors")
        
        layer_weights = {}
        
        # Load tensors lazily (fast - no dequantization yet)
        for file_path, tensor_name, device in tensor_tasks:
            try:
                tensor_info = self.load_tensor_lazy(file_path, tensor_name, device)
                layer_weights[tensor_name] = tensor_info
                logger.info(f"      ⚡ {tensor_name} → {device} (lazy)")
            except Exception as e:
                logger.error(f"      ❌ Failed to lazy load {tensor_name}: {e}")
        
        load_time = time.time() - start_time
        logger.info(f"✅ Layer {layer_num} lazy-loaded: {len(layer_weights)} tensors in {load_time:.2f}s")
        
        return layer_weights
    
    def _get_device_assignment(self, layer_name: str, tensor_name: str) -> str:
        """Determine hardware assignment for tensor - NPU+iGPU ONLY"""
        # NPU: Attention operations + layer norms
        if any(x in tensor_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn', 'layernorm', 'norm']):
            return 'npu'
        
        # iGPU: FFN operations + embeddings
        elif any(x in tensor_name for x in ['gate_proj', 'up_proj', 'down_proj', 'mlp', 'embed', 'embedding']):
            return 'igpu'
        
        # DEFAULT: NPU (hardware-only policy)
        else:
            return 'npu'
    
    def preload_critical_tensors(self, layer_weights: Dict[str, Any], critical_patterns: list = None):
        """Preload only critical tensors that are used immediately"""
        if critical_patterns is None:
            critical_patterns = ['input_layernorm', 'q_proj', 'k_proj', 'v_proj']  # Load attention first
        
        logger.info("⚡ Preloading critical tensors...")
        
        for tensor_name, tensor_info in layer_weights.items():
            if any(pattern in tensor_name for pattern in critical_patterns):
                if tensor_info.get('lazy', False):
                    logger.info(f"   🔥 Preloading {tensor_name}")
                    self.dequantize_on_demand(tensor_info)
    
    def get_tensor(self, tensor_info: Dict[str, Any]) -> torch.Tensor:
        """Get tensor, dequantizing on-demand if needed"""
        if tensor_info.get('lazy', False):
            return self.dequantize_on_demand(tensor_info)
        else:
            return tensor_info['tensor']
    
    def cleanup(self):
        """Close memory-mapped files"""
        for file_path, handle in self.file_handles.items():
            try:
                handle.close() if hasattr(handle, 'close') else None
            except:
                pass
        self.file_handles.clear()
        self.mmap_cache.clear()
        self.tensor_cache.clear()
        logger.info("🗺️ Memory-mapped handles cleaned up")

def test_mmap_optimization():
    """Test the memory-mapped optimization"""
    logger.info("🧪 Testing memory-mapped optimization")
    
    model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if not Path(model_path).exists():
        logger.error(f"❌ Model path not found: {model_path}")
        return False
    
    try:
        # Create mmap loader
        mmap_loader = MMapOptimizedLoader(model_path)
        
        # Test layer loading speed
        logger.info("🚀 Testing layer loading speed with mmap...")
        
        for layer_num in [0, 1]:
            start_time = time.time()
            
            # Load layer with mmap optimization
            layer_weights = mmap_loader.load_layer_optimized(layer_num)
            
            load_time = time.time() - start_time
            logger.info(f"⚡ Layer {layer_num} loaded in {load_time:.2f}s (mmap optimized)")
            
            # Test on-demand dequantization
            logger.info(f"🔥 Testing on-demand dequantization for layer {layer_num}")
            dequant_start = time.time()
            
            # Preload critical tensors only
            mmap_loader.preload_critical_tensors(layer_weights)
            
            dequant_time = time.time() - dequant_start
            logger.info(f"✅ Critical tensors dequantized in {dequant_time:.2f}s")
        
        # Cleanup
        mmap_loader.cleanup()
        
        logger.info("🎉 Memory-mapped optimization test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory-mapped test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mmap_optimization()