#!/usr/bin/env python3
"""
Fix the GPU loading issue by properly connecting LightningFastLoader and PureHardwarePipelineFixed
"""

import os
import shutil

def fix_gpu_loading():
    """Apply the fix to properly populate gpu_buffers"""
    
    # Read the current pipeline
    with open('pure_hardware_pipeline_fixed.py', 'r') as f:
        pipeline_content = f.read()
    
    # Fix 1: Update the _load_model_to_gpu method to handle the actual data structure
    # The loader returns weight_info with 'tensor' key containing (buffer, memory, size_bytes)
    
    # Replace the shared weights loading section
    old_shared_section = """        # First, handle shared weights (embeddings, norms)
        logger.info("📦 Loading shared weights to GPU...")
        for weight_name, weight_info in self.shared_weights.items():
            if isinstance(weight_info, dict) and 'buffer' in weight_info:
                # This is a GPU-allocated tensor, store its info
                buffer_key = f"shared_{weight_name}"
                self.gpu_buffers[buffer_key] = {
                    'buffer_info': (weight_info['buffer'], weight_info['memory'], weight_info['size_bytes']),
                    'shape': weight_info['original_shape'],
                    'dtype': weight_info['scheme'],
                    'size_mb': weight_info['size_bytes'] / (1024 * 1024),
                    'weight_info': weight_info,
                    'needs_transpose': False # Shared weights typically don't need transpose
                }
                size_mb = weight_info['size_bytes'] / (1024 * 1024)
                if 'embed_tokens' in weight_name or 'norm' in weight_name:
                    vram_used_mb += size_mb
                    logger.info(f"  ✅ {weight_name}: {size_mb:.1f}MB → VRAM")"""
    
    new_shared_section = """        # First, handle shared weights (embeddings, norms)
        logger.info("📦 Loading shared weights to GPU...")
        for weight_name, weight_info in self.shared_weights.items():
            if isinstance(weight_info, dict) and 'tensor' in weight_info:
                # The tensor key contains (buffer, memory, size_bytes) tuple
                buffer, memory, size_bytes = weight_info['tensor']
                buffer_key = f"shared_{weight_name}"
                self.gpu_buffers[buffer_key] = {
                    'buffer_info': (buffer, memory, size_bytes),
                    'shape': weight_info.get('original_shape', weight_info.get('shape')),
                    'dtype': weight_info.get('scheme', 'float32'),
                    'size_mb': size_bytes / (1024 * 1024),
                    'weight_info': weight_info,
                    'needs_transpose': False # Shared weights typically don't need transpose
                }
                size_mb = size_bytes / (1024 * 1024)
                if 'embed_tokens' in weight_name or 'norm' in weight_name:
                    vram_used_mb += size_mb
                    logger.info(f"  ✅ {weight_name}: {size_mb:.1f}MB → VRAM")"""
    
    pipeline_content = pipeline_content.replace(old_shared_section, new_shared_section)
    
    # Fix 2: Update the layer weights loading section
    old_layer_section = """            # Load each weight in the layer directly to GPU
            for weight_name, weight_info in layer_weights.items():
                if weight_name.startswith('language_model') and 'buffer' in weight_info:
                    buffer_key = f"layer_{layer_idx}_{weight_name}"
                    self.gpu_buffers[buffer_key] = {
                        'buffer_info': (weight_info['buffer'], weight_info['memory'], weight_info['size_bytes']),
                        'shape': weight_info['original_shape'],
                        'dtype': weight_info['scheme'],
                        'size_mb': weight_info['size_bytes'] / (1024 * 1024),
                        'weight_info': weight_info,
                        'needs_transpose': 'proj.weight' in weight_name # Check for transpose
                    }
                    size_mb = weight_info['size_bytes'] / (1024 * 1024)
                    layer_size_mb += size_mb
                    layer_gpu_weights[weight_name] = buffer_key"""
    
    new_layer_section = """            # Load each weight in the layer directly to GPU
            for weight_name, weight_info in layer_weights.items():
                # Skip vision components
                if 'vision' in weight_name:
                    continue
                    
                if isinstance(weight_info, dict) and 'buffer' in weight_info:
                    # Layer loader returns different structure - buffer/memory/size_bytes directly
                    buffer = weight_info['buffer']
                    memory = weight_info['memory']
                    size_bytes = weight_info['size_bytes']
                    buffer_key = f"layer_{layer_idx}_{weight_name}"
                    self.gpu_buffers[buffer_key] = {
                        'buffer_info': (buffer, memory, size_bytes),
                        'shape': weight_info.get('original_shape', weight_info.get('shape')),
                        'dtype': weight_info.get('scheme', 'float32'),
                        'size_mb': size_bytes / (1024 * 1024),
                        'weight_info': weight_info,
                        'needs_transpose': 'proj.weight' in weight_name # Check for transpose
                    }
                    size_mb = size_bytes / (1024 * 1024)
                    layer_size_mb += size_mb
                    layer_gpu_weights[weight_name] = buffer_key"""
    
    pipeline_content = pipeline_content.replace(old_layer_section, new_layer_section)
    
    # Fix 3: Update the embedding lookup to properly find the embedding weights
    old_embed_lookup = """        # Get embedding weights (cache for reuse)
        if not hasattr(self, '_cached_embed_weights_info'):
            self._cached_embed_weights_info = self.gpu_buffers.get('shared_language_model.model.embed_tokens.weight')
            if self._cached_embed_weights_info:
                logger.info(f"   ✅ Cached embedding weights info: shape {self._cached_embed_weights_info['shape']}")
            else:
                logger.warning("   ⚠️ Embedding weights not found in GPU buffers.")"""
    
    new_embed_lookup = """        # Get embedding weights (cache for reuse)
        if not hasattr(self, '_cached_embed_weights_info'):
            # Try different possible keys for embedding weights
            embed_keys = [
                'shared_language_model.model.embed_tokens.weight',
                'shared_embed_tokens.weight',
                'shared_embeddings.weight'
            ]
            for key in embed_keys:
                if key in self.gpu_buffers:
                    self._cached_embed_weights_info = self.gpu_buffers[key]
                    logger.info(f"   ✅ Found embedding weights at '{key}': shape {self._cached_embed_weights_info['shape']}")
                    break
            
            if not hasattr(self, '_cached_embed_weights_info') or self._cached_embed_weights_info is None:
                # Debug: show what keys we have
                logger.error("   ❌ Embedding weights not found in GPU buffers.")
                logger.error(f"   Available keys: {list(self.gpu_buffers.keys())[:10]}...")
                self._cached_embed_weights_info = None"""
    
    pipeline_content = pipeline_content.replace(old_embed_lookup, new_embed_lookup)
    
    # Write the fixed version
    backup_path = 'pure_hardware_pipeline_fixed_backup.py'
    shutil.copy('pure_hardware_pipeline_fixed.py', backup_path)
    
    with open('pure_hardware_pipeline_fixed.py', 'w') as f:
        f.write(pipeline_content)
    
    print("✅ Applied GPU loading fix to pure_hardware_pipeline_fixed.py")
    print(f"📁 Original backed up to: {backup_path}")
    print("\nKey changes:")
    print("1. Fixed shared weights loading to handle 'tensor' key with tuple")
    print("2. Fixed layer weights loading to properly extract buffer/memory/size_bytes")
    print("3. Added better embedding weights discovery with debug logging")
    print("\nNow test with: python3 benchmark_final_performance.py")

if __name__ == "__main__":
    fix_gpu_loading()