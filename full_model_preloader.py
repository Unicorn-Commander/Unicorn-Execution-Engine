#!/usr/bin/env python3
"""
Full Model Preloader - Ollama-style
Load entire model into 96GB RAM (40GB GTT + 56GB system) to eliminate disk I/O
"""

import torch
import time
import logging
import sys
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullModelPreloader:
    """Ollama-style model preloader - load entire model into RAM"""
    
    def __init__(self, model_path, use_fp16=True):
        self.model_path = Path(model_path)
        self.use_fp16 = use_fp16
        self.preloaded_model = {}
        self.memory_stats = {}
        
        # Memory architecture (96GB total)
        self.memory_layout = {
            'gtt_memory': 40,      # 40GB GTT (GPU-accessible)
            'system_memory': 56,   # 56GB system RAM
            'total_memory': 96,    # 96GB total
            'vram_assigned': 16    # 16GB VRAM allocation from GTT
        }
        
        logger.info(f"🦄 Full Model Preloader initialized")
        logger.info(f"   📁 Model path: {self.model_path}")
        logger.info(f"   🧠 Memory layout: {self.memory_layout}")
        
    def preload_entire_model(self):
        """Load entire model into RAM like Ollama"""
        logger.info("🚀 Starting full model preload (Ollama-style)")
        
        start_time = time.time()
        
        # Step 1: Load shared weights into GTT memory
        shared_weights = self._preload_shared_weights()
        
        # Step 2: Load all layers into system memory
        all_layers = self._preload_all_layers()
        
        # Step 3: Organize preloaded model
        self.preloaded_model = {
            'shared_weights': shared_weights,
            'layers': all_layers,
            'metadata': {
                'model_path': str(self.model_path),
                'use_fp16': self.use_fp16,
                'layer_count': len(all_layers),
                'load_time': time.time() - start_time
            }
        }
        
        # Step 4: Calculate memory usage
        self._calculate_memory_usage()
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Full model preload completed in {total_time:.2f}s")
        logger.info(f"   📊 Layers loaded: {len(all_layers)}")
        logger.info(f"   🧠 Memory usage: {self.memory_stats}")
        
        return self.preloaded_model
        
    def _preload_shared_weights(self):
        """Load shared weights into GTT memory"""
        logger.info("📦 Preloading shared weights into GTT memory...")
        
        from quantized_gemma27b_npu_igpu_loader import QuantizedGemma27BNPUIGPULoader
        
        # Create loader just to get shared weights
        loader = QuantizedGemma27BNPUIGPULoader(str(self.model_path), use_fp16=self.use_fp16)
        
        # Load model info to get shared weights
        model_info = loader.load_model_info()
        shared_weights = model_info.get('shared_weights', {})
        
        logger.info(f"✅ Shared weights loaded: {len(shared_weights)} tensors")
        
        return shared_weights
        
    def _preload_all_layers(self):
        """Load all layers into system memory with parallel processing"""
        logger.info("📦 Preloading ALL layers into system memory...")
        
        from quantized_gemma27b_npu_igpu_loader import QuantizedGemma27BNPUIGPULoader
        
        # Create loader
        loader = QuantizedGemma27BNPUIGPULoader(str(self.model_path), use_fp16=self.use_fp16)
        
        # Get layer count
        layer_count = 62  # Gemma 3 27B has 62 layers
        
        # Pre-allocate layers dictionary
        all_layers = {}
        
        # Use all available CPU cores for parallel loading
        num_workers = min(multiprocessing.cpu_count(), 16)  # Cap at 16 to avoid overload
        
        logger.info(f"🔄 Loading {layer_count} layers with {num_workers} parallel workers...")
        
        def load_single_layer(layer_num):
            """Load a single layer and return (layer_num, layer_data)"""
            try:
                logger.info(f"   📥 Loading layer {layer_num}...")
                start_time = time.time()
                
                layer_data = loader.load_layer(layer_num)
                load_time = time.time() - start_time
                
                logger.info(f"   ✅ Layer {layer_num} loaded in {load_time:.2f}s")
                return layer_num, layer_data
                
            except Exception as e:
                logger.error(f"   ❌ Layer {layer_num} failed: {e}")
                return layer_num, None
        
        # Load all layers in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all layer loading tasks
            futures = []
            for layer_num in range(layer_count):
                future = executor.submit(load_single_layer, layer_num)
                futures.append(future)
            
            # Collect results
            for future in futures:
                layer_num, layer_data = future.result()
                if layer_data is not None:
                    all_layers[layer_num] = layer_data
                    
        logger.info(f"✅ All layers preloaded: {len(all_layers)}/{layer_count} layers")
        
        return all_layers
        
    def _calculate_memory_usage(self):
        """Calculate memory usage of preloaded model"""
        logger.info("📊 Calculating memory usage...")
        
        shared_size = 0
        layer_size = 0
        
        # Calculate shared weights size
        for name, weight_info in self.preloaded_model['shared_weights'].items():
            if 'tensor' in weight_info:
                tensor = weight_info['tensor']
                size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                shared_size += size_mb
        
        # Calculate layer size
        for layer_num, layer_data in self.preloaded_model['layers'].items():
            for name, weight_info in layer_data.items():
                if 'tensor' in weight_info:
                    tensor = weight_info['tensor']
                    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                    layer_size += size_mb
        
        total_size = shared_size + layer_size
        
        self.memory_stats = {
            'shared_weights_mb': shared_size,
            'layers_mb': layer_size,
            'total_mb': total_size,
            'total_gb': total_size / 1024,
            'gtt_utilization': min(100, (total_size / 1024) / self.memory_layout['gtt_memory'] * 100),
            'system_utilization': min(100, (total_size / 1024) / self.memory_layout['system_memory'] * 100)
        }
        
        logger.info(f"📊 Memory usage calculated:")
        logger.info(f"   🧠 Shared weights: {shared_size:.1f} MB")
        logger.info(f"   📦 Layers: {layer_size:.1f} MB")
        logger.info(f"   💾 Total: {total_size/1024:.1f} GB")
        logger.info(f"   📊 GTT utilization: {self.memory_stats['gtt_utilization']:.1f}%")
        
    def get_layer_instantly(self, layer_num):
        """Get layer from preloaded memory - INSTANT ACCESS"""
        if layer_num in self.preloaded_model['layers']:
            return self.preloaded_model['layers'][layer_num]
        else:
            raise KeyError(f"Layer {layer_num} not found in preloaded model")
            
    def get_shared_weights(self):
        """Get shared weights from preloaded memory"""
        return self.preloaded_model['shared_weights']
        
    def get_memory_stats(self):
        """Get memory usage statistics"""
        return self.memory_stats
        
    def create_fast_loader(self):
        """Create a fast loader that uses preloaded model"""
        
        class FastLoader:
            def __init__(self, preloaded_model):
                self.preloaded_model = preloaded_model
                self.shared_weights = preloaded_model['shared_weights']
                self.layers = preloaded_model['layers']
                
            def load_layer(self, layer_num):
                """Load layer from memory - INSTANT"""
                if layer_num in self.layers:
                    return self.layers[layer_num]
                else:
                    raise KeyError(f"Layer {layer_num} not preloaded")
                    
            def __call__(self, layer_num):
                """Make it callable like the original loader"""
                return self.load_layer(layer_num)
                
        return FastLoader(self.preloaded_model)

def test_full_preloader():
    """Test the full model preloader"""
    logger.info("🧪 Testing full model preloader")
    
    model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
    
    if not Path(model_path).exists():
        logger.error(f"❌ Model path not found: {model_path}")
        return False
    
    # Create preloader
    preloader = FullModelPreloader(model_path, use_fp16=True)
    
    # Preload entire model
    try:
        preloaded_model = preloader.preload_entire_model()
        logger.info("✅ Full model preloaded successfully")
        
        # Test instant access
        fast_loader = preloader.create_fast_loader()
        
        # Test layer access speed
        logger.info("🚀 Testing instant layer access...")
        for layer_num in [0, 1, 2, 3]:
            start_time = time.time()
            layer_data = fast_loader(layer_num)
            access_time = time.time() - start_time
            logger.info(f"   ⚡ Layer {layer_num} accessed in {access_time*1000:.2f}ms")
            
        logger.info("🎉 Full preloader test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full preloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_preloader()