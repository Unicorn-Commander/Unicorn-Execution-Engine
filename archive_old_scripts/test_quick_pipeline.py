#!/usr/bin/env python3
"""Quick pipeline test - just initialization"""

import logging
import time
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quick_pipeline():
    """Test pipeline initialization only"""
    logger.info("🚀 Testing pipeline initialization...")
    
    start_time = time.time()
    
    try:
        # Initialize pipeline
        pipeline = PureHardwarePipelineFixed()
        model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
        
        result = pipeline.initialize(model_path)
        
        elapsed = time.time() - start_time
        
        if result:
            logger.info(f"✅ Pipeline initialized successfully in {elapsed:.1f}s!")
            logger.info(f"   NPU status: {'READY' if pipeline.npu_kernel else 'NOT AVAILABLE'}")
            logger.info(f"   GPU status: {'READY' if pipeline.vulkan_engine else 'NOT AVAILABLE'}")
            
            # Quick check of loaded layers
            loaded_layers = len(pipeline.layer_weights_gpu)
            logger.info(f"   Layers loaded to GPU: {loaded_layers}/62")
            
            # Check memory usage
            if hasattr(pipeline.vulkan_engine, 'memory_usage_mb'):
                logger.info(f"   GPU memory used: {pipeline.vulkan_engine.memory_usage_mb:.1f}MB")
                
            return True
        else:
            logger.error(f"❌ Pipeline initialization failed after {elapsed:.1f}s")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()

if __name__ == "__main__":
    success = test_quick_pipeline()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")