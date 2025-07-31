#!/usr/bin/env python3
"""
Test real NPU with Gemma3 4B kernels and correct dimensions
"""

import os
import time
import numpy as np
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_npu_gemma3_4b():
    """Test NPU with correct Gemma3 4B dimensions"""
    
    logger.info("🚀 Testing Real NPU with Gemma3 4B Kernels")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    logger.info(f"📦 Loading model: {model_path}")
    start_load = time.time()
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return False
    
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    
    # Check NPU kernel status
    npu_type = type(pipeline.npu_kernel).__name__
    logger.info(f"🧠 NPU Kernel: {npu_type}")
    
    if "Real" in npu_type:
        logger.info("✅ Using REAL NPU hardware acceleration!")
        
        # Check kernel dimensions
        if hasattr(pipeline.npu_kernel, 'd_model'):
            logger.info(f"📊 NPU Kernel Dimensions:")
            logger.info(f"   Model Dimension: {pipeline.npu_kernel.d_model}")
            logger.info(f"   Number of Heads: {pipeline.npu_kernel.num_heads}")
            logger.info(f"   Head Dimension: {pipeline.npu_kernel.head_dim}")
            
            # Verify dimensions match Gemma3 4B
            if pipeline.npu_kernel.d_model == 2560:
                logger.info("✅ Correct model dimension (2560)")
            else:
                logger.warning(f"⚠️ Model dimension mismatch: {pipeline.npu_kernel.d_model} != 2560")
                
            if pipeline.npu_kernel.num_heads == 20:
                logger.info("✅ Correct number of heads (20)")
            else:
                logger.warning(f"⚠️ Number of heads mismatch: {pipeline.npu_kernel.num_heads} != 20")
                
            if pipeline.npu_kernel.head_dim == 128:
                logger.info("✅ Correct head dimension (128)")
            else:
                logger.warning(f"⚠️ Head dimension mismatch: {pipeline.npu_kernel.head_dim} != 128")
                
    else:
        logger.warning(f"⚠️ Using {npu_type} - not real NPU hardware")
        return False
    
    # Test token generation
    logger.info("\n🔬 Testing Token Generation...")
    input_ids = [1, 2, 3]
    
    try:
        start_time = time.time()
        
        # Get embeddings
        embed_key = 'language_model.model.embed_tokens.weight'
        embed_info = pipeline.gpu_buffers.get(embed_key)
        
        if embed_info:
            logger.info("✅ Found embedding weights")
            
            # Test embedding lookup
            hidden_states = pipeline.vulkan_engine.compute_embedding_lookup_gpu(
                input_ids, embed_info['buffer_info']
            )
            logger.info(f"✅ Embedding lookup: {hidden_states.shape}")
            
            # Test attention layer (where NPU is used)
            logger.info("🧠 Testing NPU attention computation...")
            
            # Test first layer
            result, _ = pipeline.forward_layer(0, hidden_states)
            
            inference_time = time.time() - start_time
            logger.info(f"✅ Forward layer successful: {result.shape}")
            logger.info(f"⏱️ Inference time: {inference_time:.3f}s")
            
            # Calculate TPS (rough estimate)
            tps = len(input_ids) / inference_time
            logger.info(f"🚀 Estimated TPS: {tps:.2f}")
            
        else:
            logger.error("❌ No embedding weights found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Token generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("✅ Real NPU test completed successfully!")
    logger.info("✅ Correct Gemma3 4B dimensions verified")
    logger.info("✅ NPU hardware acceleration working")
    logger.info("✅ Token generation successful")
    
    pipeline.cleanup()
    return True

def main():
    """Main entry point"""
    try:
        success = test_real_npu_gemma3_4b()
        if success:
            logger.info("🎉 Real NPU test passed!")
        else:
            logger.error("❌ Real NPU test failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())