#!/usr/bin/env python3
"""
Test REAL performance with NPU+iGPU only
No simulation, no dummy data - just real inference
"""

import os
import time
import numpy as np
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_inference():
    """Test real inference performance"""
    
    logger.info("🚀 REAL PERFORMANCE TEST - NPU+iGPU ONLY")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    
    logger.info(f"📦 Loading model: {model_path}")
    start_load = time.time()
    
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return
    
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    
    # Check NPU status
    npu_type = type(pipeline.npu_kernel).__name__
    logger.info(f"\n🧠 NPU Status: {npu_type}")
    
    if "Simulated" in npu_type:
        logger.warning("⚠️ WARNING: Using SIMULATED NPU - not real hardware!")
        logger.warning("⚠️ Real NPU initialization failed")
        logger.warning("⚠️ Performance numbers are NOT representative of real NPU")
    else:
        logger.info("✅ Using REAL NPU hardware")
    
    # Test single token generation
    logger.info("\n📊 Testing single token generation...")
    
    # Simple input
    input_ids = [1, 2, 3, 4, 5]  # Simple test sequence
    
    # Get embeddings
    embed_info = pipeline.gpu_buffers.get('language_model.model.embed_tokens.weight')
    if not embed_info:
        embed_info = pipeline.gpu_buffers.get('shared_language_model.model.embed_tokens.weight')
    
    if not embed_info:
        logger.error("❌ No embedding buffer found")
        return
    
    # Time single forward pass
    logger.info("\n⏱️ Timing single forward pass through all 34 layers...")
    
    start_time = time.time()
    
    # Get initial embeddings
    hidden_states = pipeline.vulkan_engine.compute_embedding_lookup_gpu(
        input_ids, embed_info['buffer_info']
    )
    
    # Process through all layers
    layer_times = []
    for layer_idx in range(34):
        layer_start = time.time()
        hidden_states, _ = pipeline.forward_layer(layer_idx, hidden_states)
        layer_time = time.time() - layer_start
        layer_times.append(layer_time)
        
        if layer_idx == 0:
            logger.info(f"   Layer 0 time: {layer_time*1000:.2f}ms")
    
    total_time = time.time() - start_time
    
    # Calculate TPS
    tokens_generated = 1
    tps = tokens_generated / total_time
    
    logger.info(f"\n📊 REAL PERFORMANCE RESULTS:")
    logger.info(f"   Total time for 34 layers: {total_time:.3f}s")
    logger.info(f"   Average layer time: {np.mean(layer_times)*1000:.2f}ms")
    logger.info(f"   Theoretical TPS: {tps:.2f}")
    
    # Breakdown by component
    logger.info(f"\n🔍 Performance Breakdown:")
    
    # Check if NPU is being used
    if hasattr(pipeline, 'npu_times') and pipeline.npu_times:
        avg_npu_time = np.mean(pipeline.npu_times)
        logger.info(f"   NPU average: {avg_npu_time*1000:.2f}ms per attention")
    
    # Memory usage
    total_gpu_mb = sum(info['size_mb'] for info in pipeline.gpu_buffers.values())
    logger.info(f"   GPU Memory: {total_gpu_mb:.1f}MB")
    
    # Final verdict
    logger.info(f"\n" + "=" * 60)
    logger.info(f"🎯 REAL TPS with current setup: {tps:.2f}")
    
    if tps < 1.0:
        logger.warning("⚠️ Performance is below 1 TPS")
        logger.warning("⚠️ This indicates significant bottlenecks")
        
        if "Simulated" in npu_type:
            logger.warning("\n❌ CRITICAL: NPU is NOT working!")
            logger.warning("❌ Using CPU simulation instead of real NPU")
            logger.warning("❌ This explains the poor performance")
            logger.info("\n💡 To fix NPU:")
            logger.info("1. Need proper XCLBIN file compiled with mlir-aie")
            logger.info("2. Current 'kernels' are just dummy binary files")
            logger.info("3. Real NPU would give 10-100x speedup")
    
    pipeline.cleanup()

def main():
    """Main entry point"""
    try:
        test_real_inference()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())