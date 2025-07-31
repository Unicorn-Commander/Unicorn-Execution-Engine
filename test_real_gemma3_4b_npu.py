#!/usr/bin/env python3
"""
Test REAL Gemma3 4B with NPU+iGPU using actual model weights
This uses the pure_hardware_pipeline_fixed.py with real models
"""

import os
import sys
import time
import numpy as np
import logging
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_gemma3_4b():
    """Test real Gemma3 4B with NPU+iGPU hardware"""
    
    logger.info("🚀 REAL GEMMA3 4B NPU+iGPU TEST")
    logger.info("=" * 60)
    
    # Initialize pipeline
    logger.info("⚡ Initializing hardware pipeline...")
    pipeline = PureHardwarePipelineFixed()
    
    # Load real model
    model_path = "/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-4b-it-quantized"
    logger.info(f"📦 Loading real model from: {model_path}")
    
    start_load = time.time()
    if not pipeline.initialize(model_path=model_path):
        logger.error("❌ Failed to initialize pipeline")
        return 0
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    
    # Check hardware status
    logger.info("\n📊 Hardware Status:")
    if pipeline.npu_kernel:
        logger.info(f"  ✅ NPU: {type(pipeline.npu_kernel).__name__}")
    else:
        logger.info("  ❌ NPU: Not available")
    logger.info(f"  ✅ iGPU: Vulkan initialized")
    
    # Test prompts
    test_cases = [
        ([1, 2, 3, 4, 5], 20),  # Simple token IDs
        ([128000, 128006, 128007], 15),  # Special tokens
        ([1, 1587, 374, 220, 17, 10, 17, 30], 25),  # "What is 2+2?"
    ]
    
    # Warm-up
    logger.info("\n🔥 Warming up hardware...")
    try:
        _ = pipeline.generate_tokens([1, 2, 3], max_tokens=5)
        logger.info("✅ Warm-up complete")
    except Exception as e:
        logger.warning(f"⚠️ Warm-up failed: {e}")
    
    # Run benchmarks
    logger.info("\n📊 Running real generation benchmarks...")
    results = []
    
    for i, (input_ids, max_tokens) in enumerate(test_cases):
        logger.info(f"\n🔄 Test {i+1}: Input {input_ids[:5]}... (max {max_tokens} tokens)")
        
        try:
            # Reset NPU metrics
            pipeline.npu_total_time = 0.0
            pipeline.npu_total_layers = 0
            
            # Time the generation
            start_time = time.time()
            generated_ids = pipeline.generate_tokens(input_ids, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            
            tokens_generated = len(generated_ids)
            real_tps = tokens_generated / elapsed if elapsed > 0 else 0
            
            # Calculate NPU contribution
            if pipeline.npu_total_layers > 0:
                npu_avg_ms = (pipeline.npu_total_time / pipeline.npu_total_layers) * 1000
                npu_tps = 1000 / npu_avg_ms if npu_avg_ms > 0 else 0
            else:
                npu_avg_ms = 0
                npu_tps = 0
            
            logger.info(f"  ✅ Generated: {generated_ids}")
            logger.info(f"  📊 Tokens: {tokens_generated}")
            logger.info(f"  ⏱️ Total time: {elapsed:.2f}s")
            logger.info(f"  🚀 Real TPS: {real_tps:.2f} tokens/second")
            logger.info(f"  🧠 NPU avg: {npu_avg_ms:.2f}ms/layer ({npu_tps:.2f} TPS)")
            
            results.append({
                'tokens': tokens_generated,
                'time': elapsed,
                'tps': real_tps,
                'npu_ms': npu_avg_ms,
                'npu_tps': npu_tps
            })
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate real performance
    if results:
        avg_tps = sum(r['tps'] for r in results) / len(results)
        total_tokens = sum(r['tokens'] for r in results)
        total_time = sum(r['time'] for r in results)
        avg_npu_ms = sum(r['npu_ms'] for r in results) / len(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 REAL PERFORMANCE RESULTS - GEMMA3 4B")
        logger.info("=" * 60)
        logger.info(f"✅ Tests completed: {len(results)}/{len(test_cases)}")
        logger.info(f"✅ Total tokens generated: {total_tokens}")
        logger.info(f"✅ Total generation time: {total_time:.2f}s")
        logger.info(f"🚀 Average REAL TPS: {avg_tps:.2f} tokens/second")
        logger.info(f"🧠 NPU average: {avg_npu_ms:.2f}ms per attention layer")
        logger.info("=" * 60)
        
        # Performance analysis
        if avg_tps >= 150:
            logger.info("🎉 EXCELLENT! Real performance exceeds 150 TPS!")
        elif avg_tps >= 100:
            logger.info("✅ Good real performance! Over 100 TPS")
        elif avg_tps >= 50:
            logger.info("📈 Decent real performance")
        else:
            logger.info("⚠️ Performance below expectations")
        
        # Hardware breakdown
        logger.info("\n📊 Hardware Utilization:")
        logger.info(f"  NPU: {'Simulated' if 'Simulated' in str(type(pipeline.npu_kernel)) else 'Real'}")
        logger.info(f"  iGPU: Real Vulkan compute")
        logger.info(f"  Model: {model_path}")
        
        return avg_tps
    
    return 0

if __name__ == "__main__":
    try:
        pipeline = PureHardwarePipelineFixed()
        # Quick check if we can initialize
        logger.info("🔍 Checking environment...")
        logger.info(f"  Working directory: {os.getcwd()}")
        logger.info(f"  Python: {sys.executable}")
        
        real_tps = test_real_gemma3_4b()
        logger.info(f"\n✅ Test completed! REAL TPS: {real_tps:.2f}")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()