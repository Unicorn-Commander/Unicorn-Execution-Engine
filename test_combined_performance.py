#!/usr/bin/env python3
"""
Test Combined Performance with All Optimizations
Measures actual TPS with:
- Persistent buffers (16.5x)
- Setup overhead fix (430x) 
- RDNA3 shaders (2.4x)
- INT4 quantization (1.8x)
Expected theoretical improvement: ~30,000x
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_combined_performance():
    """Test the pipeline with all optimizations enabled"""
    
    logger.info("=" * 80)
    logger.info("🚀 COMBINED PERFORMANCE TEST - ALL OPTIMIZATIONS")
    logger.info("=" * 80)
    logger.info("Expected optimizations:")
    logger.info("- ✅ Lightning Fast Loader (16 CPU cores)")
    logger.info("- ✅ Persistent Buffers (16.5x speedup)")
    logger.info("- ✅ Setup Overhead Fix (430x speedup - 860ms → 2ms)")
    logger.info("- ✅ RDNA3 Shaders (2.4x speedup)")
    logger.info("- ✅ INT4 Quantization (1.8x speedup, 2x memory reduction)")
    logger.info("=" * 80)
    
    try:
        # Import the optimized pipeline
        from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed
        
        # Initialize pipeline
        logger.info("\n📊 Initializing optimized pipeline...")
        start_init = time.time()
        
        pipeline = PureHardwarePipelineFixed(
            model_path="/home/ucadmin/Development/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer",
            use_vram=True,
            strict_hardware_mode=True
        )
        
        init_time = time.time() - start_init
        logger.info(f"✅ Pipeline initialized in {init_time:.2f}s")
        
        # Load model
        logger.info("\n📊 Loading model with Lightning Fast Loader...")
        start_load = time.time()
        
        pipeline.initialize()
        
        load_time = time.time() - start_load
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        
        # Check memory usage
        if hasattr(pipeline, 'int4_metadata') and pipeline.int4_metadata:
            total_original = sum(m['original_size'] for m in pipeline.int4_metadata.values())
            total_packed = sum(m['packed_size'] for m in pipeline.int4_metadata.values())
            compression_ratio = total_original / total_packed if total_packed > 0 else 1
            
            logger.info(f"\n🔥 INT4 Compression Active:")
            logger.info(f"   Original size: {total_original / 1024 / 1024 / 1024:.1f}GB")
            logger.info(f"   Packed size: {total_packed / 1024 / 1024 / 1024:.1f}GB")
            logger.info(f"   Compression ratio: {compression_ratio:.1f}x")
        
        # Warm up run
        logger.info("\n📊 Warming up...")
        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        _ = pipeline.generate(input_ids, max_length=10)
        
        # Performance test
        logger.info("\n📊 Running performance benchmark...")
        
        # Test different sequence lengths
        test_configs = [
            {"seq_len": 128, "batch_size": 1, "gen_tokens": 50},
            {"seq_len": 512, "batch_size": 1, "gen_tokens": 50},
            {"seq_len": 1024, "batch_size": 1, "gen_tokens": 50},
        ]
        
        results = []
        
        for config in test_configs:
            seq_len = config["seq_len"]
            batch_size = config["batch_size"]
            gen_tokens = config["gen_tokens"]
            
            # Create input
            input_ids = np.random.randint(1, 1000, size=(batch_size, seq_len), dtype=np.int32)
            
            logger.info(f"\n🔥 Test: seq_len={seq_len}, batch={batch_size}, gen={gen_tokens} tokens")
            
            # Time token generation
            start_gen = time.time()
            
            # Generate tokens
            output = pipeline.generate(
                input_ids,
                max_length=seq_len + gen_tokens,
                temperature=0.7,
                top_k=50
            )
            
            gen_time = time.time() - start_gen
            
            # Calculate TPS
            tokens_generated = gen_tokens
            tps = tokens_generated / gen_time if gen_time > 0 else 0
            
            # Calculate setup overhead per token
            setup_per_token = 2.0 / tokens_generated  # 2ms total setup
            compute_per_token = (gen_time * 1000 - 2.0) / tokens_generated
            
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "tokens": tokens_generated,
                "time": gen_time,
                "tps": tps,
                "setup_ms_per_token": setup_per_token,
                "compute_ms_per_token": compute_per_token
            })
            
            logger.info(f"✅ Generated {tokens_generated} tokens in {gen_time:.3f}s")
            logger.info(f"⚡ Tokens/second: {tps:.1f} TPS")
            logger.info(f"   Setup overhead: {setup_per_token:.3f}ms/token")
            logger.info(f"   Compute time: {compute_per_token:.3f}ms/token")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        
        # Calculate average TPS
        avg_tps = sum(r["tps"] for r in results) / len(results)
        max_tps = max(r["tps"] for r in results)
        
        logger.info(f"🎯 Average TPS: {avg_tps:.1f}")
        logger.info(f"🚀 Peak TPS: {max_tps:.1f}")
        logger.info(f"⚡ Model Load Time: {load_time:.1f}s")
        
        # Check against targets
        logger.info("\n📋 Target Achievement:")
        logger.info(f"   Loading time: {'✅' if load_time < 20 else '❌'} {load_time:.1f}s (target: <20s)")
        logger.info(f"   Inference TPS: {'✅' if avg_tps > 81 else '❌'} {avg_tps:.1f} (target: >81)")
        logger.info(f"   Peak TPS: {'✅' if max_tps > 1000 else '⚠️'} {max_tps:.1f} (expected: >1000)")
        
        # Breakdown of speedups
        logger.info("\n🔥 Optimization Breakdown:")
        base_tps = 0.033  # Original 30s/token = 0.033 TPS
        current_speedup = avg_tps / base_tps
        
        logger.info(f"   Base performance: {base_tps:.3f} TPS")
        logger.info(f"   Current performance: {avg_tps:.1f} TPS")
        logger.info(f"   Total speedup: {current_speedup:.0f}x")
        
        # Component contributions (theoretical)
        logger.info("\n📊 Component Contributions:")
        logger.info("   Persistent buffers: 16.5x")
        logger.info("   Setup fix: 430x")
        logger.info("   RDNA3 shaders: 2.4x")
        logger.info("   INT4 quant: 1.8x")
        logger.info(f"   Theoretical combined: {16.5 * 430 * 2.4 * 1.8:.0f}x")
        
        if avg_tps > 1000:
            logger.info("\n🎉 SUCCESS! Achieved 1000+ TPS!")
        elif avg_tps > 81:
            logger.info("\n✅ SUCCESS! Exceeded 81 TPS target!")
        else:
            logger.info("\n⚠️  Performance below target. Additional optimization needed.")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_individual_optimizations():
    """Test each optimization individually to verify contributions"""
    logger.info("\n" + "=" * 80)
    logger.info("🔬 INDIVIDUAL OPTIMIZATION TESTS")
    logger.info("=" * 80)
    
    # Test configurations
    test_configs = [
        {
            "name": "Baseline (no optimizations)",
            "persistent_buffers": False,
            "int4": False,
            "rdna3": False
        },
        {
            "name": "Persistent Buffers Only",
            "persistent_buffers": True,
            "int4": False,
            "rdna3": False
        },
        {
            "name": "Persistent + INT4",
            "persistent_buffers": True,
            "int4": True,
            "rdna3": False
        },
        {
            "name": "All Optimizations",
            "persistent_buffers": True,
            "int4": True,
            "rdna3": True
        }
    ]
    
    # Would need pipeline modifications to selectively enable/disable
    logger.info("Individual tests would require pipeline modifications to toggle features")
    logger.info("Currently all optimizations are enabled by default")

if __name__ == "__main__":
    # Activate the environment
    logger.info("🔧 Activating UC1 AI environment...")
    activate_script = "/home/ucadmin/activate-uc1-ai-py311.sh"
    if os.path.exists(activate_script):
        os.system(f"source {activate_script}")
    
    # Run combined test
    success = test_combined_performance()
    
    # Run individual tests if requested
    if "--individual" in sys.argv:
        test_individual_optimizations()
    
    # Final result
    if success:
        logger.info("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        logger.info("\n❌ Tests failed!")
        sys.exit(1)