#!/usr/bin/env python3
"""
Test REAL Gemma3n E4B TPS with actual model loading and generation
"""

import os
import time
import logging
from gemma3n_e4b_unicorn_loader import Gemma3nE4BUnicornLoader, ModelConfig, HardwareConfig, InferenceConfig, InferenceMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_tps():
    """Test real tokens per second with Gemma3n E4B"""
    
    logger.info("🚀 REAL GEMMA3N E4B TPS TEST")
    logger.info("=" * 60)
    
    # Configure for real hardware
    model_config = ModelConfig(
        model_path="./models/gemma-3n-e4b-it",
        elastic_enabled=True,
        quantization_enabled=True,
        mix_n_match_enabled=True
    )
    
    hardware_config = HardwareConfig(
        npu_enabled=True,
        igpu_enabled=True,
        hma_enabled=True,
        turbo_mode=True
    )
    
    # Initialize loader
    logger.info("⚡ Initializing Gemma3n E4B loader...")
    loader = Gemma3nE4BUnicornLoader(model_config, hardware_config)
    
    # Load model
    logger.info("📦 Loading model to NPU+iGPU...")
    start_load = time.time()
    if not loader.load_model():
        logger.error("❌ Failed to load model")
        return 0
    load_time = time.time() - start_load
    logger.info(f"✅ Model loaded in {load_time:.2f}s")
    
    # Test prompts for real generation
    test_cases = [
        ("What is 2+2?", 20),
        ("Write a haiku about NPUs", 30),
        ("Explain AI in one sentence", 25),
        ("List three programming languages", 20),
        ("What color is the sky?", 15)
    ]
    
    # Warm-up
    logger.info("\n🔥 Warming up hardware...")
    try:
        warm_config = InferenceConfig(
            mode=InferenceMode.PERFORMANCE,
            max_tokens=10,
            temperature=0.7
        )
        _ = loader.generate("Hello", warm_config)
        logger.info("✅ Warm-up complete")
    except Exception as e:
        logger.warning(f"⚠️ Warm-up failed: {e}")
    
    # Run real benchmarks
    logger.info("\n📊 Running real generation benchmarks...")
    results = []
    
    for prompt, max_tokens in test_cases:
        logger.info(f"\n🔄 Generating: '{prompt}' (max {max_tokens} tokens)")
        
        try:
            # Configure for maximum performance
            inference_config = InferenceConfig(
                mode=InferenceMode.PERFORMANCE,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # Time the actual generation
            start_time = time.time()
            result = loader.generate(prompt, inference_config)
            elapsed = time.time() - start_time
            
            if isinstance(result, dict) and 'generated_text' in result:
                tokens_generated = result.get('tokens_generated', 0)
                tps = result.get('tokens_per_second', 0)
                
                # Calculate real TPS from elapsed time
                real_tps = tokens_generated / elapsed if elapsed > 0 else 0
                
                logger.info(f"  ✅ Generated: {result['generated_text'][:60]}...")
                logger.info(f"  📊 Tokens: {tokens_generated}")
                logger.info(f"  ⏱️ Time: {elapsed:.2f}s")
                logger.info(f"  🚀 Reported TPS: {tps:.2f}")
                logger.info(f"  🎯 Measured TPS: {real_tps:.2f}")
                
                results.append({
                    'prompt': prompt,
                    'tokens': tokens_generated,
                    'time': elapsed,
                    'reported_tps': tps,
                    'measured_tps': real_tps
                })
            else:
                logger.error(f"  ❌ Generation failed: {result}")
                
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate real performance
    if results:
        # Use measured TPS (actual elapsed time)
        avg_measured_tps = sum(r['measured_tps'] for r in results) / len(results)
        avg_reported_tps = sum(r['reported_tps'] for r in results) / len(results)
        total_tokens = sum(r['tokens'] for r in results)
        total_time = sum(r['time'] for r in results)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 REAL PERFORMANCE RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Tests completed: {len(results)}/{len(test_cases)}")
        logger.info(f"✅ Total tokens generated: {total_tokens}")
        logger.info(f"✅ Total generation time: {total_time:.2f}s")
        logger.info(f"🚀 Average MEASURED TPS: {avg_measured_tps:.2f} tokens/second")
        logger.info(f"📈 Average reported TPS: {avg_reported_tps:.2f} tokens/second")
        logger.info("=" * 60)
        
        # Real performance analysis
        real_tps = avg_measured_tps
        if real_tps >= 150:
            logger.info("🎉 EXCELLENT! Real performance exceeds 150 TPS!")
        elif real_tps >= 100:
            logger.info("✅ Good real performance! Over 100 TPS")
        elif real_tps >= 50:
            logger.info("📈 Decent real performance")
        else:
            logger.info("⚠️ Real performance below expectations")
        
        # Show hardware utilization
        try:
            status = loader.get_status()
            logger.info(f"\n📊 Hardware Status:")
            logger.info(f"  NPU: {status.get('npu', 'N/A')}")
            logger.info(f"  iGPU: {status.get('igpu', 'N/A')}")
            logger.info(f"  Memory: {status.get('memory', 'N/A')}")
        except:
            pass
        
        return avg_measured_tps
    
    return 0

if __name__ == "__main__":
    try:
        real_tps = test_real_tps()
        logger.info(f"\n✅ Test completed! REAL TPS: {real_tps:.2f}")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()