#!/usr/bin/env python3
"""
Test Real NPU+iGPU Tokens Per Second
Direct test with no CPU fallback - hardware or failure
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any
import subprocess
import os

# Import our pipeline
from pure_hardware_pipeline import PureHardwarePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrictTokenPerSecondTest:
    """Test real NPU+iGPU performance with no CPU fallback"""
    
    def __init__(self):
        self.pipeline = None
        self.hardware_verified = False
        
    def verify_hardware(self) -> bool:
        """Verify NPU and iGPU are available and working"""
        logger.info("🔍 VERIFYING REAL HARDWARE ACCELERATION...")
        
        # Check NPU Phoenix
        try:
            result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True)
            if result.returncode != 0 or 'Phoenix' not in result.stdout:
                logger.error("❌ NPU Phoenix not detected")
                return False
            logger.info("✅ NPU Phoenix detected")
        except Exception as e:
            logger.error(f"❌ NPU check failed: {e}")
            return False
            
        # Check AMD Radeon 780M iGPU
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            if result.returncode != 0 or 'amd radeon graphics' not in result.stdout.lower():
                logger.error("❌ AMD Radeon 780M iGPU not detected")
                return False
            logger.info("✅ AMD Radeon 780M iGPU detected")
        except Exception as e:
            logger.error(f"❌ iGPU check failed: {e}")
            return False
            
        self.hardware_verified = True
        return True
        
    def monitor_memory_allocation(self):
        """Monitor actual memory allocation during loading"""
        logger.info("📊 MONITORING MEMORY ALLOCATION...")
        
        # Check GPU memory usage
        try:
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\\n'):
                if 'vram' in line.lower() or 'gtt' in line.lower():
                    logger.info(f"   🎮 GPU Memory: {line.strip()}")
        except Exception as e:
            logger.debug(f"GPU memory monitoring failed: {e}")
            
        # Check system memory
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            logger.info(f"   💾 System Memory: {result.stdout.split()[12]} used")
        except Exception:
            pass
    
    def test_real_tokens_per_second(self) -> float:
        """Test REAL tokens per second with NPU+iGPU only"""
        if not self.hardware_verified:
            raise RuntimeError("Hardware not verified - cannot proceed")
            
        logger.info("🚀 TESTING REAL NPU+iGPU TOKENS PER SECOND")
        logger.info("🎯 NO CPU FALLBACK - Hardware acceleration only!")
        
        # Initialize pipeline with HMA memory distribution
        self.pipeline = PureHardwarePipeline()
        
        # Monitor memory during initialization
        logger.info("📊 BEFORE MODEL LOADING:")
        self.monitor_memory_allocation()
        
        # Load model with proper HMA distribution
        model_path = "./quantized_models/gemma-3-27b-it-layer-by-layer"
        if not self.pipeline.initialize(model_path):
            raise RuntimeError("Pipeline initialization failed")
            
        # Monitor memory after loading
        logger.info("📊 AFTER MODEL LOADING:")
        self.monitor_memory_allocation()
        
        # Verify HMA distribution is working
        logger.info("🔍 VERIFYING HMA MEMORY DISTRIBUTION:")
        logger.info(f"   🧠 NPU SRAM: {self.pipeline.current_memory['npu_sram_mb']:.1f}MB")
        logger.info(f"   ⚡ iGPU VRAM: {self.pipeline.current_memory['vram_mb']:.1f}MB") 
        logger.info(f"   💾 iGPU GTT: {self.pipeline.current_memory['gtt_mb']:.1f}MB")
        logger.info(f"   🔧 System RAM: {self.pipeline.current_memory['ram_mb']:.1f}MB")
        
        # ENFORCE hardware requirements
        if self.pipeline.current_memory['vram_mb'] < 1000:  # Less than 1GB in VRAM
            raise RuntimeError("❌ VRAM allocation failed - model not properly loaded to GPU")
        if self.pipeline.current_memory['gtt_mb'] < 5000:   # Less than 5GB in GTT  
            raise RuntimeError("❌ GTT allocation failed - model not properly distributed")
            
        logger.info("✅ HMA memory distribution verified!")
        
        # Test token generation with strict timing
        test_prompts = [
            "Hello, I am",
            "The future of AI is", 
            "Quantum computing will"
        ]
        
        total_tokens = 0
        total_time = 0
        
        for prompt in test_prompts:
            logger.info(f"🧪 Testing prompt: '{prompt}'")
            
            # Simple tokenization (character-based for this test)
            input_tokens = [ord(c) % 32000 for c in prompt]  # Simple char->token mapping
            max_new_tokens = 5  # Generate 5 tokens per prompt
            
            start_time = time.time()
            
            try:
                # Test with strict NPU+iGPU enforcement
                generated_tokens = self.pipeline.generate_tokens_streaming(
                    input_tokens, 
                    max_new_tokens=max_new_tokens,
                    enforce_hardware_only=True  # NO CPU FALLBACK
                )
                
                end_time = time.time()
                prompt_time = end_time - start_time
                
                total_tokens += max_new_tokens
                total_time += prompt_time
                
                tokens_per_second = max_new_tokens / prompt_time
                logger.info(f"   ⚡ Generated {max_new_tokens} tokens in {prompt_time:.3f}s")
                logger.info(f"   🚀 Speed: {tokens_per_second:.2f} tokens/sec")
                
            except Exception as e:
                logger.error(f"❌ Hardware-only generation failed: {e}")
                raise RuntimeError("NPU+iGPU generation failed - no fallback allowed")
        
        # Calculate final tokens per second
        if total_time > 0:
            final_tps = total_tokens / total_time
            logger.info("🎉 REAL NPU+iGPU PERFORMANCE RESULTS:")
            logger.info(f"   📊 Total tokens: {total_tokens}")
            logger.info(f"   ⏱️ Total time: {total_time:.3f}s")
            logger.info(f"   🚀 Final TPS: {final_tps:.2f} tokens/second")
            logger.info(f"   🎯 Hardware: NPU Phoenix (16 TOPS) + AMD Radeon 780M")
            return final_tps
        else:
            raise RuntimeError("No successful token generation")

def main():
    """Run the strict tokens per second test"""
    logger.info("🦄 STRICT NPU+iGPU TOKENS PER SECOND TEST")
    logger.info("=" * 60)
    
    tester = StrictTokenPerSecondTest()
    
    # Verify hardware first
    if not tester.verify_hardware():
        logger.error("❌ Hardware verification failed - cannot proceed")
        return False
    
    try:
        # Run the test
        tps = tester.test_real_tokens_per_second()
        
        logger.info("🏆 TEST SUCCESSFUL!")
        logger.info(f"   Real NPU+iGPU TPS: {tps:.2f}")
        logger.info("   No CPU fallback used")
        logger.info("   Real hardware acceleration verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if tester.pipeline:
            tester.pipeline.cleanup()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)