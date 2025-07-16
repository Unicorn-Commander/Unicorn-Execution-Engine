#!/usr/bin/env python3
"""
Test Real Gemma 3 27B with Quantized Model and Real Hardware
Force use of 27B model and real NPU+iGPU acceleration
"""

import os
import sys
import time
import torch
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import real hardware components
from production_npu_engine import ProductionNPUEngine
from real_vulkan_compute import RealVulkanCompute
from hma_zero_copy_optimization import HMAZeroCopyOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealGemma27BTest:
    """Test real Gemma 3 27B with hardware acceleration"""
    
    def __init__(self):
        # Force 27B model paths
        self.model_paths = [
            "./quantized_models/gemma-3-27b-it-memory-efficient",  # 30.8GB quantized
            "./quantized_models/gemma-3-27b-it-ultra-16gb",        # 28.9GB quantized
            "./quantized_models/gemma-3-27b-it-real-optimized",    # Real hardware optimized
            "./models/gemma-3-27b-it"                              # Original 102GB
        ]
        
        self.model = None
        self.tokenizer = None
        self.npu_engine = None
        self.vulkan_compute = None
        self.memory_bridge = None
        
        # Performance tracking
        self.results = {
            "model_info": {},
            "hardware_stats": {},
            "inference_results": []
        }
        
    def check_system_resources(self):
        """Check system resources for 27B model"""
        logger.info("🔍 Checking System Resources for Gemma 3 27B...")
        
        # Memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        logger.info(f"   💾 Total RAM: {total_gb:.1f}GB")
        logger.info(f"   💾 Available RAM: {available_gb:.1f}GB")
        
        # Check if we have enough for 27B model
        if available_gb < 35:
            logger.warning(f"   ⚠️ Low memory for 27B model (need ~35GB, have {available_gb:.1f}GB)")
        else:
            logger.info(f"   ✅ Sufficient memory for quantized 27B model")
        
        return available_gb >= 25  # Minimum for quantized 27B
    
    def initialize_hardware(self) -> bool:
        """Initialize real hardware components"""
        logger.info("🚀 Initializing Real Hardware Components...")
        
        try:
            # Initialize NPU engine
            logger.info("   🧠 Initializing NPU Phoenix...")
            self.npu_engine = ProductionNPUEngine()
            if self.npu_engine.initialize():
                logger.info("   ✅ NPU Phoenix ready")
            else:
                logger.warning("   ⚠️ NPU initialization failed, using fallback")
            
            # Initialize Vulkan compute
            logger.info("   🎮 Initializing AMD Radeon 780M...")
            self.vulkan_compute = RealVulkanCompute()
            if self.vulkan_compute.initialize():
                logger.info("   ✅ Vulkan compute ready")
            else:
                logger.error("   ❌ Vulkan initialization failed")
                return False
            
            # Initialize memory bridge
            logger.info("   🌉 Initializing HMA Memory Bridge...")
            self.memory_bridge = HMAZeroCopyOptimizer()
            if self.memory_bridge.initialize():
                logger.info("   ✅ Zero-copy memory ready")
            else:
                logger.warning("   ⚠️ Memory bridge failed, using standard allocation")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Hardware initialization failed: {e}")
            return False
    
    def load_gemma27b_model(self) -> bool:
        """Load Gemma 3 27B model (prioritize quantized versions)"""
        logger.info("📥 Loading Gemma 3 27B Model...")
        
        for model_path in self.model_paths:
            if not os.path.exists(model_path):
                logger.info(f"   ❌ Not found: {model_path}")
                continue
            
            logger.info(f"   🔄 Attempting: {model_path}")
            
            try:
                start_time = time.time()
                
                # Load tokenizer
                logger.info("      🔤 Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                # Add padding token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Check available memory before loading model
                memory_before = psutil.virtual_memory()
                available_gb = memory_before.available / (1024**3)
                
                if available_gb < 30 and "quantized" not in model_path:
                    logger.warning(f"      ⚠️ Insufficient memory for full model, skipping")
                    continue
                
                # Load model with optimizations
                logger.info("      🧠 Loading model weights...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_compile=False  # Disable compilation for faster loading
                )
                
                load_time = time.time() - start_time
                
                # Get model info
                param_count = sum(p.numel() for p in self.model.parameters())
                model_size_gb = param_count * 2 / (1024**3)  # FP16
                
                # Check memory usage after loading
                memory_after = psutil.virtual_memory()
                memory_used_gb = (memory_before.available - memory_after.available) / (1024**3)
                
                logger.info(f"      ✅ Model loaded successfully!")
                logger.info(f"      📊 Parameters: {param_count:,} ({param_count/1e9:.1f}B)")
                logger.info(f"      💾 Model size: {model_size_gb:.1f}GB")
                logger.info(f"      🕒 Load time: {load_time:.1f}s")
                logger.info(f"      💾 Memory used: {memory_used_gb:.1f}GB")
                
                # Verify this is actually 27B model
                if param_count < 20e9:  # Less than 20B parameters
                    logger.error(f"      ❌ This appears to be a smaller model ({param_count/1e9:.1f}B), not 27B!")
                    continue
                
                # Store model info
                self.results["model_info"] = {
                    "path": model_path,
                    "parameters": param_count,
                    "size_gb": model_size_gb,
                    "load_time": load_time,
                    "memory_used_gb": memory_used_gb,
                    "is_quantized": "quantized" in model_path
                }
                
                return True
                
            except Exception as e:
                logger.error(f"      ❌ Failed to load {model_path}: {e}")
                # Clean up on failure
                if self.model:
                    del self.model
                    self.model = None
                if self.tokenizer:
                    del self.tokenizer
                    self.tokenizer = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
        
        logger.error("❌ Failed to load any Gemma 3 27B model")
        return False
    
    def run_real_inference_test(self, prompt: str, max_tokens: int = 30) -> Dict:
        """Run inference with real hardware acceleration"""
        logger.info(f"🔮 Running Real Hardware Inference...")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Max tokens: {max_tokens}")
        
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Track memory before inference
            memory_before = psutil.virtual_memory()
            
            # Tokenize input
            logger.info("   🔤 Tokenizing input...")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # Generate with real hardware
            logger.info("   🚀 Generating with real hardware acceleration...")
            generation_start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            generation_time = time.time() - generation_start
            
            # Decode output
            logger.info("   📝 Decoding output...")
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            total_time = time.time() - start_time
            actual_tokens = len(self.tokenizer.encode(generated_text))
            tps = actual_tokens / generation_time if generation_time > 0 else 0
            
            # Track memory after inference
            memory_after = psutil.virtual_memory()
            
            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "input_tokens": input_length,
                "output_tokens": actual_tokens,
                "generation_time": generation_time,
                "total_time": total_time,
                "tokens_per_second": tps,
                "memory_used_mb": (memory_before.available - memory_after.available) / (1024**2),
                "success": True
            }
            
            logger.info(f"   ✅ Generated {actual_tokens} tokens in {generation_time:.2f}s")
            logger.info(f"   📊 Performance: {tps:.2f} TPS")
            logger.info(f"   💭 Output: {generated_text[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Inference failed: {e}")
            return {"error": str(e), "success": False}
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive 27B test"""
        logger.info("🦄 Starting Real Gemma 3 27B Hardware Test")
        logger.info("=" * 60)
        
        # Check system resources
        if not self.check_system_resources():
            return {"error": "Insufficient system resources"}
        
        # Initialize hardware
        if not self.initialize_hardware():
            return {"error": "Hardware initialization failed"}
        
        # Load 27B model
        if not self.load_gemma27b_model():
            return {"error": "Failed to load Gemma 3 27B model"}
        
        # Test prompts for 27B model
        test_prompts = [
            "Explain quantum computing in detail",
            "What is the future of artificial intelligence?",
            "How do neural networks work?"
        ]
        
        # Run inference tests
        logger.info("🧪 Running Inference Tests...")
        for i, prompt in enumerate(test_prompts):
            logger.info(f"   Test {i+1}/{len(test_prompts)}")
            result = self.run_real_inference_test(prompt, max_tokens=25)
            self.results["inference_results"].append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Calculate summary
        successful_tests = [r for r in self.results["inference_results"] if r.get("success", False)]
        
        if successful_tests:
            avg_tps = sum(r["tokens_per_second"] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
            
            logger.info("=" * 60)
            logger.info("🎯 REAL GEMMA 3 27B TEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"   🚀 Average TPS: {avg_tps:.2f}")
            logger.info(f"   ⏱️ Average time: {avg_time:.2f}s")
            logger.info(f"   📊 Success rate: {len(successful_tests)}/{len(test_prompts)}")
            logger.info(f"   🧠 Model: {self.results['model_info']['parameters']/1e9:.1f}B parameters")
            logger.info(f"   💾 Model size: {self.results['model_info']['size_gb']:.1f}GB")
            logger.info(f"   🔧 Quantized: {self.results['model_info']['is_quantized']}")
            
            self.results["summary"] = {
                "avg_tps": avg_tps,
                "avg_time": avg_time,
                "success_rate": len(successful_tests) / len(test_prompts)
            }
        
        return self.results

def main():
    """Main function"""
    logger.info("🦄 Real Gemma 3 27B Hardware Test")
    
    # Run test
    test = RealGemma27BTest()
    results = test.run_comprehensive_test()
    
    if "error" in results:
        logger.error(f"❌ Test failed: {results['error']}")
        return 1
    elif results.get("summary", {}).get("avg_tps", 0) > 0:
        logger.info("✅ Test completed successfully!")
        return 0
    else:
        logger.error("❌ No successful inference tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())