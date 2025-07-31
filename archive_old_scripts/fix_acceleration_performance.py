#!/usr/bin/env python3
"""
Fix Gemma 3n E4B Acceleration Performance
Identify and fix bottlenecks in the acceleration pipeline
"""

import time
import torch
import logging
import psutil
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model_bottlenecks():
    """Analyze what's causing the slow performance"""
    logger.info("🔍 ANALYZING MODEL BOTTLENECKS")
    logger.info("=" * 50)
    
    # Load model with minimal settings
    logger.info("Loading model with minimal settings...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained("./models/gemma-3n-e4b-it", trust_remote_code=True)
    
    # Try different loading strategies
    loading_strategies = [
        {
            "name": "Standard float32",
            "kwargs": {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
        },
        {
            "name": "Optimized bfloat16",
            "kwargs": {
                "torch_dtype": torch.bfloat16,
                "device_map": "cpu", 
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager"
            }
        },
        {
            "name": "Auto device map",
            "kwargs": {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
        }
    ]
    
    best_config = None
    best_tps = 0
    
    for strategy in loading_strategies:
        logger.info(f"\n🔬 Testing strategy: {strategy['name']}")
        
        try:
            # Load model
            load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                "./models/gemma-3n-e4b-it",
                **strategy["kwargs"]
            )
            load_time = time.time() - load_start
            
            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            model_size_gb = param_count * 2 / (1024**3)  # Assuming 2 bytes per param
            
            logger.info(f"  Load time: {load_time:.1f}s")
            logger.info(f"  Parameters: {param_count/1e9:.1f}B")
            logger.info(f"  Model size: {model_size_gb:.1f}GB")
            logger.info(f"  Model dtype: {next(model.parameters()).dtype}")
            
            # Quick inference test
            test_prompt = "Hello, test response"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            inference_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,  # Shorter test
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            inference_time = time.time() - inference_start
            
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            tps = tokens_generated / inference_time if inference_time > 0 else 0
            
            logger.info(f"  Inference time: {inference_time:.2f}s")
            logger.info(f"  TPS: {tps:.1f}")
            
            if tps > best_tps:
                best_tps = tps
                best_config = strategy
                
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            continue
    
    logger.info(f"\n📊 BEST CONFIGURATION: {best_config['name'] if best_config else 'None'}")
    logger.info(f"   Best TPS: {best_tps:.1f}")
    
    return best_config, best_tps

def test_hardware_utilization():
    """Test if we're actually using hardware acceleration"""
    logger.info("\n🔍 TESTING HARDWARE UTILIZATION")
    logger.info("=" * 50)
    
    # Check NPU status
    try:
        import subprocess
        result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ NPU Phoenix detected")
            npu_info = result.stdout.split('\n')[0]
            logger.info(f"   NPU info: {npu_info}")
        else:
            logger.warning("⚠️  NPU not responding")
    except:
        logger.warning("⚠️  NPU tools not available")
    
    # Check Vulkan status
    try:
        result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'phoenix' in result.stdout.lower():
            logger.info("✅ Vulkan iGPU detected")
            logger.info(f"   Vulkan device: AMD Radeon Graphics (RADV PHOENIX)")
        else:
            logger.warning("⚠️  Vulkan iGPU not detected")
    except:
        logger.warning("⚠️  Vulkan tools not available")
    
    # Check CPU utilization during inference
    logger.info(f"✅ CPU: {psutil.cpu_count()} cores available")
    logger.info(f"   Current CPU usage: {psutil.cpu_percent()}%")
    logger.info(f"   Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")

def create_optimized_loader():
    """Create an optimized model loader"""
    logger.info("\n🚀 CREATING OPTIMIZED LOADER")
    logger.info("=" * 50)
    
    class OptimizedGemma3nLoader:
        def __init__(self, model_path: str = "./models/gemma-3n-e4b-it"):
            self.model_path = model_path
            self.tokenizer = None
            self.model = None
            
        def load_optimized(self):
            """Load model with all optimizations"""
            logger.info("Loading with maximum optimizations...")
            
            # Set CPU optimizations
            torch.set_num_threads(psutil.cpu_count())
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for speed
                device_map="cpu",  # Keep on CPU for now
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            
            logger.info("✅ Optimized model loaded")
            logger.info(f"   Model dtype: {next(self.model.parameters()).dtype}")
            logger.info(f"   CPU threads: {torch.get_num_threads()}")
            
        def fast_generate(self, prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
            """Generate with optimizations"""
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = len(inputs["input_ids"][0])
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV caching
                    top_p=0.9,
                    top_k=50
                )
            
            generation_time = time.time() - start_time
            generated_tokens = len(outputs[0]) - input_length
            tps = generated_tokens / generation_time if generation_time > 0 else 0
            
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return {
                "response": response,
                "tokens_generated": generated_tokens,
                "generation_time": generation_time,
                "tokens_per_second": tps
            }
    
    # Test the optimized loader
    loader = OptimizedGemma3nLoader()
    
    load_start = time.time()
    loader.load_optimized()
    load_time = time.time() - load_start
    
    logger.info(f"✅ Optimized loader created in {load_time:.1f}s")
    
    # Test generation
    test_prompt = "Hello, I'm Aaron. Please tell me about yourself."
    result = loader.fast_generate(test_prompt, max_tokens=30)
    
    logger.info(f"📊 OPTIMIZED PERFORMANCE:")
    logger.info(f"   Generated: {result['tokens_generated']} tokens")
    logger.info(f"   Time: {result['generation_time']:.2f}s")
    logger.info(f"   TPS: {result['tokens_per_second']:.1f}")
    logger.info(f"   Response: {result['response'][:50]}...")
    
    return loader, result['tokens_per_second']

def main():
    """Main analysis and optimization"""
    logger.info("🦄 Gemma 3n E4B Performance Fix")
    logger.info("=" * 60)
    
    # Step 1: Analyze bottlenecks
    try:
        best_config, best_tps = analyze_model_bottlenecks()
        logger.info(f"\n🎯 Bottleneck analysis complete - Best TPS: {best_tps:.1f}")
    except Exception as e:
        logger.error(f"❌ Bottleneck analysis failed: {e}")
        best_tps = 0
    
    # Step 2: Test hardware utilization
    test_hardware_utilization()
    
    # Step 3: Create optimized loader
    try:
        loader, optimized_tps = create_optimized_loader()
        logger.info(f"\n🚀 Optimized loader created - TPS: {optimized_tps:.1f}")
    except Exception as e:
        logger.error(f"❌ Optimized loader failed: {e}")
        optimized_tps = 0
    
    # Step 4: Performance summary
    logger.info("\n📊 PERFORMANCE SUMMARY:")
    logger.info("=" * 40)
    logger.info(f"   Best baseline TPS: {best_tps:.1f}")
    logger.info(f"   Optimized TPS: {optimized_tps:.1f}")
    
    if optimized_tps > best_tps:
        improvement = (optimized_tps / best_tps - 1) * 100 if best_tps > 0 else 0
        logger.info(f"   Improvement: {improvement:.1f}%")
    
    # Step 5: Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    if optimized_tps < 5:
        logger.info("   ❌ Performance is still too low")
        logger.info("   - Model may be too large for this hardware")
        logger.info("   - Consider quantization (INT8/INT4)")
        logger.info("   - Check if using vision components unnecessarily")
        logger.info("   - Try smaller context windows")
    elif optimized_tps < 20:
        logger.info("   ⚠️  Performance is below target")
        logger.info("   - Implement proper hardware acceleration")
        logger.info("   - Add layer-level NPU/Vulkan processing")
        logger.info("   - Consider model pruning")
    else:
        logger.info("   ✅ Performance is acceptable - implement hardware acceleration")
        logger.info("   - Add NPU attention acceleration")
        logger.info("   - Add Vulkan FFN acceleration")
        logger.info("   - Target: 50+ TPS with hardware acceleration")

if __name__ == "__main__":
    main()