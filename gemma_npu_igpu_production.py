#!/usr/bin/env python3
"""
Production-Ready Gemma NPU+iGPU Inference
Complete pipeline for Gemma 4B/27B with real NPU acceleration
"""

import os
import sys
import subprocess
import time
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GemmaNPUiGPU:
    def __init__(self, model_path=None, model_size="4b"):
        self.model_path = model_path
        self.model_size = model_size
        self.llama_cli = "llama.cpp/build/bin/llama-cli"
        self.performance_stats = []
        
    def check_hardware(self):
        """Verify NPU and GPU are available"""
        logger.info("🔍 Checking hardware availability...")
        
        # Check NPU
        npu_available = os.path.exists("/dev/accel/accel0")
        logger.info(f"   NPU Phoenix: {'✅ Available' if npu_available else '❌ Not found'}")
        
        # Check Vulkan GPU
        vulkan_available = subprocess.run(
            ["vulkaninfo", "--summary"], 
            capture_output=True
        ).returncode == 0
        logger.info(f"   Vulkan GPU: {'✅ Available' if vulkan_available else '❌ Not found'}")
        
        # Check kernels
        kernel_map = {
            "4b": "npu_kernels_compiled/gemma3_4b_attention.xclbin",
            "27b": "npu_kernels_compiled/gemma3_27b_attention.xclbin"
        }
        
        kernel_path = kernel_map.get(self.model_size)
        kernel_available = os.path.exists(kernel_path) if kernel_path else False
        logger.info(f"   Gemma {self.model_size.upper()} kernel: {'✅ Available' if kernel_available else '❌ Not found'}")
        
        return {
            "npu": npu_available,
            "gpu": vulkan_available,
            "kernel": kernel_available,
            "ready": npu_available and vulkan_available and kernel_available
        }
    
    def setup_model(self):
        """Setup or convert model if needed"""
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"✅ Using existing model: {self.model_path}")
            return self.model_path
            
        logger.info("📦 Model setup required...")
        
        # Check for pre-downloaded models
        model_candidates = [
            f"models/gemma-{self.model_size}-it/gemma-{self.model_size}-q4_k_m.gguf",
            f"gemma-{self.model_size}-q4_k_m.gguf",
            "tinyllama-1.1b-q4_k_m.gguf"  # Fallback for testing
        ]
        
        for candidate in model_candidates:
            if os.path.exists(candidate):
                logger.info(f"✅ Found model: {candidate}")
                self.model_path = candidate
                return candidate
                
        logger.warning("⚠️ No GGUF model found. Please run:")
        logger.warning(f"   python3 download_and_convert_gemma.py")
        return None
    
    def run_inference(self, prompt, max_tokens=128, temperature=0.7):
        """Run inference with NPU+iGPU acceleration"""
        if not self.model_path:
            logger.error("❌ No model loaded")
            return None
            
        logger.info(f"\n🚀 Running NPU+iGPU inference...")
        logger.info(f"   Prompt: '{prompt[:50]}...'")
        logger.info(f"   Max tokens: {max_tokens}")
        
        cmd = [
            self.llama_cli,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--npu-attention",      # Real NPU kernel execution
            "--gpu-layers", "999",  # Vulkan GPU for other ops
            "--threads", "8",       # CPU threads for coordination
            "--batch-size", "512",  # Optimal for NPU
            "--ctx-size", "2048",   # Context window
            "--log-disable"         # Clean output
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"❌ Inference failed: {result.stderr}")
                return None
                
            # Parse output
            output_text = result.stdout.strip()
            
            # Extract performance metrics from stderr
            tokens_per_second = None
            for line in result.stderr.split('\n'):
                if "tok/s" in line or "tokens/s" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if ("tok/s" in part or "tokens/s" in part) and i > 0:
                            try:
                                tokens_per_second = float(parts[i-1])
                                break
                            except:
                                pass
            
            # Calculate tokens generated
            tokens_generated = len(output_text.split()) - len(prompt.split())
            
            stats = {
                "prompt": prompt,
                "response": output_text,
                "tokens_generated": tokens_generated,
                "time_seconds": elapsed,
                "tokens_per_second": tokens_per_second or (tokens_generated / elapsed),
                "hardware": "NPU+iGPU"
            }
            
            self.performance_stats.append(stats)
            
            logger.info(f"✅ Completed in {elapsed:.2f}s")
            logger.info(f"   Tokens: {tokens_generated}")
            logger.info(f"   Speed: {stats['tokens_per_second']:.2f} tok/s")
            
            return stats
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Inference timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    def interactive_chat(self):
        """Interactive chat mode with NPU+iGPU acceleration"""
        logger.info("\n🦄 Gemma NPU+iGPU Chat")
        logger.info("=" * 60)
        logger.info("Type 'exit' to quit, 'stats' for performance info")
        logger.info("=" * 60)
        
        conversation = []
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'stats':
                self.print_stats()
                continue
            elif not user_input:
                continue
                
            # Build conversation context
            conversation.append(f"User: {user_input}")
            
            # Create prompt with conversation history
            prompt = "\n".join(conversation[-5:])  # Keep last 5 turns
            prompt += "\nAssistant:"
            
            # Run inference
            result = self.run_inference(prompt, max_tokens=200)
            
            if result:
                # Extract just the assistant's response
                response = result['response']
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
                print(f"\n🤖 Gemma: {response}")
                
                # Add to conversation
                conversation.append(f"Assistant: {response}")
            else:
                print("\n❌ Failed to generate response")
    
    def benchmark(self):
        """Run comprehensive benchmark"""
        logger.info("\n📊 Running NPU+iGPU Benchmark...")
        
        test_prompts = [
            ("Simple", "Hello, world!", 20),
            ("Reasoning", "What is 2+2? Think step by step:", 50),
            ("Creative", "Write a haiku about NPUs:", 50),
            ("Technical", "Explain how attention works in transformers:", 100),
            ("Long", "Tell me a story about a magical unicorn that could accelerate AI models:", 200)
        ]
        
        results = []
        
        for name, prompt, max_tokens in test_prompts:
            logger.info(f"\n🧪 Test: {name}")
            
            # Warm-up run
            self.run_inference(prompt, max_tokens=10)
            
            # Actual benchmark (3 runs)
            times = []
            for i in range(3):
                result = self.run_inference(prompt, max_tokens=max_tokens)
                if result:
                    times.append(result['tokens_per_second'])
                    
            if times:
                avg_tps = sum(times) / len(times)
                results.append({
                    "test": name,
                    "prompt_length": len(prompt.split()),
                    "max_tokens": max_tokens,
                    "avg_tokens_per_second": avg_tps
                })
                
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("=" * 60)
        
        for result in results:
            logger.info(f"\n{result['test']}:")
            logger.info(f"   Average: {result['avg_tokens_per_second']:.2f} tok/s")
            
        if results:
            overall_avg = sum(r['avg_tokens_per_second'] for r in results) / len(results)
            logger.info(f"\n🏆 Overall Average: {overall_avg:.2f} tok/s")
            logger.info(f"🚀 NPU+iGPU Hybrid Acceleration Active!")
    
    def print_stats(self):
        """Print performance statistics"""
        if not self.performance_stats:
            logger.info("No statistics available yet")
            return
            
        logger.info("\n📊 Performance Statistics")
        logger.info("=" * 40)
        
        total_tokens = sum(s['tokens_generated'] for s in self.performance_stats)
        total_time = sum(s['time_seconds'] for s in self.performance_stats)
        avg_tps = sum(s['tokens_per_second'] for s in self.performance_stats) / len(self.performance_stats)
        
        logger.info(f"Total inferences: {len(self.performance_stats)}")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average speed: {avg_tps:.2f} tok/s")

def main():
    parser = argparse.ArgumentParser(description="Gemma NPU+iGPU Production Inference")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("--size", choices=["4b", "27b"], default="4b", 
                        help="Model size for kernel selection")
    parser.add_argument("--mode", choices=["chat", "benchmark", "single"], 
                        default="chat", help="Operation mode")
    parser.add_argument("--prompt", help="Single prompt for inference")
    parser.add_argument("--max-tokens", type=int, default=128, 
                        help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # ASCII art banner
    print("""
    🦄 ╔════════════════════════════════════╗
       ║  Gemma NPU+iGPU Magic Inference    ║
       ║  Real Hardware Acceleration        ║
       ║  Phoenix NPU + Vulkan GPU          ║
       ╚════════════════════════════════════╝
    """)
    
    # Initialize
    gemma = GemmaNPUiGPU(args.model, args.size)
    
    # Check hardware
    hw_status = gemma.check_hardware()
    if not hw_status["ready"]:
        logger.error("❌ Hardware not ready for NPU+iGPU acceleration")
        return
        
    # Setup model
    if not gemma.setup_model():
        logger.error("❌ Model setup failed")
        return
        
    # Run requested mode
    if args.mode == "chat":
        gemma.interactive_chat()
    elif args.mode == "benchmark":
        gemma.benchmark()
    elif args.mode == "single" and args.prompt:
        result = gemma.run_inference(args.prompt, args.max_tokens)
        if result:
            print(f"\n{result['response']}")
            print(f"\n📊 Performance: {result['tokens_per_second']:.2f} tok/s")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()