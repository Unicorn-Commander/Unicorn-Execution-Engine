#!/usr/bin/env python3
"""
Complete Hardware Inference System
- Real tokenization
- GPU-only execution (NPU when available)
- Persistent model server
- No CPU in hot path
"""

import time
import logging
import numpy as np
from gemma_tokenizer import GemmaTokenizer
from pure_hardware_pipeline_gpu_fixed import PureHardwarePipelineGPUFixed
import subprocess
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareInferenceServer:
    """Persistent hardware inference server"""
    
    def __init__(self):
        self.pipeline = None
        self.tokenizer = None
        self.model_loaded = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        
        logger.info("🚀 Initializing Hardware Inference Server...")
        
        # 1. Initialize tokenizer
        self.tokenizer = GemmaTokenizer()
        logger.info("✅ Tokenizer ready")
        
        # 2. Initialize pipeline
        self.pipeline = PureHardwarePipelineGPUFixed()
        
        # 3. Load model ONCE
        model_path = "/home/ucadmin/Development/github_repos/Unicorn-Execution-Engine/quantized_models/gemma-3-27b-it-layer-by-layer"
        
        logger.info("📦 Loading model to GPU (this will take a minute)...")
        start = time.time()
        
        if self.pipeline.initialize(model_path):
            elapsed = time.time() - start
            logger.info(f"✅ Model loaded in {elapsed:.1f}s")
            self.model_loaded = True
            
            # Check GPU memory
            gpu_info = self._get_gpu_info()
            logger.info(f"📊 GPU Memory: VRAM={gpu_info['vram_mb']:.0f}MB, GTT={gpu_info['gtt_mb']:.0f}MB")
            
            return True
        else:
            logger.error("❌ Failed to load model")
            return False
            
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text - all on GPU"""
        
        if not self.model_loaded:
            return "Error: Model not loaded"
            
        logger.info(f"\n💬 Generating response for: '{prompt}'")
        
        # Monitor GPU during generation
        gpu_monitor = GPUMonitor()
        gpu_monitor.start()
        
        start_time = time.time()
        
        try:
            # 1. Tokenize (could be GPU kernel in production)
            input_ids = self.tokenizer.encode(prompt)
            logger.info(f"📝 Tokenized: {len(input_ids)} tokens")
            
            # 2. Generate tokens on GPU
            generated_ids = self.pipeline.generate_tokens(
                input_ids,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # 3. Decode (could be GPU kernel in production)
            response = self.tokenizer.decode(generated_ids)
            
            elapsed = time.time() - start_time
            tokens_per_second = len(generated_ids) / elapsed if elapsed > 0 else 0
            
            # Get GPU usage
            max_gpu = gpu_monitor.stop()
            
            logger.info(f"✅ Generated {len(generated_ids)} tokens in {elapsed:.1f}s")
            logger.info(f"⚡ Performance: {tokens_per_second:.1f} TPS")
            logger.info(f"🎮 Max GPU usage: {max_gpu:.1f}%")
            
            if max_gpu < 10:
                logger.warning("⚠️ Low GPU usage - may be using CPU!")
                
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            gpu_monitor.stop()
            return f"Error: {str(e)}"
            
    def _get_gpu_info(self) -> dict:
        """Get current GPU memory info"""
        info = {'vram_mb': 0, 'gtt_mb': 0}
        
        try:
            result = subprocess.run(
                ['radeontop', '-d', '-', '-l', '1'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            for line in result.stdout.split('\n'):
                if 'vram' in line and 'gtt' in line:
                    # Parse VRAM
                    vram_part = line.split('vram')[1].split('mb')[0]
                    info['vram_mb'] = float(vram_part.strip().split()[-1])
                    
                    # Parse GTT
                    gtt_part = line.split('gtt')[1].split('mb')[0]
                    info['gtt_mb'] = float(gtt_part.strip().split()[-1])
                    break
        except:
            pass
            
        return info
        
    def benchmark(self):
        """Benchmark the system"""
        
        if not self.model_loaded:
            logger.error("Model not loaded")
            return
            
        logger.info("\n🏃 Running benchmark...")
        
        test_prompts = [
            "Magic Unicorn Unconventional Technology & Stuff is",
            "The future of AI is",
            "Hello world",
        ]
        
        total_tokens = 0
        total_time = 0
        
        for prompt in test_prompts:
            logger.info(f"\nTest: '{prompt}'")
            
            start = time.time()
            response = self.generate(prompt, max_tokens=20)
            elapsed = time.time() - start
            
            tokens = len(self.tokenizer.encode(response))
            total_tokens += tokens
            total_time += elapsed
            
            logger.info(f"Response: '{response}'")
            
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        logger.info(f"\n📊 Benchmark Results:")
        logger.info(f"   Total tokens: {total_tokens}")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Average TPS: {avg_tps:.1f}")
        

class GPUMonitor:
    """Monitor GPU usage during inference"""
    
    def __init__(self):
        self.max_gpu = 0
        self.monitoring = False
        self.thread = None
        
    def start(self):
        self.monitoring = True
        self.max_gpu = 0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
        return self.max_gpu
        
    def _monitor(self):
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['radeontop', '-d', '-', '-l', '0.1'],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                
                for line in result.stdout.split('\n'):
                    if 'gpu' in line:
                        parts = line.split(',')
                        for part in parts:
                            if 'gpu' in part and '%' in part:
                                gpu_str = part.split('gpu')[1].split('%')[0].strip()
                                try:
                                    gpu = float(gpu_str)
                                    self.max_gpu = max(self.max_gpu, gpu)
                                except:
                                    pass
                                break
            except:
                pass
            time.sleep(0.1)


def main():
    """Run the complete hardware inference system"""
    
    server = HardwareInferenceServer()
    
    # Initialize
    if not server.initialize():
        logger.error("Failed to initialize")
        return
        
    # Test generation
    logger.info("\n🦄 Testing Magic Unicorn generation...")
    response = server.generate(
        "Magic Unicorn Unconventional Technology & Stuff is a groundbreaking Applied AI company that",
        max_tokens=50
    )
    
    logger.info(f"\n📝 Generated text:")
    logger.info(f"'{response}'")
    
    # Run benchmark
    server.benchmark()
    
    # Cleanup
    if server.pipeline:
        server.pipeline.cleanup()


if __name__ == "__main__":
    main()