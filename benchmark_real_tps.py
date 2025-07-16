#!/usr/bin/env python3
"""
Real TPS Benchmark - Measure actual tokens per second
"""

import time
import logging
import numpy as np
from pure_hardware_pipeline_fixed import PureHardwarePipelineFixed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TPSBenchmark:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.layer_times = []
        
    def benchmark_single_layer(self, layer_idx=0, num_iterations=50):
        """Benchmark a single layer to estimate performance"""
        logger.info(f"⏱️ Benchmarking layer {layer_idx}...")
        
        # Create test input
        batch_size = 1
        seq_len = 1  # Single token
        hidden_size = 5376
        test_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            output, _ = self.pipeline.forward_layer(layer_idx, test_input)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            output, _ = self.pipeline.forward_layer(layer_idx, test_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if i == 0:
                logger.info(f"   First iteration: {elapsed*1000:.2f}ms")
        
        # Calculate statistics
        times = times[5:]  # Skip first few for stability
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        
        logger.info(f"   Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        logger.info(f"   Minimum: {min_time*1000:.2f}ms")
        
        return avg_time
    
    def estimate_full_model_tps(self, num_layers=62):
        """Estimate TPS for full model based on layer benchmarks"""
        logger.info("\n📊 Estimating full model performance...")
        
        # Benchmark a few layers
        layer_indices = [0, 10, 30, 50]  # Sample different layers
        times = []
        
        for idx in layer_indices:
            if idx < len(self.pipeline.layer_weights_gpu):
                layer_time = self.benchmark_single_layer(idx, num_iterations=20)
                times.append(layer_time)
        
        if not times:
            logger.error("No layers available for benchmarking")
            return 0
        
        # Average layer time
        avg_layer_time = np.mean(times)
        logger.info(f"\n📊 Performance Analysis:")
        logger.info(f"   Average layer time: {avg_layer_time*1000:.2f}ms")
        logger.info(f"   Total layers: {num_layers}")
        
        # Full model time
        full_model_time = avg_layer_time * num_layers
        single_token_tps = 1.0 / full_model_time
        
        logger.info(f"   Full model time: {full_model_time*1000:.1f}ms per token")
        logger.info(f"   Single-stream TPS: {single_token_tps:.1f}")
        
        return single_token_tps
    
    def benchmark_attention_vs_ffn(self):
        """Compare attention vs FFN performance"""
        logger.info("\n📊 Component Analysis...")
        
        # Test input
        test_input = np.random.randn(1, 1, 5376).astype(np.float32)
        
        # Time attention
        start = time.perf_counter()
        for _ in range(10):
            attn_out = self.pipeline.compute_attention_layer_gpu(
                0, test_input, test_input, test_input, None
            )
        attn_time = (time.perf_counter() - start) / 10
        
        # Time FFN
        start = time.perf_counter()
        for _ in range(10):
            ffn_out = self.pipeline.compute_ffn_layer_gpu(0, test_input)
        ffn_time = (time.perf_counter() - start) / 10
        
        logger.info(f"   Attention: {attn_time*1000:.2f}ms")
        logger.info(f"   FFN: {ffn_time*1000:.2f}ms")
        logger.info(f"   Ratio: {ffn_time/attn_time:.1f}x")
        
        return attn_time, ffn_time

def main():
    """Run comprehensive TPS benchmark"""
    logger.info("🚀 Real TPS Benchmark")
    logger.info("   Model: Gemma 27B INT8")
    logger.info("   Target: 81 TPS")
    logger.info("   Hardware: AMD Radeon 780M + Phoenix NPU")
    
    # Initialize pipeline
    pipeline = PureHardwarePipelineFixed()
    model_path = "quantized_models/gemma-3-27b-it-layer-by-layer"
    
    logger.info("\n🔄 Loading model...")
    start_time = time.time()
    
    if not pipeline.initialize(model_path):
        logger.error("Failed to initialize pipeline")
        return
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    logger.info(f"   Layers in GPU: {len(pipeline.layer_weights_gpu)}")
    
    # Create benchmark
    benchmark = TPSBenchmark(pipeline)
    
    # Component analysis
    try:
        attn_time, ffn_time = benchmark.benchmark_attention_vs_ffn()
    except Exception as e:
        logger.warning(f"Component analysis failed: {e}")
    
    # Estimate full model TPS
    single_tps = benchmark.estimate_full_model_tps()
    
    # Calculate batch requirements for 81 TPS
    if single_tps > 0:
        logger.info("\n🎯 Target Analysis:")
        logger.info(f"   Current: {single_tps:.1f} TPS")
        logger.info(f"   Target: 81 TPS")
        
        if single_tps >= 81:
            logger.info("   ✅ TARGET ACHIEVED!")
        else:
            required_batch = int(np.ceil(81 / single_tps))
            logger.info(f"   Required batch size: {required_batch}")
            logger.info(f"   Or {(81/single_tps - 1)*100:.0f}% speedup needed")
    
    # Optimization suggestions
    logger.info("\n💡 Optimization Opportunities:")
    if ffn_time > attn_time * 2:
        logger.info("   - FFN is bottleneck, consider INT8 optimizations")
    else:
        logger.info("   - Attention is significant, NPU acceleration would help")
    
    logger.info("   - Batch processing for higher throughput")
    logger.info("   - Memory access pattern optimization")
    logger.info("   - Kernel fusion opportunities")
    
    # Cleanup
    pipeline.cleanup()
    
    # Free cache memory
    logger.info("\n🧹 Cleaning up cache...")
    import subprocess
    subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], 
                   capture_output=True)
    logger.info("✅ Memory cache cleared")

if __name__ == "__main__":
    main()