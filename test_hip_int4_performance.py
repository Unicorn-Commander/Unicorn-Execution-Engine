#!/usr/bin/env python3.13
"""Test HIP INT4 WMMA performance"""

import time
import torch
import numpy as np
from magic_unicorn_ultra_speed import MagicUnicornUltraSpeed

def test_int4_performance():
    print("🦄 TESTING INT4 WMMA PERFORMANCE")
    print("=" * 60)
    
    # Initialize engine
    engine = MagicUnicornUltraSpeed()
    
    # Test configurations
    test_configs = [
        (1, 32, "Small context"),
        (1, 128, "Medium context"),
        (1, 256, "Large context"),
    ]
    
    for batch_size, seq_len, desc in test_configs:
        print(f"\n🧪 Testing {desc} (batch={batch_size}, seq_len={seq_len})")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, 2560, dtype=torch.float32)
        
        # Create dummy weights for testing
        hidden_size = 2560
        weights = {
            'q_proj': torch.randn(hidden_size, hidden_size),
            'k_proj': torch.randn(hidden_size, hidden_size),
            'v_proj': torch.randn(hidden_size, hidden_size),
            'o_proj': torch.randn(hidden_size, hidden_size),
            'gate_proj': torch.randn(hidden_size, hidden_size * 4),
            'up_proj': torch.randn(hidden_size, hidden_size * 4),
            'down_proj': torch.randn(hidden_size * 4, hidden_size),
        }

        # Warmup
        for _ in range(2):
            _ = engine.transformer_layer_ultra(x, weights, layer_idx=0)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            output = engine.transformer_layer_ultra(x, weights, layer_idx=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        
        # Calculate tokens/sec
        tokens_per_sec = 1.0 / (min_time * 42)  # 42 layers
        
        print(f"   Average layer time: {avg_time*1000:.1f}ms")
        print(f"   Fastest layer time: {min_time*1000:.1f}ms")
        print(f"   Projected speed: {tokens_per_sec:.3f} tokens/sec")
        print(f"   vs 21 tok/s target: {tokens_per_sec/21:.3f}x")
        
        if tokens_per_sec >= 21.0:
            print(f"   🎯 TARGET ACHIEVED WITH INT4 WMMA!")

if __name__ == "__main__":
    test_int4_performance()