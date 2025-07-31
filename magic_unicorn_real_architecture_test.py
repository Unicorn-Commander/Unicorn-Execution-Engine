#!/usr/bin/env python3.13
"""
Magic Unicorn Real Architecture Test
Test performance with REAL Gemma3n architecture (2048 hidden, 35 layers)
This could explain the 674x performance gap!
"""

import torch
import time
import numpy as np
import sys
import os

# Add project path
sys.path.append('/home/ucadmin/Development/Unicorn-Execution-Engine')

class MagicUnicornRealArchitectureTest:
    """Test with actual Gemma3n architecture discovered from config"""
    
    def __init__(self):
        print("🦄🔍 MAGIC UNICORN REAL ARCHITECTURE TEST")
        print("=" * 70)
        print("🎯 HYPOTHESIS: Real architecture explains performance gap")
        
        # REAL Gemma3n architecture from config.json
        self.real_config = {
            'hidden_size': 2048,           # vs our test 2560
            'intermediate_size': 16384,    # vs our test 5376  
            'num_hidden_layers': 35,       # vs our test 42
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'activation_sparsity_pattern': [0.95] * 10 + [0.0] * 25  # 95% sparse first 10 layers
        }
        
        # Our previous test config
        self.test_config = {
            'hidden_size': 2560,
            'intermediate_size': 5376, 
            'num_hidden_layers': 42,
            'num_attention_heads': 8,
            'num_key_value_heads': 8
        }
        
        print(f"\n📊 ARCHITECTURE COMPARISON:")
        print(f"   {'Metric':<20} {'Our Test':<15} {'Real Gemma3n':<15} {'Ratio':<10}")
        print(f"   {'-'*60}")
        print(f"   {'Hidden size':<20} {self.test_config['hidden_size']:<15} {self.real_config['hidden_size']:<15} {self.test_config['hidden_size']/self.real_config['hidden_size']:.2f}x")
        print(f"   {'Intermediate':<20} {self.test_config['intermediate_size']:<15} {self.real_config['intermediate_size']:<15} {self.test_config['intermediate_size']/self.real_config['intermediate_size']:.2f}x")
        print(f"   {'Layers':<20} {self.test_config['num_hidden_layers']:<15} {self.real_config['num_hidden_layers']:<15} {self.test_config['num_hidden_layers']/self.real_config['num_hidden_layers']:.2f}x")
        print(f"   {'KV heads':<20} {self.test_config['num_key_value_heads']:<15} {self.real_config['num_key_value_heads']:<15} {self.test_config['num_key_value_heads']/self.real_config['num_key_value_heads']:.2f}x")
        
        # Calculate complexity difference
        test_ops = self.calculate_layer_ops(self.test_config)
        real_ops = self.calculate_layer_ops(self.real_config)
        
        print(f"\n💻 COMPUTATIONAL COMPLEXITY:")
        print(f"   Test config ops per layer: {test_ops:,}")
        print(f"   Real config ops per layer: {real_ops:,}")
        print(f"   Per-layer speedup potential: {test_ops/real_ops:.2f}x")
        
        # Total model complexity
        test_total = test_ops * self.test_config['num_hidden_layers']
        real_total = real_ops * self.real_config['num_hidden_layers']
        total_speedup = test_total / real_total
        
        print(f"   Total model speedup potential: {total_speedup:.2f}x")
        
        if total_speedup > 100:
            print(f"   🎯 MAJOR DISCOVERY! This could explain much of the 674x gap!")
        elif total_speedup > 10:
            print(f"   🔥 SIGNIFICANT! This explains part of the performance gap!")
        else:
            print(f"   ⚡ MODERATE impact on performance gap")
    
    def calculate_layer_ops(self, config):
        """Calculate operations per layer (simplified)"""
        h = config['hidden_size']
        i = config['intermediate_size']
        
        # QKV projections: 3 * (h * h)
        qkv_ops = 3 * (h * h)
        
        # Attention: roughly h^2 for scores + h^2 for values
        attn_ops = 2 * (h * h)
        
        # Output projection: h * h
        out_ops = h * h
        
        # FFN: gate (h*i) + up (h*i) + down (i*h)
        ffn_ops = 3 * (h * i)
        
        total = qkv_ops + attn_ops + out_ops + ffn_ops
        return total
    
    def test_real_architecture_performance(self):
        """Test performance with real Gemma3n architecture"""
        print(f"\n🚀 TESTING REAL ARCHITECTURE PERFORMANCE")
        print("=" * 50)
        
        # Use real architecture
        config = self.real_config
        hidden_size = config['hidden_size']
        intermediate_size = config['intermediate_size']
        
        # Test different sequence lengths
        test_configs = [
            (64, "Small context"),
            (128, "Medium context"),
            (256, "Large context"),
        ]
        
        results = {}
        
        for seq_len, config_name in test_configs:
            print(f"\n🧪 Testing {config_name} (seq_len={seq_len})")
            
            # Create test data with REAL architecture
            batch_size = 1
            x = torch.randn(batch_size, seq_len, hidden_size)
            
            # Create weights with CORRECT dimensions
            weights = {
                'q_proj': torch.randn(hidden_size, hidden_size),
                'k_proj': torch.randn(hidden_size, hidden_size),
                'v_proj': torch.randn(hidden_size, hidden_size),
                'o_proj': torch.randn(hidden_size, hidden_size),
                'gate_proj': torch.randn(intermediate_size, hidden_size),  # Note: different from before
                'up_proj': torch.randn(intermediate_size, hidden_size),
                'down_proj': torch.randn(hidden_size, intermediate_size),
            }
            
            # Test with OpenCL if available
            try:
                from optimized_hybrid_pipeline import OptimizedHybridEngine
                
                engine = OptimizedHybridEngine()
                
                if engine.igpu_context is not None:
                    print(f"   ⚡ Testing with OpenCL iGPU...")
                    
                    # Warmup
                    for i in range(2):
                        _ = engine.forward_layer_optimized(x, weights)
                    
                    # Benchmark
                    times = []
                    for i in range(3):
                        start = time.time()
                        output, layer_time = engine.forward_layer_optimized(x, weights)
                        times.append(layer_time)
                    
                    avg_time = sum(times) / len(times)
                    fastest_time = min(times)
                    
                    print(f"   ✅ Layer time: {fastest_time*1000:.1f}ms")
                    
                else:
                    print(f"   ⚠️ iGPU not available, using CPU...")
                    fastest_time = self.test_cpu_layer(x, weights)
                    print(f"   ✅ CPU layer time: {fastest_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"   ⚠️ OpenCL failed ({e}), using CPU...")
                fastest_time = self.test_cpu_layer(x, weights)
                print(f"   ✅ CPU layer time: {fastest_time*1000:.1f}ms")
            
            # Calculate full model performance
            num_layers = config['num_hidden_layers']
            full_time = fastest_time * num_layers
            tokens_per_sec = 1.0 / full_time
            
            # Store results
            results[seq_len] = {
                'layer_time': fastest_time,
                'full_time': full_time,
                'tokens_per_sec': tokens_per_sec
            }
            
            print(f"   📊 Full model ({num_layers} layers): {full_time:.2f}s")
            print(f"   🎯 Single token speed: {tokens_per_sec:.3f} tokens/sec")
            print(f"   📈 vs 21 tok/s baseline: {tokens_per_sec/21:.3f}x")
            
            if tokens_per_sec >= 21.0:
                print(f"   🎯 BASELINE ACHIEVED! Real architecture works!")
                break
            elif tokens_per_sec >= 10.0:
                print(f"   🔥 GETTING VERY CLOSE!")
            elif tokens_per_sec >= 1.0:
                print(f"   ⚡ MUCH BETTER! Architecture matters!")
            else:
                print(f"   🔧 Still needs optimization")
        
        return results
    
    def test_cpu_layer(self, x, weights):
        """Simple CPU transformer layer for testing"""
        def simple_layer(x, weights):
            # QKV
            q = torch.matmul(x, weights['q_proj'].T)
            k = torch.matmul(x, weights['k_proj'].T) 
            v = torch.matmul(x, weights['v_proj'].T)
            
            # Simple attention
            scale = 1.0 / (q.shape[-1] ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            
            # Output projection
            attn_out = torch.matmul(attn_out, weights['o_proj'].T)
            x = x + attn_out
            
            # FFN
            gate = torch.matmul(x, weights['gate_proj'].T)
            up = torch.matmul(x, weights['up_proj'].T)
            hidden = torch.nn.functional.silu(gate) * up
            output = torch.matmul(hidden, weights['down_proj'].T)
            
            return x + output
        
        # Benchmark
        times = []
        for i in range(3):
            start = time.time()
            _ = simple_layer(x, weights)
            times.append(time.time() - start)
        
        return min(times)
    
    def analyze_sparsity_impact(self):
        """Analyze the impact of 95% sparsity in first 10 layers"""
        print(f"\n🔍 SPARSITY ANALYSIS")
        print("=" * 30)
        
        sparsity_pattern = self.real_config['activation_sparsity_pattern']
        sparse_layers = sum(1 for s in sparsity_pattern if s > 0.5)
        dense_layers = len(sparsity_pattern) - sparse_layers
        
        print(f"   Sparse layers (95%): {sparse_layers}")
        print(f"   Dense layers: {dense_layers}")
        
        # Estimate speedup from sparsity
        # 95% sparsity could mean ~20x speedup for those layers
        sparse_speedup = 20.0
        effective_sparse_layers = sparse_layers / sparse_speedup
        
        effective_total_layers = effective_sparse_layers + dense_layers
        sparsity_speedup = len(sparsity_pattern) / effective_total_layers
        
        print(f"   Sparsity speedup potential: {sparsity_speedup:.2f}x")
        
        if sparsity_speedup > 2.0:
            print(f"   🚀 MAJOR! Sparsity could provide significant speedup!")
        else:
            print(f"   ⚡ Moderate sparsity impact")
        
        return sparsity_speedup

def test_real_architecture():
    """Main test function"""
    print("🦄🔍 MAGIC UNICORN REAL ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print("🔍 INVESTIGATING: Does real architecture explain the 674x gap?")
    
    tester = MagicUnicornRealArchitectureTest()
    
    # Test performance
    results = tester.test_real_architecture_performance()
    
    # Analyze sparsity
    sparsity_speedup = tester.analyze_sparsity_impact()
    
    # Final analysis
    print(f"\n🏁 ARCHITECTURE ANALYSIS RESULTS:")
    print("=" * 50)
    
    best_result = max(results.values(), key=lambda x: x['tokens_per_sec'])
    best_speed = best_result['tokens_per_sec']
    
    print(f"   Best performance: {best_speed:.3f} tokens/sec")
    print(f"   vs 21 tok/s target: {best_speed/21:.3f}x")
    
    remaining_gap = 21.0 / best_speed
    
    if remaining_gap < 10:
        print(f"   🎯 MAJOR PROGRESS! Only {remaining_gap:.1f}x gap remaining!")
        print(f"   🔥 Quantization (4-8x) could close this gap!")
    elif remaining_gap < 100:
        print(f"   ⚡ GOOD PROGRESS! {remaining_gap:.1f}x gap remaining")
        print(f"   🚀 Custom kernels + quantization promising!")
    else:
        print(f"   🔧 {remaining_gap:.1f}x gap still large - more optimization needed")
    
    # Architecture impact summary
    complexity_reduction = tester.calculate_layer_ops(tester.test_config) / tester.calculate_layer_ops(tester.real_config)
    layer_reduction = tester.test_config['num_hidden_layers'] / tester.real_config['num_hidden_layers']
    total_architectural_speedup = complexity_reduction * layer_reduction * sparsity_speedup
    
    print(f"\n📊 ARCHITECTURAL SPEEDUP BREAKDOWN:")
    print(f"   Per-layer complexity: {complexity_reduction:.2f}x")
    print(f"   Fewer layers: {layer_reduction:.2f}x")
    print(f"   Sparsity potential: {sparsity_speedup:.2f}x")
    print(f"   Total architectural: {total_architectural_speedup:.2f}x")
    
    if total_architectural_speedup > 100:
        print(f"   🎯 ARCHITECTURE EXPLAINS MOST OF THE GAP!")
    elif total_architectural_speedup > 10:
        print(f"   🔥 ARCHITECTURE EXPLAINS SIGNIFICANT PORTION!")
    
    return best_speed, remaining_gap

if __name__ == "__main__":
    test_real_architecture()