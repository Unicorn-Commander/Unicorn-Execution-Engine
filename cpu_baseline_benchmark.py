#!/usr/bin/env python3.13
"""
🦄 CPU Baseline Benchmark - Real Model Inference
Test actual CPU performance with quantized models
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
import psutil

def benchmark_cpu_matrix_ops():
    """Benchmark raw CPU matrix multiplication performance"""
    print("🦄 CPU Matrix Operations Benchmark")
    print("=" * 50)
    
    # Test different sizes relevant to our models
    test_configs = [
        {"name": "4B Attention", "M": 128, "N": 2560, "K": 2560},  # seq_len x hidden x hidden
        {"name": "27B Attention", "M": 128, "N": 4608, "K": 4608},
        {"name": "4B MLP", "M": 128, "N": 10240, "K": 2560},       # hidden x ff_dim
        {"name": "27B MLP", "M": 128, "N": 18432, "K": 4608},
    ]
    
    results = {}
    
    for config in test_configs:
        name = config["name"]
        M, N, K = config["M"], config["N"], config["K"]
        
        print(f"\n📊 Testing {name} ({M}x{N} @ {N}x{K})")
        
        # Create test matrices
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)
        
        # Warm up
        for _ in range(3):
            _ = np.dot(A, B)
        
        # Benchmark
        times = []
        cpu_before = psutil.cpu_percent()
        
        for _ in range(10):
            start = time.time()
            C = np.dot(A, B)
            times.append(time.time() - start)
        
        cpu_after = psutil.cpu_percent()
        
        avg_time = np.mean(times)
        
        # Calculate FLOPS
        flops = 2 * M * N * K  # multiply-add operations
        gflops = flops / (avg_time * 1e9)
        
        # Calculate throughput
        matrix_size_mb = (A.nbytes + B.nbytes + C.nbytes) / (1024**2)
        bandwidth_gbs = matrix_size_mb / (avg_time * 1024)
        
        results[name] = {
            "time_ms": avg_time * 1000,
            "gflops": gflops,
            "bandwidth_gbs": bandwidth_gbs,
            "cpu_usage": cpu_after - cpu_before
        }
        
        print(f"   Time: {avg_time*1000:.1f}ms")
        print(f"   Performance: {gflops:.1f} GFLOPS")
        print(f"   Bandwidth: {bandwidth_gbs:.1f} GB/s")
        print(f"   CPU usage: +{cpu_after - cpu_before:.1f}%")
    
    return results

def estimate_model_performance(matrix_results):
    """Estimate full model performance based on matrix benchmarks"""
    print("\n🧮 Model Performance Estimation")
    print("=" * 40)
    
    models = {
        "4B": {
            "layers": 28,
            "hidden": 2560,
            "ff_dim": 10240,
            "heads": 20,
            "attention_key": "4B Attention",
            "mlp_key": "4B MLP"
        },
        "27B": {
            "layers": 32,
            "hidden": 4608,
            "ff_dim": 18432,
            "heads": 32,
            "attention_key": "27B Attention", 
            "mlp_key": "27B MLP"
        }
    }
    
    estimates = {}
    
    for name, config in models.items():
        print(f"\n📊 Gemma 3 {name} Estimation:")
        
        # Get matrix operation times
        attn_time = matrix_results[config["attention_key"]]["time_ms"]
        mlp_time = matrix_results[config["mlp_key"]]["time_ms"]
        
        # Estimate full layer time
        # Attention: QKV projection + attention computation + output projection
        attention_total = attn_time * 4  # Q, K, V, and output projections
        
        # MLP: up projection + down projection
        mlp_total = mlp_time * 2  # up and down projections
        
        # Layer normalization and other ops (estimate)
        other_ops = 5  # ms
        
        layer_time = attention_total + mlp_total + other_ops
        full_model_time = layer_time * config["layers"]
        
        # Tokens per second (assume generating 5 tokens on average)
        output_tokens = 5
        tps = output_tokens / (full_model_time / 1000)
        
        estimates[name] = {
            "layer_time_ms": layer_time,
            "full_model_time_ms": full_model_time,
            "estimated_tps": tps
        }
        
        print(f"   Layer time: {layer_time:.1f}ms")
        print(f"   Full model: {full_model_time:.1f}ms") 
        print(f"   Estimated TPS: {tps:.2f}")
    
    return estimates

def test_actual_inference():
    """Test actual inference if models are available"""
    print("\n🔬 Actual Model Inference Test")
    print("=" * 35)
    
    # Check if we have safetensors files
    model_4b_path = Path("quantized_models/gemma-3-4b-it-quantized")
    model_27b_path = Path("quantized_models/gemma-3-27b-it-layer-by-layer")
    
    if model_4b_path.exists():
        print(f"✅ Found 4B model: {model_4b_path}")
        safetensor_files = list(model_4b_path.glob("*.safetensors"))
        if safetensor_files:
            print(f"   Safetensors files: {len(safetensor_files)}")
            for f in safetensor_files[:3]:  # Show first 3
                size_mb = f.stat().st_size / (1024**2)
                print(f"     {f.name}: {size_mb:.1f}MB")
        
        # Try to load and test a weight tensor
        try:
            import safetensors
            import torch
            
            # Load first safetensor file to test
            first_file = safetensor_files[0]
            print(f"\n🧪 Testing weight loading from {first_file.name}...")
            
            start_time = time.time()
            with safetensors.safe_open(first_file, framework="pt") as f:
                keys = f.keys()
                first_key = list(keys)[0]
                tensor = f.get_tensor(first_key)
            load_time = time.time() - start_time
            
            print(f"   ✅ Loaded tensor '{first_key}'")
            print(f"   Shape: {tensor.shape}")
            print(f"   Dtype: {tensor.dtype}")
            print(f"   Load time: {load_time*1000:.1f}ms")
            
            # Test matrix multiplication with this weight
            if len(tensor.shape) == 2:
                input_tensor = torch.randn(128, tensor.shape[0], dtype=tensor.dtype)
                
                start_time = time.time()
                result = torch.matmul(input_tensor, tensor)
                compute_time = time.time() - start_time
                
                print(f"   ✅ Matrix mul test: {compute_time*1000:.1f}ms")
                print(f"   Result shape: {result.shape}")
                
                # Calculate performance
                flops = 2 * input_tensor.shape[0] * tensor.shape[0] * tensor.shape[1]
                gflops = flops / (compute_time * 1e9)
                print(f"   Performance: {gflops:.1f} GFLOPS")
            
        except ImportError:
            print("   ⚠️  safetensors/torch not available for loading test")
        except Exception as e:
            print(f"   ❌ Loading test failed: {e}")
    
    else:
        print("❌ No 4B model found for testing")

def main():
    print("🦄 CPU Baseline Benchmark Suite")
    print("=" * 60)
    
    # Test 1: Raw matrix operations
    matrix_results = benchmark_cpu_matrix_ops()
    
    # Test 2: Model performance estimation
    model_estimates = estimate_model_performance(matrix_results)
    
    # Test 3: Actual inference if possible
    test_actual_inference()
    
    # Summary
    print("\n" + "="*60)
    print("🏆 CPU BASELINE SUMMARY")
    print("="*60)
    
    print("\n📊 Matrix Performance:")
    for name, result in matrix_results.items():
        print(f"   {name}: {result['gflops']:.1f} GFLOPS")
    
    print("\n🚀 Estimated Model Performance:")
    for name, estimate in model_estimates.items():
        print(f"   Gemma 3 {name}: {estimate['estimated_tps']:.2f} TPS")
    
    # Write results to file for later comparison
    results = {
        "matrix_benchmarks": matrix_results,
        "model_estimates": model_estimates,
        "timestamp": time.time()
    }
    
    with open("cpu_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to cpu_baseline_results.json")

if __name__ == "__main__":
    main()