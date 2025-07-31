#!/usr/bin/env python3.13
"""
Magic Unicorn ROCm SIMPLE - Basic GPU acceleration test
Focus on getting tensor operations working on GPU first
"""

import torch
import torch.nn.functional as F
import time
import os

def setup_rocm_environment():
    """Setup ROCm environment"""
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.3'
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['AMD_SERIALIZE_KERNEL'] = '1'  # Fix: Use 1 instead of 3
    os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA for compatibility
    os.environ['ROCR_VISIBLE_DEVICES'] = '0'  # Ensure GPU visibility
    print("✅ ROCm environment configured")

def test_basic_gpu_operations():
    """Test basic GPU operations work"""
    print("\n🔧 Testing basic GPU operations...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available")
        return False
    
    device = torch.device('cuda:0')
    print(f"✅ Device: {device}")
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Test basic tensor operations
        print("🔧 Testing tensor creation...")
        a = torch.randn(100, 100, device=device)
        print("✅ Tensor creation on GPU: OK")
        
        print("🔧 Testing matrix multiplication...")
        b = torch.randn(100, 100, device=device)
        c = torch.matmul(a, b)
        print("✅ Matrix multiplication on GPU: OK")
        
        print("🔧 Testing tensor transfer...")
        c_cpu = c.cpu()
        print("✅ GPU to CPU transfer: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def simple_transformer_layer_cpu(x, weights):
    """Simple CPU transformer layer for baseline"""
    print(f"🔧 CPU Layer: {x.shape}")
    start_time = time.time()
    
    batch_size, seq_len, hidden_size = x.shape
    
    # QKV projections
    q = torch.matmul(x, weights['q_proj'])
    k = torch.matmul(x, weights['k_proj'])
    v = torch.matmul(x, weights['v_proj'])
    
    # Reshape for attention
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Simple attention
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask
    if seq_len > 1:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    
    # Output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_out, weights['o_proj'])
    
    # Residual
    x = x + attn_output
    
    # Simple FFN
    gate = torch.matmul(x, weights['gate_proj'])
    up = torch.matmul(x, weights['up_proj'])
    hidden = F.silu(gate) * up
    output = torch.matmul(hidden, weights['down_proj'])
    
    # Final residual
    x = x + output
    
    cpu_time = time.time() - start_time
    print(f"⚡ CPU layer time: {cpu_time*1000:.1f}ms")
    
    return x, cpu_time

def simple_transformer_layer_gpu(x, weights, device):
    """Simple GPU transformer layer"""
    print(f"🚀 GPU Layer: {x.shape}")
    start_time = time.time()
    
    # Move to GPU
    x_gpu = x.to(device, non_blocking=True)
    weights_gpu = {k: v.to(device, non_blocking=True) for k, v in weights.items()}
    
    batch_size, seq_len, hidden_size = x_gpu.shape
    
    # QKV projections on GPU
    q = torch.matmul(x_gpu, weights_gpu['q_proj'])
    k = torch.matmul(x_gpu, weights_gpu['k_proj'])
    v = torch.matmul(x_gpu, weights_gpu['v_proj'])
    
    # Reshape for attention
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention on GPU
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Causal mask on GPU
    if seq_len > 1:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores.masked_fill_(mask, -65504.0)
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    
    # Output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_out, weights_gpu['o_proj'])
    
    # Residual
    x_gpu = x_gpu + attn_output
    
    # FFN on GPU
    gate = torch.matmul(x_gpu, weights_gpu['gate_proj'])
    up = torch.matmul(x_gpu, weights_gpu['up_proj'])
    hidden = F.silu(gate) * up
    output = torch.matmul(hidden, weights_gpu['down_proj'])
    
    # Final residual
    x_gpu = x_gpu + output
    
    # Move back to CPU
    result = x_gpu.cpu()
    
    gpu_time = time.time() - start_time
    print(f"🚀 GPU layer time: {gpu_time*1000:.1f}ms")
    
    return result, gpu_time

def benchmark_cpu_vs_gpu():
    """Benchmark CPU vs GPU performance"""
    print("\n🦄 SIMPLE ROCm PERFORMANCE TEST")
    print("=" * 50)
    
    # Test configuration
    batch_size = 1
    seq_len = 128
    hidden_size = 2560
    
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
    
    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size)
    weights = {
        'q_proj': torch.randn(hidden_size, hidden_size),
        'k_proj': torch.randn(hidden_size, hidden_size),
        'v_proj': torch.randn(hidden_size, hidden_size),
        'o_proj': torch.randn(hidden_size, hidden_size),
        'gate_proj': torch.randn(hidden_size, hidden_size * 4),
        'up_proj': torch.randn(hidden_size, hidden_size * 4),
        'down_proj': torch.randn(hidden_size * 4, hidden_size),
    }
    
    # CPU benchmark
    print("\n⚡ CPU Benchmark...")
    cpu_times = []
    for i in range(3):
        _, cpu_time = simple_transformer_layer_cpu(x.clone(), weights)
        cpu_times.append(cpu_time)
    
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("\n🚀 GPU Benchmark...")
        
        # Warmup
        for i in range(2):
            try:
                _, _ = simple_transformer_layer_gpu(x.clone(), weights, device)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"⚠️ GPU warmup failed: {e}")
                break
        
        # Benchmark
        gpu_times = []
        for i in range(3):
            try:
                torch.cuda.synchronize()
                _, gpu_time = simple_transformer_layer_gpu(x.clone(), weights, device)
                torch.cuda.synchronize()
                gpu_times.append(gpu_time)
            except Exception as e:
                print(f"⚠️ GPU benchmark failed: {e}")
                break
        
        if gpu_times:
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time
            
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   CPU average: {avg_cpu_time*1000:.1f}ms")
            print(f"   GPU average: {avg_gpu_time*1000:.1f}ms")
            print(f"   GPU speedup: {speedup:.2f}x")
            
            # Project full model performance
            layers = 42
            cpu_full = avg_cpu_time * layers
            gpu_full = avg_gpu_time * layers
            
            cpu_tok_per_sec = 1.0 / cpu_full
            gpu_tok_per_sec = 1.0 / gpu_full
            
            print(f"\n🎯 FULL MODEL PROJECTION:")
            print(f"   CPU: {cpu_tok_per_sec:.3f} tokens/sec")
            print(f"   GPU: {gpu_tok_per_sec:.3f} tokens/sec")
            print(f"   vs Ollama (21 tok/s): {gpu_tok_per_sec/21:.3f}x")
            
            if gpu_tok_per_sec >= 1.0:
                print("   🚀 GPU acceleration working well!")
            elif gpu_tok_per_sec >= 0.5:
                print("   ⚡ Good GPU acceleration, room for optimization")
            else:
                print("   🔧 GPU setup needs optimization")
                
            return gpu_tok_per_sec
        else:
            print("❌ GPU benchmark failed")
            return None
    else:
        print("❌ No GPU available for benchmark")
        return None

def main():
    """Main test function"""
    print("🦄🚀 MAGIC UNICORN ROCm SIMPLE TEST")
    print("=" * 60)
    
    # Setup environment
    setup_rocm_environment()
    
    # Test basic GPU operations
    if test_basic_gpu_operations():
        print("✅ Basic GPU operations working!")
        
        # Run benchmark
        performance = benchmark_cpu_vs_gpu()
        
        if performance:
            print(f"\n🏁 RESULT: {performance:.3f} tokens/sec achieved")
            print("✅ ROCm GPU acceleration foundation is working!")
            print("🎯 Ready for advanced optimization!")
        else:
            print("\n⚠️ GPU benchmark issues - needs debugging")
    else:
        print("❌ Basic GPU operations failed")

if __name__ == "__main__":
    main()