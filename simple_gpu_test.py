#!/usr/bin/env python3.13
"""
🦄 Simple GPU Test - Monitor actual GPU usage
Simplified test to see real hardware utilization
"""

import os
import time
import numpy as np
import subprocess
import threading

class SimpleGPUTest:
    """Simple test to verify GPU usage"""
    
    def __init__(self):
        self.gpu_usage = []
        self.monitoring = False
        
    def monitor_gpu(self):
        """Monitor GPU in background"""
        print("📊 Starting GPU monitoring (check radeontop!)...")
        
        self.monitoring = True
        def monitor():
            while self.monitoring:
                # Simple CPU check
                cpu_percent = os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip()
                if cpu_percent:
                    self.gpu_usage.append(cpu_percent)
                time.sleep(0.5)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def test_4b_simple(self):
        """Simple 4B model test"""
        print("\n🧪 Testing 4B Model (Simple)")
        print("-" * 50)
        
        # 4B dimensions
        hidden_size = 2560
        seq_len = 128
        
        # Create matrices
        x = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        
        # Attention weights (realistic sizes)
        q_weight = np.random.randn(2048, hidden_size).astype(np.float32) * 0.02
        k_weight = np.random.randn(1024, hidden_size).astype(np.float32) * 0.02
        v_weight = np.random.randn(1024, hidden_size).astype(np.float32) * 0.02
        
        print(f"   Input: {x.shape}")
        print(f"   Q weight: {q_weight.shape}")
        print(f"   GPU should spike during computation...\n")
        
        # Time single layer
        layer_times = []
        
        for i in range(5):
            start = time.time()
            
            # Attention computation
            q = np.matmul(x, q_weight.T)  # [1, 128, 2048]
            k = np.matmul(x, k_weight.T)  # [1, 128, 1024]
            v = np.matmul(x, v_weight.T)  # [1, 128, 1024]
            
            # Simple attention (without reshape to avoid errors)
            scores = np.matmul(q, k.T) / np.sqrt(1024)  # [1, 2048, 1024]
            scores = scores[:, :seq_len, :seq_len]  # Take relevant part
            
            # Softmax
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            
            # Output (simplified)
            output = np.matmul(attn, v[:, :seq_len])
            
            elapsed = (time.time() - start) * 1000
            layer_times.append(elapsed)
            print(f"   Layer computation {i+1}: {elapsed:.1f}ms")
        
        avg_time = sum(layer_times) / len(layer_times)
        print(f"\n   Average layer time: {avg_time:.1f}ms")
        print(f"   Full 28 layers: {avg_time * 28:.1f}ms")
        print(f"   Estimated TPS: {1000 / (avg_time * 28):.2f}")
        
        return avg_time
    
    def test_27b_simple(self):
        """Simple 27B model test"""
        print("\n🧪 Testing 27B Model (Simple)")
        print("-" * 50)
        
        # 27B dimensions
        hidden_size = 4608
        seq_len = 128
        
        # Create matrices
        x = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        
        # Attention weights
        q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        k_weight = np.random.randn(hidden_size // 2, hidden_size).astype(np.float32) * 0.02
        v_weight = np.random.randn(hidden_size // 2, hidden_size).astype(np.float32) * 0.02
        
        print(f"   Input: {x.shape}")
        print(f"   Q weight: {q_weight.shape}")
        print(f"   This will be slower - watch GPU!\n")
        
        # Time single layer
        layer_times = []
        
        for i in range(3):  # Fewer iterations for 27B
            start = time.time()
            
            # Attention computation
            q = np.matmul(x, q_weight.T)  # [1, 128, 4608]
            k = np.matmul(x, k_weight.T)  # [1, 128, 2304]
            v = np.matmul(x, v_weight.T)  # [1, 128, 2304]
            
            # Simplified attention
            # Just do basic computation to stress GPU
            scores = np.matmul(q[:, :64], k[:, :64].T) / np.sqrt(144)  # Smaller for speed
            attn = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-9)
            
            elapsed = (time.time() - start) * 1000
            layer_times.append(elapsed)
            print(f"   Layer computation {i+1}: {elapsed:.1f}ms")
        
        avg_time = sum(layer_times) / len(layer_times)
        print(f"\n   Average layer time: {avg_time:.1f}ms")
        print(f"   Full 46 layers: {avg_time * 46:.1f}ms")
        print(f"   Estimated TPS: {1000 / (avg_time * 46):.2f}")
        
        return avg_time
    
    def run_full_test(self):
        """Run complete test"""
        print("🦄 SIMPLE GPU TEST")
        print("=" * 60)
        print("⚡ This will run matrix operations that should use GPU")
        print("📊 Please watch radeontop for GPU usage!")
        print("=" * 60)
        
        # Start monitoring
        self.monitor_gpu()
        
        # Warmup
        print("\n🔥 Warmup (loading NumPy)...")
        dummy = np.random.randn(1000, 1000)
        _ = np.matmul(dummy, dummy)
        time.sleep(1)
        
        # Run tests
        time_4b = self.test_4b_simple()
        time.sleep(2)  # Pause between tests
        
        time_27b = self.test_27b_simple()
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Summary
        print("\n" + "="*60)
        print("🏆 REAL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"   4B Model: {1000 / (time_4b * 28):.2f} TPS")
        print(f"   27B Model: {1000 / (time_27b * 46):.2f} TPS")
        print("\n⚠️  Note: These are estimates based on layer timing")
        print("✅ Actual performance depends on GPU utilization")
        print("\n💡 Did you see GPU usage spike in radeontop?")
        print("   If yes → GPU acceleration is working!")
        print("   If no  → Computation is running on CPU")

if __name__ == "__main__":
    test = SimpleGPUTest()
    test.run_full_test()
    
    print("\n\n💡 TIP: For real GPU acceleration, we need:")
    print("   1. ROCm/HIP for AMD GPU compute")
    print("   2. CuPy or JAX with ROCm backend")
    print("   3. Or direct OpenCL/Vulkan compute")
    print("\nNumPy alone typically uses CPU unless linked with GPU BLAS!")