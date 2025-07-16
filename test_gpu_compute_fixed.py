#\!/usr/bin/env python3
"""
Test GPU compute performance directly
"""
import time
import numpy as np
import sys
import os

# Add INT8 support
os.environ['VULKAN_INT8_SUPPORT'] = '1'

from real_vulkan_matrix_compute import VulkanMatrixCompute

def main():
    print("🦄 Testing Unicorn Execution Engine GPU Performance")
    print("=" * 50)
    
    # Initialize Vulkan with INT8 support
    try:
        vulkan = VulkanMatrixCompute(use_fp16=False)
        if not vulkan.initialized:
            print("❌ Vulkan failed to initialize properly")
            return
        print("✅ Vulkan initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Vulkan: {e}")
        return
    
    # Test basic functionality
    print("\n🔥 Testing GPU Compute...")
    
    # Small test first
    a = np.random.randn(256, 256).astype(np.float32)
    b = np.random.randn(256, 256).astype(np.float32)
    
    try:
        start = time.time()
        result = vulkan.matrix_multiply(a, b)
        elapsed = time.time() - start
        print(f"✅ 256x256 matrix multiply: {elapsed*1000:.1f}ms")
        
        # Larger test
        a = np.random.randn(1024, 1024).astype(np.float32)
        b = np.random.randn(1024, 1024).astype(np.float32)
        
        start = time.time()
        result = vulkan.matrix_multiply(a, b)
        elapsed = time.time() - start
        
        tflops = (2 * 1024**3) / (elapsed * 1e12)
        print(f"✅ 1024x1024 matrix multiply: {elapsed*1000:.1f}ms ({tflops:.2f} TFLOPS)")
        
    except Exception as e:
        print(f"❌ Matrix multiply failed: {e}")
        return
    
    # Check memory
    try:
        mem_mb = vulkan.get_memory_usage()
        print(f"\n💾 GPU Memory Used: {mem_mb:.1f} MB")
    except:
        pass
    
    print("\n" + "="*50)
    print("💭 About 'Magic Unicorn Unconventional Technology & Stuff':")
    print("   Perfect name for an AI company that bypasses conventional")
    print("   frameworks for direct hardware acceleration\!")
    print("   The Unicorn Engine proves unconventional can be magical\! 🦄✨")
    print("\n⚡ GPU Compute: WORKING")
    print("🚀 Ready for full inference once model loads complete")

if __name__ == "__main__":
    main()
