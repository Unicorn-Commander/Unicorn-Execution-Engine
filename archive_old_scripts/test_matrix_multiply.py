#!/usr/bin/env python3
"""Test matrix multiplication directly"""

import numpy as np
from real_vulkan_matrix_compute import VulkanMatrixCompute

def test_matrix_multiply():
    """Test Vulkan matrix multiplication"""
    
    print("🧪 Testing Vulkan Matrix Multiplication")
    print("=" * 60)
    
    # Initialize Vulkan
    vulkan = VulkanMatrixCompute()
    if not vulkan.initialize():
        print("❌ Failed to initialize Vulkan")
        return
    
    print("✅ Vulkan initialized")
    
    # Test cases
    tests = [
        # (A shape, B shape, description)
        ((1, 5, 512), (512, 512), "Embedding lookup (1 batch, 5 tokens)"),
        ((1, 1, 512), (512, 512), "Single token embedding"),
        ((5, 512), (512, 512), "No batch dimension"),
    ]
    
    for a_shape, b_shape, desc in tests:
        print(f"\n📊 Test: {desc}")
        print(f"   A shape: {a_shape}, B shape: {b_shape}")
        
        try:
            # Create test matrices
            A = np.random.randn(*a_shape).astype(np.float32)
            B = np.random.randn(*b_shape).astype(np.float32)
            
            # Compute with Vulkan
            result = vulkan.compute_matrix_multiply(A, B)
            
            print(f"   ✅ Result shape: {result.shape}")
            
            # Verify with numpy
            expected = np.matmul(A, B)
            if np.allclose(result, expected, rtol=1e-3):
                print(f"   ✅ Results match numpy")
            else:
                print(f"   ❌ Results differ from numpy")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    vulkan.cleanup()
    print("\n✅ Tests completed")


if __name__ == "__main__":
    test_matrix_multiply()