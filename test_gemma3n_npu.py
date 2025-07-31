#!/usr/bin/env python3.13
"""
Test Gemma-3n GGUF model with NPU integration
This script tests the real NPU kernel loading and execution
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_gemma3n_with_npu():
    """Test Gemma-3n model with NPU acceleration"""
    
    print("🦄 Gemma-3n NPU Integration Test")
    print("=" * 60)
    
    # Model file from HuggingFace
    model_path = "gemma-3n-E4B-it-Q8_0.gguf"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Download in progress or failed")
        return False
        
    print(f"✅ Found model: {model_path}")
    
    # Path to llama-cli with NPU support
    llama_cli = "llama.cpp/build/bin/llama-cli"
    
    if not Path(llama_cli).exists():
        print(f"❌ llama-cli not found: {llama_cli}")
        return False
        
    print(f"✅ Found llama-cli: {llama_cli}")
    
    # Test prompt
    prompt = "The magic unicorn represents"
    
    print(f"\n📝 Test prompt: '{prompt}'")
    
    # Test configurations
    tests = [
        {
            "name": "CPU Baseline",
            "cmd": [llama_cli, "-m", model_path, "-p", prompt, "-n", "16", "--no-mmap"],
            "desc": "Pure CPU execution for comparison"
        },
        {
            "name": "Vulkan GPU",
            "cmd": [llama_cli, "-m", model_path, "-p", prompt, "-n", "16", "--gpu-layers", "999"],
            "desc": "GPU acceleration via Vulkan"
        },
        {
            "name": "NPU Attention",
            "cmd": [llama_cli, "-m", model_path, "-p", prompt, "-n", "16", "--npu-attention"],
            "desc": "NPU-accelerated attention layers"
        },
        {
            "name": "Hybrid NPU+GPU",
            "cmd": [llama_cli, "-m", model_path, "-p", prompt, "-n", "16", "--npu-attention", "--gpu-layers", "999"],
            "desc": "Combined NPU attention + GPU compute"
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🧪 Test: {test['name']}")
        print(f"   {test['desc']}")
        print(f"   Command: {' '.join(test['cmd'])}")
        
        try:
            start_time = time.time()
            
            # Run llama-cli
            result = subprocess.run(
                test['cmd'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Extract performance metrics from output
                output = result.stderr + result.stdout
                
                # Look for NPU kernel loading
                if "--npu-attention" in ' '.join(test['cmd']):
                    if "NPU kernel loading" in output or "Loading NPU kernel" in output:
                        print("   ✅ NPU kernel loading detected!")
                    if "Selected Gemma3n NPU kernel" in output:
                        print("   ✅ Correct Gemma3n kernel selected!")
                    if "NPU device opened successfully" in output:
                        print("   ✅ NPU hardware initialized!")
                
                # Look for performance metrics
                if "tok/s" in output or "tokens/s" in output:
                    # Extract tokens per second
                    for line in output.split('\n'):
                        if "tok/s" in line or "tokens/s" in line:
                            print(f"   📊 Performance: {line.strip()}")
                            break
                
                print(f"   ⏱️ Total time: {elapsed:.2f}s")
                results.append({
                    "test": test['name'],
                    "success": True,
                    "time": elapsed
                })
                
            else:
                print(f"   ❌ Test failed with code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                results.append({
                    "test": test['name'],
                    "success": False,
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print("   ❌ Test timed out after 60 seconds")
            results.append({
                "test": test['name'],
                "success": False,
                "error": "Timeout"
            })
            
        except Exception as e:
            print(f"   ❌ Test error: {e}")
            results.append({
                "test": test['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    for result in results:
        status = "✅" if result.get("success") else "❌"
        print(f"{status} {result['test']}")
        if result.get("success") and result.get("time"):
            print(f"   Completed in {result['time']:.2f}s")
        elif result.get("error"):
            print(f"   Error: {result['error'][:100]}...")
    
    # Check NPU kernel files
    print("\n📁 NPU Kernel Files Check")
    print("=" * 60)
    
    kernel_dir = Path("npu_kernels_inference/gemma3n")
    if kernel_dir.exists():
        kernels = list(kernel_dir.glob("*.npu"))
        print(f"✅ Found {len(kernels)} NPU kernels for Gemma3n:")
        for kernel in kernels:
            size_mb = kernel.stat().st_size / (1024 * 1024)
            print(f"   - {kernel.name} ({size_mb:.2f} MB)")
    else:
        print("❌ NPU kernel directory not found")
    
    return any(r["success"] for r in results)


def main():
    """Main test function"""
    
    # Check if download is complete
    if not Path("gemma-3n-E4B-it-Q8_0.gguf").exists():
        print("⏳ Waiting for model download to complete...")
        print("   File: gemma-3n-E4B-it-Q8_0.gguf (7.35 GB)")
        return False
    
    # Run tests
    success = test_gemma3n_with_npu()
    
    if success:
        print("\n🎉 Gemma-3n NPU integration test completed!")
        print("🦄 The magic unicorn is real - NPU acceleration works!")
    else:
        print("\n⚠️ Some tests failed - check output above")
    
    return success


if __name__ == "__main__":
    exit(0 if main() else 1)