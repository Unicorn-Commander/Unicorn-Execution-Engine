#!/usr/bin/env python3
"""
Check TinyLlama model dimensions to understand NPU compatibility
"""

import subprocess
import re

def check_model_info():
    """Get TinyLlama model information"""
    print("🔍 TinyLlama Model Dimensions Analysis")
    print("=" * 50)
    
    # Run llama-cli to get model info
    cmd = ["./llama.cpp/build/bin/llama-cli", "-m", "tinyllama-1.1b-q4_k_m.gguf", "-n", "0", "--log-disable"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stderr
        
        # Extract key dimensions
        dimensions = {}
        patterns = {
            'n_ctx_train': r'n_ctx_train\s+=\s+(\d+)',
            'n_embd': r'n_embd\s+=\s+(\d+)',
            'n_layer': r'n_layer\s+=\s+(\d+)',
            'n_head': r'n_head\s+=\s+(\d+)',
            'n_head_kv': r'n_head_kv\s+=\s+(\d+)',
            'n_embd_head_k': r'n_embd_head_k\s+=\s+(\d+)',
            'n_embd_head_v': r'n_embd_head_v\s+=\s+(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                dimensions[key] = int(match.group(1))
        
        print("📊 TinyLlama Architecture:")
        for key, value in dimensions.items():
            print(f"   {key}: {value}")
        
        # Calculate attention tensor dimensions
        if 'n_embd_head_k' in dimensions and 'n_head' in dimensions:
            head_dim = dimensions['n_embd_head_k']
            num_heads = dimensions['n_head']
            num_kv_heads = dimensions.get('n_head_kv', num_heads)
            
            print(f"\n🧠 Attention Tensor Analysis:")
            print(f"   Head dimension: {head_dim}")
            print(f"   Query heads: {num_heads}")
            print(f"   KV heads: {num_kv_heads}")
            print(f"   Context length: {dimensions.get('n_ctx_train', 'unknown')}")
            
            # Check NPU kernel compatibility
            print(f"\n🔍 NPU Kernel Compatibility:")
            available_seq_lengths = [128, 256, 512, 1024]
            
            for seq_len in available_seq_lengths:
                kernel_file = f"attention_gemma3_4b_{seq_len}.xclbin"
                print(f"   Seq {seq_len}: {kernel_file} - {'✅ Available' if seq_len <= dimensions.get('n_ctx_train', 0) else '⚠️  Too long'}")
            
            # Suggest optimal configuration
            max_seq = dimensions.get('n_ctx_train', 2048)
            best_kernel = None
            for seq_len in sorted(available_seq_lengths):
                if seq_len <= max_seq:
                    best_kernel = seq_len
            
            if best_kernel:
                print(f"\n🎯 Recommended NPU Configuration:")
                print(f"   Best kernel: attention_gemma3_4b_{best_kernel}.xclbin")
                print(f"   Max sequence length: {best_kernel}")
                print(f"   Head dim: {head_dim} (kernel expects ~64)")
                print(f"   Heads: {num_heads} (kernel optimized for Gemma3)")
                
                if head_dim == 64:
                    print("   ✅ Head dimension matches NPU kernel!")
                else:
                    print("   ⚠️  Head dimension mismatch - may need kernel adjustment")
                    
            else:
                print("   ❌ No compatible NPU kernels found")
                
        print(f"\n📋 Summary:")
        print("   - TinyLlama uses different architecture than Gemma3")
        print("   - NPU kernels compiled for Gemma3 4B dimensions")
        print("   - Direct compatibility may require architecture adaptation")
        print("   - Alternative: Test with actual Gemma3 4B model")
            
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")

if __name__ == "__main__":
    check_model_info()