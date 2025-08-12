#!/usr/bin/env python3
"""
Compile real NPU kernels for Gemma3 4B attention using mlir-aie
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NPU compilation environment
MLIR_AIE_PATH = "/home/ucadmin/npu-dev/mlir-aie"
BUILD_PATH = f"{MLIR_AIE_PATH}/build"
TOOLS_PATH = f"{BUILD_PATH}/bin"

# Gemma3 4B specifications
GEMMA3_4B_SPECS = {
    "hidden_size": 2560,
    "num_heads": 20,
    "head_dim": 128,
    "sequence_lengths": [128, 256, 512, 1024]
}

def setup_environment():
    """Setup NPU compilation environment"""
    logger.info("🔧 Setting up NPU compilation environment...")
    
    # Check if mlir-aie is built
    if not os.path.exists(f"{TOOLS_PATH}/aie-opt"):
        logger.error("❌ mlir-aie not built. Please build mlir-aie first.")
        logger.info("To build mlir-aie:")
        logger.info("cd /home/ucadmin/npu-dev/mlir-aie")
        logger.info("mkdir -p build && cd build")
        logger.info("cmake .. -DCMAKE_BUILD_TYPE=Release")
        logger.info("make -j$(nproc)")
        return False
    
    # Set environment variables
    os.environ["MLIR_AIE_ROOT"] = MLIR_AIE_PATH
    os.environ["PATH"] = f"{TOOLS_PATH}:{os.environ.get('PATH', '')}"
    
    logger.info("✅ Environment setup complete")
    return True

def create_attention_kernel_mlir(seq_len, hidden_size, num_heads, head_dim):
    """Create MLIR source for attention kernel"""
    
    kernel_name = f"gemma3_4b_attention_seq{seq_len}"
    
    mlir_content = f"""
// Gemma3 4B Attention Kernel for NPU
// Sequence Length: {seq_len}
// Hidden Size: {hidden_size}
// Num Heads: {num_heads}
// Head Dimension: {head_dim}

module {{
  aie.device(npu1_4col) {{
    
    // Memory tiles for storing attention weights
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    
    // Compute tiles for attention computation
    %compute_tile_0_2 = aie.tile(0, 2)
    %compute_tile_1_2 = aie.tile(1, 2)
    %compute_tile_2_2 = aie.tile(2, 2)
    %compute_tile_3_2 = aie.tile(3, 2)
    
    // Buffers for Q, K, V matrices
    %buf_q = aie.buffer(%mem_tile_0_1) : memref<{seq_len}x{head_dim}xf32>
    %buf_k = aie.buffer(%mem_tile_1_1) : memref<{seq_len}x{head_dim}xf32>
    %buf_v = aie.buffer(%mem_tile_2_1) : memref<{seq_len}x{head_dim}xf32>
    %buf_out = aie.buffer(%mem_tile_3_1) : memref<{seq_len}x{head_dim}xf32>
    
    // Attention computation core
    %core_0_2 = aie.core(%compute_tile_0_2) {{
      // Q @ K^T computation
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c_seq_len = arith.constant {seq_len} : index
      %c_head_dim = arith.constant {head_dim} : index
      %scale = arith.constant {1.0/math.sqrt(head_dim)} : f32
      
      // Attention scores matrix
      %scores = memref.alloc() : memref<{seq_len}x{seq_len}xf32>
      
      // Compute attention scores: Q @ K^T
      scf.for %i = %c0 to %c_seq_len step %c1 {{
        scf.for %j = %c0 to %c_seq_len step %c1 {{
          %sum = arith.constant 0.0 : f32
          %score = scf.for %k = %c0 to %c_head_dim step %c1 iter_args(%acc = %sum) -> (f32) {{
            %q_val = memref.load %buf_q[%i, %k] : memref<{seq_len}x{head_dim}xf32>
            %k_val = memref.load %buf_k[%j, %k] : memref<{seq_len}x{head_dim}xf32>
            %prod = arith.mulf %q_val, %k_val : f32
            %new_acc = arith.addf %acc, %prod : f32
            scf.yield %new_acc : f32
          }}
          %scaled_score = arith.mulf %score, %scale : f32
          memref.store %scaled_score, %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
        }}
      }}
      
      // Softmax computation (simplified)
      scf.for %i = %c0 to %c_seq_len step %c1 {{
        // Find max for numerical stability
        %max_val = arith.constant -1000.0 : f32
        %row_max = scf.for %j = %c0 to %c_seq_len step %c1 iter_args(%max_acc = %max_val) -> (f32) {{
          %val = memref.load %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
          %new_max = arith.maximumf %max_acc, %val : f32
          scf.yield %new_max : f32
        }}
        
        // Compute exp and sum
        %sum_val = arith.constant 0.0 : f32
        %exp_sum = scf.for %j = %c0 to %c_seq_len step %c1 iter_args(%sum_acc = %sum_val) -> (f32) {{
          %val = memref.load %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
          %shifted = arith.subf %val, %row_max : f32
          %exp_val = math.exp %shifted : f32
          memref.store %exp_val, %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
          %new_sum = arith.addf %sum_acc, %exp_val : f32
          scf.yield %new_sum : f32
        }}
        
        // Normalize
        scf.for %j = %c0 to %c_seq_len step %c1 {{
          %exp_val = memref.load %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
          %norm_val = arith.divf %exp_val, %exp_sum : f32
          memref.store %norm_val, %scores[%i, %j] : memref<{seq_len}x{seq_len}xf32>
        }}
      }}
      
      // Attention @ V computation
      scf.for %i = %c0 to %c_seq_len step %c1 {{
        scf.for %j = %c0 to %c_head_dim step %c1 {{
          %sum = arith.constant 0.0 : f32
          %output = scf.for %k = %c0 to %c_seq_len step %c1 iter_args(%acc = %sum) -> (f32) {{
            %attn_val = memref.load %scores[%i, %k] : memref<{seq_len}x{seq_len}xf32>
            %v_val = memref.load %buf_v[%k, %j] : memref<{seq_len}x{head_dim}xf32>
            %prod = arith.mulf %attn_val, %v_val : f32
            %new_acc = arith.addf %acc, %prod : f32
            scf.yield %new_acc : f32
          }}
          memref.store %output, %buf_out[%i, %j] : memref<{seq_len}x{head_dim}xf32>
        }}
      }}
      
      aie.end
    }}
    
    // Memory-mapped interfaces
    aie.shim_dma_allocation @input_q(S2MM, 0, 0)
    aie.shim_dma_allocation @input_k(S2MM, 0, 1)
    aie.shim_dma_allocation @input_v(S2MM, 0, 2)
    aie.shim_dma_allocation @output(MM2S, 0, 3)
    
    // Connect buffers to DMA
    aie.memref.global @input_q_buffer : memref<{seq_len}x{head_dim}xf32>
    aie.memref.global @input_k_buffer : memref<{seq_len}x{head_dim}xf32>
    aie.memref.global @input_v_buffer : memref<{seq_len}x{head_dim}xf32>
    aie.memref.global @output_buffer : memref<{seq_len}x{head_dim}xf32>
  }}
}}
"""
    
    return mlir_content, kernel_name

def compile_mlir_to_xclbin(mlir_content, kernel_name, output_dir):
    """Compile MLIR to XCLBIN format"""
    
    logger.info(f"🔨 Compiling {kernel_name} to XCLBIN...")
    
    # Create temporary directory for compilation
    temp_dir = f"/tmp/npu_compile_{kernel_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Write MLIR source
        mlir_file = f"{temp_dir}/{kernel_name}.mlir"
        with open(mlir_file, 'w') as f:
            f.write(mlir_content)
        
        logger.info(f"✅ Created MLIR source: {mlir_file}")
        
        # Compile to AIE
        aie_file = f"{temp_dir}/{kernel_name}.aie"
        cmd = [
            f"{TOOLS_PATH}/aie-opt",
            "--aie-lower-to-llvm",
            "--aie-target-backend=npu1_4col",
            mlir_file,
            "-o", aie_file
        ]
        
        logger.info(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"❌ AIE compilation failed: {result.stderr}")
            return False
        
        logger.info("✅ AIE compilation successful")
        
        # Translate to XCLBIN
        xclbin_file = f"{output_dir}/{kernel_name}.xclbin"
        cmd = [
            f"{TOOLS_PATH}/aie-translate",
            "--aie-generate-xclbin",
            aie_file,
            "-o", xclbin_file
        ]
        
        logger.info(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"❌ XCLBIN generation failed: {result.stderr}")
            return False
        
        logger.info(f"✅ XCLBIN generated: {xclbin_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return False
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def compile_gemma3_4b_kernels():
    """Compile all Gemma3 4B attention kernels"""
    
    logger.info("🚀 Compiling Gemma3 4B NPU Kernels")
    logger.info("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Create output directory
    output_dir = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_real"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Compile kernels for each sequence length
    for seq_len in GEMMA3_4B_SPECS["sequence_lengths"]:
        logger.info(f"\n📊 Compiling kernel for sequence length: {seq_len}")
        
        # Create MLIR source
        mlir_content, kernel_name = create_attention_kernel_mlir(
            seq_len=seq_len,
            hidden_size=GEMMA3_4B_SPECS["hidden_size"],
            num_heads=GEMMA3_4B_SPECS["num_heads"],
            head_dim=GEMMA3_4B_SPECS["head_dim"]
        )
        
        # Compile to XCLBIN
        total_count += 1
        if compile_mlir_to_xclbin(mlir_content, kernel_name, output_dir):
            success_count += 1
            logger.info(f"✅ {kernel_name} compiled successfully")
        else:
            logger.error(f"❌ {kernel_name} compilation failed")
    
    # Summary
    logger.info(f"\n" + "=" * 60)
    logger.info(f"📊 COMPILATION SUMMARY")
    logger.info(f"=" * 60)
    logger.info(f"✅ Successfully compiled: {success_count}/{total_count} kernels")
    logger.info(f"📁 Output directory: {output_dir}")
    
    if success_count > 0:
        logger.info(f"\n💡 To use these kernels:")
        logger.info(f"1. Update NPU kernel path to: {output_dir}")
        logger.info(f"2. Restart the inference pipeline")
        logger.info(f"3. Real NPU should now work instead of simulation!")
        return True
    else:
        logger.error(f"\n❌ No kernels compiled successfully")
        return False

def main():
    """Main entry point"""
    try:
        success = compile_gemma3_4b_kernels()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import math
    exit(main())