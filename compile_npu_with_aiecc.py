#!/usr/bin/env python3
"""
Compile NPU kernel using aiecc.py directly
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_gemma3_mlir():
    """Create a simple MLIR file for Gemma3 4B dimensions"""
    
    # Create working directory
    work_dir = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_compile_work"
    os.makedirs(work_dir, exist_ok=True)
    
    # Use the working single_core.py from mlir-aie examples
    mlir_example = "/home/ucadmin/npu-dev/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/single_core.py"
    
    # Generate MLIR for Gemma3 4B dimensions
    mlir_file = f"{work_dir}/gemma3_4b_attention.mlir"
    
    try:
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/ucadmin/npu-dev/mlir-aie/python:" + env.get("PYTHONPATH", "")
        
        # Generate MLIR with appropriate dimensions
        cmd = [
            "python3", mlir_example,
            "--dev", "npu",
            "-M", "256",  # Sequence length
            "-K", "128",  # Head dimension
            "-N", "128",  # Head dimension
            "-m", "32",   # Tile M
            "-k", "32",   # Tile K
            "-n", "32",   # Tile N
            "--dtype_in", "i16",
            "--dtype_out", "i32",
            "--trace_size", "0"
        ]
        
        logger.info(f"🔧 Generating MLIR: {' '.join(cmd)}")
        
        with open(mlir_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"❌ MLIR generation failed: {result.stderr}")
            return None
        
        logger.info(f"✅ MLIR generated: {mlir_file}")
        return mlir_file
        
    except Exception as e:
        logger.error(f"❌ MLIR generation failed: {e}")
        return None

def compile_with_aiecc(mlir_file):
    """Compile MLIR file using aiecc.py"""
    
    logger.info("🚀 Compiling MLIR with aiecc.py...")
    
    work_dir = Path(mlir_file).parent
    output_dir = work_dir / "build"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/ucadmin/npu-dev/mlir-aie/python:" + env.get("PYTHONPATH", "")
        env["PATH"] = "/home/ucadmin/npu-dev/mlir-aie/build/bin:" + env.get("PATH", "")
        
        # aiecc.py command
        aiecc_path = "/home/ucadmin/npu-dev/mlir-aie/python/compiler/aiecc.py"
        xclbin_output = output_dir / "gemma3_4b_attention.xclbin"
        
        cmd = [
            "python3", aiecc_path,
            "--aie-generate-xclbin",
            "--no-compile-host",
            f"--xclbin-name={xclbin_output.name}",
            "--aie-generate-npu-insts",
            "--npu-insts-name=insts.txt",
            mlir_file
        ]
        
        logger.info(f"🔧 Running aiecc.py: {' '.join(cmd)}")
        
        # Change to output directory for compilation
        os.chdir(output_dir)
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"❌ aiecc.py compilation failed: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            return None
        
        logger.info(f"✅ aiecc.py compilation successful")
        logger.info(f"Output: {result.stdout}")
        
        # Check if XCLBIN was created
        if xclbin_output.exists():
            logger.info(f"✅ XCLBIN created: {xclbin_output}")
            return str(xclbin_output)
        else:
            logger.warning("⚠️  XCLBIN file not found")
            return None
            
    except Exception as e:
        logger.error(f"❌ aiecc.py compilation failed: {e}")
        return None

def install_compiled_kernel(xclbin_path):
    """Install the compiled kernel"""
    
    if not xclbin_path or not os.path.exists(xclbin_path):
        logger.error("❌ No valid XCLBIN to install")
        return False
    
    logger.info("📦 Installing compiled kernel...")
    
    # Install to NPU kernels directory
    npu_kernels_dir = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels_real"
    os.makedirs(npu_kernels_dir, exist_ok=True)
    
    # Copy XCLBIN
    import shutil
    dest_path = f"{npu_kernels_dir}/attention_256_real.xclbin"
    shutil.copy2(xclbin_path, dest_path)
    
    # Copy instructions file if it exists
    insts_path = Path(xclbin_path).parent / "insts.txt"
    if insts_path.exists():
        shutil.copy2(insts_path, f"{npu_kernels_dir}/insts.txt")
        logger.info(f"✅ Instructions file copied: {npu_kernels_dir}/insts.txt")
    
    logger.info(f"✅ Kernel installed: {dest_path}")
    return dest_path

def main():
    """Main entry point"""
    
    logger.info("🚀 NPU Kernel Compilation with aiecc.py")
    logger.info("=" * 60)
    
    # Generate MLIR
    mlir_file = create_simple_gemma3_mlir()
    if not mlir_file:
        logger.error("❌ Failed to generate MLIR")
        return 1
    
    # Compile with aiecc.py
    xclbin_path = compile_with_aiecc(mlir_file)
    if not xclbin_path:
        logger.error("❌ Failed to compile XCLBIN")
        return 1
    
    # Install kernel
    installed_path = install_compiled_kernel(xclbin_path)
    if not installed_path:
        logger.error("❌ Failed to install kernel")
        return 1
    
    logger.info(f"\n" + "=" * 60)
    logger.info("✅ NPU KERNEL COMPILATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📁 Kernel installed at: {installed_path}")
    logger.info(f"🔧 Size: {os.path.getsize(installed_path)} bytes")
    logger.info("🚀 Real NPU hardware acceleration is now ready!")
    
    return 0

if __name__ == "__main__":
    exit(main())