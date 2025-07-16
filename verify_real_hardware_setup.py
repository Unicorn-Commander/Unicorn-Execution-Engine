#!/usr/bin/env python3
"""
Verify Real Hardware Setup for NPU Performance Testing
Checks all requirements before running real NPU tests
"""

import sys
import subprocess
import logging
from pathlib import Path
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_npu_hardware():
    """Check NPU Phoenix hardware availability"""
    logger.info("⚡ Checking NPU Phoenix hardware...")
    
    try:
        result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ xrt-smi command failed")
            return False
        
        if 'Phoenix' in result.stdout or 'NPU' in result.stdout:
            logger.info("✅ NPU Phoenix detected")
            logger.info(f"   Device info: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ NPU Phoenix not detected")
            logger.error(f"   xrt-smi output: {result.stdout}")
            return False
    
    except FileNotFoundError:
        logger.error("❌ xrt-smi not found - XRT not installed")
        return False
    except Exception as e:
        logger.error(f"❌ NPU check failed: {e}")
        return False

def check_igpu_hardware():
    """Check iGPU Vulkan hardware availability"""
    logger.info("🎮 Checking iGPU Vulkan hardware...")
    
    try:
        result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ vulkaninfo command failed")
            return False
        
        output = result.stdout.lower()
        if 'amd radeon graphics' in output and ('radv phoenix' in output or 'phoenix' in output):
            logger.info("✅ AMD Radeon 780M iGPU detected")
            return True
        else:
            logger.error("❌ AMD Radeon 780M iGPU not detected")
            logger.error(f"   Vulkan devices: {result.stdout}")
            return False
    
    except FileNotFoundError:
        logger.error("❌ vulkaninfo not found - Vulkan not installed")
        return False
    except Exception as e:
        logger.error(f"❌ iGPU check failed: {e}")
        return False

def check_python_environment():
    """Check Python environment and required packages"""
    logger.info("🐍 Checking Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 10:
        logger.error("❌ Python 3.10+ required")
        return False
    
    # Check required packages
    required_packages = [
        'torch',
        'numpy', 
        'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"   ✅ {package}")
        except ImportError:
            logger.error(f"   ❌ {package} not found")
            missing_packages.append(package)
    
    # Check XRT Python bindings
    try:
        import xrt
        logger.info("   ✅ XRT Python bindings")
    except ImportError:
        logger.warning("   ⚠️ XRT Python bindings not available (will use fallback)")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        return False
    
    logger.info("✅ Python environment OK")
    return True

def check_compiled_kernels():
    """Check NPU kernel binaries are compiled"""
    logger.info("🔧 Checking compiled NPU kernels...")
    
    kernel_binaries = [
        "npu_binaries/gemma3_q_projection.npu_binary",
        "npu_binaries/gemma3_k_projection.npu_binary",
        "npu_binaries/gemma3_v_projection.npu_binary"
    ]
    
    missing_kernels = []
    
    for kernel in kernel_binaries:
        if Path(kernel).exists():
            size = Path(kernel).stat().st_size
            logger.info(f"   ✅ {kernel}: {size} bytes")
        else:
            logger.error(f"   ❌ {kernel}: Not found")
            missing_kernels.append(kernel)
    
    if missing_kernels:
        logger.error("❌ Missing kernel binaries - run kernel compilation first")
        logger.info("   Run: python compile_npu_kernels.py")
        return False
    
    logger.info("✅ All kernel binaries present")
    return True

def check_mlir_aie_environment():
    """Check MLIR-AIE2 environment"""
    logger.info("🔧 Checking MLIR-AIE2 environment...")
    
    # Check for MLIR-AIE2 source
    mlir_paths = [
        "mlir-aie2-src",
        "/home/ucadmin/Development/whisper_npu_project/mlir-aie/",
        "/home/ucadmin/mlir-aie2/"
    ]
    
    mlir_found = False
    for path in mlir_paths:
        if Path(path).exists():
            logger.info(f"   ✅ MLIR-AIE2 found at: {path}")
            mlir_found = True
            break
    
    if not mlir_found:
        logger.error("❌ MLIR-AIE2 source not found")
        return False
    
    # Check for MLIR-AIE Python bindings
    try:
        import aie
        logger.info("   ✅ MLIR-AIE2 Python bindings")
        return True
    except ImportError:
        logger.warning("   ⚠️ MLIR-AIE2 Python bindings not available")
        logger.info("   Note: Will attempt to use pre-built wheels from ironenv")
        return True  # Not critical, can use fallback

def check_test_data():
    """Check test data availability"""
    logger.info("📊 Checking test data...")
    
    # Check if setup script exists
    if not Path("setup_real_model_test.py").exists():
        logger.error("❌ setup_real_model_test.py not found")
        return False
    
    # Check if test data directories exist (will be created if needed)
    logger.info("   ✅ Test data setup script available")
    
    if Path("real_test_weights").exists():
        logger.info("   ✅ Test weights directory exists")
    else:
        logger.info("   📦 Test weights will be created on first run")
    
    if Path("real_test_inputs").exists():
        logger.info("   ✅ Test inputs directory exists")
    else:
        logger.info("   📦 Test inputs will be created on first run")
    
    return True

def run_full_verification():
    """Run complete hardware and software verification"""
    logger.info("🚀 Running Complete Hardware & Software Verification")
    logger.info("=" * 60)
    
    checks = [
        ("NPU Hardware", check_npu_hardware),
        ("iGPU Hardware", check_igpu_hardware),
        ("Python Environment", check_python_environment),
        ("Compiled Kernels", check_compiled_kernels),
        ("MLIR-AIE2 Environment", check_mlir_aie_environment),
        ("Test Data", check_test_data)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 {check_name}:")
        if check_func():
            passed += 1
            logger.info(f"✅ {check_name}: PASS")
        else:
            logger.error(f"❌ {check_name}: FAIL")
    
    logger.info("=" * 60)
    logger.info(f"📊 Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 ALL CHECKS PASSED - Ready for real NPU testing!")
        logger.info("\nTo run the real performance test:")
        logger.info("   python real_npu_performance_test.py")
        return True
    else:
        logger.error("❌ Some checks failed - fix issues before testing")
        logger.info("\nRequired actions:")
        if passed < total:
            logger.info("   1. Fix failed hardware/software checks")
            logger.info("   2. Ensure XRT and Vulkan are properly installed")
            logger.info("   3. Compile NPU kernels if missing")
            logger.info("   4. Re-run verification")
        return False

if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)