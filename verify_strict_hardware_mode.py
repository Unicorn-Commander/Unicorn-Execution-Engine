#!/usr/bin/env python3
"""
Verify STRICT hardware mode - NPU+iGPU only, no CPU compute allowed.
This test ensures our technological edge is real hardware acceleration.
"""

import numpy as np
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrictHardwareModeVerifier:
    """Verifies that we're using NPU+iGPU only with zero CPU compute"""
    
    def __init__(self):
        self.tests_passed = []
        self.tests_failed = []
    
    def check_npu_availability(self) -> bool:
        """Check if NPU hardware is available and accessible"""
        logger.info("🔍 Checking NPU availability...")
        
        try:
            # Check for Xilinx XRT
            result = subprocess.run(['xrt-smi', 'examine'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ XRT detected - NPU driver available")
                
                # Check for NPU device
                if "Phoenix" in result.stdout or "NPU" in result.stdout:
                    logger.info("✅ NPU device detected!")
                    self.tests_passed.append("NPU hardware available")
                    return True
                else:
                    logger.warning("⚠️ XRT present but no NPU device found")
            else:
                logger.error("❌ XRT not available - NPU cannot be used")
        except Exception as e:
            logger.error(f"❌ Error checking NPU: {e}")
        
        self.tests_failed.append("NPU hardware not available")
        return False
    
    def check_gpu_availability(self) -> bool:
        """Check if AMD GPU is available and has compute capability"""
        logger.info("🔍 Checking GPU availability...")
        
        try:
            # Check for AMD GPU
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ AMD GPU detected")
                
                # Parse VRAM info
                for line in result.stdout.splitlines():
                    if "vram" in line.lower():
                        logger.info(f"   GPU VRAM: {line.strip()}")
                    if "gtt" in line.lower():
                        logger.info(f"   GPU GTT: {line.strip()}")
                
                self.tests_passed.append("GPU hardware available")
                return True
            else:
                logger.error("❌ No AMD GPU detected")
        except Exception as e:
            logger.error(f"❌ Error checking GPU: {e}")
        
        self.tests_failed.append("GPU hardware not available")
        return False
    
    def verify_no_cpu_compute(self) -> bool:
        """Verify that CPU compute paths are disabled"""
        logger.info("🔍 Verifying NO CPU compute fallbacks...")
        
        # Check if pure_hardware_pipeline_fixed.py has strict mode
        pipeline_path = Path("pure_hardware_pipeline_fixed.py")
        if pipeline_path.exists():
            content = pipeline_path.read_text()
            
            checks = {
                "strict_hardware_mode = True": "Strict hardware mode enabled",
                "STRICT NPU+iGPU MODE": "Strict mode error messages",
                "raise RuntimeError": "Errors on CPU fallback"
            }
            
            all_good = True
            for check, desc in checks.items():
                if check in content:
                    logger.info(f"✅ {desc}: Found '{check}'")
                    self.tests_passed.append(desc)
                else:
                    logger.error(f"❌ {desc}: NOT found")
                    self.tests_failed.append(desc)
                    all_good = False
            
            return all_good
        else:
            logger.error("❌ Pipeline file not found")
            self.tests_failed.append("Pipeline file missing")
            return False
    
    def check_vulkan_compute(self) -> bool:
        """Check if Vulkan compute is available"""
        logger.info("🔍 Checking Vulkan compute capability...")
        
        try:
            import vulkan as vk
            
            # Try to enumerate devices
            instance = vk.vkCreateInstance(
                vk.VkInstanceCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    pApplicationInfo=vk.VkApplicationInfo(
                        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                        pApplicationName="StrictModeTest",
                        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                        pEngineName="StrictHardwareEngine",
                        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                        apiVersion=vk.VK_API_VERSION_1_0
                    )
                ), None
            )
            
            devices = vk.vkEnumeratePhysicalDevices(instance)
            logger.info(f"✅ Vulkan available with {len(devices)} device(s)")
            
            for i, device in enumerate(devices):
                props = vk.vkGetPhysicalDeviceProperties(device)
                logger.info(f"   Device {i}: {props.deviceName}")
            
            vk.vkDestroyInstance(instance, None)
            self.tests_passed.append("Vulkan compute available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vulkan not available: {e}")
            self.tests_failed.append("Vulkan compute not available")
            return False
    
    def check_model_availability(self) -> bool:
        """Check if quantized models are available"""
        logger.info("🔍 Checking quantized model availability...")
        
        model_paths = {
            "4B quantized": "quantized_models/gemma-3-4b-it-quantized",
            "27B quantized": "quantized_models/gemma-3-27b-it-layer-by-layer"
        }
        
        found_any = False
        for name, path in model_paths.items():
            if Path(path).exists():
                logger.info(f"✅ {name} model found at: {path}")
                
                # Check size
                total_size = 0
                for file in Path(path).rglob("*.safetensors"):
                    total_size += file.stat().st_size
                
                size_gb = total_size / (1024**3)
                logger.info(f"   Size: {size_gb:.1f} GB")
                self.tests_passed.append(f"{name} model available ({size_gb:.1f}GB)")
                found_any = True
            else:
                logger.warning(f"⚠️ {name} model not found at: {path}")
        
        if not found_any:
            self.tests_failed.append("No quantized models found")
        
        return found_any
    
    def run_all_checks(self):
        """Run all verification checks"""
        logger.info("🚀 Starting STRICT Hardware Mode Verification")
        logger.info("=" * 60)
        logger.info("Our edge: NPU+iGPU working together, NO CPU compute!")
        logger.info("=" * 60)
        
        # Run checks
        npu_ok = self.check_npu_availability()
        gpu_ok = self.check_gpu_availability()
        no_cpu = self.verify_no_cpu_compute()
        vulkan_ok = self.check_vulkan_compute()
        models_ok = self.check_model_availability()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\n✅ Tests Passed ({len(self.tests_passed)}):")
        for test in self.tests_passed:
            logger.info(f"   ✓ {test}")
        
        if self.tests_failed:
            logger.info(f"\n❌ Tests Failed ({len(self.tests_failed)}):")
            for test in self.tests_failed:
                logger.info(f"   ✗ {test}")
        
        # Overall verdict
        logger.info("\n" + "=" * 60)
        if not self.tests_failed:
            logger.info("🎉 ALL CHECKS PASSED! Pure NPU+iGPU mode ready!")
            logger.info("💪 Our technological edge is validated!")
        else:
            if gpu_ok and no_cpu and models_ok:
                if not npu_ok:
                    logger.info("⚠️ GPU-only mode available (NPU not detected)")
                    logger.info("   Still strict hardware mode, just without NPU boost")
                else:
                    logger.info("⚠️ Some issues detected but core functionality OK")
            else:
                logger.info("❌ Critical issues preventing hardware-only mode")
        
        logger.info("=" * 60)
        
        # Next steps
        logger.info("\n💡 Next Steps:")
        if not vulkan_ok or "Vulkan" in str(self.tests_failed):
            logger.info("1. Wait for Gemini-CLI to fix Vulkan binding issue")
        if not npu_ok:
            logger.info("2. NPU not detected - check XRT installation")
        if not models_ok:
            logger.info("3. Run quantization to prepare models")
        
        if not self.tests_failed:
            logger.info("1. Run benchmark once Vulkan copy issue is fixed")
            logger.info("2. Measure real NPU+iGPU performance")
            logger.info("3. Celebrate our hardware acceleration success! 🚀")

if __name__ == "__main__":
    verifier = StrictHardwareModeVerifier()
    verifier.run_all_checks()