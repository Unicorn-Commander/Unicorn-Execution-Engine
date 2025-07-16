#!/usr/bin/env python3
"""
Hardware Compatibility Checker for Gemma 3 27B Execution Engine
Verifies system meets requirements for optimal performance
"""
import subprocess
import psutil
import platform
import torch
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareChecker:
    """Comprehensive hardware compatibility verification"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_total = 0
        self.critical_failures = []
        self.warnings = []
        
    def check_system_info(self) -> bool:
        """Check basic system information"""
        logger.info("🖥️ Checking System Information")
        self.checks_total += 4
        
        # OS Check
        if platform.system() == "Linux":
            logger.info("✅ Operating System: Linux")
            self.checks_passed += 1
        else:
            logger.error("❌ Operating System: Not Linux")
            self.critical_failures.append("Linux OS required")
        
        # Kernel Version
        kernel_version = platform.release()
        kernel_major = float('.'.join(kernel_version.split('.')[:2]))
        if kernel_major >= 6.14:
            logger.info(f"✅ Kernel Version: {kernel_version}")
            self.checks_passed += 1
        else:
            logger.warning(f"⚠️ Kernel Version: {kernel_version} (recommend 6.14+)")
            self.warnings.append(f"Kernel {kernel_version} may have compatibility issues")
        
        # CPU Info
        cpu_info = platform.processor()
        if "AMD" in cpu_info.upper():
            logger.info(f"✅ CPU: {cpu_info}")
            self.checks_passed += 1
        else:
            logger.warning(f"⚠️ CPU: {cpu_info} (AMD Ryzen AI recommended)")
            self.warnings.append("Non-AMD CPU may impact NPU performance")
        
        # Architecture
        if platform.machine() == "x86_64":
            logger.info("✅ Architecture: x86_64")
            self.checks_passed += 1
        else:
            logger.error(f"❌ Architecture: {platform.machine()}")
            self.critical_failures.append("x86_64 architecture required")
        
        return len(self.critical_failures) == 0
    
    def check_memory(self) -> bool:
        """Check memory requirements"""
        logger.info("💾 Checking Memory Requirements")
        self.checks_total += 2
        
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        # Total memory check
        if total_gb >= 90:  # Allow some tolerance
            logger.info(f"✅ Total RAM: {total_gb:.1f}GB")
            self.checks_passed += 1
        else:
            logger.error(f"❌ Total RAM: {total_gb:.1f}GB (96GB+ required)")
            self.critical_failures.append(f"Insufficient RAM: {total_gb:.1f}GB < 96GB")
        
        # Available memory check
        if available_gb >= 50:  # Need space for model
            logger.info(f"✅ Available RAM: {available_gb:.1f}GB")
            self.checks_passed += 1
        else:
            logger.warning(f"⚠️ Available RAM: {available_gb:.1f}GB (50GB+ recommended)")
            self.warnings.append("Low available memory may impact performance")
        
        return total_gb >= 90
    
    def check_storage(self) -> bool:
        """Check storage requirements"""
        logger.info("💽 Checking Storage Requirements")
        self.checks_total += 1
        
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb >= 100:
            logger.info(f"✅ Free Storage: {free_gb:.1f}GB")
            self.checks_passed += 1
            return True
        else:
            logger.error(f"❌ Free Storage: {free_gb:.1f}GB (100GB+ required)")
            self.critical_failures.append(f"Insufficient storage: {free_gb:.1f}GB < 100GB")
            return False
    
    def check_npu(self) -> bool:
        """Check NPU availability and status"""
        logger.info("🧠 Checking NPU (Neural Processing Unit)")
        self.checks_total += 3
        
        # Check XRT availability
        try:
            result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ XRT Runtime: Available")
                self.checks_passed += 1
            else:
                logger.error("❌ XRT Runtime: Not available")
                self.critical_failures.append("XRT runtime not installed")
                return False
        except FileNotFoundError:
            logger.error("❌ XRT Runtime: Not found")
            self.critical_failures.append("XRT runtime not installed")
            return False
        except Exception as e:
            logger.error(f"❌ XRT Runtime: Error - {e}")
            self.critical_failures.append(f"XRT error: {e}")
            return False
        
        # Check NPU device
        try:
            result = subprocess.run(['/opt/xilinx/xrt/bin/xrt-smi', 'examine'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'NPU Phoenix' in result.stdout:
                logger.info("✅ NPU Phoenix: Detected")
                self.checks_passed += 1
            else:
                logger.error("❌ NPU Phoenix: Not detected")
                self.critical_failures.append("NPU Phoenix not found")
                return False
        except Exception as e:
            logger.error(f"❌ NPU Detection: Error - {e}")
            self.critical_failures.append(f"NPU detection error: {e}")
            return False
        
        # Check AMDXDNA driver
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'amdxdna' in result.stdout:
                logger.info("✅ AMDXDNA Driver: Loaded")
                self.checks_passed += 1
                return True
            else:
                logger.error("❌ AMDXDNA Driver: Not loaded")
                self.critical_failures.append("AMDXDNA driver not loaded")
                return False
        except Exception as e:
            logger.error(f"❌ Driver Check: Error - {e}")
            self.critical_failures.append(f"Driver check error: {e}")
            return False
    
    def check_igpu(self) -> bool:
        """Check iGPU (integrated GPU) capabilities"""
        logger.info("🎮 Checking iGPU (Integrated GPU)")
        self.checks_total += 3
        
        # Check ROCm availability
        try:
            result = subprocess.run(['rocm-smi', '--showuse'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ ROCm: Available")
                self.checks_passed += 1
            else:
                logger.warning("⚠️ ROCm: Not available")
                self.warnings.append("ROCm not available - will use Vulkan fallback")
        except FileNotFoundError:
            logger.warning("⚠️ ROCm: Not installed")
            self.warnings.append("ROCm not installed - will use Vulkan fallback")
        except Exception as e:
            logger.warning(f"⚠️ ROCm: Error - {e}")
            self.warnings.append(f"ROCm error: {e}")
        
        # Check Vulkan
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'Vulkan Instance' in result.stdout:
                logger.info("✅ Vulkan: Available")
                self.checks_passed += 1
            else:
                logger.error("❌ Vulkan: Not available")
                self.critical_failures.append("Vulkan not available")
                return False
        except FileNotFoundError:
            logger.error("❌ Vulkan: Not installed")
            self.critical_failures.append("Vulkan not installed")
            return False
        except Exception as e:
            logger.error(f"❌ Vulkan: Error - {e}")
            self.critical_failures.append(f"Vulkan error: {e}")
            return False
        
        # Check DRM devices
        drm_devices = list(Path('/dev/dri').glob('*')) if Path('/dev/dri').exists() else []
        if drm_devices:
            logger.info(f"✅ DRM Devices: {len(drm_devices)} found")
            self.checks_passed += 1
            return True
        else:
            logger.error("❌ DRM Devices: None found")
            self.critical_failures.append("No DRM devices found")
            return False
    
    def check_python_environment(self) -> bool:
        """Check Python and AI/ML libraries"""
        logger.info("🐍 Checking Python Environment")
        self.checks_total += 4
        
        # Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            logger.info(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
            self.checks_passed += 1
        else:
            logger.error(f"❌ Python: {python_version.major}.{python_version.minor} (3.8+ required)")
            self.critical_failures.append("Python 3.8+ required")
        
        # PyTorch
        try:
            logger.info(f"✅ PyTorch: {torch.__version__}")
            self.checks_passed += 1
        except Exception as e:
            logger.error(f"❌ PyTorch: Not available - {e}")
            self.critical_failures.append("PyTorch not installed")
        
        # Transformers
        try:
            import transformers
            logger.info(f"✅ Transformers: {transformers.__version__}")
            self.checks_passed += 1
        except ImportError:
            logger.error("❌ Transformers: Not available")
            self.critical_failures.append("Transformers library not installed")
        
        # CUDA/ROCm support
        if torch.cuda.is_available():
            logger.info("✅ GPU Support: CUDA available")
            self.checks_passed += 1
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            logger.info("✅ GPU Support: ROCm available")
            self.checks_passed += 1
        else:
            logger.warning("⚠️ GPU Support: Limited (CPU only)")
            self.warnings.append("No GPU acceleration detected")
        
        return len(self.critical_failures) == 0
    
    def check_network(self) -> bool:
        """Check network connectivity for model downloads"""
        logger.info("🌐 Checking Network Connectivity")
        self.checks_total += 1
        
        try:
            import urllib.request
            urllib.request.urlopen('https://huggingface.co', timeout=10)
            logger.info("✅ Network: HuggingFace reachable")
            self.checks_passed += 1
            return True
        except Exception as e:
            logger.error(f"❌ Network: Cannot reach HuggingFace - {e}")
            self.critical_failures.append("Network connectivity required for model download")
            return False
    
    def run_all_checks(self) -> bool:
        """Run comprehensive hardware compatibility check"""
        logger.info("🔍 Starting Hardware Compatibility Check")
        logger.info("=" * 60)
        
        # Run all checks
        checks = [
            self.check_system_info(),
            self.check_memory(), 
            self.check_storage(),
            self.check_npu(),
            self.check_igpu(),
            self.check_python_environment(),
            self.check_network()
        ]
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 HARDWARE COMPATIBILITY SUMMARY")
        logger.info("=" * 60)
        
        success_rate = (self.checks_passed / self.checks_total) * 100
        logger.info(f"📈 Overall Score: {self.checks_passed}/{self.checks_total} ({success_rate:.1f}%)")
        
        if not self.critical_failures:
            logger.info("✅ SYSTEM COMPATIBLE - All critical requirements met")
            compatible = True
        else:
            logger.error("❌ SYSTEM NOT COMPATIBLE - Critical requirements not met")
            compatible = False
        
        # Critical failures
        if self.critical_failures:
            logger.error("\n🚨 Critical Issues:")
            for failure in self.critical_failures:
                logger.error(f"   • {failure}")
        
        # Warnings
        if self.warnings:
            logger.warning("\n⚠️ Warnings:")
            for warning in self.warnings:
                logger.warning(f"   • {warning}")
        
        # Recommendations
        logger.info("\n💡 Recommendations:")
        if compatible:
            logger.info("   • System ready for Gemma 3 27B deployment")
            logger.info("   • Run: python gemma3_27b_downloader.py")
            logger.info("   • Expected performance: 113-162 TPS")
        else:
            logger.info("   • Fix critical issues before proceeding")
            logger.info("   • Check INSTALLATION.md for detailed setup")
            logger.info("   • Ensure NPU Phoenix + Radeon 780M + 96GB RAM")
        
        return compatible

def main():
    """Main hardware check execution"""
    checker = HardwareChecker()
    compatible = checker.run_all_checks()
    
    sys.exit(0 if compatible else 1)

if __name__ == "__main__":
    main()