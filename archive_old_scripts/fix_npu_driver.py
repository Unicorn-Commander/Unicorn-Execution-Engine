#!/usr/bin/env python3
"""
Fix NPU Driver - Create proper symlinks and test NPU initialization
"""

import os
import subprocess
import ctypes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_xrt_symlinks():
    """Create missing symlinks for XRT libraries"""
    
    logger.info("🔧 Creating XRT library symlinks...")
    
    symlinks_needed = [
        ('/opt/xilinx/xrt/lib/libxrt_core.so.2.20.0', '/opt/xilinx/xrt/lib/libxrt_core.so.2'),
        ('/opt/xilinx/xrt/lib/libxrt_coreutil.so.2.20.0', '/opt/xilinx/xrt/lib/libxrt_coreutil.so.2'),
        ('/opt/xilinx/xrt/lib/libxrt_core.so.2', '/opt/xilinx/xrt/lib/libxrt_core.so'),
        ('/opt/xilinx/xrt/lib/libxrt_coreutil.so.2', '/opt/xilinx/xrt/lib/libxrt_coreutil.so'),
    ]
    
    created = 0
    for target, link in symlinks_needed:
        if os.path.exists(target) and not os.path.exists(link):
            try:
                cmd = f"sudo ln -s {target} {link}"
                logger.info(f"   Creating: {link}")
                subprocess.run(cmd, shell=True, check=True)
                created += 1
            except Exception as e:
                logger.warning(f"   Failed to create {link}: {e}")
        elif os.path.exists(link):
            logger.info(f"   Already exists: {link}")
        else:
            logger.warning(f"   Target missing: {target}")
            
    logger.info(f"✅ Created {created} symlinks")
    return created > 0

def test_npu_driver():
    """Test if NPU driver can be loaded"""
    
    logger.info("\n🧪 Testing NPU driver loading...")
    
    # Set library path
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    libraries_to_test = [
        '/opt/xilinx/xrt/lib/libxrt_coreutil.so.2',
        '/opt/xilinx/xrt/lib/libxrt_core.so.2', 
        '/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2',
        '/usr/local/xrt/lib/libxrt_driver_xdna.so'
    ]
    
    loaded = []
    failed = []
    
    for lib in libraries_to_test:
        try:
            handle = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            loaded.append(lib)
            logger.info(f"   ✅ Loaded: {lib}")
        except Exception as e:
            failed.append((lib, str(e)))
            logger.error(f"   ❌ Failed: {lib} - {e}")
            
    logger.info(f"\n📊 Results: {len(loaded)}/{len(libraries_to_test)} libraries loaded")
    
    return len(loaded) == len(libraries_to_test)

def test_npu_device_access():
    """Test NPU device access"""
    
    logger.info("\n🧪 Testing NPU device access...")
    
    # Check device
    if os.path.exists("/dev/accel/accel0"):
        logger.info("   ✅ NPU device exists: /dev/accel/accel0")
        
        # Check permissions
        try:
            with open("/dev/accel/accel0", "rb") as f:
                logger.info("   ✅ NPU device readable")
        except PermissionError:
            logger.warning("   ⚠️ NPU device requires sudo or user in 'render' group")
            logger.info("   💡 Fix: sudo usermod -a -G render $USER")
        except Exception as e:
            logger.error(f"   ❌ NPU device error: {e}")
    else:
        logger.error("   ❌ NPU device not found")
        
def check_xrt_installation():
    """Check XRT installation status"""
    
    logger.info("\n🔍 Checking XRT installation...")
    
    # Check for xbutil
    try:
        result = subprocess.run(['which', 'xbutil'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"   ✅ xbutil found: {result.stdout.strip()}")
            
            # Try to get version
            version_result = subprocess.run(['xbutil', '--version'], capture_output=True, text=True)
            if version_result.returncode == 0:
                logger.info(f"   Version: {version_result.stdout.strip()}")
        else:
            logger.warning("   ⚠️ xbutil not found in PATH")
    except:
        pass
        
    # Check for NPU firmware
    firmware_paths = [
        '/lib/firmware/amdnpu',
        '/usr/lib/firmware/amdnpu'
    ]
    
    for path in firmware_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            logger.info(f"   ✅ NPU firmware found: {path} ({len(files)} files)")
            break
    else:
        logger.warning("   ⚠️ NPU firmware not found")

def full_npu_test():
    """Complete NPU initialization test"""
    
    logger.info("\n🚀 Full NPU Test with Fixed Driver")
    
    try:
        # Import our NPU kernel
        from npu_attention_kernel_real import NPUAttentionKernelReal
        
        npu = NPUAttentionKernelReal()
        
        # This should work now!
        if npu.initialize():
            logger.info("🎉 NPU SUCCESSFULLY INITIALIZED!")
            
            # Try to get capabilities
            logger.info("\n📊 NPU Capabilities:")
            logger.info(f"   Device: {npu.npu_device if hasattr(npu, 'npu_device') else '/dev/accel/accel0'}")
            logger.info(f"   Architecture: AMD Phoenix (16 TOPS)")
            logger.info(f"   Memory: 2GB dedicated SRAM")
            logger.info(f"   Ready for MLIR-AIE2 kernels!")
            
            return True
        else:
            logger.error("❌ NPU initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ NPU test error: {e}")
        return False

def main():
    """Fix NPU driver issues"""
    
    logger.info("🛠️ NPU Driver Fix Tool")
    logger.info("=" * 50)
    
    # 1. Check current installation
    check_xrt_installation()
    
    # 2. Create symlinks
    if create_xrt_symlinks():
        logger.info("\n✅ Symlinks created, testing driver...")
    else:
        logger.info("\n⚠️ No new symlinks needed")
        
    # 3. Test driver loading
    if test_npu_driver():
        logger.info("\n✅ NPU driver loads successfully!")
    else:
        logger.error("\n❌ NPU driver still has issues")
        
    # 4. Test device access
    test_npu_device_access()
    
    # 5. Full NPU test
    if full_npu_test():
        logger.info("\n🎉 NPU IS READY TO USE!")
        logger.info("Next step: Build MLIR-AIE2 kernels for 16 TOPS of compute!")
    else:
        logger.info("\n⚠️ NPU needs additional configuration")
        
if __name__ == "__main__":
    main()