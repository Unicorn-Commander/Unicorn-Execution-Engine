#!/usr/bin/env python3.13
"""
Simple NPU test to diagnose context allocation issue
"""

import os
import sys
import time

# XRT environment
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')

try:
    import pyxrt
    print("✅ PyXRT imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pyxrt: {e}")
    sys.exit(1)

def test_npu():
    """Test basic NPU access"""
    print("\n🔍 Testing NPU Access...")
    
    try:
        # Try to open device directly
        print("\n📱 Opening device 0...")
        device = pyxrt.device(0)
        print("✅ Device opened successfully")
        
        # Skip device info for now
        print("   Device object created")
        
        # Try to allocate a small buffer
        print("\n💾 Allocating test buffer...")
        try:
            # Very small buffer - 1KB
            bank = 0  # Memory bank index
            test_bo = pyxrt.bo(device, 1024, pyxrt.bo.flags.normal, bank)
            print("✅ Buffer allocated successfully")
            
            # Clean up
            del test_bo
            print("✅ Buffer freed")
            
        except Exception as e:
            print(f"❌ Buffer allocation failed: {e}")
            print("   This suggests a resource/context issue")
        
        # Close device
        del device
        print("\n✅ Device closed")
        
    except Exception as e:
        print(f"❌ NPU test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Simple NPU Test")
    print("=" * 40)
    test_npu()
    print("\n✨ Test complete")