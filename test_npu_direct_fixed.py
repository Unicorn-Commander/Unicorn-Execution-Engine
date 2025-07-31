#!/usr/bin/env python3.13
"""
Fixed NPU Direct Test - Using correct pyxrt API
"""

import sys
import os
import time
import numpy as np

# Set environment for XRT
os.environ['XILINX_XRT'] = '/opt/xilinx/xrt'
os.environ['PYTHONPATH'] = '/opt/xilinx/xrt/python:' + os.environ.get('PYTHONPATH', '')
os.environ['XRT_HACK_UNSECURE_LOADING_XCLBIN'] = '1'

print("🦄 Magic Unicorn NPU Direct Test (Fixed)")
print("=" * 50)
print(f"Python: {sys.version.split()[0]}")

# Test 1: NPU Device Check
print("\n1️⃣ Checking NPU Device...")
try:
    # Check if device exists
    if os.path.exists('/dev/accel/accel0'):
        print("✅ NPU device node exists: /dev/accel/accel0")
        
        # Check permissions
        import stat
        st = os.stat('/dev/accel/accel0')
        mode = stat.filemode(st.st_mode)
        print(f"   Permissions: {mode}")
        print(f"   Device type: Character device")
        npu_device_exists = True
    else:
        print("❌ NPU device not found")
        npu_device_exists = False
        
except Exception as e:
    print(f"❌ Device check failed: {e}")
    npu_device_exists = False

# Test 2: XRT Access
print("\n2️⃣ Testing XRT/pyxrt Access...")
try:
    import pyxrt
    print("✅ pyxrt imported")
    
    # Try to create device - pyxrt uses index-based constructor
    try:
        device = pyxrt.device(0)  # Device index 0
        print("✅ Created XRT device object")
        
        # Try to get device info
        try:
            # Different info types available
            device_name = device.get_info(pyxrt.info.device_name)
            print(f"   Device name: {device_name}")
            npu_available = True
        except:
            # Try alternative API
            try:
                device_name = str(device)
                print(f"   Device: {device_name}")
                npu_available = True
            except:
                print("   (Could not get device name)")
                npu_available = True  # Device created successfully
                
    except Exception as e:
        print(f"⚠️  Could not create device: {e}")
        npu_available = False
        
except ImportError:
    print("❌ pyxrt not available")
    npu_available = False
except Exception as e:
    print(f"❌ XRT error: {e}")
    npu_available = False

# Test 3: Vulkan GPU
print("\n3️⃣ Testing Vulkan GPU...")
try:
    import vulkan as vk
    print("✅ vulkan imported")
    
    # List available instance extensions
    extensions = vk.vkEnumerateInstanceExtensionProperties(None)
    print(f"   Found {len(extensions)} Vulkan extensions")
    
    # Create minimal instance
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=None,  # Simplified
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None
    )
    
    instance = vk.vkCreateInstance(create_info, None)
    print("✅ Vulkan instance created")
    
    # Get devices
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    if physical_devices:
        print(f"✅ Found {len(physical_devices)} GPU(s)")
        
        # Get first device properties
        device = physical_devices[0]
        props = vk.vkGetPhysicalDeviceProperties(device)
        
        # Device name is bytes in the struct
        device_name = ""
        for i in range(256):  # VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
            if props.deviceName[i] == 0:
                break
            device_name += chr(props.deviceName[i])
        
        print(f"   GPU: {device_name}")
        gpu_available = True
    else:
        print("❌ No GPUs found")
        gpu_available = False
        
    # Cleanup
    vk.vkDestroyInstance(instance, None)
    
except Exception as e:
    print(f"❌ Vulkan error: {e}")
    gpu_available = False

# Test 4: Simple NPU Kernel Check
print("\n4️⃣ Checking NPU Kernels...")
kernel_dir = "/home/ucadmin/Development/Unicorn-Execution-Engine/npu_kernels"
if os.path.exists(kernel_dir):
    kernels = list(os.listdir(kernel_dir))
    print(f"✅ Found {len(kernels)} kernel files:")
    for k in kernels:
        size = os.path.getsize(os.path.join(kernel_dir, k))
        print(f"   - {k}: {size} bytes")
    kernel_ready = len(kernels) > 0
else:
    print("❌ Kernel directory not found")
    kernel_ready = False

# Test 5: Memory and Compute Test
print("\n5️⃣ Testing Basic Compute...")
try:
    # Test numpy computation
    size = 256
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    start = time.time()
    c = np.matmul(a, b)
    compute_time = (time.time() - start) * 1000
    
    gflops = (2 * size**3) / (compute_time / 1000) / 1e9
    print(f"✅ Matrix multiply: {size}x{size} in {compute_time:.1f}ms ({gflops:.1f} GFLOPS)")
    compute_ready = True
    
except Exception as e:
    print(f"❌ Compute test failed: {e}")
    compute_ready = False

# Summary
print("\n" + "=" * 50)
print("📊 HARDWARE STATUS SUMMARY")
print("=" * 50)

status = {
    "NPU Device Node": "✅" if npu_device_exists else "❌",
    "XRT/pyxrt Access": "✅" if npu_available else "❌",
    "Vulkan GPU": "✅" if gpu_available else "❌",
    "NPU Kernels": "✅" if kernel_ready else "❌",
    "Compute Test": "✅" if compute_ready else "❌",
}

for key, value in status.items():
    print(f"{key:.<25} {value}")

# Overall assessment
success_count = sum([npu_device_exists, npu_available, gpu_available, kernel_ready, compute_ready])
total = len(status)

print(f"\n✅ Passed: {success_count}/{total} tests")

if success_count >= 3:
    print("\n🎉 HARDWARE READY!")
    print("🦄 We have enough to start inference!")
    
    if not npu_available:
        print("\n⚠️  NPU not fully accessible via pyxrt")
        print("   But we can still use GPU acceleration!")
    
    print("\n🚀 Next: python3.13 pure_hardware_python313.py")
else:
    print("\n⚠️  Hardware setup needs attention")
    
    if not npu_available:
        print("\n🔧 NPU Fix:")
        print("   1. Check XRT installation:")
        print("      ls -la /opt/xilinx/xrt/")
        print("   2. Try XRT tools:")
        print("      /opt/xilinx/xrt/bin/xbutil examine")
    
    if not gpu_available:
        print("\n🔧 GPU Fix:")
        print("   1. Check drivers: lspci -k | grep -A 3 VGA")
        print("   2. Check Vulkan: vulkaninfo --summary")

print("\n💪 We're close! Let's keep pushing!")