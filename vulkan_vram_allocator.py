#!/usr/bin/env python3
"""
Vulkan VRAM Allocator - Actually allocates to GPU VRAM
Fixes the issue where Vulkan was allocating to system RAM
"""

import numpy as np
import vulkan as vk
import logging
import subprocess
import time
from typing import Optional, Tuple, List
import ctypes
import struct

logger = logging.getLogger(__name__)

class VulkanVRAMAllocator:
    """
    Properly allocates memory to GPU VRAM using DEVICE_LOCAL memory
    """
    
    def __init__(self):
        self.instance = None
        self.device = None
        self.physical_device = None
        self.compute_queue = None
        self.command_pool = None
        self.initialized = False
        
        # Track allocations
        self.vram_allocations = []
        self.gtt_allocations = []
        
    def initialize(self):
        """Initialize Vulkan for VRAM allocation"""
        try:
            # Create instance
            app_info = vk.VkApplicationInfo(
                sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName='VulkanVRAMAllocator',
                applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                pEngineName='UnicornEngine',
                engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                apiVersion=vk.VK_API_VERSION_1_0
            )
            
            instance_info = vk.VkInstanceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=app_info
            )
            
            self.instance = vk.vkCreateInstance(instance_info, None)
            
            # Get physical device
            devices = vk.vkEnumeratePhysicalDevices(self.instance)
            
            for device in devices:
                props = vk.vkGetPhysicalDeviceProperties(device)
                device_name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
                
                if 'AMD' in device_name or 'Radeon' in device_name:
                    self.physical_device = device
                    logger.info(f"✅ Found GPU: {device_name}")
                    break
            
            if not self.physical_device:
                raise RuntimeError("No AMD GPU found")
            
            # Log memory properties
            mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
            
            logger.info("📊 GPU Memory Types:")
            for i in range(mem_props.memoryTypeCount):
                mem_type = mem_props.memoryTypes[i]
                heap_idx = mem_type.heapIndex
                heap_size_gb = mem_props.memoryHeaps[heap_idx].size / (1024**3)
                
                flags = []
                if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                    flags.append("DEVICE_LOCAL")
                if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
                    flags.append("HOST_VISIBLE")
                if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
                    flags.append("HOST_COHERENT")
                
                logger.info(f"   Type {i}: Heap {heap_idx} ({heap_size_gb:.1f}GB) - {', '.join(flags)}")
            
            # Find compute queue family
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
            compute_queue_family = None
            
            for i, family in enumerate(queue_families):
                if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    compute_queue_family = i
                    break
            
            if compute_queue_family is None:
                raise RuntimeError("No compute queue family found")
            
            # Create device
            queue_info = vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=compute_queue_family,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            
            device_info = vk.VkDeviceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_info]
            )
            
            self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
            self.compute_queue = vk.vkGetDeviceQueue(self.device, compute_queue_family, 0)
            
            # Create command pool
            pool_info = vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=compute_queue_family
            )
            
            self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
            
            self.initialized = True
            logger.info("✅ Vulkan VRAM allocator initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def allocate_vram(self, size_mb: float) -> Optional[Tuple]:
        """Allocate memory in GPU VRAM (DEVICE_LOCAL)"""
        if not self.initialized:
            return None
        
        size_bytes = int(size_mb * 1024 * 1024)
        
        try:
            # Create buffer
            buffer_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size_bytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
            
            # Get memory requirements
            mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
            
            # Find DEVICE_LOCAL memory type (this is VRAM!)
            mem_type_idx = self._find_memory_type(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            
            # Allocate memory
            alloc_info = vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_idx
            )
            
            memory = vk.vkAllocateMemory(self.device, alloc_info, None)
            
            # Bind buffer to memory
            vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            
            self.vram_allocations.append((buffer, memory, size_mb))
            
            logger.info(f"✅ Allocated {size_mb:.1f}MB to VRAM (DEVICE_LOCAL)")
            return buffer, memory
            
        except Exception as e:
            logger.error(f"❌ VRAM allocation failed: {e}")
            return None
    
    def allocate_gtt(self, size_mb: float) -> Optional[Tuple]:
        """Allocate memory in GTT (HOST_VISIBLE + HOST_COHERENT)"""
        if not self.initialized:
            return None
        
        size_bytes = int(size_mb * 1024 * 1024)
        
        try:
            # Create buffer
            buffer_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size_bytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
            
            # Get memory requirements
            mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
            
            # Find HOST_VISIBLE memory type (this is GTT!)
            mem_type_idx = self._find_memory_type(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            
            # Allocate memory
            alloc_info = vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_idx
            )
            
            memory = vk.vkAllocateMemory(self.device, alloc_info, None)
            
            # Bind buffer to memory
            vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            
            self.gtt_allocations.append((buffer, memory, size_mb))
            
            logger.info(f"✅ Allocated {size_mb:.1f}MB to GTT (HOST_VISIBLE)")
            return buffer, memory
            
        except Exception as e:
            logger.error(f"❌ GTT allocation failed: {e}")
            return None
    
    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        """Find suitable memory type"""
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        
        raise RuntimeError(f"No suitable memory type found for properties {properties}")
    
    def transfer_to_vram(self, data: np.ndarray) -> Optional[Tuple]:
        """Transfer numpy array to VRAM using staging buffer"""
        if not self.initialized:
            return None
        
        size_mb = data.nbytes / (1024**2)
        
        # Step 1: Create staging buffer in GTT
        staging_buffer, staging_memory = self.allocate_gtt(size_mb)
        if not staging_buffer:
            return None
        
        # Step 2: Map staging buffer and copy data
        data_ptr = vk.vkMapMemory(self.device, staging_memory, 0, data.nbytes, 0)
        
        # Copy data to staging buffer
        # Convert pointer to int and use ctypes
        ptr_int = int(data_ptr)
        ctypes.memmove(ptr_int, data.ctypes.data, data.nbytes)
        
        vk.vkUnmapMemory(self.device, staging_memory)
        
        # Step 3: Create VRAM buffer
        vram_buffer, vram_memory = self.allocate_vram(size_mb)
        if not vram_buffer:
            return None
        
        # Step 4: Copy from staging to VRAM (requires command buffer)
        self._copy_buffer(staging_buffer, vram_buffer, data.nbytes)
        
        # Clean up staging buffer
        vk.vkDestroyBuffer(self.device, staging_buffer, None)
        vk.vkFreeMemory(self.device, staging_memory, None)
        
        logger.info(f"✅ Transferred {size_mb:.1f}MB to VRAM")
        return vram_buffer, vram_memory
    
    def _copy_buffer(self, src_buffer, dst_buffer, size):
        """Copy between buffers using command buffer"""
        # Allocate command buffer
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]
        
        # Begin command buffer
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        
        # Copy command
        copy_region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vk.vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, [copy_region])
        
        vk.vkEndCommandBuffer(command_buffer)
        
        # Submit command buffer
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )
        
        vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], None)
        vk.vkQueueWaitIdle(self.compute_queue)
        
        # Free command buffer
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
    
    def get_memory_stats(self):
        """Get GPU memory statistics"""
        vram_total_mb = sum(size for _, _, size in self.vram_allocations)
        gtt_total_mb = sum(size for _, _, size in self.gtt_allocations)
        
        return {
            'vram_allocated_mb': vram_total_mb,
            'gtt_allocated_mb': gtt_total_mb,
            'vram_allocations': len(self.vram_allocations),
            'gtt_allocations': len(self.gtt_allocations)
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Free VRAM allocations
        for buffer, memory, _ in self.vram_allocations:
            vk.vkDestroyBuffer(self.device, buffer, None)
            vk.vkFreeMemory(self.device, memory, None)
        
        # Free GTT allocations
        for buffer, memory, _ in self.gtt_allocations:
            vk.vkDestroyBuffer(self.device, buffer, None)
            vk.vkFreeMemory(self.device, memory, None)
        
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        
        logger.info("✅ Cleanup complete")


def test_vram_allocation():
    """Test actual VRAM allocation"""
    print("🧪 Testing Vulkan VRAM Allocation")
    print("=" * 60)
    
    allocator = VulkanVRAMAllocator()
    
    if not allocator.initialize():
        print("❌ Failed to initialize")
        return
    
    # Get initial GPU memory
    print("\n📊 Initial GPU Memory:")
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line:
            print(f"   {line.strip()}")
    
    # Test VRAM allocation
    print("\n🔧 Allocating 1GB to VRAM...")
    buffer1, memory1 = allocator.allocate_vram(1024)
    
    time.sleep(1)  # Give system time to update
    
    # Check GPU memory after allocation
    print("\n📊 After VRAM allocation:")
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line:
            print(f"   {line.strip()}")
    
    # Test data transfer
    print("\n🔧 Transferring 512MB numpy array to VRAM...")
    test_data = np.random.randn(128, 1024, 1024).astype(np.float32)  # 512MB
    buffer2, memory2 = allocator.transfer_to_vram(test_data)
    
    time.sleep(1)
    
    # Final memory check
    print("\n📊 Final GPU Memory:")
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Used Memory' in line:
            print(f"   {line.strip()}")
    
    # Show allocator stats
    stats = allocator.get_memory_stats()
    print(f"\n📊 Allocator Stats:")
    print(f"   VRAM allocated: {stats['vram_allocated_mb']:.1f}MB")
    print(f"   GTT allocated: {stats['gtt_allocated_mb']:.1f}MB")
    
    # Cleanup
    allocator.cleanup()
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vram_allocation()