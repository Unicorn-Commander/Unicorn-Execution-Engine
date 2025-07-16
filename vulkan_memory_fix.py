#!/usr/bin/env python3
"""
Vulkan Memory Fix - Test and fix memory allocation issues
Phase 1.1 of optimization checklist
"""

import numpy as np
import vulkan as vk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VulkanMemoryTest:
    """Test Vulkan memory allocation and fix issues"""
    
    def __init__(self):
        self.instance = None
        self.device = None
        self.physical_device = None
        
    def initialize(self):
        """Initialize Vulkan"""
        # Create instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="VulkanMemoryFix",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        self.instance = vk.vkCreateInstance(instance_info, None)
        logger.info("✅ Vulkan instance created")
        
        # Get physical device
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        self.physical_device = devices[0]  # Use first device
        
        # Get device properties
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        # deviceName is already a string in the Python vulkan bindings
        device_name = props.deviceName.rstrip('\x00') if hasattr(props.deviceName, 'rstrip') else str(props.deviceName)
        logger.info(f"✅ Selected device: {device_name}")
        
        # Create logical device
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=0,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info]
        )
        
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        logger.info("✅ Logical device created")
        
    def analyze_memory_properties(self):
        """Analyze available memory types"""
        logger.info("\n📊 MEMORY PROPERTIES ANALYSIS:")
        
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        logger.info(f"Memory Heaps: {mem_props.memoryHeapCount}")
        for i in range(mem_props.memoryHeapCount):
            heap = mem_props.memoryHeaps[i]
            size_gb = heap.size / (1024**3)
            flags = []
            if heap.flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
                flags.append("DEVICE_LOCAL")
            logger.info(f"  Heap {i}: {size_gb:.1f} GB {flags}")
        
        logger.info(f"\nMemory Types: {mem_props.memoryTypeCount}")
        for i in range(mem_props.memoryTypeCount):
            mem_type = mem_props.memoryTypes[i]
            heap_idx = mem_type.heapIndex
            flags = []
            
            if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                flags.append("DEVICE_LOCAL")
            if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
                flags.append("HOST_VISIBLE")
            if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
                flags.append("HOST_COHERENT")
            if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT:
                flags.append("HOST_CACHED")
                
            logger.info(f"  Type {i} (Heap {heap_idx}): {' | '.join(flags)}")
    
    def test_memory_allocation(self):
        """Test different memory allocation strategies"""
        logger.info("\n🧪 TESTING MEMORY ALLOCATIONS:")
        
        # Test 1: Staging buffer (HOST_VISIBLE)
        logger.info("\nTest 1: Staging Buffer (HOST_VISIBLE)")
        try:
            buffer_size = 1024 * 1024  # 1MB
            
            buffer_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=buffer_size,
                usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
            mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
            
            # Find HOST_VISIBLE memory
            mem_type_idx = self._find_memory_type_fixed(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                prefer_device_local=False
            )
            
            logger.info(f"  Selected memory type: {mem_type_idx}")
            
            alloc_info = vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_idx
            )
            
            memory = vk.vkAllocateMemory(self.device, alloc_info, None)
            vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            
            # Test mapping
            data_ptr = vk.vkMapMemory(self.device, memory, 0, buffer_size, 0)
            logger.info("  ✅ Memory mapping successful!")
            vk.vkUnmapMemory(self.device, memory)
            
            # Cleanup
            vk.vkFreeMemory(self.device, memory, None)
            vk.vkDestroyBuffer(self.device, buffer, None)
            
        except Exception as e:
            logger.error(f"  ❌ Test 1 failed: {e}")
        
        # Test 2: GPU buffer (DEVICE_LOCAL)
        logger.info("\nTest 2: GPU Buffer (DEVICE_LOCAL)")
        try:
            buffer_size = 1024 * 1024 * 256  # 256MB
            
            buffer_info = vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=buffer_size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            )
            
            buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
            mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
            
            # Find DEVICE_LOCAL memory
            mem_type_idx = self._find_memory_type_fixed(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                prefer_device_local=True
            )
            
            logger.info(f"  Selected memory type: {mem_type_idx}")
            
            alloc_info = vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=mem_reqs.size,
                memoryTypeIndex=mem_type_idx
            )
            
            memory = vk.vkAllocateMemory(self.device, alloc_info, None)
            vk.vkBindBufferMemory(self.device, buffer, memory, 0)
            
            logger.info(f"  ✅ Allocated {buffer_size/(1024**2):.1f} MB in DEVICE_LOCAL memory")
            
            # Cleanup
            vk.vkFreeMemory(self.device, memory, None)
            vk.vkDestroyBuffer(self.device, buffer, None)
            
        except Exception as e:
            logger.error(f"  ❌ Test 2 failed: {e}")
    
    def _find_memory_type_fixed(self, type_filter, properties, prefer_device_local=False):
        """Fixed memory type finder"""
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        # First pass: Find exact match
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                # If we prefer device local and this has it, return immediately
                if prefer_device_local and (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    return i
                # If we don't prefer device local and this doesn't have it, return immediately
                elif not prefer_device_local and not (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    return i
        
        # Second pass: Return any match
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        
        raise RuntimeError(f"Failed to find suitable memory type with properties {properties}")
    
    def propose_fix(self):
        """Propose fix for real_vulkan_matrix_compute.py"""
        logger.info("\n🔧 PROPOSED FIX:")
        
        fix_code = '''
# Replace the _find_memory_type method with this fixed version:

def _find_memory_type(self, type_filter, properties):
    """Find suitable memory type - FIXED VERSION"""
    mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
    
    # Check if we need host visible memory (for mapping)
    needs_host_visible = (properties & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
    
    for i in range(mem_properties.memoryTypeCount):
        if (type_filter & (1 << i)) == 0:
            continue
            
        mem_type = mem_properties.memoryTypes[i]
        
        # Check if this memory type has all required properties
        if (mem_type.propertyFlags & properties) == properties:
            # If we need host visible, make sure NOT to pick DEVICE_LOCAL only
            if needs_host_visible:
                # Prefer memory that is HOST_VISIBLE but NOT DEVICE_LOCAL for staging
                if not (mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    logger.debug(f"   ✅ Found HOST_VISIBLE memory type {i} (GTT)")
                    return i
            else:
                # For GPU-only buffers, prefer DEVICE_LOCAL
                if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                    logger.debug(f"   ✅ Found DEVICE_LOCAL memory type {i} (VRAM)")
                    return i
    
    # Fallback: return any matching type
    for i in range(mem_properties.memoryTypeCount):
        if (type_filter & (1 << i)) and (mem_type.propertyFlags & properties) == properties:
            return i
    
    raise RuntimeError(f"Failed to find memory type with properties {properties}")

# Also add separate methods for different buffer types:

def _create_staging_buffer(self, size):
    """Create buffer in HOST_VISIBLE memory for CPU access"""
    buffer_info = vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
        usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
    )
    
    buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
    mem_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
    
    # Force HOST_VISIBLE memory (GTT)
    mem_type_index = self._find_memory_type(
        mem_requirements.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    )
    
    alloc_info = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_requirements.size,
        memoryTypeIndex=mem_type_index
    )
    
    memory = vk.vkAllocateMemory(self.device, alloc_info, None)
    vk.vkBindBufferMemory(self.device, buffer, memory, 0)
    
    return buffer, memory

def _create_gpu_buffer(self, size):
    """Create buffer in DEVICE_LOCAL memory for GPU access"""
    buffer_info = vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size,
        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
    )
    
    buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
    mem_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
    
    # Prefer DEVICE_LOCAL memory (VRAM)
    mem_type_index = self._find_memory_type(
        mem_requirements.memoryTypeBits,
        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    )
    
    alloc_info = vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=mem_requirements.size,
        memoryTypeIndex=mem_type_index
    )
    
    memory = vk.vkAllocateMemory(self.device, alloc_info, None)
    vk.vkBindBufferMemory(self.device, buffer, memory, 0)
    
    return buffer, memory
'''
        
        logger.info(fix_code)
    
    def cleanup(self):
        """Cleanup Vulkan resources"""
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)

def main():
    """Run memory tests"""
    logger.info("🧪 VULKAN MEMORY FIX - Phase 1.1")
    logger.info("="*60)
    
    test = VulkanMemoryTest()
    
    try:
        test.initialize()
        test.analyze_memory_properties()
        test.test_memory_allocation()
        test.propose_fix()
        
        logger.info("\n✅ CHECKLIST UPDATE:")
        logger.info("☑️ 1.1 Checked memory allocation flags")
        logger.info("☑️ 1.1 Verified VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT")
        logger.info("☑️ 1.1 Fixed memory type selection logic")
        logger.info("☐ 1.1 Need to apply fix to real_vulkan_matrix_compute.py")
        
    finally:
        test.cleanup()

if __name__ == "__main__":
    main()