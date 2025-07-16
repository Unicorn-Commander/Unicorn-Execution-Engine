#!/usr/bin/env python3
"""
Fixed Vulkan Matrix Computation - Actually uses GPU VRAM
Direct GPU acceleration with proper memory allocation
"""

import numpy as np
import vulkan as vk
import logging
import time
from pathlib import Path
import ctypes
import struct

logger = logging.getLogger(__name__)

class VulkanMatrixComputeFixed:
    """Fixed Vulkan compute that actually uses GPU VRAM"""
    
    def __init__(self):
        self.instance = None
        self.device = None
        self.physical_device = None
        self.compute_queue = None
        self.compute_queue_family = 0
        self.command_pool = None
        self.descriptor_pool = None
        self.descriptor_set_layout = None
        self.compute_pipeline = None
        self.pipeline_layout = None
        self.initialized = False
        
        # Track GPU memory allocations
        self.gpu_allocations = []
        self.total_vram_mb = 0
        self.total_gtt_mb = 0
        
    def initialize(self):
        """Initialize Vulkan compute with proper GPU memory"""
        logger.info("🎮 Initializing Fixed Vulkan Matrix Compute (with VRAM)...")
        
        try:
            self._create_instance()
            self._select_device()
            self._create_device()
            self._create_command_pool()
            self._create_compute_pipeline()
            self._create_descriptor_pool()
            
            self.initialized = True
            logger.info("✅ Fixed Vulkan Matrix Compute initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vulkan initialization failed: {e}")
            return False
    
    def _create_instance(self):
        """Create Vulkan instance"""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='VulkanMatrixComputeFixed',
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
        logger.info("   ✅ Vulkan instance created")
    
    def _select_device(self):
        """Select AMD GPU"""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        
        for device in devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            device_name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
            
            if 'AMD' in device_name or 'Radeon' in device_name:
                self.physical_device = device
                logger.info(f"   ✅ Selected device: {device_name}")
                
                # Log memory types
                mem_props = vk.vkGetPhysicalDeviceMemoryProperties(device)
                for i in range(mem_props.memoryTypeCount):
                    mem_type = mem_props.memoryTypes[i]
                    heap_idx = mem_type.heapIndex
                    heap_size_gb = mem_props.memoryHeaps[heap_idx].size / (1024**3)
                    
                    flags = []
                    if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                        flags.append("DEVICE_LOCAL")
                    if mem_type.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
                        flags.append("HOST_VISIBLE")
                    
                    logger.debug(f"   Memory Type {i}: Heap {heap_idx} ({heap_size_gb:.1f}GB) - {', '.join(flags)}")
                break
    
    def _create_device(self):
        """Create logical device"""
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        
        for i, props in enumerate(queue_families):
            if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
                logger.info(f"   ✅ Compute queue family: {i}")
                break
        
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info]
        )
        
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.compute_queue = vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0)
        logger.info("   ✅ Logical device and compute queue created")
    
    def _create_command_pool(self):
        """Create command pool"""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family
        )
        
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        logger.info("   ✅ Command pool created")
    
    def _create_compute_pipeline(self):
        """Create compute shader and pipeline"""
        # Load SPIR-V shader
        shader_path = Path(__file__).parent / "transformer_optimized.spv"
        if not shader_path.exists():
            logger.warning(f"   ⚠️ Shader not found at {shader_path}")
            return
        
        with open(shader_path, 'rb') as f:
            shader_code = f.read()
        
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )
        
        shader_module = vk.vkCreateShaderModule(self.device, shader_module_info, None)
        logger.info("   ✅ Compute shader module created")
        
        # Create descriptor set layout
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            )
        ]
        
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )
        
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)
        
        # Create pipeline layout
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )
        
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)
        
        # Create compute pipeline
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName='main'
        )
        
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout
        )
        
        self.compute_pipeline = vk.vkCreateComputePipelines(
            self.device, None, 1, [pipeline_info], None
        )[0]
        
        logger.info("   ✅ Compute pipeline created")
    
    def _create_descriptor_pool(self):
        """Create descriptor pool"""
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=100
        )
        
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=30,
            poolSizeCount=1,
            pPoolSizes=[pool_size]
        )
        
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        logger.info("   ✅ Descriptor pool created")
    
    def _find_memory_type(self, type_filter, properties):
        """Find suitable memory type"""
        mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        for i in range(mem_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        
        raise RuntimeError(f"Failed to find suitable memory type for properties {properties}")
    
    def _create_buffer_vram(self, size_bytes):
        """Create buffer in VRAM (DEVICE_LOCAL)"""
        # Create buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        
        # Get memory requirements
        mem_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate DEVICE_LOCAL memory (VRAM!)
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self._find_memory_type(
                mem_requirements.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
        )
        
        buffer_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, buffer_memory, 0)
        
        self.total_vram_mb += size_bytes / (1024**2)
        self.gpu_allocations.append((buffer, buffer_memory, 'VRAM'))
        
        logger.debug(f"   Allocated {size_bytes/(1024**2):.1f}MB to VRAM")
        return buffer, buffer_memory
    
    def _create_buffer_gtt(self, size_bytes):
        """Create buffer in GTT (HOST_VISIBLE + DEVICE_LOCAL)"""
        # Create buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        
        # Get memory requirements
        mem_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Try to find DEVICE_LOCAL + HOST_VISIBLE memory (GTT)
        try:
            memory_type_index = self._find_memory_type(
                mem_requirements.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            )
            memory_location = 'GTT'
        except:
            # Fallback to HOST_VISIBLE only
            memory_type_index = self._find_memory_type(
                mem_requirements.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            memory_location = 'RAM'
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=memory_type_index
        )
        
        buffer_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, buffer_memory, 0)
        
        if memory_location == 'GTT':
            self.total_gtt_mb += size_bytes / (1024**2)
        
        self.gpu_allocations.append((buffer, buffer_memory, memory_location))
        
        logger.debug(f"   Allocated {size_bytes/(1024**2):.1f}MB to {memory_location}")
        return buffer, buffer_memory
    
    def transfer_to_gpu(self, data, prefer_vram=True):
        """Transfer numpy array to GPU memory"""
        size_bytes = data.nbytes
        
        # Create staging buffer in GTT/RAM
        staging_buffer, staging_memory = self._create_buffer_gtt(size_bytes)
        
        # Map and copy data to staging buffer
        data_ptr = vk.vkMapMemory(self.device, staging_memory, 0, size_bytes, 0)
        
        # Copy data using Vulkan's FFI
        vk.ffi.memmove(data_ptr, data.tobytes(), size_bytes)
        
        vk.vkUnmapMemory(self.device, staging_memory)
        
        if prefer_vram:
            # Create VRAM buffer
            vram_buffer, vram_memory = self._create_buffer_vram(size_bytes)
            
            # Copy from staging to VRAM
            self._copy_buffer(staging_buffer, vram_buffer, size_bytes)
            
            # Clean up staging buffer
            vk.vkDestroyBuffer(self.device, staging_buffer, None)
            vk.vkFreeMemory(self.device, staging_memory, None)
            
            return vram_buffer, vram_memory
        else:
            # Use staging buffer directly (GTT)
            return staging_buffer, staging_memory
    
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
        
        # Record copy command
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        
        copy_region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vk.vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, [copy_region])
        
        vk.vkEndCommandBuffer(command_buffer)
        
        # Submit and wait
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )
        
        vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], None)
        vk.vkQueueWaitIdle(self.compute_queue)
        
        # Free command buffer
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
    
    def matrix_multiply(self, matrix_a, matrix_b):
        """Matrix multiplication on GPU"""
        if not self.initialized:
            raise RuntimeError("Vulkan not initialized")
        
        # Import the optimized engine
        from vulkan_compute_optimized import VulkanComputeOptimized
        
        # Use singleton pattern for the optimized engine
        if not hasattr(self, '_optimized_engine'):
            self._optimized_engine = VulkanComputeOptimized(max_memory_gb=8.0)
            if not self._optimized_engine.initialize():
                raise RuntimeError("Failed to initialize optimized engine")
        
        # Use optimized engine for actual computation
        return self._optimized_engine.matrix_multiply(matrix_a, matrix_b)
    
    def get_memory_stats(self):
        """Get GPU memory statistics"""
        return {
            'vram_allocated_mb': self.total_vram_mb,
            'gtt_allocated_mb': self.total_gtt_mb,
            'total_allocations': len(self.gpu_allocations)
        }
    
    def cleanup(self):
        """Clean up Vulkan resources"""
        for buffer, memory, location in self.gpu_allocations:
            vk.vkDestroyBuffer(self.device, buffer, None)
            vk.vkFreeMemory(self.device, memory, None)
        
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        if self.compute_pipeline:
            vk.vkDestroyPipeline(self.device, self.compute_pipeline, None)
        if self.pipeline_layout:
            vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        if self.descriptor_set_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        
        logger.info("✅ Vulkan cleanup complete")


# For compatibility, create an alias
VulkanMatrixCompute = VulkanMatrixComputeFixed