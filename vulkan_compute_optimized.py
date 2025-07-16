#!/usr/bin/env python3
"""
Optimized Vulkan Compute Engine
Actually dispatches shaders and manages memory efficiently
"""

import numpy as np
import vulkan as vk
import logging
import time
from pathlib import Path
import struct
from collections import OrderedDict

logger = logging.getLogger(__name__)

class VulkanComputeOptimized:
    """Optimized Vulkan compute with proper shader dispatch and memory management"""
    
    def __init__(self, max_memory_gb=8.0):
        self.instance = None
        self.device = None
        self.physical_device = None
        self.compute_queue = None
        self.compute_queue_family = 0
        self.command_pool = None
        self.descriptor_pool = None
        self.initialized = False
        
        # Memory management
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.current_memory_usage = 0
        
        # Buffer cache with LRU eviction
        self.buffer_cache = OrderedDict()
        self.persistent_buffers = {}  # Weights that stay in VRAM
        
        # Shader pipeline
        self.compute_pipeline = None
        self.pipeline_layout = None
        self.descriptor_set_layout = None
        
        # Command buffer for reuse
        self.command_buffer = None
        
    def initialize(self):
        """Initialize Vulkan with compute support"""
        logger.info("🚀 Initializing Optimized Vulkan Compute Engine")
        
        try:
            self._create_instance()
            self._select_device()
            self._create_device()
            self._create_command_pool()
            self._create_descriptor_pool()
            self._create_compute_pipeline()
            self._allocate_command_buffer()
            
            self.initialized = True
            logger.info("✅ Vulkan Compute Engine initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _create_instance(self):
        """Create Vulkan instance"""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='VulkanComputeOptimized',
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName='PureHardwareOptimized',
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        
        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )
        
        self.instance = vk.vkCreateInstance(instance_info, None)
        
    def _select_device(self):
        """Select AMD GPU"""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        
        for device in devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            device_name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
            
            if 'AMD' in device_name or 'Radeon' in device_name:
                self.physical_device = device
                logger.info(f"   ✅ Selected device: {device_name}")
                
                # Get memory properties
                self.mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(device)
                return
    
    def _create_device(self):
        """Create logical device"""
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        
        for i, props in enumerate(queue_families):
            if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
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
    
    def _create_command_pool(self):
        """Create command pool"""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.compute_queue_family
        )
        
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
    
    def _create_descriptor_pool(self):
        """Create descriptor pool for shader resources"""
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=3000  # Much larger for heavy workloads
        )
        
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  # Allow freeing sets
            maxSets=1000,  # Much larger for heavy workloads
            poolSizeCount=1,
            pPoolSizes=[pool_size]
        )
        
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
    
    def _create_compute_pipeline(self):
        """Create compute pipeline with shader"""
        # Load shader
        shader_path = Path(__file__).parent / "transformer_optimized.spv"
        if not shader_path.exists():
            # Try batched shader
            shader_path = Path(__file__).parent / "batched_gemm.spv"
        
        if not shader_path.exists():
            logger.warning("⚠️ No compiled shader found")
            return
        
        with open(shader_path, 'rb') as f:
            shader_code = f.read()
        
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )
        
        shader_module = vk.vkCreateShaderModule(self.device, shader_module_info, None)
        
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
        
        # Push constants for matrix dimensions
        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=5 * 4  # M, N, K, tile_size, flags (5 uints)
        )
        
        # Pipeline layout
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range]
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
        
        # Clean up shader module
        vk.vkDestroyShaderModule(self.device, shader_module, None)
        
        logger.info("   ✅ Compute pipeline created")
    
    def _allocate_command_buffer(self):
        """Allocate reusable command buffer"""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]
    
    def _find_memory_type(self, type_filter, properties):
        """Find suitable memory type"""
        for i in range(self.mem_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (self.mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")
    
    def _get_or_create_buffer(self, key, size_bytes, data=None, persistent=False):
        """Get buffer from cache or create new one"""
        
        # Check cache first
        if key in self.buffer_cache:
            # Move to end (LRU)
            self.buffer_cache.move_to_end(key)
            return self.buffer_cache[key]
        
        if key in self.persistent_buffers:
            return self.persistent_buffers[key]
        
        # Check memory limit
        if self.current_memory_usage + size_bytes > self.max_memory_bytes:
            self._evict_buffers(size_bytes)
        
        # Default to mappable
        is_mappable = True
        
        # Create new buffer
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                  vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                  vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate memory
        # For persistent buffers, try DEVICE_LOCAL + HOST_VISIBLE (GTT) first
        if persistent:
            try:
                # Try GTT (both device local and host visible)
                memory_type_index = self._find_memory_type(
                    mem_reqs.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                    vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                )
                is_mappable = True
            except:
                try:
                    # Try pure VRAM (device local only)
                    memory_type_index = self._find_memory_type(
                        mem_reqs.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    )
                    is_mappable = False
                except:
                    # Fallback to host visible
                    memory_type_index = self._find_memory_type(
                        mem_reqs.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    )
                    is_mappable = True
        else:
            memory_type_index = self._find_memory_type(
                mem_reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            is_mappable = True
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=memory_type_index
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        # Upload data if provided
        if data is not None and is_mappable:
            self._upload_to_buffer(buffer, memory, data, size_bytes)
        elif data is not None and not is_mappable:
            # For non-mappable memory, use staging buffer
            self._upload_via_staging(buffer, data, size_bytes)
        
        buffer_info = {
            'buffer': buffer,
            'memory': memory,
            'size': size_bytes,
            'mappable': is_mappable
        }
        
        # Cache the buffer
        if persistent:
            self.persistent_buffers[key] = buffer_info
        else:
            self.buffer_cache[key] = buffer_info
        
        self.current_memory_usage += size_bytes
        
        return buffer_info
    
    def _upload_to_buffer(self, buffer, memory, data, size_bytes):
        """Upload data to buffer"""
        # Map memory
        data_ptr = vk.vkMapMemory(self.device, memory, 0, size_bytes, 0)
        
        # Copy data
        vk.ffi.memmove(data_ptr, data.tobytes(), size_bytes)
        
        # Unmap
        vk.vkUnmapMemory(self.device, memory)
    
    def _upload_via_staging(self, dst_buffer, data, size_bytes):
        """Upload data via staging buffer for non-mappable memory"""
        # Create staging buffer
        staging_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        staging_buffer = vk.vkCreateBuffer(self.device, staging_info, None)
        staging_reqs = vk.vkGetBufferMemoryRequirements(self.device, staging_buffer)
        
        # Allocate host-visible memory for staging
        staging_mem_type = self._find_memory_type(
            staging_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        
        staging_alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=staging_reqs.size,
            memoryTypeIndex=staging_mem_type
        )
        
        staging_memory = vk.vkAllocateMemory(self.device, staging_alloc, None)
        vk.vkBindBufferMemory(self.device, staging_buffer, staging_memory, 0)
        
        # Upload to staging buffer
        self._upload_to_buffer(staging_buffer, staging_memory, data, size_bytes)
        
        # Copy from staging to destination
        self._copy_buffer(staging_buffer, dst_buffer, size_bytes)
        
        # Clean up staging resources
        vk.vkDestroyBuffer(self.device, staging_buffer, None)
        vk.vkFreeMemory(self.device, staging_memory, None)
    
    def _copy_buffer(self, src_buffer, dst_buffer, size):
        """Copy data between buffers"""
        # Begin command buffer
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        
        # Record copy command
        copy_region = vk.VkBufferCopy(
            srcOffset=0,
            dstOffset=0,
            size=size
        )
        
        vk.vkCmdCopyBuffer(self.command_buffer, src_buffer, dst_buffer, 1, [copy_region])
        
        vk.vkEndCommandBuffer(self.command_buffer)
        
        # Submit and wait
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer]
        )
        
        vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], None)
        vk.vkQueueWaitIdle(self.compute_queue)
        
        # Reset command buffer
        vk.vkResetCommandBuffer(self.command_buffer, 0)
    
    def _download_from_buffer(self, buffer, memory, size_bytes, dtype=np.float32):
        """Download data from buffer"""
        # Map memory
        data_ptr = vk.vkMapMemory(self.device, memory, 0, size_bytes, 0)
        
        # Create numpy array from the mapped memory pointer
        # The data_ptr is a CFFI buffer that we can directly read from
        result = np.zeros(size_bytes // 4, dtype=dtype)  # Assuming float32
        vk.ffi.memmove(vk.ffi.from_buffer(result), data_ptr, size_bytes)
        
        # Unmap
        vk.vkUnmapMemory(self.device, memory)
        
        return result
    
    def _download_via_staging(self, src_buffer, size_bytes, dtype=np.float32):
        """Download data via staging buffer for non-mappable memory"""
        # Create staging buffer
        staging_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        staging_buffer = vk.vkCreateBuffer(self.device, staging_info, None)
        staging_reqs = vk.vkGetBufferMemoryRequirements(self.device, staging_buffer)
        
        # Allocate host-visible memory for staging
        staging_mem_type = self._find_memory_type(
            staging_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        
        staging_alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=staging_reqs.size,
            memoryTypeIndex=staging_mem_type
        )
        
        staging_memory = vk.vkAllocateMemory(self.device, staging_alloc, None)
        vk.vkBindBufferMemory(self.device, staging_buffer, staging_memory, 0)
        
        # Copy from source to staging
        self._copy_buffer(src_buffer, staging_buffer, size_bytes)
        
        # Download from staging buffer
        result = self._download_from_buffer(staging_buffer, staging_memory, size_bytes, dtype)
        
        # Clean up staging resources
        vk.vkDestroyBuffer(self.device, staging_buffer, None)
        vk.vkFreeMemory(self.device, staging_memory, None)
        
        return result
    
    def _evict_buffers(self, needed_bytes):
        """Evict least recently used buffers"""
        freed = 0
        
        while freed < needed_bytes and len(self.buffer_cache) > 0:
            # Remove oldest (least recently used)
            key, buffer_info = self.buffer_cache.popitem(last=False)
            
            # Free Vulkan resources
            vk.vkDestroyBuffer(self.device, buffer_info['buffer'], None)
            vk.vkFreeMemory(self.device, buffer_info['memory'], None)
            
            freed += buffer_info['size']
            self.current_memory_usage -= buffer_info['size']
        
        logger.debug(f"Evicted {freed / (1024**2):.1f}MB from buffer cache")
    
    def matrix_multiply(self, a, b, use_fp16=False):
        """Optimized matrix multiplication with proper shader dispatch"""
        
        # Ensure inputs are contiguous float32 arrays
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        
        # Get dimensions
        M, K = a.shape
        K2, N = b.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimension mismatch: {K} != {K2}")
        
        # Create unique keys for buffers
        a_key = f"input_a_{M}x{K}"
        b_key = f"input_b_{K}x{N}"
        c_key = f"output_c_{M}x{N}"
        
        # Get or create buffers
        a_buffer = self._get_or_create_buffer(a_key, a.nbytes, a)
        b_buffer = self._get_or_create_buffer(b_key, b.nbytes, b)
        c_buffer = self._get_or_create_buffer(c_key, M * N * 4)  # float32 output
        
        # Create descriptor set
        desc_alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )
        
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, desc_alloc_info)[0]
        
        # Update descriptor set with buffers
        buffer_infos = [
            vk.VkDescriptorBufferInfo(
                buffer=a_buffer['buffer'],
                offset=0,
                range=a_buffer['size']
            ),
            vk.VkDescriptorBufferInfo(
                buffer=b_buffer['buffer'],
                offset=0,
                range=b_buffer['size']
            ),
            vk.VkDescriptorBufferInfo(
                buffer=c_buffer['buffer'],
                offset=0,
                range=c_buffer['size']
            )
        ]
        
        write_sets = []
        for i, buffer_info in enumerate(buffer_infos):
            write_set = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_info]
            )
            write_sets.append(write_set)
        
        vk.vkUpdateDescriptorSets(self.device, len(write_sets), write_sets, 0, None)
        
        # Record command buffer
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)
        
        # Bind pipeline and descriptor set
        vk.vkCmdBindPipeline(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.compute_pipeline
        )
        
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0, 1, [descriptor_set],
            0, None
        )
        
        # Push constants (matrix dimensions)
        tile_size = 16 if M >= 16 and N >= 16 else 4
        flags = 1 if use_fp16 else 0
        push_constants = struct.pack('IIIII', M, N, K, tile_size, flags)
        
        vk.vkCmdPushConstants(
            self.command_buffer,
            self.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, 20, vk.ffi.from_buffer(push_constants)
        )
        
        # Dispatch compute shader
        workgroup_x = (N + tile_size - 1) // tile_size
        workgroup_y = (M + tile_size - 1) // tile_size
        
        vk.vkCmdDispatch(self.command_buffer, workgroup_x, workgroup_y, 1)
        
        vk.vkEndCommandBuffer(self.command_buffer)
        
        # Submit command buffer
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer]
        )
        
        vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], None)
        vk.vkQueueWaitIdle(self.compute_queue)
        
        # Reset command buffer for reuse
        vk.vkResetCommandBuffer(self.command_buffer, 0)
        
        # Read result
        if c_buffer.get('mappable', True):
            result = self._download_from_buffer(
                c_buffer['buffer'], 
                c_buffer['memory'], 
                c_buffer['size']
            )
        else:
            # For non-mappable memory, use staging buffer
            result = self._download_via_staging(
                c_buffer['buffer'], 
                c_buffer['size']
            )
        
        # Free descriptor set
        vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        return result.reshape(M, N)
    
    def cache_weight(self, key, weight):
        """Cache weight tensor in VRAM persistently"""
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        self._get_or_create_buffer(key, weight.nbytes, weight, persistent=True)
        logger.debug(f"Cached weight {key} ({weight.nbytes / (1024**2):.1f}MB) in VRAM")
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        cache_size = sum(b['size'] for b in self.buffer_cache.values())
        persistent_size = sum(b['size'] for b in self.persistent_buffers.values())
        
        return {
            'cache_size_mb': cache_size / (1024**2),
            'persistent_size_mb': persistent_size / (1024**2),
            'total_usage_mb': self.current_memory_usage / (1024**2),
            'max_memory_mb': self.max_memory_bytes / (1024**2),
            'num_cached_buffers': len(self.buffer_cache),
            'num_persistent_buffers': len(self.persistent_buffers)
        }
    
    def cleanup(self):
        """Clean up all resources"""
        # Free all buffers
        for buffer_info in list(self.buffer_cache.values()):
            vk.vkDestroyBuffer(self.device, buffer_info['buffer'], None)
            vk.vkFreeMemory(self.device, buffer_info['memory'], None)
        
        for buffer_info in list(self.persistent_buffers.values()):
            vk.vkDestroyBuffer(self.device, buffer_info['buffer'], None)
            vk.vkFreeMemory(self.device, buffer_info['memory'], None)
        
        # Free Vulkan resources
        if self.command_buffer:
            vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [self.command_buffer])
        if self.compute_pipeline:
            vk.vkDestroyPipeline(self.device, self.compute_pipeline, None)
        if self.pipeline_layout:
            vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        if self.descriptor_set_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        if self.device:
            vk.vkDestroyDevice(self.device, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)


def test_optimized_engine():
    """Test the optimized Vulkan engine"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🧪 Testing Optimized Vulkan Compute Engine")
    
    engine = VulkanComputeOptimized(max_memory_gb=4.0)
    if not engine.initialize():
        return
    
    # Test matrix multiplication
    logger.info("\n📊 Testing matrix multiplication...")
    
    # Test sizes
    sizes = [(8, 512, 512), (8, 1024, 1024), (8, 2048, 2048)]
    
    for batch, m, n in sizes:
        a = np.random.randn(batch * m, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        
        # Warmup
        _ = engine.matrix_multiply(a, b)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = engine.matrix_multiply(a, b)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        gflops = (2 * batch * m * n * n) / (avg_time * 1e9)
        
        logger.info(f"   {batch}x{m}x{n}: {avg_time*1000:.2f}ms, {gflops:.1f} GFLOPS")
    
    # Test weight caching
    logger.info("\n📊 Testing weight caching...")
    
    # Cache some weights
    for i in range(5):
        weight = np.random.randn(1024, 1024).astype(np.float32)
        engine.cache_weight(f"layer_{i}_weight", weight)
    
    # Show memory stats
    stats = engine.get_memory_stats()
    logger.info(f"\n💾 Memory Stats:")
    logger.info(f"   Cache: {stats['cache_size_mb']:.1f}MB ({stats['num_cached_buffers']} buffers)")
    logger.info(f"   Persistent: {stats['persistent_size_mb']:.1f}MB ({stats['num_persistent_buffers']} buffers)")
    logger.info(f"   Total: {stats['total_usage_mb']:.1f}MB / {stats['max_memory_mb']:.1f}MB")
    
    engine.cleanup()
    logger.info("\n✅ Test complete!")


if __name__ == "__main__":
    test_optimized_engine()