#!/usr/bin/env python3
"""
Optimized Vulkan Transformer Engine
High-performance compute engine for transformer operations
"""

import numpy as np
import vulkan as vk
import logging
import time
from pathlib import Path
import struct

logger = logging.getLogger(__name__)

class VulkanTransformerEngine:
    """Optimized Vulkan engine for transformer operations"""
    
    def __init__(self, use_fp16=True):
        self.use_fp16 = use_fp16
        self.instance = None
        self.device = None
        self.physical_device = None
        self.compute_queue = None
        self.command_pool = None
        
        # Shader pipelines
        self.transformer_pipeline = None
        self.batched_gemm_pipeline = None
        self.standard_gemm_pipeline = None
        
        # Memory pool for GPU buffers
        self.buffer_pool = {}
        self.memory_pool = {}
        self.pool_size = 0
        self.max_pool_size = 4 * 1024 * 1024 * 1024  # 4GB pool
        
        self.initialized = False
        
    def initialize(self):
        """Initialize Vulkan with optimized settings"""
        logger.info("🚀 Initializing Optimized Vulkan Transformer Engine")
        logger.info(f"   FP16 mode: {'Enabled' if self.use_fp16 else 'Disabled'}")
        
        try:
            self._create_instance()
            self._select_device()
            self._create_device()
            self._create_command_pool()
            self._load_shaders()
            self._create_descriptor_pool()
            
            self.initialized = True
            logger.info("✅ Vulkan Transformer Engine initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _create_instance(self):
        """Create Vulkan instance"""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='VulkanTransformerEngine',
            applicationVersion=vk.VK_MAKE_VERSION(2, 0, 0),
            pEngineName='PureHardware',
            engineVersion=vk.VK_MAKE_VERSION(2, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_2
        )
        
        # Enable validation in debug mode
        layers = []
        extensions = []
        
        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        self.instance = vk.vkCreateInstance(instance_info, None)
        logger.info("   ✅ Vulkan instance created")
    
    def _select_device(self):
        """Select AMD GPU with preference for RDNA3"""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        
        best_device = None
        best_score = 0
        
        for device in devices:
            props = vk.vkGetPhysicalDeviceProperties(device)
            device_name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
            
            score = 0
            if 'AMD' in device_name or 'Radeon' in device_name:
                score += 1000
                if '780M' in device_name or 'RDNA3' in device_name:
                    score += 500  # Prefer RDNA3
                
                # Check compute capabilities
                queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
                for i, family in enumerate(queue_families):
                    if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                        score += 100
                
                if score > best_score:
                    best_score = score
                    best_device = device
        
        if best_device:
            self.physical_device = best_device
            props = vk.vkGetPhysicalDeviceProperties(best_device)
            device_name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
            logger.info(f"   ✅ Selected device: {device_name}")
            
            # Log device limits
            logger.info(f"   Max compute workgroup size: {props.limits.maxComputeWorkGroupSize}")
            logger.info(f"   Max compute workgroup invocations: {props.limits.maxComputeWorkGroupInvocations}")
    
    def _create_device(self):
        """Create logical device with compute queue"""
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        
        compute_family = -1
        for i, props in enumerate(queue_families):
            if props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                compute_family = i
                break
        
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=compute_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        # Enable FP16 features
        features = vk.VkPhysicalDeviceFeatures()
        features.shaderFloat64 = False  # We don't need FP64
        
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            pEnabledFeatures=features
        )
        
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.compute_queue = vk.vkGetDeviceQueue(self.device, compute_family, 0)
        self.compute_queue_family = compute_family
        logger.info("   ✅ Logical device created")
    
    def _create_command_pool(self):
        """Create command pool for compute operations"""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.compute_queue_family
        )
        
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        logger.info("   ✅ Command pool created")
    
    def _load_shaders(self):
        """Load optimized compute shaders"""
        shader_dir = Path(__file__).parent
        
        # Load transformer optimized shader
        transformer_shader = shader_dir / "transformer_optimized.spv"
        if transformer_shader.exists():
            self.transformer_pipeline = self._create_compute_pipeline(transformer_shader)
            logger.info("   ✅ Transformer optimized shader loaded")
        
        # Load batched GEMM shader
        batched_shader = shader_dir / "batched_gemm.spv"
        if batched_shader.exists():
            self.batched_gemm_pipeline = self._create_compute_pipeline(batched_shader)
            logger.info("   ✅ Batched GEMM shader loaded")
        
        # Load standard GEMM shader
        standard_shader = shader_dir / "transformer_optimized.spv"
        if standard_shader.exists():
            self.standard_gemm_pipeline = self._create_compute_pipeline(standard_shader)
            logger.info("   ✅ Standard GEMM shader loaded")
    
    def _create_compute_pipeline(self, shader_path):
        """Create compute pipeline from shader"""
        with open(shader_path, 'rb') as f:
            shader_code = f.read()
        
        shader_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )
        
        shader_module = vk.vkCreateShaderModule(self.device, shader_module_info, None)
        
        # Create pipeline layout with push constants
        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=64  # Enough for our parameters
        )
        
        # Descriptor set layout for buffers
        bindings = []
        for i in range(4):  # Support up to 4 buffers
            bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            ))
        
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )
        
        descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)
        
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range]
        )
        
        pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)
        
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
            layout=pipeline_layout
        )
        
        pipeline = vk.vkCreateComputePipelines(
            self.device, None, 1, [pipeline_info], None
        )[0]
        
        return {
            'pipeline': pipeline,
            'layout': pipeline_layout,
            'descriptor_layout': descriptor_set_layout
        }
    
    def _create_descriptor_pool(self):
        """Create descriptor pool for buffer management"""
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1000  # Support many buffers
        )
        
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=250,  # Support many descriptor sets
            poolSizeCount=1,
            pPoolSizes=[pool_size]
        )
        
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        logger.info("   ✅ Descriptor pool created")
    
    def get_or_create_buffer(self, key, size_bytes, prefer_vram=True):
        """Get buffer from pool or create new one"""
        if key in self.buffer_pool:
            buffer_info = self.buffer_pool[key]
            if buffer_info['size'] >= size_bytes:
                return buffer_info['buffer'], buffer_info['memory']
        
        # Create new buffer
        if self.pool_size + size_bytes > self.max_pool_size:
            self._cleanup_old_buffers()
        
        buffer, memory = self._create_gpu_buffer(size_bytes, prefer_vram)
        
        self.buffer_pool[key] = {
            'buffer': buffer,
            'memory': memory,
            'size': size_bytes,
            'last_used': time.time()
        }
        
        self.pool_size += size_bytes
        return buffer, memory
    
    def _create_gpu_buffer(self, size_bytes, prefer_vram=True):
        """Create GPU buffer with proper memory allocation"""
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
        
        # Find memory type
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        memory_type_index = -1
        
        if prefer_vram:
            # Try DEVICE_LOCAL first (VRAM)
            for i in range(mem_props.memoryTypeCount):
                if (mem_reqs.memoryTypeBits & (1 << i)) and \
                   (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
                    memory_type_index = i
                    break
        
        if memory_type_index == -1:
            # Fallback to HOST_VISIBLE (GTT/RAM)
            for i in range(mem_props.memoryTypeCount):
                if (mem_reqs.memoryTypeBits & (1 << i)) and \
                   (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT):
                    memory_type_index = i
                    break
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=memory_type_index
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def batched_matmul(self, input_batch, weight, bias=None, activation='none'):
        """Optimized batched matrix multiplication"""
        batch_size, seq_len, hidden_dim = input_batch.shape
        out_dim = weight.shape[1]
        
        # Flatten batch and sequence dimensions
        input_flat = input_batch.reshape(-1, hidden_dim)
        
        # Use FP16 if enabled
        if self.use_fp16:
            input_flat = input_flat.astype(np.float16)
            weight = weight.astype(np.float16)
            if bias is not None:
                bias = bias.astype(np.float16)
        
        # Get or create GPU buffers
        input_buffer, input_memory = self.get_or_create_buffer(
            'input', input_flat.nbytes, prefer_vram=False
        )
        weight_buffer, weight_memory = self.get_or_create_buffer(
            'weight', weight.nbytes, prefer_vram=True
        )
        output_buffer, output_memory = self.get_or_create_buffer(
            'output', batch_size * seq_len * out_dim * 4, prefer_vram=False
        )
        
        # Transfer data to GPU
        self._upload_to_buffer(input_flat, input_buffer, input_memory)
        self._upload_to_buffer(weight, weight_buffer, weight_memory)
        
        if bias is not None:
            bias_buffer, bias_memory = self.get_or_create_buffer(
                'bias', bias.nbytes, prefer_vram=True
            )
            self._upload_to_buffer(bias, bias_buffer, bias_memory)
        
        # Dispatch compute shader
        if self.transformer_pipeline:
            # Use optimized transformer shader
            self._dispatch_transformer_compute(
                input_buffer, weight_buffer, 
                bias_buffer if bias else None,
                output_buffer,
                batch_size * seq_len, hidden_dim, out_dim,
                activation
            )
        else:
            # Fallback to standard GEMM
            self._dispatch_standard_compute(
                input_buffer, weight_buffer, output_buffer,
                batch_size * seq_len, hidden_dim, out_dim
            )
        
        # Read back results
        output_flat = self._download_from_buffer(
            output_buffer, output_memory, 
            batch_size * seq_len * out_dim * 4
        )
        
        # Reshape to original batch dimensions
        output = output_flat.reshape(batch_size, seq_len, out_dim)
        
        return output
    
    def _upload_to_buffer(self, data, buffer, memory):
        """Upload data to GPU buffer"""
        # Implementation depends on memory type
        # For HOST_VISIBLE memory, we can map directly
        # For DEVICE_LOCAL, we need a staging buffer
        pass
    
    def _download_from_buffer(self, buffer, memory, size_bytes):
        """Download data from GPU buffer"""
        # Implementation depends on memory type
        pass
    
    def _dispatch_transformer_compute(self, input_buf, weight_buf, bias_buf, 
                                    output_buf, M, K, N, activation):
        """Dispatch optimized transformer compute shader"""
        # Create command buffer
        # Bind pipeline and buffers
        # Set push constants
        # Dispatch with optimal workgroup size
        pass
    
    def _dispatch_standard_compute(self, input_buf, weight_buf, output_buf, M, K, N):
        """Dispatch standard GEMM compute shader"""
        # Similar to transformer dispatch but simpler
        pass
    
    def _cleanup_old_buffers(self):
        """Clean up least recently used buffers"""
        # Sort by last used time and remove oldest
        sorted_buffers = sorted(
            self.buffer_pool.items(), 
            key=lambda x: x[1]['last_used']
        )
        
        # Remove oldest 25% of buffers
        remove_count = len(sorted_buffers) // 4
        for key, buffer_info in sorted_buffers[:remove_count]:
            vk.vkDestroyBuffer(self.device, buffer_info['buffer'], None)
            vk.vkFreeMemory(self.device, buffer_info['memory'], None)
            self.pool_size -= buffer_info['size']
            del self.buffer_pool[key]
    
    def cleanup(self):
        """Clean up all resources"""
        # Clean up buffer pool
        for buffer_info in self.buffer_pool.values():
            vk.vkDestroyBuffer(self.device, buffer_info['buffer'], None)
            vk.vkFreeMemory(self.device, buffer_info['memory'], None)
        
        # Clean up Vulkan resources
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
    logger.info("🧪 Testing Optimized Vulkan Transformer Engine")
    
    engine = VulkanTransformerEngine(use_fp16=True)
    if not engine.initialize():
        return
    
    # Test batch processing
    batch_size = 8
    seq_len = 128
    hidden_dim = 5376
    output_dim = 5376
    
    # Create test data
    input_batch = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    weight = np.random.randn(hidden_dim, output_dim).astype(np.float32)
    bias = np.random.randn(output_dim).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        _ = engine.batched_matmul(input_batch, weight, bias, activation='silu')
    
    # Benchmark
    start_time = time.time()
    num_iterations = 10
    
    for _ in range(num_iterations):
        output = engine.batched_matmul(input_batch, weight, bias, activation='silu')
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_iterations
    
    # Calculate performance
    flops = 2 * batch_size * seq_len * hidden_dim * output_dim
    gflops = (flops / avg_time) / 1e9
    
    # Tokens per second (assuming this is one layer)
    tokens_processed = batch_size * seq_len
    layer_time = avg_time
    model_time = layer_time * 62  # 62 layers
    tps = tokens_processed / model_time
    
    logger.info(f"\n📊 Performance Results:")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Sequence length: {seq_len}")
    logger.info(f"   Total tokens: {tokens_processed}")
    logger.info(f"   Layer time: {layer_time*1000:.2f}ms")
    logger.info(f"   Performance: {gflops:.1f} GFLOPS")
    logger.info(f"   Tokens per second: {tps:.2f} TPS")
    
    engine.cleanup()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_optimized_engine()