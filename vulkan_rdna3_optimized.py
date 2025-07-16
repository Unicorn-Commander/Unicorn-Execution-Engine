#!/usr/bin/env python3
"""
RDNA3-Optimized Vulkan Compute for AMD Radeon 780M
Optimized for maximum performance on RDNA3 architecture:
- 64 threads per workgroup (RDNA3 wavefront size)
- Memory coalescing patterns
- Device-local memory optimization
- Asynchronous execution
"""

import numpy as np
import vulkan as vk
import logging
from pathlib import Path
import ctypes
import struct
import time

logger = logging.getLogger(__name__)

class RDNA3OptimizedCompute:
    """RDNA3-optimized Vulkan compute for AMD Radeon 780M"""
    
    def __init__(self):
        self.instance = None
        self.device = None
        self.physical_device = None
        self.compute_queue = None
        self.command_pool = None
        self.initialized = False
        
        # RDNA3 optimization parameters
        self.WORKGROUP_SIZE_X = 8    # 8x8 = 64 threads (RDNA3 wavefront)
        self.WORKGROUP_SIZE_Y = 8
        self.TILE_SIZE = 16          # Optimal tile size for RDNA3
        self.MAX_COMPUTE_UNITS = 12  # AMD Radeon 780M has 12 CUs
        
    def initialize(self):
        """Initialize RDNA3-optimized Vulkan compute"""
        logger.info("🚀 Initializing RDNA3-Optimized Vulkan Compute...")
        
        try:
            # Create instance
            app_info = vk.VkApplicationInfo(
                sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName='RDNA3-Optimized-Compute',
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
            
            # Get AMD Radeon 780M device
            devices = vk.vkEnumeratePhysicalDevices(self.instance)
            self.physical_device = devices[0]
            
            # Get device properties for RDNA3 optimization
            props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
            device_name = props.deviceName if isinstance(props.deviceName, str) else props.deviceName.decode('utf-8')
            logger.info(f"   ✅ RDNA3 Device: {device_name}")
            logger.info(f"   ✅ Max Compute Units: {self.MAX_COMPUTE_UNITS}")
            logger.info(f"   ✅ Workgroup Size: {self.WORKGROUP_SIZE_X}x{self.WORKGROUP_SIZE_Y}")
            
            # Find compute queue
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
            compute_queue_family = None
            for i, family in enumerate(queue_families):
                if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    compute_queue_family = i
                    break
            
            # Create device with compute queue
            queue_create_info = vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=compute_queue_family,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            
            device_create_info = vk.VkDeviceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_create_info]
            )
            
            self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)
            self.compute_queue = vk.vkGetDeviceQueue(self.device, compute_queue_family, 0)
            
            # Create command pool
            pool_create_info = vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=compute_queue_family
            )
            self.command_pool = vk.vkCreateCommandPool(self.device, pool_create_info, None)
            
            self.initialized = True
            logger.info("✅ RDNA3-Optimized Vulkan Compute initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ RDNA3 initialization failed: {e}")
            return False
    
    def create_optimized_buffer(self, size, usage, device_local=True):
        """Create optimized buffer with device-local memory"""
        buffer_create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_create_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        # Find optimal memory type
        memory_type_index = None
        required_props = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT if device_local else vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        
        for i in range(mem_props.memoryTypeCount):
            if (mem_reqs.memoryTypeBits & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags & required_props):
                memory_type_index = i
                break
        
        if memory_type_index is None:
            raise RuntimeError("Failed to find suitable memory type")
        
        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=memory_type_index
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def create_compute_shader(self):
        """Create RDNA3-optimized compute shader"""
        # RDNA3-optimized GLSL compute shader
        shader_code = f"""
#version 450

// RDNA3-optimized workgroup size (64 threads = 1 wavefront)
layout(local_size_x = {self.WORKGROUP_SIZE_X}, local_size_y = {self.WORKGROUP_SIZE_Y}, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {{
    float A[];
}};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {{
    float B[];
}};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferC {{
    float C[];
}};

layout(push_constant) uniform PushConstants {{
    uint M;
    uint N;
    uint K;
}} pc;

// Shared memory for tiling (optimized for RDNA3)
shared float tileA[{self.TILE_SIZE}][{self.TILE_SIZE}];
shared float tileB[{self.TILE_SIZE}][{self.TILE_SIZE}];

void main() {{
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (row >= pc.M || col >= pc.N) return;
    
    float result = 0.0;
    
    // Tiled matrix multiplication for memory coalescing
    for (uint tile = 0; tile < (pc.K + {self.TILE_SIZE} - 1) / {self.TILE_SIZE}; ++tile) {{
        // Load tile into shared memory
        uint tileRow = gl_LocalInvocationID.y;
        uint tileCol = gl_LocalInvocationID.x;
        
        // Load A tile
        uint globalRow = row;
        uint globalCol = tile * {self.TILE_SIZE} + tileCol;
        if (globalRow < pc.M && globalCol < pc.K) {{
            tileA[tileRow][tileCol] = A[globalRow * pc.K + globalCol];
        }} else {{
            tileA[tileRow][tileCol] = 0.0;
        }}
        
        // Load B tile
        globalRow = tile * {self.TILE_SIZE} + tileRow;
        globalCol = col;
        if (globalRow < pc.K && globalCol < pc.N) {{
            tileB[tileRow][tileCol] = B[globalRow * pc.N + globalCol];
        }} else {{
            tileB[tileRow][tileCol] = 0.0;
        }}
        
        // Synchronize workgroup
        barrier();
        
        // Compute partial result
        for (uint k = 0; k < {self.TILE_SIZE}; ++k) {{
            result += tileA[tileRow][k] * tileB[k][tileCol];
        }}
        
        // Synchronize before loading next tile
        barrier();
    }}
    
    // Store result
    C[row * pc.N + col] = result;
}}
"""
        
        return shader_code
    
    def compile_shader(self, shader_code):
        """Compile GLSL to SPIR-V"""
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.comp', delete=False) as f:
            f.write(shader_code)
            shader_file = f.name
        
        spirv_file = shader_file + '.spv'
        
        try:
            # Compile with glslangValidator
            result = subprocess.run([
                'glslangValidator', '-V', shader_file, '-o', spirv_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Shader compilation failed: {result.stderr}")
                return None
            
            # Read SPIR-V
            with open(spirv_file, 'rb') as f:
                spirv_code = f.read()
            
            # Clean up
            Path(shader_file).unlink()
            Path(spirv_file).unlink()
            
            return spirv_code
            
        except Exception as e:
            logger.error(f"Shader compilation error: {e}")
            return None
    
    def execute_optimized_matrix_multiply(self, matrix_a, matrix_b):
        """Execute RDNA3-optimized matrix multiplication"""
        if not self.initialized:
            raise RuntimeError("RDNA3 compute not initialized")
        
        logger.info("⚡ Executing RDNA3-Optimized Matrix Multiplication...")
        
        start_time = time.time()
        
        M, K = matrix_a.shape
        K2, N = matrix_b.shape
        assert K == K2, f"Matrix dimension mismatch: {K} != {K2}"
        
        # Convert to optimal data format
        a_data = matrix_a.astype(np.float32).tobytes()
        b_data = matrix_b.astype(np.float32).tobytes()
        output_size = M * N * 4  # float32
        
        logger.info(f"   📊 Matrix A: {M}x{K}, Matrix B: {K}x{N} → Output: {M}x{N}")
        logger.info(f"   💾 Memory: {len(a_data)//1024}KB + {len(b_data)//1024}KB → {output_size//1024}KB")
        
        try:
            # Create device-local buffers for optimal performance
            buffer_a, memory_a = self.create_optimized_buffer(
                len(a_data), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, True
            )
            buffer_b, memory_b = self.create_optimized_buffer(
                len(b_data), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, True
            )
            buffer_c, memory_c = self.create_optimized_buffer(
                output_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, True
            )
            
            # Create staging buffers for data transfer
            staging_a, staging_memory_a = self.create_optimized_buffer(
                len(a_data), vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, False
            )
            staging_b, staging_memory_b = self.create_optimized_buffer(
                len(b_data), vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, False
            )
            staging_c, staging_memory_c = self.create_optimized_buffer(
                output_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, False
            )
            
            # Upload data to staging buffers
            data_ptr = vk.vkMapMemory(self.device, staging_memory_a, 0, len(a_data), 0)
            vk.ffi.memmove(data_ptr, a_data, len(a_data))
            vk.vkUnmapMemory(self.device, staging_memory_a)
            
            data_ptr = vk.vkMapMemory(self.device, staging_memory_b, 0, len(b_data), 0)
            vk.ffi.memmove(data_ptr, b_data, len(b_data))
            vk.vkUnmapMemory(self.device, staging_memory_b)
            
            # Create compute shader
            shader_code = self.create_compute_shader()
            spirv_code = self.compile_shader(shader_code)
            
            if spirv_code is None:
                raise RuntimeError("Shader compilation failed")
            
            # Create shader module
            shader_module = vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(spirv_code),
                pCode=spirv_code
            ), None)
            
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
            
            descriptor_layout = vk.vkCreateDescriptorSetLayout(self.device, vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings),
                pBindings=bindings
            ), None)
            
            # Create pipeline layout with push constants
            push_constant_range = vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=12  # 3 uint32s: M, N, K
            )
            
            pipeline_layout = vk.vkCreatePipelineLayout(self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[descriptor_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_constant_range]
            ), None)
            
            # Create compute pipeline
            pipeline_info = vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=shader_module,
                    pName='main'
                ),
                layout=pipeline_layout
            )
            
            pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None)[0]
            
            # Create descriptor pool and set
            pool_sizes = [vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=3
            )]
            
            descriptor_pool = vk.vkCreateDescriptorPool(self.device, vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes
            ), None)
            
            descriptor_set = vk.vkAllocateDescriptorSets(self.device, vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[descriptor_layout]
            ))[0]
            
            # Update descriptor set
            buffer_infos = [
                vk.VkDescriptorBufferInfo(buffer=buffer_a, offset=0, range=len(a_data)),
                vk.VkDescriptorBufferInfo(buffer=buffer_b, offset=0, range=len(b_data)),
                vk.VkDescriptorBufferInfo(buffer=buffer_c, offset=0, range=output_size)
            ]
            
            writes = [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=i,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[buffer_infos[i]]
                ) for i in range(3)
            ]
            
            vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
            
            # Create command buffer
            cmd_buffer = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            ))[0]
            
            # Record commands
            vk.vkBeginCommandBuffer(cmd_buffer, vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            ))
            
            # Copy data to device buffers
            vk.vkCmdCopyBuffer(cmd_buffer, staging_a, buffer_a, 1, [vk.VkBufferCopy(size=len(a_data))])
            vk.vkCmdCopyBuffer(cmd_buffer, staging_b, buffer_b, 1, [vk.VkBufferCopy(size=len(b_data))])
            
            # Memory barriers
            vk.vkCmdPipelineBarrier(cmd_buffer,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None, 0, None)
            
            # Bind pipeline and descriptor sets
            vk.vkCmdBindPipeline(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
            vk.vkCmdBindDescriptorSets(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, 
                                     pipeline_layout, 0, 1, [descriptor_set], 0, None)
            
            # Push constants
            push_data = struct.pack('III', M, N, K)
            vk.vkCmdPushConstants(cmd_buffer, pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(push_data), push_data)
            
            # Calculate optimal dispatch size for RDNA3
            dispatch_x = (N + self.WORKGROUP_SIZE_X - 1) // self.WORKGROUP_SIZE_X
            dispatch_y = (M + self.WORKGROUP_SIZE_Y - 1) // self.WORKGROUP_SIZE_Y
            
            logger.info(f"   🚀 Dispatching: {dispatch_x}x{dispatch_y} workgroups ({dispatch_x * dispatch_y * 64} threads)")
            
            # Dispatch compute shader
            vk.vkCmdDispatch(cmd_buffer, dispatch_x, dispatch_y, 1)
            
            # Memory barrier before copy
            vk.vkCmdPipelineBarrier(cmd_buffer,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, None, 0, None, 0, None)
            
            # Copy result back
            vk.vkCmdCopyBuffer(cmd_buffer, buffer_c, staging_c, 1, [vk.VkBufferCopy(size=output_size)])
            
            vk.vkEndCommandBuffer(cmd_buffer)
            
            # Submit and wait
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd_buffer]
            )
            
            vk.vkQueueSubmit(self.compute_queue, 1, [submit_info], vk.VK_NULL_HANDLE)
            vk.vkQueueWaitIdle(self.compute_queue)
            
            # Read result
            data_ptr = vk.vkMapMemory(self.device, staging_memory_c, 0, output_size, 0)
            result_data = vk.ffi.buffer(data_ptr, output_size)[:]
            vk.vkUnmapMemory(self.device, staging_memory_c)
            
            # Convert back to numpy
            result = np.frombuffer(result_data, dtype=np.float32).reshape(M, N)
            
            execution_time = time.time() - start_time
            throughput = (M * N * K * 2) / (execution_time * 1e9)  # GFLOPS
            
            logger.info(f"   ✅ RDNA3 execution completed in {execution_time*1000:.2f}ms")
            logger.info(f"   📊 Throughput: {throughput:.2f} GFLOPS")
            logger.info(f"   🎯 Memory bandwidth: {(len(a_data) + len(b_data) + output_size) / (execution_time * 1e9):.2f} GB/s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ RDNA3 matrix multiplication failed: {e}")
            raise

if __name__ == "__main__":
    # Test RDNA3 optimization
    compute = RDNA3OptimizedCompute()
    if compute.initialize():
        # Test with realistic dimensions
        A = np.random.randn(64, 4096).astype(np.float32) * 0.1
        B = np.random.randn(4096, 14336).astype(np.float32) * 0.1
        
        result = compute.execute_optimized_matrix_multiply(A, B)
        print(f"✅ RDNA3 optimization test completed: {result.shape}")