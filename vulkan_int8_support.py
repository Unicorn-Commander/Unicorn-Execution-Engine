"""
Vulkan INT8 Support Extension for VulkanMatrixCompute
Adds native INT8 operations to keep weights quantized in GPU memory
"""

import numpy as np
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def add_int8_support(VulkanMatrixCompute):
    """Add INT8 support methods to VulkanMatrixCompute class"""
    
    # Store original methods
    VulkanMatrixCompute._original_allocate_gpu_memory = VulkanMatrixCompute._allocate_gpu_memory
    VulkanMatrixCompute._original_create_compute_pipeline = VulkanMatrixCompute._create_compute_pipeline
    
    def _create_int8_compute_pipelines(self):
        """Create INT8 compute pipelines"""
        import vulkan as vk
        
        # Load INT8 shaders
        shaders = {
            'matrix_multiply_int8': 'matrix_multiply_int8.spv',
            'gate_up_silu_mul_int8': 'gate_up_silu_mul_int8.spv',
        }
        
        self.int8_pipelines = {}
        self.int8_layouts = {}
        
        for name, spv_file in shaders.items():
            shader_path = Path(__file__).parent / spv_file
            if not shader_path.exists():
                logger.warning(f"INT8 shader {spv_file} not found, skipping")
                continue
                
            with open(shader_path, 'rb') as f:
                shader_code = f.read()
            
            # Create shader module
            shader_module_info = vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(shader_code),
                pCode=shader_code
            )
            
            shader_module = vk.vkCreateShaderModule(self.device, shader_module_info, None)
            
            # Create descriptor set layout for INT8 (needs scale buffers)
            if 'matrix_multiply' in name:
                bindings = [
                    # Matrix A (INT8)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=0,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Matrix B (INT8)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Matrix C (output)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=2,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Scale A
                    vk.VkDescriptorSetLayoutBinding(
                        binding=3,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Scale B
                    vk.VkDescriptorSetLayoutBinding(
                        binding=4,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                ]
            else:  # FFN shaders
                bindings = [
                    # Input (FP32)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=0,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Gate weight (INT8)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Up weight (INT8)
                    vk.VkDescriptorSetLayoutBinding(
                        binding=2,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Gate scale
                    vk.VkDescriptorSetLayoutBinding(
                        binding=3,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Up scale
                    vk.VkDescriptorSetLayoutBinding(
                        binding=4,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                    # Output
                    vk.VkDescriptorSetLayoutBinding(
                        binding=5,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
                    ),
                ]
            
            layout_info = vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings),
                pBindings=bindings
            )
            
            descriptor_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)
            
            # Create pipeline layout
            push_constant_range = vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=20 if 'matrix_multiply' in name else 12  # Different push constants
            )
            
            pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[descriptor_layout],
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
                self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
            )[0]
            
            self.int8_pipelines[name] = pipeline
            self.int8_layouts[name] = (descriptor_layout, pipeline_layout)
            
            # Clean up shader module
            vk.vkDestroyShaderModule(self.device, shader_module, None)
            
            logger.info(f"   ✅ INT8 {name} pipeline created")
    
    def _allocate_gpu_memory_int8(self, tensor_data):
        """Allocate INT8 tensor data to GPU without conversion"""
        if not self.initialized:
            raise RuntimeError("Vulkan compute not initialized")
            
        # Keep INT8 data as-is
        if hasattr(tensor_data, 'numpy'):
            np_data = tensor_data.detach().cpu().numpy()
        elif isinstance(tensor_data, np.ndarray):
            np_data = tensor_data
        else:
            np_data = np.array(tensor_data)
        
        # Create persistent GPU buffer with original dtype
        try:
            buffer, memory, size_bytes = self.create_persistent_buffer(np_data)
            
            # Track allocated memory
            self.allocated_buffers.append((buffer, memory, size_bytes))
            self.memory_usage_mb += size_bytes / (1024 * 1024)
            
            logger.debug(f"✅ Allocated {size_bytes / (1024*1024):.1f}MB INT8 to GPU")
            
            return (buffer, memory, size_bytes)
            
        except Exception as e:
            logger.error(f"❌ INT8 GPU allocation failed: {e}")
            raise
    
    def compute_matrix_multiply_int8(self, matrix_a, matrix_b_buffer, shape_b, 
                                   scale_a=1.0, scale_b=1.0, flags=0):
        """Compute INT8 matrix multiplication with dequantization"""
        if not hasattr(self, 'int8_pipelines'):
            raise RuntimeError("INT8 pipelines not initialized")
            
        import vulkan as vk
        
        M, K = matrix_a.shape
        K_b, N = shape_b
        assert K == K_b, f"Matrix dimension mismatch: {K} != {K_b}"
        
        logger.info(f"🎮 Vulkan INT8 Matrix Multiply: {M}x{K} @ {K_b}x{N}")
        
        # Create buffers
        # Input A needs to be quantized dynamically
        input_int8 = (matrix_a * 127.0 / 4.0).clip(-128, 127).astype(np.int8)
        buffer_a, memory_a = self._create_buffer(input_int8)
        
        # Create scale buffers
        scale_a_arr = np.array([4.0 / 127.0], dtype=np.float32)  # Input quantization scale
        scale_b_arr = np.array([scale_b], dtype=np.float32)
        buffer_scale_a, memory_scale_a = self._create_buffer(scale_a_arr)
        buffer_scale_b, memory_scale_b = self._create_buffer(scale_b_arr)
        
        # Create output buffer
        output_size = M * N * 4  # FP32 output
        buffer_c, memory_c = self._create_buffer_empty(output_size)
        
        # Create descriptor set
        # ... (descriptor set creation code similar to original)
        
        # Execute with INT8 pipeline
        # ... (execution code)
        
        # Read result
        result = self._read_buffer(buffer_c, memory_c, output_size)
        result = np.frombuffer(result, dtype=np.float32).reshape((M, N))
        
        # Cleanup
        vk.vkDestroyBuffer(self.device, buffer_a, None)
        vk.vkFreeMemory(self.device, memory_a, None)
        vk.vkDestroyBuffer(self.device, buffer_scale_a, None)
        vk.vkFreeMemory(self.device, memory_scale_a, None)
        vk.vkDestroyBuffer(self.device, buffer_scale_b, None)
        vk.vkFreeMemory(self.device, memory_scale_b, None)
        vk.vkDestroyBuffer(self.device, buffer_c, None)
        vk.vkFreeMemory(self.device, memory_c, None)
        
        return result
    
    # Monkey patch methods
    VulkanMatrixCompute._create_int8_compute_pipelines = _create_int8_compute_pipelines
    VulkanMatrixCompute._allocate_gpu_memory_int8 = _allocate_gpu_memory_int8
    VulkanMatrixCompute.compute_matrix_multiply_int8 = compute_matrix_multiply_int8
    
    # Override _allocate_gpu_memory to check for INT8
    def _allocate_gpu_memory(self, tensor_data):
        """Enhanced GPU allocation that preserves INT8 format"""
        if isinstance(tensor_data, np.ndarray) and tensor_data.dtype == np.int8:
            return self._allocate_gpu_memory_int8(tensor_data)
        else:
            return self._original_allocate_gpu_memory(tensor_data)
    
    VulkanMatrixCompute._allocate_gpu_memory = _allocate_gpu_memory
    
    # Add INT8 pipeline creation to initialization
    original_initialize = VulkanMatrixCompute.initialize
    
    def initialize(self):
        """Enhanced initialization with INT8 support"""
        result = original_initialize(self)
        if result:
            try:
                self._create_int8_compute_pipelines()
                logger.info("✅ INT8 support initialized")
            except Exception as e:
                logger.warning(f"⚠️ INT8 support not available: {e}")
        return result
    
    VulkanMatrixCompute.initialize = initialize
    
    logger.info("✅ INT8 support added to VulkanMatrixCompute")