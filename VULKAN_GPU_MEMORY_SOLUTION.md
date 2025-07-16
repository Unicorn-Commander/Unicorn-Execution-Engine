# Vulkan GPU Memory Allocation Solution

## The Issue Resolved

The model was loading to system RAM because the Vulkan implementation was using `HOST_VISIBLE | HOST_COHERENT` memory, which allocates in system RAM, not GPU VRAM.

## The Solution

### Key Discovery: Memory Types in Vulkan

```
Type 0: Heap 1 (36.0GB) - DEVICE_LOCAL              ← This is VRAM!
Type 2: Heap 0 (18.0GB) - HOST_VISIBLE, HOST_COHERENT  ← This is system RAM
Type 3: Heap 1 (36.0GB) - DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT  ← This is GTT!
```

### To Allocate to VRAM:
Use `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`:
```cpp
vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  // VRAM allocation
```

### To Allocate to GTT:
Use `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`:
```cpp
vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  // GTT allocation
```

### To Allocate to System RAM:
Use `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`:
```cpp
vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  // System RAM (current implementation)
```

## Verification

The test shows VRAM allocation working:
- Initial VRAM: 1.1GB used
- After 1GB allocation: 2.2GB used ✅

## Implementation Steps

1. **Modify `real_vulkan_matrix_compute.py`** to use DEVICE_LOCAL memory:
   ```python
   # Change from:
   memory_type_index = self._find_memory_type(
       mem_requirements.memoryTypeBits, 
       vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
   )
   
   # To:
   memory_type_index = self._find_memory_type(
       mem_requirements.memoryTypeBits, 
       vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  # For VRAM
   )
   ```

2. **Use staging buffers** for VRAM allocation:
   - Create HOST_VISIBLE buffer for staging
   - Copy data to staging buffer
   - Create DEVICE_LOCAL buffer in VRAM
   - Use vkCmdCopyBuffer to transfer from staging to VRAM

3. **Memory hierarchy**:
   - Critical layers → VRAM (DEVICE_LOCAL)
   - Bulk layers → GTT (DEVICE_LOCAL + HOST_VISIBLE)
   - Overflow → System RAM (HOST_VISIBLE)

## Summary

The pure hardware pipeline concept is correct - we don't need PyTorch/ROCm. We just need to:
1. Use the correct Vulkan memory flags
2. Implement staging buffers for VRAM transfers
3. The MLIR-AIE2 NPU kernels and Vulkan shaders remain the same

This achieves the original goal of bypassing ML frameworks while actually using GPU memory!