# 🔍 Debugging OpenCL Segmentation Fault in MLP Pipeline

## The Issue
The MLP kernels work in isolation but cause segfault in the full pipeline. This suggests:

1. **Buffer Reuse Issue**: Buffers from previous operations might be in an inconsistent state
2. **Memory Alignment**: Larger allocations in pipeline might have different alignment
3. **Queue Synchronization**: Race condition between kernels
4. **Buffer Size Calculation**: Off-by-one errors that only manifest with certain sizes

## Immediate Debugging Steps

### 1. Add Explicit Synchronization
```python
# After EACH kernel call in the pipeline:
event.wait()
queue.finish()  # Force complete synchronization
```

### 2. Check Buffer States
```python
# Before MLP block, verify buffer contents:
def verify_buffer(buf, expected_size, name):
    test_data = np.zeros(expected_size, dtype=np.float32)
    try:
        cl.enqueue_copy(queue, test_data, buf)
        queue.finish()
        print(f"✓ {name} buffer readable, size: {expected_size}")
    except Exception as e:
        print(f"✗ {name} buffer error: {e}")
```

### 3. Guard Against Edge Cases
The kernels might have issues with:
- Non-multiple of 16 dimensions
- Empty work items at boundaries
- Uninitialized memory

## Common OpenCL Segfault Causes

1. **Work Size Mismatch**
   ```python
   # Always check:
   assert global_size[0] * global_size[1] <= total_elements
   ```

2. **Buffer Overflow**
   ```c
   // In kernel, always guard:
   if (row >= M || col >= N) return;
   ```

3. **Local Memory Overrun**
   ```python
   # Check LDS usage:
   max_lds = device.local_mem_size
   required_lds = calculate_lds_usage()
   assert required_lds < max_lds
   ```

## Safer MLP Implementation

Try this defensive approach:

```python
# In phase1_fused_pipeline.py, replace MLP block with:

# 1. Ensure complete synchronization before MLP
queue.finish()

# 2. Create fresh buffers (don't reuse)
mlp_gate_up_buf = cl.Buffer(ctx, mf.READ_WRITE, 
                            size=int(batch_seq * 2 * ff_dim * 4))
mlp_intermediate_buf = cl.Buffer(ctx, mf.READ_WRITE, 
                                size=int(batch_seq * ff_dim * 4))
mlp_output_buf = cl.Buffer(ctx, mf.READ_WRITE, 
                          size=int(batch_seq * hidden_size * 4))

# 3. Use defensive global sizes
gate_up_global = (int(batch_seq), int(2 * ff_dim))
gelu_global = (int(batch_seq * ff_dim),)
down_global = (int(batch_seq), int(hidden_size))

# 4. Set local_size explicitly to None
local_size = None  # Let OpenCL choose

# 5. Add error checking after each kernel
try:
    event1 = program.gate_up_fused_fixed(...)
    event1.wait()
    queue.finish()
    print("✓ Gate-up kernel completed")
except Exception as e:
    print(f"✗ Gate-up kernel failed: {e}")
    raise
```

## Memory Layout Debugging

Add this debug kernel to pinpoint the exact failure:

```c
__kernel void debug_memory_access(
    __global float* buffer,
    const int size
) {
    int idx = get_global_id(0);
    if (idx == 0) {
        printf("Buffer address: %p, size: %d\n", buffer, size);
        // Try to read first and last elements
        float first = buffer[0];
        float last = buffer[size-1];
        printf("First: %f, Last: %f\n", first, last);
    }
}
```

## Alternative: Simplified MLP

If segfaults persist, try this minimal MLP that avoids the 2*ff_dim complexity:

```c
__kernel void mlp_simple_safe(
    __global const float* input,      // [M, hidden_size]
    __global const float* W_gate,     // [hidden_size, ff_dim]  
    __global const float* W_up,       // [hidden_size, ff_dim]
    __global const float* W_down,     // [ff_dim, hidden_size]
    __global float* output,           // [M, hidden_size]
    __global float* temp_buffer,      // [M, ff_dim] workspace
    const int M,
    const int hidden_size,
    const int ff_dim
) {
    int gid = get_global_id(0);
    if (gid >= M * hidden_size) return;
    
    int row = gid / hidden_size;
    int col = gid % hidden_size;
    
    // For this output element, compute the full MLP
    float sum = 0.0f;
    
    // Loop over ff_dim
    for (int k = 0; k < ff_dim; k++) {
        // Compute gate[row,k]
        float gate_val = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            gate_val += input[row * hidden_size + j] * W_gate[j * ff_dim + k];
        }
        
        // Compute up[row,k]
        float up_val = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            up_val += input[row * hidden_size + j] * W_up[j * ff_dim + k];
        }
        
        // Apply GELU to gate
        float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_val));
        float gelu_gate = gate_val * sigmoid;
        
        // Accumulate down projection
        sum += gelu_gate * up_val * W_down[k * hidden_size + col];
    }
    
    output[row * hidden_size + col] = sum;
}
```

This kernel:
- Avoids the 2*ff_dim intermediate buffer
- Has simpler indexing
- Less likely to segfault

## Pipeline-Specific Issues

The segfault in pipeline but not isolation suggests:

1. **Memory Pressure**: Full pipeline uses more GPU memory
2. **Buffer Interference**: Previous kernels leave buffers in bad state
3. **Timing Issues**: Race conditions that only appear under load

Try running with:
```bash
export GPU_MAX_ALLOC_PERCENT=90
export GPU_SINGLE_ALLOC_PERCENT=90
```

## If All Else Fails

1. **Use CPU for MLP temporarily**: Get Phase 1 working with CPU MLP
2. **Binary search**: Comment out half the pipeline, narrow down the conflict
3. **Use AMD ROCm tools**: `rocprof` can show exact crash location
4. **Simplify dimensions**: Test with powers of 2 only (128, 256, 512)