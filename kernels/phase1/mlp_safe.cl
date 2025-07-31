// Safe MLP Implementation - Avoids segfault issues
// Single kernel approach with minimal intermediate buffers

__kernel void mlp_safe_single(
    __global const float* restrict input,      // [M, hidden_size]
    __global const float* restrict W_gate,     // [hidden_size, ff_dim]
    __global const float* restrict W_up,       // [hidden_size, ff_dim]  
    __global const float* restrict W_down,     // [ff_dim, hidden_size]
    __global float* restrict output,           // [M, hidden_size]
    const int M,
    const int hidden_size,
    const int ff_dim
) {
    // Each work item computes one output element
    int gid = get_global_id(0);
    if (gid >= M * hidden_size) return;
    
    int row = gid / hidden_size;
    int col = gid % hidden_size;
    
    // Bounds checking
    if (row >= M || col >= hidden_size) return;
    
    float accumulator = 0.0f;
    
    // Process in chunks to avoid register pressure
    const int CHUNK_SIZE = 64;
    
    for (int k_start = 0; k_start < ff_dim; k_start += CHUNK_SIZE) {
        int k_end = min(k_start + CHUNK_SIZE, ff_dim);
        
        for (int k = k_start; k < k_end; k++) {
            // Compute gate projection: input[row,:] @ W_gate[:,k]
            float gate_val = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                gate_val += input[row * hidden_size + j] * W_gate[j * ff_dim + k];
            }
            
            // Compute up projection: input[row,:] @ W_up[:,k]  
            float up_val = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                up_val += input[row * hidden_size + j] * W_up[j * ff_dim + k];
            }
            
            // Apply GELU activation to gate
            float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_val));
            float activated = gate_val * sigmoid;
            
            // Element-wise multiply with up
            float intermediate = activated * up_val;
            
            // Down projection: intermediate * W_down[k,col]
            accumulator += intermediate * W_down[k * hidden_size + col];
        }
    }
    
    output[row * hidden_size + col] = accumulator;
}

// Debug kernel to check memory access patterns
__kernel void debug_mlp_memory(
    __global const float* input,
    __global const float* W_gate,
    __global const float* W_up,
    __global const float* W_down,
    __global float* output,
    const int M,
    const int hidden_size, 
    const int ff_dim
) {
    int gid = get_global_id(0);
    
    // Only first work item does debugging
    if (gid != 0) return;
    
    printf("MLP Debug - Kernel started\n");
    printf("M=%d, hidden_size=%d, ff_dim=%d\n", M, hidden_size, ff_dim);
    printf("Expected input size: %d elements\n", M * hidden_size);
    printf("Expected output size: %d elements\n", M * hidden_size);
    
    // Test first element access
    if (M > 0 && hidden_size > 0) {
        float test_input = input[0];
        printf("First input element: %f\n", test_input);
    }
    
    // Test weight matrix access
    if (hidden_size > 0 && ff_dim > 0) {
        float test_gate = W_gate[0];
        float test_up = W_up[0]; 
        float test_down = W_down[0];
        printf("First weight elements - gate:%f, up:%f, down:%f\n", 
               test_gate, test_up, test_down);
    }
    
    printf("MLP Debug - Memory access test passed\n");
}

// Conservative three-kernel approach with extra safety
__kernel void gate_up_safe(
    __global const float* restrict input,
    __global const float* restrict W_gate,
    __global const float* restrict W_up,
    __global float* restrict gate_out,      // [M, ff_dim]
    __global float* restrict up_out,        // [M, ff_dim] 
    const int M,
    const int hidden_size,
    const int ff_dim
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // Conservative bounds checking
    if (row >= M || col >= ff_dim) return;
    if (row < 0 || col < 0) return;
    
    // Compute gate projection
    float gate_sum = 0.0f;
    for (int k = 0; k < hidden_size; k++) {
        gate_sum += input[row * hidden_size + k] * W_gate[k * ff_dim + col];
    }
    gate_out[row * ff_dim + col] = gate_sum;
    
    // Compute up projection
    float up_sum = 0.0f;
    for (int k = 0; k < hidden_size; k++) {
        up_sum += input[row * hidden_size + k] * W_up[k * ff_dim + col];
    }
    up_out[row * ff_dim + col] = up_sum;
}

__kernel void gelu_multiply_safe(
    __global const float* restrict gate,     // [M, ff_dim]
    __global const float* restrict up,       // [M, ff_dim]
    __global float* restrict output,         // [M, ff_dim]
    const int M,
    const int ff_dim
) {
    int gid = get_global_id(0);
    if (gid >= M * ff_dim) return;
    if (gid < 0) return;
    
    int row = gid / ff_dim;
    int col = gid % ff_dim;
    
    if (row >= M || col >= ff_dim) return;
    
    float gate_val = gate[row * ff_dim + col];
    float up_val = up[row * ff_dim + col];
    
    // GELU approximation
    float sigmoid = 1.0f / (1.0f + exp(-1.702f * gate_val));
    float gelu_result = gate_val * sigmoid;
    
    output[row * ff_dim + col] = gelu_result * up_val;
}

__kernel void down_projection_safe(
    __global const float* restrict input,    // [M, ff_dim]
    __global const float* restrict W_down,   // [ff_dim, hidden_size]
    __global float* restrict output,         // [M, hidden_size]
    const int M,
    const int ff_dim,
    const int hidden_size
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= M || col >= hidden_size) return;
    if (row < 0 || col < 0) return;
    
    float sum = 0.0f;
    for (int k = 0; k < ff_dim; k++) {
        sum += input[row * ff_dim + k] * W_down[k * hidden_size + col];
    }
    
    output[row * hidden_size + col] = sum;
}