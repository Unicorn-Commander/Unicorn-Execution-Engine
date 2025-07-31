import pyopencl as cl
import numpy as np
import time

# Model configurations
HIDDEN_SIZE_4B = 2560
NUM_HEADS_4B = 20
HEAD_DIM_4B = 128

HIDDEN_SIZE_27B = 4608
NUM_HEADS_27B = 32
HEAD_DIM_27B = 144

def get_kernel_execution_time(event):
    return (event.profile.end - event.profile.start) * 1e-6 # ns to ms

def test_attention_softmax_fusion(num_heads, head_dim, seq_len=128):
    print(f"\n--- Testing Attention+Softmax Fusion (heads={num_heads}, head_dim={head_dim}) ---")

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Load and build the kernel
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/attention_softmax_fused.cl", "r") as f:
        kernel_code = f.read()
    build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math"
    prg = cl.Program(ctx, kernel_code).build(options=build_options)

    # Input data
    Q = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    # --- Reference implementation (NumPy) ---
    start_time = time.time()
    attention_scores = (Q @ K.transpose(0, 2, 1)) * scale
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    attention_scores[:, mask] = -np.inf
    ref_softmax = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
    ref_softmax /= np.sum(ref_softmax, axis=-1, keepdims=True)
    ref_time = time.time() - start_time
    print(f"Reference (NumPy) time: {ref_time * 1000:.4f} ms")

    # --- Fused kernel implementation ---
    attention_weights_fused = np.empty((num_heads, seq_len, seq_len), dtype=np.float32)

    # Create buffers
    q_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Q)
    k_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=K)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, attention_weights_fused.nbytes)

    # Execute kernel
    global_size = (num_heads, seq_len)
    
    # Warmup
    prg.attention_softmax_fused(queue, global_size, None, q_buf, k_buf, output_buf, np.int32(num_heads), np.int32(seq_len), np.int32(head_dim), np.float32(scale))
    queue.finish()

    # Timed execution
    num_iterations = 100
    total_fused_time = 0
    for _ in range(num_iterations):
        evt = prg.attention_softmax_fused(queue, global_size, None, q_buf, k_buf, output_buf, np.int32(num_heads), np.int32(seq_len), np.int32(head_dim), np.float32(scale))
        evt.wait()
        total_fused_time += get_kernel_execution_time(evt)
    
    avg_fused_time = total_fused_time / num_iterations
    print(f"Fused kernel GPU time: {avg_fused_time:.4f} ms")

    # Read back the result
    cl.enqueue_copy(queue, attention_weights_fused, output_buf).wait()

    # Verification
    np.testing.assert_allclose(ref_softmax, attention_weights_fused, rtol=1e-5, atol=1e-5)
    print("Verification successful!")

if __name__ == "__main__":
    test_attention_softmax_fusion(NUM_HEADS_4B, HEAD_DIM_4B)
    test_attention_softmax_fusion(NUM_HEADS_27B, HEAD_DIM_27B)
