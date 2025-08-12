import pyopencl as cl
import numpy as np
import time

# Model configurations
HIDDEN_SIZE_4B = 2560
HIDDEN_SIZE_27B = 4608

def get_kernel_execution_time(event):
    return (event.profile.end - event.profile.start) * 1e-6 # ns to ms

def test_qkv_fusion(hidden_size, batch_size=1, seq_len=128):
    print(f"\n--- Testing QKV Fusion for model with hidden_size={hidden_size} ---")

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Load and build kernels
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/qkv_fused.cl", "r") as f:
        fused_kernel_code = f.read()
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/gemm_unfused.cl", "r") as f:
        unfused_kernel_code = f.read()
    
    build_options = "-cl-std=CL2.0"
    fused_prg = cl.Program(ctx, fused_kernel_code).build(options=build_options)
    unfused_prg = cl.Program(ctx, unfused_kernel_code).build(options=build_options)

    # Input data
    M = batch_size * seq_len
    N = 3 * hidden_size
    K = hidden_size

    input_data = np.eye(M, K, dtype=np.float32)
    W_q = np.eye(K, hidden_size, dtype=np.float32)
    W_k = np.eye(K, hidden_size, dtype=np.float32)
    W_v = np.eye(K, hidden_size, dtype=np.float32)
    W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)

    # --- Reference implementation (3 separate GPU kernels) ---
    Q_ref = input_data @ W_q
    K_ref = input_data @ W_k
    V_ref = input_data @ W_v

    # --- Fused kernel implementation ---
    output_fused = np.empty((M, N), dtype=np.float32)

    # Create buffers
    input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
    w_qkv_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_qkv)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_fused.nbytes)

    # Execute kernel
    local_size = (16, 16)
    global_size = (M, N)
    global_size = (
        ((global_size[0] + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((global_size[1] + local_size[1] - 1) // local_size[1]) * local_size[1]
    )

    # Timed execution
    num_iterations = 100
    evt = fused_prg.qkv_projection_fused(queue, global_size, local_size, input_buf, w_qkv_buf, output_buf, np.int32(M), np.int32(N), np.int32(K))
    evt.wait()
    
    # Read back the result
    cl.enqueue_copy(queue, output_fused, output_buf).wait()

    # Verification
    Q_fused = output_fused[:, :hidden_size]
    K_fused = output_fused[:, hidden_size:2*hidden_size]
    V_fused = output_fused[:, 2*hidden_size:]

    np.testing.assert_allclose(Q_ref, Q_fused, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(K_ref, K_fused, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(V_ref, V_fused, rtol=1e-4, atol=1e-4)
    print("Verification successful!")

if __name__ == "__main__":
    test_qkv_fusion(HIDDEN_SIZE_4B)
    test_qkv_fusion(HIDDEN_SIZE_27B)
