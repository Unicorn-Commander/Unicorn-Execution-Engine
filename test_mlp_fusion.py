import pyopencl as cl
import numpy as np
import time
# from scipy.special import erf # Removed for simpler GELU

# Model configurations
HIDDEN_SIZE_4B = 2560
FF_DIM_4B = 10240

HIDDEN_SIZE_27B = 4608
FF_DIM_27B = 18432

def get_kernel_execution_time(event):
    return (event.profile.end - event.profile.start) * 1e-6 # ns to ms

def sigmoid(x):
    # Clamp input to prevent overflow/underflow in exp
    x = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x))

def gelu(x):
    return x * sigmoid(1.702 * x) # Approximation: x * sigmoid(1.702 * x)

def test_mlp_fusion(hidden_size, ff_dim, batch_size=1, seq_len=128):
    print(f"\n--- Testing MLP Fusion (hidden={hidden_size}, ff_dim={ff_dim}) ---")

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Load and build kernels
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/gate_up_fused.cl", "r") as f:
        gate_up_kernel_code = f.read()
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/gelu_down_fused.cl", "r") as f:
        gelu_down_kernel_code = f.read()
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/gelu_multiply_kernel.cl", "r") as f:
        gelu_multiply_kernel_code = f.read()
    
    build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math"
    gate_up_prg = cl.Program(ctx, gate_up_kernel_code).build(options=build_options)
    gelu_down_prg = cl.Program(ctx, gelu_down_kernel_code).build(options=build_options)
    gelu_multiply_prg = cl.Program(ctx, gelu_multiply_kernel_code).build(options=build_options)

    # Input data
    M = batch_size * seq_len
    K_hidden = hidden_size
    K_ff = ff_dim
    N_gate_up = 2 * K_ff
    N_down = hidden_size

    input_data = np.random.randn(M, K_hidden).astype(np.float32)
    W_gate_up = np.random.randn(K_hidden, N_gate_up).astype(np.float32)
    W_down = np.random.randn(K_ff, N_down).astype(np.float32)

    # --- Reference implementation (NumPy) ---
    gate_up_ref = input_data @ W_gate_up
    gate_proj_ref = gate_up_ref[:, :K_ff]
    up_proj_ref = gate_up_ref[:, K_ff:]
    activated_ref = gelu(gate_proj_ref) * up_proj_ref
    ref_output = activated_ref @ W_down

    # --- Fused kernel implementation (3 kernels) ---
    gate_up_output = np.empty((M, N_gate_up), dtype=np.float32)
    activated_output = np.empty((M, K_ff), dtype=np.float32)
    fused_output = np.empty((M, N_down), dtype=np.float32)

    # Create buffers
    input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
    w_gate_up_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_gate_up)
    w_down_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_down)
    gate_up_out_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, gate_up_output.nbytes)
    activated_out_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, activated_output.nbytes)
    final_out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, fused_output.nbytes)

    # Execute kernels
    local_size = (16, 16)
    gate_up_global_size = (M, N_gate_up)
    gelu_multiply_global_size = (M * K_ff,)
    gelu_down_global_size = (M, N_down)

    gate_up_global_size = (
        ((gate_up_global_size[0] + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((gate_up_global_size[1] + local_size[1] - 1) // local_size[1]) * local_size[1]
    )
    gelu_down_global_size = (
        ((gelu_down_global_size[0] + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((gelu_down_global_size[1] + local_size[1] - 1) // local_size[1]) * local_size[1]
    )

    # Timed execution
    num_iterations = 10
    evt1 = gate_up_prg.gate_up_fused(queue, gate_up_global_size, local_size, np.int32(M), np.int32(N_gate_up), np.int32(K_hidden), input_buf, w_gate_up_buf, gate_up_out_buf)
    evt2 = gelu_multiply_prg.gelu_multiply_kernel(queue, gelu_multiply_global_size, None, gate_up_out_buf, activated_out_buf, np.int32(M * K_ff), np.int32(K_ff), wait_for=[evt1])
    evt3 = gelu_down_prg.gelu_down_fused(queue, gelu_down_global_size, local_size, np.int32(M), np.int32(N_down), np.int32(K_ff), activated_out_buf, w_down_buf, final_out_buf, wait_for=[evt2])
    evt3.wait()

    # Read back the result
    cl.enqueue_copy(queue, fused_output, final_out_buf).wait()

    # Verification
    np.testing.assert_allclose(ref_output, fused_output, rtol=1e-3, atol=1e-3)
    print("Verification successful!")

if __name__ == "__main__":
    test_mlp_fusion(HIDDEN_SIZE_4B, FF_DIM_4B)
    test_mlp_fusion(HIDDEN_SIZE_27B, FF_DIM_27B)