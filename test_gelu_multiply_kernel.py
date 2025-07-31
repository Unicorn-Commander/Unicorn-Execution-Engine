import pyopencl as cl
import numpy as np
import time
from scipy.special import erf

# Model configurations
FF_DIM_4B = 10240
FF_DIM_27B = 18432

def get_kernel_execution_time(event):
    return (event.profile.end - event.profile.start) * 1e-6 # ns to ms

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2)))

def test_gelu_multiply_kernel(ff_dim, batch_size=1, seq_len=128):
    print(f"\n--- Testing GELU Multiply Kernel (ff_dim={ff_dim}) ---")

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Load and build kernel
    with open("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/gelu_multiply_kernel.cl", "r") as f:
        kernel_code = f.read()
    
    build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math"
    prg = cl.Program(ctx, kernel_code).build(options=build_options)

    # Input data
    M = batch_size * seq_len
    N_gate_up = 2 * ff_dim
    num_elements = M * ff_dim

    gate_up_output_data = np.random.randn(M, N_gate_up).astype(np.float32)

    # --- Reference implementation (NumPy) ---
    gate_proj_ref = gate_up_output_data[:, :ff_dim]
    up_proj_ref = gate_up_output_data[:, ff_dim:]
    ref_output = gelu(gate_proj_ref) * up_proj_ref # Re-enable GELU

    # --- Kernel implementation ---
    output_data = np.empty((M, ff_dim), dtype=np.float32)

    # Create buffers
    gate_up_output_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=gate_up_output_data)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_data.nbytes)

    # Execute kernel
    global_size = (num_elements,)
    
    # Timed execution
    num_iterations = 100
    evt = prg.gelu_multiply_kernel(queue, global_size, None, gate_up_output_buf, output_buf, np.int32(num_elements), np.int32(ff_dim))
    evt.wait()

    # Read back the result
    cl.enqueue_copy(queue, output_data, output_buf).wait()

    # Verification
    np.testing.assert_allclose(ref_output, output_data, rtol=1e-3, atol=1e-3)
    print("Verification successful!")

if __name__ == "__main__":
    test_gelu_multiply_kernel(FF_DIM_4B)
    test_gelu_multiply_kernel(FF_DIM_27B)
