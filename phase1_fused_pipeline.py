import pyopencl as cl
import numpy as np
from pathlib import Path

# Model configurations (Gemma 3)
HIDDEN_SIZE_4B = 2560
NUM_HEADS_4B = 20
HEAD_DIM_4B = 128  # HIDDEN_SIZE_4B / NUM_HEADS_4B
FF_DIM_4B = 10240

HIDDEN_SIZE_27B = 4608
NUM_HEADS_27B = 32
HEAD_DIM_27B = 144 # HIDDEN_SIZE_27B / NUM_HEADS_27B
FF_DIM_27B = 18432

class Phase1FusedPipeline:
    def __init__(self, model_type="4b"):
        self.model_type = model_type
        if model_type == "4b":
            self.hidden_size = HIDDEN_SIZE_4B
            self.num_heads = NUM_HEADS_4B
            self.head_dim = HEAD_DIM_4B
            self.ff_dim = FF_DIM_4B
        elif model_type == "27b":
            self.hidden_size = HIDDEN_SIZE_27B
            self.num_heads = NUM_HEADS_27B
            self.head_dim = HEAD_DIM_27B
            self.ff_dim = FF_DIM_27B
        else:
            raise ValueError("model_type must be '4b' or '27b'")

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable"

        self.load_fused_kernels()

    def load_fused_kernels(self):
        # Load QKV fused kernel
        qkv_kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/qkv_fused.cl")
        with open(qkv_kernel_path, 'r') as f:
            qkv_kernel_source = f.read()
        self.qkv_prg = cl.Program(self.ctx, qkv_kernel_source).build(self.build_options)

        # Load Attention+Softmax fused kernel
        attn_softmax_kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/attention_softmax_fused.cl")
        with open(attn_softmax_kernel_path, 'r') as f:
            attn_softmax_kernel_source = f.read()
        self.attn_softmax_prg = cl.Program(self.ctx, attn_softmax_kernel_source).build(self.build_options)

        # Load MLP safe kernel
        mlp_safe_kernel_path = Path("/home/ucadmin/Development/Unicorn-Execution-Engine/kernels/phase1/mlp_safe.cl")
        with open(mlp_safe_kernel_path, 'r') as f:
            mlp_safe_kernel_source = f.read()
        self.mlp_safe_prg = cl.Program(self.ctx, mlp_safe_kernel_source).build(self.build_options)

    def forward_layer(self, hidden_states_np, W_qkv_np, W_gate_np, W_up_np, W_down_np):
        # hidden_states_np: [batch_seq_len, hidden_size]
        # W_qkv_np: [hidden_size, 3 * hidden_size]
        # W_gate_np: [hidden_size, ff_dim]
        # W_up_np: [hidden_size, ff_dim]
        # W_down_np: [ff_dim, hidden_size]

        mf = cl.mem_flags
        batch_seq_len = np.int32(hidden_states_np.shape[0])

        # --- QKV Projection Fusion (Temporarily commented out for MLP debugging) ---
        # --- Attention Score + Softmax Fusion (Temporarily commented out for MLP debugging) ---

        # --- MLP Block Fusion (using mlp_safe_single) ---
        # W_gate_np: [hidden_size, ff_dim], W_up_np: [hidden_size, ff_dim]
        # W_down_np: [ff_dim, hidden_size]

        hidden_states_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hidden_states_np)
        W_gate_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_gate_np)
        W_up_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_up_np)
        W_down_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W_down_np)

        final_mlp_output_np = np.empty((batch_seq_len, self.hidden_size), dtype=np.float32)
        final_mlp_out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, final_mlp_output_np.nbytes)

        mlp_global_size = (batch_seq_len * self.hidden_size,)
        mlp_local_size = None

        self.mlp_safe_prg.mlp_safe_single(
            self.queue, mlp_global_size, mlp_local_size,
            hidden_states_buf, W_gate_buf, W_up_buf, W_down_buf, final_mlp_out_buf,
            np.int32(batch_seq_len), np.int32(self.hidden_size), np.int32(self.ff_dim)
        ).wait()

        # Return the final output of the MLP block (as an example)
        cl.enqueue_copy(self.queue, final_mlp_output_np, final_mlp_out_buf).wait()
        return final_mlp_output_np

if __name__ == "__main__":
    # Example usage:
    pipeline = Phase1FusedPipeline(model_type="4b")

    # Create dummy input data and weights
    batch_seq_len = 128
    hidden_size = pipeline.hidden_size
    ff_dim = pipeline.ff_dim

    hidden_states = np.random.randn(batch_seq_len, hidden_size).astype(np.float32)
    W_qkv = np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
    W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32)
    W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32)
    W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32)

    print("Running forward pass...")
    output = pipeline.forward_layer(hidden_states, W_qkv, W_gate, W_up, W_down)
    print("Forward pass complete. Output shape:", output.shape)

    # You can add more comprehensive testing here, comparing with a CPU reference
    # and measuring performance.

    # Example of a very basic CPU reference for the MLP part
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))

    gate_ref = np.dot(hidden_states, W_gate)
    up_ref = np.dot(hidden_states, W_up)
    intermediate_ref = gelu(gate_ref) * up_ref
    mlp_output_ref = np.dot(intermediate_ref, W_down)

    # Compare only the MLP part of the output for now
    # Note: This is a very simplified comparison and doesn't cover QKV or Attention
    # A full comparison would require running the entire reference model.
    # For now, we're just checking if the MLP output is numerically close.

    # Since forward_layer returns the MLP output, we can compare it directly
    max_error = np.max(np.abs(output - mlp_output_ref))
    mean_error = np.mean(np.abs(output - mlp_output_ref))

    print(f"\nMLP Output Correctness Check:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  Pass: {max_error < 1e-3}")

    if max_error < 1e-3:
        print("Phase 1 Pipeline MLP output is correct within tolerance.")
    else:
        print("Phase 1 Pipeline MLP output has errors.")
