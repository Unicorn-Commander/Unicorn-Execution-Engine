import pyopencl as cl
import numpy as np
import time
from phase1_fused_pipeline import Phase1FusedPipeline

# Model configurations (Gemma 3)
HIDDEN_SIZE_4B = 2560
NUM_HEADS_4B = 20
HEAD_DIM_4B = 128  # HIDDEN_SIZE_4B / NUM_HEADS_4B
FF_DIM_4B = 10240

# Create dummy input data and weights
batch_seq_len = 128
hidden_size = HIDDEN_SIZE_4B
ff_dim = FF_DIM_4B

hidden_states = np.random.randn(batch_seq_len, hidden_size).astype(np.float32)
W_qkv = np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32)
W_gate = np.random.randn(hidden_size, ff_dim).astype(np.float32)
W_up = np.random.randn(hidden_size, ff_dim).astype(np.float32)
W_down = np.random.randn(ff_dim, hidden_size).astype(np.float32)

print("Initializing Phase1FusedPipeline...")
pipeline = Phase1FusedPipeline(model_type="4b")
print("Pipeline initialized.")

print("Running forward pass...")
start_time = time.time()
output = pipeline.forward_layer(hidden_states, W_qkv, W_gate, W_up, W_down)
end_time = time.time()
print(f"Forward pass complete in {(end_time - start_time) * 1000:.2f} ms.")
print("Output shape:", output.shape)

# Add basic correctness check (e.g., compare with a simple CPU reference if possible)
# For a full benchmark, use benchmark_fusion.py

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

# You would need to extract the final hidden states from the full model output
# For this test, we're just checking the MLP output from the pipeline

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
