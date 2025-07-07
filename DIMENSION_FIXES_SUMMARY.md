# Gemma3n E2B Weight Dimension Fixes Summary

## Issues Identified

### 1. Matrix Multiplication Dimension Mismatch
**Error**: `Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 512 is different from 2048)`

**Root Cause**: The safetensors weights are stored in format `(output_dim, input_dim)` but the code expected `(input_dim, output_dim)` for direct matrix multiplication.

**Solution**: Transpose all weight matrices during extraction to convert from `(output_dim, input_dim)` to `(input_dim, output_dim)`.

### 2. Embedding Dimension Mismatch  
**Error**: `operands could not be broadcast together with shapes (1,5,7680) (2048,)`

**Root Cause**: The model has two embedding matrices:
- `model.language_model.embed_tokens.weight`: (262400, 2048) - standard embedding
- `model.language_model.embed_tokens_per_layer.weight`: (262144, 7680) - per-layer embedding

The code was using the wrong embedding matrix (7680 dim instead of 2048 dim).

**Solution**: Filter to use only the standard embedding matrix, excluding the per-layer variant.

## Fixes Implemented

### File: `direct_safetensors_loader.py`

#### 1. Embedding Weight Selection Fix
```python
# Extract embeddings (use standard embedding, not per-layer)
for name, tensor in organized['embeddings'].items():
    if 'embed_tokens' in name and 'per_layer' not in name:
        npu_weights['embedding'] = tensor.astype(np.float32)
        logger.info(f"✅ Embedding: {tensor.shape}")
        break  # Use first standard embedding found
```

#### 2. Attention Weight Transpose Fix
```python
if 'self_attn.q_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['attention']['q_proj'] = tensor_f32.T
elif 'self_attn.k_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['attention']['k_proj'] = tensor_f32.T
elif 'self_attn.v_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['attention']['v_proj'] = tensor_f32.T
elif 'self_attn.o_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['attention']['o_proj'] = tensor_f32.T
```

#### 3. MLP Weight Transpose Fix
```python
elif 'mlp.gate_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['mlp']['gate_proj'] = tensor_f32.T
elif 'mlp.up_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['mlp']['up_proj'] = tensor_f32.T
elif 'mlp.down_proj' in name:
    # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
    layer_info['mlp']['down_proj'] = tensor_f32.T
```

#### 4. Output Projection Fix with Tied Weights Fallback
```python
# Extract output projection (look for lm_head or use embedding weights transposed)
output_found = False
for name, tensor in organized['output'].items():
    if 'lm_head' in name:
        # Transpose weight for correct matrix multiplication: (input_dim, output_dim)
        npu_weights['output_projection'] = tensor.astype(np.float32).T
        logger.info(f"✅ Output projection: {tensor.shape} -> {tensor.T.shape}")
        output_found = True
        break

# If no lm_head found, use embedding weights transposed (tied weights)
if not output_found and npu_weights['embedding'] is not None:
    npu_weights['output_projection'] = npu_weights['embedding'].T
    logger.info(f"✅ Output projection (tied weights): {npu_weights['embedding'].T.shape}")
```

### File: `direct_npu_attention.py`

#### 1. KV Dimension Calculation Fix
```python
def __init__(self, hidden_size: int = 2048, num_heads: int = 8, num_kv_heads: int = 2):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = hidden_size // num_heads
    self.kv_head_dim = hidden_size // num_heads  # Same head dimension for both Q and KV
    self.kv_dim = self.kv_head_dim * num_kv_heads  # Total KV dimension
```

#### 2. Test Weight Generation Fix
```python
# Generate test weights for attention (accounting for KV heads)
# Weights are in the correct format: (input_dim, output_dim) after transpose
kv_dim = accelerator.npu_attention.kv_dim
attention_weights = {
    'q_proj': np.random.randn(hidden_size, hidden_size).astype(np.float32),
    'k_proj': np.random.randn(hidden_size, kv_dim).astype(np.float32),
    'v_proj': np.random.randn(hidden_size, kv_dim).astype(np.float32),
    'o_proj': np.random.randn(hidden_size, hidden_size).astype(np.float32)
}
```

## Correct Weight Dimensions

After the fixes, the weight dimensions are:

### Language Model Configuration
- **Hidden size**: 2048
- **Attention heads**: 8
- **KV heads**: 2
- **Head dimension**: 256 (2048 / 8)
- **KV dimension**: 512 (256 * 2)

### Weight Matrix Dimensions (after transpose)
- **Q projection**: 2048 x 2048 (input_dim x output_dim)
- **K projection**: 2048 x 512 (input_dim x kv_dim)
- **V projection**: 2048 x 512 (input_dim x kv_dim)
- **O projection**: 2048 x 2048 (input_dim x output_dim)
- **Embedding**: 262400 x 2048 (vocab_size x hidden_size)
- **Output projection**: 2048 x 262400 (hidden_size x vocab_size)

### Matrix Multiplication Flow
```
hidden_states: (batch_size, seq_len, 2048)
q_proj:        (2048, 2048)
k_proj:        (2048, 512)
v_proj:        (2048, 512)
o_proj:        (2048, 2048)

Computation:
query = hidden_states @ q_proj  -> (batch_size, seq_len, 2048)
key   = hidden_states @ k_proj  -> (batch_size, seq_len, 512)
value = hidden_states @ v_proj  -> (batch_size, seq_len, 512)
output = attention_output @ o_proj -> (batch_size, seq_len, 2048)
```

## Validation Results

✅ **Weight extraction**: All weights now have correct dimensions
✅ **Matrix multiplication**: No more dimension mismatch errors
✅ **Embedding lookup**: Using correct 2048-dimensional embedding
✅ **Forward pass**: Successfully processes input through layers
✅ **Output shape**: Correct vocabulary size (262400) in final output

## Files Modified

1. `/home/ucadmin/Development/Unicorn-Execution-Engine/direct_safetensors_loader.py`
2. `/home/ucadmin/Development/Unicorn-Execution-Engine/direct_npu_attention.py`
3. `/home/ucadmin/Development/Unicorn-Execution-Engine/integrated_acceleration_engine.py`

The dimension fixes ensure proper integration of the Gemma3n E2B model with the NPU acceleration framework.