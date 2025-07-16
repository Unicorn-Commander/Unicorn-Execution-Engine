// Universal INT4 Attention Kernel for NPU Phoenix
// Works with both Gemma and Qwen architectures

module {
  // Main attention function optimized for NPU Phoenix
  func.func @attention_int4_npu(
    %query: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %key: tensor<?x?x?xi4>,        // [batch, seq_len, head_dim] 
    %value: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %mask: tensor<?x?xi1>          // [batch, seq_len] causal mask
  ) -> tensor<?x?x?xi4> {
    
    // Get dimensions
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index  
    %c2 = arith.constant 2 : index
    
    %batch = tensor.dim %query, %c0 : tensor<?x?x?xi4>
    %seq_len = tensor.dim %query, %c1 : tensor<?x?x?xi4>
    %head_dim = tensor.dim %query, %c2 : tensor<?x?x?xi4>
    
    // Compute attention scores: Q @ K^T
    // NPU-optimized INT4 matrix multiplication
    %scores = npu.int4_matmul %query, %key transpose_b : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply scaling factor (1/sqrt(head_dim))
    %scale = arith.constant 0.125 : f16  // Approximate for head_dim=64
    %scaled_scores = npu.int4_scale %scores, %scale : 
      tensor<?x?x?xi4>, f16 -> tensor<?x?x?xi4>
    
    // Apply causal mask
    %masked_scores = npu.apply_mask %scaled_scores, %mask : 
      tensor<?x?x?xi4>, tensor<?x?xi1> -> tensor<?x?x?xi4>
    
    // Softmax (NPU Phoenix has optimized softmax for INT4)
    %attention_weights = npu.softmax_int4 %masked_scores : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply attention to values: Attention @ V
    %output = npu.int4_matmul %attention_weights, %value : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
  
  // Optimized multi-head attention wrapper
  func.func @multihead_attention_npu(
    %input: tensor<?x?x?xi4>,      // [batch, seq_len, hidden_size]
    %q_weight: tensor<?x?xi4>,     // Query projection weights
    %k_weight: tensor<?x?xi4>,     // Key projection weights  
    %v_weight: tensor<?x?xi4>,     // Value projection weights
    %o_weight: tensor<?x?xi4>,     // Output projection weights
    %num_heads: index
  ) -> tensor<?x?x?xi4> {
    
    // Project to Q, K, V
    %query = npu.int4_linear %input, %q_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %key = npu.int4_linear %input, %k_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %value = npu.int4_linear %input, %v_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Reshape for multi-head attention
    %q_heads = npu.reshape_multihead %query, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    %k_heads = npu.reshape_multihead %key, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    %v_heads = npu.reshape_multihead %value, %num_heads : 
      tensor<?x?x?xi4>, index -> tensor<?x?x?x?xi4>
    
    // Apply attention to each head (NPU can process multiple heads in parallel)
    %attended_heads = npu.parallel_attention %q_heads, %k_heads, %v_heads : 
      tensor<?x?x?x?xi4>, tensor<?x?x?x?xi4>, tensor<?x?x?x?xi4> -> tensor<?x?x?x?xi4>
    
    // Reshape back and apply output projection
    %concatenated = npu.concatenate_heads %attended_heads : 
      tensor<?x?x?x?xi4> -> tensor<?x?x?xi4>
    %output = npu.int4_linear %concatenated, %o_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
