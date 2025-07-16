// Qwen Rotary Position Embedding (RoPE) Kernel for NPU Phoenix
// Optimized for Qwen's RoPE attention mechanism

module {
  func.func @qwen_rope_attention_npu(
    %query: tensor<?x?x?xi4>,      // [batch, seq_len, head_dim]
    %key: tensor<?x?x?xi4>,        // [batch, seq_len, head_dim]
    %rope_cos: tensor<?x?xf16>,    // Cosine values for RoPE
    %rope_sin: tensor<?x?xf16>     // Sine values for RoPE
  ) -> (tensor<?x?x?xi4>, tensor<?x?x?xi4>) {
    
    // Apply RoPE to query
    %q_rotated = npu.apply_rope_int4 %query, %rope_cos, %rope_sin : 
      tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?x?xi4>
    
    // Apply RoPE to key
    %k_rotated = npu.apply_rope_int4 %key, %rope_cos, %rope_sin : 
      tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16> -> tensor<?x?x?xi4>
    
    return %q_rotated, %k_rotated : tensor<?x?x?xi4>, tensor<?x?x?xi4>
  }
  
  // Qwen-specific attention with RoPE
  func.func @qwen_attention_with_rope_npu(
    %input: tensor<?x?x?xi4>,
    %q_weight: tensor<?x?xi4>,
    %k_weight: tensor<?x?xi4>, 
    %v_weight: tensor<?x?xi4>,
    %o_weight: tensor<?x?xi4>,
    %rope_cos: tensor<?x?xf16>,
    %rope_sin: tensor<?x?xf16>
  ) -> tensor<?x?x?xi4> {
    
    // Project to Q, K, V
    %query = npu.int4_linear %input, %q_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %key = npu.int4_linear %input, %k_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    %value = npu.int4_linear %input, %v_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply RoPE
    %q_rope, %k_rope = func.call @qwen_rope_attention_npu(%query, %key, %rope_cos, %rope_sin) : 
      (tensor<?x?x?xi4>, tensor<?x?x?xi4>, tensor<?x?xf16>, tensor<?x?xf16>) -> 
      (tensor<?x?x?xi4>, tensor<?x?x?xi4>)
    
    // Standard attention computation
    %scores = npu.int4_matmul %q_rope, %k_rope transpose_b : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    %attention_weights = npu.softmax_int4 %scores : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    %attended = npu.int4_matmul %attention_weights, %value : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Output projection
    %output = npu.int4_linear %attended, %o_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
