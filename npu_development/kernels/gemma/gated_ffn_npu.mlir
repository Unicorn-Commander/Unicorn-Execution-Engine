// Gemma Gated FFN Kernel for NPU Phoenix
// Optimized for Gemma's SiLU + gated architecture

module {
  func.func @gemma_gated_ffn_npu(
    %input: tensor<?x?x?xi4>,      // [batch, seq_len, hidden_size]
    %gate_weight: tensor<?x?xi4>,  // Gate projection weights
    %up_weight: tensor<?x?xi4>,    // Up projection weights
    %down_weight: tensor<?x?xi4>   // Down projection weights
  ) -> tensor<?x?x?xi4> {
    
    // Gate projection
    %gate_proj = npu.int4_linear %input, %gate_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Up projection  
    %up_proj = npu.int4_linear %input, %up_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    // Apply SiLU activation to gate (NPU Phoenix has optimized SiLU)
    %gate_activated = npu.silu_int4 %gate_proj : 
      tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Element-wise multiplication (gating mechanism)
    %gated = npu.int4_mul %gate_activated, %up_proj : 
      tensor<?x?x?xi4>, tensor<?x?x?xi4> -> tensor<?x?x?xi4>
    
    // Down projection
    %output = npu.int4_linear %gated, %down_weight : 
      tensor<?x?x?xi4>, tensor<?x?xi4> -> tensor<?x?x?xi4>
    
    return %output : tensor<?x?x?xi4>
  }
}
