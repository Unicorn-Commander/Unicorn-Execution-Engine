
// Fused Multi-Head Attention - Single kernel for entire attention
// Target: Minimize memory transfers, maximize NPU utilization
// Features: QKV fusion, output projection, residual connection

module {
  func.func @fused_multihead_attention(
    %hidden_states: memref<1x256x5376xf16>,     // Input sequence
    %q_weight: memref<32x168x5376xf16>,         // Q projection weights
    %k_weight: memref<16x168x5376xf16>,         // K projection weights  
    %v_weight: memref<16x168x5376xf16>,         // V projection weights
    %o_weight: memref<5376x5376xf16>,           // Output projection
    %output: memref<1x256x5376xf16>             // Final output
  ) {
    
    // Allocate intermediate tensors in NPU SRAM
    %q_proj = memref.alloc() : memref<32x256x168xf16>
    %k_proj = memref.alloc() : memref<16x256x168xf16>  
    %v_proj = memref.alloc() : memref<16x256x168xf16>
    %attn_out = memref.alloc() : memref<32x256x168xf16>
    %concat_out = memref.alloc() : memref<1x256x5376xf16>
    
    // Fused QKV projection with 16-way vectorization
    call @fused_qkv_projection_16way(%hidden_states, %q_weight, %k_weight, %v_weight,
                                     %q_proj, %k_proj, %v_proj)
      : (memref<1x256x5376xf16>, memref<32x168x5376xf16>, memref<16x168x5376xf16>, 
         memref<16x168x5376xf16>, memref<32x256x168xf16>, memref<16x256x168xf16>, 
         memref<16x256x168xf16>) -> ()
    
    // Multi-head attention computation
    call @flash_attention_16way(%q_proj, %k_proj, %v_proj, %attn_out)
      : (memref<32x256x168xf16>, memref<16x256x168xf16>, memref<16x256x168xf16>,
         memref<32x256x168xf16>) -> ()
    
    // Concatenate heads and output projection
    call @concat_heads_and_project(%attn_out, %o_weight, %concat_out)
      : (memref<32x256x168xf16>, memref<5376x5376xf16>, memref<1x256x5376xf16>) -> ()
    
    // Add residual connection
    call @add_residual_16way(%hidden_states, %concat_out, %output)
      : (memref<1x256x5376xf16>, memref<1x256x5376xf16>, memref<1x256x5376xf16>) -> ()
    
    return
  }
}
