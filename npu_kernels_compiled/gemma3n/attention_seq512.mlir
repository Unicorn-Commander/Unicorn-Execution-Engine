
// Auto-generated NPU kernel for gemma3n
// Sequence length: 512
// Target: AMD Phoenix NPU (XDNA1)

module @gemma3n_attention_seq512 {
  // Model parameters
  %hidden_size = arith.constant 1536 : index
  %num_heads = arith.constant 12 : index
  %head_dim = arith.constant 128 : index
  %kv_heads = arith.constant 12 : index
  %seq_len = arith.constant 512 : index
  
  // Phoenix NPU tile configuration
  %num_tiles = arith.constant 16 : index
  
  func.func @attention_forward(
    %hidden_states: tensor<1x512x1536xi8>,
    %q_weight: tensor<1536x1536xi8>,
    %k_weight: tensor<1536x1536xi8>,
    %v_weight: tensor<1536x1536xi8>,
    %o_weight: tensor<1536x1536xi8>,
    %q_scale: f32, %k_scale: f32, %v_scale: f32, %o_scale: f32
  ) -> tensor<1x512x1536xi8> {
    
    // Tile parallel execution across Phoenix NPU
    %tiles = aie.tiles(4, 4)  // 4x4 tile grid
    
    // QKV Projections - distributed across tiles
    %q_int32 = linalg.matmul ins(%hidden_states, %q_weight : 
      tensor<1x512x1536xi8>, 
      tensor<1536x1536xi8>) 
      outs(%q_out : tensor<1x512x1536xi32>)
      
    %k_int32 = linalg.matmul ins(%hidden_states, %k_weight :
      tensor<1x512x1536xi8>,
      tensor<1536x1536xi8>)
      outs(%k_out : tensor<1x512x1536xi32>)
      
    %v_int32 = linalg.matmul ins(%hidden_states, %v_weight :
      tensor<1x512x1536xi8>,
      tensor<1536x1536xi8>)
      outs(%v_out : tensor<1x512x1536xi32>)
    
    // Dequantize to FP16 for attention computation
    %q_fp16 = arith.mulf %q_int32, %q_scale : tensor<...xf16>
    %k_fp16 = arith.mulf %k_int32, %k_scale : tensor<...xf16>
    %v_fp16 = arith.mulf %v_int32, %v_scale : tensor<...xf16>
    
    // Reshape for multi-head attention
    %q_heads = tensor.reshape %q_fp16 : 
      tensor<1x512x1536xf16> to 
      tensor<1x512x12x128xf16>
      
    %k_heads = tensor.reshape %k_fp16 :
      tensor<1x512x1536xf16> to
      tensor<1x512x12x128xf16>
      
    %v_heads = tensor.reshape %v_fp16 :
      tensor<1x512x1536xf16> to
      tensor<1x512x12x128xf16>

    %k_expanded = %k_heads
    %v_expanded = %v_heads

    // Compute attention scores (Q @ K^T)
    %scores = linalg.batch_matmul_transpose_b ins(%q_heads, %k_expanded :
      tensor<1x512x12x128xf16>,
      tensor<1x512x12x128xf16>)
      outs(%score_out : tensor<1x12x512x512xf16>)
    
    // Scale by 1/sqrt(head_dim)
    %scale_factor = arith.constant 0.088388 : f16
    %scaled_scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scores) outs(%scores) {
    ^bb0(%in: f16, %out: f16):
      %scaled = arith.mulf %in, %scale_factor : f16
      linalg.yield %scaled : f16
    }
    
    // Softmax 
    %attention_weights = "tosa.softmax"(%scaled_scores) {axis = 3 : i64}
    
    // Apply attention to values
    %attention_output = linalg.batch_matmul ins(%attention_weights, %v_expanded :
      tensor<1x12x512x512xf16>,
      tensor<1x512x12x128xf16>)
      outs(%attn_out : tensor<1x512x12x128xf16>)
    
    // Reshape and output projection
    %output_2d = tensor.reshape %attention_output :
      tensor<1x512x12x128xf16> to
      tensor<1x512x1536xf16>
    
    // Quantize back to INT8
    %output_scaled = arith.divf %output_2d, %o_scale : tensor<...xf16>
    %output_i8 = arith.fptosi %output_scaled : tensor<...xf16> to tensor<...xi8>
    
    return %output_i8 : tensor<1x512x1536xi8>
  }
  
  // DMA configuration for Phoenix NPU
  aie.device(npu) {
    // Use memory banks from transcription project
    %tile_0_0 = aie.tile(0, 0)
    %buf_dma = aie.buffer(%tile_0_0) {address = 131071 : ui32} : memref<65536xi8>
    %buf_compute0 = aie.buffer(%tile_0_0) {address = 65536 : ui32} : memref<32768xi8>
    %buf_compute1 = aie.buffer(%tile_0_0) {address = 65537 : ui32} : memref<32768xi8>
  }
}
