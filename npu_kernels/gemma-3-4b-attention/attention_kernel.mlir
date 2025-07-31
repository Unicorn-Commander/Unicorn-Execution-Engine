
// Gemma3 4B Attention Kernel for AMD Phoenix NPU
// Optimized for INT8 computation with 2560 hidden dimension

module @gemma3_4b_attention {
  // Constants for Gemma3 4B
  %hidden_size = arith.constant 2560 : index
  %num_heads = arith.constant 32 : index  
  %head_dim = arith.constant 80 : index
  %kv_heads = arith.constant 16 : index  // GQA
  
  func.func @attention_forward(
    %hidden_states: tensor<1x?x2560xf32>,
    %q_weight: tensor<2560x2560xi8>,
    %k_weight: tensor<2560x1280xi8>,  // KV heads = 16, so 16*80=1280
    %v_weight: tensor<2560x1280xi8>,
    %o_weight: tensor<2560x2560xi8>
  ) -> tensor<1x?x2560xf32> {
    
    // Get dynamic sequence length
    %c1 = arith.constant 1 : index
    %seq_len = tensor.dim %hidden_states, %c1 : tensor<1x?x2560xf32>
    
    // Project to Q, K, V
    %q = linalg.matmul ins(%hidden_states, %q_weight)
    %k = linalg.matmul ins(%hidden_states, %k_weight)
    %v = linalg.matmul ins(%hidden_states, %v_weight)
    
    // Reshape for multi-head attention
    %q_heads = tensor.reshape %q : tensor<1x?x2560xf32> to tensor<1x?x32x80xf32>
    %k_heads = tensor.reshape %k : tensor<1x?x1280xf32> to tensor<1x?x16x80xf32>
    %v_heads = tensor.reshape %v : tensor<1x?x1280xf32> to tensor<1x?x16x80xf32>
    
    // Expand K,V for GQA (repeat 2x to match Q heads)
    %k_expanded = tensor.expand_shape %k_heads [[0], [1], [2, 3], [4]] 
      : tensor<1x?x16x80xf32> into tensor<1x?x16x2x80xf32>
    %k_full = tensor.reshape %k_expanded : tensor<1x?x16x2x80xf32> to tensor<1x?x32x80xf32>
    
    // Compute attention scores
    %scores = linalg.batch_matmul ins(%q_heads, %k_full)
    
    // Apply scaling
    %scale = arith.constant 0.1118 : f32  // 1/sqrt(80)
    %scaled_scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scores) outs(%scores) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %scale : f32
      linalg.yield %scaled : f32
    }
    
    // Softmax
    %attention_weights = "tosa.softmax"(%scaled_scores) {axis = 3 : i64}
    
    // Apply attention to values
    %v_full = tensor.reshape %v_expanded : tensor<1x?x16x2x80xf32> to tensor<1x?x32x80xf32>
    %attention_output = linalg.batch_matmul ins(%attention_weights, %v_full)
    
    // Reshape and project output
    %output_2d = tensor.reshape %attention_output : tensor<1x?x32x80xf32> to tensor<1x?x2560xf32>
    %output = linalg.matmul ins(%output_2d, %o_weight)
    
    return %output : tensor<1x?x2560xf32>
  }
}
