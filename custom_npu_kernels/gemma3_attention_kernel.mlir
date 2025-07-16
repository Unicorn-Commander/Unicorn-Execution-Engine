// Custom NPU Kernel for Gemma 3 27B Scaled Dot-Product Attention
// Target: NPU Phoenix 16 TOPS, optimized for Grouped Query Attention
// Architecture: 32 Q heads, 16 K/V heads, 128 head_dim
// Memory: 2GB SRAM, 16 compute tiles

module {
  aie.device(xcve2802) {
    
    // Gemma 3 27B Grouped Query Attention layout
    // Q: [1, 64, 32, 128] - 32 query heads
    // K: [1, 64, 16, 128] - 16 key heads (repeated for GQA)
    // V: [1, 64, 16, 128] - 16 value heads (repeated for GQA)
    // Output: [1, 64, 32, 128] -> [1, 64, 4096]
    
    // Input buffers
    %buf_q = aie.buffer(%tile_0_0) { sym_name = "q_input" } : memref<1x64x32x128xf16>
    %buf_k = aie.buffer(%tile_0_1) { sym_name = "k_input" } : memref<1x64x16x128xf16>
    %buf_v = aie.buffer(%tile_0_2) { sym_name = "v_input" } : memref<1x64x16x128xf16>
    
    // Intermediate buffers for attention computation
    %buf_scores = aie.buffer(%tile_1_0) { sym_name = "attention_scores" } : memref<1x32x64x64xf16>
    %buf_probs = aie.buffer(%tile_1_1) { sym_name = "attention_probs" } : memref<1x32x64x64xf16>
    
    // Output buffer
    %buf_out = aie.buffer(%tile_2_0) { sym_name = "attention_output" } : memref<1x64x32x128xf16>
    
    // Constants for attention computation
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    
    // Scale factor: 1/sqrt(128) = 0.088388
    %scale = arith.constant 0.088388 : f16
    
    // Define compute tiles for parallel attention heads
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_3_1 = aie.tile(3, 1)
    
    // Core 1: Compute attention scores Q @ K^T
    %core_scores = aie.core(%tile_1_0) {
      // Process each query head
      scf.for %h = %c0 to %c32 step %c1 {
        // For Grouped Query Attention: map 32 Q heads to 16 K heads
        %kv_head = arith.remui %h, %c16 : index
        
        // Compute Q @ K^T for this head
        scf.for %i = %c0 to %c64 step %c1 {
          scf.for %j = %c0 to %c64 step %c1 {
            %sum = arith.constant 0.0 : f16
            %score = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %sum) -> (f16) {
              // Load Q[batch=0, seq_i, head_h, dim_k]
              %q_val = memref.load %buf_q[%c0, %i, %h, %k] : memref<1x64x32x128xf16>
              
              // Load K[batch=0, seq_j, kv_head, dim_k] 
              %k_val = memref.load %buf_k[%c0, %j, %kv_head, %k] : memref<1x64x16x128xf16>
              
              // Multiply and accumulate
              %prod = arith.mulf %q_val, %k_val : f16
              %new_acc = arith.addf %acc, %prod : f16
              scf.yield %new_acc : f16
            }
            
            // Scale by 1/sqrt(head_dim)
            %scaled_score = arith.mulf %score, %scale : f16
            memref.store %scaled_score, %buf_scores[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
          }
        }
      }
      aie.end
    }
    
    // Core 2: Apply softmax to attention scores
    %core_softmax = aie.core(%tile_1_1) {
      scf.for %h = %c0 to %c32 step %c1 {
        scf.for %i = %c0 to %c64 step %c1 {
          // Find maximum for numerical stability
          %neg_inf = arith.constant -65504.0 : f16  // -inf for f16
          %max_val = scf.for %j = %c0 to %c64 step %c1 iter_args(%max = %neg_inf) -> (f16) {
            %score = memref.load %buf_scores[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
            %new_max = arith.maximumf %max, %score : f16
            scf.yield %new_max : f16
          }
          
          // Compute exponentials and sum
          %zero = arith.constant 0.0 : f16
          %exp_sum = scf.for %j = %c0 to %c64 step %c1 iter_args(%sum = %zero) -> (f16) {
            %score = memref.load %buf_scores[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
            %shifted = arith.subf %score, %max_val : f16
            %exp_val = math.exp %shifted : f16
            %new_sum = arith.addf %sum, %exp_val : f16
            
            // Store exp temporarily in probs buffer
            memref.store %exp_val, %buf_probs[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
            scf.yield %new_sum : f16
          }
          
          // Normalize to get probabilities
          scf.for %j = %c0 to %c64 step %c1 {
            %exp_val = memref.load %buf_probs[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
            %prob = arith.divf %exp_val, %exp_sum : f16
            memref.store %prob, %buf_probs[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
          }
        }
      }
      aie.end
    }
    
    // Core 3: Compute final attention output: Probs @ V
    %core_output = aie.core(%tile_2_0) {
      scf.for %h = %c0 to %c32 step %c1 {
        // Map to corresponding K/V head for GQA
        %kv_head = arith.remui %h, %c16 : index
        
        scf.for %i = %c0 to %c64 step %c1 {
          scf.for %d = %c0 to %c128 step %c1 {
            %sum = arith.constant 0.0 : f16
            %output = scf.for %j = %c0 to %c64 step %c1 iter_args(%acc = %sum) -> (f16) {
              // Load attention probability
              %prob = memref.load %buf_probs[%c0, %h, %i, %j] : memref<1x32x64x64xf16>
              
              // Load value
              %v_val = memref.load %buf_v[%c0, %j, %kv_head, %d] : memref<1x64x16x128xf16>
              
              // Multiply and accumulate
              %prod = arith.mulf %prob, %v_val : f16
              %new_acc = arith.addf %acc, %prod : f16
              scf.yield %new_acc : f16
            }
            memref.store %output, %buf_out[%c0, %i, %h, %d] : memref<1x64x32x128xf16>
          }
        }
      }
      aie.end
    }
  }
}