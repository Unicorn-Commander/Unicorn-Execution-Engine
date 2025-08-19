// UC1-NPU-Pro Optimized Transformer Kernel
// Fused 12-layer transformer with persistent execution and INT4 support

module @uc1_transformer_fused {
  aie.device(npu) {
    // Memory hierarchy optimized for persistent execution
    %tile_mem = aie.tile(0, 0)  // Memory tile for weights
    %tile_compute_0 = aie.tile(0, 1)  // Compute tile 0
    %tile_compute_1 = aie.tile(0, 2)  // Compute tile 1
    %tile_compute_2 = aie.tile(1, 1)  // Compute tile 2
    %tile_compute_3 = aie.tile(1, 2)  // Compute tile 3
    
    // Persistent weight buffers (stay loaded)
    %weights_embed = aie.buffer(%tile_mem) { sym_name = "weights_embed" } : memref<50000x768xi4>
    %weights_attn = aie.buffer(%tile_mem) { sym_name = "weights_attn" } : memref<12x768x768xi4>
    %weights_ffn1 = aie.buffer(%tile_mem) { sym_name = "weights_ffn1" } : memref<12x768x3072xi4>
    %weights_ffn2 = aie.buffer(%tile_mem) { sym_name = "weights_ffn2" } : memref<12x3072x768xi4>
    
    // Quantization parameters
    %scales_embed = aie.buffer(%tile_mem) { sym_name = "scales_embed" } : memref<50000xf32>
    %scales_attn = aie.buffer(%tile_mem) { sym_name = "scales_attn" } : memref<12x768xf32>
    %scales_ffn1 = aie.buffer(%tile_mem) { sym_name = "scales_ffn1" } : memref<12x768xf32>
    %scales_ffn2 = aie.buffer(%tile_mem) { sym_name = "scales_ffn2" } : memref<12x3072xf32>
    
    // Input/output buffers (reused per request)
    %input_tokens = aie.buffer(%tile_mem) { sym_name = "input_tokens" } : memref<64x512xi32>
    %hidden_states = aie.buffer(%tile_mem) { sym_name = "hidden_states" } : memref<64x512x768xf32>
    %output_embeddings = aie.buffer(%tile_mem) { sym_name = "output_embeddings" } : memref<64x768xf32>
    
    // Intermediate buffers
    %attn_temp = aie.buffer(%tile_compute_0) { sym_name = "attn_temp" } : memref<512x768xf32>
    %ffn_temp = aie.buffer(%tile_compute_1) { sym_name = "ffn_temp" } : memref<512x3072xf32>
    %layer_temp = aie.buffer(%tile_compute_2) { sym_name = "layer_temp" } : memref<512x768xf32>
    
    // Core 0: Embedding lookup with INT4 dequantization
    %core_embed = aie.core(%tile_compute_0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c768 = arith.constant 768 : index
      
      // Process each sample in batch
      scf.for %batch = %c0 to %c64 step %c1 {
        scf.for %seq = %c0 to %c512 step %c1 {
          // Load token ID
          %token_id = memref.load %input_tokens[%batch, %seq] : memref<64x512xi32>
          %token_idx = arith.index_cast %token_id : i32 to index
          
          // Load embedding vector with INT4 dequantization
          scf.for %dim = %c0 to %c768 step %c1 {
            // Load INT4 weight and scale
            %weight_i4 = memref.load %weights_embed[%token_idx, %dim] : memref<50000x768xi4>
            %scale = memref.load %scales_embed[%token_idx] : memref<50000xf32>
            
            // Dequantize: (INT4 - 8) * scale
            %weight_i32 = arith.extui %weight_i4 : i4 to i32
            %weight_f32 = arith.uitofp %weight_i32 : i32 to f32
            %offset = arith.constant 8.0 : f32
            %centered = arith.subf %weight_f32, %offset : f32
            %dequant = arith.mulf %centered, %scale : f32
            
            memref.store %dequant, %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
          }
        }
      }
      
      aie.end
    }
    
    // Core 1: Fused attention layers (parallel processing)
    %core_attn = aie.core(%tile_compute_1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c768 = arith.constant 768 : index
      
      // Process all 12 layers in sequence
      scf.for %layer = %c0 to %c12 step %c1 {
        scf.for %batch = %c0 to %c64 step %c1 {
          
          // Simplified attention (real would include Q, K, V projections)
          scf.for %seq = %c0 to %c512 step %c1 {
            scf.for %dim = %c0 to %c768 step %c1 {
              
              // Load input
              %input = memref.load %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
              
              // Load attention weight with INT4 dequantization
              %weight_i4 = memref.load %weights_attn[%layer, %dim, %dim] : memref<12x768x768xi4>
              %scale = memref.load %scales_attn[%layer, %dim] : memref<12x768xf32>
              
              // Dequantize weight
              %weight_i32 = arith.extui %weight_i4 : i4 to i32
              %weight_f32 = arith.uitofp %weight_i32 : i32 to f32
              %offset = arith.constant 8.0 : f32
              %centered = arith.subf %weight_f32, %offset : f32
              %dequant_weight = arith.mulf %centered, %scale : f32
              
              // Compute attention (simplified)
              %attn_out = arith.mulf %input, %dequant_weight : f32
              
              // Apply activation (tanh)
              %activated = math.tanh %attn_out : f32
              
              // Store intermediate result
              memref.store %activated, %attn_temp[%seq, %dim] : memref<512x768xf32>
            }
          }
          
          // Copy back to hidden states
          scf.for %seq = %c0 to %c512 step %c1 {
            scf.for %dim = %c0 to %c768 step %c1 {
              %val = memref.load %attn_temp[%seq, %dim] : memref<512x768xf32>
              memref.store %val, %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
            }
          }
        }
      }
      
      aie.end
    }
    
    // Core 2: Fused FFN layers (parallel processing)
    %core_ffn = aie.core(%tile_compute_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c768 = arith.constant 768 : index
      %c3072 = arith.constant 3072 : index
      
      // Process all 12 layers
      scf.for %layer = %c0 to %c12 step %c1 {
        scf.for %batch = %c0 to %c64 step %c1 {
          
          // FFN Layer 1: 768 -> 3072
          scf.for %seq = %c0 to %c512 step %c1 {
            scf.for %out_dim = %c0 to %c3072 step %c1 {
              
              %sum_init = arith.constant 0.0 : f32
              %sum = scf.for %in_dim = %c0 to %c768 step %c1 iter_args(%acc = %sum_init) -> f32 {
                // Load input
                %input = memref.load %hidden_states[%batch, %seq, %in_dim] : memref<64x512x768xf32>
                
                // Load FFN1 weight with INT4 dequantization
                %weight_i4 = memref.load %weights_ffn1[%layer, %in_dim, %out_dim] : memref<12x768x3072xi4>
                %scale = memref.load %scales_ffn1[%layer, %in_dim] : memref<12x768xf32>
                
                // Dequantize
                %weight_i32 = arith.extui %weight_i4 : i4 to i32
                %weight_f32 = arith.uitofp %weight_i32 : i32 to f32
                %offset = arith.constant 8.0 : f32
                %centered = arith.subf %weight_f32, %offset : f32
                %dequant_weight = arith.mulf %centered, %scale : f32
                
                // Multiply and accumulate
                %prod = arith.mulf %input, %dequant_weight : f32
                %new_acc = arith.addf %acc, %prod : f32
                scf.yield %new_acc : f32
              }
              
              // Apply ReLU activation
              %zero = arith.constant 0.0 : f32
              %relu_out = arith.maximumf %sum, %zero : f32
              
              memref.store %relu_out, %ffn_temp[%seq, %out_dim] : memref<512x3072xf32>
            }
          }
          
          // FFN Layer 2: 3072 -> 768
          scf.for %seq = %c0 to %c512 step %c1 {
            scf.for %out_dim = %c0 to %c768 step %c1 {
              
              %sum_init = arith.constant 0.0 : f32
              %sum = scf.for %in_dim = %c0 to %c3072 step %c1 iter_args(%acc = %sum_init) -> f32 {
                // Load intermediate
                %input = memref.load %ffn_temp[%seq, %in_dim] : memref<512x3072xf32>
                
                // Load FFN2 weight with INT4 dequantization
                %weight_i4 = memref.load %weights_ffn2[%layer, %in_dim, %out_dim] : memref<12x3072x768xi4>
                %scale = memref.load %scales_ffn2[%layer, %in_dim] : memref<12x3072xf32>
                
                // Dequantize
                %weight_i32 = arith.extui %weight_i4 : i4 to i32
                %weight_f32 = arith.uitofp %weight_i32 : i32 to f32
                %offset = arith.constant 8.0 : f32
                %centered = arith.subf %weight_f32, %offset : f32
                %dequant_weight = arith.mulf %centered, %scale : f32
                
                // Multiply and accumulate
                %prod = arith.mulf %input, %dequant_weight : f32
                %new_acc = arith.addf %acc, %prod : f32
                scf.yield %new_acc : f32
              }
              
              // Store back to hidden states
              memref.store %sum, %layer_temp[%seq, %out_dim] : memref<512x768xf32>
            }
          }
          
          // Add residual connection
          scf.for %seq = %c0 to %c512 step %c1 {
            scf.for %dim = %c0 to %c768 step %c1 {
              %original = memref.load %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
              %ffn_out = memref.load %layer_temp[%seq, %dim] : memref<512x768xf32>
              %residual = arith.addf %original, %ffn_out : f32
              memref.store %residual, %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
            }
          }
        }
      }
      
      aie.end
    }
    
    // Core 3: Pooling and output generation
    %core_pool = aie.core(%tile_compute_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c768 = arith.constant 768 : index
      
      // Mean pooling across sequence dimension
      scf.for %batch = %c0 to %c64 step %c1 {
        scf.for %dim = %c0 to %c768 step %c1 {
          
          %sum_init = arith.constant 0.0 : f32
          %sum = scf.for %seq = %c0 to %c512 step %c1 iter_args(%acc = %sum_init) -> f32 {
            %val = memref.load %hidden_states[%batch, %seq, %dim] : memref<64x512x768xf32>
            %new_acc = arith.addf %acc, %val : f32
            scf.yield %new_acc : f32
          }
          
          // Divide by sequence length for mean
          %seq_len = arith.constant 512.0 : f32
          %mean = arith.divf %sum, %seq_len : f32
          
          memref.store %mean, %output_embeddings[%batch, %dim] : memref<64x768xf32>
        }
      }
      
      // L2 normalization
      scf.for %batch = %c0 to %c64 step %c1 {
        // Calculate L2 norm
        %norm_sq_init = arith.constant 0.0 : f32
        %norm_squared = scf.for %dim = %c0 to %c768 step %c1 iter_args(%acc = %norm_sq_init) -> f32 {
          %val = memref.load %output_embeddings[%batch, %dim] : memref<64x768xf32>
          %sq = arith.mulf %val, %val : f32
          %new_acc = arith.addf %acc, %sq : f32
          scf.yield %new_acc : f32
        }
        
        %norm = math.sqrt %norm_squared : f32
        %eps = arith.constant 1.0e-12 : f32
        %norm_safe = arith.addf %norm, %eps : f32
        
        // Normalize
        scf.for %dim = %c0 to %c768 step %c1 {
          %val = memref.load %output_embeddings[%batch, %dim] : memref<64x768xf32>
          %normalized = arith.divf %val, %norm_safe : f32
          memref.store %normalized, %output_embeddings[%batch, %dim] : memref<64x768xf32>
        }
      }
      
      aie.end
    }
    
    // DMA configuration for persistent operation
    %dma_controller = aie.mem(%tile_mem) {
      %dma_start = aie.dma_start("MM2S", 0, ^input_bd, ^complete)
      
      // Input buffer descriptor
      ^input_bd:
        aie.use_lock(%lock_input, "Acquire", 1)
        aie.dma_bd(%input_tokens : memref<64x512xi32>, 0, 131072)  // 64*512*4 bytes
        aie.use_lock(%lock_input, "Release", 0)
        aie.next_bd ^output_bd
      
      // Output buffer descriptor  
      ^output_bd:
        aie.use_lock(%lock_output, "Acquire", 0)
        aie.dma_bd(%output_embeddings : memref<64x768xf32>, 0, 196608)  // 64*768*4 bytes
        aie.use_lock(%lock_output, "Release", 1)
        aie.next_bd ^complete
      
      ^complete:
        aie.end
    }
    
    // Synchronization locks for persistent execution
    %lock_input = aie.lock(%tile_mem, 0) { sym_name = "lock_input" }
    %lock_output = aie.lock(%tile_mem, 1) { sym_name = "lock_output" }
    %lock_weights = aie.lock(%tile_mem, 2) { sym_name = "lock_weights" }
    %lock_ready = aie.lock(%tile_mem, 3) { sym_name = "lock_ready" }
  }
}