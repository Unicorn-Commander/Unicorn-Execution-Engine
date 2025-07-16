
// NPU Attention Kernel for Gemma 27B
// Optimized for AMD Phoenix NPU (16 TOPS)
// Sequence length: 1, Heads: 4, Head dim: 64

module {
  // NPU memory configuration
  %npu_sram = aie.tile(0, 0) : !aie.tile<0, 0>
  %compute_tile = aie.tile(1, 1) : !aie.tile<1, 1>
  
  // Memory allocation for attention matrices
  %q_buffer = aie.buffer(%npu_sram) {sym_name = "q_buffer"} : memref<1x64xf16>
  %k_buffer = aie.buffer(%npu_sram) {sym_name = "k_buffer"} : memref<1x64xf16>
  %v_buffer = aie.buffer(%npu_sram) {sym_name = "v_buffer"} : memref<1x64xf16>
  %scores_buffer = aie.buffer(%npu_sram) {sym_name = "scores_buffer"} : memref<1x1xf16>
  %output_buffer = aie.buffer(%npu_sram) {sym_name = "output_buffer"} : memref<1x64xf16>
  
  // DMA configuration for data movement
  %dma_q = aie.dma_start(S2MM, 0, ^q_transfer, ^end)
  ^q_transfer:
    aie.use_lock(%q_lock, Acquire, 0)
    aie.dma_bd(%q_buffer : memref<1x64xf16>, 0, 64)
    aie.use_lock(%q_lock, Release, 1)
    aie.next_bd ^end
  ^end:
    aie.end
  
  // Attention computation kernel
  func.func @attention_kernel(%q : memref<1x64xf16>,
                             %k : memref<1x64xf16>, 
                             %v : memref<1x64xf16>,
                             %output : memref<1x64xf16>) {
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %seq_len_const = arith.constant 1 : index
    %head_dim_const = arith.constant 64 : index
    
    // Scale factor for attention (1/sqrt(head_dim))
    %scale = arith.constant 0.125 : f16
    
    // Compute attention scores: Q * K^T
    scf.for %i = %c0 to %seq_len_const step %c1 {
      scf.for %j = %c0 to %seq_len_const step %c1 {
        %score = arith.constant 0.0 : f16
        
        // Vectorized dot product for Q[i] * K[j]
        %score_final = scf.for %k = %c0 to %head_dim_const step %c1 
                              iter_args(%acc = %score) -> (f16) {
          %q_val = memref.load %q[%i, %k] : memref<1x64xf16>
          %k_val = memref.load %k[%j, %k] : memref<1x64xf16>
          %prod = arith.mulf %q_val, %k_val : f16
          %new_acc = arith.addf %acc, %prod : f16
          scf.yield %new_acc : f16
        }
        
        // Apply scale factor
        %scaled_score = arith.mulf %score_final, %scale : f16
        memref.store %scaled_score, %scores_buffer[%i, %j] : memref<1x1xf16>
      }
    }
    
    // Softmax computation (simplified for NPU)
    scf.for %i = %c0 to %seq_len_const step %c1 {
      // Find max for numerical stability
      %max_val = arith.constant -65504.0 : f16  // FP16 min
      %row_max = scf.for %j = %c0 to %seq_len_const step %c1 
                        iter_args(%max_acc = %max_val) -> (f16) {
        %score = memref.load %scores_buffer[%i, %j] : memref<1x1xf16>
        %new_max = arith.maxf %max_acc, %score : f16
        scf.yield %new_max : f16
      }
      
      // Compute exp and sum
      %sum = arith.constant 0.0 : f16
      %row_sum = scf.for %j = %c0 to %seq_len_const step %c1 
                        iter_args(%sum_acc = %sum) -> (f16) {
        %score = memref.load %scores_buffer[%i, %j] : memref<1x1xf16>
        %shifted = arith.subf %score, %row_max : f16
        %exp_val = math.exp %shifted : f16
        memref.store %exp_val, %scores_buffer[%i, %j] : memref<1x1xf16>
        %new_sum = arith.addf %sum_acc, %exp_val : f16
        scf.yield %new_sum : f16
      }
      
      // Normalize
      scf.for %j = %c0 to %seq_len_const step %c1 {
        %exp_val = memref.load %scores_buffer[%i, %j] : memref<1x1xf16>
        %normalized = arith.divf %exp_val, %row_sum : f16
        memref.store %normalized, %scores_buffer[%i, %j] : memref<1x1xf16>
      }
    }
    
    // Apply attention to values: Attention * V
    scf.for %i = %c0 to %seq_len_const step %c1 {
      scf.for %k = %c0 to %head_dim_const step %c1 {
        %result = arith.constant 0.0 : f16
        
        %final_result = scf.for %j = %c0 to %seq_len_const step %c1 
                               iter_args(%acc = %result) -> (f16) {
          %attn_weight = memref.load %scores_buffer[%i, %j] : memref<1x1xf16>
          %v_val = memref.load %v[%j, %k] : memref<1x64xf16>
          %weighted = arith.mulf %attn_weight, %v_val : f16
          %new_acc = arith.addf %acc, %weighted : f16
          scf.yield %new_acc : f16
        }
        
        memref.store %final_result, %output[%i, %k] : memref<1x64xf16>
      }
    }
    
    return
  }
}
