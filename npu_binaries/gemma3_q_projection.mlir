
// Gemma 3 27B Q Projection Kernel - NPU Phoenix Optimized
// Matrix: [5376, 4096] INT8 weights
// Tiles: 16 compute tiles

module @gemma3_q_projection {
  func.func @q_projection_kernel(
    %input: memref<?x5376xf16>,        // Input activations [seq_len, 5376]
    %weight: memref<5376x4096xi8>,     // Q weight matrix INT8
    %scale: memref<1xf16>,             // Quantization scale
    %output: memref<?x4096xf16>        // Output [seq_len, 4096]
  ) {
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %seq_len = memref.dim %input, %c0 : memref<?x5376xf16>
    
    // Tiling parameters for 16 tiles
    %tile_m = arith.constant 64 : index
    %tile_n = arith.constant 128 : index
    %tile_k = arith.constant 256 : index
    
    // Parallel execution across NPU tiles
    scf.parallel (%tile_id) = (%c0) to (%c16) step (%c1) {
      
      // Calculate tile boundaries
      %start_row = arith.muli %tile_id, %tile_m : index
      %end_row = arith.addi %start_row, %tile_m : index
      
      // Tile-based matrix multiplication with INT8 dequantization
      scf.for %i = %start_row to %end_row step %c1 {
        scf.for %j = %c0 to %c4096 step %tile_n {
          
          // Load quantization scale
          %scale_val = memref.load %scale[%c0] : memref<1xf16>
          
          // Accumulator for dot product
          %acc = arith.constant 0.0 : f16
          
          // Inner loop over hidden dimension (reduction)
          %final_acc = scf.for %k = %c0 to %c5376 step %tile_k 
                       iter_args(%acc_iter = %acc) -> (f16) {
            
            // Load input activation (FP16)
            %input_val = memref.load %input[%i, %k] : memref<?x5376xf16>
            
            // Load quantized weight (INT8) and dequantize
            %weight_i8 = memref.load %weight[%k, %j] : memref<5376x4096xi8>
            %weight_f16 = arith.sitofp %weight_i8 : i8 to f16
            %weight_dequant = arith.mulf %weight_f16, %scale_val : f16
            
            // Multiply and accumulate
            %prod = arith.mulf %input_val, %weight_dequant : f16
            %new_acc = arith.addf %acc_iter, %prod : f16
            
            scf.yield %new_acc : f16
          }
          
          // Store result
          memref.store %final_acc, %output[%i, %j] : memref<?x4096xf16>
        }
      }
      
      scf.yield
    }
    
    return
  }
}
