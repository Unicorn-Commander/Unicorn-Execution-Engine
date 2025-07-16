
// 16-way Vectorized FFN - Maximum throughput FFN computation
// Target: Utilize all NPU resources for FFN layers
// Features: Fused gate/up projection, SiLU activation, vectorization

module {
  func.func @vectorized_ffn_16way(
    %input: memref<1x256x5376xf16>,           // Input hidden states
    %gate_weight: memref<14336x5376xf16>,     // Gate projection weights
    %up_weight: memref<14336x5376xf16>,       // Up projection weights
    %down_weight: memref<5376x14336xf16>,     // Down projection weights
    %output: memref<1x256x5376xf16>           // Output hidden states
  ) {
    
    // Allocate intermediate tensors
    %gate_proj = memref.alloc() : memref<1x256x14336xf16>
    %up_proj = memref.alloc() : memref<1x256x14336xf16>
    %gated = memref.alloc() : memref<1x256x14336xf16>
    
    // Fused gate and up projections with 16-way vectorization
    scf.parallel (%seq) = (0) to (256) step (1) {
      scf.parallel (%ffn_dim) = (0) to (14336) step (16) {
        
        // Process 16 FFN dimensions simultaneously
        %gate_vec = arith.constant dense<0.0> : vector<16xf16>
        %up_vec = arith.constant dense<0.0> : vector<16xf16>
        
        // Vectorized matrix multiplication
        scf.for %hidden_dim = (0) to (5376) step (8) {
          
          // Load input vector (8 elements)
          %input_vec = vector.load %input[0, %seq, %hidden_dim] : memref<1x256x5376xf16>, vector<8xf16>
          
          // Load gate weights (16x8 matrix)
          scf.for %i = (0) to (16) step (1) {
            %gate_dim = arith.addi %ffn_dim, %i : index
            %gate_weight_vec = vector.load %gate_weight[%gate_dim, %hidden_dim] : memref<14336x5376xf16>, vector<8xf16>
            %up_weight_vec = vector.load %up_weight[%gate_dim, %hidden_dim] : memref<14336x5376xf16>, vector<8xf16>
            
            // Vectorized multiply-accumulate
            %gate_prod = arith.mulf %input_vec, %gate_weight_vec : vector<8xf16>
            %up_prod = arith.mulf %input_vec, %up_weight_vec : vector<8xf16>
            
            %gate_sum = vector.reduction <add>, %gate_prod : vector<8xf16> into f16
            %up_sum = vector.reduction <add>, %up_prod : vector<8xf16> into f16
            
            %gate_vec = vector.insertelement %gate_sum, %gate_vec[%i] : vector<16xf16>
            %up_vec = vector.insertelement %up_sum, %up_vec[%i] : vector<16xf16>
          }
        }
        
        // Store gate and up projections
        vector.store %gate_vec, %gate_proj[0, %seq, %ffn_dim] : memref<1x256x14336xf16>, vector<16xf16>
        vector.store %up_vec, %up_proj[0, %seq, %ffn_dim] : memref<1x256x14336xf16>, vector<16xf16>
        
        // Fused SiLU activation and element-wise multiplication
        %silu_gate = call @vectorized_silu_16way(%gate_vec) : (vector<16xf16>) -> vector<16xf16>
        %gated_vec = arith.mulf %silu_gate, %up_vec : vector<16xf16>
        
        vector.store %gated_vec, %gated[0, %seq, %ffn_dim] : memref<1x256x14336xf16>, vector<16xf16>
      }
    }
    
    // Down projection with 16-way vectorization
    scf.parallel (%seq) = (0) to (256) step (1) {
      scf.parallel (%hidden_dim) = (0) to (5376) step (16) {
        
        %output_vec = arith.constant dense<0.0> : vector<16xf16>
        
        scf.for %ffn_dim = (0) to (14336) step (8) {
          
          %gated_vec = vector.load %gated[0, %seq, %ffn_dim] : memref<1x256x14336xf16>, vector<8xf16>
          
          scf.for %i = (0) to (16) step (1) {
            %out_dim = arith.addi %hidden_dim, %i : index
            %down_weight_vec = vector.load %down_weight[%out_dim, %ffn_dim] : memref<5376x14336xf16>, vector<8xf16>
            
            %prod = arith.mulf %gated_vec, %down_weight_vec : vector<8xf16>
            %sum = vector.reduction <add>, %prod : vector<8xf16> into f16
            
            %output_vec = vector.insertelement %sum, %output_vec[%i] : vector<16xf16>
          }
        }
        
        vector.store %output_vec, %output[0, %seq, %hidden_dim] : memref<1x256x5376xf16>, vector<16xf16>
      }
    }
    
    return
  }
  
  // Vectorized SiLU activation: x * sigmoid(x)
  func.func @vectorized_silu_16way(%input: vector<16xf16>) -> vector<16xf16> {
    %sigmoid = call @vectorized_sigmoid_16way(%input) : (vector<16xf16>) -> vector<16xf16>
    %result = arith.mulf %input, %sigmoid : vector<16xf16>
    return %result : vector<16xf16>
  }
  
  func.func @vectorized_sigmoid_16way(%input: vector<16xf16>) -> vector<16xf16> {
    // Fast sigmoid approximation for NPU
    %one = arith.constant dense<1.0> : vector<16xf16>
    %neg_input = arith.negf %input : vector<16xf16>
    %exp_neg = math.exp %neg_input : vector<16xf16>
    %one_plus_exp = arith.addf %one, %exp_neg : vector<16xf16>
    %result = arith.divf %one, %one_plus_exp : vector<16xf16>
    return %result : vector<16xf16>
  }
}
