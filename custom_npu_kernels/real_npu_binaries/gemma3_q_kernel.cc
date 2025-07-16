// Gemma 3 27B Q Projection Kernel for NPU Phoenix
// Optimized INT8 symmetric quantization with BF16 scales
// [1, 64, 5376] @ [5376, 4096] -> [1, 64, 4096]

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

// Gemma 3 Q projection: hidden_size=5376, output_size=4096
void gemma3_q_projection(
    const aie::vector<float, 16>* input,      // [1, 64, 5376] input tensor
    const aie::vector<int8, 32>* weight,      // [5376, 4096] quantized weights
    const float scale,                        // Dequantization scale
    aie::vector<float, 16>* output           // [1, 64, 4096] output tensor
) {
    static_assert(5376 % 16 == 0, "Hidden size must be divisible by vector width");
    static_assert(4096 % 16 == 0, "Output size must be divisible by vector width");
    
    const int HIDDEN_SIZE = 5376;
    const int OUTPUT_SIZE = 4096;
    const int SEQ_LEN = 64;
    const int VEC_WIDTH = 16;
    
    // Outer loop: sequence dimension
    for (int seq = 0; seq < SEQ_LEN; seq++) {
        // Middle loop: output dimension
        for (int out = 0; out < OUTPUT_SIZE; out += VEC_WIDTH) {
            aie::vector<float, 16> acc = aie::zeros<float, 16>();
            
            // Inner loop: hidden dimension (vectorized)
            for (int hidden = 0; hidden < HIDDEN_SIZE; hidden += VEC_WIDTH) {
                // Load input vector [16 elements]
                aie::vector<float, 16> input_vec = input[seq * (HIDDEN_SIZE/VEC_WIDTH) + hidden/VEC_WIDTH];
                
                // Load and dequantize weight vectors
                for (int out_elem = 0; out_elem < VEC_WIDTH; out_elem++) {
                    // Load quantized weights for this output element
                    aie::vector<int8, 16> weight_vec;
                    for (int h = 0; h < VEC_WIDTH; h++) {
                        int weight_idx = (hidden + h) * OUTPUT_SIZE + (out + out_elem);
                        weight_vec[h] = reinterpret_cast<const int8*>(weight)[weight_idx];
                    }
                    
                    // Dequantize to float
                    aie::vector<float, 16> weight_float = aie::to_float(weight_vec) * scale;
                    
                    // Dot product and accumulate
                    float dot_result = aie::reduce_add(aie::mul(input_vec, weight_float));
                    acc[out_elem] += dot_result;
                }
            }
            
            // Store output vector
            output[seq * (OUTPUT_SIZE/VEC_WIDTH) + out/VEC_WIDTH] = acc;
        }
    }
}

// Kernel wrapper for NPU execution
extern "C" void gemma3_q_kernel_wrapper(
    float* input_data,    // [1, 64, 5376]
    int8_t* weight_data,  // [5376, 4096] 
    float scale_val,      // Dequantization scale
    float* output_data    // [1, 64, 4096]
) {
    gemma3_q_projection(
        reinterpret_cast<const aie::vector<float, 16>*>(input_data),
        reinterpret_cast<const aie::vector<int8, 32>*>(weight_data),
        scale_val,
        reinterpret_cast<aie::vector<float, 16>*>(output_data)
    );
}
