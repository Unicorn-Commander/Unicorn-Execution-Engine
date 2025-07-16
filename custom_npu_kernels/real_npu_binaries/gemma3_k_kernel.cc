// Gemma 3 27B K Projection Kernel for NPU Phoenix
// [1, 64, 5376] @ [5376, 2048] -> [1, 64, 2048]

#include <aie_api/aie.hpp>

void gemma3_k_projection(
    const aie::vector<float, 16>* input,
    const aie::vector<int8, 32>* weight,
    const float scale,
    aie::vector<float, 16>* output
) {
    const int HIDDEN_SIZE = 5376;
    const int OUTPUT_SIZE = 2048;  // K/V output size
    const int SEQ_LEN = 64;
    const int VEC_WIDTH = 16;
    
    for (int seq = 0; seq < SEQ_LEN; seq++) {
        for (int out = 0; out < OUTPUT_SIZE; out += VEC_WIDTH) {
            aie::vector<float, 16> acc = aie::zeros<float, 16>();
            
            for (int hidden = 0; hidden < HIDDEN_SIZE; hidden += VEC_WIDTH) {
                aie::vector<float, 16> input_vec = input[seq * (HIDDEN_SIZE/VEC_WIDTH) + hidden/VEC_WIDTH];
                
                for (int out_elem = 0; out_elem < VEC_WIDTH; out_elem++) {
                    aie::vector<int8, 16> weight_vec;
                    for (int h = 0; h < VEC_WIDTH; h++) {
                        int weight_idx = (hidden + h) * OUTPUT_SIZE + (out + out_elem);
                        weight_vec[h] = reinterpret_cast<const int8*>(weight)[weight_idx];
                    }
                    
                    aie::vector<float, 16> weight_float = aie::to_float(weight_vec) * scale;
                    float dot_result = aie::reduce_add(aie::mul(input_vec, weight_float));
                    acc[out_elem] += dot_result;
                }
            }
            
            output[seq * (OUTPUT_SIZE/VEC_WIDTH) + out/VEC_WIDTH] = acc;
        }
    }
}

extern "C" void gemma3_k_kernel_wrapper(
    float* input_data, int8_t* weight_data, float scale_val, float* output_data
) {
    gemma3_k_projection(
        reinterpret_cast<const aie::vector<float, 16>*>(input_data),
        reinterpret_cast<const aie::vector<int8, 32>*>(weight_data),
        scale_val,
        reinterpret_cast<aie::vector<float, 16>*>(output_data)
    );
}
