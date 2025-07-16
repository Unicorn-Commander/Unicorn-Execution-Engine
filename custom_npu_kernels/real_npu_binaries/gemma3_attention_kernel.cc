// Gemma 3 27B Attention Kernel for NPU Phoenix
// Grouped Query Attention: 32 Q heads, 16 K/V heads, 128 head_dim

#include <aie_api/aie.hpp>
#include <cmath>

void gemma3_attention(
    const aie::vector<float, 16>* q_input,    // [1, 64, 32, 128]
    const aie::vector<float, 16>* k_input,    // [1, 64, 16, 128]
    const aie::vector<float, 16>* v_input,    // [1, 64, 16, 128]
    aie::vector<float, 16>* output           // [1, 64, 32, 128]
) {
    const int SEQ_LEN = 64;
    const int Q_HEADS = 32;
    const int KV_HEADS = 16;
    const int HEAD_DIM = 128;
    const int VEC_WIDTH = 16;
    const float SCALE = 0.088388;  // 1/sqrt(128)
    
    // Temporary buffers for attention scores and probabilities
    alignas(32) float scores[Q_HEADS][SEQ_LEN][SEQ_LEN];
    alignas(32) float probs[Q_HEADS][SEQ_LEN][SEQ_LEN];
    
    // Process each query head
    for (int q_head = 0; q_head < Q_HEADS; q_head++) {
        // Map to corresponding K/V head (Grouped Query Attention)
        int kv_head = q_head % KV_HEADS;
        
        // Compute attention scores: Q @ K^T
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                float score = 0.0f;
                
                // Dot product over head dimension
                for (int d = 0; d < HEAD_DIM; d += VEC_WIDTH) {
                    // Load Q and K vectors
                    int q_idx = i * Q_HEADS * (HEAD_DIM/VEC_WIDTH) + q_head * (HEAD_DIM/VEC_WIDTH) + d/VEC_WIDTH;
                    int k_idx = j * KV_HEADS * (HEAD_DIM/VEC_WIDTH) + kv_head * (HEAD_DIM/VEC_WIDTH) + d/VEC_WIDTH;
                    
                    aie::vector<float, 16> q_vec = q_input[q_idx];
                    aie::vector<float, 16> k_vec = k_input[k_idx];
                    
                    // Compute dot product and accumulate
                    float partial_dot = aie::reduce_add(aie::mul(q_vec, k_vec));
                    score += partial_dot;
                }
                
                scores[q_head][i][j] = score * SCALE;
            }
        }
        
        // Apply softmax
        for (int i = 0; i < SEQ_LEN; i++) {
            // Find maximum for numerical stability
            float max_val = scores[q_head][i][0];
            for (int j = 1; j < SEQ_LEN; j++) {
                max_val = std::max(max_val, scores[q_head][i][j]);
            }
            
            // Compute exponentials and sum
            float exp_sum = 0.0f;
            for (int j = 0; j < SEQ_LEN; j++) {
                float exp_val = std::exp(scores[q_head][i][j] - max_val);
                probs[q_head][i][j] = exp_val;
                exp_sum += exp_val;
            }
            
            // Normalize probabilities
            for (int j = 0; j < SEQ_LEN; j++) {
                probs[q_head][i][j] /= exp_sum;
            }
        }
        
        // Compute output: Probs @ V
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d += VEC_WIDTH) {
                aie::vector<float, 16> output_vec = aie::zeros<float, 16>();
                
                for (int j = 0; j < SEQ_LEN; j++) {
                    float prob = probs[q_head][i][j];
                    
                    // Load V vector
                    int v_idx = j * KV_HEADS * (HEAD_DIM/VEC_WIDTH) + kv_head * (HEAD_DIM/VEC_WIDTH) + d/VEC_WIDTH;
                    aie::vector<float, 16> v_vec = v_input[v_idx];
                    
                    // Multiply by probability and accumulate
                    output_vec = aie::add(output_vec, aie::mul(v_vec, prob));
                }
                
                // Store output
                int out_idx = i * Q_HEADS * (HEAD_DIM/VEC_WIDTH) + q_head * (HEAD_DIM/VEC_WIDTH) + d/VEC_WIDTH;
                output[out_idx] = output_vec;
            }
        }
    }
}

extern "C" void gemma3_attention_kernel_wrapper(
    float* q_data,       // [1, 64, 32, 128]
    float* k_data,       // [1, 64, 16, 128]
    float* v_data,       // [1, 64, 16, 128]
    float* output_data   // [1, 64, 32, 128]
) {
    gemma3_attention(
        reinterpret_cast<const aie::vector<float, 16>*>(q_data),
        reinterpret_cast<const aie::vector<float, 16>*>(k_data),
        reinterpret_cast<const aie::vector<float, 16>*>(v_data),
        reinterpret_cast<aie::vector<float, 16>*>(output_data)
    );
}
