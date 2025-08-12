// Fused Attention Score + Softmax Kernel
// Computes Q @ K^T, applies causal masking, and performs a stable softmax in one pass.

__kernel void attention_softmax_fused(
    __global const float* restrict Q,           // Query tensor [num_heads, seq_len, head_dim]
    __global const float* restrict K,           // Key tensor [num_heads, seq_len, head_dim]
    __global float* restrict attention_weights, // Output tensor [num_heads, seq_len, seq_len]
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale                         // 1.0f / sqrt(head_dim)
) {
    // Each work-item computes one row of the attention matrix for one head.
    const int head = get_global_id(0);
    const int row = get_global_id(1);

    if (head >= num_heads || row >= seq_len) return;

    // Pointers to the start of the current head's Q and K data
    const __global float* q_row = Q + head * seq_len * head_dim + row * head_dim;
    const __global float* k_base = K + head * seq_len * head_dim;

    // --- Pass 1: Compute dot products and find the max score for stable softmax ---
    float max_score = -FLT_MAX;
    for (int col = 0; col <= row; col++) { // Causal mask: only attend to previous tokens
        float score = 0.0f;
        const __global float* k_col = k_base + col * head_dim;

        // Compute dot product Q[row] . K[col]
        for (int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_col[d];
        }
        score *= scale;

        // Store the raw score temporarily in the output buffer
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    // --- Pass 2: Compute exponentials and the sum for the denominator ---
    float sum_exp = 0.0f;
    for (int col = 0; col <= row; col++) {
        // Load the raw score, subtract max, and compute exponential
        float val = attention_weights[head * seq_len * seq_len + row * seq_len + col];
        float exp_val = exp(val - max_score);
        
        // Store the exponential value back
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = exp_val;
        sum_exp += exp_val;
    }

    // --- Pass 3: Normalize to get the final attention weights ---
    float inv_sum_exp = 1.0f / sum_exp;
    for (int col = 0; col <= row; col++) {
        attention_weights[head * seq_len * seq_len + row * seq_len + col] *= inv_sum_exp;
    }

    // --- Pass 4: Explicitly zero out the masked-out upper triangle ---
    for (int col = row + 1; col < seq_len; col++) {
        attention_weights[head * seq_len * seq_len + row * seq_len + col] = 0.0f;
    }
}
