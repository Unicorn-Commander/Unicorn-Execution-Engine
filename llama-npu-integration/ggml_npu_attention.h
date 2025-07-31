/*
 * GGML NPU Attention Header
 * Provides GGML-compatible NPU attention functions
 */

#ifndef GGML_NPU_ATTENTION_H
#define GGML_NPU_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of GGML types (simplified)
struct ggml_tensor;
struct ggml_context;

// NPU attention function compatible with GGML flash attention signature
extern "C" struct ggml_tensor * ggml_npu_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * mask,
    float                 scale,
    float                 max_bias,
    float                 logit_softcap);

// Check if NPU can handle this attention operation
bool ggml_npu_can_flash_attn(
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v);

#ifdef __cplusplus
}
#endif

#endif // GGML_NPU_ATTENTION_H