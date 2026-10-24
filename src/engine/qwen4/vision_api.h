#ifndef DS4_ENGINE_QWEN4_VISION_API_H
#define DS4_ENGINE_QWEN4_VISION_API_H

#include "../model_api.h"
#include "../vision_api.h"

/* The table borrows offsets from the sidecar mapping. Architecture is checked
 * before clearing it; later failures retain partial metadata/bindings. */
bool ds4_native_qwen4_vision_weights_bind(ds4_qwen4_vision_weights *,
    const ds4_model *, char *error, size_t error_capacity);
void ds4_native_qwen4_vision_pos_embed(const float *table, uint32_t side,
    uint32_t width, uint32_t grid_height, uint32_t grid_width, float *out);

/* Actual shader leaves. The native controller owns allocation, operation order,
 * activation choice and the complete 27-layer loop. Objective-C owns binding,
 * ARC and dispatch geometry. */
int ds4_native_qwen4_vision_patch(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *first, const ds4_gpu_tensor *second,
    const ds4_gpu_tensor *position, const void *map, uint64_t size,
    uint64_t bias, uint32_t rows, uint32_t width);
int ds4_native_qwen4_vision_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const void *map, uint64_t size, uint64_t weight, uint64_t bias,
    uint32_t rows, uint32_t width, float epsilon);
int ds4_native_qwen4_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
    ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, const void *map, uint64_t size,
    uint64_t bias, uint32_t rows, uint32_t width, uint32_t grid_width,
    uint32_t heads, uint32_t head_width);
int ds4_native_qwen4_vision_attention(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
    uint32_t rows, uint32_t width, uint32_t heads, uint32_t head_width, float scale);
int ds4_native_qwen4_vision_residual(ds4_gpu_tensor *residual,
    const ds4_gpu_tensor *input, const void *map, uint64_t size,
    uint64_t bias, uint32_t rows, uint32_t width);
int ds4_native_qwen4_vision_bias_activation(ds4_gpu_tensor *tensor,
    const void *map, uint64_t size, uint64_t bias,
    uint32_t rows, uint32_t width, uint32_t mode);

#endif
