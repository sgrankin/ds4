#ifndef DS4_ENGINE_VISION_API_H
#define DS4_ENGINE_VISION_API_H

#include "ds4_gpu.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* Real leaf encoders only. Native vision owns its complete workspace, shape
 * policy and layer flow; Objective-C retains ARC/PSO/shader-packet mechanics. */
int ds4_native_vision_rms(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t width, uint32_t rows, float epsilon, const char *label);
int ds4_native_vision_bias(const char *kernel, ds4_gpu_tensor *tensor,
    const ds4_gpu_tensor *residual, const void *model_map, uint64_t model_size,
    uint64_t bias_offset, uint32_t width, uint32_t rows, const char *label);
int ds4_native_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k, ds4_gpu_tensor *v,
    const ds4_gpu_tensor *qkv, const void *model_map, uint64_t model_size,
    uint64_t bias_offset, uint32_t rows, uint32_t grid_width);
int ds4_native_vision_attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, uint32_t rows);
int ds4_native_vision_round(ds4_gpu_tensor *tensor, uint32_t width, uint32_t rows);
int ds4_native_vision_two_rows(const char *kernel, ds4_gpu_tensor *first,
    const ds4_gpu_tensor *second, uint32_t width, uint32_t rows, const char *label);
int ds4_native_vision_reorder(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    uint32_t grid_height, uint32_t grid_width, uint32_t output_rows);
int ds4_native_vision_gelu_bias(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const void *model_map, uint64_t model_size, uint64_t bias_offset,
    uint32_t rows, uint32_t width);

#endif
