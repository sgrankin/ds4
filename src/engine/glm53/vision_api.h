#ifndef DS4_ENGINE_GLM53_VISION_API_H
#define DS4_ENGINE_GLM53_VISION_API_H

#include "../model_api.h"
#include "../vision_api.h"

bool ds4_native_glm53_vision_weights_bind(ds4_glm53_vision_weights *,
    const ds4_model *, char *error, size_t error_capacity);

/* Four model-specific leaves complement the shared RMS, bias, attention and
 * matmul primitives. Native vision owns workspace and the complete tower. */
int ds4_native_glm53_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
    ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, const void *map, uint64_t size,
    const ds4_glm53_vision_layer_weights *, uint32_t rows,
    uint32_t grid_height, uint32_t grid_width);
int ds4_native_glm53_vision_swiglu_bias(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
    const void *map, uint64_t size, uint64_t gate_bias, uint64_t up_bias,
    uint32_t width, uint32_t rows);
int ds4_native_glm53_vision_reorder(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *input, uint32_t rows);
int ds4_native_glm53_vision_merger_norm(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *input, const void *map, uint64_t size,
    uint64_t weight, uint64_t bias, uint32_t rows);

#endif
