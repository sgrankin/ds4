#ifndef DS4_ENGINE_VISION_RUNTIME_API_H
#define DS4_ENGINE_VISION_RUNTIME_API_H
#include "ds4_engine_vision.h"
#include "qwen4/vision_api.h"
#include "glm53/vision_api.h"

/* Actual component records, borrowed only for this synchronous transaction.
 * Image rows remain caller-owned until a complete prompt span accepts them. */
int ds4_native_vision_encode_file(ds4_engine_core *, const ds4_shape *,
    const char *, ds4_vision_embedding *, char *, size_t);
int ds4_native_vision_encode_memory(ds4_engine_core *, const ds4_shape *,
    const uint8_t *, size_t, ds4_vision_embedding *, char *, size_t);
int ds4_native_vision_append(ds4_engine_core *, const ds4_shape *,
    ds4_tokens *, ds4_vision_span *, ds4_vision_embedding *, char *, size_t);
int ds4_native_vision_message(ds4_engine_core *, const ds4_shape *,
    ds4_tokens *, const char *, const char *const *, ds4_vision_embedding *,
    size_t, ds4_vision_span *, char *, size_t);
/* Diagnostic callers share the same model-owned preparation/position flow. */
int ds4_native_qwen4_vision_image(const ds4_model *,
    const ds4_qwen4_vision_weights *, const ds4_image *, uint32_t, uint32_t,
    float **, ds4_image_patches *, char *, size_t);
uint32_t ds4_native_qwen4_vision_max_tokens(void);
#endif
