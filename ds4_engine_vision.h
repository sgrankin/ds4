#ifndef DS4_ENGINE_VISION_H
#define DS4_ENGINE_VISION_H
#include "ds4_engine_core.h"
#include "ds4_image.h"
/* Existing embedding layout discriminator, defined once for both languages. */
#define DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL 1u

/* Metal DS41 dispatch passes actual shared Core/Shape, never Engine storage.
 * All calls are synchronous; output bytes/spans remain libc-owned. */
int ds4_native_vision_encode_file41(ds4_engine_core *, const ds4_shape *,
    const char *path, ds4_vision_embedding *, char *error, size_t capacity);
int ds4_native_vision_encode_memory41(ds4_engine_core *, const ds4_shape *,
    const uint8_t *encoded, size_t length, ds4_vision_embedding *,
    char *error, size_t capacity);
int ds4_native_vision_append41(ds4_engine_core *, const ds4_shape *,
    ds4_tokens *, ds4_vision_span *, ds4_vision_embedding *, char *error, size_t capacity);
int ds4_native_vision_message41(ds4_engine_core *, const ds4_shape *,
    ds4_tokens *, const char *role, const char *const *parts,
    ds4_vision_embedding *, size_t image_count, ds4_vision_span *,
    char *error, size_t capacity);
#endif
