#ifndef DS4_ENGINE_VISION_WEIGHTS_API_H
#define DS4_ENGINE_VISION_WEIGHTS_API_H
#include "ds4_model_weights.h"
#include "ds4_gpu.h"
/* Actual public weight record remains in ds4_gpu.h, defined once. This strict
 * directory binder borrows Model and publishes offsets only; it initializes no
 * GPU resources and can be called by existing model/vision fixtures directly. */
void ds4_native_vision_weights_bind(ds4_deepseek4_vision_weights *,
    const ds4_model *, const ds4_shape *);
#endif
