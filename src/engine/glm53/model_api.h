#ifndef DS4_ENGINE_GLM53_MODEL_API_H
#define DS4_ENGINE_GLM53_MODEL_API_H

#include "../model_api.h"
#include "memory_api.h"
#include "ds4_glm_graph.h"
#include <math.h>
#include <stdio.h>

/* Configuration commits only on success. Bound tables borrow model's tensor
 * descriptors and preserve partial bindings on failure. C owns fatal reporting.
 * Weight operations retain both Flash and legacy GLM layout branches, with an
 * explicit validated shape rather than an active process-global model. */
bool ds4_native_glm53_model_config(const ds4_model *, ds4_glm_model_config *,
                                  char *error, size_t error_capacity);
bool ds4_native_glm53_weights_bind(ds4_weights *, const ds4_model *,
    const ds4_glm_model_config *, bool load_slice, uint32_t start, uint32_t end,
    bool require_output, bool optional_output, char *error, size_t error_capacity);
bool ds4_native_glm53_weights_validate(const ds4_weights *,
    const ds4_glm_model_config *, uint32_t start, uint32_t end,
    bool require_embedding, bool require_output, char *error, size_t error_capacity);
bool ds4_native_glm53_layer_has_required(const ds4_layer_weights *, uint32_t layer,
                                        const ds4_glm_model_config *);

#endif
