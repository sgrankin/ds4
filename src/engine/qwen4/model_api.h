#ifndef DS4_ENGINE_QWEN4_MODEL_API_H
#define DS4_ENGINE_QWEN4_MODEL_API_H

#include "../model_api.h"
#include "ds4_qwen4_graph.h"
#include <math.h>
#include <stdio.h>

/* Borrow actual GGUF records. Configuration owns its values, and weight tables
 * borrow tensor descriptors from model. C retains its fatal-error policy.
 * Configuration is committed only on success; a failed weight bind retains
 * the partial pointer table, matching the existing loader's diagnostics. */
ds4_shape ds4_native_qwen4_default_shape(bool mini);

bool ds4_native_qwen4_model_config(const ds4_model *, ds4_qwen4_config *,
                                  char *error, size_t error_capacity);
bool ds4_native_qwen4_weights_bind(ds4_weights *, const ds4_model *,
    const ds4_qwen4_config *, bool load_slice, uint32_t start, uint32_t end,
    bool require_output, bool optional_output, char *error, size_t error_capacity);
bool ds4_native_qwen4_weights_validate(const ds4_weights *,
    const ds4_qwen4_config *, uint32_t start, uint32_t end,
    bool require_embedding, bool require_output, char *error, size_t error_capacity);
bool ds4_native_qwen4_layer_has_required(const ds4_layer_weights *, uint32_t layer,
                                        const ds4_qwen4_config *);

#endif
