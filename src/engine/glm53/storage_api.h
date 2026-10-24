#ifndef DS4_GLM53_STORAGE_API_H
#define DS4_GLM53_STORAGE_API_H
#include "../../ds4_glm_graph.h"
#include "../../ds4_streaming_policy.h"
#include "../model_spans_api.h"
#include "../pagein_api.h"

/* Graph configuration and the process policy remain actual borrowed owners.
 * The compatibility C boundary only captures the latter's address. */
bool ds4_native_glm_storage_selected(const ds4_glm_gpu_graph *,
    const ds4_layer_weights *, uint32_t, const ds4_streaming_policy *);
bool ds4_native_glm_storage_mtp(const ds4_glm_gpu_graph *,
    const ds4_layer_weights *, uint32_t);
bool ds4_native_glm_storage_seed(const ds4_glm_gpu_graph *, const ds4_weights *,
    const ds4_layer_weights *, uint32_t, const ds4_streaming_policy *);
bool ds4_native_glm_storage_token(ds4_glm_gpu_graph *, const ds4_model *, const ds4_weights *);
bool ds4_native_glm_storage_output(ds4_glm_gpu_graph *, const ds4_model *, const ds4_weights *);
bool ds4_native_glm_storage_begin(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, bool, bool *, const ds4_streaming_policy *);
bool ds4_native_glm_storage_layer(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, uint32_t, const ds4_streaming_policy *);
bool ds4_native_glm_storage_prefill(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, uint32_t, uint32_t, bool, const ds4_streaming_policy *);
#endif
