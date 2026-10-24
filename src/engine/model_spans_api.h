#ifndef DS4_ENGINE_MODEL_SPANS_API_H
#define DS4_ENGINE_MODEL_SPANS_API_H

#include "pagein_api.h"
#include "engine_cache_sizes_api.h"
#include "../ds4_streaming_policy.h"

/* CPU-safe span/policy ownership. Vec records are actual pagein_api.h types;
 * successful builders transfer a malloc-compatible vector to their caller. */
bool ds4_native_model_spans_full(const ds4_weights *, const ds4_shape *,
    uint32_t first, uint32_t last, bool output, ds4_model_map_span_vec *);
bool ds4_native_model_spans_static(const ds4_weights *, const ds4_shape *,
    uint32_t first, uint32_t last, bool token, bool output, ds4_model_map_span_vec *);
bool ds4_native_model_spans_runtime(const ds4_weights *, const ds4_shape *,
    const ds4_streaming_policy *, uint32_t first, uint32_t last,
    bool token, bool output, ds4_model_map_span_vec *);
bool ds4_native_model_spans_layer(const ds4_weights *, const ds4_shape *,
    const ds4_streaming_policy *, uint32_t layer, bool decode, ds4_model_map_span_vec *);
bool ds4_native_model_spans_token(const ds4_weights *, ds4_model_map_span_vec *);
bool ds4_native_model_spans_output(const ds4_weights *, ds4_model_map_span_vec *);
uint64_t ds4_native_model_spans_total(const ds4_model_map_span_vec *);
bool ds4_native_model_spans_static_bytes(const ds4_weights *, const ds4_shape *,
    bool token, bool output, uint64_t *bytes);
bool ds4_native_model_spans_uniform(const ds4_weights *, const ds4_shape *, uint32_t layer);
bool ds4_native_model_spans_resident_supported(const ds4_shape *, const ds4_layer_weights *, uint32_t layer);
bool ds4_native_model_spans_resident_layer(const ds4_shape *, const ds4_streaming_policy *,
    const ds4_layer_weights *, uint32_t layer);
bool ds4_native_model_spans_static_supported(const ds4_weights *, const ds4_shape *, const ds4_streaming_policy *);
bool ds4_native_model_spans_glm_addr_layout(const ds4_weights *, const ds4_shape *, const ds4_layer_weights *, uint32_t layer);
bool ds4_native_model_spans_glm_selected_layout(const ds4_shape *, const ds4_layer_weights *, uint32_t layer);
bool ds4_native_model_spans_glm_streamed(const ds4_weights *, const ds4_shape *, const ds4_layer_weights *, uint32_t layer);

#endif
