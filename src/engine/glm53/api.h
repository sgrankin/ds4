#ifndef DS4_ENGINE_GLM53_API_H
#define DS4_ENGINE_GLM53_API_H

#include "../../ds4_glm_graph.h"
#include "memory_api.h"
#include "../diagnostics_api.h"
#include "../payload_io.h"
#include "../stream_io.h"
#include "../../ds4_session_frontier.h"
#include "../../ds4_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Capture the validated shape and execution settings once. Platform admission
 * still precedes native tensor allocation at the transitional C boundary. */
void ds4_native_glm_graph_configure(ds4_glm_gpu_graph *, const ds4_shape *);
/* Returns false for legacy multi-tier ownership, which remains in C. */
bool ds4_native_glm_graph_free(ds4_glm_gpu_graph *);
bool ds4_native_glm_allocate_tensors(ds4_glm_gpu_graph *);
bool ds4_native_glm_plan_allocation(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, uint32_t first_layer, uint32_t last_layer,
    bool require_embedding, bool require_output);
bool ds4_native_glm_reset_kda(ds4_glm_gpu_graph *);
void ds4_native_glm_prefill_workspace_free(ds4_glm_gpu_graph *);
bool ds4_native_glm_prefill_workspace_ensure(ds4_glm_gpu_graph *, uint32_t rows);
bool ds4_native_glm_session_batch_logits_ensure(ds4_glm_gpu_graph *, uint32_t rows);

bool ds4_native_glm_matmul(ds4_gpu_tensor *out, const ds4_model *,
    const ds4_tensor *weight, uint32_t inputs, uint32_t outputs,
    const ds4_gpu_tensor *input, uint32_t rows);
bool ds4_native_glm_hc_pre(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_tensor *fn, const ds4_tensor *scale, const ds4_tensor *base,
    const ds4_tensor *norm, const ds4_gpu_tensor *residual,
    ds4_gpu_tensor *collapsed, ds4_gpu_tensor *normalized);
bool ds4_native_glm_hc_pre_rows(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_tensor *fn, const ds4_tensor *scale, const ds4_tensor *base,
    const ds4_tensor *norm, const ds4_gpu_tensor *residual,
    ds4_gpu_tensor *collapsed, ds4_gpu_tensor *normalized,
    ds4_gpu_tensor *flat, ds4_gpu_tensor *mix, ds4_gpu_tensor *split,
    uint32_t rows);
bool ds4_native_glm_kda(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer);
bool ds4_native_glm_kda_rows(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, uint32_t position,
    uint32_t rows, ds4_gpu_tensor *output);

/* Per-token selection can reuse a dense contiguous fill across DSA layers.
 * Sparse pooled selections remain layer-specific, including their padding. */
typedef struct {
    const ds4_gpu_tensor *selected;
    uint32_t count;
    uint32_t guaranteed_prefix;
} ds4_glm_decode_selection;
bool ds4_native_glm_attention_decode(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, uint32_t position, bool indexed,
    ds4_glm_decode_selection *, uint32_t ablate_mask);
bool ds4_native_glm_attention_rows(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, uint32_t position,
    uint32_t rows, uint32_t ablate_mask);

bool ds4_native_glm_output_head_from(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, const ds4_gpu_tensor *hidden);
bool ds4_native_glm_output_head(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *);
bool ds4_native_glm_output_head_forward(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, const ds4_gpu_tensor *, float *logits);

uint64_t ds4_native_glm_spec_state_bytes(const ds4_glm_gpu_graph *);
bool ds4_native_glm_spec_state_copy(ds4_glm_gpu_graph *, bool save);
bool ds4_native_glm_mtp_ensure(ds4_glm_gpu_graph *);
bool ds4_native_glm_mtp_stream_cache(const ds4_glm_gpu_graph *,
    const ds4_layer_weights *, uint32_t layer);
bool ds4_native_glm_mtp_step(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, int next_token, uint32_t position,
    uint32_t minimum_position, int *draft);

/* Existing cache admission/boosted-layer policy, without numerical dispatch. */
bool ds4_native_glm_stream_selected_cache(const ds4_glm_gpu_graph *,
    const ds4_layer_weights *, uint32_t layer);
bool ds4_native_glm_seed_layer_expected(const ds4_glm_gpu_graph *,
    const ds4_weights *, const ds4_layer_weights *, uint32_t layer);
bool ds4_native_glm_experts_uniform(const ds4_weights *, uint32_t layer);
bool ds4_native_glm_map_token(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *);
bool ds4_native_glm_map_prefill_layer(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, uint32_t layer, uint32_t rows, bool full_layer);
bool ds4_native_glm_map_decode_begin(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, bool from_hidden, bool *static_map);
bool ds4_native_glm_map_decode_layer(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, uint32_t layer);
bool ds4_native_glm_map_output(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *);
bool ds4_native_glm_forward_token(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, int token, const float *input_hc, uint32_t position,
    float *output_hc, float *logits, bool defer_completion,
    uint32_t ablate_mask);

/* One synchronous chunk call. Views and the page-in slot borrow their C chunk
 * owner's lifetime; only hc_current/hc_next are swapped by native traversal. */
typedef struct {
    uint32_t position;
    uint32_t rows;
    bool full_layer_prefill;
    bool layer_prepare;
    bool streaming_sync_each_layer;
    bool allow_complete;
    uint32_t flush_interval;
    uint32_t drain_interval;
    ds4_session_progress_fn progress;
    void *progress_user;
    uint32_t absolute_base;
    uint32_t work_done_base;
    uint32_t work_total;
    ds4_gpu_tensor *current;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *after_attention;
    ds4_gpu_tensor *hc_after_attention;
    ds4_gpu_tensor *hc_current;
    ds4_gpu_tensor *hc_next;
    void *pagein_slot;
} ds4_glm_prefill_pass;
bool ds4_native_glm_prefill_layers(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, ds4_glm_prefill_pass *, uint32_t ablate_mask);
bool ds4_native_glm_chunk_forward(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_weights *, const int *tokens, const float *input_hc,
    const ds4_vision_span *, size_t image_count, uint32_t position,
    uint32_t rows, float *output_hc, float *logits,
    ds4_session_progress_fn, void *progress_user, uint32_t absolute_base,
    uint32_t done_base, uint32_t work_total, uint32_t ablate_mask);

/* Borrow each session's actual cache/state; batching must not turn independent
 * recurrent streams into a causal token sequence. */
typedef struct {
    ds4_glm_gpu_graph *graph;
    uint32_t position;
    int32_t token;
} ds4_glm_session_row;
bool ds4_native_glm_session_batch(const ds4_model *, const ds4_weights *,
    const ds4_glm_session_row *, uint32_t rows, uint32_t ablate_mask);
bool ds4_native_glm_ffn_decode(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, uint32_t position,
    uint32_t ablate_mask);
bool ds4_native_glm_ffn_rows(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, uint32_t position,
    uint32_t rows, ds4_gpu_tensor *output, bool full_layer_prefill,
    uint32_t ablate_mask);
bool ds4_native_glm_routed_one(const ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, ds4_gpu_tensor *output,
    ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
    const ds4_gpu_tensor *weights, const ds4_gpu_tensor *input,
    bool force_resident);
bool ds4_native_glm_routed_rows(ds4_glm_gpu_graph *, const ds4_model *,
    const ds4_layer_weights *, uint32_t layer, ds4_gpu_tensor *output,
    ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
    const ds4_gpu_tensor *weights, const ds4_gpu_tensor *input,
    uint32_t rows, uint32_t mid_stride, bool force_resident,
    bool direct_scalar_q4);

uint64_t ds4_native_glm_payload_bytes(const ds4_glm_gpu_graph *,
    const ds4_session_frontier *, uint32_t dense_cache_len);
int ds4_native_glm_payload_save(ds4_glm_gpu_graph *,
    const ds4_session_frontier *, uint32_t dense_cache_len, FILE *,
    char *error, size_t error_size);
/* The outer session clears its MTP proposal after a successful restore, or
 * after this call invalidates the frontier before writing device state. */
int ds4_native_glm_payload_load(ds4_glm_gpu_graph *, ds4_session_frontier *,
    uint32_t *dense_cache_len, FILE *, const uint32_t *header,
    uint64_t remaining, char *error, size_t error_size);

uint64_t ds4_native_glm_payload_live_bytes(const ds4_glm_gpu_graph *,
    uint32_t full_live, uint32_t compact_live);

#endif
