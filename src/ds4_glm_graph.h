#ifndef DS4_GLM_GRAPH_H
#define DS4_GLM_GRAPH_H

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_model_weights.h"

/* The actual transitional graph record. C session/checkpoint callers and the
 * native owner share these fields; this is not a second Engine/Session mirror.
 * Tensor handles own retained allocations/views; weight and transport pointers
 * borrow their engine's lifetime. Never free a shallow execution-row copy. */
struct ds4_imatrix_collector;

typedef struct {
    uint32_t dst_row;
    uint32_t src_row;
    uint32_t rows;
} glm_vision_overlay_segment;

typedef struct {
    ds4_gpu_tensor *tensor;
    glm_vision_overlay_segment *segments;
    size_t count;
} glm_vision_overlay;

typedef struct {
    ds4_shape shape;
    uint32_t prefill_chunk;
    uint32_t index_pool_size;
    bool reference_hc;
    bool fused_bf16_qkv;
    bool stream_selected_async;
    bool flash_prefill;
    int32_t decode_flush_interval; /* -1 selects the attention-path default. */
} ds4_glm_model_config;

typedef struct ds4_glm_gpu_graph {
    const ds4_weights *weights;
    uint32_t ctx_size;
    uint32_t ctx_cap;
    uint32_t normal_layers;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t layer_count;
    uint64_t q_dim;
    uint64_t q_nope;
    uint64_t heads_dim;
    uint64_t kv_raw_dim;
    uint64_t dense_hidden_max;
    uint64_t ffn_mid_elems;

    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    /* GLM-5.3 keeps four mHC residual streams. cur/next remain the plain
     * one-stream scratch used by the attention and FFN implementations. */
    ds4_gpu_tensor *hc_cur;
    ds4_gpu_tensor *hc_next;
    ds4_gpu_tensor *hc_flat;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *hc_after_attn;
    ds4_gpu_tensor *hc_output;
    ds4_gpu_tensor *hc_mean_weights;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *q_rank;
    ds4_gpu_tensor *q_rank_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv_norm;
    ds4_gpu_tensor *k_nope;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *ffn_sum;
    ds4_gpu_tensor *kda_q;
    ds4_gpu_tensor *kda_k;
    ds4_gpu_tensor *kda_v;
    ds4_gpu_tensor *kda_lowrank;
    ds4_gpu_tensor *kda_raw_gate;
    ds4_gpu_tensor *kda_raw_beta;
    ds4_gpu_tensor *kda_output_gate;
    ds4_gpu_tensor *kda_out;
    ds4_gpu_tensor *layer_kda_conv_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_kda_recurrent_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *prefill_seed_router_selected;
    uint32_t prefill_seed_tokens;
    bool prefill_seed_layer_captured[DS4_MAX_LAYER];

    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur;
    ds4_gpu_tensor *batch_next;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_q_rank;
    ds4_gpu_tensor *batch_q_rank_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv_norm;
    ds4_gpu_tensor *batch_k_nope;
    ds4_gpu_tensor *batch_value;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_after_attn;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_ffn_gate;
    ds4_gpu_tensor *batch_ffn_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_ffn_mid;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_ffn_out;
    bool batch_routed_mid_is_f16;
    uint32_t batch_cap;

    /* GLM-5.3 layer-major prefill workspace. It is allocated at graph
     * creation for the full supported prefill chunk. */
    uint32_t glm53_prefill_cap;
    ds4_gpu_tensor *batch_hc_cur;
    ds4_gpu_tensor *batch_hc_next;
    ds4_gpu_tensor *batch_hc_flat;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_hc_after_attn;
    ds4_gpu_tensor *batch_kda_q;
    ds4_gpu_tensor *batch_kda_k;
    ds4_gpu_tensor *batch_kda_v;
    ds4_gpu_tensor *batch_kda_lowrank;
    ds4_gpu_tensor *batch_kda_raw_gate;
    ds4_gpu_tensor *batch_kda_raw_beta;
    ds4_gpu_tensor *batch_kda_output_gate;
    ds4_gpu_tensor *batch_kda_out;
    ds4_gpu_tensor *session_batch_logits;
    uint32_t session_batch_logits_cap;

    uint32_t compact_cache_cap;
    uint32_t indexed_prefill_cap;
    uint32_t indexed_prefill_score_cap;
    uint32_t indexer_full_layers;
    ds4_gpu_tensor *indexer_k;
    ds4_gpu_tensor *indexer_gate;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *indexer_pool_selected;
    ds4_gpu_tensor *indexer_selected;
    ds4_gpu_tensor *qk_low;
    ds4_gpu_tensor *attn_partial_lora;
    ds4_gpu_tensor *attn_partial_ms;
    ds4_gpu_tensor *batch_indexer_k;
    ds4_gpu_tensor *batch_indexer_gate;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_indexer_scores;
    ds4_gpu_tensor *batch_indexer_pool_selected;
    ds4_gpu_tensor *batch_indexer_selected;
    ds4_gpu_tensor *batch_qk_low;
    ds4_gpu_tensor *batch_attn_lora;
    ds4_gpu_tensor *layer_kv_lora_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_k_rope_cache[DS4_MAX_LAYER];
    /* GLM MTP (nextn block) drafting: private compact caches for the nextn
     * layer (slot = absolute position; only [mtp_min_pos..pos] is ever
     * selected) plus small scratch. Allocated lazily on first draft. */
    ds4_gpu_tensor *mtp_kv_lora_cache;
    ds4_gpu_tensor *mtp_k_rope_cache;
    ds4_gpu_tensor *mtp_concat;
    ds4_gpu_tensor *mtp_selected;
    ds4_gpu_tensor *mtp_state_backup;
    float          *mtp_logits_host;
    int             mtp_ready;
    ds4_gpu_tensor *layer_indexer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_indexer_tail_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_indexer_tail_gate[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_value_cache[DS4_MAX_LAYER];
    bool full_kv_cache;
    bool has_token_embd;
    bool has_output_head;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool generic_routed_moe;
    bool glm53;
    struct ds4_imatrix_collector *imatrix;
    ds4_gpu_tensor *directional_steering_dirs_by_tier[DS4_MAX_GPUS];
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int directional_steering_mode;
    uint32_t directional_steering_layers;
    bool streaming_static_decode_map_current;
    /* Tensor parallelism (50/50 expert sharding): tp_world 2 means
     * this rank computes only its contiguous half of the routed experts
     * and exchanges the 24KB routed-FFN partial at one gate per sparse
     * layer.  Views alias the engine's TP slab slots [layer*2 + FFN]. */
    uint32_t tp_world;
    uint32_t tp_rank;
    ds4_gpu_tensor **tp_out;
    ds4_gpu_tensor **tp_in;
    /* Prefill batch gate bounce buffers (shared storage; grow on demand). */
    ds4_gpu_tensor *tp_bounce_out;
    ds4_gpu_tensor *tp_bounce_in;
    /* CUDA multi-tier placement and device-local decode scratch mirrors. */
    const int *placement;
#define DS4_GLM_WS_SLOTS 29
    ds4_gpu_tensor *ws_mirror[DS4_MAX_GPUS][DS4_GLM_WS_SLOTS];
    ds4_gpu_tensor *ws_orig[DS4_GLM_WS_SLOTS];
    int ws_ready;
    int ws_tier;
#define DS4_GLM_VERIFY_WS_SLOTS 28
    ds4_gpu_tensor *verify_ws_mirror[DS4_MAX_GPUS][DS4_GLM_VERIFY_WS_SLOTS];
    ds4_gpu_tensor *verify_ws_orig[DS4_GLM_VERIFY_WS_SLOTS];
    int verify_ws_ready;
    int verify_ws_tier;
    /* Validated dimensions and model policies are captured once at allocation.
     * Native model code never reads the legacy process-global shape. */
    ds4_glm_model_config config;
} ds4_glm_gpu_graph;

#endif
