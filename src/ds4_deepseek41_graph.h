#ifndef DS4_DEEPSEEK41_GRAPH_H
#define DS4_DEEPSEEK41_GRAPH_H

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_engram.h"
#include "ds4_model_weights.h"
#include "engine/stream_io.h"

/* Actual transitional graph record shared with C sessions, state serialization,
 * suffix relocation, batching and DSpark. It is not an Engine/Session mirror.
 * The native owner allocates/releases resources; shallow execution-row copies
 * borrow every tensor/table/host pointer and must never be passed to free.
 */
typedef struct ds4_imatrix_collector ds4_imatrix_collector;
typedef struct ds41_dspark ds41_dspark;

/* A scoped encoder-page lease borrows the admitted expert-cache budget.
 * Release every locked span before restoring that budget, including partial
 * acquisition or cancellation; the model mapping itself remains borrowed. */
typedef struct {
    uint32_t cache_budget;
    struct { uint64_t off, bytes, locked; } span[60];
} ds41_encoder_residency;

/* One view handle per field. Aliases share storage through the GPU tensor API,
 * not by duplicating handles; every successfully created view is freed once. */
typedef struct {
    ds4_gpu_tensor *residual;
    ds4_gpu_tensor *pre;
    ds4_gpu_tensor *selected_comp;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_split;
    ds4_gpu_tensor *attn_split;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv;
    ds4_gpu_tensor *latent;
    ds4_gpu_tensor *index_k;
    ds4_gpu_tensor *pool_kv;
    ds4_gpu_tensor *pool_score;
    ds4_gpu_tensor *index_q;
    ds4_gpu_tensor *index_weights;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *low;
    ds4_gpu_tensor *block;
    ds4_gpu_tensor *mix;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *route_logits;
    ds4_gpu_tensor *route_probs;
    ds4_gpu_tensor *route_weights;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *block_mask;
    ds4_gpu_tensor *engram_rows;
    ds4_gpu_tensor *engram_kv;
    ds4_gpu_tensor *flat_norm;
    ds4_gpu_tensor *experts;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared;
    ds4_gpu_tensor *routed;
} ds41_prefill_row;

/* Original normalized pre-RoPE BF16 tails live separately from post-RoPE KV.
 * Epoch/publication coverage is explicit: restoring an ordinary system prefix
 * cannot invent originals. The archive owns its GPU tensors and host bitmaps. */
typedef struct {
    ds4_gpu_tensor *raw[40], *compressed[4], *index[4];
    uint32_t raw_epoch[40][128];
    uint8_t *published[4]; /* bit 1: index, bit 2: compressed */
    uint64_t bytes;
} ds41_prerope_archive;

typedef struct {
    uint32_t ctx, pos, prefill_cap, carry_cap, prefill_max;
    /* Pruned rows retain the full tile's kernel-selection thresholds. */
    uint32_t plan_rows;
    uint32_t decoder_replay_floor; /* Active inside a bounded raw replay. */
    uint32_t immutable_cache_end; /* Experimental SWA recovery: suppress publication before this endpoint. */
    uint64_t allocation_bytes;
    /* Immutable layer directions are session-owned; scales may change live.
     * Row/workspace views borrow this pointer with the rest of the graph. */
    ds4_gpu_tensor *steering_dirs;
    float steering_attn_scale, steering_ffn_scale;
    int steering_mode;
    uint32_t steering_layers;
    bool steering_capture_enabled, steering_capture_prompt;
    bool valid, streaming, encoder_resident, compact_carry, prefill_alias, quality, workspace_failed;
    bool compressed_kv_f16;
    bool compressed_kv_fp4; /* Physical 288-byte E2M1/E4M3 main-cache rows. */
    bool index_fp4; /* Physical 68-byte index rows; scratch queries remain F32. */
    uint32_t tp_world, tp_rank;
    bool scalar_decode;
    /* Per-invocation fusion state, never serialized as session state. */
    bool hc_attn_deferred, hc_ffn_deferred, kv_stored, attn_expanded, low_unrounded, shared_done;
    bool tp_balanced_route, tp_gate_fused, defer_shared_down;
    ds4_gpu_tensor *selected_half, *route_weights_half;
    ds4_gpu_tensor *rope_tab;
    uint32_t rope_tab_pos; /* Position + 1; zero outside a scalar step. */
    /* Immutable decode copies belong to the engine, not borrowed row graphs. */
    void *decode_weights;
    ds4_gpu_tensor *tp_logits_half;
    ds4_gpu_tensor **tp_out, **tp_in;
    ds4_gpu_tensor **tp_batch_out, **tp_batch_in;
    ds4_imatrix_collector *imatrix;
    const ds4_vision_span *images;
    size_t image_count;
    ds4_engram_layout engram;
    ds4_engram_history history;
    uint32_t *token_map;
    uint32_t (*prefill_ids)[2][DS4_ENGRAM_COLS];
    ds4_engram_table table[2];
    ds4_engram_reader *engram_reader;
    float rows[2][DS4_ENGRAM_COLS * DS4_ENGRAM_DIM];
    ds4_gpu_tensor *window[40];
    ds41_prerope_archive *prerope;
    ds4_gpu_tensor *compressed[4], *index_cache[4];
    ds4_gpu_tensor *index_bf16[4], *index_bf16_flag;
    bool index_bf16_ok; /* Derived views only; ordinary F32 storage is canonical. */
    ds4_gpu_tensor *previous_kv[4], *previous_score[4];
    ds4_gpu_tensor *verify_previous_kv[4], *verify_previous_score[4];
    bool verify_prefix;
    bool score_selected_blocks; /* Only the active scalar sweep owns layer 20 block IDs. */
    ds4_gpu_tensor *engram_q_norm[2], *engram_k_norm[2];
    ds41_prefill_row batch, *rows_view;
    ds41_prefill_row carry;
    ds4_gpu_tensor *prefill_tokens;
    ds41_dspark *dspark;
    ds4_gpu_tensor *oracle_selected; /* Diagnostic deferred route validation. */
    ds4_gpu_tensor *predict_selected;
    metal_graph_selected_async_load predict_load;
    bool predict_active;
    float resident_route_gap; /* Decode, or an explicitly enabled selected prefill. */
    float resident_route_max_mass;
    float resident_route_max_total_mass;
    uint32_t resident_route_max_rank;
    uint32_t resident_route_max_subs;
    uint32_t resident_route_mode;
    bool resident_route_profile;
    bool resident_route_drop_weak;
    bool resident_route_native_weights;
    bool resident_prefill_active;
    uint8_t resident_prefill_snapshot[40][384];
    ds4_gpu_tensor *image_text_mask;
    ds4_gpu_tensor *residual;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *flat_norm;
    ds4_gpu_tensor *mix;
    ds4_gpu_tensor *attn_split;
    ds4_gpu_tensor *ffn_split;
    ds4_gpu_tensor *pre;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *block;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv;
    ds4_gpu_tensor *latent;
    ds4_gpu_tensor *raw_prefill;
    ds4_gpu_tensor *pool_kv;
    ds4_gpu_tensor *pool_score;
    ds4_gpu_tensor *index_q;
    ds4_gpu_tensor *index_k;
    ds4_gpu_tensor *index_weights;
    ds4_gpu_tensor *index_scores;
    ds4_gpu_tensor *selected_comp;
    ds4_gpu_tensor *index_packed;
    ds4_gpu_tensor *selected_kv;
    ds4_gpu_tensor *block_scores;
    ds4_gpu_tensor *block_selected;
    ds4_gpu_tensor *block_mask;
    ds4_gpu_tensor *block_list;
    const ds4_gpu_tensor *block_list_mask;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *low;
    ds4_gpu_tensor *route_logits;
    ds4_gpu_tensor *route_probs;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *route_weights;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *experts;
    ds4_gpu_tensor *routed;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared;
    ds4_gpu_tensor *engram_rows;
    ds4_gpu_tensor *engram_rows_b;
    ds4_gpu_tensor *engram_prefetch;
    ds4_gpu_tensor *engram_kv;
    ds4_gpu_tensor *logits;
} ds41_gpu_graph;

/* Existing private controllers retained during this resource-lifetime slice.
 * These wrappers release/reset their own resources, not the graph record. The
 * selected-load finish must join its service-thread work before tensors retire.
 */
void ds4_native_graph_dspark_free(ds41_dspark *dspark);
void ds4_native_graph_dspark_reset(ds41_dspark *dspark);
void ds4_native_graph_selected_finish(metal_graph_selected_async_load *load);

#endif
