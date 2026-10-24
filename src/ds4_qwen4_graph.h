#ifndef DS4_QWEN4_GRAPH_H
#define DS4_QWEN4_GRAPH_H

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_model_weights.h"
#include "ds4_session_frontier.h"

/* Actual shared model records, not mirrors of Engine or Session. Model and
 * weight pointers remain borrowed. Only the owning graph releases resources;
 * batch row copies borrow all handles and must never be passed to free. */
typedef struct {
    uint32_t n_heads;
    uint64_t multiplier[DS4_MAX_PLE_NGRAM];
    uint32_t head_offset[DS4_MAX_PLE_HEADS];
    uint32_t head_vocab[DS4_MAX_PLE_HEADS];
    uint64_t n_rows;
} ds4_qwen4_ple_hash;

typedef struct {
    ds4_shape shape;
    uint32_t compress_ratios[DS4_MAX_LAYER];
    ds4_qwen4_ple_hash ple;
    float rope_freq[32];
    float rope_mscale;
    uint32_t native_ctx;
    bool rope_yarn;
} ds4_qwen4_config;

/* Engine-owned recurrent storage. Session graph views retain individual slots;
 * ad-hoc sessions beyond this pool use their existing private allocations. */
typedef struct {
    ds4_gpu_tensor *state[DS4_MAX_LAYER];
    ds4_gpu_tensor *history[DS4_MAX_LAYER];
    uint32_t slots;
    uint64_t used;
} ds4_qwen4_state_pool;

/* Scratch is recomputed by every forward. An engine lends one arena to
 * serialized session forwards; recurrent state, caches, logits and snapshots
 * remain private. A multi-session batch is one combined forward, not several
 * forwards encoded into the same command batch over aliased scratch. */
#define DS4_QWEN4_SCRATCH_FIELDS(X) \
    X(R) X(xn) X(lo) X(inj) X(inj_alt) X(mixed) X(blk) X(qkv) X(z) X(ga) \
    X(gb) X(lin_o) X(ple_emb) X(ple_key) X(ple_val) X(ple_gated) X(ple_normed) \
    X(qg) X(kp) X(vp) X(iq) X(ik) X(q) X(gate) X(iqn) X(attn_o) \
    X(score) X(tile_max) X(sel_blocks) X(sel_tokens) X(n_sel) X(attn_part) \
    X(router) X(selected) X(weights) X(mid) X(part) X(sh_gate_logit) \
    X(moe_lists) X(moe_counts) X(sh_gate) X(sh_up) X(sh_mid) X(sh_out) \
    X(hc_u) X(hc_lo_act)

typedef struct ds4_qwen4_gpu_graph {
    uint32_t ctx_cap;
    uint32_t pos;
    uint32_t cap_tokens;
    uint32_t k_blocks;
    uint32_t n_block_cap;
    uint32_t sel_stride;
    /* false when the scratch above is borrowed from the engine's arena */
    bool owns_scratch;
    ds4_gpu_tensor *R, *xn, *lo, *inj, *mixed, *blk;
    ds4_gpu_tensor *qkv, *z, *ga, *gb, *lin_o;
    ds4_gpu_tensor *ple_emb, *ple_key, *ple_val, *ple_gated, *ple_normed, *ple_hist;
    int ple_prev[DS4_MAX_PLE_NGRAM];
    ds4_gpu_tensor *qg, *kp, *vp, *iq, *ik, *q, *gate, *iqn, *attn_o;
    ds4_gpu_tensor *score, *tile_max, *sel_blocks, *sel_tokens, *n_sel, *attn_part;
    ds4_gpu_tensor *router, *selected, *weights, *mid, *part, *sh_gate_logit;
    ds4_gpu_tensor *moe_lists, *moe_counts, *sh_gate, *sh_up, *sh_mid, *sh_out, *hc_u, *hc_lo_act;
    ds4_gpu_tensor *logits;
    /* Arena only, grown on demand: one logit row per batch member, the
     * attention rows' cache table (one entry per row and layer) and their
     * split partials. */
    ds4_gpu_tensor *batch_logits, *batch_attn_rows, *batch_attn_part, *batch_head_x, *batch_gdn_rows;
    uint32_t batch_logit_rows;
    ds4_gpu_tensor *layer_lin_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_lin_hist[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_k_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_v_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_ik_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_block_key[DS4_MAX_LAYER];
    /* A separate injection buffer keeps combined normalization race-free. */
    ds4_gpu_tensor *inj_alt;
    /* MTP: staged predictor input, its residual, and the recurrent-state
     * snapshot that verify rollback restores */
    ds4_gpu_tensor *mtp_e, *mtp_cat, *mtp_proj, *mtp_R, *mtp_argmax, *mtp_argmax_tmp;
    /* MTP draft head over a ranked vocabulary subset: gathered Q8 rows, their
     * token ids, and whether the lazy load was attempted */
    ds4_gpu_tensor *draft_head;
    int32_t *draft_ids;
    uint32_t draft_rows;
    bool draft_head_tried;
    ds4_gpu_tensor *snap_lin_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap_lin_hist[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap_ple_hist;
    int snap_ple_prev[DS4_MAX_PLE_NGRAM];
    uint32_t snap_pos;
    /* Second snapshot set (after row 1) for 3-row MTP verifies: a partial
     * accept of two drafts restores this instead of replaying a token. */
    ds4_gpu_tensor *snap2_lin_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap2_lin_hist[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap2_ple_hist;
    int snap2_ple_prev[DS4_MAX_PLE_NGRAM];
    uint32_t snap2_pos;
    int32_t snap2_mrope_delta;
    bool snap2_valid;
    /* Pre-verify snapshot set (state at the block start).  Exact sampling
     * rewinds to the block start to resample the boundary token when a
     * verified block crosses a sampling-mode boundary; without this set
     * that rewind resets the recurrent state and replays the whole kept
     * context on the next eval. */
    ds4_gpu_tensor *snap0_lin_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap0_lin_hist[DS4_MAX_LAYER];
    ds4_gpu_tensor *snap0_ple_hist;
    int snap0_ple_prev[DS4_MAX_PLE_NGRAM];
    uint32_t snap0_pos;
    int32_t snap0_mrope_delta;
    bool snap0_valid;
    uint32_t mtp_pos;
    uint32_t n_logit_rows;
    bool snap_after_first;   /* set by the caller for a 2-token verify: snapshot the state after row 0 */
    bool snap_after_second;  /* 3-token verify: also snapshot the state after row 1 */
    bool verify_rows_exact;  /* 3-token verify: split attention into 2/1-row sub-batches */
    bool snap_valid;
    /* multimodal: per-position (t, h, w) rope positions, the text counter
     * offset, and the image spans of the prompt being prefilled */
    ds4_gpu_tensor *pos3;
    uint32_t *host_pos3;
    int32_t mrope_delta;
    int32_t snap_mrope_delta;
    const ds4_vision_span *vis_spans;
    uint32_t mtp_last_rows;  /* rows of the last predictor pass resident in mtp_R */
    size_t vis_span_count;
    float *host_row;
    float *host_logits;
    ds4_gpu_tensor *steer_dirs;
    float steer_attn_scale;
    float steer_ffn_scale;
    bool dump_prompt_rows;
    /* Captured once at allocation; hot loops never read family globals. */
    ds4_qwen4_config config;
} ds4_qwen4_gpu_graph;

/* Borrowed views for one encoded batch. The frontend supplies initialized
 * same-engine graphs and each slot index (-1 for private recurrent storage).
 * tokens[0] is the next input; ragged verification may add tokens[1]. */
typedef struct {
    ds4_qwen4_gpu_graph *graph;
    ds4_session_frontier *frontier;
    int tokens[2];
    uint32_t n;
    uint32_t row0;
    int slot;
} ds4_qwen4_batch_member;

bool ds4_native_qwen4_graph_alloc(ds4_qwen4_gpu_graph *graph,
    const ds4_weights *weights, uint32_t context, uint32_t capacity, bool mtp,
    const ds4_qwen4_gpu_graph *shared, ds4_gpu_tensor *const *state_pool,
    ds4_gpu_tensor *const *history_pool, uint32_t slot,
    const ds4_qwen4_config *config);
void ds4_native_qwen4_graph_free(ds4_qwen4_gpu_graph *graph);
void ds4_native_qwen4_graph_reset(ds4_qwen4_gpu_graph *graph);
void ds4_native_qwen4_graph_transfer_scratch(ds4_qwen4_gpu_graph *destination,
    ds4_qwen4_gpu_graph *source);
uint64_t ds4_native_qwen4_graph_scratch_bytes(const ds4_qwen4_gpu_graph *graph);
bool ds4_native_qwen4_graph_load_steering(ds4_qwen4_gpu_graph *graph,
    const char *path, float attention_scale, float ffn_scale);
bool ds4_native_qwen4_ngram_read(const ds4_model *model, const uint32_t *rows,
    size_t count, float *out);
void ds4_native_qwen4_ple_step(const ds4_qwen4_config *config, int token,
    int previous[DS4_MAX_PLE_NGRAM], uint32_t *rows);
bool ds4_native_qwen4_graph_forward_tokens(ds4_qwen4_gpu_graph *graph,
    const ds4_model *model, const ds4_weights *weights, const int *tokens,
    uint32_t count, float *logits, bool all_rows);
bool ds4_native_qwen4_graph_mtp_steps(ds4_qwen4_gpu_graph *graph,
    const ds4_model *model, const ds4_weights *weights, uint32_t row,
    const int *tokens, uint32_t count, uint32_t position, bool want_logits,
    float *logits, int *draft);
bool ds4_native_qwen4_pool_ensure(ds4_qwen4_state_pool *pool,
    const ds4_qwen4_config *config, uint32_t slots);
int ds4_native_qwen4_pool_take(ds4_qwen4_state_pool *pool);
void ds4_native_qwen4_pool_release(ds4_qwen4_state_pool *pool, uint32_t slot);
void ds4_native_qwen4_pool_free(ds4_qwen4_state_pool *pool);
uint64_t ds4_native_qwen4_payload_tensor_bytes(const ds4_qwen4_config *config,
    uint32_t rows, uint32_t mtp_rows);
int ds4_native_qwen4_payload_save(ds4_qwen4_gpu_graph *graph,
    const ds4_session_frontier *frontier, bool ready, bool rewound, FILE *file,
    char *error, size_t error_capacity);
/* mutated becomes true immediately before the first destination write.
 * Caller clears its pending draft counters then, and rewound on success.
 * Success commits the actual frontier; failures retain existing reset rules. */
int ds4_native_qwen4_payload_load(ds4_qwen4_gpu_graph *graph,
    ds4_session_frontier *frontier, bool ready, FILE *file,
    const uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS], uint64_t *remaining,
    bool *mutated, char *error, size_t error_capacity);
bool ds4_native_qwen4_batch_supported(const ds4_qwen4_gpu_graph *arena,
    const ds4_qwen4_batch_member *members, uint32_t count, bool speculative);
/* Encode-only, called inside the frontend's bounded batch/cancellation scope.
 * Advances live recurrent/KV and PLE staging state, leaving commands open.
 * Caller waits/readbacks, then commits graph.pos and frontier tokens/logits;
 * on failure caller invalidates affected sessions. No cancellation callback or
 * independent per-token commit is introduced here. */
bool ds4_native_qwen4_batch_encode(const ds4_qwen4_batch_member *members,
    uint32_t count, ds4_qwen4_gpu_graph *arena,
    const ds4_qwen4_state_pool *pool, const ds4_model *model,
    const ds4_weights *weights);

uint32_t ds4_native_qwen4_prefill_chunk(uint32_t context);
ds4_context_memory ds4_native_qwen4_context_memory(
    const ds4_qwen4_config *config, uint32_t context, uint32_t prefill_chunk);

#endif
