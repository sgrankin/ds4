#ifndef DS4_DSPARK_CYCLE_H
#define DS4_DSPARK_CYCLE_H

#include "ds4.h"
#include "ds4_tp.h"
#include "ds4_deepseek41_dspark_state.h"

#define DS4_DSPARK_MAX_BLOCK_SIZE 16

/* The existing counters, shared once with the C session and diagnostics.
 * Cycles with no ready draft still count the first target token. Timed proposal
 * work excludes the fallback ordinary target step and retains its old scopes. */
typedef struct ds4_dspark_spec_stats {
    uint64_t cycles;
    uint64_t first_tokens;
    uint64_t seed_batches;
    uint64_t proposed_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t full_accepts;
    uint64_t partial_accepts;
    uint64_t direct_full_commits;
    uint64_t direct_partial_commits;
    uint64_t replay_fallbacks;
    uint64_t first_misses;
    uint64_t no_draft;
    uint64_t no_room;
    uint64_t invalid_draft;
    uint64_t draft_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t accepted_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t scheduler_skips;
    uint64_t tail_skips;
    uint64_t verifier_unavailable;
    uint64_t verifier_errors;
    double target_ms;
    double saved_ms;
    double propose_ms;
    double propose_stage0_ms;
    double propose_setup_ms;
    double propose_cache_ms;
    double propose_chain_ms;
    double propose_hidden_ms;
    double propose_conf0_ms;
    double propose_logits_ms;
    double propose_markov_ms;
    double propose_confidence_ms;
    double snapshot_ms;
    double verify_ms;
    double verify_upload_ms;
    double verify_layer_ms;
    double verify_head_ms;
    double verify_read_ms;
    uint64_t verifier_fused_head;
    double replay_ms;
    double total_ms;
} ds4_dspark_spec_stats;

/* Immutable call policy, copied from the existing engine fields. */
typedef struct {
    bool quality, strict, exact_sampling;
    float confidence_threshold;
} ds4_dspark_cycle_policy;

/* Existing transport identity; NULL handle means no active TP session.
 * The C worker's receive/dispatch loop remains the transport coordinator. */
typedef struct {
    ds4_tp *handle;
    ds4_tp *verify_handle; /* Existing bound block-gate context, not a getter. */
    uint64_t session_id;
    int rank;
    bool vocab_split;
} ds4_dspark_cycle_transport;

/* Borrowed arguments for one synchronous action, not a session layout.
 * Every owner outlives the action. Session/engine are opaque public handles;
 * all other pointers are actual mutable owners already stored by that session.
 * logits/sample_probs each have config.shape.n_vocab entries. All pointers
 * are non-NULL except cancel, cancel_userdata and the TP handles. No ownership moves.
 * The action updates the checkpoint only after a successful verifier commit. */
typedef struct {
    ds4_session *session;
    ds4_engine *engine;
    ds4_tokens *checkpoint;
    bool *checkpoint_valid;
    float *logits, *sample_probs;
    ds4_session_cancel_fn cancel;
    void *cancel_userdata;
    ds4_dspark_spec_stats *stats;
    ds4_dspark_cycle_policy policy;
    ds4_dspark_cycle_transport tp;
} ds4_dspark_cycle_args;

#endif
