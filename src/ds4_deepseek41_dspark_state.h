#ifndef DS4_DEEPSEEK41_DSPARK_STATE_H
#define DS4_DEEPSEEK41_DSPARK_STATE_H

#include "ds4_deepseek41_graph.h"
#include "ds4_dspark_weights.h"

#define DS41_DRAFT_WINDOW 128u
#define DS41_DRAFT_ROWS 5u
#define DS41_VERIFY_ROWS (DS41_DRAFT_ROWS + 1u)

/* Actual V4.1 draft owner. Sessions inspect the accepted feature positions,
 * host sampling rows and rewind frontier directly during the transition.
 * verified[] are shallow target-graph copies: only their explicitly retained
 * residual/pre/logits/pool views belong to this owner, never their caches. */
struct ds41_dspark {
    const ds4_dspark_weights *weights;
    ds41_prefill_row batch;
    ds4_gpu_tensor *target[3], *window[3];
    uint32_t positions[3][DS41_DRAFT_WINDOW];
    uint32_t kv_end;
    bool kv_valid;
    ds4_gpu_tensor *mean_weights, *mean, *input, *projected, *main_x, *main_kv;
    ds4_gpu_tensor *logits, *tokens, *markov_embed, *markov_logits, *confidence_input, *confidence;
    float *host_logits, *draft_probs;
    ds4_gpu_tensor *verify_residual, *verify_pre, *verify_logits, *verify_pool;
    ds4_gpu_tensor *old_raw, *old_features;
    uint32_t old_positions[3][DS41_VERIFY_ROWS];
    ds41_gpu_graph verified[DS41_VERIFY_ROWS + 1u];
    float *host_verified;
    uint32_t verify_start, verify_count, rewind_end;
};

#endif
