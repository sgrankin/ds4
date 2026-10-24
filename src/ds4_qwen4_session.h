#ifndef DS4_QWEN4_SESSION_H
#define DS4_QWEN4_SESSION_H

#include "ds4_qwen4_graph.h"
#include "ds4_model_speculation.h"
#include "ds4_session_control.h"

/* Actual model session storage, embedded once in Session. The host logit
 * allocation retains libc ownership; graph ownership is unchanged. */
typedef struct {
    ds4_qwen4_gpu_graph graph;
    bool ready;
    int slot; /* -1: private recurrence, otherwise an engine pool slot. */
    bool rewound;
    float *verify_logits;
    uint64_t spec_cycles;
    uint64_t spec_accepted;
    uint32_t depth_window;
    uint32_t depth_accepted;
    uint32_t depth_cycles;
    uint32_t reject2_streak;
    bool depth3_engaged;
} ds4_qwen4_session_state;

/* Engine-wide measured policy for the existing batched speculative schedule. */
typedef struct {
    float p;
    float ms[2];
    uint32_t cycles;
    uint32_t spec_cycles;
    uint32_t width;
} ds4_qwen4_batch_policy;

/* Borrowed for one synchronous call. sync re-enters the common public session
 * service during stale-state replay (validation/image identity/observability),
 * whose Qwen dispatch invokes the native controller below. No C model graph
 * or speculative algorithm is called through this callback. */
typedef struct {
    ds4_qwen4_session_state *state;
    ds4_session_frontier *frontier;
    ds4_model_speculation_state *pending;
    ds4_session_control *control;
    const ds4_model *model;
    const ds4_weights *weights;
    bool exact_sampling;
    int (*sync)(void *user, const ds4_tokens *tokens, char *error, size_t capacity);
    void *user;
} ds4_qwen4_runtime;

int ds4_native_qwen4_session_sync(ds4_qwen4_runtime *runtime,
    const ds4_tokens *prompt, char *error, size_t capacity);
int ds4_native_qwen4_session_eval(ds4_qwen4_runtime *runtime, int token,
    char *error, size_t capacity);
/* Restore graph/logits only. The common caller truncates the checkpoint and
 * clears pending speculation after this model operation. */
void ds4_native_qwen4_session_rewind(ds4_qwen4_session_state *state,
    ds4_session_frontier *frontier, uint32_t position);
int ds4_native_qwen4_session_speculate(ds4_qwen4_runtime *runtime, int first,
    float temperature, int top_k, float top_p, float min_p, uint64_t *rng,
    bool exact_sampling, int *accepted, int accepted_capacity,
    char *error, size_t capacity);
/* Caller validates engine/backend/token admission first. On failure it must
 * invalidate every affected session through the common public service. */
int ds4_native_qwen4_batch_speculate(ds4_qwen4_runtime *runtimes,
    const int *tokens, uint32_t count, ds4_qwen4_gpu_graph *arena,
    ds4_qwen4_batch_policy *policy, int (*accepted)[2], int *counts,
    char *error, size_t capacity);

#endif
