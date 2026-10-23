#ifndef DS4_SESSION_SAMPLING_H
#define DS4_SESSION_SAMPLING_H

#include "ds4_session_frontier.h"
#include "ds4_vocab.h"
#include "ds4_sampling.h"

/* The five existing Engine fields, embedded ONCE. Configuration/token mapping
 * stays with the engine's vocabulary owner; sampling only borrows this policy.
 * Only exact leading-space vocabulary tokens are mapped: penalizing the first
 * subtoken of a split word would suppress unrelated words sharing its prefix. */
typedef struct {
    float penalty;
    bool think_only;
    int after;
    int ids[50];
    int count;
} ds4_overthink_policy;

/* Synchronous borrowed inputs, not an Engine or Session layout. The frontier
 * owns its optional adjusted-logit buffer through the existing C allocator.
 * Raw logprobs/copies always read logits; sampling/greedy actions may read the
 * separately adjusted buffer. No action modifies the raw distribution. */
typedef struct {
    ds4_session_frontier *frontier;
    const ds4_overthink_policy *policy;
    const ds4_vocab *vocab;
    uint32_t count;
} ds4_session_sampling_view;

int ds4_native_session_argmax(const ds4_session_sampling_view *);
int ds4_native_session_argmax_excluding(const ds4_session_sampling_view *, int);
int ds4_native_session_argmax_ignoring_eos(const ds4_session_sampling_view *, ds4_think_mode);
int ds4_native_session_sample(const ds4_session_sampling_view *, float temperature,
    int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_native_session_top_logprobs(const ds4_session_sampling_view *, ds4_token_score *, int);
int ds4_native_session_token_logprob(const ds4_session_sampling_view *, int, ds4_token_score *);
int ds4_native_session_copy_logits(const ds4_session_sampling_view *, float *);
int ds4_native_session_copy_sampling_logits(const ds4_session_sampling_view *, float *);
int ds4_native_session_set_logits(const ds4_session_sampling_view *, const float *);
int ds4_native_frontier_common_prefix(ds4_session_frontier *, const ds4_tokens *);

#endif
