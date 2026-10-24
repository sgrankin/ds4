#ifndef DS4_ENGINE_GLM53_SESSION_API_H
#define DS4_ENGINE_GLM53_SESSION_API_H

#include "../../ds4_glm_session.h"
#include "../../ds4_sampling.h"
#include "api.h"
#include <limits.h>
#include <sys/time.h>

/* Shared optional logit-dump service, implemented by the native diagnostics owner. */
void ds4_native_session_prefill_logit_dump(const float *, uint32_t vocabulary);

void ds4_native_glm_session_dense_reset(ds4_glm_session_state *,
    ds4_model_speculation_state *);
void ds4_native_glm_session_dense_cap(ds4_glm_session_state *,
    const ds4_session_frontier *);
void ds4_native_glm_session_dense_note(ds4_glm_session_state *,
    const ds4_session_frontier *, uint32_t position, uint32_t count);
int ds4_native_glm_session_sync(ds4_glm_runtime *, const ds4_tokens *,
    char *error, size_t capacity);
int ds4_native_glm_session_eval(ds4_glm_runtime *, int token,
    char *error, size_t capacity);
int ds4_native_glm_session_replay(ds4_glm_runtime *, char *error, size_t capacity);
/* Restore state/logits only. The common rewind owner truncates transcript and
 * clears pending proposals after this call, exactly as for the other models. */
bool ds4_native_glm_session_rewind(ds4_glm_runtime *, uint32_t position);
int ds4_native_glm_session_speculate(ds4_glm_runtime *, int first, int eos,
    float temperature, int top_k, float top_p, float min_p, uint64_t *rng,
    bool exact_sampling, int *accepted, int accepted_capacity,
    char *error, size_t capacity);

#endif
