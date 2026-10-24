#ifndef DS4_ENGINE_SESSION_BATCH_API_H
#define DS4_ENGINE_SESSION_BATCH_API_H
#include "engine/session_api.h"
#include "ds4_session_resource_pool.h"

/* Actual component views/tokens are borrowed for this call only. C captures
 * the records; native owns validation, optimized dispatch and publication.
 * DS41 retains its existing complete native coordinator. */
int ds4_native_sessions_eval_batch(ds4_session_view *, const int *, int count,
    ds4_session_resource_pool *, char *, size_t);
int ds4_native_sessions_speculative_batch(ds4_session_view *, const int *,
    int count, ds4_session_resource_pool *, ds4_qwen4_batch_policy *,
    int (*accepted)[2], int *counts, char *, size_t);
int ds4_native_sessions_mixed_batch(ds4_session_view *, const int *, int count,
    ds4_session_resource_pool *, ds4_session_view *prefill,
    const ds4_tokens *, char *, size_t);
#endif
