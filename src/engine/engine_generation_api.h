#ifndef DS4_ENGINE_GENERATION_API_H
#define DS4_ENGINE_GENERATION_API_H
#include "ds4_engine_core.h"
#include "ds4_qwen4_graph.h"
#include "ds4_sampling.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Existing optimized top-one facade; its ordinary DS41 route is native. */
int ds4_session_eval_argmax(ds4_session *, int token, char *error, size_t capacity);
/* Opaque engine/session APIs for DS41, GLM and Flint; Qwen's stateless native
 * graph borrows the actual Core/config to retain final-head-only prefill. */
int ds4_native_engine_generate_argmax(ds4_engine *, const ds4_engine_core *,
    const ds4_qwen4_config *, const ds4_tokens *, int count, int context,
    ds4_token_emit_fn, ds4_generation_done_fn, void *,
    ds4_session_progress_fn, void *);
#endif
