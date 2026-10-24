#ifndef DS4_GLM_SESSION_H
#define DS4_GLM_SESSION_H

#include "ds4_glm_graph.h"
#include "ds4_model_speculation.h"
#include "ds4_session_frontier.h"
#include "ds4_session_control.h"
#include "ds4_engine_core.h"

/* Actual model session storage, embedded once in Session. Device allocations
 * belong to graph; engine-wide admission counters account reserved_graph_bytes.
 * Pending host MTP buffers remain in the shared model_speculation record. */
typedef struct {
    ds4_glm_gpu_graph graph;
    bool ready;
    uint32_t dense_cache_len;
    uint64_t reserved_graph_bytes;
} ds4_glm_session_state;

/* Borrowed for a synchronous operation. The optional public sync service owns
 * common request/image validation and observability; its GLM53 dispatch returns
 * to the native model controller. No C model algorithm crosses this boundary. */
typedef struct {
    ds4_glm_session_state *state;
    ds4_session_frontier *frontier;
    ds4_model_speculation_state *pending;
    ds4_session_control *control;
    const ds4_engine_core *core;
    uint32_t ablate_mask;
    int (*sync)(void *user, const ds4_tokens *, char *error, size_t capacity);
    void *user;
} ds4_glm_runtime;

#endif
