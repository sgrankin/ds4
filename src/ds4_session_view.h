#ifndef DS4_SESSION_VIEW_H
#define DS4_SESSION_VIEW_H

#include "ds4_engine_core.h"
#include "ds4_engine_tp_state.h"
#include "ds4_session_frontier.h"
#include "ds4_session_control.h"
#include "ds4_session_resources.h"
#include "ds4_qwen4_session.h"
#include "ds4_glm_session.h"
#include "models/flint/engine.h"

typedef enum {
    DS4_SESSION_DS41,
    DS4_SESSION_FLINT,
    DS4_SESSION_QWEN,
    DS4_SESSION_GLM53,
} ds4_session_kind;

/* A synchronous borrow, not a Session layout. Only the selected family's
 * state pointer is populated. All mutable records are their actual owners;
 * no KV, callback, pending proposal or accounting field is copied here.
 * config41/memory41 remain live until the call and its callbacks return.
 * Flint points at the actual owning slot so resource init/free can publish
 * it without knowing the private C Session's conditional layout. */
typedef struct {
    ds4_session_kind kind;
    ds4_session_frontier *frontier;
    ds4_session_control *control;
    ds4_engine_core *core;
    ds4_model_speculation_state *pending;
    ds4_deepseek41_session_device *ds41;
    ds4_qwen4_session_state *qwen;
    ds4_glm_session_state *glm;
    flint_session **flint;
    ds4_engine_tp_state *transport;
    const ds4_model_config *config41;
    const ds4_engine_memory_inputs *memory41;
    bool exact41;
    uint32_t ablate_mask;
} ds4_session_view;

#endif
