#ifndef DS4_ENGINE_SESSION_API_H
#define DS4_ENGINE_SESSION_API_H

#include "ds4_session_view.h"
#include "payload_io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Ordinary single-Metal ownership. The compatibility shell only captures
 * these borrowed owners; legacy/reference/distributed routes stay separate. */
int ds4_native_session_sync(ds4_session_view *, const ds4_tokens *, char *, size_t);
int ds4_native_session_multimodal_sync(ds4_session_view *, const ds4_tokens *,
    const ds4_vision_span *, size_t count, char *, size_t);
int ds4_native_session_eval(ds4_session_view *, int token, char *, size_t);
void ds4_native_session_invalidate(ds4_session_view *);
float ds4_native_session_steering_scale(const ds4_session_view *);
int ds4_native_session_steering_set(ds4_session_view *, float scale);
ds4_session_rewrite_result ds4_native_session_rewrite(ds4_session *,
    const ds4_session_frontier *, const ds4_tokens *, int common,
    char *error, size_t error_capacity);
/* Qwen/GLM/Flint common continuation policy. DS41 retains its complete native
 * coordinator; private compatibility paths never enter these routes. */
int ds4_native_session_speculate(ds4_session_view *, int first, int maximum,
    int eos, float temperature, int top_k, float top_p, float min_p,
    uint64_t *rng, bool ignore_eos, int *accepted, int accepted_capacity,
    char *error, size_t error_capacity);
void ds4_native_session_rewind(ds4_session_view *, int position);
uint64_t ds4_native_session_payload_bytes(ds4_session_view *);
int ds4_native_session_payload_save(ds4_session_view *, FILE *, char *, size_t);
int ds4_native_session_payload_load(ds4_session_view *, FILE *, uint64_t bytes,
    char *, size_t);

/* CPU-safe owners: no model/GPU call enters these objects. Existing frontier
 * vision ABI names remain unchanged when moved out of the hot graph object. */
void ds4_native_control_progress(ds4_session_control *,
    ds4_session_progress_fn, void *);
void ds4_native_control_display(ds4_session_control *,
    ds4_session_progress_fn, void *);
void ds4_native_control_cancel(ds4_session_control *,
    ds4_session_cancel_fn, void *);
bool ds4_native_control_cancelled(const ds4_session_control *);
void ds4_native_control_report(const ds4_session_control *, const char *, int, int);

#endif
