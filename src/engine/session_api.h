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
