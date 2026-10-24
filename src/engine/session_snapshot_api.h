#ifndef DS4_ENGINE_SESSION_SNAPSHOT_API_H
#define DS4_ENGINE_SESSION_SNAPSHOT_API_H

#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "ds4.h"
#include "payload_io.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Common FILE/memory envelopes borrow the opaque session and call its public
 * model payload API. The C shell keeps backend capability rejection while
 * model codecs own byte layout and checkpoint mutation/commit policy. */
int ds4_native_session_save_snapshot(ds4_session *, ds4_session_snapshot *,
    char *, size_t);
int ds4_native_session_load_snapshot(ds4_session *, const ds4_session_snapshot *,
    char *, size_t);
int ds4_native_session_stage_payload(ds4_session *, ds4_session_payload_file *,
    char *, size_t);

#endif
