#ifndef DS4_AGENT_CONTEXT_H
#define DS4_AGENT_CONTEXT_H

#include "ds4.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The actual existing turn record, moved once out of ds4_agent.c. IDs identify
 * real role tokens in this worker; display labels never enter model input. */
typedef struct {
    uint64_t id;
    int start, end, role_token;
} agent_live_context_turn;

/* Cohesive extraction of the worker's existing snapshot/turns/count fields,
 * not a copy of its layout. Worker embeds this owner and all callers use it.
 * Tokens/turn storage is libc-owned and released by context_clear. */
typedef struct {
    ds4_tokens snapshot;
    agent_live_context_turn *turns;
    size_t turn_count;
} ds4_agent_context;

/* A staged text edit owns no graph/session. prepare never mutates transcript
 * or Context. Only commit, after the caller's successful graft/exact decision,
 * moves these owners. ready also distinguishes a labelled no-op from a raw
 * byte-identical no-op; structured no-ops still update their retained ID map. */
typedef struct {
    ds4_tokens tokens;
    agent_live_context_turn *turns;
    size_t turn_count;
    bool ready;
    bool structured;
} ds4_agent_context_edit;

void ds4_agent_context_clear(ds4_agent_context *context);
char *ds4_agent_context_body_text(const char *text, size_t len, bool quote);
char *ds4_agent_context_render(ds4_engine *engine, ds4_tokens *transcript,
    const ds4_tokens *system, ds4_agent_context *context, size_t *out_len);
bool ds4_agent_context_text_safe(ds4_engine *engine, const ds4_tokens *tokens,
    const char *text, size_t len, char *err, size_t errlen);
/* text is len bytes with a trailing NUL, as returned by agent_file read.
 * Return -1 rejected, 0 unchanged, 1 changed. Errors leave both owners intact;
 * free the staged edit on every exit. Context budget includes the old256-token
 * acknowledgment/prefix reserve. No session mutation/token evaluation occurs. */
int ds4_agent_context_prepare(ds4_engine *engine, ds4_tokens *transcript,
    const ds4_tokens *system, ds4_agent_context *context,
    const char *text, size_t len, size_t image_count, int ctx_size,
    ds4_agent_context_edit *edit, char *err, size_t errlen);
void ds4_agent_context_commit(ds4_tokens *transcript, ds4_agent_context *context,
    ds4_agent_context_edit *edit);
void ds4_agent_context_edit_free(ds4_agent_context_edit *edit);

#endif
