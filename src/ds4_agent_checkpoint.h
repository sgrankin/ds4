#ifndef DS4_AGENT_CHECKPOINT_H
#define DS4_AGENT_CHECKPOINT_H
#include "ds4_agent_sessions.h"

/* One private native owner for the actual open FILE/text/title/header. Engine
 * and Session are borrowed; no Worker/Session layout is copied into this ABI. */
typedef struct ds4_agent_checkpoint_load ds4_agent_checkpoint_load;

/* Return -1 rejected, 0 payload loaded, 1 ordinary sync required. On success,
 * out receives one owner; a rebuild publishes a fresh owned token vector.
 * Worker performs its existing approximate invalidation and sync/progress/
 * cancellation actions, then calls finish, and frees the load on every exit.
 * The held file stays open through sync, matching the original transaction. */
bool ds4_agent_checkpoint_requires_rebuild(uint64_t payload_bytes, bool tensor_parallel);
int ds4_agent_checkpoint_prepare_load(ds4_engine *, ds4_session *, const char *path,
    const char *expected_sha, const char *expected_text, size_t expected_text_len,
    bool tensor_parallel, ds4_tokens *rebuilt, ds4_agent_checkpoint_load **out,
    char *error, size_t error_cap);
bool ds4_agent_checkpoint_finish_load(ds4_agent_checkpoint_load *, bool synchronized,
    ds4_tokens *loaded, agent_kv_session_meta *, char *error, size_t error_cap);
void ds4_agent_checkpoint_free_load(ds4_agent_checkpoint_load *);

/* Complete staging/write/close/rename action. Worker retains its context-sized
 * text-only decision and commits identity/status only after success. */
bool ds4_agent_checkpoint_save(ds4_engine *, ds4_session *, const char *path,
    const ds4_tokens *, const char *reason, char sha_out[41], const char *title,
    uint64_t created_at, bool text_only, char *error, size_t error_cap);
#endif
