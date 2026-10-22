#ifndef DS4_SERVER_PERSIST_H
#define DS4_SERVER_PERSIST_H

#include "ds4_server_tool_memory.h"

/* Complete cache transactions over the actual runtime records. Canonical
 * payload, replay and suffix bytes remain owned by their public libraries. */
bool ds4_server_persist_store(server *runtime, server_slot *slot,
    const ds4_tokens *tokens, int length, const char *reason,
    const char *visible, uint8_t extension, const char *key);
void ds4_server_persist_store_current(server *runtime, server_slot *slot,
    const char *reason);
int ds4_server_persist_continued_target(server *runtime, server_slot *slot,
    int live_tokens);
void ds4_server_persist_note_store(server_slot *slot, int tokens);
int ds4_server_persist_suppress(server *runtime, server_slot *slot, int tokens);
void ds4_server_persist_restore_suppressed(server_slot *slot, int old_tokens,
    int suppressed_tokens);
void ds4_server_persist_discard(server *runtime, server_slot *slot,
    const char *path);
void ds4_server_persist_maybe_store(server *runtime, server_slot *slot);
int ds4_server_persist_load(server *runtime, server_slot *slot,
    const char *prompt_text, ds4_tokens *effective_prompt, char **path,
    uint8_t *extension, bool responses);
void ds4_server_persist_restore_tools(server *runtime,
    const chat_msgs *messages);
/* load_state is called while inference admission owns the session/cache. */
int ds4_server_persist_load_state(server *runtime, server_slot *slot,
    const ds4_tokens *prompt);
void ds4_server_persist_endpoint(server *runtime, server_slot *slot);
int ds4_server_persist_suffix(server *runtime, server_slot *slot,
    const ds4_tokens *prompt, ds4_tokens *padded,
    ds4_suffix_relocation_stats *stats, const volatile sig_atomic_t *stop);

/* One existing synchronized diagnostic action. No object/state is queried and
 * the borrowed complete message does not escape the call. */
void ds4_server_persist_log(ds4_log_type type, const char *message);

#endif
