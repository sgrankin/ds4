#ifndef DS4_AGENT_CONTEXT_RUNTIME_H
#define DS4_AGENT_CONTEXT_RUNTIME_H
#include "ds4_agent_worker.h"
/* Actual Worker is shared once. These two existing whole operations retain
 * progress/status/system-prompt policy until the queue/turn owner migrates. */
void ds4_agent_runtime_build_system(agent_worker *, ds4_tokens *);
int ds4_agent_runtime_sync(agent_worker *, const ds4_tokens *, bool,
                           char *, size_t);

void ds4_agent_runtime_images_clear(agent_worker *);
void ds4_agent_runtime_images_append(agent_worker *, ds4_vision_span *, size_t);
bool ds4_agent_runtime_images_fit(const agent_worker *, const ds4_tokens *);
bool ds4_agent_runtime_cache_allowed(agent_worker *);
bool ds4_agent_runtime_cache_materialize(agent_worker *, ds4_state_cache_ref *,
    ds4_suffix_cache *, ds4_state_cache_kind *, char *, size_t);
bool ds4_agent_runtime_cache_put(agent_worker *, const ds4_suffix_cache *, char[41]);
void ds4_agent_runtime_cache_endpoint(agent_worker *);
bool ds4_agent_runtime_cache_restore(agent_worker *, const ds4_tokens *);
bool ds4_agent_runtime_compact_prefix(agent_worker *, const ds4_tokens *);
void ds4_agent_runtime_frontier_drop(agent_worker *, int);
void ds4_agent_runtime_frontiers_clear(agent_worker *);
int ds4_agent_runtime_frontier_slots(agent_worker *, uint64_t *);
bool ds4_agent_runtime_capture(agent_worker *);
bool ds4_agent_runtime_restore(agent_worker *, const ds4_tokens *);
void ds4_agent_runtime_context_clear(agent_worker *);
char *ds4_agent_runtime_render(agent_worker *, size_t *);
bool ds4_agent_runtime_export(agent_worker *, char *, size_t);
int ds4_agent_runtime_try_suffix(agent_worker *, const ds4_tokens *, const ds4_tokens *, char *, size_t);
int ds4_agent_runtime_import(agent_worker *, char *, size_t);
#endif
