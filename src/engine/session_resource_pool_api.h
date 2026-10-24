#ifndef DS4_ENGINE_SESSION_RESOURCE_POOL_API_H
#define DS4_ENGINE_SESSION_RESOURCE_POOL_API_H
#include "ds4_session_resource_pool.h"

/* CPU-safe admission arithmetic, shared with private legacy host fixtures. */
uint64_t ds4_native_session_graph_budget(const ds4_session_resource_pool *,
    int session_count_hint, uint64_t next_graph_bytes);
#endif
