#ifndef DS4_AGENT_RESULT_H
#define DS4_AGENT_RESULT_H
#include "ds4_agent_context_runtime.h"
#include "ds4_agent_compaction.h"
/* Worker retains parser/Observation cleanup and status publication after the
 * returned whole transaction. Observation is borrowed and may be replaced by
 * the existing bounded retry receipt; its storage remains caller-owned. */
enum { DS4_AGENT_RESULT_OK = 0, DS4_AGENT_RESULT_INTERRUPTED = 1,
       DS4_AGENT_RESULT_FAILED = 2 };
int ds4_agent_result_after_tools(agent_worker *, agent_tool_observation *);
int ds4_agent_result_append(agent_worker *, agent_tool_observation *, char *, size_t);
/* Temporary complete private summary inference/rebuild/rollback action. */
bool ds4_agent_result_compact(agent_worker *, const char *, char *, size_t);
bool ds4_agent_result_commit(agent_worker *, const agent_tool_observation *, char *, size_t);
#endif
