#ifndef DS4_AGENT_SUMMARY_H
#define DS4_AGENT_SUMMARY_H
#include "ds4_agent_turn.h"
#include "ds4_agent_result.h"
#include "ds4_agent_compaction.h"
#include "ds4_agent_summary_prompts.h"
/* Existing actual Worker and whole-action signatures. These declarations
 * already serve Context/Turn/Result; no second state or service ABI is added. */
void ds4_agent_runtime_build_system(agent_worker *, ds4_tokens *);
bool ds4_agent_turn_compact_private(agent_worker *, const char *, int *,
                                    char *, size_t);
bool ds4_agent_result_compact(agent_worker *, const char *, char *, size_t);
#endif
