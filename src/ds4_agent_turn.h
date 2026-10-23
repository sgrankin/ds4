#ifndef DS4_AGENT_TURN_H
#define DS4_AGENT_TURN_H
#include "ds4_agent_communications.h"
#include "ds4_agent_stream.h"
#include "ds4_agent_edit.h"
#include "ds4_agent_prompts.h"
int ds4_agent_turn_run(agent_worker *, const char *);
int ds4_agent_turn_raw(agent_worker *, const char *);
int ds4_agent_turn_sync(agent_worker *, const ds4_tokens *, bool, char *, size_t);
/* One remaining whole private summary inference/graph-rebuild transaction. */
bool ds4_agent_turn_compact_private(agent_worker *, const char *, int *, char *, size_t);
char *ds4_agent_turn_tools_prompt(ds4_engine *, bool);
char *ds4_agent_turn_dsml_prompt(bool, bool);
char *ds4_agent_turn_glm_prompt(bool, bool);
char *ds4_agent_turn_qwen_prompt(bool, bool);
char *ds4_agent_turn_dsml41_prompt(const char *);
const char *ds4_agent_turn_hints_note(agent_hints *);
void ds4_agent_turn_hints_compacted(agent_hints *);
void ds4_agent_turn_hints_context(agent_worker *);
void ds4_agent_turn_append_system(ds4_engine *, ds4_tokens *, const char *, bool);
void ds4_agent_turn_system_status(agent_worker *, const char *);
void ds4_agent_turn_datetime(agent_worker *);
void ds4_agent_turn_system_reminder(agent_worker *);
void ds4_agent_turn_trace_message(agent_worker *, const char *);
void ds4_agent_turn_trace_text(agent_worker *, const char *, const char *, size_t);
void ds4_agent_turn_trace_token(agent_worker *, int, const char *, size_t, int);
void ds4_agent_turn_trace_tokens(agent_worker *, const char *, const ds4_tokens *, int);
#endif
