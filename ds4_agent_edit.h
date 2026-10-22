#ifndef DS4_AGENT_EDIT_H
#define DS4_AGENT_EDIT_H
#include "ds4_agent_tool_actions.h"

#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2

/* Actual per-generation state, moved once. The parser/call/file bytes remain
 * borrowed; a successful unique-prefix probe marks this argument done. */
typedef struct {
    bool active;
    bool done;
} agent_edit_upto_forcer;

/* Results are libc-owned strings. allow_upto is the existing caller policy;
 * the owner has no Worker/config/session record or UI side effects. */
char *ds4_agent_edit_write(const agent_tool_call *);
char *ds4_agent_edit_apply(bool allow_upto, const agent_tool_call *);
bool ds4_agent_edit_preflight(bool allow_upto, const agent_tool_call *,
    char *error, size_t error_size);
bool ds4_agent_edit_force_upto(agent_edit_upto_forcer *, agent_dsml_parser *,
    const char *next_text, size_t next_length);
#endif
