#ifndef DS4_AGENT_HISTORY_H
#define DS4_AGENT_HISTORY_H
#include "ds4_agent_stream.h"

#define DS4_AGENT_HISTORY_DEFAULT_TURNS 3
#define DS4_AGENT_HISTORY_MAX_TURNS 200
#define DS4_AGENT_HISTORY_ASSISTANT_MAX_LINES 80
#define DS4_AGENT_HISTORY_ASSISTANT_MAX_BYTES 12000

/* Borrowed transcript and the existing output action. The opaque context is
 * also passed to Stream's existing trace action; it remains the actual Worker.
 * Replay never executes tools or mutates transcript, session or KV owners. */
void ds4_agent_history_render(ds4_engine *, agent_tool_syntax, void *,
    ds4_agent_presentation_emit_fn, bool use_color, const char *, size_t,
    int user_turns);
#endif
