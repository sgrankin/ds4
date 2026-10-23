#ifndef DS4_AGENT_EXECUTOR_H
#define DS4_AGENT_EXECUTOR_H
#include "ds4_agent_tool_actions.h"
#include "ds4_agent_edit.h"
#include "ds4_agent_bash.h"
#include "ds4_agent_presentation.h"
#include "ds4_web.h"

#define DS4_AGENT_WEB_HEAD_BYTES (8 * 1024)
#define DS4_AGENT_WEB_HEAD_LINES 100

typedef struct agent_bash_job agent_bash_job;
/* Process owns its monitor, descriptors and process group. ToolRuntime owns
 * the list node and its next link; the process itself stays opaque. */
struct agent_bash_job {
    ds4_agent_bash *process;
    struct agent_bash_job *next;
};

/* The four existing Worker fields form one actual tool runtime owner. Web is
 * created/freed by Worker startup/shutdown; native actions borrow its handle. */
typedef struct {
    ds4_agent_pager pager;
    agent_bash_job *jobs;
    int next_job_id;
    ds4_web *web;
} ds4_agent_tool_runtime;

/* Immutable whole-action inputs, not a Worker record. The output/cancel and
 * terminal-change callbacks are the existing actions used by the public
 * session/Bash APIs. No field getters or per-tool services are introduced. */
char *ds4_agent_executor_call(ds4_agent_tool_runtime *, int context_size,
    bool allow_upto, bool non_interactive, bool use_color, void *context,
    ds4_agent_presentation_emit_fn, ds4_session_cancel_fn,
    ds4_agent_bash_terminal_fn, const agent_tool_call *);
agent_bash_job *ds4_agent_executor_start(ds4_agent_tool_runtime *, const char *,
    int timeout_sec, ds4_agent_bash_terminal_fn, void *context, char *, size_t);
void ds4_agent_executor_jobs_free(ds4_agent_tool_runtime *);
agent_bash_job *ds4_agent_executor_find(ds4_agent_tool_runtime *, int id, pid_t);
void ds4_agent_executor_remove(ds4_agent_tool_runtime *, agent_bash_job *);
char *ds4_agent_executor_compaction_observation(ds4_agent_tool_runtime *);
char *ds4_agent_executor_terminal_safe(const char *, size_t);
#endif
