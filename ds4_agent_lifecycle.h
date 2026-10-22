#ifndef DS4_AGENT_LIFECYCLE_H
#define DS4_AGENT_LIFECYCLE_H
#include "ds4_agent_summary.h"
#include "ds4_agent_checkpoint.h"
#include "ds4_agent_history.h"

/* Whole live-session actions over the one actual shared Worker. */
bool ds4_agent_worker_load(agent_worker *, const char *, const char *,
    const char *, size_t, ds4_tokens *, agent_kv_session_meta *, char *, size_t);
bool ds4_agent_worker_reset(agent_worker *, char *, size_t);
bool ds4_agent_worker_has_user(agent_worker *);
bool ds4_agent_worker_needs_save(agent_worker *);
bool ds4_agent_worker_save_now(agent_worker *, char[41], int *, char *, size_t);
bool ds4_agent_worker_save(agent_worker *, char *, size_t);
bool ds4_agent_worker_history(agent_worker *, int, char *, size_t);
bool ds4_agent_worker_switch(agent_worker *, const char *, int, char *, size_t);
void ds4_agent_worker_numeric_prefix(ds4_engine *, ds4_think_mode, ds4_tokens *);
void ds4_agent_worker_apply_thinking(agent_worker *);
bool ds4_agent_worker_wait_route(agent_worker *, char *, size_t);
void ds4_agent_worker_deferred_save(agent_worker *);
void ds4_agent_worker_deferred_compact(agent_worker *);
void *ds4_agent_worker_thread(void *);
int ds4_agent_worker_init(agent_worker *, ds4_engine *, agent_config *);
void ds4_agent_worker_free(agent_worker *);

#endif
