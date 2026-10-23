#ifndef DS4_AGENT_COMMUNICATIONS_H
#define DS4_AGENT_COMMUNICATIONS_H
#include "ds4_agent_worker.h"
/* Whole existing mutex/pipe/condition operations over the SAME Worker. No
 * field getter ABI or second synchronized copy of its state is introduced. */
void ds4_agent_comms_wake_locked(agent_worker *);
void ds4_agent_comms_publish(agent_worker *, const char *, size_t);
void ds4_agent_comms_status(agent_worker *, agent_worker_state);
void ds4_agent_comms_error(agent_worker *, const char *);
void ds4_agent_comms_progress(void *, const char *, int, int);
bool ds4_agent_comms_interrupted(agent_worker *);
void ds4_agent_comms_clear_interrupt(agent_worker *);
bool ds4_agent_comms_cancel(void *);
int ds4_agent_comms_web_confirm(void *, const char *, char *, size_t);
bool ds4_agent_comms_take_web(agent_worker *, char *, size_t);
void ds4_agent_comms_answer_web(agent_worker *, bool, const char *);
char *ds4_agent_comms_request_drain(agent_worker *);
bool ds4_agent_comms_take_drain(agent_worker *);
void ds4_agent_comms_answer_drain(agent_worker *, char *);
void ds4_agent_comms_request_save(agent_worker *);
void ds4_agent_comms_request_compact(agent_worker *);
void ds4_agent_comms_request_power(agent_worker *, int);
bool ds4_agent_comms_take_save(agent_worker *);
bool ds4_agent_comms_take_compact(agent_worker *);
void ds4_agent_comms_apply_power(agent_worker *);
bool ds4_agent_comms_take_raw_restore(agent_worker *);
void ds4_agent_comms_drain_fd(int);
bool ds4_agent_comms_submit(agent_worker *, const char *);
void ds4_agent_comms_interrupt(agent_worker *);
void ds4_agent_comms_stop(agent_worker *);
void ds4_agent_comms_consume(agent_worker *, char **, size_t *, agent_status *);
void ds4_agent_comms_get_status(agent_worker *, agent_status *);
bool ds4_agent_comms_idle(agent_worker *);
bool ds4_agent_comms_initialized(agent_worker *, agent_status *);
/* One original process-wide label sequence. Its storage lives only in the
 * sole facade; pure owner imports cannot accidentally instantiate duplicates. */
unsigned ds4_agent_comms_next_label(void);
#endif
