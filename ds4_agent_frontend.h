#ifndef DS4_AGENT_FRONTEND_H
#define DS4_AGENT_FRONTEND_H
#include "ds4_agent_terminal.h"
#include <signal.h>

/* No additional Worker/UI record. The sole facade owns the original
 * completion callback slot and signal flag; pure owners borrow their storage. */
/* Same sole callback slot; existing embedded fixture borrows it. */
extern agent_worker *ds4_agent_completion_worker;
void ds4_agent_frontend_complete(const char *, linenoiseCompletions *);
int ds4_agent_frontend_main(int, char **, ds4_backend);
agent_config ds4_agent_frontend_options(int, char **, ds4_backend);
void ds4_agent_frontend_defaults(ds4_engine *, agent_generation_options *);
int ds4_agent_frontend_headless(ds4_engine *, agent_config *);
int ds4_agent_frontend_interactive(ds4_engine *, agent_config *);
bool ds4_agent_frontend_power(const char *, int *);
bool ds4_agent_frontend_steering(const char *, float *);
bool ds4_agent_frontend_hints(const char *, bool *);
bool ds4_agent_frontend_with_args(const char *, const char *);
bool ds4_agent_frontend_known(const char *);

/* Actual OS signal action, with the caller's original handler/mask restored.
 * install=false retires it. Signal state itself belongs only to the facade. */
bool ds4_agent_frontend_signal(bool install, void (*handler)(int), struct sigaction *previous);
#endif
