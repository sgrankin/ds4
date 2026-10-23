#ifndef DS4_AGENT_TERMINAL_API_H
#define DS4_AGENT_TERMINAL_API_H
#include "lifecycle_api.h"
#include "ds4_agent_terminal.h"
/* Existing public symbol declared privately by the original Agent. */
int linenoiseEditInsert(struct linenoiseState *, const char *, size_t);
#include <sys/ioctl.h>
#include <poll.h>
#include <unistd.h>
#endif
