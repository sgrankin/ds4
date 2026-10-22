#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _FORTIFY_SOURCE 0
#include "ds4_agent_bash.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
extern char **environ;
