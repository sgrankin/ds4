#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _FORTIFY_SOURCE 0
#include "ds4_conversations.h"
#include "ds4_server_output.h"
#include "ds4_response_store.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Import the SDK initializer and actual pthread type, never a Darwin layout. */
static const pthread_mutex_t ds4_conversation_mutex_init = PTHREAD_MUTEX_INITIALIZER;
