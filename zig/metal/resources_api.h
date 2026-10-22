#ifndef DS4_ZIG_METAL_RESOURCES_API_H
#define DS4_ZIG_METAL_RESOURCES_API_H
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _FORTIFY_SOURCE 0
#include "resources.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* Import the SDK's typed initializer rather than guessing pthread layout. */
static const pthread_mutex_t ds4_metal_resource_mutex_init = PTHREAD_MUTEX_INITIALIZER;
#endif
