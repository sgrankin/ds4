#ifndef DS4_ENGINE_STREAM_SELECTED_API_H
#define DS4_ENGINE_STREAM_SELECTED_API_H

#include "stream_io.h"
#include "metal/context.h"
#include <stdio.h>
#include <string.h>

/* Import SDK initializers and actual pthread types, never a Darwin layout. */
static const pthread_mutex_t ds4_stream_selected_mutex_init = PTHREAD_MUTEX_INITIALIZER;
static const pthread_cond_t ds4_stream_selected_cond_init = PTHREAD_COND_INITIALIZER;

#endif
