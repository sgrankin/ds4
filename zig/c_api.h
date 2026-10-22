#ifndef DS4_ZIG_C_API_H
#define DS4_ZIG_C_API_H

/* Declaration-only bridge for handwritten Zig modules. Keep generated output
 * in Zig's build cache; the public C headers remain the layout authority. */
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0
#endif

#include "ds4_engram.h"
#include "ds4_kvstore.h"
#include "ds4_prompt_prefix.h"
#include <ctype.h>
#include <dirent.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Import the SDK initializer with its declared platform type. */
static const pthread_once_t ds4_zig_pthread_once_init = PTHREAD_ONCE_INIT;

#endif
