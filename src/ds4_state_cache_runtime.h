#ifndef DS4_STATE_CACHE_RUNTIME_H
#define DS4_STATE_CACHE_RUNTIME_H

#include "ds4_state_cache.h"
#include "ds4_kvstore.h"
#include "ds4_tp.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Two frontends use the same opt-in cache settings and weight/policy identity.
 * A directory has one process owner; server clients share that owner. */
extern char **environ;
/* Shared Zig transaction used by both frontends. */
ds4_state_cache *ds4_state_cache_runtime_open(ds4_engine *engine,
    const ds4_engine_options *options, int ctx, char profile[41],
    char *err, size_t errlen);

#endif
