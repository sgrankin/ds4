#ifndef DS4_ENGINE_CORE_API_H
#define DS4_ENGINE_CORE_API_H

/* Include this from the existing graph_api.h/shared gguf_c declaration Module.
 * Owners are ordinary pure Zig imports in the ONE graph object. */
#include "ds4_engine_core.h"
#include <float.h>


/* Existing process services and mmap/page warming. The instance lock is
 * acquired before opening the target by C's one-family routing prologue. */
void ds4_core_profile_init(const char *, const char *);
void ds4_core_profile_close(void);
void ds4_core_threads_shutdown(void);
void ds4_core_release_instance_lock(void);
void ds4_core_native_admission_acquire(void);
void ds4_core_native_admission_release(void);
void ds4_core_warm_model(const ds4_model *);
void ds4_core_warm_shard(const ds4_model *, const ds4_weights *, int);
uint64_t ds4_core_wired_limit_bytes(void);
void ds4_core_reset_streaming(void);
void ds4_core_memory_summary(const ds4_engine_core *, int);
#endif
