#ifndef DS4_METAL_EXACT_CACHE_H
#define DS4_METAL_EXACT_CACHE_H
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
static const pthread_once_t ds4_metal_exact_once_init = PTHREAD_ONCE_INIT;

/* The existing lifetime alternatives, shared with the native range owner.
 * Cached views retain one native-cache reference. The other two transfer one
 * retained handle back to ARC; transient views also enter the command's array. */
typedef enum {
    DS4_GPU_EXACT_VIEW_CACHED,
    DS4_GPU_EXACT_VIEW_TRANSIENT,
    DS4_GPU_EXACT_VIEW_OWNED,
} ds4_gpu_exact_view_lifetime;

void ds4_metal_exact_cache_init(void);
void ds4_metal_exact_cache_cleanup(void);
void ds4_metal_exact_cache_clear(const char *reason);
void ds4_metal_exact_cache_maybe_evict(const char *reason);
void ds4_metal_exact_cache_report(void);
void *ds4_metal_exact_cache_wrap(const void *, uint64_t model_size,
    uint64_t offset, uint64_t bytes, uint64_t *inner_offset,
    ds4_gpu_exact_view_lifetime lifetime);

/* True driver/ARC leaves. create returns one retained opaque Metal handle;
 * the native cache or the caller's bridge-transfer owns that reference. */
uint64_t ds4_metal_model_view_max_buffer_length(void);
void *ds4_metal_exact_view_create(const void *address, uint64_t bytes,
    ds4_gpu_exact_view_lifetime lifetime);
void ds4_metal_exact_view_release(void *handle);
void **ds4_metal_exact_cache_state_slot(void);
#endif
