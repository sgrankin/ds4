#ifndef DS4_METAL_EXPERT_LOAD_H
#define DS4_METAL_EXPERT_LOAD_H
#include "expert_cache.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <sys/resource.h>
#ifdef __OBJC__
#define DS4_LOAD_STRONG __strong
#else
#define DS4_LOAD_STRONG
#endif
/* Actual pread/pending records moved once; remaining C and native callers
 * borrow the SAME state. Strong fields are written only by ARC actions. */
typedef struct {
    uint64_t offset;
    uint64_t len;
    uint8_t *dst;
    uint64_t read_bytes;
    double ms;
    int ok;
    int done; /* Published under the pread-pool mutex. */
    int gate_stage;
} ds4_gpu_stream_expert_pread_task;

typedef struct {
    int active;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t missing_mask;
    uint32_t load_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t source_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t n_loads;
    uint32_t n_tasks;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t gate_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t up_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t down_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    DS4_LOAD_STRONG ds4_metal_expert_buffer_ref
        gate_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    DS4_LOAD_STRONG ds4_metal_expert_buffer_ref up_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    DS4_LOAD_STRONG ds4_metal_expert_buffer_ref
        down_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uintptr_t gate_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uintptr_t up_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uintptr_t down_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    ds4_gpu_stream_expert_pread_task tasks[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED * 3u];
    double start_ms;
    double prepare_ms;
} ds4_gpu_stream_expert_pending_load;

typedef struct {
    int active;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
} ds4_gpu_glm_stream_selected_prefetch;

typedef struct {
    ds4_gpu_stream_expert_pread_task *tasks;
    uint32_t n_tasks;
    uint32_t worker_index;
    uint32_t n_workers;
} ds4_gpu_stream_expert_pread_worker_args;

#undef DS4_LOAD_STRONG
/* Original persistent pool fields, grouped once with the two actual records.
 * C initializes these SDK mutex/condition objects with their real constants. */
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    pthread_t threads[18];
    uint32_t thread_count;
    uint32_t active_workers;
    uint32_t remaining_workers;
    uint32_t n_tasks;
    uint32_t next_task;
    uint64_t generation;
    ds4_gpu_stream_expert_pread_task *tasks;
    int initialized;
    int stopping;
} ds4_metal_expert_pread_pool;
typedef struct {
    ds4_gpu_stream_expert_pending_load pending;
    ds4_gpu_glm_stream_selected_prefetch prefetch;
    ds4_metal_expert_pread_pool pool;
} ds4_metal_expert_load_state;
typedef struct {
    ds4_metal_expert_cache_state *cache;
    ds4_metal_expert_load_state *load;
    int model_fd;
    bool streaming;
    bool device_ready;
    bool batch_active;
    bool quality;
    bool prefill_building;
    bool glm_building;
} ds4_metal_expert_load_view;
/* One coherent immutable driver snapshot plus the actual state addresses. */
ds4_metal_expert_load_view ds4_metal_expert_load_driver(void);
typedef struct ds4_metal_expert_owned_buffer ds4_metal_expert_owned_buffer;
ds4_metal_expert_owned_buffer *ds4_metal_expert_buffer_new(uint64_t bytes, const char *label);
void ds4_metal_expert_buffer_release(ds4_metal_expert_owned_buffer *buffer);
void ds4_metal_expert_reuse_assign(ds4_gpu_stream_expert_reusable_buffers *out,
                                   ds4_metal_expert_buffer_ref gate, ds4_metal_expert_buffer_ref up,
                                   ds4_metal_expert_buffer_ref down, uintptr_t gate_inner,
                                   uintptr_t up_inner, uintptr_t down_inner);
void ds4_metal_expert_slab_assign(ds4_metal_expert_cache_state *, uint32_t,
                                  ds4_metal_expert_buffer_ref);
void ds4_metal_expert_pending_assign(ds4_gpu_stream_expert_pending_load *, uint32_t,
                                     const ds4_gpu_stream_expert_reusable_buffers *);
void ds4_metal_expert_pending_retire(ds4_gpu_stream_expert_pending_load *);
/* Existing command/storage leaves, no scheduling or cache policy. */
typedef struct {
    struct ds4_metal_model_buffer *gate, *up, *down;
    uintptr_t gate_inner, up_inner, down_inner;
} ds4_metal_expert_copy_sources;
bool ds4_metal_expert_seed_blit(const ds4_metal_expert_copy_sources *sources,
                                const ds4_gpu_stream_expert_reusable_buffers *destinations,
                                uint32_t count, uint64_t gate_bytes, uint64_t down_bytes);
bool ds4_metal_expert_load_addresses(uint32_t layer,
                                     ds4_gpu_stream_expert_reusable_buffers *addresses);
bool ds4_metal_expert_load_address_install(uint32_t layer, uint32_t expert,
                                           const ds4_gpu_stream_expert_reusable_buffers *buffers);
/* Sole native facade. Original private C wrappers retain their signatures and
 * output-pointer/ARC writeback contracts; no controller body is forwarded. */
int ds4_native_expert_pending_finish(ds4_gpu_stream_expert_cache_entry **);
void ds4_native_expert_pending_clear(void);
int ds4_native_expert_pending_matches(const ds4_gpu_stream_expert_table *, const int32_t *,
                                      uint32_t);
int ds4_native_expert_pending_gate_wait(void);
void ds4_native_expert_pread_shutdown(void);
uint32_t ds4_native_expert_locked_slots(void);
int ds4_native_expert_load_missing(const ds4_gpu_stream_expert_table *, const int32_t *, uint32_t,
                                   const uint64_t *, const uint64_t *, const uint64_t *, uint32_t,
                                   ds4_gpu_stream_expert_cache_entry **);
int ds4_native_expert_prefetch_take(const ds4_gpu_stream_expert_table *, uint32_t, int32_t *);
int ds4_native_expert_prepare_batch(const ds4_gpu_stream_expert_table *, const ds4_gpu_tensor *,
                                    uint32_t, uint32_t, ds4_gpu_stream_expert_reusable_buffers *,
                                    ds4_gpu_stream_expert_cache_entry **, uint32_t *, uint32_t *,
                                    ds4_gpu_stream_expert_reusable_buffers *);
int ds4_native_expert_address_requested(void);
int ds4_native_expert_prefill_enabled(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
#endif
