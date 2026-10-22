#ifndef DS4_ZIG_METAL_MODEL_VIEWS_H
#define DS4_ZIG_METAL_MODEL_VIEWS_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_METAL_MAX_MODEL_VIEWS 4096
/* Compatibility callers without a parsed GGUF maximum tensor span. */
#define DS4_METAL_FALLBACK_MAX_TENSOR_BYTES (4ull * 1024ull * 1024ull * 1024ull)

/* One real record shared by the native owner and the ARC leaves. Native code
 * never writes buffer: assignment, retirement and compaction use ARC leaves.
 * C imports see only its opaque pointer; Objective-C keeps its strong field. */
typedef struct {
#ifdef __OBJC__
    __strong id<MTLBuffer> buffer;
#else
    void *buffer;
#endif
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
} ds4_gpu_model_view;

typedef struct {
    ds4_gpu_model_view entries[DS4_METAL_MAX_MODEL_VIEWS];
    uint32_t count;
    uint64_t wrap_count;
    uint64_t wrap_bytes;
    uint64_t wrap_max_bytes;
    /* Actual public map publication and residency owner, moved once from
     * the original globals. C fallback uses this same storage. */
    const void *published_map;
    uint64_t published_size;
    uint64_t published_offset;
    uint64_t published_bytes;
    uint64_t published_max_tensor_bytes;
#ifdef __OBJC__
    __strong id residency_set;
#else
    void *residency_set;
#endif
    uint64_t residency_count;
    int residency_added_to_queue;
    int residency_skipped;
} ds4_metal_model_views;

/* Driver/ARC operations only. Callers own GPU initialization and command scope. */
uint64_t ds4_metal_model_view_max_buffer_length(void);
int ds4_metal_model_view_install(ds4_gpu_model_view *view, const void *address,
    uint64_t bytes, uint32_t index);
void ds4_metal_model_view_retire(ds4_gpu_model_view *view);
void ds4_metal_model_view_move(ds4_gpu_model_view *destination,
    const ds4_gpu_model_view *source);

void ds4_native_metal_model_views_clear(ds4_metal_model_views *views);
void ds4_native_metal_model_views_remove_map(ds4_metal_model_views *views,
    const void *model_map);
int ds4_native_metal_model_views_add(ds4_metal_model_views *views,
    const void *model_map, uint64_t model_size, uint64_t map_offset,
    uint64_t map_size, uint64_t max_tensor_bytes, bool use_default_view_cap,
    uint64_t *mapped_model_size_out);
uint32_t ds4_native_metal_model_views_find(const ds4_metal_model_views *views,
    const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes);
int ds4_native_metal_model_views_cover_range(const ds4_metal_model_views *views,
    const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes);
int ds4_native_metal_model_views_cover_spans(const ds4_metal_model_views *views,
    const void *model_map, uint64_t model_size, const uint64_t *offsets,
    const uint64_t *sizes, uint32_t count);
/* Lifecycle ARC leaves: resource ownership and fixed shader bindings only. */
typedef struct ds4_metal_model_warm_pipeline ds4_metal_model_warm_pipeline;
typedef struct {
    uint64_t bytes, destination, groups;
} ds4_metal_model_warm_range;
void ds4_metal_model_view_residency_clear(ds4_metal_model_views *state);
int ds4_metal_model_view_residency_install(ds4_metal_model_views *state,
    bool attach_queue);
const ds4_metal_model_warm_pipeline *ds4_metal_model_view_warm_pipeline(void);
int ds4_metal_model_view_warm_encode(const ds4_metal_model_views *state,
    const ds4_metal_model_warm_pipeline *pipeline,
    const ds4_metal_model_warm_range *ranges, uint64_t total, uint64_t stride);
double ds4_metal_model_view_now_ms(void);
void ds4_metal_model_view_keepalive(void);
void ds4_metal_model_view_progress(uint32_t stage, const char *text);
int ds4_native_metal_model_map_range(ds4_metal_model_views *state,
    bool streaming, const void *map, uint64_t size, uint64_t offset,
    uint64_t bytes, uint64_t maximum);
int ds4_native_metal_model_map_spans(ds4_metal_model_views *state,
    bool streaming, const void *map, uint64_t size, const uint64_t *offsets,
    const uint64_t *sizes, uint32_t count, uint64_t maximum);
#endif
