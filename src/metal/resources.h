#ifndef DS4_ZIG_METAL_RESOURCES_H
#define DS4_ZIG_METAL_RESOURCES_H

#include "../../ds4_gpu.h"

/* A borrowed, immutable region description, not the Objective-C class layout.
 * The caller must retain the tensor while reading it or using its contents. */
typedef struct {
    uint64_t offset;
    uint64_t bytes;
    uint8_t owner;
} ds4_metal_tensor_region;

typedef struct {
    uint64_t live_bytes;
    uint64_t peak_bytes;
} ds4_metal_tensor_stats;

/* These create one retained DS4MetalTensor handle. A view independently retains
 * the parent's MTLBuffer; disposing either handle does not retire the other.
 * Registration and range/offset policy belong to the native resource owner. */
ds4_gpu_tensor *ds4_metal_resource_new(uint64_t bytes);
ds4_gpu_tensor *ds4_metal_resource_view(const ds4_gpu_tensor *base,
    uint64_t absolute_offset, uint64_t bytes);
ds4_metal_tensor_region ds4_metal_resource_region(const ds4_gpu_tensor *tensor);
void ds4_metal_resource_dispose(ds4_gpu_tensor *tensor);
void *ds4_metal_resource_contents(ds4_gpu_tensor *tensor);

/* Absolute offsets are already checked/planned by the native owner. This only
 * enqueues into the existing caller batch; it never starts/commits/waits one. */
int ds4_metal_resource_blit(ds4_gpu_tensor *dst, uint64_t absolute_dst,
    const ds4_gpu_tensor *src, uint64_t absolute_src, uint64_t bytes);

/* Whole tracking actions used by memory diagnostics and GPU cleanup. Reset
 * discards bookkeeping only, exactly as the C fallback; it does not free live
 * caller handles or make later use of those handles valid after GPU cleanup. */
ds4_metal_tensor_stats ds4_native_metal_tensor_stats(void);
void ds4_native_metal_tensor_tracking_reset(void);

#endif
