#ifndef DS4_ENGINE_CACHE_SIZES_API_H
#define DS4_ENGINE_CACHE_SIZES_API_H
#include "ds4_model_weights.h"
#include "ds4_engine_memory.h"
#include "model_api.h"

/* CPU-safe format/size accounting. No graph allocation or device query. */
bool ds4_native_cache_per_expert(const ds4_shape *, const ds4_weights *, uint64_t *);
bool ds4_native_cacheable_count(const ds4_shape *, const ds4_weights *,
    uint64_t *experts, uint32_t *layers);
bool ds4_native_cache_prefill_headroom(const ds4_shape *, const ds4_weights *, uint64_t *);
uint64_t ds4_native_cache_dynamic_bytes(const ds4_engine_memory *, bool streaming,
    const ds4_shape *, const ds4_weights *);
#endif
