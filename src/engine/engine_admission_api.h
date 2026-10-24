#ifndef DS4_ENGINE_ADMISSION_API_H
#define DS4_ENGINE_ADMISSION_API_H
#include "ds4_engine_core.h"
#include "ds4_session_resource_pool.h"
#include "ds4_streaming_policy.h"
#include "ds4_model_residency.h"
#include "engine_cache_sizes_api.h"
#include "engine_admission_math_api.h"
#include "model_spans_api.h"
#include "glm53/memory_api.h"
#include <sys/sysctl.h>
#include <sys/time.h>
#include <ctype.h>
#include <math.h>

/* Engine policy owns these algorithms over the actual common owners. The
 * immutable placement inputs describe the already-validated local request;
 * legacy TP/distributed and other hardware retain their own compatibility path. */
void ds4_native_engine_fit_glm_cache(ds4_engine_core *,
    const ds4_session_resource_pool *, const ds4_shape *, ds4_streaming_policy *,
    const ds4_model_residency_inputs *, int session_count_hint);
bool ds4_native_engine_admit_glm(const ds4_engine_core *, const ds4_shape *,
    const ds4_model_residency_inputs *,
    uint32_t compact_capacity, uint64_t extra_bytes, const char *phase);
bool ds4_native_engine_seed_hotlist(const ds4_engine_core *, const ds4_shape *);

/* Native startup cache policy, shared by lifecycle's pre-inspect/manual fit
 * and post-Metal automatic/configured cache construction. */
bool ds4_native_engine_auto_cache(ds4_engine_core *, const ds4_shape *,
    int context, uint32_t prefill_chunk);
bool ds4_native_engine_reserve_cache(ds4_engine_core *, const ds4_shape *,
    ds4_streaming_policy *);
uint64_t ds4_native_engine_manual_safe_bytes(const ds4_engine_core *,
    const ds4_shape *, int context, uint32_t prefill_chunk);
#endif
