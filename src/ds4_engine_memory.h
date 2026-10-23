#ifndef DS4_ENGINE_MEMORY_H
#define DS4_ENGINE_MEMORY_H

#include <stdbool.h>
#include <stdint.h>

/* Existing selected-address prefill reserve, shared with C fallback. */
enum { DS4_STREAMING_PREFILL_HEADROOM_LAYERS = 2 };

/* The actual Engine accounting record. All families keep their existing
 * counters here; the native owner changes only V4.1 admission and growth.
 * Graph bytes count live sessions, including conservative failed allocations.
 * Cache bytes/count describe the one configured GPU expert cache, not a copy. */
typedef struct {
    uint64_t ds41_session_bytes;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint64_t ssd_streaming_prefill_headroom_bytes;
    uint64_t ssd_streaming_full_layer_bytes;
    uint64_t ssd_streaming_decode_map_bytes;
    uint64_t ssd_streaming_decode_map_extra_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t startup_model_span_bytes;
    bool ssd_streaming_budget_finalized;
} ds4_engine_memory;

/* Immutable resource view for one admission/sync call. Target bytes retain
 * the existing TP-shard override. Bound-weight predicates select the static
 * spans actually owned by this Engine. No Engine or Session layout is shared. */
typedef struct {
    uint64_t target_model_bytes;
    uint64_t vision_model_bytes;
    uint64_t mtp_model_bytes;
    uint32_t session_count;
    bool ssd_streaming;
    bool dspark;
    bool include_token;
    bool include_output;
} ds4_engine_memory_inputs;

#endif
