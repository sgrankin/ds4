#ifndef DS4_MODEL_RESIDENCY_H
#define DS4_MODEL_RESIDENCY_H
#include <stdbool.h>
#include <stdint.h>

/* The existing three Engine-open locals, now one owned mapping selection.
 * Arrays are libc-owned and borrowed by the remaining optional-cache action.
 * An empty selection denotes the ordinary full-range map. No Engine/Session
 * record or Metal/ARC buffer layout is duplicated here. */
typedef struct {
    uint64_t *offsets;
    uint64_t *sizes;
    uint32_t count;
} ds4_model_map_selection;

/* Immutable startup placement facts, derived from the existing public options.
 * load_output already includes the optional-but-actually-bound output head. */
typedef struct {
    bool load_slice;
    uint32_t layer_start;
    uint32_t layer_end;
    bool load_output;
    bool tp_shard;
    int tp_rank;
    int context;
    uint32_t chunk;
} ds4_model_residency_inputs;
#endif
