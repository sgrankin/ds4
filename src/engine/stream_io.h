#ifndef DS4_ENGINE_STREAM_IO_H
#define DS4_ENGINE_STREAM_IO_H

#include "../ds4.h"
#include "../ds4_gpu.h"
#include "../ds4_model_weights.h"
#include <pthread.h>

/* One selected-expert worker. Finish joins before the borrowed model, layer or
 * selected tensor may retire. Retained IDs allow a synchronous retry. */
typedef struct metal_graph_selected_async_load {
    bool active;
    bool ok;
    bool ids_ok;
    ds4_gpu_tensor *router_selected;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    uint32_t il;
    uint64_t event_value;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_MAX_EXPERT_USED];
} metal_graph_selected_async_load;

bool ds4_native_moe_selected_start(metal_graph_selected_async_load *,
    ds4_gpu_tensor *selected, const ds4_model *, const ds4_layer_weights *,
    uint32_t layer_index, uint64_t event, uint64_t gate_expert_bytes,
    uint64_t down_expert_bytes);
bool ds4_native_moe_selected_finish(metal_graph_selected_async_load *);

/* Actual existing page-in service records, shared once with native callers.
 * Worker/range allocations are released by join, including failed jobs. */
typedef struct {
    uint64_t off;
    uint64_t size;
} metal_graph_stream_pagein_range;

typedef struct {
    pthread_t thread;
    const ds4_model *model;
    metal_graph_stream_pagein_range *ranges;
    pthread_t *threads;
    struct metal_graph_stream_pagein_worker *workers;
    uint32_t n_ranges;
    uint32_t n_threads;
    uint32_t layer;
    uint32_t n_tokens;
    uint32_t unique;
    uint64_t bytes;
    uint64_t touched;
    double read_ms;
    double thread_ms;
    bool profile;
    bool madvise_only;
    bool pread_only;
    bool readahead_only;
    bool started;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_job;

typedef struct metal_graph_stream_pagein_worker {
    metal_graph_stream_pagein_job *job;
    uint32_t first;
    uint32_t stride;
    uint64_t touched;
    double thread_ms;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_worker;

typedef struct {
    metal_graph_stream_pagein_job job;
    uint32_t layer;
    bool active;
} metal_graph_stream_prepare_slot;

enum {
    DS4_STREAM_PREPARE_MADVISE = 1u,
    DS4_STREAM_PREPARE_PREAD = 2u,
    DS4_STREAM_PREPARE_READAHEAD = 4u,
    DS4_STREAM_PREPARE_DECODE = 8u,
};
bool ds4_native_stream_prepare_start(const ds4_model *, const ds4_weights *,
    uint32_t layer, uint32_t rows, uint32_t flags,
    metal_graph_stream_prepare_slot *, uint32_t slots);
bool ds4_native_stream_prepare_join_layer(const ds4_model *, const ds4_weights *,
    uint32_t layer, uint32_t rows, uint32_t flags,
    metal_graph_stream_prepare_slot *, uint32_t slots);
bool ds4_native_stream_prepare_join_all(metal_graph_stream_prepare_slot *,
    uint32_t slots);

#endif
