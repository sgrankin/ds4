#ifndef DS4_ENGINE_PAGEIN_API_H
#define DS4_ENGINE_PAGEIN_API_H
#define _DARWIN_C_SOURCE 1

#include "stream_io.h"
#include "engine_cache_sizes_api.h"
#include "../ds4_streaming_policy.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Actual model-view range records, lifted from the C planner. The planner
 * selects model-specific tensors; native I/O owns partitioning and workers. */
typedef struct {
    uint64_t off;
    uint64_t end;
    bool isolate;
} ds4_model_map_span;
typedef struct {
    ds4_model_map_span *v;
    uint32_t len;
    uint32_t cap;
    uint64_t max_tensor_bytes;
} ds4_model_map_span_vec;

bool ds4_native_pagein_install_spans(const ds4_model *,
    const ds4_model_map_span_vec *, const char *label);

bool ds4_native_pagein_touch(const ds4_model *, uint64_t offset, uint64_t bytes,
    uint64_t *touched, uint8_t *sink);
bool ds4_native_pagein_pread(const ds4_model *, uint64_t offset, uint64_t bytes,
    uint64_t *read_bytes, uint8_t *sink);
bool ds4_native_pagein_madvise(const ds4_model *, uint64_t offset, uint64_t bytes,
    bool enabled, uint64_t *advised);
void ds4_native_pagein_readahead(const ds4_model *, uint64_t offset, uint64_t bytes,
    bool enabled);

bool ds4_native_pagein_layer_start(const ds4_model *, const ds4_weights *,
    uint32_t layer, uint32_t rows, uint32_t flags, bool have_graph,
    bool streaming, const ds4_shape *, const ds4_streaming_policy *,
    metal_graph_stream_pagein_job *);
bool ds4_native_pagein_selected_start(const ds4_model *, const ds4_layer_weights *,
    ds4_gpu_tensor *selected, uint32_t layer, uint32_t rows, uint64_t gate_bytes,
    uint64_t down_bytes, bool have_graph, bool streaming, uint32_t expert_count,
    uint32_t selected_count, metal_graph_stream_pagein_job *);
bool ds4_native_pagein_join(metal_graph_stream_pagein_job *, bool selected);
bool ds4_native_pagein_prepare_start(const ds4_model *, const ds4_weights *,
    uint32_t layer, uint32_t rows, uint32_t flags, bool have_graph,
    bool streaming, const ds4_shape *, const ds4_streaming_policy *,
    metal_graph_stream_prepare_slot *, uint32_t slot_count);
bool ds4_native_pagein_prepare_join_layer(const ds4_model *, const ds4_weights *,
    uint32_t layer, uint32_t rows, uint32_t flags, bool have_graph,
    bool streaming, const ds4_shape *, const ds4_streaming_policy *,
    metal_graph_stream_prepare_slot *, uint32_t slot_count);
bool ds4_native_pagein_prepare_join_all(metal_graph_stream_prepare_slot *, uint32_t);

#endif
