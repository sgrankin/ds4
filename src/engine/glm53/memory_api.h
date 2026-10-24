#ifndef DS4_ENGINE_GLM53_MEMORY_API_H
#define DS4_ENGINE_GLM53_MEMORY_API_H

#include "ds4.h"
#include "ds4_model_config.h"

/* CPU-safe model geometry. No graph handles or device service dependencies. */
uint32_t ds4_native_glm_full_attention_cap(uint32_t context, bool streaming);
bool ds4_native_glm_context_request(const ds4_shape *, int context,
    uint32_t *output);
uint32_t ds4_native_glm_indexed_chunk(uint32_t compact_capacity);
uint32_t ds4_native_glm_score_rows(uint32_t indexed_rows, uint32_t compact_capacity);
uint32_t ds4_native_glm_batch_cap(uint32_t work_capacity, uint32_t indexed_rows);
uint64_t ds4_native_glm_compact_bytes(const ds4_shape *, uint32_t attention_layers,
    uint32_t indexer_layers, uint32_t compact_capacity);
uint64_t ds4_native_glm_indexed_scratch_bytes(const ds4_shape *,
    uint32_t work_capacity, uint32_t compact_capacity);
uint64_t ds4_native_glm_workspace_bytes(const ds4_shape *,
    uint32_t work_capacity, uint32_t compact_capacity);
uint64_t ds4_native_glm_fixed_state_bytes(const ds4_shape *,
    uint32_t first_layer, uint32_t last_layer);
uint32_t ds4_native_glm_indexer_layers(const ds4_shape *,
    uint32_t first_layer, uint32_t last_layer);
ds4_context_memory ds4_native_glm_memory(const ds4_shape *, uint32_t context,
    uint32_t work_capacity, uint32_t compact_capacity, uint32_t first_layer,
    uint32_t last_layer);

#endif
