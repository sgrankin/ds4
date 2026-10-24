#ifndef DS4_SESSION_RESOURCE_POOL_H
#define DS4_SESSION_RESOURCE_POOL_H

#include "ds4_qwen4_graph.h"

/* Actual engine-owned session resources, embedded once in Engine. Qwen
 * sessions borrow its scratch arena and retained recurrence-slot views. GLM
 * reservations include existing sessions and slots still awaiting creation.
 * Keep this record identical in CPU fixtures: it contains opaque GPU handles,
 * not a GPU implementation or a second Engine/Session layout. */
typedef struct {
    ds4_qwen4_gpu_graph *qwen4_shared_workspace;
    uint32_t qwen4_arena_users;
    ds4_qwen4_state_pool qwen4_pool;
    uint32_t glm_session_count;
    uint64_t glm_session_graph_bytes;
} ds4_session_resource_pool;

#endif
