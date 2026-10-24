#ifndef DS4_ENGINE_MOE_SERVICES_H
#define DS4_ENGINE_MOE_SERVICES_H

#include "ds4_deepseek41_graph.h"
#include "ds4_resident_route.h"
#include <time.h>

/* One scalar-call snapshot of existing process-wide diagnostics. The metrics
 * record is borrowed. Native MoE owns scheduling, readback and timing; C keeps
 * the oracle files and the resident-cache feature's existing private state. */
typedef struct {
    ds4_resident_route_layer_metrics *profile;
    bool oracle_deferred;
} ds4_native_moe_services;

/* Diagnostic actions only. Deferred capture preserves the GPU-selected tape
 * before returning checked recorded IDs. Observe checks/records native IDs and
 * writes the optional cache snapshot, retaining existing fatal diagnostics. */
bool ds4_native_moe_oracle_capture_deferred(ds41_gpu_graph *graph,
    uint32_t layer_index, int32_t *selected_ids);
void ds4_native_moe_oracle_observe(const ds4_gpu_stream_expert_table *table,
    uint32_t layer_index, uint32_t token, const int32_t *selected_ids);

/* Existing resident selected-set adjustment, including its cache snapshots,
 * readbacks and metrics. It does not launch shared/routed expert arithmetic. */
bool ds4_native_moe_resident_apply(ds41_gpu_graph *graph,
    const ds4_model *model, const ds4_layer_weights *layer,
    uint32_t layer_index, const ds4_gpu_stream_expert_table *table,
    int32_t *selected_ids);

#endif
