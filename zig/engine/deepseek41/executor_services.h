#ifndef DS4_ENGINE_EXECUTOR_SERVICES_H
#define DS4_ENGINE_EXECUTOR_SERVICES_H

#include "moe_services.h"

/* Transitional private feature actions. Native execution owns numerical and
 * command ordering; these existing owners retain their tapes/counters. */
bool ds4_native_executor_capture(ds41_dspark *draft, uint32_t layer,
    uint32_t start, const ds4_gpu_tensor *residual, uint32_t count);
bool ds4_native_executor_route_probe(ds41_gpu_graph *graph,
    const ds4_model *model, const ds4_layer_weights *layer, uint32_t token);
bool ds4_native_executor_predict_prefetch(ds41_gpu_graph *graph,
    const ds4_model *model, const ds4_layer_weights *layer,
    uint32_t layer_index, uint32_t token);
ds4_native_moe_services ds4_native_executor_moe_snapshot(ds41_gpu_graph *graph,
    uint32_t layer_index);
bool ds4_native_executor_resident_prefill(ds41_gpu_graph *graph,
    const ds4_model *model, const ds4_layer_weights *layer,
    uint32_t layer_index, uint32_t count);
/* Begin returns the current oracle-active policy for gate-prefetch admission;
 * end completes its existing deferred validation and diagnostic publication. */
bool ds4_native_executor_begin_token(ds41_gpu_graph *graph, uint32_t token,
    bool decode);
bool ds4_native_executor_end_token(ds41_gpu_graph *graph, bool success);
bool ds4_native_executor_oracle_prefetch(const ds4_model *model,
    const ds4_layer_weights *layer, uint32_t layer_index);
bool ds4_native_executor_observe_imatrix(ds41_gpu_graph *graph,
    uint32_t layer_index);

bool ds4_native_executor_begin_resident_prefill(ds41_gpu_graph *graph,
    const ds4_model *model, const ds4_weights *weights, bool selected);

#endif
