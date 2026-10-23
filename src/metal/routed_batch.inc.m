/* Fixed ARC/resource leaves for the native batched routed-MoE owner. MSL and
 * the existing expert-cache/table libraries are unchanged. */
#include "routed_batch.h"
@interface DS4RoutedBatchResources : NSObject {
  @public
    ds4_metal_routed_request request;
    id<MTLBuffer> gateWeight, upWeight, downWeight;
    uint64_t gateInner, upInner, downInner;
    id<MTLBuffer> downScratch, mapBuffer, packedBuffer;
    id<MTLBuffer> selectedExecution, addendBuffer;
    NSUInteger selectedExecutionOffset, addendOffset;
    DS4MetalQ4ExpertTable *gateTable, *upTable, *downTable;
    id layerResidency;
    id<MTLBuffer> gateAddress, upAddress, downAddress;
    id<MTLBuffer> overflowGate, overflowUp, overflowDown;
    ds4_gpu_stream_expert_cache_entry *entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    uint32_t entryCount;
}
@end
@implementation DS4RoutedBatchResources
@end
#define ROUTED_CONTEXT ((DS4RoutedBatchResources *)(__bridge id)opaque)
double ds4_metal_routed_now_ms(void) { return ds4_gpu_now_ms(); }
void ds4_metal_routed_get_facts(uint32_t total, ds4_metal_routed_facts *out) {
    @autoreleasepool {
        memset(out, 0, sizeof(*out));
        out->flags =
            (g_quality_mode ? DS4_ROUTED_QUALITY : 0) |
            (g_ssd_streaming_mode ? DS4_ROUTED_SSD : 0) |
            (ds4_gpu_device_is_m5_apple_silicon() ? DS4_ROUTED_M5 : 0) |
            (ds4_gpu_device_is_pre_m5_apple_silicon() ? DS4_ROUTED_PRE_M5 : 0) |
            (g_glm_model_mode ? DS4_ROUTED_GLM : 0) |
            (ds4_gpu_mpp_available() ? DS4_ROUTED_MPP : 0) |
            (ds4_gpu_q4_table_queue_residency_available() ? DS4_ROUTED_QUEUE_RESIDENCY : 0) |
            (g_glm_streaming_prefill_full_layer_runtime ? DS4_ROUTED_FULL_LAYER : 0);
        out->rank = g_tp_split_rank;
        out->world = g_tp_split_world;
        ds4_gpu_tp_expert_range(total, &out->first_expert, &out->bound_experts);
        out->cache_budget = ds4_gpu_stream_expert_cache_configured_count();
        out->tests = g_test_flags;
        out->max_buffer_bytes = g_device ? [g_device maxBufferLength] : 0;
        out->table_encoders = g_moe_table_q4_pair_gate_encoder && g_moe_table_q4_pair_up_encoder &&
                              g_moe_table_q4_sum_down_encoder;
#define P(key, value)                                                                              \
    out->cached[key] = (const ds4_metal_hc_pipeline *)(__bridge const void *)(value)
        P(DS4_ROUTED_IQ2, g_moe_mul_mv_id_iq2_xxs_pipeline);
        P(DS4_ROUTED_Q2, g_moe_mul_mv_id_q2_k_pipeline);
        P(DS4_ROUTED_Q4, g_moe_mul_mv_id_q4_k_pipeline);
        P(DS4_ROUTED_MX, g_moe_mul_mv_id_mxfp4_pipeline);
        P(DS4_ROUTED_IQ2_PAIR, g_moe_mul_mv_id_iq2_xxs_pair_pipeline);
        P(DS4_ROUTED_Q4_PAIR, g_moe_mul_mv_id_q4_k_pair_pipeline);
        P(DS4_ROUTED_IQ2_SWI, g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline);
        P(DS4_ROUTED_Q4_SWI, g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_MX_SWI, g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline);
        P(DS4_ROUTED_Q2_SUM, g_moe_mul_mv_id_q2_k_sum6_pipeline);
        P(DS4_ROUTED_Q4_SUM, g_moe_mul_mv_id_q4_k_sum6_pipeline);
        P(DS4_ROUTED_MX_SUM, g_moe_mul_mv_id_mxfp4_sum6_pipeline);
        P(DS4_ROUTED_MX_TP_SWI, g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1);
        P(DS4_ROUTED_MX_TP_SUM, g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1);
        P(DS4_ROUTED_ADDR_IQ2_SWI, g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline);
        P(DS4_ROUTED_ADDR_IQ2, g_moe_mul_mv_addr_iq2_xxs_pipeline);
        P(DS4_ROUTED_ADDR_Q2_SUM, g_moe_mul_mv_addr_q2_k_sum6_pipeline);
        P(DS4_ROUTED_TABLE_Q4_SWI, g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_TABLE_Q4_SUM, g_moe_mul_mv_table_q4_k_sum6_pipeline);
        P(DS4_ROUTED_GROUP_Q4_SWI, g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_GROUP_Q4_SUM, g_moe_mul_mv_group_q4_k_sum6_pipeline);
        P(DS4_ROUTED_GROUP6_SWI, g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_GROUP6_SUM, g_moe_mul_mv_group6_q4_k_sum6_pipeline);
        P(DS4_ROUTED_GROUP8_SWI, g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_GROUP8_SUM, g_moe_mul_mv_group8_q4_k_sum6_pipeline);
        P(DS4_ROUTED_GROUP24, g_moe_mul_mv_group24_q4_k_id_pipeline);
        P(DS4_ROUTED_GROUP24_SUM, g_moe_mul_mv_group24_q4_k_sum6_pipeline);
        P(DS4_ROUTED_SLOTS_Q4_SWI, g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_SLOTS_Q4_SUM, g_moe_mul_mv_slots6_q4_k_sum6_pipeline);
        P(DS4_ROUTED_SLOTS_MX_SWI, g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline);
        P(DS4_ROUTED_SLOTS_MX_SUM, g_moe_mul_mv_slots6_mxfp4_sum6_pipeline);
        P(DS4_ROUTED_ADDR_Q4_SWI, g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline);
        P(DS4_ROUTED_ADDR_Q4_SUM, g_moe_mul_mv_addr_q4_k_sum6_pipeline);
        P(DS4_ROUTED_CLAMP, g_unary_clamp_pipeline);
        P(DS4_ROUTED_SILU, g_unary_silu_pipeline);
        P(DS4_ROUTED_MUL, g_mul_pipeline);
        P(DS4_ROUTED_WEIGHT, g_bin_mul_scalar_pipeline);
        P(DS4_ROUTED_ADD, g_add_pipeline);
        P(DS4_ROUTED_SWIGLU, g_swiglu_flat_pipeline);
        P(DS4_ROUTED_SUM6, g_moe_sum6_pipeline);
        P(DS4_ROUTED_SUM8, g_moe_sum8_pipeline);
#undef P
    }
}
const ds4_metal_hc_pipeline *ds4_metal_routed_pipeline(uint32_t kind, const char *name,
                                                       int16_t nsg) {
    @autoreleasepool {
        id<MTLComputePipelineState> p = kind == 1   ? ds4_gpu_get_mul_mv_pipeline(name, nsg)
                                        : kind == 2 ? ds4_gpu_get_mul_mm_id_pipeline(name, false)
                                                    : ds4_gpu_get_pipeline(name);
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)p;
    }
}
uint64_t ds4_metal_routed_max_threads(const ds4_metal_hc_pipeline *p) {
    return [(__bridge id<MTLComputePipelineState>)p maxTotalThreadsPerThreadgroup];
}
ds4_metal_routed_resources *ds4_metal_routed_open(const ds4_metal_routed_request *request) {
    @autoreleasepool {
        DS4RoutedBatchResources *context = [DS4RoutedBatchResources new];
        if (!context)
            return NULL;
        context->request = *request;
        return (ds4_metal_routed_resources *)(__bridge_retained void *)context;
    }
}
void ds4_metal_routed_close(ds4_metal_routed_resources *opaque) {
    @autoreleasepool {
        id held = (__bridge_transfer id)opaque;
        (void)held;
    }
}
bool ds4_metal_routed_wrap(ds4_metal_routed_resources *opaque, uint64_t gate_bytes,
                           uint64_t down_bytes) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        r->gateWeight = ds4_gpu_wrap_model_range(r->request.model, r->request.model_bytes,
                                                 r->request.gate_offset, gate_bytes, &r->gateInner);
        r->upWeight = ds4_gpu_wrap_model_range(r->request.model, r->request.model_bytes,
                                               r->request.up_offset, gate_bytes, &r->upInner);
        r->downWeight = ds4_gpu_wrap_model_range(r->request.model, r->request.model_bytes,
                                                 r->request.down_offset, down_bytes, &r->downInner);
        return r->gateWeight && r->upWeight && r->downWeight;
    }
}
void ds4_metal_routed_debug_bind(ds4_metal_routed_resources *opaque,
                                 const ds4_gpu_mul_mv_id_args *args, uint64_t gate_bytes) {
    DS4RoutedBatchResources *r = ROUTED_CONTEXT;
    fprintf(stderr,
            "ds4: batch mv binds l=%u rank=%d base=%d gate=%llu+%llu up=%llu "
            "down=%llu inner=%llu/%llu/%llu ne02=%d nei0=%d nr0=%d\n",
            r->request.layer, g_tp_split_rank, args->tp_expert_base,
            (unsigned long long)r->request.gate_offset, (unsigned long long)gate_bytes,
            (unsigned long long)r->request.up_offset, (unsigned long long)r->request.down_offset,
            (unsigned long long)r->gateInner, (unsigned long long)r->upInner,
            (unsigned long long)r->downInner, args->ne02, args->nei0, args->nr0);
}
bool ds4_metal_routed_scratch(ds4_metal_routed_resources *opaque, uint32_t kind, uint64_t bytes) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        if (kind == 0) {
            if (!ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                               &g_moe_down_scratch_bytes, bytes,
                                               "ds4_moe_down_scratch"))
                return false;
            r->downScratch = g_moe_down_scratch_buffer;
        } else if (kind == 1) {
            if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer, &g_moe_id_map_bytes, bytes,
                                               "ds4_moe_id_map"))
                return false;
            r->mapBuffer = g_moe_id_map_buffer;
        } else {
            if (!ds4_gpu_ensure_scratch_buffer(&g_moe_packed_rhs_buffer, &g_moe_packed_rhs_bytes,
                                               bytes, "ds4_moe_packed_rhs"))
                return false;
            r->packedBuffer = g_moe_packed_rhs_buffer;
        }
        return true;
    }
}
uint32_t ds4_metal_routed_cache_admit(uint64_t gate, uint64_t down) {
    return ds4_gpu_stream_expert_cache_note_expert_size(gate, down)
               ? ds4_gpu_stream_expert_cache_configured_count()
               : 0;
}
bool ds4_metal_routed_take_ready(const ds4_gpu_tensor *selected) {
    bool ready = g_stream_expert_batch_ready == selected;
    g_stream_expert_batch_ready = NULL;
    return ready;
}
bool ds4_metal_routed_cache_prepare(ds4_metal_routed_resources *opaque, uint32_t *unique,
                                    uint32_t *bad, uint32_t *count) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        *bad = UINT32_MAX;
        g_stream_prefill_batch_selected_addr_building++;
        /* ARC writeback uses local strong owners, then transfers the retained
         * resources into this invocation's bundle. No cache state is copied. */
        id<MTLBuffer> gateAddress = nil, upAddress = nil, downAddress = nil;
        id<MTLBuffer> overflowGate = nil, overflowUp = nil, overflowDown = nil;
        int ok = ds4_gpu_stream_expert_cache_prepare_selected_batch(
            q->model, q->model_bytes, q->layer, q->selected, q->rows, q->total_experts,
            q->used_experts, q->gate_offset, q->up_offset, q->down_offset, q->gate_expert_bytes,
            q->down_expert_bytes, &gateAddress, &upAddress, &downAddress, r->entries,
            &r->entryCount, unique, &overflowGate, &overflowUp, &overflowDown);
        r->gateAddress = gateAddress;
        r->upAddress = upAddress;
        r->downAddress = downAddress;
        r->overflowGate = overflowGate;
        r->overflowUp = overflowUp;
        r->overflowDown = overflowDown;
        g_stream_prefill_batch_selected_addr_building--;
        *count = r->entryCount;
        if (ok)
            for (uint32_t i = 0; i < r->entryCount; i++) {
                ds4_gpu_stream_expert_cache_entry *entry = r->entries[i];
                if (!entry || !entry->valid || !entry->gate_buffer || !entry->up_buffer ||
                    !entry->down_buffer) {
                    *bad = i;
                    break;
                }
            }
        return ok != 0;
    }
}
void ds4_metal_routed_cache_clear(uint32_t layer) {
    @autoreleasepool {
        ds4_gpu_stream_expert_cache_clear_layer(layer);
    }
}
bool ds4_metal_routed_mark(ds4_metal_routed_resources *opaque) {
    DS4RoutedBatchResources *r = ROUTED_CONTEXT;
    return !r->entryCount ||
           ds4_gpu_stream_expert_cache_mark_entries_inflight(r->entries, r->entryCount, 0);
}
int ds4_metal_routed_tables(ds4_metal_routed_resources *opaque, bool automatic) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        r->gateTable =
            ds4_gpu_q4_expert_table(q->model, q->model_bytes, q->gate_offset, q->gate_expert_bytes,
                                    q->total_experts, g_moe_table_q4_pair_gate_encoder);
        r->upTable =
            ds4_gpu_q4_expert_table(q->model, q->model_bytes, q->up_offset, q->gate_expert_bytes,
                                    q->total_experts, g_moe_table_q4_pair_up_encoder);
        r->downTable =
            ds4_gpu_q4_expert_table(q->model, q->model_bytes, q->down_offset, q->down_expert_bytes,
                                    q->total_experts, g_moe_table_q4_sum_down_encoder);
        if (!r->gateTable || !r->upTable || !r->downTable)
            return 0;
        r->layerResidency = ds4_gpu_q4_expert_layer_residency_set(r->gateTable, r->upTable,
                                                                  r->downTable, automatic);
        return r->layerResidency ? 1 : 2;
    }
}
bool ds4_metal_routed_residency(ds4_metal_routed_resources *opaque, void *command, bool model,
                                bool queue) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (model && !ds4_gpu_use_model_residency_set(cb))
            return false;
        if (!queue && r->layerResidency && [cb respondsToSelector:@selector(useResidencySet:)])
            [cb useResidencySet:r->layerResidency];
        return true;
    }
}
static id<MTLBuffer> ds4_routed_buffer(DS4RoutedBatchResources *r, uint32_t role,
                                       NSUInteger *offset) {
    *offset = 0;
    const ds4_gpu_tensor *tensor = NULL;
    switch (role) {
    case DS4_ROUTED_X:
        tensor = r->request.x;
        break;
    case DS4_ROUTED_GATE:
        tensor = r->request.gate;
        break;
    case DS4_ROUTED_UP:
        tensor = r->request.up;
        break;
    case DS4_ROUTED_MID:
        tensor = r->request.mid;
        break;
    case DS4_ROUTED_OUT:
        tensor = r->request.out;
        break;
    case DS4_ROUTED_WEIGHTS:
        tensor = r->request.weights;
        break;
    case DS4_ROUTED_SELECTED:
        if (r->selectedExecution) {
            *offset = r->selectedExecutionOffset;
            return r->selectedExecution;
        }
        tensor = r->request.selected;
        break;
    case DS4_ROUTED_DOWN:
        if (r->request.used_experts == 1)
            tensor = r->request.out;
        else if (ds4_gpu_tensor_buffer(r->request.experts))
            tensor = r->request.experts;
        else
            return r->downScratch;
        break;
    case DS4_ROUTED_W_GATE:
        *offset = r->gateInner;
        return r->gateWeight;
    case DS4_ROUTED_W_UP:
        *offset = r->upInner;
        return r->upWeight;
    case DS4_ROUTED_W_DOWN:
        *offset = r->downInner;
        return r->downWeight;
    case DS4_ROUTED_MAP_BUFFER:
        return r->mapBuffer;
    case DS4_ROUTED_PACKED:
        return r->packedBuffer;
    }
    *offset = ds4_gpu_tensor_offset(tensor);
    return ds4_gpu_tensor_buffer(tensor);
}

bool ds4_metal_routed_encode(ds4_metal_routed_resources *opaque, void *command,
                             const ds4_metal_routed_launch *p) {
    @autoreleasepool {
        DS4RoutedBatchResources *r = ROUTED_CONTEXT;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !pipeline)
            return false;
        const bool address = p->kind == DS4_ROUTED_ADDR_PAIR || p->kind == DS4_ROUTED_ADDR_MV ||
                             p->kind == DS4_ROUTED_ADDR_SUM;
        if (address) {
            /* Pair and direct-sum helpers mark even an empty list; the old cache
             * action rejects it. Ordinary address MV skips that mark when empty. */
            if (p->kind != DS4_ROUTED_ADDR_MV && r->entryCount == 0)
                return false;
            id<MTLBuffer> addresses = p->kind == DS4_ROUTED_ADDR_PAIR ? r->gateAddress
                                      : p->weight == 0                ? r->gateAddress
                                      : p->weight == 1                ? r->upAddress
                                                                      : r->downAddress;
            id<MTLBuffer> overflow = p->kind == DS4_ROUTED_ADDR_PAIR ? r->overflowGate
                                     : p->weight == 0                ? r->overflowGate
                                     : p->weight == 1                ? r->overflowUp
                                                                     : r->overflowDown;
            if (!addresses || (!r->entryCount && !overflow) ||
                (p->kind == DS4_ROUTED_ADDR_PAIR && !r->upAddress))
                return false;
        }
        if (p->kind == DS4_ROUTED_TABLE_PAIR &&
            (!r->gateTable.argumentBuffer || !r->upTable.argumentBuffer))
            return false;
        if (p->kind == DS4_ROUTED_TABLE_SUM && !r->downTable.argumentBuffer)
            return false;
        if ((p->kind == DS4_ROUTED_MAP || p->kind == DS4_ROUTED_MM ||
             p->kind == DS4_ROUTED_MM_PAIR || p->kind == DS4_ROUTED_PACK_OFFSETS ||
             p->kind == DS4_ROUTED_PACK_RHS) &&
            !r->mapBuffer)
            return false;
        if ((p->kind == DS4_ROUTED_PACK_OFFSETS || p->kind == DS4_ROUTED_PACK_RHS) &&
            !r->packedBuffer)
            return false;
        if ((p->kind == DS4_ROUTED_ADDR_PAIR || p->kind == DS4_ROUTED_ADDR_MV ||
             p->kind == DS4_ROUTED_ADDR_SUM) &&
            !ds4_metal_routed_mark(opaque))
            return false;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
#define B(slot, role, extra)                                                                       \
    do {                                                                                           \
        NSUInteger at;                                                                             \
        id<MTLBuffer> buffer = ds4_routed_buffer(r, (role), &at);                                  \
        [enc setBuffer:buffer offset:at + (extra)atIndex:(slot)];                                  \
    } while (0)
#define A(slot, field) [enc setBytes:&p->args.field length:sizeof(p->args.field) atIndex:(slot)]
#define ACT [enc setBytes:&p->activation length:sizeof(p->activation) atIndex:1]
#define DIRECT(slot, buffer) [enc setBuffer:(buffer) offset:0 atIndex:(slot)]
        const bool addr = p->kind == DS4_ROUTED_ADDR_MV || p->kind == DS4_ROUTED_ADDR_SUM;
        switch (p->kind) {
        case DS4_ROUTED_MV:
        case DS4_ROUTED_DOWN_SUM:
        case DS4_ROUTED_ADDR_MV:
        case DS4_ROUTED_ADDR_SUM:
        case DS4_ROUTED_TABLE_SUM:
            A(0, mv);
            if (addr)
                DIRECT(1, p->weight == 0   ? r->gateAddress
                          : p->weight == 1 ? r->upAddress
                                           : r->downAddress);
            else if (p->kind == DS4_ROUTED_TABLE_SUM)
                DIRECT(1, r->downTable.argumentBuffer);
            else
                B(1,
                  p->weight == 0   ? DS4_ROUTED_W_GATE
                  : p->weight == 1 ? DS4_ROUTED_W_UP
                                   : DS4_ROUTED_W_DOWN,
                  0);
            B(2, p->source, p->source_extra);
            B(3, p->destination, p->destination_extra);
            B(4, DS4_ROUTED_SELECTED, 0);
            if (p->kind == DS4_ROUTED_DOWN_SUM) {
                if (r->addendBuffer)
                    [enc setBuffer:r->addendBuffer offset:r->addendOffset atIndex:5];
                else
                    B(5, p->destination, p->destination_extra);
            }
            if (p->kind == DS4_ROUTED_TABLE_SUM) {
                if (!ds4_gpu_bind_q4_expert_table_anchors(enc, r->downTable, 5, 6)) {
                    ds4_gpu_end_compute_encoder(cb, enc);
                    return false;
                }
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->downTable, p->queue_residency);
            }
            if (addr) {
                for (uint32_t i = 0; i < r->entryCount; i++) {
                    ds4_gpu_stream_expert_cache_entry *entry = r->entries[i];
                    id<MTLBuffer> buffer = p->weight == 0   ? entry->gate_buffer
                                           : p->weight == 1 ? entry->up_buffer
                                                            : entry->down_buffer;
                    [enc useResource:buffer usage:MTLResourceUsageRead];
                }
                id<MTLBuffer> overflow = p->weight == 0   ? r->overflowGate
                                         : p->weight == 1 ? r->overflowUp
                                                          : r->overflowDown;
                if (overflow)
                    [enc useResource:overflow usage:MTLResourceUsageRead];
            }
            break;
        case DS4_ROUTED_PAIR:
            A(0, mv);
            B(1, DS4_ROUTED_W_GATE, 0);
            B(2, DS4_ROUTED_W_UP, 0);
            B(3, DS4_ROUTED_X, 0);
            B(4, DS4_ROUTED_GATE, 0);
            B(5, DS4_ROUTED_UP, 0);
            B(6, DS4_ROUTED_SELECTED, 0);
            break;
        case DS4_ROUTED_PAIR_SWI:
        case DS4_ROUTED_ADDR_PAIR:
        case DS4_ROUTED_TABLE_PAIR:
            A(0, mv);
            ACT;
            if (p->kind == DS4_ROUTED_ADDR_PAIR) {
                DIRECT(2, r->gateAddress);
                DIRECT(3, r->upAddress);
            } else if (p->kind == DS4_ROUTED_TABLE_PAIR) {
                DIRECT(2, r->gateTable.argumentBuffer);
                DIRECT(3, r->upTable.argumentBuffer);
            } else {
                B(2, DS4_ROUTED_W_GATE, 0);
                B(3, DS4_ROUTED_W_UP, 0);
            }
            B(4, DS4_ROUTED_X, 0);
            B(5, DS4_ROUTED_GATE, 0);
            B(6, DS4_ROUTED_UP, 0);
            B(7, DS4_ROUTED_MID, 0);
            B(8, DS4_ROUTED_SELECTED, 0);
            B(9, DS4_ROUTED_WEIGHTS, 0);
            if (p->kind == DS4_ROUTED_TABLE_PAIR) {
                if (!ds4_gpu_bind_q4_expert_table_anchors(enc, r->gateTable, 10, 6) ||
                    !ds4_gpu_bind_q4_expert_table_anchors(enc, r->upTable, 16, 6)) {
                    ds4_gpu_end_compute_encoder(cb, enc);
                    return false;
                }
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->gateTable, p->queue_residency);
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->upTable, p->queue_residency);
            }
            if (p->kind == DS4_ROUTED_ADDR_PAIR) {
                for (uint32_t i = 0; i < r->entryCount; i++) {
                    [enc useResource:r->entries[i]->gate_buffer usage:MTLResourceUsageRead];
                    [enc useResource:r->entries[i]->up_buffer usage:MTLResourceUsageRead];
                }
                if (r->overflowGate)
                    [enc useResource:r->overflowGate usage:MTLResourceUsageRead];
                if (r->overflowUp)
                    [enc useResource:r->overflowUp usage:MTLResourceUsageRead];
            }
            break;
        case DS4_ROUTED_MAP:
            A(0, map);
            B(1, DS4_ROUTED_SELECTED, 0);
            DIRECT(2, r->mapBuffer);
            [enc setBuffer:r->mapBuffer offset:p->counts_bytes atIndex:3];
            [enc setBuffer:r->mapBuffer offset:p->work_offset atIndex:4];
            break;
        case DS4_ROUTED_MM:
            A(0, mm);
            if (p->cached)
                DIRECT(1, p->weight == 0   ? r->gateAddress
                          : p->weight == 1 ? r->upAddress
                                           : r->downAddress);
            else
                B(1,
                  p->weight == 0   ? DS4_ROUTED_W_GATE
                  : p->weight == 1 ? DS4_ROUTED_W_UP
                                   : DS4_ROUTED_W_DOWN,
                  0);
            B(2, p->source, p->source_extra);
            DIRECT(3, r->mapBuffer);
            [enc setBuffer:r->mapBuffer offset:p->counts_bytes atIndex:4];
            B(5, p->destination, p->destination_extra);
            [enc setBuffer:r->mapBuffer offset:p->work_offset atIndex:6];
            if (p->cached) {
                for (uint32_t i = 0; i < r->entryCount; i++) {
                    ds4_gpu_stream_expert_cache_entry *entry = r->entries[i];
                    id<MTLBuffer> buffer = p->weight == 0   ? entry->gate_buffer
                                           : p->weight == 1 ? entry->up_buffer
                                                            : entry->down_buffer;
                    [enc useResource:buffer usage:MTLResourceUsageRead];
                }
            }
            {
                id<MTLBuffer> overflow = p->weight == 0   ? r->overflowGate
                                         : p->weight == 1 ? r->overflowUp
                                                          : r->overflowDown;
                if (overflow)
                    [enc useResource:overflow usage:MTLResourceUsageRead];
            }
            if (p->use_resources) {
                NSUInteger ignored;
                id<MTLBuffer> buffer = p->cached
                                           ? (p->weight == 0   ? r->gateAddress
                                              : p->weight == 1 ? r->upAddress
                                                               : r->downAddress)
                                           : ds4_routed_buffer(r,
                                                               p->weight == 0   ? DS4_ROUTED_W_GATE
                                                               : p->weight == 1 ? DS4_ROUTED_W_UP
                                                                                : DS4_ROUTED_W_DOWN,
                                                               &ignored);
                [enc useResource:buffer usage:MTLResourceUsageRead];
            }
            break;
        case DS4_ROUTED_MM_PAIR:
            A(0, mm);
            ACT;
            B(2, DS4_ROUTED_W_GATE, 0);
            B(3, DS4_ROUTED_W_UP, 0);
            B(4, DS4_ROUTED_X, 0);
            DIRECT(5, r->mapBuffer);
            [enc setBuffer:r->mapBuffer offset:p->counts_bytes atIndex:6];
            B(7, DS4_ROUTED_MID, 0);
            B(8, DS4_ROUTED_WEIGHTS, 0);
            [enc setBuffer:r->mapBuffer offset:p->work_offset atIndex:9];
            break;
        case DS4_ROUTED_PACK_OFFSETS:
            A(0, count);
            DIRECT(1, r->mapBuffer);
            DIRECT(2, r->packedBuffer);
            break;
        case DS4_ROUTED_PACK_RHS:
            A(0, mm);
            B(1, p->source, 0);
            DIRECT(2, r->mapBuffer);
            [enc setBuffer:r->mapBuffer offset:p->counts_bytes atIndex:3];
            [enc setBuffer:r->mapBuffer offset:p->work_offset atIndex:4];
            DIRECT(5, r->packedBuffer);
            break;
        case DS4_ROUTED_ACTIVATION:
            A(0, activation);
            B(1, DS4_ROUTED_GATE, 0);
            B(2, DS4_ROUTED_UP, 0);
            B(3, DS4_ROUTED_MID, 0);
            B(4, DS4_ROUTED_WEIGHTS, 0);
            break;
        case DS4_ROUTED_UNARY:
            A(0, unary);
            B(1, p->source, p->source_extra);
            B(2, p->destination, p->destination_extra);
            break;
        case DS4_ROUTED_BIN:
            A(0, binary);
            B(1, p->source, p->source_extra);
            B(2, p->second, p->second_extra);
            B(3, p->destination, p->destination_extra);
            break;
        case DS4_ROUTED_FLAT:
            A(0, glu);
            B(1, DS4_ROUTED_GATE, 0);
            B(2, DS4_ROUTED_UP, 0);
            B(3, DS4_ROUTED_MID, 0);
            break;
        case DS4_ROUTED_SUM:
            A(0, sum);
            B(1, DS4_ROUTED_DOWN, 0);
            B(2, DS4_ROUTED_OUT, 0);
            break;
        default:
            ds4_gpu_end_compute_encoder(cb, enc);
            return false;
        }
        if (p->shared_bytes)
            [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, p->groups_z)
            threadsPerThreadgroup:MTLSizeMake(p->threads_x, p->threads_y, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return true;
#undef B
#undef A
#undef ACT
#undef DIRECT
    }
}
#undef ROUTED_CONTEXT
