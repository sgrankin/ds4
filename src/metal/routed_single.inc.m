void **ds4_metal_single_diagnostics_slot(void) { return &ds4_metal_context_current()->native_diagnostics; }
ds4_metal_selected_override *ds4_metal_single_override_state(void) { return &ds4_metal_context_current()->selected_override; }
/* ARC resource leaves for the complete native single-token routed owner.
 * Cache/table operations are explicit existing whole libraries. This file
 * contains no routed-path selection or old encode-controller forwarding. */
#include "routed_single.h"
@interface DS4RoutedSingleResources : DS4RoutedBatchResources {
  @public
    id<MTLBuffer> slotBuffers[3][8];
    NSUInteger slotOffsets[3][8];
    id<MTLBuffer> groupBuffers[3][24];
    NSUInteger groupOffsets[3][24];
    ds4_gpu_stream_expert_cache_entry *slots[8];
}
@end
@implementation DS4RoutedSingleResources
@end
#define SINGLE_CONTEXT ((DS4RoutedSingleResources *)(__bridge id)opaque)
bool ds4_metal_single_scope_begin(void) { return g_parallel_q8_pending; }
void ds4_metal_single_scope_end(bool armed) {
    BOOL scope = armed;
    ds4_gpu_parallel_ffn_scope_cleanup(&scope);
}
void ds4_metal_single_get_facts(uint32_t total, ds4_metal_single_facts *out) {
    @autoreleasepool {
        memset(out, 0, sizeof(*out));
        ds4_metal_routed_get_facts(total, &out->batch);
        out->parallel_pending = g_parallel_q8_pending;
        out->parallel_mode = g_parallel_ffn_mode;
        out->parallel_stage = g_parallel_ffn_stage;
#define P(key, value)                                                                              \
    out->cached[key] = (const ds4_metal_hc_pipeline *)(__bridge const void *)(value)
        P(DS4_SINGLE_MASK_PAIR, g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline);
        P(DS4_SINGLE_MASK_SUM, g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline);
        P(DS4_SINGLE_IQ2_PACK2, g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline);
        P(DS4_SINGLE_IQ2_SUM, g_moe_mul_mv_id_iq2_xxs_sum6_pipeline);
        P(DS4_SINGLE_MX_FIXED_PAIR, g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1);
        P(DS4_SINGLE_MX_STATIC_PAIR,
          g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1);
        P(DS4_SINGLE_MX_NSG1_PAIR, g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1);
        P(DS4_SINGLE_MX_TG_PAIR, g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple);
        P(DS4_SINGLE_MX_FULL_SUM, g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1);
        P(DS4_SINGLE_MX_STATIC_SUM,
          g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1);
        P(DS4_SINGLE_MX_FIXED_SUM, g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1);
        P(DS4_SINGLE_MX_NSG1_SUM, g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1);
        P(DS4_SINGLE_MX_TG_SUM, g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple);
        P(DS4_SINGLE_SLOTS_IQ2_PAIR, g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline);
        P(DS4_SINGLE_SLOTS_Q2_SUM, g_moe_mul_mv_slots6_q2_k_sum6_pipeline);
        P(DS4_SINGLE_Q4_GATHER, g_moe_q4_gather_slots6_pipeline);
        P(DS4_SINGLE_VALIDATE, g_moe_stream_expert_cache_validate_pipeline);
#undef P
    }
}
ds4_metal_single_resources *ds4_metal_single_open(const ds4_metal_routed_request *request,
                                                  const ds4_gpu_tensor *add_in) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = [DS4RoutedSingleResources new];
        if (!r)
            return NULL;
        r->request = *request;
        r->addendBuffer = ds4_gpu_tensor_buffer(add_in);
        r->addendOffset = ds4_gpu_tensor_offset(add_in);
        return (ds4_metal_single_resources *)(__bridge_retained void *)r;
    }
}
void ds4_metal_single_close(ds4_metal_single_resources *opaque) {
    @autoreleasepool {
        id held = (__bridge_transfer id)opaque;
        (void)held;
    }
}
ds4_metal_routed_resources *ds4_metal_single_base(ds4_metal_single_resources *opaque) {
    return (ds4_metal_routed_resources *)opaque;
}
void ds4_metal_single_note_token(uint32_t layer) { ds4_gpu_stream_expert_cache_note_token(layer); }
int ds4_metal_single_tables(ds4_metal_single_resources *opaque, bool address, bool automatic) {
    if (!address)
        return ds4_metal_routed_tables((ds4_metal_routed_resources *)opaque, automatic);
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        r->gateTable = ds4_gpu_q4_expert_address_table(q->model, q->model_bytes, q->gate_offset,
                                                       q->gate_expert_bytes, q->total_experts);
        r->upTable = ds4_gpu_q4_expert_address_table(q->model, q->model_bytes, q->up_offset,
                                                     q->gate_expert_bytes, q->total_experts);
        r->downTable = ds4_gpu_q4_expert_address_table(q->model, q->model_bytes, q->down_offset,
                                                       q->down_expert_bytes, q->total_experts);
        if (!r->gateTable || !r->upTable || !r->downTable)
            return 0;
        r->layerResidency = ds4_gpu_q4_expert_layer_residency_set(r->gateTable, r->upTable,
                                                                  r->downTable, automatic);
        return r->layerResidency ? 1 : 2;
    }
}
static id<MTLBuffer> ds4_single_model_view(DS4RoutedSingleResources *r, uint64_t offset,
                                           uint64_t bytes, uint32_t mode, uint64_t *inner) {
    if (mode == 1)
        return ds4_gpu_wrap_model_exact_range(r->request.model, r->request.model_bytes, offset,
                                              bytes, inner);
    if (mode == 2)
        return ds4_gpu_wrap_model_exact_range_transient(r->request.model, r->request.model_bytes,
                                                        offset, bytes, inner);
    return ds4_gpu_wrap_model_range(r->request.model, r->request.model_bytes, offset, bytes, inner);
}
bool ds4_metal_single_wrap(ds4_metal_single_resources *opaque, uint32_t mask, uint64_t gate,
                           uint64_t up, uint64_t down, uint64_t gate_bytes, uint64_t down_bytes,
                           uint32_t mode, uint32_t index) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        if (mask & 1)
            r->gateWeight = ds4_single_model_view(r, gate, gate_bytes, mode, &r->gateInner);
        if (mask & 2)
            r->upWeight = ds4_single_model_view(r, up, gate_bytes, mode, &r->upInner);
        if (mask & 4)
            r->downWeight = ds4_single_model_view(r, down, down_bytes, mode, &r->downInner);
        if (((mask & 1) && !r->gateWeight) || ((mask & 2) && !r->upWeight) ||
            ((mask & 4) && !r->downWeight))
            return false;
        if (index < 24) {
            if (mask & 1) {
                r->groupBuffers[0][index] = r->gateWeight;
                r->groupOffsets[0][index] = r->gateInner;
            }
            if (mask & 2) {
                r->groupBuffers[1][index] = r->upWeight;
                r->groupOffsets[1][index] = r->upInner;
            }
            if (mask & 4) {
                r->groupBuffers[2][index] = r->downWeight;
                r->groupOffsets[2][index] = r->downInner;
            }
        }
        return true;
    }
}
bool ds4_metal_single_wrap_slot(ds4_metal_single_resources *opaque, uint32_t slot, uint64_t gate,
                                uint64_t up, uint64_t down, uint32_t mode) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        uint64_t inner = 0;
        r->slotBuffers[0][slot] =
            ds4_single_model_view(r, gate, r->request.gate_expert_bytes, mode, &inner);
        r->slotOffsets[0][slot] = inner;
        inner = 0;
        r->slotBuffers[1][slot] =
            ds4_single_model_view(r, up, r->request.gate_expert_bytes, mode, &inner);
        r->slotOffsets[1][slot] = inner;
        inner = 0;
        r->slotBuffers[2][slot] =
            ds4_single_model_view(r, down, r->request.down_expert_bytes, mode, &inner);
        r->slotOffsets[2][slot] = inner;
        return r->slotBuffers[0][slot] && r->slotBuffers[1][slot] && r->slotBuffers[2][slot];
    }
}
bool ds4_metal_single_gather_storage(ds4_metal_single_resources *opaque, uint64_t gate_bytes,
                                     uint64_t down_bytes) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        if (!ds4_gpu_ensure_scratch_buffer(&g_moe_q4_gate_slots_buffer, &g_moe_q4_gate_slots_bytes,
                                           gate_bytes, "ds4_moe_q4_gate_slots") ||
            !ds4_gpu_ensure_scratch_buffer(&g_moe_q4_up_slots_buffer, &g_moe_q4_up_slots_bytes,
                                           gate_bytes, "ds4_moe_q4_up_slots") ||
            !ds4_gpu_ensure_scratch_buffer(&g_moe_q4_down_slots_buffer, &g_moe_q4_down_slots_bytes,
                                           down_bytes, "ds4_moe_q4_down_slots"))
            return false;
        for (uint32_t i = 0; i < 6; i++) {
            r->slotBuffers[0][i] = g_moe_q4_gate_slots_buffer;
            r->slotBuffers[1][i] = g_moe_q4_up_slots_buffer;
            r->slotBuffers[2][i] = g_moe_q4_down_slots_buffer;
            r->slotOffsets[0][i] = (uint64_t)i * r->request.gate_expert_bytes;
            r->slotOffsets[1][i] = r->slotOffsets[0][i];
            r->slotOffsets[2][i] = (uint64_t)i * r->request.down_expert_bytes;
        }
        return true;
    }
}
void ds4_metal_single_cache_stats(ds4_metal_single_stats *out, bool query_summary) {
    *out = (ds4_metal_single_stats){g_stream_expert_cache_hits, g_stream_expert_cache_misses,
                                    g_stream_expert_cache_wraps, g_stream_expert_cache_evictions,
                                    query_summary &&
                                        ds4_gpu_stream_expert_timing_summary_enabled() != 0};
}
bool ds4_metal_single_cache_note_size(ds4_metal_single_resources *opaque) {
    DS4RoutedSingleResources *r = SINGLE_CONTEXT;
    return ds4_gpu_stream_expert_cache_note_expert_size(r->request.gate_expert_bytes,
                                                        r->request.down_expert_bytes) != 0;
}
bool ds4_metal_single_cache_available(ds4_metal_single_resources *opaque) {
    DS4RoutedSingleResources *r = SINGLE_CONTEXT;
    return ds4_gpu_stream_expert_cache_effective_cap(r->request.layer, r->request.total_experts,
                                                     r->request.used_experts) != 0;
}
bool ds4_metal_single_split_ready(void) { return ds4_gpu_stream_expert_split_ready(); }
bool ds4_metal_single_addresses(ds4_metal_single_resources *opaque) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        id<MTLBuffer> gate = nil, up = nil, down = nil;
        int ok = ds4_gpu_stream_expert_cache_addr_buffers(r->request.layer, &gate, &up, &down);
        r->gateAddress = gate;
        r->upAddress = up;
        r->downAddress = down;
        return ok != 0;
    }
}
bool ds4_metal_single_full_addresses(ds4_metal_single_resources *opaque) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        id<MTLBuffer> gate = nil, up = nil, down = nil;
        ds4_gpu_stream_expert_cache_entry *entry = NULL;
        int ok = ds4_gpu_stream_full_expert_addr_table_prepare(
            q->model, q->model_bytes, q->layer, q->total_experts, q->gate_offset, q->up_offset,
            q->down_offset, q->gate_expert_bytes, q->down_expert_bytes, &gate, &up, &down, &entry);
        r->gateAddress = gate;
        r->upAddress = up;
        r->downAddress = down;
        if (ok) {
            for (uint32_t i = 0; i < q->used_experts; i++) {
                r->slots[i] = entry;
                r->entries[i] = entry;
            }
            r->entryCount = q->used_experts;
        }
        return ok != 0;
    }
}


/* Complete existing cache action; file policy never lives in this binding. */
void ds4_metal_single_note_selected_hotness(uint32_t layer,
                                           const int32_t *ids,
                                           uint32_t count) {
    ds4_gpu_stream_expert_cache_note_selected_hotness(layer, ids, count);
}
int ds4_metal_single_prefetch_take(ds4_metal_single_resources *opaque, int32_t *ids) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        if (!ds4_gpu_glm_stream_selected_prefetch_take(
                q->model, q->model_bytes, q->layer, q->total_experts, q->used_experts,
                q->gate_offset, q->up_offset, q->down_offset, q->gate_expert_bytes,
                q->down_expert_bytes, ids))
            return 0;
        return g_batch_cb && g_batch_has_work && g_stream_expert_pending_load.active ? 2 : 1;
    }
}
bool ds4_metal_single_validate(ds4_metal_single_resources *opaque, int32_t *ids, uint32_t *all,
                               uint32_t *miss, uint32_t *invalid) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        return ds4_gpu_stream_expert_cache_validate_selected(
            r->request.selected, r->gateAddress, r->upAddress, r->downAddress,
            r->request.total_experts, r->request.used_experts, ids, all, miss, invalid);
    }
}
bool ds4_metal_single_readback_event(void) {
    return ds4_gpu_signal_batch_and_wait_event("selected-id readback");
}
static void ds4_single_bind_entry(DS4RoutedSingleResources *r, uint32_t slot) {
    ds4_gpu_stream_expert_cache_entry *entry = r->slots[slot];
    r->slotBuffers[0][slot] = entry->gate_buffer;
    r->slotOffsets[0][slot] = entry->gate_inner;
    r->slotBuffers[1][slot] = entry->up_buffer;
    r->slotOffsets[1][slot] = entry->up_inner;
    r->slotBuffers[2][slot] = entry->down_buffer;
    r->slotOffsets[2][slot] = entry->down_inner;
    r->entries[slot] = entry;
    r->entryCount = r->request.used_experts;
}
bool ds4_metal_single_peek(ds4_metal_single_resources *opaque, uint32_t slot, uint32_t expert,
                           uint64_t gate, uint64_t up, uint64_t down) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        r->slots[slot] = ds4_gpu_stream_expert_cache_peek(
            q->model, q->model_bytes, q->layer, expert, q->total_experts, q->used_experts, gate, up,
            down, q->gate_expert_bytes, q->down_expert_bytes);
        if (!r->slots[slot])
            return false;
        ds4_single_bind_entry(r, slot);
        return true;
    }
}
static ds4_gpu_stream_expert_table ds4_single_table(DS4RoutedSingleResources *r) {
    ds4_metal_routed_request *q = &r->request;
    return (ds4_gpu_stream_expert_table){.model_map = q->model,
                                         .model_size = q->model_bytes,
                                         .layer = q->layer,
                                         .n_total_expert = q->total_experts,
                                         .gate_offset = q->gate_offset,
                                         .up_offset = q->up_offset,
                                         .down_offset = q->down_offset,
                                         .gate_expert_bytes = q->gate_expert_bytes,
                                         .down_expert_bytes = q->down_expert_bytes};
}
bool ds4_metal_single_begin_load(ds4_metal_single_resources *opaque, const int32_t *ids) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        const ds4_gpu_stream_expert_table table = ds4_single_table(r);
        return ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids,
                                                               r->request.used_experts);
    }
}
int ds4_metal_single_pending_gate(ds4_metal_single_resources *opaque, const int32_t *ids,
                                  uint32_t missing) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        if (!ds4_gpu_stream_expert_pending_load_matches(q->model, q->model_bytes, q->layer, ids,
                                                        q->total_experts, q->used_experts,
                                                        q->gate_expert_bytes, q->down_expert_bytes))
            return 0;
        const int ready = ds4_gpu_stream_expert_pending_gate_wait();
        if (!ready)
            return -1;
        if (ready == 2) {
            ds4_gpu_stream_expert_pending_load *pending = &g_stream_expert_pending_load;
            for (uint32_t i = 0; i < q->used_experts; i++) {
                if (!(missing & (1u << i)))
                    continue;
                const uint32_t source = pending->source_slots[i];
                uint32_t j = 0;
                while (j < pending->n_loads && pending->load_slots[j] != source)
                    j++;
                if (j == pending->n_loads)
                    return -1;
                r->slotBuffers[0][i] = pending->gate_bufs[j];
                r->slotBuffers[1][i] = pending->up_bufs[j];
                r->slotOffsets[0][i] = pending->gate_inners[j];
                r->slotOffsets[1][i] = pending->up_inners[j];
            }
        }
        return ready;
    }
}
bool ds4_metal_single_load_missing(ds4_metal_single_resources *opaque, const int32_t *ids,
                                   const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                                   uint32_t missing) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        return ds4_gpu_stream_expert_cache_load_selected_missing(
            q->model, q->model_bytes, q->layer, ids, q->total_experts, q->used_experts, gate, up,
            down, q->gate_expert_bytes, q->down_expert_bytes, missing, r->slots);
    }
}
bool ds4_metal_single_bind_loaded(ds4_metal_single_resources *opaque, uint32_t missing) {
    DS4RoutedSingleResources *r = SINGLE_CONTEXT;
    for (uint32_t i = 0; i < r->request.used_experts; i++) {
        if (!(missing & (1u << i)))
            continue;
        if (!r->slots[i])
            return false;
        ds4_single_bind_entry(r, i);
    }
    return true;
}
bool ds4_metal_single_mark(ds4_metal_single_resources *opaque, uint32_t active) {
    DS4RoutedSingleResources *r = SINGLE_CONTEXT;
    return ds4_gpu_stream_expert_cache_mark_entries_inflight(r->slots, r->request.used_experts,
                                                             active);
}
bool ds4_metal_single_mark_present(ds4_metal_single_resources *opaque) {
    DS4RoutedSingleResources *r = SINGLE_CONTEXT;
    for (uint32_t i = 0; i < r->request.used_experts; i++)
        if (r->slots[i] && !ds4_gpu_stream_expert_cache_mark_inflight(r->slots[i]))
            return false;
    return true;
}
bool ds4_metal_single_publish_addresses(ds4_metal_single_resources *opaque, const int32_t *ids) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        for (uint32_t i = 0; i < r->request.used_experts; i++) {
            ds4_gpu_stream_expert_cache_entry *entry = r->slots[i];
            if (!entry)
                return false;
            if (!ds4_gpu_stream_expert_cache_set_addr_slot_raw(
                    r->request.layer, (uint32_t)ids[i], entry->gate_buffer, entry->gate_inner,
                    entry->up_buffer, entry->up_inner, entry->down_buffer, entry->down_inner))
                return false;
        }
        return true;
    }
}
void ds4_metal_single_prune(ds4_metal_single_resources *opaque, const int32_t *ids) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        ds4_metal_routed_request *q = &r->request;
        ds4_gpu_stream_expert_cache_prune_layer(q->layer, q->total_experts, q->used_experts, ids,
                                                q->used_experts);
        ds4_gpu_stream_expert_cache_prune_global(q->layer, ids, q->used_experts);
    }
}
bool ds4_metal_single_compact(ds4_metal_single_resources *opaque) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        id<MTLBuffer> gate = nil, up = nil, down = nil, selected = nil;
        int ok = ds4_gpu_stream_compact_addr_prepare(
            r->request.layer, r->slots, r->request.used_experts, &gate, &up, &down, &selected);
        r->gateAddress = gate;
        r->upAddress = up;
        r->downAddress = down;
        r->selectedExecution = selected;
        r->selectedExecutionOffset = 0;
        return ok != 0;
    }
}
bool ds4_metal_single_host_ids(ds4_metal_single_resources *opaque, const int32_t *ids) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        id<MTLBuffer> selected = nil;
        NSUInteger offset = 0;
        int ok = ds4_gpu_stream_selected_ids_prepare(r->request.layer, ids, r->request.used_experts,
                                                     &selected, &offset);
        r->selectedExecution = selected;
        r->selectedExecutionOffset = offset;
        return ok != 0;
    }
}
void ds4_metal_single_gate_submitted(void) { g_stream_expert_gate_submissions++; }
bool ds4_metal_single_wait_resident(void) {
    return ds4_gpu_wait_pending_command_buffers("streaming expert split resident");
}
void ds4_metal_single_note_selected(double sync, double copy, double bind, uint32_t resident,
                                    uint32_t missing, bool cache) {
    if (cache)
        ds4_gpu_stream_expert_timing_note_cache_class(resident, missing);
    ds4_gpu_stream_expert_timing_note_selected(sync, copy, bind);
}
void ds4_metal_single_note_split(uint32_t resident, uint32_t missing, double submit, double total,
                                 const double *details) {
    ds4_gpu_stream_expert_timing_note_split(resident, missing, submit, total);
    ds4_gpu_stream_expert_timing_note_split_missing_detail(details[0], details[1], details[2],
                                                           details[3], details[4]);
}
bool ds4_metal_single_parallel_pending(ds4_metal_single_resources *opaque, void *command,
                                       ds4_metal_single_parallel *after) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        bool ok = !g_parallel_q8_pending || ds4_gpu_parallel_q8_matvec_encode_pending(
                                                cb, ds4_gpu_tensor_buffer(r->request.mid));
        *after = (ds4_metal_single_parallel){g_parallel_q8_pending, g_parallel_ffn_mode,
                                             g_parallel_ffn_stage};
        return ok;
    }
}
bool ds4_metal_single_parallel_second(void *command) {
    @autoreleasepool {
        return g_parallel_ffn_mode != 2 || g_parallel_ffn_stage != 1 ||
               ds4_gpu_parallel_ffn_encode_second_stage((__bridge id<MTLCommandBuffer>)command) !=
                   0;
    }
}
bool ds4_metal_single_encode(ds4_metal_single_resources *opaque, void *command,
                             const ds4_metal_single_launch *p) {
    @autoreleasepool {
        DS4RoutedSingleResources *r = SINGLE_CONTEXT;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !pipeline)
            return false;
        // Preserve the original helpers' nil-buffer admission before opening
        // an encoder. The native controller owns dimensions and dispatches.
        switch (p->kind) {
        case DS4_SINGLE_ARRAY_PAIR:
        case DS4_SINGLE_ARRAY_SUM:
            for (uint32_t i = 0; i < 6; i++) {
                if (p->kind == DS4_SINGLE_ARRAY_PAIR
                        ? (!r->slotBuffers[0][i] || !r->slotBuffers[1][i])
                        : !r->slotBuffers[2][i])
                    return false;
            }
            break;
        case DS4_SINGLE_GROUP_ARRAY_PAIR:
        case DS4_SINGLE_GROUP_ARRAY_SUM:
            for (uint32_t i = 0; i < p->array_count; i++) {
                if (p->kind == DS4_SINGLE_GROUP_ARRAY_PAIR
                        ? (!r->groupBuffers[0][i] || !r->groupBuffers[1][i])
                        : !r->groupBuffers[2][i])
                    return false;
            }
            break;
        case DS4_SINGLE_GROUP24_MV:
        case DS4_SINGLE_GROUP24_SUM:
            for (uint32_t i = 0; i < 24; i++)
                if (!r->groupBuffers[p->part][i])
                    return false;
            break;
        case DS4_SINGLE_GATHER:
            for (uint32_t i = 0; i < 6; i++)
                if (!r->groupBuffers[p->part][i])
                    return false;
            if (!r->slotBuffers[p->part][0])
                return false;
            break;
        case DS4_SINGLE_GROUP_PAIR:
            if (!r->gateWeight || !r->upWeight)
                return false;
            break;
        case DS4_SINGLE_GROUP_SUM:
            if (!r->downWeight)
                return false;
            break;
        case DS4_SINGLE_Q4_ADDRESS_PAIR:
            if (!r->gateTable.addressBuffer || !r->upTable.addressBuffer)
                return false;
            break;
        case DS4_SINGLE_Q4_ADDRESS_SUM:
        case DS4_SINGLE_MASKED_SUM:
            if (p->kind == DS4_SINGLE_Q4_ADDRESS_SUM ? !r->downTable.addressBuffer
                                                     : !r->downAddress)
                return false;
            break;
        case DS4_SINGLE_MASKED_PAIR:
            if (!r->gateAddress || !r->upAddress)
                return false;
            break;
        default:
            return false;
        }
        const bool masked = p->kind == DS4_SINGLE_MASKED_PAIR || p->kind == DS4_SINGLE_MASKED_SUM;
        if (masked) {
            for (uint32_t i = 0; i < 6; i++) {
                if (!(p->split.active_mask & (1u << i)))
                    continue;
                ds4_gpu_stream_expert_cache_entry *entry = r->slots[i];
                if (!entry ||
                    (p->kind == DS4_SINGLE_MASKED_PAIR ? (!entry->gate_buffer || !entry->up_buffer)
                                                       : !entry->down_buffer))
                    return false;
            }
            if (!ds4_metal_single_mark(opaque, p->split.active_mask))
                return false;
        }
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
#define A(slot, field) [enc setBytes:&p->field length:sizeof(p->field) atIndex:slot]
#define B(slot, role)                                                                              \
    do {                                                                                           \
        NSUInteger at;                                                                             \
        id<MTLBuffer> b = ds4_routed_buffer(r, role, &at);                                         \
        [enc setBuffer:b offset:at atIndex:slot];                                                  \
    } while (0)
#define G(slot, part, index)                                                                       \
    [enc setBuffer:r->groupBuffers[part][index] offset:r->groupOffsets[part][index] atIndex:slot]
#define S(slot, part, index)                                                                       \
    [enc setBuffer:r->slotBuffers[part][index] offset:r->slotOffsets[part][index] atIndex:slot]
#define D(slot, buffer) [enc setBuffer:buffer offset:0 atIndex:slot]
        switch (p->kind) {
        case DS4_SINGLE_GROUP_PAIR:
            A(0, mv);
            A(1, activation);
            A(2, group);
            B(3, DS4_ROUTED_W_GATE);
            B(4, DS4_ROUTED_W_UP);
            B(5, DS4_ROUTED_X);
            B(6, DS4_ROUTED_GATE);
            B(7, DS4_ROUTED_UP);
            B(8, DS4_ROUTED_MID);
            B(9, DS4_ROUTED_SELECTED);
            B(10, DS4_ROUTED_WEIGHTS);
            break;
        case DS4_SINGLE_GROUP_SUM:
            A(0, mv);
            A(1, group);
            B(2, DS4_ROUTED_W_DOWN);
            B(3, DS4_ROUTED_MID);
            B(4, DS4_ROUTED_OUT);
            B(5, DS4_ROUTED_SELECTED);
            break;
        case DS4_SINGLE_GROUP_ARRAY_PAIR:
            A(0, mv);
            A(1, activation);
            for (uint32_t i = 0; i < p->array_count; i++)
                G(2 + i, 0, i);
            for (uint32_t i = 0; i < p->array_count; i++)
                G(2 + p->array_count + i, 1, i);
            {
                const uint32_t first = 2 + 2 * p->array_count;
                B(first, DS4_ROUTED_X);
                B(first + 1, DS4_ROUTED_GATE);
                B(first + 2, DS4_ROUTED_UP);
                B(first + 3, DS4_ROUTED_MID);
                B(first + 4, DS4_ROUTED_SELECTED);
                B(first + 5, DS4_ROUTED_WEIGHTS);
            }
            break;
        case DS4_SINGLE_GROUP_ARRAY_SUM:
            A(0, mv);
            for (uint32_t i = 0; i < p->array_count; i++)
                G(1 + i, 2, i);
            B(1 + p->array_count, DS4_ROUTED_MID);
            B(2 + p->array_count, DS4_ROUTED_OUT);
            B(3 + p->array_count, DS4_ROUTED_SELECTED);
            break;
        case DS4_SINGLE_GROUP24_MV:
        case DS4_SINGLE_GROUP24_SUM:
            A(0, mv);
            for (uint32_t i = 0; i < 24; i++)
                G(1 + i, p->part, i);
            B(25, p->kind == DS4_SINGLE_GROUP24_SUM ? DS4_ROUTED_MID : DS4_ROUTED_X);
            B(26, p->kind == DS4_SINGLE_GROUP24_SUM ? DS4_ROUTED_OUT
                  : p->part == 0                    ? DS4_ROUTED_GATE
                                                    : DS4_ROUTED_UP);
            B(27, DS4_ROUTED_SELECTED);
            break;
        case DS4_SINGLE_ARRAY_PAIR:
            A(0, mv);
            A(1, activation);
            for (uint32_t i = 0; i < 6; i++)
                S(2 + i, 0, i);
            for (uint32_t i = 0; i < 6; i++)
                S(8 + i, 1, i);
            B(14, DS4_ROUTED_X);
            B(15, DS4_ROUTED_GATE);
            B(16, DS4_ROUTED_UP);
            B(17, DS4_ROUTED_MID);
            B(18, DS4_ROUTED_WEIGHTS);
            break;
        case DS4_SINGLE_ARRAY_SUM:
            A(0, mv);
            for (uint32_t i = 0; i < 6; i++)
                S(1 + i, 2, i);
            B(7, DS4_ROUTED_MID);
            B(8, DS4_ROUTED_OUT);
            break;
        case DS4_SINGLE_Q4_ADDRESS_PAIR:
            A(0, mv);
            A(1, activation);
            D(2, r->gateTable.addressBuffer);
            D(3, r->upTable.addressBuffer);
            B(4, DS4_ROUTED_X);
            B(5, DS4_ROUTED_GATE);
            B(6, DS4_ROUTED_UP);
            B(7, DS4_ROUTED_MID);
            B(8, DS4_ROUTED_SELECTED);
            B(9, DS4_ROUTED_WEIGHTS);
            if (!ds4_gpu_bind_q4_expert_table_anchors(enc, r->gateTable, 10, 6) ||
                !ds4_gpu_bind_q4_expert_table_anchors(enc, r->upTable, 16, 6)) {
                ds4_gpu_end_compute_encoder(cb, enc);
                return false;
            }
            if (p->use_resources) {
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->gateTable, false);
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->upTable, false);
            }
            break;
        case DS4_SINGLE_Q4_ADDRESS_SUM:
            A(0, mv);
            D(1, r->downTable.addressBuffer);
            B(2, DS4_ROUTED_MID);
            B(3, DS4_ROUTED_OUT);
            B(4, DS4_ROUTED_SELECTED);
            if (!ds4_gpu_bind_q4_expert_table_anchors(enc, r->downTable, 5, 6)) {
                ds4_gpu_end_compute_encoder(cb, enc);
                return false;
            }
            if (p->use_resources)
                ds4_gpu_use_q4_expert_table_resources(cb, enc, r->downTable, false);
            break;
        case DS4_SINGLE_MASKED_PAIR:
            A(0, mv);
            A(1, activation);
            A(2, split);
            D(3, r->gateAddress);
            D(4, r->upAddress);
            B(5, DS4_ROUTED_X);
            B(6, DS4_ROUTED_GATE);
            B(7, DS4_ROUTED_UP);
            B(8, DS4_ROUTED_MID);
            B(9, DS4_ROUTED_SELECTED);
            B(10, DS4_ROUTED_WEIGHTS);
            for (uint32_t i = 0; i < 6; i++)
                if (p->split.active_mask & (1u << i)) {
                    [enc useResource:r->slots[i]->gate_buffer usage:MTLResourceUsageRead];
                    [enc useResource:r->slots[i]->up_buffer usage:MTLResourceUsageRead];
                }
            break;
        case DS4_SINGLE_MASKED_SUM:
            A(0, mv);
            A(1, split);
            D(2, r->downAddress);
            B(3, DS4_ROUTED_MID);
            B(4, DS4_ROUTED_OUT);
            B(5, DS4_ROUTED_SELECTED);
            for (uint32_t i = 0; i < 6; i++)
                if (p->split.active_mask & (1u << i))
                    [enc useResource:r->slots[i]->down_buffer usage:MTLResourceUsageRead];
            break;
        case DS4_SINGLE_GATHER:
            A(0, gather);
            for (uint32_t i = 0; i < 6; i++)
                G(1 + i, p->part, i);
            B(7, DS4_ROUTED_SELECTED);
            S(8, p->part, 0);
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
#undef A
#undef B
#undef G
#undef S
#undef D
    }
}
#undef SINGLE_CONTEXT
