/* ARC/Metal leaves for the shared actual streaming-cache records. Native code
 * never writes a strong field, including during reset or reuse transfer. */
void ds4_metal_expert_reuse_reset(ds4_gpu_stream_expert_reusable_buffers *reuse) {
    if (!reuse) return;
    @autoreleasepool {
        reuse->gate_buffer = nil;
        reuse->up_buffer = nil;
        reuse->down_buffer = nil;
        reuse->gate_inner = reuse->up_inner = reuse->down_inner = 0;
    }
}
void ds4_metal_expert_entry_retire(ds4_gpu_stream_expert_cache_entry *entry,
                                   ds4_gpu_stream_expert_reusable_buffers *reuse) {
    @autoreleasepool {
        if (reuse) {
            /* Strong destinations retain BEFORE the original references retire. */
            reuse->gate_buffer = entry->gate_buffer;
            reuse->up_buffer = entry->up_buffer;
            reuse->down_buffer = entry->down_buffer;
            reuse->gate_inner = entry->gate_inner;
            reuse->up_inner = entry->up_inner;
            reuse->down_inner = entry->down_inner;
        }
        entry->gate_buffer = nil;
        entry->up_buffer = nil;
        entry->down_buffer = nil;
    }
}
void ds4_metal_expert_entry_retain(ds4_gpu_stream_expert_cache_entry *entry,
        ds4_metal_expert_buffer_ref gate, ds4_metal_expert_buffer_ref up,
        ds4_metal_expert_buffer_ref down) {
    @autoreleasepool {
        entry->gate_buffer = gate;
        entry->up_buffer = up;
        entry->down_buffer = down;
    }
}
ds4_metal_expert_buffer_memory ds4_metal_expert_buffer_storage(ds4_metal_expert_buffer_ref buffer) {
    @autoreleasepool {
        /* The caller borrows a still-retained entry/reuse/slab reference. The
         * bytes remain valid after this pool ends, until that owner retires. */
        return (ds4_metal_expert_buffer_memory){buffer ? [buffer contents] : NULL,
                                               buffer ? (uint64_t)[buffer length] : 0};
    }
}
void ds4_metal_expert_zero_address(uint32_t layer, uint32_t expert) {
    @autoreleasepool { ds4_gpu_stream_expert_cache_zero_addr_slot(layer, expert); }
}
void ds4_metal_expert_full_layer_retire(uint32_t layer) {
    /* Existing complete address-table library action. Its metadata construction
     * and retirement remain C until the subsequent table transaction owner. */
    @autoreleasepool { ds4_gpu_stream_full_expert_addr_clear_layer(layer); }
}
void ds4_metal_expert_zero_layer_address(uint32_t layer) {
    @autoreleasepool {
        id<MTLBuffer> buffers[3] = {g_stream_expert_cache_gate_addr_buffers[layer],
                                  g_stream_expert_cache_up_addr_buffers[layer],
                                  g_stream_expert_cache_down_addr_buffers[layer]};
        for (uint32_t i = 0; i < 3; i++) {
            if (buffers[i]) memset([buffers[i] contents], 0,
                (NSUInteger)DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT * sizeof(uint64_t));
        }
    }
}
void ds4_metal_expert_retire_slab(ds4_metal_expert_cache_state *state, uint32_t index) {
    @autoreleasepool { state->cache_slabs[index] = nil; }
}
bool ds4_metal_expert_install_address(uint32_t layer, uint32_t expert,
        ds4_metal_expert_buffer_ref gate, uintptr_t gate_inner,
        ds4_metal_expert_buffer_ref up, uintptr_t up_inner,
        ds4_metal_expert_buffer_ref down, uintptr_t down_inner) {
    @autoreleasepool {
        return ds4_gpu_stream_expert_cache_set_addr_slot(layer, expert, gate, gate_inner,
                                                          up, up_inner, down, down_inner) != 0;
    }
}
/* Whole dependencies explicitly retained for the next pending-load owner and
 * the actual Metal command driver. No per-field callback or copied cache. */
void ds4_metal_expert_pending_clear(void) { ds4_gpu_stream_expert_pending_load_clear(); }
bool ds4_metal_expert_wait(const char *label) {
    @autoreleasepool { return ds4_gpu_stream_expert_cache_wait_inflight(label) != 0; }
}
bool ds4_metal_expert_on_service_thread(void) { return ds4_gpu_stream_expert_cache_on_service_thread() != 0; }
double ds4_metal_expert_now_ms(void) { return ds4_gpu_now_ms(); }
_Static_assert(sizeof(uintptr_t) == sizeof(NSUInteger), "shared Metal inner offsets");
_Static_assert(offsetof(ds4_gpu_stream_expert_cache_entry, model_map) == 3 * sizeof(id<MTLBuffer>),
               "shared strong entry prefix");
_Static_assert(sizeof(ds4_gpu_stream_expert_reusable_buffers) == 6 * sizeof(uintptr_t),
               "shared retained reuse record");
