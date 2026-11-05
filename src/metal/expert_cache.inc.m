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
bool ds4_metal_expert_slab_residency_add(ds4_metal_expert_cache_state *state,
        ds4_metal_expert_buffer_ref buffer, bool queue, bool profile) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            if (!g_device || (queue && (!g_queue ||
                ![g_queue respondsToSelector:@selector(addResidencySet:)] ||
                ![g_queue respondsToSelector:@selector(removeResidencySet:)]))) {
                fprintf(stderr, "ds4: streaming expert slab residency queue unavailable; using ordinary slab declarations\n");
                return false;
            }
            id<MTLResidencySet> set = state->cache_slab_residency_set;
            if (!set) {
                MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
                desc.label = @"ds4_stream_expert_slabs";
                desc.initialCapacity = DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS;
                NSError *error = nil;
                set = [g_device newResidencySetWithDescriptor:desc error:&error];
                if (!set) {
                    fprintf(stderr, "ds4: streaming expert slab residency creation failed: %s\n",
                            [[error localizedDescription] UTF8String]);
                    return false;
                }
            }
            /* Commit actual slabs before publication. Mode 1 attaches here;
             * mode 2 attaches at expert consumption, never CPU cache pinning.
             * Existing declarations retain hazard tracking in both modes. */
            [set addAllocation:buffer];
            [set commit];
            state->cache_slab_residency_used_batch = 0;
            if (!state->cache_slab_residency_set) {
                state->cache_slab_residency_set = set;
                state->cache_slab_residency_uses = 0;
                if (queue) {
                    [g_queue addResidencySet:set];
                    state->cache_slab_residency_attached = 1;
                }
            }
            if (profile)
                fprintf(stderr, "ds4: streaming expert slab residency add allocations=%llu bytes=%.2f GiB mode=%s\n",
                        (unsigned long long)set.allocationCount,
                        (double)set.allocatedSize / 1073741824.0,
                        state->cache_slab_residency_attached ? "queue" : "command");
            return true;
        }
#endif
        fprintf(stderr, "ds4: streaming expert slab residency requires macOS 15 or newer; using ordinary slab declarations\n");
        return false;
    }
}
void ds4_metal_expert_slab_residency_use(void *command) {
    ds4_metal_expert_cache_state *state = &g_expert_cache;
    if (!state->cache_slab_residency_set || state->cache_slab_residency_attached || !command)
        return;
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLResidencySet> set = state->cache_slab_residency_set;
        const char *once_setting = getenv("DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE");
        const bool once = cb == g_batch_cb && once_setting && !strcmp(once_setting, "1");
        /* The active batch's creation hook resets this marker, so no raw CB
         * identity survives its lifetime. Standalone CBs keep ordinary use;
         * set commits/clear conservatively invalidate the current proof. */
        if (once && state->cache_slab_residency_used_batch)
            return;
        if (![cb respondsToSelector:@selector(useResidencySet:)]) {
            state->cache_slab_residency_disabled = 1;
            ds4_metal_expert_slab_residency_clear(state,
                ds4_gpu_stream_expert_timing_summary_enabled(), "command API unavailable");
            return;
        }
        /* This covers the whole CB, including any earlier/later router work.
         * Cache pins and useResource still own actual reads and hazards. The
         * extra reference only keeps this set object alive across pressure
         * clear when the caller opted into unretained command buffers. */
        [cb useResidencySet:set];
        if (!cb.retainedReferences) {
            [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                (void)completed;
                __attribute__((objc_precise_lifetime)) id keepAlive = set;
                (void)keepAlive;
            }];
        }
        if (once)
            state->cache_slab_residency_used_batch = 1;
        state->cache_slab_residency_uses++;
        if (g_timeline_enabled && g_timeline_batch && cb == g_batch_cb &&
            ds4_gpu_stream_expert_timing_summary_enabled())
            fprintf(stderr, "ds4: streaming expert slab residency use seq=%llu mode=command\n",
                    (unsigned long long)g_timeline_batch->seq);
    }
#endif
}
void ds4_metal_expert_slab_residency_clear(ds4_metal_expert_cache_state *state,
        bool profile, const char *reason) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            id<MTLResidencySet> set = state->cache_slab_residency_set;
            if (set) {
                if (profile)
                    fprintf(stderr, "ds4: streaming expert slab residency clear reason=%s allocations=%llu bytes=%.2f GiB mode=%s uses=%llu\n",
                            reason, (unsigned long long)set.allocationCount,
                            (double)set.allocatedSize / 1073741824.0,
                            state->cache_slab_residency_attached ? "queue" : "command",
                            (unsigned long long)state->cache_slab_residency_uses);
                if (state->cache_slab_residency_attached && g_queue &&
                    [g_queue respondsToSelector:@selector(removeResidencySet:)])
                    [g_queue removeResidencySet:set];
                [set endResidency];
                [set removeAllAllocations];
                [set commit];
            }
        }
#endif
        state->cache_slab_residency_attached = 0;
        state->cache_slab_residency_set = nil;
        state->cache_slab_residency_used_batch = 0;
        state->cache_slab_residency_uses = 0;
    }
}
void ds4_metal_expert_retire_slab(ds4_metal_expert_cache_state *state, uint32_t index) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            if (state->cache_slab_residency_set && state->cache_slabs[index]) {
                [state->cache_slab_residency_set removeAllocation:state->cache_slabs[index]];
                [state->cache_slab_residency_set commit];
                state->cache_slab_residency_used_batch = 0;
            }
        }
#endif
        state->cache_slabs[index] = nil;
    }
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
