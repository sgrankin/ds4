/* Fixed ARC/storage driver for the native loader. No load scheduling, cache
 * selection, slab policy, IDs or byte-range calculation lives here. */
ds4_metal_expert_load_view ds4_metal_expert_load_driver(void) {
    /* Scalar/pointer facts only: pread workers do not enter Foundation or Metal. */
    return (ds4_metal_expert_load_view){
        .cache = &g_expert_cache,
        .load = &g_expert_load,
        .model_fd = g_model_fd,
        .streaming = g_ssd_streaming_mode,
        .device_ready = g_device != nil,
        .batch_active = g_batch_cb != nil,
        .quality = g_quality_mode,
        .prefill_building = g_stream_prefill_batch_selected_addr_building,
        .glm_building = g_glm_stream_expert_addr_table_building,
    };
}
ds4_metal_expert_owned_buffer *ds4_metal_expert_buffer_new(uint64_t bytes, const char *label) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                                     options:MTLResourceStorageModeShared];
        if (buffer)
            buffer.label = label ? [NSString stringWithUTF8String:label] : nil;
        return buffer ? (ds4_metal_expert_owned_buffer *)(__bridge_retained void *)buffer : NULL;
    }
}
void ds4_metal_expert_buffer_release(ds4_metal_expert_owned_buffer *opaque) {
    @autoreleasepool {
        id held = (__bridge_transfer id)opaque;
        (void)held;
    }
}
void ds4_metal_expert_reuse_assign(ds4_gpu_stream_expert_reusable_buffers *out,
                                   ds4_metal_expert_buffer_ref gate, ds4_metal_expert_buffer_ref up,
                                   ds4_metal_expert_buffer_ref down, uintptr_t gate_inner,
                                   uintptr_t up_inner, uintptr_t down_inner) {
    @autoreleasepool {
        out->gate_buffer = gate;
        out->up_buffer = up;
        out->down_buffer = down;
        out->gate_inner = gate_inner;
        out->up_inner = up_inner;
        out->down_inner = down_inner;
    }
}
void ds4_metal_expert_slab_assign(ds4_metal_expert_cache_state *state, uint32_t slot,
                                  ds4_metal_expert_buffer_ref buffer) {
    @autoreleasepool {
        state->cache_slabs[slot] = buffer;
    }
}
void ds4_metal_expert_pending_assign(ds4_gpu_stream_expert_pending_load *pending, uint32_t slot,
                                     const ds4_gpu_stream_expert_reusable_buffers *buffers) {
    @autoreleasepool {
        pending->gate_bufs[slot] = buffers->gate_buffer;
        pending->up_bufs[slot] = buffers->up_buffer;
        pending->down_bufs[slot] = buffers->down_buffer;
        pending->gate_inners[slot] = buffers->gate_inner;
        pending->up_inners[slot] = buffers->up_inner;
        pending->down_inners[slot] = buffers->down_inner;
    }
}
void ds4_metal_expert_pending_retire(ds4_gpu_stream_expert_pending_load *pending) {
    @autoreleasepool {
        for (uint32_t slot = 0; slot < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; slot++) {
            pending->gate_bufs[slot] = nil;
            pending->up_bufs[slot] = nil;
            pending->down_bufs[slot] = nil;
            pending->gate_inners[slot] = pending->up_inners[slot] = pending->down_inners[slot] = 0;
        }
    }
}
bool ds4_metal_expert_load_addresses(uint32_t layer,
                                     ds4_gpu_stream_expert_reusable_buffers *addresses) {
    @autoreleasepool {
        id<MTLBuffer> gate = nil, up = nil, down = nil;
        int ok = ds4_gpu_stream_expert_cache_addr_buffers(layer, &gate, &up, &down);
        if (ok && addresses)
            ds4_metal_expert_reuse_assign(addresses, gate, up, down, 0, 0, 0);
        return ok != 0;
    }
}
bool ds4_metal_expert_load_address_install(uint32_t layer, uint32_t expert,
                                           const ds4_gpu_stream_expert_reusable_buffers *buffers) {
    @autoreleasepool {
        return ds4_gpu_stream_expert_cache_set_addr_slot(
                   layer, expert, buffers->gate_buffer, buffers->gate_inner, buffers->up_buffer,
                   buffers->up_inner, buffers->down_buffer, buffers->down_inner) != 0;
    }
}
bool ds4_metal_expert_seed_blit(const ds4_metal_expert_copy_sources *sources,
                                const ds4_gpu_stream_expert_reusable_buffers *destinations,
                                uint32_t count, uint64_t gate_bytes, uint64_t down_bytes) {
    @autoreleasepool {
        ds4_gpu_close_batch_encoder();
        id<MTLBlitCommandEncoder> blit =
            ds4_gpu_blit_encoder(g_batch_cb, "expert_cache_fill", count * 3u);
        if (!blit)
            return false;
        for (uint32_t slot = 0; slot < count; slot++) {
            const ds4_metal_expert_copy_sources *source = &sources[slot];
            const ds4_gpu_stream_expert_reusable_buffers *destination = &destinations[slot];
            [blit copyFromBuffer:(__bridge id<MTLBuffer>)source->gate
                     sourceOffset:source->gate_inner
                         toBuffer:destination->gate_buffer
                destinationOffset:destination->gate_inner
                             size:(NSUInteger)gate_bytes];
            [blit copyFromBuffer:(__bridge id<MTLBuffer>)source->up
                     sourceOffset:source->up_inner
                         toBuffer:destination->up_buffer
                destinationOffset:destination->up_inner
                             size:(NSUInteger)gate_bytes];
            [blit copyFromBuffer:(__bridge id<MTLBuffer>)source->down
                     sourceOffset:source->down_inner
                         toBuffer:destination->down_buffer
                destinationOffset:destination->down_inner
                             size:(NSUInteger)down_bytes];
        }
        [blit endEncoding];
        g_batch_has_work = YES;
        return true;
    }
}
