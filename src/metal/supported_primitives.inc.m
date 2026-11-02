/* Fixed bindings for the original carry/pre-RoPE/candidate shader packets. */
int ds4_metal_supported_encode(void *command,
                               const ds4_metal_supported_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)p->pipeline;
    if (!cb || !pipeline)
      return 0;
    const NSUInteger words =
        p->kind == DS4_TAIL_CARRY                                    ? 5u
        : (p->kind == DS4_TAIL_BLOCKS || p->kind == DS4_TAIL_FILTER) ? 5u
                                                                     : 3u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
      return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:p->args length:words * sizeof(uint32_t) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)p->first.buffer
            offset:p->first.offset
           atIndex:1];
    if (p->kind != DS4_TAIL_QUANTIZE)
      [enc setBuffer:(__bridge id<MTLBuffer>)p->second.buffer
              offset:p->second.offset
             atIndex:2];
    if (p->kind == DS4_TAIL_BLOCKS || p->kind == DS4_TAIL_FILTER)
      [enc setBuffer:(__bridge id<MTLBuffer>)p->third.buffer
              offset:p->third.offset
             atIndex:3];
    if (p->dispatch_threads)
      [enc dispatchThreads:MTLSizeMake(p->x, p->y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    else
      [enc dispatchThreadgroups:MTLSizeMake(p->x, p->y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}

int ds4_metal_v41_encode(void *command, const ds4_metal_v41_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = p->nsg > 0 ?
            ds4_gpu_get_mul_mv_pipeline(p->kernel, p->nsg) : ds4_gpu_get_pipeline(p->kernel);
        if (!cb || !pipeline || !p->tx || p->tx*p->ty > pipeline.maxTotalThreadsPerThreadgroup) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        ds4_metal_v41_note(p->kernel);
        [enc setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < p->count; i++) {
            const ds4_metal_v41_binding *b = &p->bindings[i];
            if (b->bytes) [enc setBytes:b->bytes length:(NSUInteger)b->length atIndex:b->index];
            else [enc setBuffer:(__bridge id<MTLBuffer>)b->view.buffer offset:(NSUInteger)b->view.offset atIndex:b->index];
        }
        if (p->shared_bytes) [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->gx, p->gy, p->gz)
             threadsPerThreadgroup:MTLSizeMake(p->tx, p->ty, 1)];
        if (p->barrier && enc.dispatchType == MTLDispatchTypeConcurrent)
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
ds4_metal_attention_view ds4_metal_v41_completion(const char *name, uint64_t bytes) {
    @autoreleasepool {
        if (!g_dsv4_completion_cache) return (ds4_metal_attention_view){0};
        NSString *key = [NSString stringWithUTF8String:name];
        id<MTLBuffer> buffer = [g_dsv4_completion_cache objectForKey:key];
        if (!buffer || buffer.length < bytes) {
            buffer = [g_device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
            if (!buffer) return (ds4_metal_attention_view){0};
            memset(buffer.contents, 0, (size_t)bytes);
            [g_dsv4_completion_cache setObject:buffer forKey:key];
        }
        [g_transient_buffers addObject:buffer];
        return (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge_retained void *)buffer, 0, buffer.length};
    }
}

uint32_t ds4_gpu_v41_off(void) {
    DS4MetalContext *context = ds4_metal_context_current();
    if (!context->v41_off_read) {
        static const char *names[] = {
            "DS4_METAL_DISABLE_V41_DEFERRED_HC", "DS4_METAL_DISABLE_V41_LITERAL_SHAPES",
            "DS4_METAL_DISABLE_V41_INDEXER_E4M3", "DS4_METAL_MODEL_TRACKED",
            "DS4_METAL_DISABLE_V41_ROPE_TABLE", "DS4_METAL_DISABLE_V41_SPLIT_REDUCE",
            "DS4_METAL_DISABLE_V41_INDEX_BF16", "DS4_METAL_DISABLE_V41_DECODE_TOPK",
        };
        for (uint32_t i = 0; i < sizeof(names)/sizeof(*names); i++) {
            const char *value = getenv(names[i]);
            if (value && *value && strcmp(value, "0")) context->v41_off |= 1u << i;
        }
        context->v41_off_read = true;
    }
    return context->v41_off;
}
void ds4_gpu_test_set_v41_off(uint32_t off) {
    DS4MetalContext *context = ds4_metal_context_current();
    context->v41_off = off;
    context->v41_off_read = true;
}
int ds4_metal_v41_on(uint32_t path) { return (ds4_gpu_v41_off() & path) == 0; }
uint32_t ds4_metal_v41_test_flags(void) { return g_test_flags; }
void ds4_metal_v41_note(const char *name) {
    if (!(g_test_flags & DS4_GPU_TEST_COUNT_KERNELS)) return;
    @autoreleasepool {
        DS4MetalContext *context = ds4_metal_context_current();
        if (!context->v41_kernel_runs) context->v41_kernel_runs = [NSMutableDictionary dictionary];
        NSString *key = [NSString stringWithUTF8String:name];
        context->v41_kernel_runs[key] = @([context->v41_kernel_runs[key] unsignedLongLongValue] + 1);
    }
}
uint64_t ds4_gpu_test_kernel_runs(const char *name) {
    @autoreleasepool {
        return [ds4_metal_context_current()->v41_kernel_runs[[NSString stringWithUTF8String:name]] unsignedLongLongValue];
    }
}

int ds4_metal_v41_dense_decode_allowed(void) { return !g_quality_mode && !g_ssd_streaming_mode && !ds4_gpu_tp_world_is_two(); }

ds4_metal_v41_topk_state *ds4_metal_v41_topk_state_get(void) { return &ds4_metal_context_current()->v41_topk_state; }
