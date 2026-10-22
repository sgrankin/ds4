/* Fixed ARC/PSO/binding driver. Native owns positions, kernel policy, packets,
 * validation and all grid/thread decisions. No old RoPE encoder is called. */
uint32_t ds4_metal_rope_facts(void) {
    @autoreleasepool {
        return (ds4_gpu_device_name_contains("M3") ? DS4_ROPE_M3 : 0) |
               (ds4_gpu_device_name_contains("M5") ? DS4_ROPE_M5 : 0) |
               (g_rope_tail_inplace_pair_pipeline ? DS4_ROPE_PAIR_AVAILABLE : 0) |
               (g_rope_tail_inplace_pair_shared4_pipeline ? DS4_ROPE_SHARED_AVAILABLE : 0) |
               (g_rope_tail_inplace_pair_affine_pipeline ? DS4_ROPE_AFFINE_AVAILABLE : 0);
    }
}
ds4_metal_rope_pipeline ds4_metal_rope_pipeline_get(uint32_t kind) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
        case DS4_ROPE_REFERENCE:
            pipeline = g_rope_tail_batch_pipeline;
            break;
        case DS4_ROPE_PAIR:
            pipeline = g_rope_tail_inplace_pair_pipeline;
            break;
        case DS4_ROPE_SHARED:
            pipeline = g_rope_tail_inplace_pair_shared4_pipeline;
            break;
        case DS4_ROPE_AFFINE:
            pipeline = g_rope_tail_inplace_pair_affine_pipeline;
            break;
        case DS4_ROPE_PLAIN_HEAD:
            pipeline = g_rms_norm_plain_pipeline;
            break;
        case DS4_ROPE_FUSED_HEAD:
            pipeline = g_use_dsv4_head_rms_norm_rope_tail_pipeline
                           ? g_dsv4_head_rms_norm_rope_tail_pipeline
                           : ds4_gpu_get_pipeline("kernel_dsv4_head_rms_norm_rope_tail_f32");
            break;
        case DS4_ROPE_GLM:
            pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_rope_tail_pipeline,
                                            "kernel_glm_indexer_rope_tail_f32");
            break;
        case DS4_ROPE_V41:
            pipeline = ds4_gpu_get_pipeline("kernel_dsv41_rope");
            break;
        }
        return (ds4_metal_rope_pipeline){
            (const ds4_metal_hc_pipeline *)(__bridge void *)pipeline,
            pipeline ? (uint64_t)[pipeline maxTotalThreadsPerThreadgroup] : 1024u};
    }
}
int ds4_metal_rope_encode(void *command, const ds4_metal_rope_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)p->input.buffer;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !input || !pipeline)
            return 0;
        id<MTLBuffer> positions = nil;
        const bool generic =
            p->kind == DS4_ROPE_REFERENCE || p->kind == DS4_ROPE_PAIR || p->kind == DS4_ROPE_SHARED;
        if (generic && p->position_buffer) {
            /* Metal setBytes is for small constants. Long-prefill positions
             * retain the existing transient-buffer lifetime through the CB. */
            positions =
                ds4_gpu_new_transient_buffer((NSUInteger)p->position_bytes, "ds4_rope_positions");
            if (!positions)
                return 0;
            memcpy([positions contents], p->positions, (size_t)p->position_bytes);
        }
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc)
            return 0;
        [enc setComputePipelineState:pipeline];
        switch (p->kind) {
        case DS4_ROPE_REFERENCE:
        case DS4_ROPE_PAIR:
        case DS4_ROPE_SHARED:
            [enc setBytes:&p->args.tail length:sizeof(p->args.tail) atIndex:0];
            [enc setBuffer:input offset:p->input.offset atIndex:1];
            if (positions)
                [enc setBuffer:positions offset:0 atIndex:2];
            else
                [enc setBytes:p->positions length:(NSUInteger)p->position_bytes atIndex:2];
            [enc setBuffer:input offset:p->input.offset atIndex:3];
            [enc setBuffer:input offset:p->input.offset atIndex:4];
            break;
        case DS4_ROPE_AFFINE:
            [enc setBytes:&p->args.affine length:sizeof(p->args.affine) atIndex:0];
            [enc setBuffer:input offset:p->input.offset atIndex:1];
            [enc setBuffer:input offset:p->input.offset atIndex:4];
            break;
        case DS4_ROPE_PLAIN_HEAD:
            [enc setBytes:&p->args.norm length:sizeof(p->args.norm) atIndex:0];
            for (NSUInteger slot = 1; slot <= 4; slot++)
                [enc setBuffer:input offset:p->input.offset atIndex:slot];
            break;
        case DS4_ROPE_FUSED_HEAD:
            [enc setBytes:&p->args.head length:sizeof(p->args.head) atIndex:0];
            [enc setBuffer:input offset:p->input.offset atIndex:1];
            break;
        case DS4_ROPE_GLM:
            [enc setBytes:&p->args.glm length:sizeof(p->args.glm) atIndex:0];
            [enc setBuffer:input offset:p->input.offset atIndex:1];
            break;
        case DS4_ROPE_V41:
            [enc setBytes:&p->args.v41 length:sizeof(p->args.v41) atIndex:0];
            [enc setBuffer:input offset:p->input.offset atIndex:1];
            break;
        default:
            return 0;
        }
        if (p->shared_bytes)
            [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
