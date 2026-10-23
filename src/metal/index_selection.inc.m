/* Only cached PSO/resource access and fixed Metal encoder messages live here.
 * Native policy owns buffer bounds, causal geometry and scratch ping-pong. */
ds4_metal_index_pipeline ds4_metal_index_get_pipeline(uint32_t kind) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
            case DS4_METAL_INDEX_SCORE: pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_score_one_pipeline, "kernel_glm_indexer_score_one"); break;
            case DS4_METAL_INDEX_DIRECT: pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_score_one_direct_pipeline, "kernel_glm_indexer_score_one_direct"); break;
            case DS4_METAL_INDEX_BATCH: pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_scores_batch_pipeline, "kernel_glm_indexer_scores_batch"); break;
            case DS4_METAL_INDEX_TILED: pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_scores_tiled_pipeline, "kernel_glm_indexer_scores_tiled"); break;
            case DS4_METAL_INDEX_TILED_F32: pipeline = ds4_gpu_hot_pipeline(g_glm_indexer_scores_tiled_f32_pipeline, "kernel_glm_indexer_scores_tiled_f32"); break;
            case DS4_METAL_INDEX_PACK: pipeline = ds4_gpu_get_pipeline("kernel_dsv41_indexer_pack"); break;
            case DS4_METAL_INDEX_PACKED_SCORE: pipeline = ds4_gpu_get_pipeline("kernel_dsv41_indexer_scores_packed"); break;
            case DS4_METAL_INDEX_SORT: pipeline = g_argsort_f32_i32_desc_pipeline; break;
            case DS4_METAL_INDEX_SORT_CAUSAL: pipeline = ds4_gpu_get_pipeline("kernel_argsort_f32_i32_desc_causal"); break;
            case DS4_METAL_INDEX_SORT_SHUFFLE: pipeline = ds4_gpu_get_pipeline("kernel_argsort_f32_i32_desc_causal_shuffle"); break;
            case DS4_METAL_INDEX_MERGE: pipeline = g_argsort_merge_f32_i32_desc_pipeline; break;
            case DS4_METAL_INDEX_MERGE_CAUSAL: pipeline = ds4_gpu_get_pipeline("kernel_argsort_merge_f32_i32_desc_causal"); break;
            case DS4_METAL_INDEX_MERGE_PREFIX: pipeline = ds4_gpu_get_pipeline("kernel_argsort_merge_f32_i32_desc_causal_prefix"); break;
            case DS4_METAL_INDEX_MASK_FILL: pipeline = g_dsv4_topk_mask_pipeline; break;
            case DS4_METAL_INDEX_MASK_SCATTER: pipeline = g_dsv4_topk_mask_scatter_pipeline; break;
            case DS4_METAL_INDEX_ROW_SORT: pipeline = ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline, "kernel_dsv4_sort_i32_rows_asc"); break;
            default: break;
        }
        return (ds4_metal_index_pipeline){
            .handle = (const struct ds4_metal_hc_pipeline *)(__bridge const void *)pipeline,
            .max_threads = pipeline.maxTotalThreadsPerThreadgroup,
        };
    }
}
ds4_metal_index_facts ds4_metal_index_get_facts(void) {
    @autoreleasepool {
        return (ds4_metal_index_facts){g_quality_mode,
            g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup,
            g_device.maxThreadgroupMemoryLength};
    }
}
int ds4_metal_index_reserve_scratch(uint64_t bytes) {
    @autoreleasepool {
        return ds4_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
            &g_indexer_topk_bytes, (NSUInteger)bytes, "ds4_indexer_topk");
    }
}
int ds4_metal_index_score_encode(void *command, const ds4_metal_index_score_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline.handle;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        if (launch->batch) [enc setBytes:&launch->args.batch length:sizeof(launch->args.batch) atIndex:0];
        else [enc setBytes:&launch->args.one length:sizeof(launch->args.one) atIndex:0];
        const ds4_gpu_tensor *tensors[] = {launch->q, launch->weights, launch->keys, launch->scores};
        for (NSUInteger i = 0; i < 4; i++)
            [enc setBuffer:ds4_gpu_tensor_buffer(tensors[i]) offset:ds4_gpu_tensor_offset(tensors[i]) atIndex:i + 1];
        [enc setThreadgroupMemoryLength:(NSUInteger)launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(launch->threads_x, launch->threads_y, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_index_topk_encode(void *command, const ds4_metal_index_topk_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline.handle;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        if (launch->merge) [enc setBytes:&launch->args.merge length:sizeof(launch->args.merge) atIndex:0];
        else [enc setBytes:&launch->args.sort length:sizeof(launch->args.sort) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(launch->scores) offset:ds4_gpu_tensor_offset(launch->scores) atIndex:1];
        id<MTLBuffer> destination = launch->to_selected ? ds4_gpu_tensor_buffer(launch->selected) : g_indexer_topk_buffer;
        NSUInteger target = launch->to_selected ? ds4_gpu_tensor_offset(launch->selected) : (NSUInteger)launch->target_offset;
        if (launch->merge) {
            [enc setBuffer:g_indexer_topk_buffer offset:(NSUInteger)launch->source_offset atIndex:2];
            [enc setBuffer:destination offset:target atIndex:3];
        } else {
            [enc setBuffer:destination offset:target atIndex:2];
            [enc setThreadgroupMemoryLength:(NSUInteger)launch->shared_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_index_rows_encode(void *command, const ds4_metal_index_rows_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline.handle;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&launch->args length:sizeof(launch->args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(launch->source) offset:ds4_gpu_tensor_offset(launch->source) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(launch->destination) offset:ds4_gpu_tensor_offset(launch->destination) atIndex:2];
        if (launch->shared_bytes) [enc setThreadgroupMemoryLength:(NSUInteger)launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_index_pack_encode(void *command, const ds4_metal_index_pack_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline.handle;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:launch->args length:sizeof(launch->args) atIndex:0];
        if (launch->scores) {
            [enc setBytes:launch->range length:sizeof(launch->range) atIndex:1];
            const ds4_gpu_tensor *inputs[] = {launch->q, launch->weights, launch->keys, launch->output, launch->packed_source};
            for (NSUInteger i = 0; i < 5; i++)
                [enc setBuffer:ds4_gpu_tensor_buffer(inputs[i]) offset:ds4_gpu_tensor_offset(inputs[i]) atIndex:i + 2];
            const NSUInteger base = ds4_gpu_tensor_offset(launch->packed_source);
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->packed_source) offset:base + launch->flags_offset atIndex:7];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->packed_source) offset:base + launch->keys_offset atIndex:8];
        } else {
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->q) offset:ds4_gpu_tensor_offset(launch->q) atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->keys) offset:ds4_gpu_tensor_offset(launch->keys) atIndex:2];
            const NSUInteger base = ds4_gpu_tensor_offset(launch->output);
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->output) offset:base atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->output) offset:base + launch->flags_offset atIndex:4];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->output) offset:base + launch->keys_offset atIndex:5];
        }
        [enc setThreadgroupMemoryLength:(NSUInteger)launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
