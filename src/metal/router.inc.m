/* Router ARC facts/scratch and fixed shader bindings. No original selection
 * controller or primitive encoder is called through this boundary. */
#include "router.h"
uint32_t ds4_metal_router_facts(void) {
    return (g_quality_mode ? DS4_ROUTER_QUALITY : 0) |
        (ds4_gpu_device_is_pre_m5_apple_silicon() ? DS4_ROUTER_PRE_M5 : 0) |
        (ds4_gpu_device_name_contains("M5") ? DS4_ROUTER_M5 : 0) |
        (ds4_gpu_device_name_contains("M3") ? DS4_ROUTER_M3 : 0) |
        (g_ssd_streaming_mode ? DS4_ROUTER_SSD : 0);
}
ds4_metal_router_pipeline ds4_metal_router_pipeline_get(uint32_t kind, bool hot) {
    @autoreleasepool {
        id<MTLComputePipelineState> p = nil;
        const char *name = NULL;
#define PIPE(K, P, N) case K: p=P; name=N; break
        switch (kind) {
        PIPE(DS4_ROUTER_SOFTPLUS, g_unary_softplus_pipeline, NULL);
        PIPE(DS4_ROUTER_SQRT, g_unary_sqrt_pipeline, NULL);
        PIPE(DS4_ROUTER_TRANSFORM, g_dsv4_softplus_sqrt_pipeline, "kernel_dsv4_softplus_sqrt_f32_4");
        PIPE(DS4_ROUTER_FINALIZE, g_dsv4_router_finalize_one_pipeline, "kernel_dsv4_router_finalize_one");
        PIPE(DS4_ROUTER_SIMD, g_dsv4_router_finalize_one_simd_pipeline, "kernel_dsv4_router_finalize_one_simd");
        PIPE(DS4_ROUTER_SIMD_WEIGHTS, g_dsv4_router_finalize_weights_one_simd_pipeline, "kernel_dsv4_router_finalize_weights_one_simd");
        PIPE(DS4_ROUTER_TRANSFORM_WEIGHTS, g_dsv4_router_transform_finalize_weights_one_simd_pipeline, "kernel_dsv4_router_transform_finalize_weights_one_simd");
        PIPE(DS4_ROUTER_WEIGHTS_ONE, g_dsv4_router_weights_one_pipeline, "kernel_dsv4_router_weights_one");
        PIPE(DS4_ROUTER_WEIGHTS_BATCH, g_dsv4_router_weights_batch_pipeline, NULL);
        PIPE(DS4_ROUTER_VISUAL_PSO, g_dsv4_router_select_visual_batch_pipeline, NULL);
        PIPE(DS4_ROUTER_ADD, g_add_pipeline, NULL);
        PIPE(DS4_ROUTER_DIVIDE, g_bin_div_row_pipeline, NULL);
        PIPE(DS4_ROUTER_SCALE_PSO, g_bin_mul_scalar_pipeline, NULL);
        PIPE(DS4_ROUTER_CLAMP, g_unary_clamp_pipeline, NULL);
        PIPE(DS4_ROUTER_SUM_PSO, g_sum_rows_f32_f32_pipeline, NULL);
        PIPE(DS4_ROUTER_GATHER_I32, g_get_rows_i32_pipeline, NULL);
        PIPE(DS4_ROUTER_GATHER_F32, g_get_rows_f32_pipeline, NULL);
        default: return (ds4_metal_router_pipeline){0};
        }
#undef PIPE
        if (hot) p = ds4_gpu_hot_pipeline(p, name);
        return (ds4_metal_router_pipeline){ (const ds4_metal_hc_pipeline *)(__bridge const void *)p,
            p.maxTotalThreadsPerThreadgroup, p.threadExecutionWidth };
    }
}
ds4_gpu_tensor *ds4_metal_router_selection_scratch(uint64_t bytes) {
    @autoreleasepool {
        if (!ds4_gpu_ensure_scratch_buffer(&g_router_selection_buffer,
            &g_router_selection_bytes, bytes, "ds4_router_selection")) return NULL;
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = g_router_selection_buffer; view.offset = 0;
        view.bytes = bytes; view.owner = 0;
        return (ds4_gpu_tensor *)(__bridge_retained void *)view;
    }
}
ds4_metal_attention_view ds4_metal_router_sum_scratch(uint64_t bytes) {
    @autoreleasepool {
        if (!ds4_gpu_ensure_scratch_buffer(&g_router_weight_sum_buffer,
            &g_router_weight_sum_bytes, bytes, "ds4_router_weight_sum")) return (ds4_metal_attention_view){0};
        return (ds4_metal_attention_view){ (ds4_metal_model_buffer *)(__bridge_retained void *)g_router_weight_sum_buffer,
            0, (uint64_t)g_router_weight_sum_buffer.length };
    }
}
int ds4_metal_router_encode(void *command, const ds4_metal_router_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline.handle;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        const float zero_f = 0;
        const int32_t zero_i = 0;
#define B(I, V) [enc setBuffer:(__bridge id<MTLBuffer>)p->V.buffer offset:p->V.offset atIndex:I]
#define PACK(V) [enc setBytes:&p->args.V length:sizeof(p->args.V) atIndex:0]
        switch (p->kind) {
        case DS4_ROUTER_UNARY: PACK(unary); B(1,source); B(2,target); break;
        case DS4_ROUTER_BINARY: PACK(binary); B(1,source); B(2,other); B(3,target); break;
        case DS4_ROUTER_SCALE:
            PACK(binary); B(1,source); [enc setBytes:&p->scale length:sizeof(p->scale) atIndex:2]; B(3,target); break;
        case DS4_ROUTER_GATHER:
            PACK(gather); B(1,source);
            if (p->inline_token) [enc setBytes:&p->token length:sizeof(p->token) atIndex:2];
            else { B(2,other); }
            B(3,target); break;
        case DS4_ROUTER_SUM: PACK(sum); B(1,source); B(2,target); break;
        case DS4_ROUTER_SELECT_ONE:
            PACK(one); B(1,probs);
            if (p->args.one.has_bias) { B(2,bias); }
            else [enc setBytes:&zero_f length:sizeof(zero_f) atIndex:2];
            if (p->args.one.hash_mode) { B(3,hash); }
            else [enc setBytes:&zero_i length:sizeof(zero_i) atIndex:3];
            if (p->args.one.use_token_buffer) { B(4,tokens); }
            else [enc setBytes:&zero_i length:sizeof(zero_i) atIndex:4];
            B(5,selected); if (p->fused_weights) { B(6,weights); } break;
        case DS4_ROUTER_SELECT_TRANSFORM:
            PACK(one); B(1,source); B(2,probs);
            if (p->args.one.has_bias) { B(3,bias); }
            else [enc setBytes:&zero_f length:sizeof(zero_f) atIndex:3];
            [enc setBytes:&zero_i length:sizeof(zero_i) atIndex:4];
            if (p->args.one.use_token_buffer) { B(5,tokens); }
            else [enc setBytes:&zero_i length:sizeof(zero_i) atIndex:5];
            B(6,selected); B(7,weights); break;
        case DS4_ROUTER_SELECT_VISUAL:
            PACK(visual); B(1,probs);
            if (p->args.visual.has_bias) { B(2,bias); }
            else [enc setBytes:&zero_f length:sizeof(zero_f) atIndex:2];
            B(3,visual);
            if (p->args.visual.hash_mode) { B(4,hash); }
            else [enc setBytes:&zero_i length:sizeof(zero_i) atIndex:4];
            B(5,tokens); B(6,selected); break;
        case DS4_ROUTER_WEIGHT_ONE: B(0,probs); B(1,selected); B(2,weights); break;
        case DS4_ROUTER_WEIGHT_BATCH:
            [enc setBytes:&p->scale length:sizeof(p->scale) atIndex:0];
            B(1,probs); B(2,selected); B(3,weights); break;
        default: return 0;
        }
#undef PACK
#undef B
        if (p->shared) [enc setThreadgroupMemoryLength:p->shared atIndex:0];
        if (p->direct_threads) [enc dispatchThreads:MTLSizeMake(p->groups_x,p->groups_y,1)
            threadsPerThreadgroup:MTLSizeMake(p->threads,1,1)];
        else [enc dispatchThreadgroups:MTLSizeMake(p->groups_x,p->groups_y,1)
            threadsPerThreadgroup:MTLSizeMake(p->threads,1,1)];
        ds4_gpu_end_compute_encoder(cb,enc);
        return 1;
    }
}
