/* Included after existing Metal helpers. No dispatch policy lives here. */
#include "dense_matmul.h"
uint32_t ds4_metal_dense_facts(bool paired) {
    @autoreleasepool {
        return (ds4_gpu_mpp_available() ? DS4_DENSE_MPP : 0u) |
            (g_quality_mode ? DS4_DENSE_QUALITY : 0u) |
            (ds4_gpu_device_is_m5_apple_silicon() ? DS4_DENSE_M5 : 0u) |
            (g_qwen4_verify_rows_exact ? DS4_DENSE_VERIFY_EXACT : 0u) |
            (ds4_gpu_tp_world_is_two() ? DS4_DENSE_TP_TWO : 0u) |
            (ds4_gpu_device_name_contains("M3 Ultra") ? DS4_DENSE_M3_ULTRA : 0u) |
            (paired && ds4_gpu_use_compressor_pair_nr4() ? DS4_DENSE_PAIR_NR4 : 0u);
    }
}
const ds4_metal_hc_pipeline *ds4_metal_dense_pipeline(uint32_t kind,
        const char *name, int16_t nsg, int16_t nxpsg, bool bc_in, bool bc_out) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
            case DS4_DENSE_MV: pipeline = ds4_gpu_get_mul_mv_pipeline(name, nsg); break;
            case DS4_DENSE_EXT: pipeline = ds4_gpu_get_mul_mv_ext_pipeline(name, nsg, nxpsg); break;
            case DS4_DENSE_MM: pipeline = ds4_gpu_get_mul_mm_pipeline(name, bc_in, bc_out); break;
            default: return NULL;
        }
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)pipeline;
    }
}
int ds4_metal_dense_encode(void *command, const ds4_metal_dense_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        id<MTLBuffer> weights = (__bridge id<MTLBuffer>)p->weights;
        id<MTLBuffer> x = ds4_gpu_tensor_buffer(p->x), out = ds4_gpu_tensor_buffer(p->out);
        if (!cb || !pipeline || !weights || !x || !out) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        if (p->kind == DS4_DENSE_MV) [enc setBytes:&p->args.mv length:sizeof(p->args.mv) atIndex:0];
        else if (p->kind == DS4_DENSE_EXT) [enc setBytes:&p->args.ext length:sizeof(p->args.ext) atIndex:0];
        else [enc setBytes:&p->args.mm length:sizeof(p->args.mm) atIndex:0];
        [enc setBuffer:weights offset:p->weight_offset atIndex:1];
        if (p->shared_bytes) [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        for (uint64_t row = 0; row < p->repeats; ++row) {
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(p->x) + p->x_offset + row*p->x_step atIndex:2];
            [enc setBuffer:out offset:ds4_gpu_tensor_offset(p->out) + p->out_offset + row*p->out_step atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, 1)
                 threadsPerThreadgroup:MTLSizeMake(p->threads_x, p->threads_y, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_dense_pair_encode(void *command, const ds4_metal_dense_pair_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&p->args0 length:sizeof(p->args0) atIndex:0];
        const NSUInteger first = p->two_args ? 2u : 1u;
        if (p->two_args) [enc setBytes:&p->args1 length:sizeof(p->args1) atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->weights0 offset:p->weight_offset0 atIndex:first];
        [enc setBuffer:(__bridge id<MTLBuffer>)p->weights1 offset:p->weight_offset1 atIndex:first+1];
        [enc setBuffer:ds4_gpu_tensor_buffer(p->x) offset:ds4_gpu_tensor_offset(p->x) atIndex:first+2];
        [enc setBuffer:ds4_gpu_tensor_buffer(p->out0) offset:ds4_gpu_tensor_offset(p->out0) atIndex:first+3];
        [enc setBuffer:ds4_gpu_tensor_buffer(p->out1) offset:ds4_gpu_tensor_offset(p->out1) atIndex:first+4];
        if (p->shared_bytes) [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, p->simdgroups, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
ds4_metal_model_buffer *ds4_metal_dense_retain_scratch(uint64_t bytes) {
    @autoreleasepool {
        if (!ds4_gpu_ensure_scratch_buffer(&g_q8_prefill_scratch_buffer,
                &g_q8_prefill_scratch_bytes, (NSUInteger)bytes, "q8 prefill weights")) return NULL;
        return (ds4_metal_model_buffer *)(__bridge_retained void *)g_q8_prefill_scratch_buffer;
    }
}
const ds4_metal_hc_pipeline *ds4_metal_dense_unpack_pipeline(void) {
    @autoreleasepool {
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)ds4_gpu_get_pipeline("kernel_q8_prefill_unpack");
    }
}
int ds4_metal_dense_unpack(void *command, const ds4_metal_hc_pipeline *pso,
        const ds4_metal_model_buffer *weights, uint64_t offset,
        const ds4_metal_model_buffer *scratch, uint32_t blocks, uint64_t groups) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pso;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&blocks length:sizeof(blocks) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)scratch offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
void ds4_metal_dense_warn_mpp(void) { ds4_gpu_warn_mpp_fallback(); }
double ds4_metal_dense_now_ms(void) { return ds4_gpu_now_ms(); }
