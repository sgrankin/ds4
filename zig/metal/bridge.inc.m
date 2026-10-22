/* Included at the end of ds4_metal.m for the native Metal controllers.
 * ARC objects, command ownership and encoder batching remain private here. */
#include "bridge.h"

static id<MTLComputePipelineState> ds4_zig_metal_copy_pipeline(uint32_t pipeline) {
    switch (pipeline) {
        case DS4_METAL_COPY_CONTIG_F32_F16: return g_cpy_contig_f32_f16_pipeline;
        case DS4_METAL_COPY_F32_F16: return g_cpy_f32_f16_pipeline;
        default: return nil;
    }
}

int ds4_zig_metal_env_bool(const char *name) {
    return ds4_gpu_env_bool(name);
}

uint64_t ds4_zig_metal_copy_max_threads(uint32_t pipeline) {
    @autoreleasepool {
        return ds4_zig_metal_copy_pipeline(pipeline).maxTotalThreadsPerThreadgroup;
    }
}

void *ds4_zig_metal_command_acquire(int *owned) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(owned);
        return cb ? (__bridge_retained void *)cb : NULL;
    }
}

int ds4_zig_metal_copy_encode(void *command, const ds4_metal_copy_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLBuffer> src = ds4_gpu_tensor_buffer(launch->src);
        id<MTLBuffer> dst = ds4_gpu_tensor_buffer(launch->dst);
        if (!cb || !src || !dst) return 0;
        id<MTLComputePipelineState> pipeline = ds4_zig_metal_copy_pipeline(launch->pipeline);
        if (!pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:launch->args length:(NSUInteger)launch->args_bytes atIndex:0];
        [enc setBuffer:src offset:ds4_gpu_tensor_offset(launch->src) + launch->src_offset atIndex:1];
        [enc setBuffer:dst offset:ds4_gpu_tensor_offset(launch->dst) + launch->dst_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

int ds4_zig_metal_command_release(void *command, int owned, int ok, const char *label) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)command;
        return cb && ok ? ds4_gpu_finish_command_buffer(cb, owned, label) : 0;
    }
}

/* Shared environment, tensor and model-buffer boundaries for native controllers. */
int ds4_zig_metal_env_present(const char *name) {
    return getenv(name) != NULL;
}

int ds4_zig_metal_error(const char *message) {
    fprintf(stderr, "%s\n", message);
    return 0;
}

int ds4_zig_metal_tensor_has_buffer(const ds4_gpu_tensor *tensor) {
    @autoreleasepool {
        return ds4_gpu_tensor_buffer(tensor) != nil;
    }
}

uint32_t ds4_zig_metal_hc_facts(void) {
    @autoreleasepool {
        return (ds4_gpu_device_name_contains("M3") ? DS4_METAL_HC_M3 : 0u) |
            ((g_test_flags & DS4_GPU_TEST_HC_RMS_SCALE_PROJ) ? DS4_METAL_HC_TEST : 0u) |
            (g_rms_norm_scale_pipeline ? DS4_METAL_HC_SCALE_PRESENT : 0u);
    }
}

const ds4_metal_hc_pipeline *ds4_zig_metal_hc_mm_pipeline(int bc_in, int bc_out) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_mul_mm_pipeline(
                "kernel_mul_mm_f16_f32_scaled", bc_in != 0, bc_out != 0);
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)pipeline;
    }
}

const ds4_metal_hc_pipeline *ds4_zig_metal_hc_scale_pipeline(void) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
                g_rms_norm_scale_pipeline, "kernel_rms_norm_scale_f32_4");
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)pipeline;
    }
}

ds4_metal_model_buffer *ds4_metal_retain_model_range(const void *map, uint64_t size,
        uint64_t offset, uint64_t bytes, uint64_t *inner_offset) {
    @autoreleasepool {
        id<MTLBuffer> weights = ds4_gpu_wrap_model_range(map, size, offset, bytes, inner_offset);
        return weights ? (ds4_metal_model_buffer *)(__bridge_retained void *)weights : NULL;
    }
}

void ds4_metal_release_model_buffer(ds4_metal_model_buffer *weights) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)weights;
        (void)buffer;
    }
}

int ds4_zig_metal_hc_encode(void *command, const ds4_metal_hc_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline;
        id<MTLBuffer> x = ds4_gpu_tensor_buffer(launch->x);
        id<MTLBuffer> scales = ds4_gpu_tensor_buffer(launch->scales);
        if (!cb || !pipeline || !x || !scales) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:launch->args length:(NSUInteger)launch->args_bytes atIndex:0];
        if (launch->stage == DS4_METAL_HC_SCALE_STAGE) {
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:1];
            [enc setBuffer:scales offset:ds4_gpu_tensor_offset(launch->scales) atIndex:2];
        } else {
            id<MTLBuffer> weights = (__bridge id<MTLBuffer>)launch->weights;
            [enc setBuffer:weights offset:launch->weight_offset atIndex:1];
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:2];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->out)
                    offset:ds4_gpu_tensor_offset(launch->out) atIndex:3];
            [enc setBuffer:scales offset:ds4_gpu_tensor_offset(launch->scales) atIndex:4];
        }
        [enc setThreadgroupMemoryLength:launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

/* RMS resource bindings; validation, packets and shape policy are Zig. */
int ds4_zig_metal_norm_encode(void *command, const ds4_metal_norm_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb) return 0;
        id<MTLComputePipelineState> pipeline = nil;
        switch (launch->mode) {
            case DS4_METAL_NORM_PLAIN: pipeline = g_rms_norm_plain_pipeline; break;
            case DS4_METAL_NORM_WEIGHTED: pipeline = g_rms_norm_pipeline; break;
            case DS4_METAL_NORM_WEIGHTED_BF16:
                pipeline = ds4_gpu_get_pipeline("kernel_rms_norm_mul_bf16_f32_4"); break;
            case DS4_METAL_NORM_ADD: pipeline = g_add_rms_norm_pipeline; break;
            case DS4_METAL_NORM_QKV: pipeline = g_dsv4_qkv_rms_norm_pipeline; break;
            default: return 0;
        }
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc || !pipeline) return 0;
        [enc setComputePipelineState:pipeline];
        if (launch->mode == DS4_METAL_NORM_QKV)
            [enc setBytes:&launch->args.qkv length:sizeof(launch->args.qkv) atIndex:0];
        else
            [enc setBytes:&launch->args.rms length:sizeof(launch->args.rms) atIndex:0];
        id<MTLBuffer> x = ds4_gpu_tensor_buffer(launch->x);
        id<MTLBuffer> out = ds4_gpu_tensor_buffer(launch->out);
        id<MTLBuffer> weights0 = (__bridge id<MTLBuffer>)launch->weights0;
        if (launch->mode == DS4_METAL_NORM_ADD) {
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:1];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->y) offset:ds4_gpu_tensor_offset(launch->y) atIndex:2];
            [enc setBuffer:weights0 offset:launch->weight_offset0 atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->aux_out) offset:ds4_gpu_tensor_offset(launch->aux_out) atIndex:4];
            [enc setBuffer:out offset:ds4_gpu_tensor_offset(launch->out) atIndex:5];
        } else if (launch->mode == DS4_METAL_NORM_QKV) {
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:1];
            [enc setBuffer:weights0 offset:launch->weight_offset0 atIndex:2];
            [enc setBuffer:out offset:ds4_gpu_tensor_offset(launch->out) atIndex:3];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->y) offset:ds4_gpu_tensor_offset(launch->y) atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)launch->weights1 offset:launch->weight_offset1 atIndex:5];
            [enc setBuffer:ds4_gpu_tensor_buffer(launch->aux_out) offset:ds4_gpu_tensor_offset(launch->aux_out) atIndex:6];
        } else {
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:1];
            const BOOL plain = launch->mode == DS4_METAL_NORM_PLAIN;
            [enc setBuffer:plain ? x : weights0 offset:plain ? ds4_gpu_tensor_offset(launch->x) : launch->weight_offset0 atIndex:2];
            [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:3];
            [enc setBuffer:out offset:ds4_gpu_tensor_offset(launch->out) atIndex:4];
        }
        [enc setThreadgroupMemoryLength:launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->rows, launch->tasks, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

uint64_t ds4_zig_metal_tensor_offset(const ds4_gpu_tensor *tensor) {
    return ds4_gpu_tensor_offset(tensor);
}

int ds4_zig_metal_quant_encode(void *command, const ds4_metal_quant_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(launch->linear ?
            "kernel_dsv41_bf16_linear" : "kernel_dsv41_quantize");
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        if (launch->linear) [enc setBytes:&launch->count length:sizeof(launch->count) atIndex:0];
        else [enc setBytes:&launch->args length:sizeof(launch->args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(launch->x) offset:ds4_gpu_tensor_offset(launch->x) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

int ds4_zig_metal_engram_encode(void *command, const ds4_metal_engram_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv41_engram_add");
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&launch->args length:sizeof(launch->args) atIndex:0];
        const ds4_gpu_tensor *inputs[] = {launch->residual, launch->kv,
            launch->q_weight, launch->k_weight, launch->mask};
        for (NSUInteger i = 0; i < 5; ++i)
            [enc setBuffer:ds4_gpu_tensor_buffer(inputs[i]) offset:ds4_gpu_tensor_offset(inputs[i]) atIndex:i + 1];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

int ds4_zig_metal_pool_encode(void *command, const ds4_metal_pool_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv41_pool2");
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&launch->args length:sizeof(launch->args) atIndex:0];
        const ds4_gpu_tensor *inputs[] = {launch->out, launch->kv, launch->scores,
            launch->previous_kv, launch->previous_scores};
        for (NSUInteger i = 0; i < 5; ++i)
            [enc setBuffer:ds4_gpu_tensor_buffer(inputs[i]) offset:ds4_gpu_tensor_offset(inputs[i]) atIndex:i + 1];
        [enc dispatchThreads:MTLSizeMake(launch->grid_x, launch->grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(launch->threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}

/* F32 matmul resource queries/bindings. Shape/kernel policy remains native. */
uint32_t ds4_zig_metal_f32_facts(void) {
    @autoreleasepool {
        return (ds4_gpu_device_is_m5_apple_silicon() ? DS4_METAL_F32_M5 : 0u) |
            (g_quality_mode ? DS4_METAL_F32_QUALITY : 0u) |
            (g_qwen4_verify_rows_exact ? DS4_METAL_F32_VERIFY_EXACT : 0u);
    }
}
uint64_t ds4_zig_metal_env_u64(const char *name, uint64_t fallback, uint64_t min, uint64_t max) {
    return ds4_gpu_env_u64(name, fallback, min, max);
}
const ds4_metal_hc_pipeline *ds4_zig_metal_f32_pipeline(uint32_t kind,
        const char *name, int16_t nsg, int16_t nxpsg, int bc_out) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
            case DS4_METAL_F32_MV: pipeline = ds4_gpu_get_mul_mv_pipeline(name, nsg); break;
            case DS4_METAL_F32_MV_EXT: pipeline = ds4_gpu_get_mul_mv_ext_pipeline(name, nsg, nxpsg); break;
            case DS4_METAL_F32_MM:
                pipeline = ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_f32_f32_full", false, bc_out != 0); break;
            default: return NULL;
        }
        return (const ds4_metal_hc_pipeline *)(__bridge const void *)pipeline;
    }
}
int ds4_zig_metal_f32_encode(void *command, const ds4_metal_f32_launch *launch) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)launch->pipeline;
        id<MTLBuffer> weights = (__bridge id<MTLBuffer>)launch->weights;
        id<MTLBuffer> x = ds4_gpu_tensor_buffer(launch->x);
        id<MTLBuffer> out = ds4_gpu_tensor_buffer(launch->out);
        if (!cb || !pipeline || !weights || !x || !out) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        /* Match the old MM branch's explicit nil-encoder rejection. */
        if (!enc && launch->kind == DS4_METAL_F32_MM) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:launch->args length:(NSUInteger)launch->args_bytes atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)launch->weight_offset atIndex:1];
        [enc setBuffer:x offset:ds4_gpu_tensor_offset(launch->x) atIndex:2];
        [enc setBuffer:out offset:ds4_gpu_tensor_offset(launch->out) atIndex:3];
        if (launch->shared_bytes)
            [enc setThreadgroupMemoryLength:launch->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(launch->groups_x, launch->groups_y, 1)
             threadsPerThreadgroup:MTLSizeMake(launch->threads_x, launch->threads_y, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
