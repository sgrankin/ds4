/* Driver/ARC leaves only; native chooses stages, policy, packets and grids. */
#include "attention_decode.h"
static ds4_metal_attention_view ds4_metal_attention_retain_view(id<MTLBuffer> buffer,
        uint64_t offset, uint64_t bytes) {
    return (ds4_metal_attention_view){
        .buffer = buffer ? (ds4_metal_model_buffer *)(__bridge_retained void *)buffer : NULL,
        .offset = offset, .bytes = bytes,
    };
}
ds4_metal_attention_view ds4_metal_attention_tensor_view(const ds4_gpu_tensor *tensor) {
    @autoreleasepool {
        return ds4_metal_attention_retain_view(ds4_gpu_tensor_buffer(tensor),
            ds4_gpu_tensor_offset(tensor), ds4_gpu_tensor_bytes(tensor));
    }
}
void *ds4_metal_attention_contents(ds4_metal_attention_view view) {
    @autoreleasepool { return [(__bridge id<MTLBuffer>)view.buffer contents]; }
}
ds4_metal_attention_view ds4_metal_attention_scratch(uint32_t kind, uint64_t bytes,
        const char *label) {
    @autoreleasepool {
        id<MTLBuffer> result = nil;
        if (kind == DS4_ATTN_TRANSIENT_MASK) {
            result = ds4_gpu_new_transient_buffer(bytes, label);
        } else if (kind == DS4_ATTN_ZERO_MASK) {
            if (ds4_gpu_ensure_zero_attention_mask(bytes)) result = g_flash_attn_zero_mask_buffer;
        } else {
            id<MTLBuffer> __strong *buffer = NULL;
            NSUInteger *capacity = NULL;
            switch (kind) {
                case DS4_ATTN_MASK: buffer = &g_flash_attn_mask_buffer; capacity = &g_flash_attn_mask_bytes; break;
                case DS4_ATTN_KV: buffer = &g_flash_attn_kv_buffer; capacity = &g_flash_attn_kv_bytes; break;
                case DS4_ATTN_PAD: buffer = &g_flash_attn_pad_buffer; capacity = &g_flash_attn_pad_bytes; break;
                case DS4_ATTN_TMP: buffer = &g_flash_attn_tmp_buffer; capacity = &g_flash_attn_tmp_bytes; break;
                default: return (ds4_metal_attention_view){0};
            }
            if (ds4_gpu_ensure_scratch_buffer(buffer, capacity, bytes, label)) result = *buffer;
        }
        return ds4_metal_attention_retain_view(result, 0, result.length);
    }
}
ds4_metal_attention_facts ds4_metal_attention_get_facts(void) {
    @autoreleasepool {
        return (ds4_metal_attention_facts){
            .flags = (ds4_gpu_device_is_pre_m5_apple_silicon() ? DS4_ATTN_PRE_M5 : 0u) |
                (ds4_gpu_device_is_m5_apple_silicon() ? DS4_ATTN_M5 : 0u) |
                (ds4_gpu_device_name_contains("M3") ? DS4_ATTN_M3 : 0u) |
                (g_quality_mode ? DS4_ATTN_QUALITY : 0u) |
                (g_ssd_streaming_mode ? DS4_ATTN_SSD : 0u) |
                (g_tp_split_world == 1 ? DS4_ATTN_TP_SINGLE : 0u) |
                (g_flash_kv_stage_f16_pipeline ? DS4_ATTN_STAGE_AVAILABLE : 0u),
            .max_shared_bytes = g_device.maxThreadgroupMemoryLength,
            .stage_threads = g_flash_kv_stage_f16_pipeline.maxTotalThreadsPerThreadgroup,
            .fill_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup,
            .copy_threads = {g_cpy_contig_f32_f16_pipeline.maxTotalThreadsPerThreadgroup,
                g_cpy_f32_f16_pipeline.maxTotalThreadsPerThreadgroup,
                g_cpy_contig_f16_f16_pipeline.maxTotalThreadsPerThreadgroup},
        };
    }
}
ds4_metal_attention_pipeline ds4_metal_attention_get_pipeline(uint32_t kind,
        uint32_t width, uint32_t nsg, uint32_t nwg, bool pad, bool shared_pad) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
            case DS4_ATTN_PAD_PSO: pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, 32); break;
            case DS4_ATTN_VEC_PSO:
            case DS4_ATTN_PACKED_PSO:
                pipeline = ds4_gpu_get_flash_attn_vec_pipeline(kind == DS4_ATTN_PACKED_PSO ?
                    "kernel_dsv4_flash_attn_vec_packed32_reduce_rope_f16_dk512_dv512" :
                    "kernel_flash_attn_ext_vec_f16_dk512_dv512",
                    true, true, false, false, pad, shared_pad, width, width, nsg, nwg); break;
            case DS4_ATTN_REDUCE_PSO: pipeline = ds4_gpu_get_flash_attn_reduce_pipeline(width, nwg); break;
            case DS4_ATTN_ROPE_PSO: pipeline = ds4_gpu_get_flash_attn_reduce_rope_pipeline(width, nwg); break;
            default: break;
        }
        return (ds4_metal_attention_pipeline){
            .handle = (const ds4_metal_hc_pipeline *)(__bridge const void *)pipeline,
            .threads = pipeline.maxTotalThreadsPerThreadgroup,
            .simd_width = pipeline.threadExecutionWidth,
        };
    }
}
int ds4_metal_attention_break_encoder(void *command) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb) return 0;
        ds4_gpu_end_compute_encoder(cb, ds4_gpu_compute_encoder(cb));
        return 1;
    }
}
int ds4_metal_attention_copy(void *command, uint32_t kind,
        const ds4_metal_copy_args *args, uint32_t count,
        ds4_metal_attention_view source, ds4_metal_attention_view target,
        uint64_t groups, uint64_t threads) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.buffer;
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)target.buffer;
        if (!cb || !src || !dst) return 0;
        if (kind == DS4_ATTN_COPY_F16_BLIT) {
            if (g_batch_cb && cb == g_batch_cb) ds4_gpu_close_batch_encoder();
            id<MTLBlitCommandEncoder> blit = ds4_gpu_blit_encoder(cb, "f16_copy", 1u);
            if (!blit) return 0;
            [blit copyFromBuffer:src sourceOffset:source.offset toBuffer:dst
                destinationOffset:target.offset size:(NSUInteger)count*sizeof(uint16_t)];
            [blit endEncoding];
            return 1;
        }
        id<MTLComputePipelineState> pipeline = kind == DS4_ATTN_COPY_F32_CONTIG ?
            g_cpy_contig_f32_f16_pipeline : kind == DS4_ATTN_COPY_F32_GENERIC ?
            g_cpy_f32_f16_pipeline : g_cpy_contig_f16_f16_pipeline;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        if (kind == DS4_ATTN_COPY_F32_GENERIC) [enc setBytes:args length:sizeof(*args) atIndex:0];
        else [enc setBytes:&count length:sizeof(count) atIndex:0];
        [enc setBuffer:src offset:source.offset atIndex:1];
        [enc setBuffer:dst offset:target.offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_fill(void *command, const ds4_metal_attention_unary_args *args,
        ds4_metal_attention_view mask, uint64_t groups, uint64_t threads) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mask.buffer;
        if (!cb || !buffer) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_unary_fill_f16_pipeline];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:buffer offset:mask.offset atIndex:1];
        [enc setBuffer:buffer offset:mask.offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_stage(void *command, const ds4_metal_attention_stage_args *args,
        ds4_metal_attention_view raw, ds4_metal_attention_view comp,
        ds4_metal_attention_view dst, ds4_metal_attention_view mask,
        ds4_metal_attention_view pad, uint64_t groups, uint64_t threads) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_flash_kv_stage_f16_pipeline];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {raw,comp,dst,mask,pad};
        for (NSUInteger i=0;i<5;i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_pad(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_pad_args *args, ds4_metal_attention_view kv,
        ds4_metal_attention_view mask, ds4_metal_attention_view pad) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)kv.buffer offset:kv.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)kv.buffer offset:kv.offset atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)mask.buffer offset:mask.offset atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)pad.buffer offset:pad.offset atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_vec(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_vec_args *args, ds4_metal_attention_view q,
        ds4_metal_attention_view kv, ds4_metal_attention_view mask,
        ds4_metal_attention_view sinks, ds4_metal_attention_view pad,
        ds4_metal_attention_view out, const ds4_metal_attention_rope_args *rope,
        uint64_t gx,uint64_t gy,uint64_t gz,uint64_t tx,uint64_t ty,uint64_t shared) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {q,kv,kv,mask,sinks,pad,out};
        for (NSUInteger i=0;i<7;i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        if (rope) [enc setBytes:rope length:sizeof(*rope) atIndex:8];
        [enc setThreadgroupMemoryLength:shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(gx,gy,gz) threadsPerThreadgroup:MTLSizeMake(tx,ty,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_reduce(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_reduce_args *args, ds4_metal_attention_view tmp,
        ds4_metal_attention_view out, const ds4_metal_attention_rope_args *rope,
        uint64_t rows, uint64_t threads) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)tmp.buffer offset:tmp.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out.buffer offset:out.offset atIndex:2];
        if (rope) [enc setBytes:rope length:sizeof(*rope) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
