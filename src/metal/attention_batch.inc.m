/* Fixed ARC/Metal leaves. Masks, geometry and dispatch policy live in Zig. */
#include "attention_batch.h"
ds4_metal_attention_cache ds4_metal_attention_cache_acquire(
        const ds4_metal_attention_cache_key *key) {
    @autoreleasepool {
        bool created = false;
        ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
            ds4_gpu_get_zero_prefix_prefill_mask_cache(key->kind, key->n_tokens,
                key->n_comp, key->n_keys, key->window, key->ratio,
                key->nqptg, key->ncpsg, key->has_kvpad, key->bc_mask,
                key->mask_bytes, key->blk_bytes, &created);
        if (!entry) return (ds4_metal_attention_cache){0};
        return (ds4_metal_attention_cache){
            .mask = ds4_metal_attention_retain_view(entry->mask, 0, entry->mask_bytes),
            .block = ds4_metal_attention_retain_view(entry->blk, 0, entry->blk_bytes),
            .entry = entry, .created = created, .block_ready = entry->blk_ready,
        };
    }
}
void ds4_metal_attention_cache_publish(void *pointer, bool mask_valid, bool block_ready) {
    ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry = pointer;
    if (!entry) return;
    if (mask_valid) entry->valid = true;
    if (block_ready) entry->blk_ready = true;
}
ds4_metal_attention_view ds4_metal_attention_batch_scratch(uint32_t kind,
        uint64_t bytes, const char *label) {
    @autoreleasepool {
        id<MTLBuffer> __strong *buffer = kind == DS4_ATTN_BATCH_BLOCK ?
            &g_flash_attn_blk_buffer : &g_indexed_topk_buffer;
        NSUInteger *capacity = kind == DS4_ATTN_BATCH_BLOCK ?
            &g_flash_attn_blk_bytes : &g_indexed_topk_bytes;
        if (!ds4_gpu_ensure_scratch_buffer(buffer, capacity, bytes, label))
            return (ds4_metal_attention_view){0};
        return ds4_metal_attention_retain_view(*buffer, 0, [*buffer length]);
    }
}
ds4_metal_attention_pipeline ds4_metal_attention_batch_pipeline(uint32_t kind,
        uint32_t nsg, uint32_t nwg, bool pad, bool bc, bool sinks) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = nil;
        switch (kind) {
            case DS4_ATTN_BATCH_PAD: pso = ds4_gpu_get_flash_attn_pad_pipeline(true, 64); break;
            case DS4_ATTN_BATCH_BLOCK_MAP: pso = ds4_gpu_get_flash_attn_blk_pipeline(8, 64); break;
            case DS4_ATTN_BATCH_VECTOR:
                pso = ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                    true, sinks, false, false, pad, false, 512, 512, nsg, nwg); break;
            case DS4_ATTN_BATCH_NONVECTOR:
                pso = ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                    true, sinks, false, false, pad, bc, 512, 512, nsg); break;
            case DS4_ATTN_BATCH_CAUSAL: pso = ds4_gpu_get_pipeline("kernel_dsv4_causal_comp_mask"); break;
            case DS4_ATTN_BATCH_SORT: pso = ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline, "kernel_dsv4_sort_i32_rows_asc"); break;
            case DS4_ATTN_BATCH_INDEX_SPLIT: pso = ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_split_pipeline, "kernel_dsv4_indexed_mixed_attention_heads8_split"); break;
            case DS4_ATTN_BATCH_INDEX_REDUCE: pso = ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_split_reduce_pipeline, "kernel_dsv4_indexed_mixed_attention_heads8_split_reduce"); break;
            case DS4_ATTN_BATCH_INDEX_DECODE: pso = ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_rb16_pipeline, "kernel_dsv4_indexed_mixed_attention_heads8_rb16"); break;
            case DS4_ATTN_BATCH_INDEX_DUAL: pso = ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads16_dual_pipeline, "kernel_dsv4_indexed_mixed_attention_heads16_dual"); break;
            case DS4_ATTN_BATCH_INDEX: pso = ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_pipeline, "kernel_dsv4_indexed_mixed_attention_heads8"); break;
        }
        return (ds4_metal_attention_pipeline){
            .handle = (const ds4_metal_hc_pipeline *)(__bridge const void *)pso,
            .threads = pso.maxTotalThreadsPerThreadgroup, .simd_width = pso.threadExecutionWidth,
        };
    }
}
bool ds4_metal_attention_batch_mpp(void) { return ds4_gpu_mpp_available(); }
bool ds4_metal_attention_profile_enabled(void) {
    return getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
}
static void ds4_metal_attention_replace_command(void **token, id<MTLCommandBuffer> cb) {
    id<MTLCommandBuffer> previous = (__bridge_transfer id<MTLCommandBuffer>)*token;
    (void)previous;
    *token = (__bridge_retained void *)cb;
}
int ds4_metal_attention_profile_begin(void **token, double *started) {
    @autoreleasepool {
        if (!token || !*token || !started ||
            !ds4_gpu_end_commands() || !ds4_gpu_begin_commands()) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) return 0;
        ds4_metal_attention_replace_command(token, cb);
        *started = ds4_gpu_now_ms();
        return 1;
    }
}
int ds4_metal_attention_profile_stage(void **token, const char *mode,
        const char *stage, uint32_t rows, uint32_t comp, uint32_t keys,
        uint32_t heads, uint32_t width, uint32_t window, uint32_t ratio, double *started) {
    @autoreleasepool {
        if (!token || !*token) return 0;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)*token;
        if (!ds4_gpu_flash_attn_stage_profile_boundary(&cb, mode, stage,
                rows, comp, keys, heads, width, window, ratio, started)) return 0;
        ds4_metal_attention_replace_command(token, cb);
        return 1;
    }
}
int ds4_metal_attention_batch_pad(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_pad_args *args, ds4_metal_attention_view kv,
        ds4_metal_attention_view mask, ds4_metal_attention_view pad, uint64_t groups) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {kv,kv,mask,pad};
        for (NSUInteger i=0;i<4;i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_block_map(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_block_args *args, ds4_metal_attention_view mask,
        ds4_metal_attention_view block, uint64_t gx, uint64_t gy) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)mask.buffer offset:mask.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)block.buffer offset:block.offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_nonvector(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_vec_args *args, ds4_metal_attention_view q,
        ds4_metal_attention_view kv, ds4_metal_attention_view mask,
        ds4_metal_attention_view sinks, ds4_metal_attention_view pad,
        ds4_metal_attention_view block, ds4_metal_attention_view dst,
        uint64_t gx, uint64_t gy, uint64_t nsg, uint64_t shared) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {q,kv,kv,mask,sinks,pad,block,dst};
        for (NSUInteger i=0;i<8;i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        [enc setThreadgroupMemoryLength:shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_causal_mask(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_causal_args *args, ds4_metal_attention_view src,
        ds4_metal_attention_view dst, uint64_t threads) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:src.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:dst.offset atIndex:2];
        [enc dispatchThreads:MTLSizeMake(args->n_comp,args->n_tokens,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_index_sort(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_index_rows_args *args, ds4_metal_attention_view src,
        ds4_metal_attention_view dst, uint64_t groups, uint64_t threads, uint64_t shared) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:src.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:dst.offset atIndex:2];
        [enc setThreadgroupMemoryLength:shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_index_run(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_indexed_args *args, ds4_metal_attention_view q,
        ds4_metal_attention_view raw, ds4_metal_attention_view comp,
        ds4_metal_attention_view topk, ds4_metal_attention_view sinks,
        ds4_metal_attention_view dst, bool split,
        uint64_t gx, uint64_t gy, uint64_t gz, uint64_t shared) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {q,raw,comp,topk,sinks,dst};
        for (NSUInteger i=0;i<(split ? 4u : 6u);i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        if (split) [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:dst.offset atIndex:5];
        [enc setThreadgroupMemoryLength:shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(gx,gy,gz) threadsPerThreadgroup:MTLSizeMake(32,8,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_attention_index_reduce(void *command, ds4_metal_attention_pipeline pso,
        const ds4_metal_attention_indexed_args *args, ds4_metal_attention_view src,
        ds4_metal_attention_view sinks, ds4_metal_attention_view dst, uint64_t rows) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        if (!cb || !pso.handle) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.handle];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        ds4_metal_attention_view views[] = {src,sinks,dst};
        for (NSUInteger i=0;i<3;i++) [enc setBuffer:(__bridge id<MTLBuffer>)views[i].buffer offset:views[i].offset atIndex:i+1];
        [enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
