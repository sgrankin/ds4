/* ARC resource factories and fixed shader bindings. Every stage, packet, range
 * and dispatch shape is selected by the native compressor owner. */
#include "compressor.h"
ds4_metal_attention_view ds4_metal_compressor_scratch(uint32_t kind, uint64_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> __strong *buffer = NULL;
        NSUInteger *capacity = NULL;
        const char *label = NULL;
#define SCRATCH(K, N)                                                                              \
    case K:                                                                                        \
        buffer = &g_compressor_##N##_buffer;                                                       \
        capacity = &g_compressor_##N##_bytes;                                                      \
        label = "ds4_compressor_" #N;                                                              \
        break
        switch (kind) {
            SCRATCH(DS4_COMP_KV, pool_kv);
            SCRATCH(DS4_COMP_SCORE, pool_score);
            SCRATCH(DS4_COMP_SCORE_CONT, pool_score_cont);
            SCRATCH(DS4_COMP_SCRATCH_SOFTMAX, pool_softmax);
            SCRATCH(DS4_COMP_PRODUCT, pool_product);
            SCRATCH(DS4_COMP_APE, store_ape);
            SCRATCH(DS4_COMP_ADDED_SCORE, store_score);
        default:
            return (ds4_metal_attention_view){0};
        }
#undef SCRATCH
        if (!ds4_gpu_ensure_scratch_buffer(buffer, capacity, (NSUInteger)bytes, label))
            return (ds4_metal_attention_view){0};
        return (ds4_metal_attention_view){
            (ds4_metal_model_buffer *)(__bridge_retained void *)*buffer, 0,
            (uint64_t)[*buffer length]};
    }
}
ds4_metal_attention_pipeline ds4_metal_compressor_pipeline(uint32_t kind, bool half) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        switch (kind) {
        case DS4_COMP_COPY:
            pipeline = g_cpy_f32_f32_pipeline;
            break;
        case DS4_COMP_ADD:
            pipeline = g_add2_pipeline;
            break;
        case DS4_COMP_FILL:
            pipeline = g_unary_fill_pipeline;
            break;
        case DS4_COMP_SET_ROWS:
            pipeline = g_set_rows_f32_i32_pipeline;
            break;
        case DS4_COMP_SUM:
            pipeline = g_sum_rows_f32_f32_pipeline;
            break;
        case DS4_COMP_MUL:
            pipeline = g_mul_pipeline;
            break;
        case DS4_COMP_CONCAT:
            pipeline = g_concat_pipeline;
            break;
        case DS4_COMP_SOFTMAX:
            pipeline = half ? g_soft_max_f32_4_pipeline : g_soft_max_f32_pipeline;
            break;
        case DS4_COMP_TILE_APE:
            pipeline = ds4_gpu_get_pipeline(half ? "kernel_cpy_tile_rows_f16_f32"
                                                 : "kernel_cpy_tile_rows_f32_f32");
            break;
        case DS4_COMP_SCORE_APE:
            pipeline = ds4_gpu_get_pipeline(half ? "kernel_dsv4_compressor_score_ape_f16"
                                                 : "kernel_dsv4_compressor_score_ape_f32");
            break;
#define HOT(K, N, NAME)                                                                            \
    case K:                                                                                        \
        pipeline = ds4_gpu_hot_pipeline(g_##N##_pipeline, NAME);                                   \
        break
            HOT(DS4_COMP_STORE, dsv4_compressor_store_one, "kernel_dsv4_compressor_store_one");
        case DS4_COMP_POOL:
            pipeline = g_dsv4_softmax_pool_pipeline;
            break;
            HOT(DS4_COMP_DIRECT_POOL, dsv4_softmax_pool_ratio4_direct,
                "kernel_dsv4_softmax_pool_ratio4_direct");
            HOT(DS4_COMP_PACK, dsv4_compressor_pack_ratio4, "kernel_dsv4_compressor_pack_ratio4");
            HOT(DS4_COMP_DECODE_PACK, dsv4_compressor_pack_ratio4_decode_ggml,
                "kernel_dsv4_compressor_pack_ratio4_decode_ggml");
            HOT(DS4_COMP_EXACT_REDUCE, dsv4_compressor_exact_softmax_product,
                "kernel_dsv4_compressor_exact_softmax_product_ratio4");
            HOT(DS4_COMP_EXACT_POOL, dsv4_compressor_exact_pool_ratio4,
                "kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml");
        case DS4_COMP_SHIFT:
            pipeline = g_dsv4_ratio4_shift_pipeline;
            break;
#undef HOT
        }
        return (ds4_metal_attention_pipeline){(__bridge const ds4_metal_hc_pipeline *)pipeline,
                                              (uint64_t)pipeline.maxTotalThreadsPerThreadgroup,
                                              (uint64_t)pipeline.threadExecutionWidth};
    }
}
/* Availability is an immutable initialized-PSO fact, distinct from a hot
 * lookup that can re-resolve a pipeline when static lookup is disabled. */
uint32_t ds4_metal_compressor_cached_pipelines(void) {
    return (g_dsv4_compressor_pack_ratio4_pipeline ? 1u << DS4_COMP_PACK : 0) |
           (g_dsv4_compressor_pack_ratio4_decode_ggml_pipeline ? 1u << DS4_COMP_DECODE_PACK : 0) |
           (g_dsv4_softmax_pool_ratio4_direct_pipeline ? 1u << DS4_COMP_DIRECT_POOL : 0);
}
/* A process-cached public tuning fact, shared with existing dense projections. */
bool ds4_metal_compressor_pair_nr4(void) { return ds4_gpu_use_compressor_pair_nr4() != 0; }
int ds4_metal_compressor_encode(void *command, const ds4_metal_compressor_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)p->pipeline.handle;
        if (!cb || !pipeline)
            return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
#define PACK(F) [enc setBytes:&p->args.F length:sizeof(p->args.F) atIndex:0]
#define B(I, V)                                                                                    \
    [enc setBuffer:(__bridge id<MTLBuffer>)p->V.buffer offset:(NSUInteger)p->V.offset atIndex:I]
        switch (p->kind) {
        case DS4_COMP_COPY:
            PACK(copy);
            B(1, kv);
            B(2, out);
            break;
        case DS4_COMP_ADD:
            PACK(add);
            B(1, kv);
            B(2, score);
            B(3, out);
            break;
        case DS4_COMP_FILL:
            PACK(fill);
            B(1, out);
            B(2, out);
            break;
        case DS4_COMP_SET_ROWS:
            PACK(set_rows);
            B(1, kv);
            if (p->indices.buffer) {
                B(2, indices);
            } else
                [enc setBytes:p->rows length:(NSUInteger)p->row_count * sizeof(int32_t) atIndex:2];
            B(3, out);
            break;
        case DS4_COMP_SUM:
            PACK(sum);
            B(1, kv);
            B(2, out);
            break;
        case DS4_COMP_MUL:
            PACK(multiply);
            B(1, kv);
            B(2, score);
            B(3, out);
            break;
        case DS4_COMP_CONCAT:
            PACK(concat);
            B(1, kv);
            B(2, score);
            B(3, out);
            break;
        case DS4_COMP_SOFTMAX:
            PACK(softmax);
            B(1, kv);
            B(2, kv);
            B(3, kv);
            B(4, out);
            break;
        case DS4_COMP_TILE_APE:
            PACK(tile);
            B(1, ape);
            B(2, out);
            break;
        case DS4_COMP_SCORE_APE:
            PACK(score_ape);
            B(1, score);
            B(2, ape);
            B(3, out);
            break;
        case DS4_COMP_STORE:
            PACK(store);
            B(1, kv);
            B(2, score);
            B(3, ape);
            B(4, state_kv);
            B(5, state_score);
            break;
        case DS4_COMP_POOL:
            PACK(pool);
            B(1, kv);
            B(2, score);
            B(3, out);
            break;
        case DS4_COMP_DIRECT_POOL:
            PACK(direct_pool);
            B(1, kv);
            B(2, score);
            B(3, state_kv);
            B(4, state_score);
            B(5, out);
            break;
        case DS4_COMP_PACK:
            PACK(pack);
            B(1, kv);
            B(2, score);
            B(3, state_kv);
            B(4, state_score);
            B(5, product);
            B(6, score_cont);
            break;
        case DS4_COMP_DECODE_PACK:
            PACK(pack);
            B(1, state_kv);
            B(2, state_score);
            B(3, product);
            B(4, score_cont);
            break;
        case DS4_COMP_EXACT_REDUCE:
            PACK(pack);
            B(1, product);
            B(2, score_cont);
            B(3, softmax);
            B(4, product);
            break;
        case DS4_COMP_EXACT_POOL:
            PACK(pack);
            B(1, state_kv);
            B(2, state_score);
            B(3, softmax);
            B(4, product);
            B(5, out);
            break;
        case DS4_COMP_SHIFT:
            PACK(shift);
            B(1, state_kv);
            B(2, state_score);
            break;
        default:
            return 0;
        }
#undef PACK
#undef B
        if (p->shared)
            [enc setThreadgroupMemoryLength:(NSUInteger)p->shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->gx, p->gy, p->gz)
            threadsPerThreadgroup:MTLSizeMake(p->tx, p->ty, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
int ds4_metal_compressor_fused_encode(void *command, const ds4_metal_compressor_fused *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !pipeline)
            return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
#define P(I, V) [enc setBytes:&p->V length:sizeof(p->V) atIndex:I]
#define B(I, V)                                                                                    \
    [enc setBuffer:(__bridge id<MTLBuffer>)p->V.buffer offset:(NSUInteger)p->V.offset atIndex:I]
        switch (p->kind) {
        case DS4_COMP_FUSED_PAIR:
            P(0, mv[0]);
            P(1, store[0]);
            B(2, pairs[0].weight_kv);
            B(3, pairs[0].weight_score);
            B(4, input);
            B(5, pairs[0].out_kv);
            B(6, pairs[0].out_score);
            B(7, pairs[0].ape);
            B(8, pairs[0].state_kv);
            B(9, pairs[0].state_score);
            break;
        case DS4_COMP_FUSED_QUAD:
            P(0, mv[0]);
            P(1, store[0]);
            P(2, store[1]);
            B(3, pairs[0].weight_kv);
            B(4, pairs[0].weight_score);
            B(5, pairs[1].weight_kv);
            B(6, pairs[1].weight_score);
            B(7, input);
            B(8, pairs[0].out_kv);
            B(9, pairs[0].out_score);
            B(10, pairs[1].out_kv);
            B(11, pairs[1].out_score);
            B(12, pairs[0].ape);
            B(13, pairs[1].ape);
            B(14, pairs[0].state_kv);
            B(15, pairs[0].state_score);
            B(16, pairs[1].state_kv);
            B(17, pairs[1].state_score);
            break;
        case DS4_COMP_FUSED_QKV:
            P(0, mv[0]);
            P(1, mv[1]);
            P(2, mv[2]);
            P(3, store[0]);
            P(4, store[1]);
            P(5, pair_groups);
            B(6, q_weights);
            B(7, kv_weights);
            B(8, pairs[0].weight_kv);
            B(9, pairs[0].weight_score);
            B(10, pairs[1].weight_kv);
            B(11, pairs[1].weight_score);
            B(12, input);
            B(13, qr);
            B(14, raw);
            B(15, pairs[0].out_kv);
            B(16, pairs[0].out_score);
            B(17, pairs[1].out_kv);
            B(18, pairs[1].out_score);
            B(19, pairs[0].ape);
            B(20, pairs[1].ape);
            B(21, pairs[0].state_kv);
            B(22, pairs[0].state_score);
            B(23, pairs[1].state_kv);
            B(24, pairs[1].state_score);
            break;
        case DS4_COMP_FUSED_FINALIZE:
            P(0, finalize);
            B(1, attn_stage);
            B(2, attn_norm);
            B(3, attn_cache);
            B(4, index_cache);
            B(5, index_norm);
            B(6, pairs[0].state_kv);
            B(7, pairs[0].state_score);
            B(8, pairs[1].state_kv);
            B(9, pairs[1].state_score);
            break;
        default:
            return 0;
        }
#undef P
#undef B
        if (p->shared)
            [enc setThreadgroupMemoryLength:(NSUInteger)p->shared atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(p->kind == DS4_COMP_FUSED_FINALIZE ? 256 : 32, p->nsg,
                                              1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
