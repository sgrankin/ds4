/* Fixed ARC resource factory and named bindings for native Shared/Router.
 * No gate/shape/fallback/dispatch planning remains in these leaves. */
#include "router_shared.h"
ds4_metal_attention_view ds4_metal_router_completion(ds4_metal_attention_view logits) {
    @autoreleasepool {
        if (!g_dsv4_completion_cache) return (ds4_metal_attention_view){0};
        NSString *key = [NSString stringWithFormat:@"router:%p:%llu", (void *)logits.buffer,
            (unsigned long long)logits.offset];
        id<MTLBuffer> completion = [g_dsv4_completion_cache objectForKey:key];
        if (!completion) {
            completion = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            if (!completion) return (ds4_metal_attention_view){0};
            *((uint32_t *)[completion contents]) = 0u;
            [g_dsv4_completion_cache setObject:completion forKey:key];
        }
        [g_transient_buffers addObject:completion];
        return (ds4_metal_attention_view){ (ds4_metal_model_buffer *)(__bridge_retained void *)completion, 0, sizeof(uint32_t) };
    }
}
int ds4_metal_router_shared_encode(void *command, const ds4_metal_router_shared_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        if (!cb || !pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
#define BUFFER(I, V) [enc setBuffer:(__bridge id<MTLBuffer>)p->V.buffer offset:p->V.offset atIndex:I]
        switch (p->kind) {
        case DS4_SHARED_VECTOR:
        case DS4_SHARED_EXT:
            if (p->kind == DS4_SHARED_VECTOR) [enc setBytes:&p->shared length:sizeof(p->shared) atIndex:0];
            else [enc setBytes:&p->ext length:sizeof(p->ext) atIndex:0];
            BUFFER(1, gate_weight); BUFFER(2, up_weight); BUFFER(3, input);
            BUFFER(4, gate); BUFFER(5, up); BUFFER(6, mid);
            [enc setBytes:&p->clamp length:sizeof(p->clamp) atIndex:7];
            break;
        case DS4_ROUTER_SHARED:
            [enc setBytes:&p->router length:sizeof(p->router) atIndex:0];
            [enc setBytes:&p->shared length:sizeof(p->shared) atIndex:1];
            BUFFER(2, router_weight); BUFFER(3, gate_weight); BUFFER(4, up_weight);
            BUFFER(5, input); BUFFER(6, logits); BUFFER(7, gate); BUFFER(8, up); BUFFER(9, mid);
            [enc setBytes:&p->clamp length:sizeof(p->clamp) atIndex:10];
            break;
        case DS4_ROUTER_PROJECT:
            [enc setBytes:&p->router length:sizeof(p->router) atIndex:0];
            [enc setBytes:&p->select length:sizeof(p->select) atIndex:1];
            BUFFER(2, router_weight); BUFFER(3, input); BUFFER(4, logits); BUFFER(5, probs);
            if (p->has_bias) { BUFFER(6, bias); }
            else { const float zero = 0; [enc setBytes:&zero length:sizeof(zero) atIndex:6]; }
            BUFFER(7, selected); BUFFER(8, weights); BUFFER(9, completion);
            break;
        default: return 0;
        }
#undef BUFFER
        if (p->shared_bytes) [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(p->threads_x, p->threads_y, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
