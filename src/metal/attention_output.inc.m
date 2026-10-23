/* Fixed packet bindings and ARC resource factories only. Native supplies all
 * dimensions, offsets, work-map layout, stages and dispatch geometry. */
#include "attention_output.h"
ds4_metal_attention_view ds4_metal_output_scratch(uint32_t kind, uint64_t bytes, bool transient,
                                                  const char *label) {
    @autoreleasepool {
        id<MTLBuffer> buffer = nil;
        if (kind == DS4_OUT_IDS_SCRATCH) {
            if (transient)
                buffer = ds4_gpu_new_transient_buffer((NSUInteger)bytes, label);
            else if (ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                   &g_attn_out_group_ids_bytes, (NSUInteger)bytes,
                                                   "ds4_attention_output_group_ids"))
                buffer = g_attn_out_group_ids_buffer;
        } else if (kind == DS4_OUT_WORKMAP_SCRATCH) {
            if (ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer, &g_moe_id_map_bytes,
                                              (NSUInteger)bytes, "ds4_moe_id_map"))
                buffer = g_moe_id_map_buffer;
        }
        return (ds4_metal_attention_view){
            (ds4_metal_model_buffer *)(__bridge_retained void *)buffer, 0,
            buffer ? (uint64_t)[buffer length] : 0};
    }
}
ds4_metal_output_fold ds4_metal_output_fold_take(const ds4_gpu_tensor *out, uint64_t bytes) {
    ds4_metal_output_fold result = {0};
    result.active = ds4_gpu_tp_flag_fold_take(out, bytes, &result.slot, &result.value) != 0;
    return result;
}
void ds4_metal_output_fold_publish(void) { g_tp_flag_prepublished_seq = g_tp_seq + 1u; }
int ds4_metal_output_encode(void *command, const ds4_metal_output_launch *p) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
        id<MTLBuffer> weight = (__bridge id<MTLBuffer>)p->weights.buffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)p->input.buffer;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)p->output.buffer;
        id<MTLBuffer> ids = (__bridge id<MTLBuffer>)p->ids.buffer;
        id<MTLBuffer> work = (__bridge id<MTLBuffer>)p->workmap.buffer;
        if (!cb || !pipeline)
            return 0;
        if (p->kind == DS4_OUT_MAP) {
            if (!ids || !work)
                return 0;
        } else if (!weight || !input || !output)
            return 0;
        if (p->kind == DS4_OUT_LOW_IDS && !ids)
            return 0;
        if (p->kind == DS4_OUT_MAPPED && !work)
            return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        switch (p->kind) {
        case DS4_OUT_MAP:
            [enc setBytes:&p->args.map length:sizeof(p->args.map) atIndex:0];
            [enc setBuffer:ids offset:p->ids.offset atIndex:1];
            [enc setBuffer:work offset:p->workmap.offset atIndex:2];
            [enc setBuffer:work offset:p->workmap.offset + p->work_ids_offset atIndex:3];
            [enc setBuffer:work offset:p->workmap.offset + p->work_list_offset atIndex:4];
            break;
        case DS4_OUT_MAPPED:
            [enc setBytes:&p->args.grouped length:sizeof(p->args.grouped) atIndex:0];
            [enc setBuffer:weight offset:p->weights.offset atIndex:1];
            [enc setBuffer:input offset:p->input.offset atIndex:2];
            [enc setBuffer:work offset:p->workmap.offset atIndex:3];
            [enc setBuffer:work offset:p->workmap.offset + p->work_ids_offset atIndex:4];
            [enc setBuffer:output offset:p->output.offset atIndex:5];
            [enc setBuffer:work offset:p->workmap.offset + p->work_list_offset atIndex:6];
            if (p->hint_weights)
                [enc useResource:weight usage:MTLResourceUsageRead];
            break;
        default:
            if (p->kind == DS4_OUT_KSLICE_MV)
                [enc setBytes:&p->args.vector length:sizeof(p->args.vector) atIndex:0];
            else if (p->kind == DS4_OUT_KSLICE_MM)
                [enc setBytes:&p->args.matrix length:sizeof(p->args.matrix) atIndex:0];
            else if (p->kind == DS4_OUT_MPP)
                [enc setBytes:&p->args.grouped length:sizeof(p->args.grouped) atIndex:0];
            else
                [enc setBytes:&p->args.low length:sizeof(p->args.low) atIndex:0];
            [enc setBuffer:weight offset:p->weights.offset atIndex:1];
            [enc setBuffer:input offset:p->input.offset atIndex:2];
            [enc setBuffer:output offset:p->output.offset atIndex:3];
            if (p->kind == DS4_OUT_LOW_IDS)
                [enc setBuffer:ids offset:p->ids.offset atIndex:4];
            if (p->fold) {
                uint32_t groups = (uint32_t)p->groups_x;
                [enc setBuffer:g_tp_slab_buffer
                        offset:(NSUInteger)(g_tp_slab_buffer_off + g_tp_gpu_flags_off +
                                            (uint64_t)p->fold_slot * 4u)
                       atIndex:4];
                [enc setBuffer:g_tp_check_buffer offset:(NSUInteger)p->fold_slot * 4u atIndex:5];
                [enc setBytes:&p->fold_value length:sizeof(p->fold_value) atIndex:6];
                [enc setBuffer:g_tp_fold_ctl offset:(NSUInteger)p->fold_slot * 8u atIndex:7];
                [enc setBytes:&groups length:sizeof(groups) atIndex:8];
                [enc setThreadgroupMemoryLength:16 atIndex:1];
            }
            break;
        }
        if (p->shared_bytes)
            [enc setThreadgroupMemoryLength:p->shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, p->groups_z)
            threadsPerThreadgroup:MTLSizeMake(p->threads_x, p->threads_y, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }
}
