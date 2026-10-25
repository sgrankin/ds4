/* True ARC/driver leaves for the native command, parallel FFN and KV owner.
 * Policy lives in native code; no old public controller is called here. */
#include "scheduling.h"
ds4_metal_scheduling *ds4_metal_scheduling_state(void) { return &g_schedule; }
void *ds4_metal_schedule_retain(void *object) {
    return object ? (__bridge_retained void *)(__bridge id)object : NULL;
}
void ds4_metal_schedule_drop(void *object) {
    if (object) { id value = (__bridge_transfer id)object; (void)value; }
}
void *ds4_metal_schedule_new_command(bool unretained) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = unretained ? [g_queue commandBufferWithUnretainedReferences]
                                           : [g_queue commandBuffer];
        return cb ? (__bridge_retained void *)cb : NULL;
    }
}
void ds4_metal_schedule_set_batch(void *cb) {
    g_schedule.commands.batch = (__bridge id<MTLCommandBuffer>)cb;
}
void ds4_metal_schedule_set_encoder(void *enc) {
    g_schedule.commands.encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
}
void *ds4_metal_schedule_new_encoder(void *command, bool concurrent, bool timeline, uintptr_t caller) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
        id<MTLComputeCommandEncoder> enc = timeline
            ? ds4_gpu_timeline_new_encoder(cb, concurrent, caller)
            : concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                         : [cb computeCommandEncoder];
        return enc ? (__bridge_retained void *)enc : NULL;
    }
}
void ds4_metal_schedule_end_encoder(void *enc) {
    [(__bridge id<MTLComputeCommandEncoder>)enc endEncoding];
}
bool ds4_metal_schedule_is_concurrent(void *enc) {
    return [(__bridge id<MTLComputeCommandEncoder>)enc dispatchType] == MTLDispatchTypeConcurrent;
}
bool ds4_metal_schedule_timeline_active(void) { return g_timeline_enabled && g_timeline_batch; }
void ds4_metal_schedule_timeline_attach(void *cb) {
    ds4_gpu_timeline_attach((__bridge id<MTLCommandBuffer>)cb);
}
void ds4_metal_schedule_commit(void *cb) { ds4_runtime_metal_commit(cb, &g_bootstrap, false); }
uint64_t ds4_metal_schedule_pending_count(void) { return [g_schedule.commands.pending count]; }
void *ds4_metal_schedule_pending_at(uint64_t index) {
    return (__bridge_retained void *)g_schedule.commands.pending[(NSUInteger)index];
}
void ds4_metal_schedule_pending_append(void *cb) {
    [g_schedule.commands.pending addObject:(__bridge id<MTLCommandBuffer>)cb];
}
void ds4_metal_schedule_pending_clear(void) { [g_schedule.commands.pending removeAllObjects]; }
int ds4_metal_schedule_wait(void *cb, const char *label) {
    /* This existing driver wait includes status/error diagnostics and drops
     * completion-counter cache ownership on a failed command buffer. */
    return ds4_gpu_wait_command_buffer((__bridge id<MTLCommandBuffer>)cb, label);
}
void ds4_metal_schedule_invalidate_maps(void) { ds4_gpu_invalidate_zero_prefix_prefill_block_maps(); }
void ds4_metal_schedule_clear_transients(bool profile) {
    @autoreleasepool {
        if (profile) {
            uint64_t bytes = 0;
            for (id<MTLBuffer> b in g_transient_buffers) bytes += [b length];
            fprintf(stderr, "ds4: transient buffers this batch: %lu (%.1f MiB)\n",
                    (unsigned long)[g_transient_buffers count], bytes / 1048576.0);
        }
        [g_transient_buffers removeAllObjects];
    }
}
void ds4_metal_schedule_evict(const char *label) { ds4_gpu_model_buffer_cache_maybe_evict(label); }
void ds4_metal_schedule_cache_event(uint32_t event) {
    switch (event) {
        case DS4_SCHEDULE_OWNED_CREATED: ds4_gpu_stream_expert_cache_note_owned_created(); break;
        case DS4_SCHEDULE_OWNED_DONE: ds4_gpu_stream_expert_cache_note_owned_completed(); break;
        case DS4_SCHEDULE_BATCH_CREATED: ds4_gpu_stream_expert_cache_note_batch_created(); break;
        case DS4_SCHEDULE_BATCH_COMMITTED: ds4_gpu_stream_expert_cache_note_batch_committed(); break;
        case DS4_SCHEDULE_PENDING_DONE: ds4_gpu_stream_expert_cache_note_pending_completed(); break;
        case DS4_SCHEDULE_BATCH_TO_OWNED:
            g_stream_expert_cache_owned_seq = g_stream_expert_cache_batch_seq;
            g_stream_expert_cache_batch_seq = 0;
            break;
    }
}
uint32_t ds4_metal_schedule_facts(void) {
    return (g_ssd_streaming_mode ? DS4_SCHEDULE_STREAMING : 0) |
        (ds4_gpu_device_is_pre_m5_apple_silicon() ? DS4_SCHEDULE_PRE_M5 : 0);
}
uint64_t ds4_metal_schedule_pipeline_threads(const ds4_metal_hc_pipeline *p) {
    return [(__bridge id<MTLComputePipelineState>)p maxTotalThreadsPerThreadgroup];
}
void ds4_metal_schedule_finish_trace(void *command, const char *label,
                                    double commit_ms, double created_ms, uintptr_t caller) {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    const double done_ms = ds4_gpu_now_ms();
    struct timespec mono;
    clock_gettime(CLOCK_MONOTONIC, &mono);
    const double mono_ms = (double)mono.tv_sec * 1e3 + (double)mono.tv_nsec / 1e6;
    fprintf(stderr, "ds4: cb '%s': mono %.1f ms: ", label ? label : "?", mono_ms);
    fprintf(stderr, "encode %.1f ms, commit->done %.1f ms, gpu span %.1f ms, gpu start +%.1f ms, caller 0x%llx\n",
        commit_ms - created_ms, done_ms - commit_ms,
        (cb.GPUEndTime - cb.GPUStartTime) * 1e3,
        cb.GPUStartTime * 1e3 - (mono_ms - (done_ms - commit_ms)),
        (unsigned long long)(caller - (uintptr_t)_dyld_get_image_vmaddr_slide(0)));
    if (done_ms - commit_ms > 100.0 && (cb.GPUEndTime - cb.GPUStartTime) * 1e3 < 10.0) {
        void *frames[12];
        const int count = backtrace(frames, 12);
        fprintf(stderr, "ds4: cb stall backtrace:");
        for (int i = 1; i < count; i++)
            fprintf(stderr, " 0x%llx", (unsigned long long)((uintptr_t)frames[i] - (uintptr_t)_dyld_get_image_vmaddr_slide(0)));
        fprintf(stderr, "\n");
    }
}

void ds4_metal_schedule_parallel_install(const ds4_metal_attention_view *v,
    const ds4_metal_hc_pipeline *gate, const ds4_metal_hc_pipeline *down) {
    ds4_metal_parallel_ffn *p = &g_schedule.parallel;
    p->gate_pipeline = (__bridge id<MTLComputePipelineState>)gate;
    p->down_pipeline = (__bridge id<MTLComputePipelineState>)down;
    p->gate_weight = (__bridge id<MTLBuffer>)v[0].buffer;
    p->up_weight = (__bridge id<MTLBuffer>)v[1].buffer;
    p->gate_x = (__bridge id<MTLBuffer>)v[2].buffer;
    p->gate_out = (__bridge id<MTLBuffer>)v[3].buffer;
    p->up_out = (__bridge id<MTLBuffer>)v[4].buffer;
    p->mid_out = (__bridge id<MTLBuffer>)v[5].buffer;
    p->down_weight = (__bridge id<MTLBuffer>)v[6].buffer;
    p->down_x = (__bridge id<MTLBuffer>)v[5].buffer;
    p->down_out = (__bridge id<MTLBuffer>)v[7].buffer;
    p->ids = (__bridge id<MTLBuffer>)v[8].buffer;
}
void ds4_metal_schedule_parallel_retire(void) {
    ds4_metal_parallel_ffn *p = &g_schedule.parallel;
    p->down_pipeline = nil; p->gate_pipeline = nil;
    p->down_weight = nil; p->down_x = nil; p->down_out = nil;
    p->gate_weight = nil; p->up_weight = nil; p->gate_x = nil;
    p->gate_out = nil; p->up_out = nil; p->mid_out = nil; p->ids = nil;
}
void ds4_metal_schedule_parallel_gate(void *encoder, uint64_t groups) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    ds4_metal_parallel_ffn *p = &g_schedule.parallel;
    [enc setComputePipelineState:p->gate_pipeline];
    [enc setBytes:&p->gate_args length:sizeof(p->gate_args) atIndex:0];
    const NSUInteger base = p->split ? 2 : 1;
    if (p->split) [enc setBytes:&p->split_args length:sizeof(p->split_args) atIndex:1];
    [enc setBuffer:p->gate_weight offset:p->gate_weight_offset atIndex:base];
    [enc setBuffer:p->up_weight offset:p->up_weight_offset atIndex:base + 1];
    [enc setBuffer:p->gate_x offset:p->gate_x_offset atIndex:base + 2];
    [enc setBuffer:p->gate_out offset:p->gate_out_offset atIndex:base + 3];
    [enc setBuffer:p->up_out offset:p->up_out_offset atIndex:base + 4];
    [enc setBuffer:p->mid_out offset:p->mid_out_offset atIndex:base + 5];
    [enc setBytes:&p->clamp length:sizeof(p->clamp) atIndex:base + 6];
    if (p->split) [enc setBuffer:p->ids offset:p->ids_offset atIndex:9];
    [enc setThreadgroupMemoryLength:2u * p->shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, p->nsg, 1)];
}
void ds4_metal_schedule_parallel_down(void *encoder, uint64_t groups) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    ds4_metal_parallel_ffn *p = &g_schedule.parallel;
    [enc setComputePipelineState:p->down_pipeline];
    [enc setBytes:&p->down_args length:sizeof(p->down_args) atIndex:0];
    const NSUInteger base = p->split ? 2 : 1;
    if (p->split) [enc setBytes:&p->split_args length:sizeof(p->split_args) atIndex:1];
    [enc setBuffer:p->down_weight offset:p->down_weight_offset atIndex:base];
    [enc setBuffer:p->down_x offset:p->down_x_offset atIndex:base + 1];
    [enc setBuffer:p->down_out offset:p->down_out_offset atIndex:base + 2];
    if (p->split) [enc setBuffer:p->ids offset:p->ids_offset atIndex:5];
    [enc setThreadgroupMemoryLength:256u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}
void ds4_metal_schedule_parallel_barrier(void *encoder, void *routed_mid) {
    id<MTLResource> resources[2] = { (__bridge id<MTLBuffer>)routed_mid, g_schedule.parallel.mid_out };
    [(__bridge id<MTLComputeCommandEncoder>)encoder memoryBarrierWithResources:resources count:2];
}
void ds4_metal_schedule_kv_install(const ds4_metal_attention_view *v) {
    ds4_metal_kv_task *t = &g_schedule.kv;
    t->q = (__bridge id<MTLBuffer>)v[0].buffer;
    t->q_w = (__bridge id<MTLBuffer>)v[1].buffer;
    t->q_out = (__bridge id<MTLBuffer>)v[2].buffer;
    t->kv = (__bridge id<MTLBuffer>)v[3].buffer;
    t->kv_w = (__bridge id<MTLBuffer>)v[4].buffer;
    t->kv_out = (__bridge id<MTLBuffer>)v[5].buffer;
    t->raw = (__bridge id<MTLBuffer>)v[6].buffer;
}
void ds4_metal_schedule_kv_encode(void *encoder, const ds4_metal_hc_pipeline *pipeline,
                                   const ds4_metal_kv_launch *p) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    ds4_metal_qkv_rms_args args = p->norm;
    args.tasks = p->tasks;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBytes:&p->rope length:sizeof(p->rope) atIndex:1];
    [enc setBytes:&p->store length:sizeof(p->store) atIndex:2];
    for (NSUInteger i = 0; i < 7; i++)
        [enc setBuffer:(__bridge id<MTLBuffer>)p->buffers[i].buffer offset:p->buffers[i].offset atIndex:i + 3];
    [enc setThreadgroupMemoryLength:(32u + 64u) * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, p->groups_y, 1)
        threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
}
