/* Actual ARC event/buffer storage, fixed shader bindings and the existing
 * transport queue. All gate policy and publication order are native. */
#include "tp_control.h"
ds4_metal_tp_policy *ds4_metal_tp_policy_state(void) {
  return &g_tp_native_policy;
}
ds4_metal_tp_facts ds4_metal_tp_get_facts(void) {
  @autoreleasepool {
    bool available = false;
    if (@available(macOS 12.0, *))
      available = true;
#define V(B)                                                                   \
  (ds4_metal_attention_view) {                                                 \
    (ds4_metal_model_buffer *)(__bridge void *)(B), 0, (uint64_t)[(B) length]  \
  }
    return (ds4_metal_tp_facts){
        .initialized = g_initialized,
        .batch = g_batch_cb != nil,
        .service_running = g_tp_thread_running != 0,
        .ssd = g_ssd_streaming_mode,
        .checks = g_tp_check_words != NULL,
        .shared_events = available,
        .selected_event = g_selected_readback_event != nil,
        .pending_count = (uint64_t)[g_pending_cbs count],
        .slab = V(g_tp_slab_buffer),
        .checks_buffer = V(g_tp_check_buffer),
        .poll = V(g_tp_poll_buffer)};
#undef V
  }
}
ds4_metal_attention_view ds4_metal_tp_fold_control(void) {
  @autoreleasepool {
    if (!g_tp_fold_ctl) {
      g_tp_fold_ctl =
          [g_device newBufferWithLength:(NSUInteger)DS4_TP_POLL_MAX_SLOTS * 8u
                                options:MTLResourceStorageModeShared];
      if (!g_tp_fold_ctl)
        return (ds4_metal_attention_view){0};
      memset(g_tp_fold_ctl.contents, 0, (size_t)DS4_TP_POLL_MAX_SLOTS * 8u);
    }
    return (ds4_metal_attention_view){
        (ds4_metal_model_buffer *)(__bridge void *)g_tp_fold_ctl, 0,
        (uint64_t)g_tp_fold_ctl.length};
  }
}
void ds4_metal_tp_signal(uint32_t kind, uint64_t sequence) {
  @autoreleasepool {
    id<MTLSharedEvent> event = kind == DS4_TP_EVENT_ROW ? g_tp_gpu_event
                               : kind == DS4_TP_EVENT_BATCH
                                   ? g_tp_batch_gpu_event
                                   : g_selected_readback_event;
    [g_batch_cb encodeSignalEvent:event value:sequence];
    if (kind == DS4_TP_EVENT_SELECTED)
      g_batch_has_work = YES;
  }
}
void ds4_metal_tp_wait(uint32_t kind, uint64_t sequence) {
  @autoreleasepool {
    [g_batch_cb
        encodeWaitForEvent:kind == DS4_TP_EVENT_ROW ? g_tp_cpu_event
                                                    : g_tp_batch_cpu_event
                     value:sequence];
  }
}
bool ds4_metal_tp_selected_ready(void) {
  @autoreleasepool {
    if (!g_selected_readback_event)
      g_selected_readback_event = [g_device newSharedEvent];
    return g_selected_readback_event != nil;
  }
}
bool ds4_metal_tp_wait_selected(uint64_t sequence) {
  @autoreleasepool {
    return [g_selected_readback_event waitUntilSignaledValue:sequence
                                                   timeoutMS:60000];
  }
}
bool ds4_metal_tp_enqueue(const ds4_gpu_tp_request *request) {
  pthread_mutex_lock(&g_tp_mutex);
  if (g_tp_queue_count >= DS4_GPU_TP_QUEUE) {
    pthread_mutex_unlock(&g_tp_mutex);
    fprintf(stderr, "ds4: TP gate queue overflow\n");
    return false;
  }
  uint32_t tail = (g_tp_queue_head + g_tp_queue_count) % DS4_GPU_TP_QUEUE;
  g_tp_queue[tail] = *request;
  g_tp_queue_count++;
  pthread_cond_signal(&g_tp_cond);
  pthread_mutex_unlock(&g_tp_mutex);
  return true;
}
void ds4_metal_tp_wait_pending_index(uint64_t index) {
  @autoreleasepool {
    [g_pending_cbs[(NSUInteger)index] waitUntilCompleted];
  }
}
void *ds4_metal_tp_take_batch(uint32_t work_mode) {
  id<MTLCommandBuffer> cb = g_batch_cb;
  g_batch_cb = nil;
  if (work_mode != 2)
    g_batch_has_work = work_mode != 0;
  return (__bridge_retained void *)cb;
}
void ds4_metal_tp_signal_detached(void *command, uint64_t sequence) {
  @autoreleasepool {
    [(__bridge id<MTLCommandBuffer>)command
        encodeSignalEvent:g_selected_readback_event
                    value:sequence];
    g_batch_has_work = YES;
  }
}
void ds4_metal_tp_commit(void *command) {
  @autoreleasepool {
    [(__bridge id<MTLCommandBuffer>)command commit];
    ds4_gpu_stream_expert_cache_note_batch_committed();
  }
}
void ds4_metal_tp_append_pending(void *command) {
  [g_pending_cbs addObject:(__bridge id<MTLCommandBuffer>)command];
}
bool ds4_metal_tp_command_error(void *command, const char *label) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    if (cb.status != MTLCommandBufferStatusError)
      return false;
    fprintf(stderr, "ds4: Metal %s failed: %s\n", label,
            [[cb.error localizedDescription] UTF8String]);
    return true;
  }
}
void ds4_metal_tp_drop(void *command) {
  if (command) {
    __unused id retired = (__bridge_transfer id)command;
  }
}
void ds4_metal_tp_clear_transients(void) {
  [g_transient_buffers removeAllObjects];
}
bool ds4_metal_tp_new_batch(void) {
  g_batch_cb = ds4_gpu_new_command_buffer();
  g_batch_has_work = NO;
  if (g_batch_cb)
    ds4_gpu_stream_expert_cache_note_batch_created();
  return g_batch_cb != nil;
}
/* Named temporary whole library, not relabeled as a driver leaf. */
int ds4_metal_tp_prefetch(uint32_t gate) {
  return ds4_gpu_tp_encode_gate_prefetch(gate);
}
int ds4_metal_tp_encode(void *command, const ds4_metal_tp_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    if (!cb || !p->pipeline)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)p->
                                 pipeline];
#define B(I, F)                                                                \
  [enc setBuffer:(__bridge id<MTLBuffer>)p->F.buffer                           \
          offset:(NSUInteger)p->F.offset                                       \
         atIndex:I]
#define A(I, F) [enc setBytes:&p->F length:sizeof(p->F) atIndex:I]
    switch (p->kind) {
    case DS4_TP_FLAG:
      B(0, flags);
      A(1, value);
      break;
    case DS4_TP_CHECKED_FLAG:
      B(0, flags);
      B(1, checks);
      A(2, value);
      B(3, payload);
      A(4, words);
      break;
    case DS4_TP_POLL:
      B(0, flags);
      A(1, value);
      A(2, words);
      B(3, checks);
      break;
    case DS4_TP_FOLD_ADD:
      A(0, count);
      B(1, a);
      B(2, b);
      B(3, out);
      B(4, flags);
      B(5, checks);
      A(6, value);
      B(7, control);
      A(8, groups);
      break;
    default:
      return 0;
    }
    if (p->shared_bytes)
      [enc setThreadgroupMemoryLength:(NSUInteger)p->shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(p->groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    if (p->end_encoder)
      ds4_gpu_end_compute_encoder(cb, enc);
#undef B
#undef A
    return 1;
  }
}
