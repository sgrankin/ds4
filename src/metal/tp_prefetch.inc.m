/* Actual retained scratch and address query; touch geometry and plan policy
 * are entirely native. These do not call an old prefetch controller. */
#include "tp_prefetch.h"
ds4_metal_tp_prefetch_policy *ds4_metal_tp_prefetch_policy_state(void) {
  return &g_tp_native_prefetch;
}
ds4_metal_attention_view ds4_metal_tp_prefetch_storage(void) {
  @autoreleasepool {
    if (!g_tp_prefetch_scratch) {
      g_tp_prefetch_scratch =
          [g_device newBufferWithLength:(NSUInteger)(1u << 20)
                                options:MTLResourceStorageModePrivate];
    }
    return (ds4_metal_attention_view){
        (ds4_metal_model_buffer *)(__bridge void *)g_tp_prefetch_scratch, 0,
        (uint64_t)g_tp_prefetch_scratch.length};
  }
}
uint64_t ds4_metal_tp_prefetch_address(ds4_metal_attention_view view) {
  return ds4_gpu_buffer_address((__bridge id<MTLBuffer>)view.buffer,
                                (NSUInteger)view.offset);
}
int ds4_metal_tp_prefetch_encode(void *command,
                                 const ds4_metal_tp_prefetch_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    if (!cb)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (p->count) {
      id<MTLResource> resources[DS4_TP_PREFETCH_MAX];
      for (uint32_t i = 0; i < p->count; i++)
        resources[i] = (__bridge id<MTLBuffer>)p->ranges[i].buffer;
      [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)p->
                                   pipeline];
      [enc useResources:resources count:p->count usage:MTLResourceUsageRead];
      [enc setBytes:p->addresses length:sizeof(p->addresses) atIndex:0];
      [enc setBytes:p->lines length:sizeof(p->lines) atIndex:1];
      [enc setBytes:&p->count length:sizeof(p->count) atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)p->scratch.buffer
              offset:(NSUInteger)p->scratch.offset
             atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(p->groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    }
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}
