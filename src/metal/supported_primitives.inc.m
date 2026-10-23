/* Fixed bindings for the original carry/pre-RoPE/candidate shader packets. */
int ds4_metal_supported_encode(void *command,
                               const ds4_metal_supported_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)p->pipeline;
    if (!cb || !pipeline)
      return 0;
    const NSUInteger words =
        p->kind == DS4_TAIL_CARRY                                    ? 5u
        : (p->kind == DS4_TAIL_BLOCKS || p->kind == DS4_TAIL_FILTER) ? 4u
                                                                     : 3u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
      return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:p->args length:words * sizeof(uint32_t) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)p->first.buffer
            offset:p->first.offset
           atIndex:1];
    if (p->kind != DS4_TAIL_QUANTIZE)
      [enc setBuffer:(__bridge id<MTLBuffer>)p->second.buffer
              offset:p->second.offset
             atIndex:2];
    if (p->kind == DS4_TAIL_BLOCKS || p->kind == DS4_TAIL_FILTER)
      [enc setBuffer:(__bridge id<MTLBuffer>)p->third.buffer
              offset:p->third.offset
             atIndex:3];
    if (p->dispatch_threads)
      [enc dispatchThreads:MTLSizeMake(p->x, p->y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    else
      [enc dispatchThreadgroups:MTLSizeMake(p->x, p->y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}
