/* Fixed ARC/PSO and named buffer bindings only; host policy is native. */
ds4_metal_attention_pipeline ds4_metal_primitive_pipeline(uint32_t kind) {
  @autoreleasepool {
    id<MTLComputePipelineState> p = nil;
    switch (kind) {
    case DS4_PRIM_FP8:
      p = g_dsv4_fp8_kv_quantize_pipeline;
      break;
    case DS4_PRIM_QAT:
      p = g_dsv4_indexer_qat_pipeline;
      break;
    case DS4_PRIM_FP8_STORE:
      p = g_dsv4_kv_fp8_store_pipeline;
      break;
    case DS4_PRIM_ROPE_STORE:
      p = g_dsv4_kv_rope_fp8_store_pipeline;
      break;
    case DS4_PRIM_ADD:
      p = g_add2_pipeline;
      break;
    case DS4_PRIM_ADD3:
      p = g_add3_pipeline;
      break;
    case DS4_PRIM_SWIGLU:
      p = g_swiglu_flat_pipeline;
      break;
    case DS4_PRIM_COPY_F32:
      p = g_cpy_f32_f32_pipeline;
      break;
    case DS4_PRIM_COPY_F32_F16:
      p = g_cpy_f32_f16_pipeline;
      break;
    case DS4_PRIM_COPY_F16:
      p = g_cpy_f16_f16_pipeline;
      break;
    case DS4_PRIM_COPY_F16_F32:
      p = g_cpy_f16_f32_pipeline;
      break;
    case DS4_PRIM_COPY_F32_F16_CONTIG:
      p = g_cpy_contig_f32_f16_pipeline;
      break;
    case DS4_PRIM_COPY_F16_F32_CONTIG:
      p = g_cpy_contig_f16_f32_pipeline;
      break;
    case DS4_PRIM_COPY_F16_CONTIG:
      p = g_cpy_contig_f16_f16_pipeline;
      break;
    }
    return (ds4_metal_attention_pipeline){
        (const ds4_metal_hc_pipeline *)(__bridge void *)p,
        p ? (uint64_t)p.maxTotalThreadsPerThreadgroup : 0,
        p ? (uint64_t)p.threadExecutionWidth : 0};
  }
}
ds4_metal_attention_view ds4_metal_primitive_round_scratch(bool half,
                                                           uint64_t bytes) {
  @autoreleasepool {
    id<MTLBuffer> __strong *buffer =
        half ? &g_f16_round_scratch_buffer : &g_raw_store_round_buffer;
    NSUInteger *capacity =
        half ? &g_f16_round_scratch_bytes : &g_raw_store_round_bytes;
    const char *label = half ? "ds4_f16_round_scratch" : "ds4_raw_store_round";
    if (!ds4_gpu_ensure_scratch_buffer(buffer, capacity, (NSUInteger)bytes,
                                       label))
      return (ds4_metal_attention_view){0};
    return (ds4_metal_attention_view){
        (ds4_metal_model_buffer *)(__bridge_retained void *)*buffer, 0,
        *capacity};
  }
}
int ds4_metal_primitive_copy(void *command, uint32_t kind,
                             ds4_metal_attention_pipeline p,
                             const ds4_metal_copy_args *args, uint32_t count,
                             ds4_metal_attention_view source,
                             ds4_metal_attention_view target, uint64_t groups_x,
                             uint64_t groups_y, uint64_t threads) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)source.buffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)target.buffer;
    if (!cb || !src || !dst)
      return 0;
    if (kind == DS4_PRIM_COPY_F16_BLIT) {
      if (g_batch_cb && cb == g_batch_cb)
        ds4_gpu_close_batch_encoder();
      id<MTLBlitCommandEncoder> blit = ds4_gpu_blit_encoder(cb, "f16_copy", 1u);
      if (!blit)
        return 0;
      [blit copyFromBuffer:src
               sourceOffset:(NSUInteger)source.offset
                   toBuffer:dst
          destinationOffset:(NSUInteger)target.offset
                       size:(NSUInteger)count * sizeof(uint16_t)];
      [blit endEncoding];
      return 1;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)p.handle;
    if (!pipeline)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
      return 0;
    [enc setComputePipelineState:pipeline];
    const bool contiguous = kind == DS4_PRIM_COPY_F32_F16_CONTIG ||
                            kind == DS4_PRIM_COPY_F16_F32_CONTIG ||
                            kind == DS4_PRIM_COPY_F16_CONTIG;
    if (contiguous)
      [enc setBytes:&count length:sizeof(count) atIndex:0];
    else
      [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src offset:(NSUInteger)source.offset atIndex:1];
    [enc setBuffer:dst offset:(NSUInteger)target.offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}
int ds4_metal_primitive_encode(void *command,
                               const ds4_metal_primitive_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)p->pipeline.handle;
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)p->input.buffer;
    if (!cb || !pipeline || !input)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
      return 0;
    [enc setComputePipelineState:pipeline];
#define B(slot, field)                                                         \
  [enc setBuffer:(__bridge id<MTLBuffer>)p->field.buffer                       \
          offset:p->field.offset                                               \
         atIndex:slot]
    switch (p->kind) {
    case DS4_PRIM_FP8:
      [enc setBytes:&p->args.fp8 length:sizeof(p->args.fp8) atIndex:0];
      B(1, input);
      B(2, input);
      break;
    case DS4_PRIM_QAT:
      [enc setBytes:&p->args.qat length:sizeof(p->args.qat) atIndex:0];
      B(1, input);
      break;
    case DS4_PRIM_FP8_STORE:
      [enc setBytes:&p->args.store length:sizeof(p->args.store) atIndex:0];
      B(1, input);
      B(2, out);
      break;
    case DS4_PRIM_ROPE_STORE:
      [enc setBytes:&p->args.store length:sizeof(p->args.store) atIndex:0];
      [enc setBytes:&p->rope length:sizeof(p->rope) atIndex:1];
      B(2, input);
      B(3, out);
      break;
    case DS4_PRIM_SWIGLU:
      [enc setBytes:&p->args.glu length:sizeof(p->args.glu) atIndex:0];
      B(1, input);
      B(2, other);
      B(3, out);
      break;
    case DS4_PRIM_ADD:
    case DS4_PRIM_ADD3:
      [enc setBytes:&p->args.add length:sizeof(p->args.add) atIndex:0];
      B(1, input);
      B(2, other);
      if (p->kind == DS4_PRIM_ADD3) {
        B(3, third);
        B(4, out);
      } else {
        B(3, out);
      }
      break;
    default:
      return 0;
    }
#undef B
    if (p->shared)
      [enc setThreadgroupMemoryLength:p->shared atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(p->groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}
