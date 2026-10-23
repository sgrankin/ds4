/* ARC-only embedding resources and fixed shader bindings. All shape/range,
 * exact-row policy, packet construction and launch geometry live in Zig. */
ds4_metal_embedding_pipeline ds4_metal_embedding_pipeline_get(uint32_t kind) {
  @autoreleasepool {
    id<MTLComputePipelineState> pipeline = nil;
    switch (kind) {
    case DS4_EMBED_F16:
      pipeline = g_get_rows_f16_pipeline;
      break;
    case DS4_EMBED_Q8:
      pipeline = g_get_rows_q8_0_pipeline;
      break;
    case DS4_EMBED_Q4_0:
      pipeline = g_get_rows_q4_0_pipeline;
      break;
    case DS4_EMBED_Q4_K:
      pipeline = g_get_rows_q4_K_pipeline;
      break;
    case DS4_EMBED_REPEAT:
      pipeline = g_repeat_f32_pipeline;
      break;
    case DS4_EMBED_BF16:
      pipeline = ds4_gpu_get_pipeline("kernel_glm53_embedding_bf16");
      break;
    case DS4_EMBED_F32_GATHER:
      pipeline = g_get_rows_f32_pipeline;
      break;
    case DS4_EMBED_PACKED_GATHER:
      pipeline = ds4_gpu_get_pipeline("kernel_dsv41_gather_kv_f16_packed");
      break;
    }
    return (ds4_metal_embedding_pipeline){
        (const ds4_metal_hc_pipeline *)(__bridge void *)pipeline,
        pipeline ? (uint64_t)pipeline.maxTotalThreadsPerThreadgroup : 0};
  }
}
ds4_metal_attention_view ds4_metal_embedding_scratch(uint64_t bytes) {
  @autoreleasepool {
    if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                       &g_embed_rows_bytes, (NSUInteger)bytes,
                                       "ds4_embed_rows"))
      return (ds4_metal_attention_view){0};
    return (ds4_metal_attention_view){
        (ds4_metal_model_buffer *)(__bridge_retained void *)g_embed_rows_buffer,
        0, (uint64_t)g_embed_rows_bytes};
  }
}
ds4_metal_model_buffer *ds4_metal_embedding_exact_view(const void *model,
                                                       uint64_t size,
                                                       uint64_t offset,
                                                       uint64_t bytes,
                                                       uint64_t *inner) {
  @autoreleasepool {
    id<MTLBuffer> buffer =
        ds4_gpu_wrap_model_exact_range(model, size, offset, bytes, inner);
    /* The exact-view cache lends its buffer. Give this lexical native owner
     * a separate +1, just as retain_model_range does for ordinary views. */
    return (ds4_metal_model_buffer *)(__bridge_retained void *)buffer;
  }
}
int ds4_metal_embedding_encode(void *command,
                               const ds4_metal_embedding_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)p->pipeline;
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)p->source.buffer;
    id<MTLBuffer> ids = (__bridge id<MTLBuffer>)p->ids.buffer;
    id<MTLBuffer> out = (__bridge id<MTLBuffer>)p->out.buffer;
    if (!cb || !pipeline || !source || !out)
      return 0;
    if (p->kind != DS4_EMBED_REPEAT && !ids && !p->inline_token)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    if (!enc)
      return 0;
    [enc setComputePipelineState:pipeline];
    switch (p->kind) {
    case DS4_EMBED_F16:
    case DS4_EMBED_F32_GATHER:
      [enc setBytes:&p->args.rows length:sizeof(p->args.rows) atIndex:0];
      break;
    case DS4_EMBED_Q8:
    case DS4_EMBED_Q4_0:
    case DS4_EMBED_Q4_K:
      [enc setBytes:&p->args.quant length:sizeof(p->args.quant) atIndex:0];
      break;
    case DS4_EMBED_REPEAT:
      [enc setBytes:&p->args.repeat length:sizeof(p->args.repeat) atIndex:0];
      break;
    case DS4_EMBED_BF16:
      [enc setBytes:&p->args.bf16 length:sizeof(p->args.bf16) atIndex:0];
      break;
    case DS4_EMBED_PACKED_GATHER:
      [enc setBytes:p->args.packed_rows
             length:sizeof(p->args.packed_rows)
            atIndex:0];
      break;
    default:
      return 0;
    }
    [enc setBuffer:source offset:(NSUInteger)p->source.offset atIndex:1];
    if (p->kind == DS4_EMBED_REPEAT) {
      [enc setBuffer:out offset:(NSUInteger)p->out.offset atIndex:2];
    } else {
      if (p->inline_token)
        [enc setBytes:&p->token length:sizeof(p->token) atIndex:2];
      else
        [enc setBuffer:ids offset:(NSUInteger)p->ids.offset atIndex:2];
      [enc setBuffer:out offset:(NSUInteger)p->out.offset atIndex:3];
    }
    if (p->dispatch_threads)
      [enc dispatchThreads:MTLSizeMake(p->groups_x, p->groups_y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    else
      [enc dispatchThreadgroups:MTLSizeMake(p->groups_x, p->groups_y, 1)
          threadsPerThreadgroup:MTLSizeMake(p->threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
  }
}
