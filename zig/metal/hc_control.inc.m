/* Fixed ARC resource factories and shader bindings. Native HC owns all packet,
 * shape, tuning, generation and dispatch decisions. */
#include "hc_control.h"
ds4_metal_attention_pipeline ds4_metal_hc_control_pipeline(uint32_t kind) {
  @autoreleasepool {
    id<MTLComputePipelineState> p = nil;
    switch (kind) {
    case DS4_HC_SPLIT:
      p = g_hc_split_sinkhorn_pipeline;
      break;
    case DS4_HC_SUM:
      p = g_hc_weighted_sum_pipeline;
      break;
    case DS4_HC_SPLIT_SUM:
      p = g_hc_split_weighted_sum_pipeline;
      break;
    case DS4_HC_SPLIT_NORM:
      p = ds4_gpu_hot_pipeline(g_hc_split_weighted_sum_norm_pipeline,
                               "kernel_dsv4_hc_split_weighted_sum_norm4");
      break;
    case DS4_HC_NORM_MIX:
      p = ds4_gpu_get_pipeline("kernel_dsv4_hc_rms_norm_mix_f16");
      break;
    case DS4_HC_NORM_MIX_CLUSTER:
      p = ds4_gpu_get_pipeline("kernel_dsv4_hc_rms_norm_mix_f16_cluster2");
      break;
    case DS4_HC_PRODUCER:
      if (!g_dsv4_hc_producer_pre_norm_pipeline)
        g_dsv4_hc_producer_pre_norm_pipeline = ds4_gpu_get_pipeline(
            "kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm");
      p = g_dsv4_hc_producer_pre_norm_pipeline;
      break;
    case DS4_HC_EXPAND_PRODUCER:
      if (!g_dsv4_hc_expand_producer_pre_norm_pipeline)
        g_dsv4_hc_expand_producer_pre_norm_pipeline = ds4_gpu_get_pipeline(
            "kernel_dsv4_hc_expand4_rms_norm_mix_f16_cluster2_pre_norm");
      p = g_dsv4_hc_expand_producer_pre_norm_pipeline;
      break;
    case DS4_HC_OUTPUT4:
      p = g_output_hc_weights4_pipeline;
      break;
    case DS4_HC_OUTPUT_MUL:
      p = g_bin_mul_scalar_pipeline;
      break;
    case DS4_HC_OUTPUT_ADD:
      p = g_add_pipeline;
      break;
    case DS4_HC_OUTPUT_SIGMOID:
      p = g_unary_sigmoid_pipeline;
      break;
    case DS4_HC_OUTPUT_SCALE:
      p = g_unary_scale_pipeline;
      break;
    case DS4_HC_EXPAND:
      p = g_hc_expand_pipeline;
      break;
    case DS4_HC_EXPAND4:
      p = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                               "kernel_dsv4_hc_expand4");
      break;
    }
    return (ds4_metal_attention_pipeline){
        (__bridge const ds4_metal_hc_pipeline *)p,
        (uint64_t)p.maxTotalThreadsPerThreadgroup,
        (uint64_t)p.threadExecutionWidth};
  }
}
bool ds4_metal_hc_control_output4_test(void) {
  return (g_test_flags & DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4) != 0u;
}
bool ds4_metal_hc_control_tp_two(void) {
  return ds4_gpu_tp_world_is_two() != 0;
}
bool ds4_metal_hc_control_completion(ds4_metal_attention_view mix,
                                     bool expanded,
                                     ds4_metal_hc_completion *out) {
  @autoreleasepool {
    if (!g_dsv4_completion_cache)
      return false;
    id<MTLBuffer> mixbuf = (__bridge id<MTLBuffer>)mix.buffer;
    id<MTLBuffer> completion = nil;
    if (g_dsv4_hc_producer_last_mix_buffer == mixbuf &&
        g_dsv4_hc_producer_last_mix_offset == (NSUInteger)mix.offset) {
      completion = g_dsv4_hc_producer_last_completion;
    } else {
      NSString *key =
          [NSString stringWithFormat:@"%p:%llu", (void *)mixbuf,
                                     (unsigned long long)mix.offset];
      completion = [g_dsv4_completion_cache objectForKey:key];
      if (!completion) {
        completion =
            [g_device newBufferWithLength:sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared];
        if (!completion)
          return false;
        *((uint32_t *)[completion contents]) = 0u;
        [g_dsv4_completion_cache setObject:completion forKey:key];
      }
      g_dsv4_hc_producer_last_mix_buffer = mixbuf;
      g_dsv4_hc_producer_last_mix_offset = (NSUInteger)mix.offset;
      g_dsv4_hc_producer_last_completion = completion;
    }
    [g_transient_buffers addObject:completion];
    id<MTLBuffer> barrier = nil;
    uint32_t generation = 0;
    if (expanded) {
      NSString *key =
          [NSString stringWithFormat:@"%p:%llu", (void *)mixbuf,
                                     (unsigned long long)mix.offset];
      if (!g_dsv4_hc_barrier_cache) {
        g_dsv4_hc_barrier_cache = [NSMutableDictionary dictionary];
        g_dsv4_hc_barrier_gen = [NSMutableDictionary dictionary];
      }
      barrier = [g_dsv4_hc_barrier_cache objectForKey:key];
      if (!barrier) {
        barrier = [g_device newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
        if (!barrier)
          return false;
        *((uint32_t *)[barrier contents]) = 0u;
        [g_dsv4_hc_barrier_cache setObject:barrier forKey:key];
      }
      generation =
          (uint32_t)[[g_dsv4_hc_barrier_gen objectForKey:key] unsignedIntValue];
    }
    *out = (ds4_metal_hc_completion){
        .completion = {(ds4_metal_model_buffer *)(__bridge_retained void *)
                           completion,
                       0, sizeof(uint32_t)},
        .barrier = {(ds4_metal_model_buffer *)(__bridge_retained void *)barrier,
                    0, barrier ? sizeof(uint32_t) : 0},
        .generation = generation};
    return true;
  }
}
void ds4_metal_hc_control_generation(ds4_metal_attention_view mix,
                                     uint32_t generation) {
  @autoreleasepool {
    NSString *key = [NSString stringWithFormat:@"%p:%llu", (void *)mix.buffer,
                                               (unsigned long long)mix.offset];
    [g_dsv4_hc_barrier_gen setObject:@(generation) forKey:key];
    [g_transient_buffers addObject:[g_dsv4_hc_barrier_cache objectForKey:key]];
  }
}
int ds4_metal_hc_control_encode(void *command,
                                const ds4_metal_hc_control_launch *p) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command;
    if (!cb)
      return 0;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
#define PSO(P)                                                                 \
  [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)          \
                                   p->P.handle]
#define A(I, F) [enc setBytes:&p->F length:sizeof(p->F) atIndex:I]
#define B(I, F)                                                                \
  [enc setBuffer:(__bridge id<MTLBuffer>)p->F.buffer                           \
          offset:(NSUInteger)p->F.offset                                       \
         atIndex:I]
    if (p->kind != DS4_HC_OUTPUT_STAGES) {
      PSO(pipeline);
    }
    switch (p->kind) {
    case DS4_HC_SPLIT:
      A(0, split_args);
      B(1, mix);
      B(2, scale);
      B(3, base);
      B(4, out);
      break;
    case DS4_HC_SUM:
      A(0, sum_args);
      B(1, residual);
      B(2, post);
      B(3, out);
      break;
    case DS4_HC_SPLIT_SUM:
      A(0, split_sum_args);
      B(1, mix);
      B(2, scale);
      B(3, base);
      B(4, residual);
      B(5, split);
      B(6, out);
      break;
    case DS4_HC_SPLIT_NORM:
      A(0, split_norm_args);
      B(1, mix);
      B(2, scale);
      B(3, base);
      B(4, residual);
      B(5, split);
      B(6, out);
      B(7, norm_weights);
      B(8, norm_out);
      break;
    case DS4_HC_NORM_MIX:
      A(0, norm_mix_args);
      B(1, input);
      B(2, weights);
      B(3, out);
      break;
    case DS4_HC_PRODUCER:
      A(0, norm_mix_args);
      A(1, split_norm_args);
      B(2, residual);
      B(3, weights);
      B(4, mix);
      B(5, scale);
      B(6, base);
      B(7, split);
      B(8, out);
      B(9, norm_weights);
      B(10, norm_out);
      B(11, completion);
      break;
    case DS4_HC_EXPAND_PRODUCER:
      A(0, norm_mix_args);
      A(1, split_norm_args);
      A(2, expand_args);
      A(3, barrier_target);
      B(4, residual);
      B(5, weights);
      B(6, mix);
      B(7, scale);
      B(8, base);
      B(9, split);
      B(10, out);
      B(11, norm_weights);
      B(12, norm_out);
      B(13, completion);
      B(14, barrier);
      B(15, block);
      B(16, block_add);
      B(17, previous);
      B(18, post);
      B(19, comb);
      break;
    case DS4_HC_OUTPUT4:
      A(0, output4_args);
      B(1, input);
      B(2, scale);
      B(3, base);
      B(4, out);
      break;
    case DS4_HC_EXPAND:
      A(0, expand_args);
      B(1, block);
      B(2, residual);
      B(3, post);
      B(4, comb);
      B(5, block_add);
      B(6, out);
      break;
    case DS4_HC_SHARED_DOWN:
      A(0, mv_args);
      A(1, expand_args);
      B(2, weights);
      B(3, input);
      B(4, block);
      B(5, block_add);
      B(6, residual);
      B(7, post);
      B(8, comb);
      B(9, out);
      break;
    case DS4_HC_MATMUL:
      A(0, mv_args);
      A(1, expand_args);
      B(2, weights);
      B(3, input);
      B(4, block);
      B(5, residual);
      B(6, post);
      B(7, comb);
      B(8, out);
      break;
    case DS4_HC_OUTPUT_STAGES:
      PSO(mul);
      A(0, mul_args);
      B(1, input);
      B(2, scale);
      B(3, out);
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p->mul_args.ne01,
                                            (NSUInteger)p->mul_args.ne02,
                                            (NSUInteger)p->mul_args.ne03)
          threadsPerThreadgroup:MTLSizeMake(p->mul_threads, 1, 1)];
      PSO(add_pipeline);
      A(0, add_args);
      B(1, out);
      B(2, base);
      B(3, out);
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p->add_args.ne01,
                                            (NSUInteger)p->add_args.ne02,
                                            (NSUInteger)p->add_args.ne03)
          threadsPerThreadgroup:MTLSizeMake(p->add_threads, 1, 1)];
      PSO(sigmoid);
      A(0, sigmoid_args);
      B(1, out);
      B(2, out);
      [enc dispatchThreadgroups:MTLSizeMake(p->unary_groups,
                                            (NSUInteger)p->sigmoid_args.ne02,
                                            (NSUInteger)p->sigmoid_args.ne03)
          threadsPerThreadgroup:MTLSizeMake(p->unary_threads, 1, 1)];
      PSO(scale_pipeline);
      A(0, scale_args);
      B(1, out);
      B(2, out);
      [enc dispatchThreadgroups:MTLSizeMake(p->unary_groups,
                                            (NSUInteger)p->scale_args.ne02,
                                            (NSUInteger)p->scale_args.ne03)
          threadsPerThreadgroup:MTLSizeMake(p->unary_threads, 1, 1)];
      ds4_gpu_end_compute_encoder(cb, enc);
      return 1;
    default:
      ds4_gpu_end_compute_encoder(cb, enc);
      return 0;
    }
    if (p->shared_bytes)
      [enc setThreadgroupMemoryLength:(NSUInteger)p->shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(p->gx, p->gy, p->gz)
        threadsPerThreadgroup:MTLSizeMake(p->tx, p->ty, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
#undef PSO
#undef A
#undef B
    return 1;
  }
}
