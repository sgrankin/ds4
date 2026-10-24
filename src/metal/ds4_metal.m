#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <execinfo.h>

#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/resource.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <objc/runtime.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "bootstrap.h"
#include "tp_prefetch.h"
#include "tp_control.h"
#include "primitives.h"
#include "router.h"
#include "router_shared.h"
#include "hc_control.h"
#include "embedding.h"
#include "supported_primitives.h"
#include "scheduling.h"
#include "compressor.h"
#include "model_views.h"
#include "exact_cache.h"
#include "bridge.h"
#include "index_selection.h"
#include "dense_matmul.h"
#include "attention_decode.h"
#include "attention_batch.h"
#include "routed_batch.h"
#include "attention_output.h"
#include "rope.h"
#include "expert_load.h"
#include "routed_single.h"
#include "resources.h"
#include "ds4_image.h"

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

/* Tensor type tags are shared in routed_batch.h. */

@class DS4MetalQ4ExpertTable;

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static ds4_metal_scheduling g_schedule;
#define g_batch_cb g_schedule.commands.batch
#define g_batch_enc g_schedule.commands.encoder
#define g_batch_encoder_concurrent g_schedule.commands.concurrent
#define g_batch_has_work g_schedule.commands.has_work

/* ---- Encoder timeline diagnostic (DS4_METAL_ENCODER_TIMELINE=<file>) ----
 * Gives every batch dispatch group its own compute pass with stage-boundary
 * GPU timestamps, so one decode token can be read as a per-kernel GPU
 * timeline (durations and the gaps between them) without ending command
 * buffers. Pass boundaries add a little serialization, so absolute numbers
 * are an upper bound; the anatomy is what this exists for. Off by default. */
#define DS4_TIMELINE_NAME_MAX 80
#define DS4_TIMELINE_SAMPLES_PER_BUFFER 4096
typedef struct {
    uintptr_t caller;
    uint32_t n_dispatch;
    uint32_t tg[3];
    uint32_t tpt[3];
    char kernel[DS4_TIMELINE_NAME_MAX];
} ds4_timeline_rec;

@interface DS4TimelineBatch : NSObject {
@public
    NSMutableArray<id<MTLCounterSampleBuffer>> *samples;
    ds4_timeline_rec *recs;
    uint32_t count;
    uint32_t cap;
    uint64_t seq;
}
@end
@implementation DS4TimelineBatch
- (void)dealloc { free(recs); }
@end

static BOOL g_timeline_enabled;
static FILE *g_timeline_file;
static pthread_mutex_t g_timeline_mutex = PTHREAD_MUTEX_INITIALIZER;
static DS4TimelineBatch *g_timeline_batch;
static uint64_t g_timeline_seq;
static NSMutableDictionary<NSNumber *, NSString *> *g_timeline_pso_names;
static id<MTLCounterSet> g_timeline_counter_set;
static BOOL g_timeline_encoder_hooked;

static void ds4_gpu_timeline_note_pso_name(id pso, NSString *name) {
    if (!g_timeline_enabled || !pso || !name) return;
    pthread_mutex_lock(&g_timeline_mutex);
    if (!g_timeline_pso_names) g_timeline_pso_names = [NSMutableDictionary dictionary];
    g_timeline_pso_names[@((uintptr_t)pso)] = name;
    pthread_mutex_unlock(&g_timeline_mutex);
}

@interface NSObject (DS4TimelineSwizzle)
- (id)ds4_tl_newComputePipelineStateWithFunction:(id<MTLFunction>)fn
                                            error:(NSError * __autoreleasing *)error NS_RETURNS_RETAINED;
- (id)ds4_tl_newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            options:(MTLPipelineOption)options
                                         reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                              error:(NSError * __autoreleasing *)error NS_RETURNS_RETAINED;
- (void)ds4_tl_setComputePipelineState:(id<MTLComputePipelineState>)pso;
- (void)ds4_tl_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt;
- (void)ds4_tl_dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tpt;
@end

static ds4_timeline_rec *ds4_gpu_timeline_current_rec(void) {
    DS4TimelineBatch *b = g_timeline_batch;
    if (!b || b->count == 0) return NULL;
    return &b->recs[b->count - 1];
}

@implementation NSObject (DS4TimelineSwizzle)
- (id)ds4_tl_newComputePipelineStateWithFunction:(id<MTLFunction>)fn
                                            error:(NSError * __autoreleasing *)error {
    id pso = [self ds4_tl_newComputePipelineStateWithFunction:fn error:error];
    ds4_gpu_timeline_note_pso_name(pso, [fn name]);
    return pso;
}
- (id)ds4_tl_newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            options:(MTLPipelineOption)options
                                         reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                              error:(NSError * __autoreleasing *)error {
    id pso = [self ds4_tl_newComputePipelineStateWithDescriptor:descriptor
                                                        options:options
                                                     reflection:reflection
                                                          error:error];
    ds4_gpu_timeline_note_pso_name(pso, [[descriptor computeFunction] name]);
    return pso;
}
- (void)ds4_tl_setComputePipelineState:(id<MTLComputePipelineState>)pso {
    [self ds4_tl_setComputePipelineState:pso];
    ds4_timeline_rec *rec = ds4_gpu_timeline_current_rec();
    if (!rec) return;
    NSString *name = nil;
    pthread_mutex_lock(&g_timeline_mutex);
    if (g_timeline_pso_names) name = g_timeline_pso_names[@((uintptr_t)pso)];
    pthread_mutex_unlock(&g_timeline_mutex);
    const char *cname = name ? [name UTF8String] : "?";
    if (rec->kernel[0] == '\0') {
        snprintf(rec->kernel, sizeof(rec->kernel), "%s", cname);
    } else if (strstr(rec->kernel, cname) == NULL) {
        size_t used = strlen(rec->kernel);
        if (used + 1 < sizeof(rec->kernel))
            snprintf(rec->kernel + used, sizeof(rec->kernel) - used, "+%s", cname);
    }
}
- (void)ds4_tl_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    [self ds4_tl_dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
    ds4_timeline_rec *rec = ds4_gpu_timeline_current_rec();
    if (!rec) return;
    if (rec->n_dispatch == 0) {
        rec->tg[0] = (uint32_t)tg.width; rec->tg[1] = (uint32_t)tg.height; rec->tg[2] = (uint32_t)tg.depth;
        rec->tpt[0] = (uint32_t)tpt.width; rec->tpt[1] = (uint32_t)tpt.height; rec->tpt[2] = (uint32_t)tpt.depth;
    }
    rec->n_dispatch++;
}
- (void)ds4_tl_dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tpt {
    [self ds4_tl_dispatchThreads:threads threadsPerThreadgroup:tpt];
    ds4_timeline_rec *rec = ds4_gpu_timeline_current_rec();
    if (!rec) return;
    if (rec->n_dispatch == 0) {
        rec->tg[0] = (uint32_t)((threads.width + tpt.width - 1) / (tpt.width ? tpt.width : 1));
        rec->tg[1] = (uint32_t)((threads.height + tpt.height - 1) / (tpt.height ? tpt.height : 1));
        rec->tg[2] = (uint32_t)((threads.depth + tpt.depth - 1) / (tpt.depth ? tpt.depth : 1));
        rec->tpt[0] = (uint32_t)tpt.width; rec->tpt[1] = (uint32_t)tpt.height; rec->tpt[2] = (uint32_t)tpt.depth;
    }
    rec->n_dispatch++;
}
@end

static void ds4_gpu_timeline_swizzle(Class cls, SEL orig, SEL repl) {
    Method m_orig = class_getInstanceMethod(cls, orig);
    Method m_repl = class_getInstanceMethod(cls, repl);
    if (!m_orig || !m_repl) {
        fprintf(stderr, "ds4: timeline hook missing for %s\n", sel_getName(orig));
        return;
    }
    if (class_addMethod(cls, orig, method_getImplementation(m_repl),
                        method_getTypeEncoding(m_orig))) {
        class_replaceMethod(cls, repl, method_getImplementation(m_orig),
                            method_getTypeEncoding(m_orig));
    } else {
        method_exchangeImplementations(m_orig, m_repl);
    }
}

/* Called once the device exists: opens the log and hooks pipeline creation so
 * every pipeline made afterwards has a kernel name. */
static void ds4_gpu_timeline_probe(id<MTLDevice> device) {
    static int probed;
    if (probed) return;
    probed = 1;
    const char *path = getenv("DS4_METAL_ENCODER_TIMELINE");
    if (!path || !path[0] || !device) return;
    g_timeline_file = fopen(path, "a");
    if (!g_timeline_file) {
        fprintf(stderr, "ds4: cannot open encoder timeline %s\n", path);
        return;
    }
    for (id<MTLCounterSet> cs in [device counterSets]) {
        if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) g_timeline_counter_set = cs;
    }
    if (!g_timeline_counter_set ||
        ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        fprintf(stderr, "ds4: encoder timeline unsupported on this device\n");
        fclose(g_timeline_file);
        g_timeline_file = NULL;
        return;
    }
    g_timeline_enabled = YES;
    Class dev_cls = object_getClass(device);
    ds4_gpu_timeline_swizzle(dev_cls,
        @selector(newComputePipelineStateWithFunction:error:),
        @selector(ds4_tl_newComputePipelineStateWithFunction:error:));
    ds4_gpu_timeline_swizzle(dev_cls,
        @selector(newComputePipelineStateWithDescriptor:options:reflection:error:),
        @selector(ds4_tl_newComputePipelineStateWithDescriptor:options:reflection:error:));
    fprintf(g_timeline_file, "# ds4 encoder timeline; slide=0x%llx pid=%d\n",
            (unsigned long long)_dyld_get_image_vmaddr_slide(0), (int)getpid());
    fprintf(g_timeline_file, "# B <seq> <n_encoders> <gpu_start_ns> <gpu_end_ns>\n");
    fprintf(g_timeline_file, "# E <seq> <idx> <start_ns> <end_ns> <dur_us> <gap_us> <caller_unslid> <n_dispatch> <tg> <tpt> <kernel>\n");
    fflush(g_timeline_file);
    fprintf(stderr, "ds4: encoder timeline -> %s\n", path);
}

static void ds4_gpu_timeline_hook_encoder(id<MTLComputeCommandEncoder> enc) {
    if (g_timeline_encoder_hooked || !enc) return;
    g_timeline_encoder_hooked = YES;
    Class cls = object_getClass(enc);
    ds4_gpu_timeline_swizzle(cls, @selector(setComputePipelineState:),
                             @selector(ds4_tl_setComputePipelineState:));
    ds4_gpu_timeline_swizzle(cls, @selector(dispatchThreadgroups:threadsPerThreadgroup:),
                             @selector(ds4_tl_dispatchThreadgroups:threadsPerThreadgroup:));
    ds4_gpu_timeline_swizzle(cls, @selector(dispatchThreads:threadsPerThreadgroup:),
                             @selector(ds4_tl_dispatchThreads:threadsPerThreadgroup:));
}

static void ds4_gpu_timeline_resolve(DS4TimelineBatch *b, id<MTLCommandBuffer> cb) {
    if (!b || !g_timeline_file) return;
    pthread_mutex_lock(&g_timeline_mutex);
    fprintf(g_timeline_file, "B %llu %u %.0f %.0f\n", (unsigned long long)b->seq, b->count,
            cb.GPUStartTime * 1e9, cb.GPUEndTime * 1e9);
    uint64_t prev_end = 0;
    for (uint32_t i = 0; i < b->count; i++) {
        const uint32_t sb_index = i / (DS4_TIMELINE_SAMPLES_PER_BUFFER / 2);
        const uint32_t slot = (i % (DS4_TIMELINE_SAMPLES_PER_BUFFER / 2)) * 2u;
        uint64_t start = 0, end = 0;
        if (sb_index < [b->samples count]) {
            NSData *d = [b->samples[sb_index] resolveCounterRange:NSMakeRange(slot, 2)];
            if (d && [d length] >= 2 * sizeof(uint64_t)) {
                const uint64_t *ts = (const uint64_t *)[d bytes];
                start = ts[0];
                end = ts[1];
            }
        }
        const ds4_timeline_rec *r = &b->recs[i];
        const double dur_us = (end >= start) ? (double)(end - start) / 1000.0 : -1.0;
        const double gap_us = (prev_end && start >= prev_end) ? (double)(start - prev_end) / 1000.0 : 0.0;
        fprintf(g_timeline_file, "E %llu %u %llu %llu %.2f %.2f 0x%llx %u %ux%ux%u %ux%ux%u %s\n",
                (unsigned long long)b->seq, i,
                (unsigned long long)start, (unsigned long long)end, dur_us, gap_us,
                (unsigned long long)(r->caller - (uintptr_t)_dyld_get_image_vmaddr_slide(0)),
                r->n_dispatch, r->tg[0], r->tg[1], r->tg[2], r->tpt[0], r->tpt[1], r->tpt[2],
                r->kernel[0] ? r->kernel : "?");
        if (end) prev_end = end;
    }
    fflush(g_timeline_file);
    pthread_mutex_unlock(&g_timeline_mutex);
}

/* Attach to a freshly created batch command buffer. */
static void ds4_gpu_timeline_attach(id<MTLCommandBuffer> cb) {
    if (!g_timeline_enabled || !cb) return;
    DS4TimelineBatch *b = [DS4TimelineBatch new];
    b->samples = [NSMutableArray array];
    b->cap = 1024;
    b->recs = calloc(b->cap, sizeof(ds4_timeline_rec));
    b->count = 0;
    b->seq = ++g_timeline_seq;
    g_timeline_batch = b;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        ds4_gpu_timeline_resolve(b, done);
    }];
}

static id<MTLComputeCommandEncoder> ds4_gpu_timeline_new_encoder(
        id<MTLCommandBuffer> cb, BOOL concurrent, uintptr_t caller) {
    DS4TimelineBatch *b = g_timeline_batch;
    if (!b) return nil;
    const uint32_t per_buffer = DS4_TIMELINE_SAMPLES_PER_BUFFER / 2;
    const uint32_t sb_index = b->count / per_buffer;
    if (sb_index >= [b->samples count]) {
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = g_timeline_counter_set;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = DS4_TIMELINE_SAMPLES_PER_BUFFER;
        NSError *error = nil;
        id<MTLCounterSampleBuffer> sb = [g_device newCounterSampleBufferWithDescriptor:d error:&error];
        if (!sb) {
            fprintf(stderr, "ds4: timeline sample buffer failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [b->samples addObject:sb];
    }
    if (b->count == b->cap) {
        b->cap *= 2;
        b->recs = realloc(b->recs, (size_t)b->cap * sizeof(ds4_timeline_rec));
    }
    const uint32_t slot = (b->count % per_buffer) * 2u;
    MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.dispatchType = concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
    MTLComputePassSampleBufferAttachmentDescriptor *att = pd.sampleBufferAttachments[0];
    att.sampleBuffer = b->samples[sb_index];
    att.startOfEncoderSampleIndex = slot;
    att.endOfEncoderSampleIndex = slot + 1u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pd];
    if (!enc) return nil;
    ds4_timeline_rec *rec = &b->recs[b->count++];
    memset(rec, 0, sizeof(*rec));
    rec->caller = caller;
    ds4_gpu_timeline_hook_encoder(enc);
    return enc;
}
/* Blit encoder with timeline sampling (blit passes are otherwise invisible
 * to the encoder timeline and show up as gaps). */
static id<MTLBlitCommandEncoder> ds4_gpu_blit_encoder(id<MTLCommandBuffer> cb, const char *label,
                                                     uint32_t n_ops) {
    DS4TimelineBatch *b = g_timeline_batch;
    if (!g_timeline_enabled || !b || cb != g_batch_cb) return [cb blitCommandEncoder];
    const uint32_t per_buffer = DS4_TIMELINE_SAMPLES_PER_BUFFER / 2;
    const uint32_t sb_index = b->count / per_buffer;
    if (sb_index >= [b->samples count]) {
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = g_timeline_counter_set;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = DS4_TIMELINE_SAMPLES_PER_BUFFER;
        NSError *error = nil;
        id<MTLCounterSampleBuffer> sb = [g_device newCounterSampleBufferWithDescriptor:d error:&error];
        if (!sb) return [cb blitCommandEncoder];
        [b->samples addObject:sb];
    }
    if (b->count == b->cap) {
        b->cap *= 2;
        b->recs = realloc(b->recs, (size_t)b->cap * sizeof(ds4_timeline_rec));
    }
    const uint32_t slot = (b->count % per_buffer) * 2u;
    MTLBlitPassDescriptor *pd = [MTLBlitPassDescriptor blitPassDescriptor];
    MTLBlitPassSampleBufferAttachmentDescriptor *att = pd.sampleBufferAttachments[0];
    att.sampleBuffer = b->samples[sb_index];
    att.startOfEncoderSampleIndex = slot;
    att.endOfEncoderSampleIndex = slot + 1u;
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoderWithDescriptor:pd];
    if (!enc) return [cb blitCommandEncoder];
    ds4_timeline_rec *rec = &b->recs[b->count++];
    memset(rec, 0, sizeof(*rec));
    rec->caller = (uintptr_t)__builtin_return_address(0);
    rec->n_dispatch = n_ops;
    snprintf(rec->kernel, sizeof(rec->kernel), "blit:%s", label ? label : "?");
    return enc;
}

static void ds4_gpu_parallel_ffn_reset_state(BOOL close_encoder);
#define g_pending_cbs g_schedule.commands.pending
static id<MTLSharedEvent> g_selected_readback_event;
static ds4_metal_tp_policy g_tp_native_policy;
#define g_tp_seq (g_tp_native_policy.row_sequence)
#define g_tp_batch_seq (g_tp_native_policy.batch_sequence)
#define g_tp_flag_prepublished_seq (g_tp_native_policy.prepublished_sequence)
#define g_tp_session_batch_mode (g_tp_native_policy.session_batch_mode)
#define g_tp_flag_gates (g_tp_native_policy.flag_gates)
#define g_tp_poll_gates (g_tp_native_policy.poll_gates)
#define g_tp_out_off (g_tp_native_policy.out_offset)
#define g_tp_vec_bytes (g_tp_native_policy.vector_bytes)
#define g_tp_slab_buffer_off (g_tp_native_policy.slab_offset)
#define g_tp_gpu_flags_off (g_tp_native_policy.flags_offset)
#define g_tp_fold_req_active (g_tp_native_policy.fold_requested)
#define g_tp_fold_req_layer (g_tp_native_policy.fold_layer)
#define g_tp_fold_req_gate (g_tp_native_policy.fold_gate)
#define g_selected_readback_event_value (g_tp_native_policy.selected_sequence)

static id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f16_pipeline;
static id<MTLComputePipelineState> g_get_rows_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_q8_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_K_pipeline;
static id<MTLComputePipelineState> g_repeat_f32_pipeline;
static id<MTLComputePipelineState> g_concat_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f16_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f16_f16_pipeline;
static id<MTLComputePipelineState> g_flash_kv_stage_f16_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_swiglu_flat_pipeline;
static id<MTLComputePipelineState> g_add_pipeline;
static id<MTLComputePipelineState> g_add2_pipeline;
static id<MTLComputePipelineState> g_add3_pipeline;
static id<MTLComputePipelineState> g_moe_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_sum8_pipeline;
static id<MTLComputePipelineState> g_mul_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_plain_pipeline;
static id<MTLComputePipelineState> g_add_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_scale_pipeline;
static id<MTLComputePipelineState> g_dsv4_qkv_rms_norm_pipeline;
static id<MTLComputePipelineState> g_dsv4_head_rms_norm_rope_tail_pipeline;
#define g_use_dsv4_head_rms_norm_rope_tail_pipeline g_schedule.commands.static_head_norm
static id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_norm_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_producer_pre_norm_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_expand_producer_pre_norm_pipeline;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_dsv4_hc_barrier_cache;
static NSMutableDictionary<NSString *, NSNumber *> *g_dsv4_hc_barrier_gen;
static id<MTLComputePipelineState> g_hc_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_output_hc_weights4_pipeline;
static uint32_t g_test_flags;
static id<MTLComputePipelineState> g_hc_expand_pipeline;
static id<MTLComputePipelineState> g_unary_sigmoid_pipeline;
static id<MTLComputePipelineState> g_unary_silu_pipeline;
static id<MTLComputePipelineState> g_unary_softplus_pipeline;
static id<MTLComputePipelineState> g_unary_sqrt_pipeline;
static id<MTLComputePipelineState> g_unary_clamp_pipeline;
static id<MTLComputePipelineState> g_unary_scale_pipeline;
static id<MTLComputePipelineState> g_unary_fill_pipeline;
static id<MTLComputePipelineState> g_unary_fill_f16_pipeline;
static id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;
static id<MTLComputePipelineState> g_bin_div_row_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group6_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group8_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group24_q4_k_id_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group24_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_mxfp4_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline;
static id<MTLComputePipelineState> g_moe_stream_expert_cache_validate_pipeline;
static id<MTLComputePipelineState> g_moe_q4_gather_slots6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_table_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q4_k_sum6_pipeline;
static id<MTLArgumentEncoder> g_moe_table_q4_pair_gate_encoder;
static id<MTLArgumentEncoder> g_moe_table_q4_pair_up_encoder;
static id<MTLArgumentEncoder> g_moe_table_q4_sum_down_encoder;
static id<MTLComputePipelineState> g_rope_tail_batch_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_shared4_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_affine_pipeline;
static id<MTLComputePipelineState> g_dsv4_fp8_kv_quantize_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_qat_pipeline;
static id<MTLComputePipelineState> g_dsv4_kv_fp8_store_pipeline;
static id<MTLComputePipelineState> g_dsv4_kv_rope_fp8_store_pipeline;
static id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_pack_ratio4_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_pack_ratio4_decode_ggml_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_exact_softmax_product_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_exact_pool_ratio4_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_ratio4_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_4_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_store_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_rb16_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads16_dual_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_split_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_split_reduce_pipeline;
static id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_weights_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_transform_finalize_weights_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_select_visual_batch_pipeline;
static NSCache<NSString *, id<MTLBuffer>> *g_dsv4_completion_cache;
static __weak id<MTLBuffer> g_dsv4_hc_producer_last_mix_buffer;
static NSUInteger g_dsv4_hc_producer_last_mix_offset;
static id<MTLBuffer> g_dsv4_hc_producer_last_completion;
static id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;
static id<MTLComputePipelineState> g_glm_router_select_one_pipeline;
static id<MTLComputePipelineState> g_glm_kv_lora_rms_norm_pipeline;
static id<MTLComputePipelineState> g_glm_k_b_project_pipeline;
static id<MTLComputePipelineState> g_glm_store_compact_kv_pipeline;
static id<MTLComputePipelineState> g_glm_qkv_norm_store_compact_kv_pipeline;
static id<MTLComputePipelineState> g_glm_store_indexer_k_pipeline;
static id<MTLComputePipelineState> g_glm53_indexer_pool_update_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_decode_group4_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_flash_pipeline;
static id<MTLComputePipelineState> g_glm_attention_full_pipeline;
static id<MTLComputePipelineState> g_glm_fill_selected_range_pipeline;
static id<MTLComputePipelineState> g_glm_fill_selected_range_batch_pipeline;
static id<MTLComputePipelineState> g_glm53_expand_pool_selection_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_rope_tail_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_score_one_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_batch_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_tiled_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_tiled_f32_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_glm52_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_glm52_sg_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_batch_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_batch_glm52_t4_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_batch_heads_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_batch_heads_mma_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_partial_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_reduce_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_reduce16_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_group2_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_q2_group4_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_group8_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu4_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_pair_swiglu2_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q6_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_batch_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_expand4_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;

enum {
    DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS = 64,
    DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES = 96,
};
/*
 * Sentinel nxpsg stored by the single-constant ds4_gpu_get_mul_mv_pipeline
 * path, which has no second SIMD-group constant. The extended
 * ds4_gpu_get_mul_mv_ext_pipeline path derives nxpsg as a positive divisor of
 * the 32-row output tile, so INT16_MIN can never collide with a real value.
 */
#define DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE INT16_MIN
_Static_assert(
    (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS &
     (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u)) == 0,
    "decode pipeline fast cache slot count must be a power of two");

typedef struct {
    id<MTLComputePipelineState> __strong pipeline;
    uint64_t hash;
    int16_t nsg;
    int16_t nxpsg;
    uint16_t name_len;
    bool used;
    char name[DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES];
} ds4_gpu_decode_pipeline_fast_cache_entry;

static ds4_gpu_decode_pipeline_fast_cache_entry
    g_decode_pipeline_fast_cache[DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS];
/* Command encoding and the existing NSMutableDictionary pipeline cache are
 * backend-serialized; this hot-path mirror intentionally shares that model. */
static bool g_decode_pipeline_fast_lookup_active;
static uint32_t g_decode_pipeline_fast_cache_entries;
static NSMutableDictionary<NSString *, DS4MetalQ4ExpertTable *> *g_q4_expert_table_cache;
static NSMutableDictionary<NSString *, id> *g_q4_expert_layer_residency_cache;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;

typedef struct {
    id<MTLBuffer> __strong mask;
    id<MTLBuffer> __strong blk;
    NSUInteger mask_bytes;
    NSUInteger blk_bytes;
    uint32_t kind;
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t n_keys;
    uint32_t window;
    uint32_t ratio;
    uint32_t nqptg;
    uint32_t ncpsg;
    bool has_kvpad;
    bool bc_mask;
    bool valid;
    bool blk_ready;
} ds4_gpu_zero_prefix_prefill_mask_cache_entry;

/* Actual cache kinds live once in attention_batch.h. */

static ds4_gpu_zero_prefix_prefill_mask_cache_entry
    g_zero_prefix_prefill_mask_cache[DS4_GPU_PREFILL_MASK_CACHE_SLOTS];
static void ds4_gpu_invalidate_zero_prefix_prefill_block_maps(void);
static id<MTLBuffer> g_flash_attn_mask_buffer;
static id<MTLBuffer> g_flash_attn_zero_mask_buffer;
static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_flash_attn_blk_buffer;
static id<MTLBuffer> g_flash_attn_ring_buffer;
static id<MTLBuffer> g_flash_attn_kv_buffer;
static id<MTLBuffer> g_glm_flash_attn_mask_buffer;
static id<MTLBuffer> g_compressor_pool_kv_buffer;
static id<MTLBuffer> g_compressor_pool_score_buffer;
static id<MTLBuffer> g_compressor_pool_score_cont_buffer;
static id<MTLBuffer> g_compressor_pool_softmax_buffer;
static id<MTLBuffer> g_compressor_pool_product_buffer;
static id<MTLBuffer> g_compressor_store_ape_buffer;
static id<MTLBuffer> g_compressor_store_score_buffer;
static id<MTLBuffer> g_embed_rows_buffer;
static id<MTLBuffer> g_router_selection_buffer;
static id<MTLBuffer> g_router_weight_sum_buffer;
static id<MTLBuffer> g_indexer_head_scores_buffer;
static id<MTLBuffer> g_indexer_topk_buffer;
static id<MTLBuffer> g_indexed_topk_buffer;
static id<MTLBuffer> g_f16_round_scratch_buffer;
static id<MTLBuffer> g_q8_prefill_scratch_buffer;
static id<MTLBuffer> g_qwen4_conv_halo_buffer;
static NSUInteger g_qwen4_conv_halo_bytes;
static id<MTLBuffer> g_raw_store_round_buffer;
static id<MTLBuffer> g_moe_gate_scratch_buffer;
static id<MTLBuffer> g_moe_down_scratch_buffer;
static id<MTLBuffer> g_moe_id_map_buffer;
static id<MTLBuffer> g_moe_packed_rhs_buffer;
static NSUInteger g_moe_packed_rhs_bytes;
static id<MTLBuffer> g_moe_q4_gate_slots_buffer;
static id<MTLBuffer> g_moe_q4_up_slots_buffer;
static id<MTLBuffer> g_moe_q4_down_slots_buffer;
static id<MTLBuffer> g_attn_out_group_ids_buffer;
static int g_model_fd = -1;

static int g_stream_prefill_batch_selected_addr_building;
static const ds4_gpu_tensor *g_stream_expert_batch_ready;
static int g_glm_stream_expert_addr_table_building;
#include "expert_cache.h"
#include "expert_tables.h"
static ds4_metal_expert_cache_state g_expert_cache = {
    .cache_first_decode_layer = UINT32_MAX,
};
#define g_stream_expert_cache_bytes (g_expert_cache.cache_bytes)
#define g_stream_expert_cache_expert_bytes (g_expert_cache.cache_expert_bytes)
#define g_stream_expert_cache_entry_count (g_expert_cache.cache_entry_count)
#define g_stream_expert_cache_budget_override (g_expert_cache.cache_budget_override)
#define g_stream_expert_pending_wait_ms (g_expert_cache.pending_wait_ms)
#define g_stream_expert_pending_joins (g_expert_cache.pending_joins)
#define g_stream_expert_gate_probes (g_expert_cache.gate_probes)
#define g_stream_expert_gate_submissions (g_expert_cache.gate_submissions)
#define g_stream_expert_cache_hits (g_expert_cache.cache_hits)
#define g_stream_expert_cache_misses (g_expert_cache.cache_misses)
#define g_stream_expert_cache_evictions (g_expert_cache.cache_evictions)
#define g_stream_expert_cache_wraps (g_expert_cache.cache_wraps)
#define g_stream_expert_cache_clock (g_expert_cache.cache_clock)
#define g_stream_expert_cache_evict_advise_bytes (g_expert_cache.cache_evict_advise_bytes)
#define g_stream_expert_cache_willneed_advise_bytes (g_expert_cache.cache_willneed_advise_bytes)
#define g_stream_expert_cache_pread_bytes (g_expert_cache.cache_pread_bytes)
#define g_stream_expert_cache_pread_ms (g_expert_cache.cache_pread_ms)
#define g_stream_expert_io_metrics_enabled (g_expert_cache.io_metrics_enabled)
#define g_stream_expert_io_pread_calls (g_expert_cache.io_pread_calls)
#define g_stream_expert_io_requested_bytes (g_expert_cache.io_requested_bytes)
#define g_stream_expert_io_returned_bytes (g_expert_cache.io_returned_bytes)
#define g_stream_expert_cache_buffer_allocs (g_expert_cache.cache_buffer_allocs)
#define g_stream_expert_cache_buffer_reuses (g_expert_cache.cache_buffer_reuses)
#define g_stream_expert_cache_decode_tokens (g_expert_cache.cache_decode_tokens)
#define g_stream_expert_cache_first_decode_layer (g_expert_cache.cache_first_decode_layer)
#define g_stream_expert_cache_hotness_decay_token (g_expert_cache.cache_hotness_decay_token)
#define g_stream_expert_timing_selected_calls (g_expert_cache.timing_selected_calls)
#define g_stream_expert_timing_selected_read_ms (g_expert_cache.timing_selected_read_ms)
#define g_stream_expert_timing_selected_sync_ms (g_expert_cache.timing_selected_sync_ms)
#define g_stream_expert_timing_selected_copy_ms (g_expert_cache.timing_selected_copy_ms)
#define g_stream_expert_timing_selected_bind_ms (g_expert_cache.timing_selected_bind_ms)
#define g_stream_expert_timing_split_layers (g_expert_cache.timing_split_layers)
#define g_stream_expert_timing_split_resident_experts (g_expert_cache.timing_split_resident_experts)
#define g_stream_expert_timing_split_missing_experts (g_expert_cache.timing_split_missing_experts)
#define g_stream_expert_timing_split_resident_ms (g_expert_cache.timing_split_resident_ms)
#define g_stream_expert_timing_split_missing_ms (g_expert_cache.timing_split_missing_ms)
#define g_stream_expert_timing_split_missing_load_ms (g_expert_cache.timing_split_missing_load_ms)
#define g_stream_expert_timing_split_missing_slot_ms (g_expert_cache.timing_split_missing_slot_ms)
#define g_stream_expert_timing_split_missing_prune_ms (g_expert_cache.timing_split_missing_prune_ms)
#define g_stream_expert_timing_split_missing_addr_ms (g_expert_cache.timing_split_missing_addr_ms)
#define g_stream_expert_timing_split_missing_wait_ms (g_expert_cache.timing_split_missing_wait_ms)
#define g_stream_expert_timing_load_calls (g_expert_cache.timing_load_calls)
#define g_stream_expert_timing_load_prepare_ms (g_expert_cache.timing_load_prepare_ms)
#define g_stream_expert_timing_load_pread_ms (g_expert_cache.timing_load_pread_ms)
#define g_stream_expert_timing_load_install_ms (g_expert_cache.timing_load_install_ms)
#define g_stream_expert_timing_prepare_batch_reuse_calls (g_expert_cache.timing_prepare_batch_reuse_calls)
#define g_stream_expert_timing_prepare_batch_reuse_ms (g_expert_cache.timing_prepare_batch_reuse_ms)
#define g_stream_expert_timing_prepare_buffer_calls (g_expert_cache.timing_prepare_buffer_calls)
#define g_stream_expert_timing_prepare_buffer_ms (g_expert_cache.timing_prepare_buffer_ms)
#define g_stream_expert_timing_prepare_task_experts (g_expert_cache.timing_prepare_task_experts)
#define g_stream_expert_timing_prepare_task_ms (g_expert_cache.timing_prepare_task_ms)
#define g_stream_expert_timing_reuse_scan_calls (g_expert_cache.timing_reuse_scan_calls)
#define g_stream_expert_timing_reuse_scan_entries (g_expert_cache.timing_reuse_scan_entries)
#define g_stream_expert_timing_reuse_scan_ms (g_expert_cache.timing_reuse_scan_ms)
#define g_stream_expert_timing_reuse_clear_ms (g_expert_cache.timing_reuse_clear_ms)
#define g_stream_expert_timing_readahead_calls (g_expert_cache.timing_readahead_calls)
#define g_stream_expert_timing_readahead_bytes (g_expert_cache.timing_readahead_bytes)
#define g_stream_expert_timing_readahead_ms (g_expert_cache.timing_readahead_ms)
#define g_stream_expert_timing_cache_all_resident_layers (g_expert_cache.timing_cache_all_resident_layers)
#define g_stream_expert_timing_cache_all_missing_layers (g_expert_cache.timing_cache_all_missing_layers)
#define g_stream_expert_timing_cache_mixed_layers (g_expert_cache.timing_cache_mixed_layers)
#define g_stream_expert_timing_cache_resident_experts (g_expert_cache.timing_cache_resident_experts)
#define g_stream_expert_timing_cache_missing_experts (g_expert_cache.timing_cache_missing_experts)
#define g_stream_expert_timing_last_report (g_expert_cache.timing_last_report)
#define g_stream_expert_cache (g_expert_cache.entries)
#define g_stream_full_expert_addr_entry (g_expert_cache.full_entries)
#define g_stream_expert_cache_layer_count (g_expert_cache.cache_layer_count)
#define g_stream_expert_cache_layer_hits (g_expert_cache.cache_layer_hits)
#define g_stream_expert_cache_layer_misses (g_expert_cache.cache_layer_misses)
#define g_stream_expert_cache_layer_evictions (g_expert_cache.cache_layer_evictions)
#define g_stream_expert_cache_layer_pread_bytes (g_expert_cache.cache_layer_pread_bytes)
#define g_stream_expert_cache_layer_pread_ms (g_expert_cache.cache_layer_pread_ms)
#define g_stream_expert_cache_layer_last_hits (g_expert_cache.cache_layer_last_hits)
#define g_stream_expert_cache_layer_last_misses (g_expert_cache.cache_layer_last_misses)
#define g_stream_expert_cache_layer_last_evictions (g_expert_cache.cache_layer_last_evictions)
#define g_stream_expert_cache_layer_last_pread_bytes (g_expert_cache.cache_layer_last_pread_bytes)
#define g_stream_expert_cache_layer_last_pread_ms (g_expert_cache.cache_layer_last_pread_ms)
#define g_stream_expert_cache_route_hotness (g_expert_cache.cache_route_hotness)
#define g_stream_expert_cache_gate_addr_buffers (g_expert_cache.cache_gate_addr_buffers)
#define g_stream_expert_cache_up_addr_buffers (g_expert_cache.cache_up_addr_buffers)
#define g_stream_expert_cache_down_addr_buffers (g_expert_cache.cache_down_addr_buffers)
#define g_stream_expert_cache_slabs (g_expert_cache.cache_slabs)
#define g_stream_expert_cache_slab_start_slot (g_expert_cache.cache_slab_start_slot)
#define g_stream_expert_cache_slab_slot_count (g_expert_cache.cache_slab_slot_count)
#define g_stream_expert_cache_slab_slots_used (g_expert_cache.cache_slab_slots_used)
#define g_stream_expert_cache_slab_count (g_expert_cache.cache_slab_count)
#define g_stream_expert_cache_slab_total_slots (g_expert_cache.cache_slab_total_slots)
#define g_stream_expert_cache_free_slots (g_expert_cache.cache_free_slots)
#define g_stream_expert_cache_free_slot_count (g_expert_cache.cache_free_slot_count)
#define g_stream_expert_cache_slab_slot_locked (g_expert_cache.cache_slab_slot_locked)
#define g_stream_expert_cache_mlock_budget_cap (g_expert_cache.cache_mlock_budget_cap)
#define g_stream_expert_cache_mlock_relief_applied (g_expert_cache.cache_mlock_relief_applied)
#define g_stream_expert_cache_mlock_bytes (g_expert_cache.cache_mlock_bytes)
#define g_stream_expert_cache_mlock_fail_bytes (g_expert_cache.cache_mlock_fail_bytes)
#define g_stream_expert_cache_mlock_failures (g_expert_cache.cache_mlock_failures)
#define g_stream_expert_cache_mlock_ms (g_expert_cache.cache_mlock_ms)
#define g_stream_expert_cache_mlock_warned (g_expert_cache.cache_mlock_warned)
#define g_stream_expert_cache_slab_slot_bytes (g_expert_cache.cache_slab_slot_bytes)
#define g_stream_expert_cache_cb_seq (g_expert_cache.cache_cb_seq)
#define g_stream_expert_cache_done_seq (g_expert_cache.cache_done_seq)
#define g_stream_expert_cache_batch_seq (g_expert_cache.cache_batch_seq)
#define g_stream_expert_cache_owned_seq (g_expert_cache.cache_owned_seq)
#define g_stream_expert_cache_pending_max_seq (g_expert_cache.cache_pending_max_seq)
#define g_stream_compact_gate_addr_buffers (g_expert_cache.compact_gate_addr_buffers)
#define g_stream_compact_up_addr_buffers (g_expert_cache.compact_up_addr_buffers)
#define g_stream_compact_down_addr_buffers (g_expert_cache.compact_down_addr_buffers)
#define g_stream_compact_selected_buffers (g_expert_cache.compact_selected_buffers)
#define g_stream_selected_id_buffers (g_expert_cache.selected_id_buffers)
#define g_stream_expert_validate_status_buffer (g_expert_cache.validate_status_buffer)

static int g_glm_model_mode;
static int g_ssd_streaming_mode;
static int g_glm_streaming_prefill_full_layer_runtime;
static ds4_metal_bootstrap_state g_bootstrap;
#define g_initialized (g_bootstrap.initialized)
#define g_metal4_runtime_available (g_bootstrap.runtime_available)
#define g_metal4_family_supported (g_bootstrap.family_supported)
#define g_metal4_queue_supported (g_bootstrap.queue_supported)
#define g_metal4_m5_neural_accelerators_hint (g_bootstrap.neural_accelerators_hint)
#define g_metal4_tensor_api_enabled (g_bootstrap.tensor_enabled)
#define g_metal4_tensor_api_compile_supported (g_bootstrap.tensor_compile_supported)
#define g_metal_device_name (g_bootstrap.device_name)
static int ds4_gpu_stream_expert_cache_note_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
static uint32_t ds4_gpu_stream_expert_cache_configured_budget(void);
static void ds4_gpu_stream_expert_cache_clear_all(int reset_stats);
static int ds4_gpu_stream_expert_cache_trim(uint32_t experts);
static void ds4_gpu_stream_expert_pending_load_clear(void);
static void ds4_gpu_stream_expert_pread_pool_shutdown(void);
static int ds4_gpu_stream_expert_timing_summary_enabled(void);
static int ds4_gpu_stream_expert_cache_entry_protected(
        uint32_t       layer,
        uint32_t       expert,
        uint32_t       protect_layer,
        const int32_t *protect_ids,
        uint32_t       n_protect);

/* The async selected-load worker registers itself so cache paths that would
 * flush/wait on command buffers (a race against the encoding thread) fail
 * the load instead; the caller then retries on the main thread. */
static pthread_t g_stream_expert_service_thread;
static int g_stream_expert_service_thread_set;

void ds4_gpu_stream_expert_cache_note_service_thread(void) {
    g_stream_expert_service_thread = pthread_self();
    g_stream_expert_service_thread_set = 1;
}

static int ds4_gpu_stream_expert_cache_on_service_thread(void) {
    return g_stream_expert_service_thread_set &&
           pthread_equal(pthread_self(), g_stream_expert_service_thread);
}
static NSUInteger g_flash_attn_mask_bytes;
static NSUInteger g_flash_attn_zero_mask_bytes;
static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_flash_attn_blk_bytes;
static NSUInteger g_flash_attn_ring_bytes;
static NSUInteger g_flash_attn_kv_bytes;
static NSUInteger g_glm_flash_attn_mask_bytes;
static uint32_t g_glm_flash_attn_mask_pos0;
static uint32_t g_glm_flash_attn_mask_tokens;
static uint32_t g_glm_flash_attn_mask_cache_len;
static int g_glm_flash_attn_mask_valid;
static NSUInteger g_compressor_pool_kv_bytes;
static NSUInteger g_compressor_pool_score_bytes;
static NSUInteger g_compressor_pool_score_cont_bytes;
static NSUInteger g_compressor_pool_softmax_bytes;
static NSUInteger g_compressor_pool_product_bytes;
static NSUInteger g_compressor_store_ape_bytes;
static NSUInteger g_compressor_store_score_bytes;
static NSUInteger g_embed_rows_bytes;
static NSUInteger g_router_selection_bytes;
static NSUInteger g_router_weight_sum_bytes;
static NSUInteger g_indexer_head_scores_bytes;
static NSUInteger g_indexer_topk_bytes;
static NSUInteger g_indexed_topk_bytes;
static NSUInteger g_f16_round_scratch_bytes;
static NSUInteger g_q8_prefill_scratch_bytes;
static NSUInteger g_raw_store_round_bytes;
static NSUInteger g_moe_gate_scratch_bytes;
static NSUInteger g_moe_down_scratch_bytes;
static NSUInteger g_moe_id_map_bytes;
static NSUInteger g_moe_q4_gate_slots_bytes;
static NSUInteger g_moe_q4_up_slots_bytes;
static NSUInteger g_moe_q4_down_slots_bytes;
static NSUInteger g_attn_out_group_ids_bytes;
static int g_quality_mode;
static int g_mpp_invalid_env_reported;

static double ds4_gpu_gib(uint64_t bytes);

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

/* Compatibility fallback for callers that cannot provide a parsed GGUF tensor
 * span. The normal DS4 engine passes the exact maximum tensor byte size. */

static ds4_metal_model_views g_model_views;










@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
- (ds4_metal_tensor_region)ds4ResourceRegion;
@end

@implementation DS4MetalTensor
/* One metadata snapshot avoids three property messages on native region reads. */
- (ds4_metal_tensor_region)ds4ResourceRegion {
    return (ds4_metal_tensor_region){_offset, _bytes, _owner};
}
@end

@interface DS4MetalQ4ExpertTable : NSObject
@property(nonatomic, strong) id<MTLBuffer> argumentBuffer;
@property(nonatomic, strong) id<MTLBuffer> addressBuffer;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *expertBuffers;
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL residencySetAddedToQueue;
@property(nonatomic, assign) uint32_t nExpert;
@property(nonatomic, assign) uint64_t expertBytes;
@end

@implementation DS4MetalQ4ExpertTable
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_residencySetAddedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

@interface DS4MetalQ4LayerResidency : NSObject
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL addedToQueue;
@end

@implementation DS4MetalQ4LayerResidency
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_addedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

static DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

static void ds4_gpu_tensor_tracking_reset(void) {
    ds4_native_metal_tensor_tracking_reset();
}

static id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void);
static void ds4_gpu_stream_expert_cache_note_owned_created(void);

static id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    return (__bridge_transfer id<MTLCommandBuffer>)ds4_native_metal_command_acquire(owned);
}

static id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    return (__bridge_transfer id<MTLComputeCommandEncoder>)ds4_native_metal_compute_encoder(
        (__bridge void *)cb, (uintptr_t)__builtin_return_address(0));
}

static void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    ds4_native_metal_end_encoder((__bridge void *)cb, (__bridge void *)enc);
}

static void ds4_gpu_close_batch_encoder(void) {
    ds4_native_metal_close_encoder();
}

static double g_gpu_busy_accum;
static uint64_t g_gpu_busy_cbs;

/* A failed command buffer can leave a cross-threadgroup arrival counter at an
 * arbitrary partial value.  Drop cached ownership instead of CPU-resetting
 * buffers that another in-flight command buffer might still reference; bound
 * resources remain retained by their command buffers/transient list. */
static void ds4_gpu_invalidate_completion_counters(void) {
    [g_dsv4_completion_cache removeAllObjects];
    g_dsv4_hc_producer_last_mix_buffer = nil;
    g_dsv4_hc_producer_last_mix_offset = 0;
    g_dsv4_hc_producer_last_completion = nil;
}

static int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (getenv("DS4_METAL_CB_TIMES")) {
        static double prev_gpu_end;
        static uint64_t n_printed;
        if (n_printed < 400) {
            n_printed++;
            fprintf(stderr,
                    "ds4: cb %s: driver %.0f us, queue-wait %.0f us, gpu %.0f us, gap-from-prev-gpu-end %.0f us\n",
                    label,
                    (cb.kernelEndTime - cb.kernelStartTime) * 1e6,
                    (cb.GPUStartTime - cb.kernelEndTime) * 1e6,
                    (cb.GPUEndTime - cb.GPUStartTime) * 1e6,
                    prev_gpu_end ? (cb.GPUStartTime - prev_gpu_end) * 1e6 : 0.0);
        }
        prev_gpu_end = cb.GPUEndTime;
    }
    if (getenv("DS4_METAL_GPU_BUSY_PROFILE")) {
        const double busy = cb.GPUEndTime - cb.GPUStartTime;
        if (busy > 0) g_gpu_busy_accum += busy;
        if ((++g_gpu_busy_cbs % 64u) == 0u) {
            fprintf(stderr, "ds4: gpu busy accum %.1f ms over %llu cbs\n",
                    g_gpu_busy_accum * 1000.0,
                    (unsigned long long)g_gpu_busy_cbs);
        }
    }
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        ds4_gpu_invalidate_completion_counters();
        return 0;
    }
    return 1;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void) {
    return (__bridge_transfer id<MTLCommandBuffer>)ds4_native_metal_new_command();
}

static void ds4_gpu_model_buffer_cache_clear(const char *reason) {
    ds4_metal_exact_cache_clear(reason);
}
static void ds4_gpu_model_buffer_cache_maybe_evict(const char *reason) {
    ds4_metal_exact_cache_maybe_evict(reason);
}

static uint64_t ds4_gpu_stream_expert_cache_next_cb_seq(void){
    return ds4_native_expert_cache_next_cb_seq(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_batch_created(void){
    ds4_native_expert_cache_note_batch_created(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_batch_committed(void){
    ds4_native_expert_cache_note_batch_committed(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_owned_created(void){
    ds4_native_expert_cache_note_owned_created(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_pending_completed(void){
    ds4_native_expert_cache_note_pending_completed(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_owned_completed(void){
    ds4_native_expert_cache_note_owned_completed(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static int ds4_gpu_stream_expert_cache_entry_inflight(
        const ds4_gpu_stream_expert_cache_entry *e){
    return ds4_native_expert_cache_entry_inflight(&g_expert_cache, g_ssd_streaming_mode != 0, e);
}

static int ds4_gpu_stream_expert_cache_mark_inflight(
        ds4_gpu_stream_expert_cache_entry *e){
    return ds4_native_expert_cache_mark_inflight(&g_expert_cache, g_ssd_streaming_mode != 0, e);
}

static int ds4_gpu_stream_expert_cache_mark_entries_inflight(
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t n_entries,
        uint32_t active_mask){
    return ds4_native_expert_cache_mark_entries_inflight(&g_expert_cache, g_ssd_streaming_mode != 0, entries, n_entries, active_mask);
}

static int ds4_gpu_stream_expert_cache_wait_inflight(const char *label);

static int ds4_gpu_wait_pending_command_buffers(const char *label) {
    return ds4_native_metal_wait_pending(label);
}

#define g_batch_cb_created_ms g_schedule.commands.created_ms
static double ds4_gpu_now_ms(void);
static void ds4_gpu_queue_keepalive_start(void);
static void ds4_gpu_queue_keepalive_stop_thread(void);
static int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    return ds4_native_metal_finish_command((__bridge void *)cb, owned, label,
        (uintptr_t)__builtin_return_address(0));
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = ds4_gpu_device_name_contains("M5");
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strcmp(label, "ds4_attention_output_group_ids") == 0;
}

static MTLResourceOptions ds4_gpu_model_resource_options(void) {
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (getenv("DS4_METAL_MODEL_UNTRACKED") != NULL) {
        options |= MTLResourceHazardTrackingModeUntracked;
    }
    return options;
}

static int ds4_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (ds4_gpu_use_m5_private_scratch() &&
        !ds4_gpu_scratch_needs_cpu_access(label)) {
        /*
         * M5 scratch buffers that only flow between Metal kernels do not need
         * CPU-visible shared storage. This reduces shared-memory traffic and
         * residency pressure for the long prefill scratch pools without
         * changing the public buffer lifetime model. Keep default hazard
         * tracking because the graph reuses these buffers across dependent
         * compute encoders.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

static int ds4_gpu_ensure_zero_attention_mask(NSUInteger bytes) {
    const NSUInteger capacity = 8192u * sizeof(uint16_t);
    if (bytes > capacity) return 0;
    if (g_flash_attn_zero_mask_buffer &&
        g_flash_attn_zero_mask_bytes >= capacity) {
        return 1;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_zero_mask_buffer,
                                         &g_flash_attn_zero_mask_bytes,
                                         capacity,
                                         "ds4_flash_attn_zero_mask")) {
        return 0;
    }
    void *contents = [g_flash_attn_zero_mask_buffer contents];
    if (!contents) return 0;
    memset(contents, 0, g_flash_attn_zero_mask_bytes);
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}


static id<MTLComputePipelineState> ds4_gpu_get_pipeline(const char *function_name);
static double ds4_gpu_gib(uint64_t bytes);

static double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}





/* Legacy GLM and selective C callers share the sole native diagnostic state. */
static int ds4_gpu_moe_selected_hotlist_record(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected,
        uint32_t       n_total_expert) {
    return ds4_native_metal_selected_hotlist_record(layer, selected_ids, n_selected, n_total_expert);
}




static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

static void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

static void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_model_views_clear(void) {
    ds4_native_metal_model_views_clear(&g_model_views);
}

static __attribute__((unused)) void ds4_gpu_model_views_remove_map(const void *model_map) {
    ds4_native_metal_model_views_remove_map(&g_model_views, model_map);
}


static void ds4_gpu_model_residency_clear(void) {
    ds4_metal_model_view_residency_clear(&g_model_views);
}

/* TP sharding keeps only this rank's expert ranges warm,
 * so whole-view residency requests (which would page in the full file)
 * must be skipped; pages fault in lazily through the same view buffers,
 * exactly like ssd-streaming mode. */

void ds4_gpu_model_residency_skip(int skip) {
    g_model_views.residency_skipped = skip;
}


static __attribute__((unused)) int ds4_gpu_add_model_view_range(
        const void *model_map, uint64_t model_size, uint64_t map_offset,
        uint64_t map_size, uint64_t max_tensor_bytes, bool use_default_view_cap,
        uint64_t *mapped_model_size_out) {
    return ds4_native_metal_model_views_add(&g_model_views, model_map,
        model_size, map_offset, map_size, max_tensor_bytes,
        use_default_view_cap, mapped_model_size_out);
}




static id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label) {
    if (bytes == 0) bytes = 1;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal transient buffer %s (%llu bytes)\n",
                label ? label : "(unnamed)", (unsigned long long)bytes);
        return nil;
    }
    if (label) buffer.label = [NSString stringWithUTF8String:label];

    /*
     * CPU-filled buffers must survive until their command buffer completes.
     * A local ObjC strong variable is not enough when the encoder function
     * returns before the caller commits the command buffer.
     */
    [g_transient_buffers addObject:buffer];
    return buffer;
}

static int ds4_gpu_zero_prefix_prefill_mask_cache_enabled(void) {
    if (getenv("DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE") != NULL ||
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL) {
        return 0;
    }
    return ds4_gpu_device_name_contains("M3");
}

static ds4_gpu_zero_prefix_prefill_mask_cache_entry *
ds4_gpu_get_zero_prefix_prefill_mask_cache(
        uint32_t   kind,
        uint32_t   n_tokens,
        uint32_t   n_comp,
        uint32_t   n_keys,
        uint32_t   window,
        uint32_t   ratio,
        uint32_t   nqptg,
        uint32_t   ncpsg,
        bool       has_kvpad,
        bool       bc_mask,
        NSUInteger mask_bytes,
        NSUInteger blk_bytes,
        bool      *created) {
    if (created) *created = false;
    if (!created || !ds4_gpu_zero_prefix_prefill_mask_cache_enabled() ||
        mask_bytes == 0 || blk_bytes == 0) {
        return NULL;
    }

    uint32_t slot = UINT32_MAX;
    switch (kind) {
        case DS4_GPU_PREFILL_MASK_CACHE_RAW:      slot = 0; break;
        case DS4_GPU_PREFILL_MASK_CACHE_RATIO4:   slot = 1; break;
        case DS4_GPU_PREFILL_MASK_CACHE_RATIO128: slot = 2; break;
        default: return NULL;
    }

    ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
        &g_zero_prefix_prefill_mask_cache[slot];
    if (entry->valid && entry->mask && entry->blk &&
        entry->mask_bytes == mask_bytes && entry->blk_bytes == blk_bytes &&
        entry->kind == kind && entry->n_tokens == n_tokens &&
        entry->n_comp == n_comp && entry->n_keys == n_keys &&
        entry->window == window && entry->ratio == ratio &&
        entry->nqptg == nqptg && entry->ncpsg == ncpsg &&
        entry->has_kvpad == has_kvpad && entry->bc_mask == bc_mask) {
        return entry;
    }

    id<MTLBuffer> mask = [g_device newBufferWithLength:mask_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> blk = [g_device newBufferWithLength:blk_bytes
                                               options:MTLResourceStorageModePrivate];
    if (!blk) {
        blk = [g_device newBufferWithLength:blk_bytes
                                    options:MTLResourceStorageModeShared];
    }
    if (!mask || !blk) return NULL;

    mask.label = [NSString stringWithFormat:@"ds4_prefill_mask_cache_%u", kind];
    blk.label = [NSString stringWithFormat:@"ds4_prefill_blk_cache_%u", kind];

    /* A replaced entry may still be referenced by an uncommitted or in-flight
     * command buffer. Keep its resources alive with the other batch-scoped
     * buffers until command completion instead of mutating or releasing them. */
    if (entry->mask) [g_transient_buffers addObject:entry->mask];
    if (entry->blk) [g_transient_buffers addObject:entry->blk];

    entry->mask = mask;
    entry->blk = blk;
    entry->mask_bytes = mask_bytes;
    entry->blk_bytes = blk_bytes;
    entry->kind = kind;
    entry->n_tokens = n_tokens;
    entry->n_comp = n_comp;
    entry->n_keys = n_keys;
    entry->window = window;
    entry->ratio = ratio;
    entry->nqptg = nqptg;
    entry->ncpsg = ncpsg;
    entry->has_kvpad = has_kvpad;
    entry->bc_mask = bc_mask;
    /* The caller publishes the entry only after synchronously filling mask. */
    entry->valid = false;
    entry->blk_ready = false;
    *created = true;
    return entry;
}

static void ds4_gpu_invalidate_zero_prefix_prefill_block_maps(void) {
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        g_zero_prefix_prefill_mask_cache[i].blk_ready = false;
    }
}

static void ds4_gpu_clear_zero_prefix_prefill_mask_cache(void) {
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
            &g_zero_prefix_prefill_mask_cache[i];
        entry->mask = nil;
        entry->blk = nil;
        entry->mask_bytes = 0;
        entry->blk_bytes = 0;
        entry->kind = 0;
        entry->n_tokens = 0;
        entry->n_comp = 0;
        entry->n_keys = 0;
        entry->window = 0;
        entry->ratio = 0;
        entry->nqptg = 0;
        entry->ncpsg = 0;
        entry->has_kvpad = false;
        entry->bc_mask = false;
        entry->valid = false;
        entry->blk_ready = false;
    }
}

void ds4_gpu_release_zero_prefix_prefill_mask_cache(void) {
    /* Layer-major prefill waits each layer before advancing, so its final
     * release point has no outstanding cache users. Keep the guard here for
     * diagnostic callers that may have an open or asynchronously flushed CB. */
    if (!g_initialized || g_batch_cb ||
        (g_pending_cbs && [g_pending_cbs count] != 0)) {
        return;
    }
    ds4_gpu_clear_zero_prefix_prefill_mask_cache();
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d",
                     function_name, bc_inp ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        /* Failed lookups are remembered as NSNull so the error prints once
         * per name instead of on every dispatch. */
        if (cached == (id<MTLComputePipelineState>)[NSNull null]) return nil;
        return cached;
    }

    const bool profile_create = getenv("DS4_METAL_PIPELINE_CREATE_PROFILE") != NULL;
    const double create_start = profile_create ? ds4_gpu_now_ms() : 0.0;
    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        [g_pipeline_cache setObject:(id<MTLComputePipelineState>)[NSNull null] forKey:key];
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        [g_pipeline_cache setObject:(id<MTLComputePipelineState>)[NSNull null] forKey:key];
        return nil;
    }

    if (profile_create) fprintf(stderr, "ds4: Metal pipeline create %s %.3f ms\n",
        [key UTF8String], ds4_gpu_now_ms() - create_start);
    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static int ds4_gpu_disable_hot_pipeline_statics(void) {
    static int initialized;
    static int disabled;
    if (!initialized) {
        disabled = getenv("DS4_METAL_DISABLE_HOT_PIPELINE_STATICS") != NULL;
        initialized = 1;
    }
    return disabled;
}

static id<MTLComputePipelineState> ds4_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    if (!ds4_gpu_disable_hot_pipeline_statics()) return pipeline;
    return ds4_gpu_get_pipeline(fallback_name);
}

static int ds4_gpu_use_compressor_pair_nr4(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_COMPRESSOR_PAIR_NR4") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_env_value_eq(const char *v, size_t n, const char *literal) {
    size_t m = strlen(literal);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) != tolower((unsigned char)literal[i])) return 0;
    }
    return 1;
}

static int ds4_gpu_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v) return -1;

    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (n == 0) return 1;

    if (ds4_gpu_env_value_eq(v, n, "1") ||
        ds4_gpu_env_value_eq(v, n, "true") ||
        ds4_gpu_env_value_eq(v, n, "yes") ||
        ds4_gpu_env_value_eq(v, n, "on")) {
        return 1;
    }
    if (ds4_gpu_env_value_eq(v, n, "0") ||
        ds4_gpu_env_value_eq(v, n, "false") ||
        ds4_gpu_env_value_eq(v, n, "no") ||
        ds4_gpu_env_value_eq(v, n, "off")) {
        return 0;
    }

    if (!g_mpp_invalid_env_reported) {
        fprintf(stderr,
                "ds4: invalid Metal boolean environment value %s=%.*s; treating presence as enabled\n",
                name, (int)n, v);
        g_mpp_invalid_env_reported = 1;
    }
    return 1;
}

static uint64_t ds4_gpu_env_u64(const char *name,
                                uint64_t    fallback,
                                uint64_t    min_value,
                                uint64_t    max_value) {
    const char *v = getenv(name);
    if (!v) return fallback;
    while (isspace((unsigned char)*v)) v++;
    if (!*v) return fallback;

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(v, &end, 10);
    if (end == v || errno == ERANGE) return fallback;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return fallback;

    if (parsed < min_value) return fallback;
    uint64_t value = (uint64_t)parsed;
    if (value > max_value) value = max_value;
    return value;
}

static uint32_t ds4_gpu_glm_full_attention_max_cache_len(void) {
    /*
     * kernel_glm_attention_full stores one score per visible token in
     * threadgroup memory, plus 256 reduction slots.  Keep the default under
     * the 32 KiB envelope used by current Apple GPUs while allowing short
     * dense decode to step past the 4096-token prefill boundary.
     */
    return 7680u;
}

static uint32_t ds4_gpu_glm_flash_attention_max_cache_len(void) {
    /*
     * Staged FlashAttention uses fixed-size threadgroup scratch and separate
     * KV staging buffers, so it is not bound by the legacy full-attention
     * kernel's one-score-per-token threadgroup-memory envelope.
     */
    return 8192u;
}

static int ds4_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * DS4_METAL_DISABLE_METAL4 comparison switch.  Benchmark-only alternatives that
 * lost during M5 work are removed or kept out of the dispatch path so future
 * changes do not accidentally turn old experiments into new modes.
 */

/* The retained tile constant lives once in attention_output.h. */

static void ds4_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "ds4: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

static int ds4_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

int ds4_gpu_device_is_pre_m5_apple_silicon(void) {
    return strncmp(g_metal_device_name, "Apple M", 7) == 0 &&
           g_metal_device_name[7] >= '1' &&
           g_metal_device_name[7] <= '4' &&
           (g_metal_device_name[8] == '\0' ||
            g_metal_device_name[8] == ' ');
}

int ds4_gpu_device_is_m5_apple_silicon(void) {
    return strncmp(g_metal_device_name, "Apple M5", 8) == 0 &&
           (g_metal_device_name[8] == '\0' ||
            g_metal_device_name[8] == ' ');
}

static bool ds4_gpu_ported_m5_decode_feature_enabled(
        const char *pre_m5_disable_env,
        const char *m5_disable_env) {
    if (m5_disable_env && getenv(m5_disable_env) != NULL) return false;
    const bool pre_m5 = ds4_gpu_device_is_pre_m5_apple_silicon();
    if (pre_m5 &&
        (getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") != NULL ||
         (pre_m5_disable_env && getenv(pre_m5_disable_env) != NULL))) {
        return false;
    }
    return pre_m5 || ds4_gpu_device_is_m5_apple_silicon();
}

static int ds4_gpu_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void ds4_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library = [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"ds4_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}



static const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}

static void ds4_gpu_decode_pipeline_fast_cache_reset(void) {
    for (uint32_t i = 0; i < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS; i++) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[i];
        /* ARC must see every strong release; do not memset this table. */
        entry->pipeline = nil;
        entry->hash = 0;
        entry->nsg = 0;
        entry->name_len = 0;
        entry->used = false;
        entry->name[0] = '\0';
    }
    g_decode_pipeline_fast_lookup_active = false;
    g_decode_pipeline_fast_cache_entries = 0;
}

static bool ds4_gpu_decode_pipeline_fast_key(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg,
        uint16_t   *name_len_out,
        uint64_t   *hash_out) {
    if (!function_name || !name_len_out || !hash_out) return false;

    uint64_t hash = UINT64_C(14695981039346656037);
    size_t name_len = 0;
    while (function_name[name_len] != '\0') {
        if (name_len + 1u >= DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES) {
            return false;
        }
        hash ^= (uint8_t)function_name[name_len++];
        hash *= UINT64_C(1099511628211);
    }
    const uint16_t nsg_bits = (uint16_t)nsg;
    hash ^= (uint8_t)nsg_bits;
    hash *= UINT64_C(1099511628211);
    hash ^= (uint8_t)(nsg_bits >> 8u);
    hash *= UINT64_C(1099511628211);
    const uint16_t nxpsg_bits = (uint16_t)nxpsg;
    hash ^= (uint8_t)nxpsg_bits;
    hash *= UINT64_C(1099511628211);
    hash ^= (uint8_t)(nxpsg_bits >> 8u);
    hash *= UINT64_C(1099511628211);

    *name_len_out = (uint16_t)name_len;
    *hash_out = hash;
    return true;
}

static id<MTLComputePipelineState> ds4_gpu_decode_pipeline_fast_cache_lookup(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg,
        uint16_t    name_len,
        uint64_t    hash) {
    const uint32_t mask = DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u;
    uint32_t slot = (uint32_t)hash & mask;
    for (uint32_t probe = 0;
         probe < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS;
         probe++, slot = (slot + 1u) & mask) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[slot];
        if (!entry->used) break;
        if (entry->hash == hash &&
            entry->nsg == nsg &&
            entry->nxpsg == nxpsg &&
            entry->name_len == name_len &&
            memcmp(entry->name, function_name, name_len) == 0) {
            return entry->pipeline;
        }
    }
    return nil;
}

static void ds4_gpu_decode_pipeline_fast_cache_insert(
        const char                  *function_name,
        int16_t                      nsg,
        int16_t                      nxpsg,
        uint16_t                     name_len,
        uint64_t                     hash,
        id<MTLComputePipelineState>  pipeline) {
    if (!pipeline ||
        g_decode_pipeline_fast_cache_entries >=
            DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS) {
        return;
    }

    const uint32_t mask = DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u;
    uint32_t slot = (uint32_t)hash & mask;
    for (uint32_t probe = 0;
         probe < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS;
         probe++, slot = (slot + 1u) & mask) {
        ds4_gpu_decode_pipeline_fast_cache_entry *entry =
            &g_decode_pipeline_fast_cache[slot];
        if (entry->used) {
            if (entry->hash == hash &&
                entry->nsg == nsg &&
                entry->nxpsg == nxpsg &&
                entry->name_len == name_len &&
                memcmp(entry->name, function_name, name_len) == 0) {
                entry->pipeline = pipeline;
                return;
            }
            continue;
        }
        memcpy(entry->name, function_name, name_len);
        entry->name[name_len] = '\0';
        entry->hash = hash;
        entry->nsg = nsg;
        entry->nxpsg = nxpsg;
        entry->name_len = name_len;
        entry->pipeline = pipeline;
        entry->used = true;
        g_decode_pipeline_fast_cache_entries++;
        return;
    }
}

int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled) {
    const int previous = g_decode_pipeline_fast_lookup_active ? 1 : 0;
    g_decode_pipeline_fast_lookup_active = enabled != 0;
    return previous;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    uint16_t fast_name_len = 0;
    uint64_t fast_hash = 0;
    const bool fast_key_valid =
        g_decode_pipeline_fast_lookup_active &&
        ds4_gpu_decode_pipeline_fast_key(
            function_name, nsg,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            &fast_name_len, &fast_hash);
    if (fast_key_valid) {
        id<MTLComputePipelineState> fast_cached =
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                function_name, nsg,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                fast_name_len, fast_hash);
        if (fast_cached) return fast_cached;
    }

    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (fast_key_valid) {
            ds4_gpu_decode_pipeline_fast_cache_insert(
                function_name, nsg,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                fast_name_len, fast_hash, cached);
        }
        return cached;
    }

    const bool profile_create = getenv("DS4_METAL_PIPELINE_CREATE_PROFILE") != NULL;
    const double create_start = profile_create ? ds4_gpu_now_ms() : 0.0;
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    if (profile_create) fprintf(stderr, "ds4: Metal pipeline create %s %.3f ms\n",
        [key UTF8String], ds4_gpu_now_ms() - create_start);
    [g_pipeline_cache setObject:pipeline forKey:key];
    if (fast_key_valid) {
        ds4_gpu_decode_pipeline_fast_cache_insert(
            function_name, nsg,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            fast_name_len, fast_hash, pipeline);
    }
    return pipeline;
}

/* The ordinary mul-mv cache key covers function name and nsg only. Keep the
 * descriptor-hinted PSO separate so a cache hit cannot erase this compiler
 * contract or substitute it for the fallback pipeline. */
static id<MTLComputePipelineState> ds4_gpu_new_mul_mv_tg_multiple_pipeline(
        const char *function_name,
        int16_t     nsg) {
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s tg-multiple function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
    descriptor.label = name;
    descriptor.computeFunction = fn;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

    error = nil;
    id<MTLComputePipelineState> pipeline =
        [g_device newComputePipelineStateWithDescriptor:descriptor
                                                options:MTLPipelineOptionNone
                                             reflection:nil
                                                  error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s tg-multiple pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }
    return pipeline;
}

int ds4_gpu_test_decode_pipeline_fast_lookup(void) {
    if (!g_initialized || !g_pipeline_cache || !g_library || !g_device) {
        return 0;
    }

    static const char base_name[] = "kernel_mul_mv_id_mxfp4_f32";
    char copied_name[sizeof(base_name)];
    memcpy(copied_name, base_name, sizeof(base_name));

    ds4_gpu_decode_pipeline_fast_cache_reset();
    id<MTLComputePipelineState> baseline =
        ds4_gpu_get_mul_mv_pipeline(base_name, 2);
    if (!baseline ||
        g_decode_pipeline_fast_cache_entries != 0u) {
        ds4_gpu_decode_pipeline_fast_cache_reset();
        return 0;
    }

    (void)ds4_gpu_set_decode_pipeline_fast_lookup(1);
    id<MTLComputePipelineState> populated =
        ds4_gpu_get_mul_mv_pipeline(base_name, 2);
    id<MTLComputePipelineState> hit =
        ds4_gpu_get_mul_mv_pipeline(copied_name, 2);
    uint16_t copied_name_len = 0;
    uint64_t copied_hash = 0;
    const bool copied_key_valid =
        ds4_gpu_decode_pipeline_fast_key(
            copied_name, 2,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            &copied_name_len, &copied_hash);
    id<MTLComputePipelineState> direct_hit =
        copied_key_valid ?
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                copied_name, 2,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                copied_name_len, copied_hash) : nil;
    id<MTLComputePipelineState> distinct_name =
        ds4_gpu_get_mul_mv_pipeline(
            "kernel_mul_mv_id_mxfp4_pair_swiglu_f32", 2);
    id<MTLComputePipelineState> distinct_nsg =
        ds4_gpu_get_mul_mv_pipeline(base_name, 4);
    const bool populated_and_hit =
        populated == baseline &&
        hit == baseline &&
        direct_hit == baseline &&
        distinct_name != nil &&
        distinct_nsg != nil &&
        g_decode_pipeline_fast_cache_entries == 3u;

    (void)ds4_gpu_set_decode_pipeline_fast_lookup(0);
    const uint32_t entries_before_disabled_lookup =
        g_decode_pipeline_fast_cache_entries;
    id<MTLComputePipelineState> disabled =
        ds4_gpu_get_mul_mv_pipeline(base_name, 2);
    const bool inactive_unchanged =
        disabled == baseline &&
        g_decode_pipeline_fast_cache_entries == entries_before_disabled_lookup;

    ds4_gpu_decode_pipeline_fast_cache_reset();
    (void)ds4_gpu_set_decode_pipeline_fast_lookup(1);
    bool filled = true;
    for (uint32_t i = 0;
         i < DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS && filled;
         i++) {
        char synthetic_name[DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES];
        const int written = snprintf(
            synthetic_name, sizeof(synthetic_name),
            "ds4_test_decode_pipeline_fast_%u", i);
        uint16_t synthetic_name_len = 0;
        uint64_t synthetic_hash = 0;
        filled = written > 0 &&
                 (size_t)written < sizeof(synthetic_name) &&
                 ds4_gpu_decode_pipeline_fast_key(
                     synthetic_name, 2,
                     DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                     &synthetic_name_len, &synthetic_hash);
        if (filled) {
            ds4_gpu_decode_pipeline_fast_cache_insert(
                synthetic_name, 2,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                synthetic_name_len,
                synthetic_hash, baseline);
        }
    }
    id<MTLComputePipelineState> full_fallback =
        ds4_gpu_get_mul_mv_pipeline(base_name, 2);
    const bool full_table_unchanged =
        filled && full_fallback == baseline &&
        g_decode_pipeline_fast_cache_entries ==
            DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS;

    ds4_gpu_decode_pipeline_fast_cache_reset();
    return populated_and_hit && inactive_unchanged && full_table_unchanged;
}

/*
 * Exercises the second SIMD-group constant key shape used by
 * ds4_gpu_get_mul_mv_ext_pipeline (name, nsg, nxpsg). Confirms the extended key
 * populates and hits, distinguishes a real nxpsg from the single-constant
 * sentinel, and coexists with the single-constant mv entries in the shared
 * 64-slot table. Uses baseline as a stand-in pipeline object so the guard does
 * not depend on compiling a specific extended-kernel function.
 */
int ds4_gpu_test_decode_pipeline_fast_lookup_ext(void) {
    if (!g_initialized || !g_pipeline_cache || !g_library || !g_device) {
        return 0;
    }

    static const char base_name[] = "kernel_mul_mv_id_mxfp4_f32";
    char copied_name[sizeof(base_name)];
    memcpy(copied_name, base_name, sizeof(base_name));

    id<MTLComputePipelineState> baseline =
        ds4_gpu_get_mul_mv_pipeline(base_name, 2);
    if (!baseline) return 0;

    ds4_gpu_decode_pipeline_fast_cache_reset();
    (void)ds4_gpu_set_decode_pipeline_fast_lookup(1);

    const int16_t ext_nsg = 2;
    const int16_t ext_nxpsg = 4;
    uint16_t ext_name_len = 0;
    uint64_t ext_hash = 0;
    const bool ext_key_valid =
        ds4_gpu_decode_pipeline_fast_key(
            copied_name, ext_nsg, ext_nxpsg, &ext_name_len, &ext_hash);
    id<MTLComputePipelineState> ext_populated = nil;
    if (ext_key_valid) {
        ds4_gpu_decode_pipeline_fast_cache_insert(
            copied_name, ext_nsg, ext_nxpsg, ext_name_len, ext_hash, baseline);
        ext_populated = ds4_gpu_decode_pipeline_fast_cache_lookup(
            copied_name, ext_nsg, ext_nxpsg, ext_name_len, ext_hash);
    }

    /*
     * The single-constant sentinel must not match an extended key even when the
     * name and nsg agree, proving the two key shapes never alias.
     */
    uint16_t none_name_len = 0;
    uint64_t none_hash = 0;
    const bool none_key_valid =
        ds4_gpu_decode_pipeline_fast_key(
            copied_name, ext_nsg,
            DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
            &none_name_len, &none_hash);
    id<MTLComputePipelineState> none_hit =
        none_key_valid ?
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                copied_name, ext_nsg,
                DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE,
                none_name_len, none_hash) : nil;

    const int16_t alt_nxpsg = 8;
    uint16_t alt_name_len = 0;
    uint64_t alt_hash = 0;
    const bool alt_key_valid =
        ds4_gpu_decode_pipeline_fast_key(
            copied_name, ext_nsg, alt_nxpsg, &alt_name_len, &alt_hash);
    id<MTLComputePipelineState> alt_hit =
        alt_key_valid ?
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                copied_name, ext_nsg, alt_nxpsg, alt_name_len, alt_hash) : nil;

    const bool ext_populated_and_hit =
        ext_key_valid &&
        ext_populated == baseline &&
        g_decode_pipeline_fast_cache_entries == 1u &&
        none_hit == nil &&
        alt_hit == nil;

    (void)ds4_gpu_set_decode_pipeline_fast_lookup(0);
    ds4_gpu_decode_pipeline_fast_cache_reset();
    return ext_populated_and_hit;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    uint16_t fast_name_len = 0;
    uint64_t fast_hash = 0;
    const bool fast_key_valid =
        g_decode_pipeline_fast_lookup_active &&
        ds4_gpu_decode_pipeline_fast_key(
            function_name, nsg, nxpsg, &fast_name_len, &fast_hash);
    if (fast_key_valid) {
        id<MTLComputePipelineState> fast_cached =
            ds4_gpu_decode_pipeline_fast_cache_lookup(
                function_name, nsg, nxpsg, fast_name_len, fast_hash);
        if (fast_cached) return fast_cached;
    }

    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (fast_key_valid) {
            ds4_gpu_decode_pipeline_fast_cache_insert(
                function_name, nsg, nxpsg, fast_name_len, fast_hash, cached);
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (fast_key_valid) {
        ds4_gpu_decode_pipeline_fast_cache_insert(
            function_name, nsg, nxpsg, fast_name_len, fast_hash, pipeline);
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     * The rollback switch restores the dictionary path for same-binary A/B.
     */
    static struct {
        bool m;
        int32_t nc;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.m == has_mask && memo.nc == ncpsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ has_mask, ncpsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ has_mask, ncpsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline(
        int32_t nqptg,
        int32_t ncpsg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     * The rollback switch restores the dictionary path for same-binary A/B.
     */
    static struct {
        int32_t nq;
        int32_t nc;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.nq == nqptg && memo.nc == ncpsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_blk_nqptg=%d_ncpsg=%d",
                     (int)nqptg, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ nqptg, ncpsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ nqptg, ncpsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        bc_mask,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg) {
    /*
     * Prefill and batched decode call this once per layer with identical
     * arguments, so memoize the last hit and skip the NSString key + dictionary
     * lookup on the hot path.  The generic cache below remains the fallback for
     * new variants.  The rollback switch restores the dictionary path for
     * same-binary A/B.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, bc;
        int32_t n10, n20, sg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    const bool memo_disabled =
        getenv("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO") != NULL;
    if (!memo_disabled && memo.pipeline && memo.fn != NULL &&
        strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.bc == bc_mask &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     bc_mask ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        if (!memo_disabled) {
            memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                                   has_scap, has_kvpad, bc_mask, ns10, ns20,
                                   nsg, cached };
        }
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    if (!memo_disabled) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, bc_mask, ns10, ns20,
                               nsg, pipeline };
    }
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        shared_kvpad,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, sp;
        int32_t n10, n20, sg, wg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    if (memo.pipeline && memo.fn != NULL && strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.sp == shared_kvpad &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg && memo.wg == nwg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_sharedpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     shared_kvpad ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                               nsg, nwg, cached };
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&shared_kvpad type:MTLDataTypeBool atIndex:405];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                           has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                           nsg, nwg, pipeline };
    return pipeline;
}

/* Pipeline for the RoPE-fused decode reduce. Same function constants as the
 * plain reduce so the split-K geometry is identical. */
static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_rope_pipeline(
        int32_t dv,
        int32_t nwg) {
    static int32_t memo_dv, memo_nwg;
    static id<MTLComputePipelineState> memo_pipeline;
    if (memo_pipeline && memo_dv == dv && memo_nwg == nwg) {
        return memo_pipeline;
    }
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_rope_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo_dv = dv; memo_nwg = nwg; memo_pipeline = cached;
        return cached;
    }
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];
    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce_rope"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce_rope not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }
    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce_rope pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }
    [g_pipeline_cache setObject:pipeline forKey:key];
    memo_dv = dv; memo_nwg = nwg; memo_pipeline = pipeline;
    return pipeline;
}

int ds4_gpu_decode_attn_rope_fuse_available(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_rope_tail_inplace_pair_affine_pipeline == nil) return 0;
    if (getenv("DS4_METAL_DISABLE_INPLACE_ROPE_PAIR") != NULL) return 0;
    if (getenv("DS4_METAL_DISABLE_AFFINE_ROPE_PAIR") != NULL) return 0;
    if (!ds4_gpu_device_name_contains("M3") && !ds4_gpu_device_name_contains("M5")) return 0;
    return 1;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline(
        int32_t dv,
        int32_t nwg) {
    /* Same per-layer memo pattern as the vec getter above. */
    static int32_t memo_dv, memo_nwg;
    static id<MTLComputePipelineState> memo_pipeline;
    if (memo_pipeline && memo_dv == dv && memo_nwg == nwg) {
        return memo_pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo_dv = dv; memo_nwg = nwg; memo_pipeline = cached;
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo_dv = dv; memo_nwg = nwg; memo_pipeline = pipeline;
    return pipeline;
}



static double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static ds4_gpu_stream_expert_timing_snapshot
ds4_gpu_stream_expert_timing_current(void) {
    return (ds4_gpu_stream_expert_timing_snapshot) {
        .selected_calls = g_stream_expert_timing_selected_calls,
        .selected_read_ms = g_stream_expert_timing_selected_read_ms,
        .selected_sync_ms = g_stream_expert_timing_selected_sync_ms,
        .selected_copy_ms = g_stream_expert_timing_selected_copy_ms,
        .selected_bind_ms = g_stream_expert_timing_selected_bind_ms,
        .split_layers = g_stream_expert_timing_split_layers,
        .split_resident_experts = g_stream_expert_timing_split_resident_experts,
        .split_missing_experts = g_stream_expert_timing_split_missing_experts,
        .split_resident_ms = g_stream_expert_timing_split_resident_ms,
        .split_missing_ms = g_stream_expert_timing_split_missing_ms,
        .split_missing_load_ms =
            g_stream_expert_timing_split_missing_load_ms,
        .split_missing_slot_ms =
            g_stream_expert_timing_split_missing_slot_ms,
        .split_missing_prune_ms =
            g_stream_expert_timing_split_missing_prune_ms,
        .split_missing_addr_ms =
            g_stream_expert_timing_split_missing_addr_ms,
        .split_missing_wait_ms =
            g_stream_expert_timing_split_missing_wait_ms,
        .load_calls = g_stream_expert_timing_load_calls,
        .load_prepare_ms = g_stream_expert_timing_load_prepare_ms,
        .load_pread_ms = g_stream_expert_timing_load_pread_ms,
        .load_install_ms = g_stream_expert_timing_load_install_ms,
        .prepare_batch_reuse_calls =
            g_stream_expert_timing_prepare_batch_reuse_calls,
        .prepare_batch_reuse_ms =
            g_stream_expert_timing_prepare_batch_reuse_ms,
        .prepare_buffer_calls =
            g_stream_expert_timing_prepare_buffer_calls,
        .prepare_buffer_ms =
            g_stream_expert_timing_prepare_buffer_ms,
        .prepare_task_experts =
            g_stream_expert_timing_prepare_task_experts,
        .prepare_task_ms =
            g_stream_expert_timing_prepare_task_ms,
        .reuse_scan_calls =
            g_stream_expert_timing_reuse_scan_calls,
        .reuse_scan_entries =
            g_stream_expert_timing_reuse_scan_entries,
        .reuse_scan_ms =
            g_stream_expert_timing_reuse_scan_ms,
        .reuse_clear_ms =
            g_stream_expert_timing_reuse_clear_ms,
        .readahead_calls =
            g_stream_expert_timing_readahead_calls,
        .readahead_bytes =
            g_stream_expert_timing_readahead_bytes,
        .readahead_ms =
            g_stream_expert_timing_readahead_ms,
        .cache_all_resident_layers =
            g_stream_expert_timing_cache_all_resident_layers,
        .cache_all_missing_layers =
            g_stream_expert_timing_cache_all_missing_layers,
        .cache_mixed_layers = g_stream_expert_timing_cache_mixed_layers,
        .cache_resident_experts =
            g_stream_expert_timing_cache_resident_experts,
        .cache_missing_experts =
            g_stream_expert_timing_cache_missing_experts,
    };
}

static uint64_t ds4_gpu_stream_expert_timing_delta_u64(
        uint64_t current,
        uint64_t previous) {
    return current >= previous ? current - previous : current;
}

static double ds4_gpu_stream_expert_timing_delta_f64(
        double current,
        double previous) {
    return current >= previous ? current - previous : current;
}

static ds4_gpu_stream_expert_timing_snapshot
ds4_gpu_stream_expert_timing_delta(
        ds4_gpu_stream_expert_timing_snapshot current,
        ds4_gpu_stream_expert_timing_snapshot previous) {
    return (ds4_gpu_stream_expert_timing_snapshot) {
        .selected_calls =
            ds4_gpu_stream_expert_timing_delta_u64(current.selected_calls,
                                                   previous.selected_calls),
        .selected_read_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_read_ms,
                                                   previous.selected_read_ms),
        .selected_sync_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_sync_ms,
                                                   previous.selected_sync_ms),
        .selected_copy_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_copy_ms,
                                                   previous.selected_copy_ms),
        .selected_bind_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_bind_ms,
                                                   previous.selected_bind_ms),
        .split_layers =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_layers,
                                                   previous.split_layers),
        .split_resident_experts =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_resident_experts,
                                                   previous.split_resident_experts),
        .split_missing_experts =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_missing_experts,
                                                   previous.split_missing_experts),
        .split_resident_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.split_resident_ms,
                                                   previous.split_resident_ms),
        .split_missing_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.split_missing_ms,
                                                   previous.split_missing_ms),
        .split_missing_load_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_load_ms,
                    previous.split_missing_load_ms),
        .split_missing_slot_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_slot_ms,
                    previous.split_missing_slot_ms),
        .split_missing_prune_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_prune_ms,
                    previous.split_missing_prune_ms),
        .split_missing_addr_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_addr_ms,
                    previous.split_missing_addr_ms),
        .split_missing_wait_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_wait_ms,
                    previous.split_missing_wait_ms),
        .load_calls =
            ds4_gpu_stream_expert_timing_delta_u64(current.load_calls,
                                                   previous.load_calls),
        .load_prepare_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_prepare_ms,
                                                   previous.load_prepare_ms),
        .load_pread_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_pread_ms,
                                                   previous.load_pread_ms),
        .load_install_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_install_ms,
                                                   previous.load_install_ms),
        .prepare_batch_reuse_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_batch_reuse_calls,
                    previous.prepare_batch_reuse_calls),
        .prepare_batch_reuse_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_batch_reuse_ms,
                    previous.prepare_batch_reuse_ms),
        .prepare_buffer_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_buffer_calls,
                    previous.prepare_buffer_calls),
        .prepare_buffer_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_buffer_ms,
                    previous.prepare_buffer_ms),
        .prepare_task_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_task_experts,
                    previous.prepare_task_experts),
        .prepare_task_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_task_ms,
                    previous.prepare_task_ms),
        .reuse_scan_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.reuse_scan_calls,
                    previous.reuse_scan_calls),
        .reuse_scan_entries =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.reuse_scan_entries,
                    previous.reuse_scan_entries),
        .reuse_scan_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.reuse_scan_ms,
                    previous.reuse_scan_ms),
        .reuse_clear_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.reuse_clear_ms,
                    previous.reuse_clear_ms),
        .readahead_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.readahead_calls,
                    previous.readahead_calls),
        .readahead_bytes =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.readahead_bytes,
                    previous.readahead_bytes),
        .readahead_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.readahead_ms,
                    previous.readahead_ms),
        .cache_all_resident_layers =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_all_resident_layers,
                    previous.cache_all_resident_layers),
        .cache_all_missing_layers =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_all_missing_layers,
                    previous.cache_all_missing_layers),
        .cache_mixed_layers =
            ds4_gpu_stream_expert_timing_delta_u64(current.cache_mixed_layers,
                                                   previous.cache_mixed_layers),
        .cache_resident_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_resident_experts,
                    previous.cache_resident_experts),
        .cache_missing_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_missing_experts,
                    previous.cache_missing_experts),
    };
}

static int ds4_gpu_stream_expert_timing_has_data(
        ds4_gpu_stream_expert_timing_snapshot s) {
    return s.selected_calls != 0 ||
           s.split_layers != 0 ||
           s.load_calls != 0 ||
           s.cache_all_resident_layers != 0 ||
           s.cache_all_missing_layers != 0 ||
           s.cache_mixed_layers != 0;
}

static void ds4_gpu_stream_expert_timing_print(
        const char *scope,
        ds4_gpu_stream_expert_timing_snapshot s) {
    if (!ds4_gpu_stream_expert_timing_has_data(s)) return;
    const double selected_calls = (double)s.selected_calls;
    const double split_layers = (double)s.split_layers;
    const double selected_read_avg =
        selected_calls != 0.0 ? s.selected_read_ms / selected_calls : 0.0;
    const double selected_sync_avg =
        selected_calls != 0.0 ? s.selected_sync_ms / selected_calls : 0.0;
    const double selected_copy_avg =
        selected_calls != 0.0 ? s.selected_copy_ms / selected_calls : 0.0;
    const double selected_bind_avg =
        selected_calls != 0.0 ? s.selected_bind_ms / selected_calls : 0.0;
    const double split_resident_avg =
        split_layers != 0.0 ? s.split_resident_ms / split_layers : 0.0;
    const double split_missing_avg =
        split_layers != 0.0 ? s.split_missing_ms / split_layers : 0.0;
    const double split_missing_load_avg =
        split_layers != 0.0 ?
            s.split_missing_load_ms / split_layers : 0.0;
    const double split_missing_slot_avg =
        split_layers != 0.0 ?
            s.split_missing_slot_ms / split_layers : 0.0;
    const double split_missing_prune_avg =
        split_layers != 0.0 ?
            s.split_missing_prune_ms / split_layers : 0.0;
    const double split_missing_addr_avg =
        split_layers != 0.0 ?
            s.split_missing_addr_ms / split_layers : 0.0;
    const double split_missing_wait_avg =
        split_layers != 0.0 ?
            s.split_missing_wait_ms / split_layers : 0.0;
    const double load_calls = (double)s.load_calls;
    const double load_prepare_avg =
        load_calls != 0.0 ? s.load_prepare_ms / load_calls : 0.0;
    const double load_pread_avg =
        load_calls != 0.0 ? s.load_pread_ms / load_calls : 0.0;
    const double load_install_avg =
        load_calls != 0.0 ? s.load_install_ms / load_calls : 0.0;
    const double prepare_batch_reuse_avg =
        s.prepare_batch_reuse_calls != 0 ?
            s.prepare_batch_reuse_ms /
                (double)s.prepare_batch_reuse_calls : 0.0;
    const double prepare_buffer_avg =
        s.prepare_buffer_calls != 0 ?
            s.prepare_buffer_ms / (double)s.prepare_buffer_calls : 0.0;
    const double prepare_task_avg =
        s.prepare_task_experts != 0 ?
            s.prepare_task_ms / (double)s.prepare_task_experts : 0.0;
    const double reuse_scan_avg =
        s.reuse_scan_calls != 0 ?
            s.reuse_scan_ms / (double)s.reuse_scan_calls : 0.0;
    const double reuse_scan_entries_avg =
        s.reuse_scan_calls != 0 ?
            (double)s.reuse_scan_entries / (double)s.reuse_scan_calls : 0.0;
    const double readahead_avg =
        s.readahead_calls != 0 ?
            s.readahead_ms / (double)s.readahead_calls : 0.0;
    const double split_resident_experts_avg =
        split_layers != 0.0 ?
            (double)s.split_resident_experts / split_layers : 0.0;
    const double split_missing_experts_avg =
        split_layers != 0.0 ?
            (double)s.split_missing_experts / split_layers : 0.0;
    const uint64_t cache_layers =
        s.cache_all_resident_layers +
        s.cache_all_missing_layers +
        s.cache_mixed_layers;
    const double cache_layer_count = (double)cache_layers;
    const double cache_resident_experts_avg =
        cache_layer_count != 0.0 ?
            (double)s.cache_resident_experts / cache_layer_count : 0.0;
    const double cache_missing_experts_avg =
        cache_layer_count != 0.0 ?
            (double)s.cache_missing_experts / cache_layer_count : 0.0;
    fprintf(stderr,
            "ds4:   streaming expert timing %s selected_calls=%llu read_avg=%.3f ms sync_avg=%.3f ms copy_avg=%.3f ms bind_avg=%.3f ms read_total=%.3f ms sync_total=%.3f ms copy_total=%.3f ms bind_total=%.3f ms split_layers=%llu resident_experts_avg=%.2f missing_experts_avg=%.2f resident_submit_avg=%.3f ms missing_bind_avg=%.3f ms resident_submit_total=%.3f ms missing_bind_total=%.3f ms missing_load_avg=%.3f ms missing_slot_avg=%.3f ms missing_prune_avg=%.3f ms missing_addr_avg=%.3f ms missing_wait_avg=%.3f ms missing_wait_total=%.3f ms load_calls=%llu load_prepare_avg=%.3f ms load_pread_avg=%.3f ms load_install_avg=%.3f ms prepare_batch_reuse_calls=%llu prepare_batch_reuse_avg=%.3f ms prepare_batch_reuse_total=%.3f ms prepare_buffer_calls=%llu prepare_buffer_avg=%.3f ms prepare_buffer_total=%.3f ms prepare_task_experts=%llu prepare_task_avg=%.3f ms prepare_task_total=%.3f ms reuse_scan_calls=%llu reuse_scan_entries_avg=%.1f reuse_scan_avg=%.3f ms reuse_scan_total=%.3f ms reuse_clear_total=%.3f ms readahead_calls=%llu readahead_avg=%.3f ms readahead_total=%.3f ms readahead_gib=%.2f cache_all_resident=%llu cache_all_missing=%llu cache_mixed=%llu cache_resident_avg=%.2f cache_missing_avg=%.2f\n",
            scope ? scope : "total",
            (unsigned long long)s.selected_calls,
            selected_read_avg,
            selected_sync_avg,
            selected_copy_avg,
            selected_bind_avg,
            s.selected_read_ms,
            s.selected_sync_ms,
            s.selected_copy_ms,
            s.selected_bind_ms,
            (unsigned long long)s.split_layers,
            split_resident_experts_avg,
            split_missing_experts_avg,
            split_resident_avg,
            split_missing_avg,
            s.split_resident_ms,
            s.split_missing_ms,
            split_missing_load_avg,
            split_missing_slot_avg,
            split_missing_prune_avg,
            split_missing_addr_avg,
            split_missing_wait_avg,
            s.split_missing_wait_ms,
            (unsigned long long)s.load_calls,
            load_prepare_avg,
            load_pread_avg,
            load_install_avg,
            (unsigned long long)s.prepare_batch_reuse_calls,
            prepare_batch_reuse_avg,
            s.prepare_batch_reuse_ms,
            (unsigned long long)s.prepare_buffer_calls,
            prepare_buffer_avg,
            s.prepare_buffer_ms,
            (unsigned long long)s.prepare_task_experts,
            prepare_task_avg,
            s.prepare_task_ms,
            (unsigned long long)s.reuse_scan_calls,
            reuse_scan_entries_avg,
            reuse_scan_avg,
            s.reuse_scan_ms,
            s.reuse_clear_ms,
            (unsigned long long)s.readahead_calls,
            readahead_avg,
            s.readahead_ms,
            ds4_gpu_gib(s.readahead_bytes),
            (unsigned long long)s.cache_all_resident_layers,
            (unsigned long long)s.cache_all_missing_layers,
            (unsigned long long)s.cache_mixed_layers,
            cache_resident_experts_avg,
            cache_missing_experts_avg);
}

static void ds4_gpu_print_task_memory_report(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(),
                                       TASK_VM_INFO,
                                       (task_info_t)&info,
                                       &count);
    if (kr != KERN_SUCCESS) return;

    fprintf(stderr,
            "ds4:   macOS task memory footprint %.2f GiB, resident %.2f GiB, virtual %.2f GiB\n",
            ds4_gpu_gib((uint64_t)info.phys_footprint),
            ds4_gpu_gib((uint64_t)info.resident_size),
            ds4_gpu_gib((uint64_t)info.virtual_size));
}

void ds4_gpu_print_memory_report(const char *label) {
    uint64_t cached_prefill_mask_bytes = 0;
    uint64_t cached_prefill_blk_bytes = 0;
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        const ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
            &g_zero_prefix_prefill_mask_cache[i];
        if (entry->mask) cached_prefill_mask_bytes += entry->mask_bytes;
        if (entry->blk) cached_prefill_blk_bytes += entry->blk_bytes;
    }
    const uint64_t scratch =
        (uint64_t)g_flash_attn_mask_bytes +
        (uint64_t)g_flash_attn_zero_mask_bytes +
        cached_prefill_mask_bytes +
        (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        cached_prefill_blk_bytes +
        (uint64_t)g_flash_attn_ring_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_glm_flash_attn_mask_bytes +
        (uint64_t)g_compressor_pool_kv_bytes +
        (uint64_t)g_compressor_pool_score_bytes +
        (uint64_t)g_compressor_pool_score_cont_bytes +
        (uint64_t)g_compressor_pool_softmax_bytes +
        (uint64_t)g_compressor_pool_product_bytes +
        (uint64_t)g_compressor_store_ape_bytes +
        (uint64_t)g_compressor_store_score_bytes +
        (uint64_t)g_embed_rows_bytes +
        (uint64_t)g_router_selection_bytes +
        (uint64_t)g_router_weight_sum_bytes +
        (uint64_t)g_indexer_head_scores_bytes +
        (uint64_t)g_indexer_topk_bytes +
        (uint64_t)g_indexed_topk_bytes +
        (uint64_t)g_f16_round_scratch_bytes +
        (uint64_t)g_q8_prefill_scratch_bytes + (uint64_t)g_qwen4_conv_halo_bytes +
        (uint64_t)g_raw_store_round_bytes +
        (uint64_t)g_moe_gate_scratch_bytes +
        (uint64_t)g_moe_down_scratch_bytes +
        (uint64_t)g_moe_id_map_bytes + (uint64_t)g_moe_packed_rhs_bytes +
        (uint64_t)g_moe_q4_gate_slots_bytes +
        (uint64_t)g_moe_q4_up_slots_bytes +
        (uint64_t)g_moe_q4_down_slots_bytes;

    const ds4_metal_tensor_stats tensor_stats = ds4_native_metal_tensor_stats();
    const uint64_t tensor_live_snap = tensor_stats.live_bytes;
    const uint64_t tensor_peak_snap = tensor_stats.peak_bytes;

    uint64_t tracked_live = tensor_live_snap;
    if (tracked_live > UINT64_MAX - g_stream_expert_cache_bytes) {
        tracked_live = UINT64_MAX;
    } else {
        tracked_live += g_stream_expert_cache_bytes;
    }

    const bool color = ds4_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";
    fprintf(stderr,
            "%sds4: Metal memory%s%s: runtime %.2f GiB + streaming experts %.2f GiB = %s%.2f GiB tracked live%s\n",
            green,
            label && label[0] ? " " : "",
            label && label[0] ? label : "",
            ds4_gpu_gib(tensor_live_snap),
            ds4_gpu_gib(g_stream_expert_cache_bytes),
            bright_green,
            ds4_gpu_gib(tracked_live),
            reset);
    if (color) fputs(green, stderr);
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_gpu_mib(tensor_live_snap),
            ds4_gpu_mib(tensor_peak_snap));
    ds4_gpu_print_task_memory_report();
    fprintf(stderr,
            "ds4:   mmap model wrapper creations %llu buffers %.2f GiB cumulative, %.2f GiB max (not copied)\n",
            (unsigned long long)g_model_views.wrap_count,
            ds4_gpu_gib(g_model_views.wrap_bytes),
            ds4_gpu_gib(g_model_views.wrap_max_bytes));
    ds4_metal_exact_cache_report();
    if (g_stream_expert_gate_probes) {
        fprintf(stderr, "ds4:   staged experts probes=%llu submissions=%llu\n",
            (unsigned long long)g_stream_expert_gate_probes,
            (unsigned long long)g_stream_expert_gate_submissions);
    }
    if (g_stream_expert_pending_joins) {
        fprintf(stderr, "ds4:   streaming pending joins=%llu wait_ms=%.3f (CPU wait only)\n",
            (unsigned long long)g_stream_expert_pending_joins, g_stream_expert_pending_wait_ms);
    }
    if (g_stream_expert_cache_hits != 0 ||
        g_stream_expert_cache_misses != 0 ||
        g_stream_expert_cache_bytes != 0) {
        const uint64_t budget = ds4_gpu_stream_expert_cache_configured_budget();
        uint64_t target_bytes = 0;
        if (budget != 0 && g_stream_expert_cache_expert_bytes != 0) {
            target_bytes =
                budget > UINT64_MAX / g_stream_expert_cache_expert_bytes ?
                    UINT64_MAX :
                    budget * g_stream_expert_cache_expert_bytes;
        }
        const uint64_t lookups = g_stream_expert_cache_hits + g_stream_expert_cache_misses;
        const double hit_rate = lookups ?
            (double)g_stream_expert_cache_hits / (double)lookups : 0.0;
        if (g_stream_expert_cache_evict_advise_bytes != 0 ||
            g_stream_expert_cache_willneed_advise_bytes != 0 ||
            g_stream_expert_cache_pread_bytes != 0) {
            fprintf(stderr,
                    "ds4:   streaming expert cache budget=%llu experts entries=%u expert=%.2f MiB target=%.2f GiB live=%.2f GiB, hits=%llu misses=%llu hit_rate=%.3f wraps=%llu evictions=%llu buffer_allocs=%llu buffer_reuses=%llu evict_dontneed=%.2f GiB miss_willneed=%.2f GiB miss_pread=%.2f GiB pread_ms=%.3f\n",
                    (unsigned long long)budget,
                    g_stream_expert_cache_entry_count,
                    ds4_gpu_mib(g_stream_expert_cache_expert_bytes),
                    ds4_gpu_gib(target_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_bytes),
                    (unsigned long long)g_stream_expert_cache_hits,
                    (unsigned long long)g_stream_expert_cache_misses,
                    hit_rate,
                    (unsigned long long)g_stream_expert_cache_wraps,
                    (unsigned long long)g_stream_expert_cache_evictions,
                    (unsigned long long)g_stream_expert_cache_buffer_allocs,
                    (unsigned long long)g_stream_expert_cache_buffer_reuses,
                    ds4_gpu_gib(g_stream_expert_cache_evict_advise_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_willneed_advise_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_pread_bytes),
                    g_stream_expert_cache_pread_ms);
        } else {
            fprintf(stderr,
                    "ds4:   streaming expert cache budget=%llu experts entries=%u expert=%.2f MiB target=%.2f GiB live=%.2f GiB, hits=%llu misses=%llu hit_rate=%.3f wraps=%llu evictions=%llu buffer_allocs=%llu buffer_reuses=%llu\n",
                    (unsigned long long)budget,
                    g_stream_expert_cache_entry_count,
                    ds4_gpu_mib(g_stream_expert_cache_expert_bytes),
                    ds4_gpu_gib(target_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_bytes),
                    (unsigned long long)g_stream_expert_cache_hits,
                    (unsigned long long)g_stream_expert_cache_misses,
                    hit_rate,
                    (unsigned long long)g_stream_expert_cache_wraps,
                    (unsigned long long)g_stream_expert_cache_evictions,
                    (unsigned long long)g_stream_expert_cache_buffer_allocs,
                    (unsigned long long)g_stream_expert_cache_buffer_reuses);
        }
        if (g_stream_expert_cache_mlock_bytes != 0 ||
            g_stream_expert_cache_mlock_failures != 0) {
            fprintf(stderr,
                    "ds4:   streaming expert buffer mlock locked=%.2f GiB failed=%.2f GiB failures=%llu time=%.3f ms\n",
                    ds4_gpu_gib(g_stream_expert_cache_mlock_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_mlock_fail_bytes),
                    (unsigned long long)g_stream_expert_cache_mlock_failures,
                    g_stream_expert_cache_mlock_ms);
        }
        if (ds4_gpu_stream_expert_timing_summary_enabled()) {
            fprintf(stderr,
                    "ds4:   streaming cache aging tokens=%llu first_layer=%u decay_token=%llu\n",
                    (unsigned long long)g_stream_expert_cache_decode_tokens,
                    g_stream_expert_cache_first_decode_layer,
                    (unsigned long long)g_stream_expert_cache_hotness_decay_token);
            const ds4_gpu_stream_expert_timing_snapshot total =
                ds4_gpu_stream_expert_timing_current();
            if (ds4_gpu_stream_expert_timing_has_data(total)) {
                const ds4_gpu_stream_expert_timing_snapshot delta =
                    ds4_gpu_stream_expert_timing_delta(
                            total,
                            g_stream_expert_timing_last_report);
                ds4_gpu_stream_expert_timing_print("total", total);
                ds4_gpu_stream_expert_timing_print("delta", delta);
                g_stream_expert_timing_last_report = total;
            }
        }
        if (getenv("DS4_METAL_STREAMING_EXPERT_LAYER_STATS") != NULL) {
            fprintf(stderr, "ds4:   streaming expert cache per-layer stats:\n");
            for (uint32_t layer = 0;
                 layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
                 layer++) {
                const uint64_t hits = g_stream_expert_cache_layer_hits[layer];
                const uint64_t misses = g_stream_expert_cache_layer_misses[layer];
                const uint64_t lookups = hits + misses;
                const uint64_t evictions =
                    g_stream_expert_cache_layer_evictions[layer];
                const uint64_t pread_bytes =
                    g_stream_expert_cache_layer_pread_bytes[layer];
                const double pread_ms =
                    g_stream_expert_cache_layer_pread_ms[layer];
                const uint32_t cached =
                    g_stream_expert_cache_layer_count[layer];
                const uint32_t layer_slots =
                    ds4_gpu_stream_expert_cache_configured_count() != 0 ?
                        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT : 0;
                if (lookups == 0 && evictions == 0 && pread_bytes == 0 &&
                    cached == 0) {
                    continue;
                }
                const double layer_hit_rate = lookups ?
                    (double)hits / (double)lookups : 0.0;
                fprintf(stderr,
                        "ds4:     layer=%u layer_slots=%u cached=%u hits=%llu misses=%llu hit_rate=%.3f evictions=%llu miss_pread=%.2f GiB pread_ms=%.3f\n",
                        layer,
                        layer_slots,
                        cached,
                        (unsigned long long)hits,
                        (unsigned long long)misses,
                        layer_hit_rate,
                        (unsigned long long)evictions,
                        ds4_gpu_gib(pread_bytes),
                        pread_ms);
            }
            if (getenv("DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA") != NULL) {
                fprintf(stderr, "ds4:   streaming expert cache per-layer delta:\n");
                for (uint32_t layer = 0;
                     layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
                     layer++) {
                    const uint64_t hits =
                        g_stream_expert_cache_layer_hits[layer];
                    const uint64_t misses =
                        g_stream_expert_cache_layer_misses[layer];
                    const uint64_t evictions =
                        g_stream_expert_cache_layer_evictions[layer];
                    const uint64_t pread_bytes =
                        g_stream_expert_cache_layer_pread_bytes[layer];
                    const double pread_ms =
                        g_stream_expert_cache_layer_pread_ms[layer];
                    const uint64_t delta_hits =
                        hits - g_stream_expert_cache_layer_last_hits[layer];
                    const uint64_t delta_misses =
                        misses - g_stream_expert_cache_layer_last_misses[layer];
                    const uint64_t delta_evictions =
                        evictions -
                        g_stream_expert_cache_layer_last_evictions[layer];
                    const uint64_t delta_pread_bytes =
                        pread_bytes -
                        g_stream_expert_cache_layer_last_pread_bytes[layer];
                    const double delta_pread_ms =
                        pread_ms -
                        g_stream_expert_cache_layer_last_pread_ms[layer];
                    const uint64_t lookups = delta_hits + delta_misses;
                    if (lookups == 0 && delta_evictions == 0 &&
                        delta_pread_bytes == 0) {
                        continue;
                    }
                    const double hit_rate = lookups ?
                        (double)delta_hits / (double)lookups : 0.0;
                    fprintf(stderr,
                            "ds4:     layer=%u cached=%u hits=%llu misses=%llu hit_rate=%.3f evictions=%llu miss_pread=%.2f GiB pread_ms=%.3f\n",
                            layer,
                            g_stream_expert_cache_layer_count[layer],
                            (unsigned long long)delta_hits,
                            (unsigned long long)delta_misses,
                            hit_rate,
                            (unsigned long long)delta_evictions,
                            ds4_gpu_gib(delta_pread_bytes),
                            delta_pread_ms);
                    g_stream_expert_cache_layer_last_hits[layer] = hits;
                    g_stream_expert_cache_layer_last_misses[layer] = misses;
                    g_stream_expert_cache_layer_last_evictions[layer] =
                        evictions;
                    g_stream_expert_cache_layer_last_pread_bytes[layer] =
                        pread_bytes;
                    g_stream_expert_cache_layer_last_pread_ms[layer] =
                        pread_ms;
                }
            }
        }
    }
    fprintf(stderr,
            "ds4:   model residency requests %llu%s\n",
            (unsigned long long)g_model_views.residency_count,
            g_ssd_streaming_mode ? " (ssd-streaming)" :
            (getenv("DS4_METAL_NO_RESIDENCY") != NULL ? " (disabled)" : ""));
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   accelerated Metal path %s%s\n",
            ds4_gpu_mpp_available() ? "enabled" : "disabled",
            g_quality_mode ? " by --quality" :
                (!g_metal4_tensor_api_enabled ? " (tensor API unavailable)" : ""));
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   scratch %.2f MiB (flash mask %.2f, pad %.2f, tmp %.2f, blk %.2f, ring %.2f, kv %.2f, compressor %.2f, router %.2f, indexer %.2f, moe %.2f, f16 %.2f, raw-store %.2f)\n",
            ds4_gpu_mib(scratch),
            ds4_gpu_mib((uint64_t)g_flash_attn_mask_bytes +
                        (uint64_t)g_glm_flash_attn_mask_bytes +
                        (uint64_t)g_flash_attn_zero_mask_bytes +
                        cached_prefill_mask_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_pad_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_tmp_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_blk_bytes +
                        cached_prefill_blk_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_ring_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_kv_bytes),
            ds4_gpu_mib((uint64_t)g_compressor_pool_kv_bytes +
                          (uint64_t)g_compressor_pool_score_bytes +
                          (uint64_t)g_compressor_pool_score_cont_bytes +
                          (uint64_t)g_compressor_pool_softmax_bytes +
                          (uint64_t)g_compressor_pool_product_bytes +
                          (uint64_t)g_compressor_store_ape_bytes +
                          (uint64_t)g_compressor_store_score_bytes +
                          (uint64_t)g_embed_rows_bytes),
            ds4_gpu_mib((uint64_t)g_router_selection_bytes +
                          (uint64_t)g_router_weight_sum_bytes),
            ds4_gpu_mib((uint64_t)g_indexer_head_scores_bytes +
                          (uint64_t)g_indexer_topk_bytes +
                          (uint64_t)g_indexed_topk_bytes),
            ds4_gpu_mib((uint64_t)g_moe_gate_scratch_bytes +
                          (uint64_t)g_moe_down_scratch_bytes +
                          (uint64_t)g_moe_id_map_bytes + (uint64_t)g_moe_packed_rhs_bytes +
                          (uint64_t)g_moe_q4_gate_slots_bytes +
                          (uint64_t)g_moe_q4_up_slots_bytes +
                          (uint64_t)g_moe_q4_down_slots_bytes),
            ds4_gpu_mib((uint64_t)g_f16_round_scratch_bytes),
            ds4_gpu_mib((uint64_t)g_raw_store_round_bytes));
    if (color) fputs(reset, stderr);
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void ds4_gpu_set_glm_model(bool enabled) {
    g_glm_model_mode = enabled ? 1 : 0;
}

void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    ds4_gpu_stream_expert_cache_clear_all(1);
    /* clear_all joins outstanding reads before changing the workers' mode. */
    const char *metrics = getenv("DS4_METAL_STREAMING_EXPERT_IO_METRICS");
    g_stream_expert_io_metrics_enabled = metrics && *metrics && strcmp(metrics, "0");
    if (g_ssd_streaming_mode) {
        fprintf(stderr,
                "ds4: Metal SSD streaming mode enabled; full model residency and warmup are skipped\n");
    }
}

int ds4_gpu_stream_expert_io_metrics_read(ds4_gpu_stream_expert_io_metrics *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    out->explicit_io_metrics_enabled = g_stream_expert_io_metrics_enabled;
    out->pread_calls = __atomic_load_n(&g_stream_expert_io_pread_calls, __ATOMIC_RELAXED);
    out->pread_requested_bytes = __atomic_load_n(&g_stream_expert_io_requested_bytes, __ATOMIC_RELAXED);
    out->pread_returned_bytes = __atomic_load_n(&g_stream_expert_io_returned_bytes, __ATOMIC_RELAXED);
    out->cache_loader_returned_bytes = g_stream_expert_cache_pread_bytes;
    out->cache_hits = g_stream_expert_cache_hits;
    out->cache_misses = g_stream_expert_cache_misses;
    out->cache_evictions = g_stream_expert_cache_evictions;
    struct rusage_info_v4 process = {0};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&process) == 0) {
        out->process_io_valid = true;
        out->process_disk_read_bytes = process.ri_diskio_bytesread;
    }
    return 1;
}

void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    g_glm_streaming_prefill_full_layer_runtime = enabled ? 1 : 0;
}

void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    /* MTLBuffer.contents returns an inner pointer. ARC autoreleases its
     * receiver while clearing address slots and unlocking slabs; C callers
     * need a pool here so discarded cache buffers can actually be released. */
    @autoreleasepool {
        if (experts > DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) {
            experts = DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES;
        }
        const uint32_t previous = g_stream_expert_cache_budget_override;
        if (experts && previous && experts < previous &&
            ds4_gpu_stream_expert_cache_trim(experts)) {
            g_stream_expert_cache_budget_override = experts;
            return;
        }
        g_stream_expert_cache_budget_override = experts;
        ds4_gpu_stream_expert_cache_clear_all(1);
    }
}

void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    /*
     * Pre-seed the cache's single slab size class with the model's uniform
     * per-expert bytes (first routed layer). With a mixed-precision GGUF this
     * pins the class to the majority layers so the boosted ones are rejected
     * deterministically from startup, instead of depending on which layer
     * happens to touch the cache first.
     */
    g_stream_expert_cache_expert_bytes = bytes;
}

uint64_t ds4_gpu_recommended_working_set_size(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_device) return 0;
    return (uint64_t)[g_device recommendedMaxWorkingSetSize];
}


static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"// Gate-time prefetch: one dispatch over up to 8 ranges given as GPU\n"
"// addresses (bindless views over the model), 128 B per thread, ranges\n"
"// assigned by cumulative line counts.  One launch instead of one per\n"
"// range: the fixed cost per dispatch is what a rank pays when its peer's\n"
"// partial is already there.\n"
"kernel void kernel_touch_u8_stride_table(\n"
"        constant ulong        *addr       [[buffer(0)]],\n"
"        constant uint         *lines      [[buffer(1)]],\n"
"        constant uint         &count      [[buffer(2)]],\n"
"        device uchar          *dst        [[buffer(3)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    uint line = gid;\n"
"    uint i = 0;\n"
"    while (i < count && line >= lines[i]) { line -= lines[i]; i++; }\n"
"    if (i >= count) return;\n"
"    device const uchar *src = (device const uchar *)addr[i];\n"
"    dst[gid] = src[(ulong)line * 128ul];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n"
"\n";

static NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  Environment overrides are still honored so a diagnostic
     * run can swap one source file without changing the executable.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"DS4_METAL_FLASH_ATTN_SOURCE", @"src/metal/flash_attn.metal"],
        @[@"DS4_METAL_DENSE_SOURCE",      @"src/metal/dense.metal"],
        @[@"DS4_METAL_GLM53_BF16_SOURCE", @"src/metal/glm53_bf16.metal"],
        @[@"DS4_METAL_GLM53_VISION_SOURCE", @"src/metal/glm53_vision.metal"],
        @[@"DS4_METAL_DEEPSEEK4_VISION_SOURCE", @"src/metal/deepseek4_vision.metal"],
        @[@"DS4_METAL_GLM53_KDA_SOURCE",  @"src/metal/glm53_kda.metal"],
        @[@"DS4_METAL_MOE_SOURCE",        @"src/metal/moe.metal"],
        @[@"DS4_METAL_DSV4_HC_SOURCE",    @"src/metal/dsv4_hc.metal"],
        @[@"DS4_METAL_UNARY_SOURCE",      @"src/metal/unary.metal"],
        @[@"DS4_METAL_DSV4_KV_SOURCE",    @"src/metal/dsv4_kv.metal"],
        @[@"DS4_METAL_DSV41_SOURCE",      @"src/metal/dsv41.metal"],
        @[@"DS4_METAL_DSV4_ROPE_SOURCE",  @"src/metal/dsv4_rope.metal"],
        @[@"DS4_METAL_DSV4_MISC_SOURCE",  @"src/metal/dsv4_misc.metal"],
        @[@"DS4_METAL_ARGSORT_SOURCE",    @"src/metal/argsort.metal"],
        @[@"DS4_METAL_CPY_SOURCE",        @"src/metal/cpy.metal"],
        @[@"DS4_METAL_CONCAT_SOURCE",     @"src/metal/concat.metal"],
        @[@"DS4_METAL_GET_ROWS_SOURCE",   @"src/metal/get_rows.metal"],
        @[@"DS4_METAL_SUM_ROWS_SOURCE",   @"src/metal/sum_rows.metal"],
        @[@"DS4_METAL_SOFTMAX_SOURCE",    @"src/metal/softmax.metal"],
        @[@"DS4_METAL_REPEAT_SOURCE",     @"src/metal/repeat.metal"],
        @[@"DS4_METAL_GLU_SOURCE",        @"src/metal/glu.metal"],
        @[@"DS4_METAL_NORM_SOURCE",       @"src/metal/norm.metal"],
        @[@"DS4_METAL_BIN_SOURCE",        @"src/metal/bin.metal"],
        @[@"DS4_METAL_SET_ROWS_SOURCE",   @"src/metal/set_rows.metal"],
        @[@"DS4_METAL_QWEN4_SOURCE",      @"src/metal/qwen4.metal"],
        @[@"DS4_METAL_QWEN4_VISION_SOURCE", @"src/metal/qwen4_vision.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0]) {
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr,
                    "ds4: Metal source %s not found (set %s to override)\n",
                    [spec[1] UTF8String], [spec[0] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef ds4_metal_embedding_rows_args ds4_gpu_get_rows_args;

typedef ds4_metal_embedding_quant_args ds4_gpu_get_rows_q8_0_args;

typedef ds4_metal_embedding_repeat_args ds4_gpu_repeat_args;

/* ds4_gpu_set_rows_args lives once in compressor.h. */

/* ds4_gpu_concat_args lives once in compressor.h. */

typedef ds4_metal_copy_args ds4_gpu_cpy_args;

static ds4_gpu_cpy_args ds4_gpu_make_cpy_1d_args(
        uint32_t n,
        uint64_t src_elem,
        uint64_t dst_elem) {
    return (ds4_gpu_cpy_args) {
        .nk0 = (int64_t)n,
        .ne00 = (int64_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = src_elem,
        .nb01 = (uint64_t)n * src_elem,
        .nb02 = (uint64_t)n * src_elem,
        .nb03 = (uint64_t)n * src_elem,
        .ne0 = (int64_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = dst_elem,
        .nb1 = (uint64_t)n * dst_elem,
        .nb2 = (uint64_t)n * dst_elem,
        .nb3 = (uint64_t)n * dst_elem,
    };
}

static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static float ds4_gpu_negative_infinity(void) {
    union { uint32_t u; float f; } v = { 0xff800000u };
    return v.f;
}

static float ds4_gpu_positive_infinity(void) {
    union { uint32_t u; float f; } v = { 0x7f800000u };
    return v.f;
}

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);


static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value);

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n);

/* ds4_gpu_glu_args is declared once in routed_batch.h. */

/* ds4_gpu_add_flat_args lives once in compressor.h. */

/* ds4_gpu_bin_args is declared once in routed_batch.h. */

typedef ds4_metal_attention_unary_args ds4_gpu_unary_args;

static ds4_gpu_bin_args ds4_gpu_make_bin_rows_args(uint32_t n, uint32_t rows, uint32_t rhs_n) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t rhs_row_bytes = (uint64_t)rhs_n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)rhs_n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = rhs_row_bytes,
        .nb13 = rhs_row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_unary_args ds4_gpu_make_unary_rows_args(
        uint32_t n,
        uint32_t rows,
        int      c4,
        float    scale,
        float    bias) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint32_t n_kernel = c4 ? n / 4u : n;
    return (ds4_gpu_unary_args) {
        .ne00 = (int32_t)n_kernel,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne0 = (int32_t)n_kernel,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .slope = 0.0f,
        .scale = scale,
        .bias = bias,
        .val = 0.0f,
        .min = 0.0f,
        .max = 0.0f,
    };
}


static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off);

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows);

typedef ds4_metal_mv_args ds4_gpu_q8_0_matvec_args;

typedef ds4_metal_mm_args ds4_gpu_mul_mm_args;

typedef ds4_metal_mv_ext_args ds4_gpu_mul_mv_ext_args;

typedef ds4_gpu_q8_0_matvec_args ds4_gpu_f16_matvec_args;

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_f16_matvec_args ds4_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    return (ds4_gpu_f16_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_f32_mv_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_vec) {
    const uint64_t row_bytes = in_dim * sizeof(float);
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_vec,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_vec * sizeof(float),
        .nb13 = in_dim * n_vec * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_vec,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} ds4_gpu_mv_dispatch;

static int ds4_gpu_tp_world_is_two(void);

static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void) {
    const uint64_t default_nsg = ds4_gpu_tp_world_is_two() ? 2u : 4u;
    const int16_t nsg =
        (int16_t)ds4_gpu_env_u64("DS4_METAL_Q8_MV_NSG", default_nsg, 1u, 8u);
    return (ds4_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

/* Single-token F16/F32 matvecs with few output rows (the Qwen hyper-connection
 * low-rank down projections and routers) launch one row per SIMD group on M5,
 * where two-row tiles leave most of the 40 cores idle.  The per-row K walk,
 * simdgroup count and reduction tree are unchanged.  DS4_METAL_PLAIN_MV_NR0
 * forces 1 or 2 rows on any device. */

static ds4_gpu_mv_dispatch ds4_gpu_make_plain_mv_dispatch(
        uint64_t in_dim,
        int      f32_weights) {
    if (in_dim < 32) {
        return (ds4_gpu_mv_dispatch) {
            .function_name = f32_weights ? "kernel_mul_mv_f32_f32_short" : "kernel_mul_mv_f16_f32_short",
            .nsg = 1,
            .nr0 = 32,
            .smem = 0,
        };
    }

    const int16_t nsg = (int16_t)((in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u);
    const int use_4 = (in_dim % 4u) == 0;
    return (ds4_gpu_mv_dispatch) {
        .function_name = f32_weights
            ? (use_4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32")
            : (use_4 ? "kernel_mul_mv_f16_f32_4" : "kernel_mul_mv_f16_f32"),
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

/* Simdgroups per threadgroup for the few-row (2..16 tokens) Q8/F16 matvecs
 * that carry the MTP verify rows.  Rows per threadgroup only: each row keeps
 * its lane walk and shuffle tree, so outputs are byte-identical at any
 * value (tests/test_qwen4_kernels.c pins 1/2/4/8). */
/* Row-exact geometry for the 3-row speculative verify: the few-row matvec
 * and routed-expert dispatches keep the lane maps and kernel choices of the
 * T <= 2 decode paths, so every committed row stays byte-identical. */
static bool g_qwen4_verify_rows_exact;
void ds4_gpu_qwen4_set_verify_rows_exact(bool on) {
    g_qwen4_verify_rows_exact = on;
}

static int16_t ds4_gpu_mv_ext_nsg(void) {
    return (int16_t)ds4_gpu_env_u64("DS4_METAL_MV_EXT_NSG", 2u, 1u, 8u);
}

static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && (n_tok < 3 || (n_tok == 3 && g_qwen4_verify_rows_exact))) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return n_tok > 8 ? 4 : 0;
    }
}





typedef ds4_metal_rms_args ds4_gpu_rms_norm_args;

typedef ds4_metal_qkv_rms_args ds4_gpu_qkv_rms_norm_args;


static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args(
        uint32_t n0,
        uint32_t n1,
        uint32_t n2,
        float    eps) {
    const uint64_t row_bytes = (uint64_t)n0 * sizeof(float);
    const uint64_t plane_bytes = row_bytes * n1;
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n0,
        .ne00_t = (int32_t)(n0 / 4u),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = plane_bytes * n2,
        .eps = eps,
        .nef1 = { (int32_t)n1, 1, 1 },
        .nef2 = { (int32_t)n2, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { plane_bytes, row_bytes, row_bytes },
        .nbf3 = { plane_bytes * n2, row_bytes, row_bytes },
    };
}

static NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

static NSUInteger ds4_gpu_rms_norm_pipeline_threads(
        uint32_t                  n,
        id<MTLComputePipelineState> pipeline) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger max_threads = pipeline ? [pipeline maxTotalThreadsPerThreadgroup] : 1024u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

/* ds4_gpu_hc_split_args lives once in hc_control.h. */

/* ds4_gpu_hc_weighted_sum_args lives once in hc_control.h. */

/* ds4_gpu_output_hc_weights4_args lives once in hc_control.h. */

/* ds4_gpu_hc_split_weighted_sum_args lives once in hc_control.h. */

/* ds4_gpu_hc_split_weighted_sum_norm_args lives once in hc_control.h. */

/* ds4_gpu_hc_expand_args lives once in hc_control.h. */

/* ds4_gpu_mul_mv_id_args is declared once in routed_batch.h. */

typedef struct {
    uint32_t n_total_expert;
    uint32_t n_expert;
} ds4_gpu_stream_expert_validate_args;

/* ds4_gpu_stream_expert_split_args is shared once in routed_single.h. */

/* ds4_gpu_mul_mm_id_map_args is declared once in routed_batch.h. */

/* ds4_gpu_mul_mm_id_args is declared once in routed_batch.h. */

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0);



static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);
static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size);

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off);
static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes);
static int ds4_gpu_encode_mul_mm_id_addr_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t                    resource_count,
        uint32_t                    resource_kind,
        id<MTLBuffer>               overflow_resource);

typedef ds4_metal_attention_pad_args ds4_gpu_flash_attn_pad_args;

typedef ds4_metal_attention_stage_args ds4_gpu_flash_kv_stage_f16_args;



typedef ds4_metal_attention_block_args ds4_gpu_flash_attn_blk_args;

typedef ds4_metal_attention_vec_args ds4_gpu_flash_attn_vec_args;

typedef ds4_metal_attention_reduce_args ds4_gpu_flash_attn_reduce_args;

typedef ds4_metal_rope_tail_args ds4_gpu_rope_tail_batch_args;

typedef ds4_metal_attention_rope_args ds4_gpu_rope_affine_pair_args;

/* Matches ds4_metal_args_dsv4_comp_finalize in dsv4_rope.metal. */
/* ds4_gpu_comp_finalize_args lives once in compressor.h. */


_Static_assert(sizeof(ds4_gpu_rope_affine_pair_args) == 64,
               "Metal affine RoPE argument ABI changed");



typedef ds4_metal_fp8_quant_args ds4_gpu_dsv4_fp8_kv_quantize_args;

typedef ds4_metal_kv_store_args ds4_gpu_dsv4_kv_fp8_store_args;

typedef ds4_metal_qat_args ds4_gpu_dsv4_indexer_qat_args;

/* ds4_gpu_dsv4_ratio4_shift_args lives once in compressor.h. */

/* ds4_gpu_dsv4_compressor_pack_ratio4_args lives once in compressor.h. */

/* ds4_gpu_dsv4_softmax_pool_ratio4_direct_args lives once in compressor.h. */

/* ds4_gpu_dsv4_compressor_store_one_args lives once in compressor.h. */

/* ds4_gpu_dsv4_softmax_pool_args lives once in compressor.h. */

/* ds4_gpu_dsv4_compressor_score_ape_args lives once in compressor.h. */

typedef ds4_metal_index_sort_args ds4_gpu_kargs_argsort;

typedef ds4_metal_index_merge_args ds4_gpu_kargs_argsort_merge;

/* ds4_gpu_kargs_sum_rows lives once in compressor.h. */

/* ds4_gpu_softmax_args lives once in compressor.h. */

typedef ds4_metal_index_rows_args ds4_gpu_dsv4_topk_mask_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
} ds4_gpu_dsv4_indexer_weighted_sum_args;

typedef ds4_metal_router_select_one_args ds4_gpu_dsv4_router_select_one_args;

typedef ds4_metal_router_visual_args ds4_gpu_dsv4_router_select_visual_args;

typedef struct {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
} ds4_gpu_glm_router_select_one_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
} ds4_gpu_glm_kv_lora_rms_norm_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
} ds4_gpu_glm_k_b_project_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
} ds4_gpu_glm_store_compact_kv_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t q_n;
    uint32_t q_n4;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_lora_n4;
    uint32_t qk_rope;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
} ds4_gpu_glm_qkv_norm_store_compact_kv_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    pad1;
} ds4_gpu_glm_store_indexer_k_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t pool_size;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
} ds4_gpu_glm53_indexer_pool_update_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
} ds4_gpu_glm_build_kv_cache_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_len;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t qk_dim;
    uint32_t value_dim;
    uint32_t pad0;
    uint32_t cache_f16;
    uint32_t pad1;
    uint32_t pad2;
    float    scale;
} ds4_gpu_glm_attention_full_args;

typedef struct {
    uint32_t n_selected;
} ds4_gpu_glm_fill_selected_range_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
} ds4_gpu_glm_fill_selected_range_batch_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t selected_pools;
    uint32_t index_topk;
    uint32_t pool_size;
    uint32_t output_width;
} ds4_gpu_glm53_expand_pool_selection_args;

typedef ds4_metal_rope_glm_args ds4_gpu_glm_rope_tail_args;

typedef ds4_metal_index_score_args ds4_gpu_glm_indexer_score_one_args;

typedef ds4_metal_index_batch_args ds4_gpu_glm_indexer_scores_batch_args;

typedef struct {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
} ds4_gpu_glm_qk_lowrank_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t head_base;
} ds4_gpu_glm_qk_lowrank_batch_args;

typedef struct {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
} ds4_gpu_glm_attention_indexed_decode_args;

typedef struct {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t block_rows;
    uint32_t n_blocks;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
} ds4_gpu_glm_attention_indexed_decode_split_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
} ds4_gpu_glm_attention_indexed_batch_args;

typedef struct {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    float    swiglu_clamp;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
} ds4_gpu_glm_routed_moe_args;

typedef ds4_metal_attention_indexed_args ds4_gpu_dsv4_indexed_attention_args;

typedef struct {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
} ds4_gpu_dsv4_indexer_scores_fused_args;

/* ds4_gpu_dsv4_moe_swiglu_weight_args is declared once in routed_batch.h. */

/* ds4_gpu_moe_expert_group_args is shared once in routed_single.h. */

/* ds4_gpu_q4_gather_slots6_args is shared once in routed_single.h. */

/* ds4_gpu_dsv4_moe_sum6_args is declared once in routed_batch.h. */

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    /* Preserve the original already-initialized no-pool fast path. */
    if (g_initialized) return 1;
    @autoreleasepool {
        ds4_metal_bootstrap_context context = ds4_metal_bootstrap_driver();
        return ds4_native_metal_bootstrap_init(&context);
    }
}

/* Model-independent Metal access. A Flint-only process needs the same device
 * and queue, without compiling the DeepSeek pipeline catalogue. */
bool ds4_gpu_metal_shared_context(void **device, void **queue) {
    @autoreleasepool {
        if (!g_device && !ds4_metal_bootstrap_device()) return false;
        if (!g_queue && !ds4_metal_bootstrap_queue()) return false;
        *device = (__bridge void *)g_device;
        *queue = (__bridge void *)g_queue;
        return true;
    }
}

int ds4_gpu_test_mxfp4_down_half_lut(uint16_t *legacy_bits,
                                     uint16_t *lut_bits) {
    if (!legacy_bits || !lut_bits) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;

    @autoreleasepool {
        const NSUInteger count = 4096u;
        const NSUInteger bytes = count * sizeof(uint16_t);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_test_mxfp4_down_half_lut");
        id<MTLBuffer> legacy = [g_device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> lut = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
        if (!pipeline || !legacy || !lut || !cb) {
            fprintf(stderr, "ds4: Metal MXFP4 half-LUT test setup failed\n");
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:legacy offset:0 atIndex:0];
        [enc setBuffer:lut offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((count + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
        [enc endEncoding];
        [cb commit];
        if (!ds4_gpu_wait_command_buffer(cb, "MXFP4 half-LUT raw-bit test")) {
            return 0;
        }
        memcpy(legacy_bits, [legacy contents], bytes);
        memcpy(lut_bits, [lut contents], bytes);
    }
    return 1;
}

void ds4_gpu_test_set_flags(uint32_t flags) {
    g_test_flags = flags;
}



int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !slots || n_rows == 0 || width == 0 || n_slots == 0 ||
        slot_cap == 0 || n_rows > slot_cap) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> slotsbuf = ds4_gpu_tensor_buffer(slots);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        uint64_t row_bytes = 0;
        uint64_t slot_plane_bytes = 0;
        uint64_t slots_bytes = 0;
        uint64_t out_rows = 0;
        uint64_t out_bytes = 0;
        if ((uint64_t)width > UINT64_MAX / sizeof(float)) return 0;
        row_bytes = (uint64_t)width * sizeof(float);
        if ((uint64_t)slot_cap > UINT64_MAX / row_bytes) return 0;
        slot_plane_bytes = (uint64_t)slot_cap * row_bytes;
        if ((uint64_t)n_slots > UINT64_MAX / slot_plane_bytes) return 0;
        slots_bytes = (uint64_t)n_slots * slot_plane_bytes;
        if ((uint64_t)n_rows > UINT64_MAX / n_slots) return 0;
        out_rows = (uint64_t)n_rows * n_slots;
        if (out_rows > UINT64_MAX / row_bytes) return 0;
        out_bytes = out_rows * row_bytes;
        if (!slotsbuf || !outbuf ||
            ds4_gpu_tensor_bytes(slots) < slots_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal slot-row pack received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                        slotsbuf,
                                                        ds4_gpu_tensor_offset(slots),
                                                        outbuf,
                                                        ds4_gpu_tensor_offset(out),
                                                        width,
                                                        n_slots,
                                                        n_rows,
                                                        sizeof(float),
                                                        slot_plane_bytes,
                                                        row_bytes,
                                                        row_bytes,
                                                        (uint64_t)n_slots * row_bytes)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "slot-row pack")) return 0;
    }

    return 1;
}

int ds4_gpu_begin_commands(void) {
    return ds4_native_metal_begin_commands();
}

int ds4_gpu_flush_encoder(void) {
    return ds4_native_metal_flush_encoder();
}

int ds4_gpu_flush_commands(void) {
    return ds4_native_metal_flush_commands();
}

int ds4_gpu_commands_active(void) {
    return ds4_native_metal_commands_active();
}

/* Exact M5 full-FFN overlap inside one concurrent compute encoder.  Shared
 * gate/up and routed IQ2 pair-SwiGLU launch together; explicit level barriers
 * precede the routed Q2 and shared Q8 down consumers. */
typedef ds4_metal_shared_split_args ds4_gpu_shared_split_args;
#define g_parallel_q8_pipeline g_schedule.parallel.down_pipeline
#define g_parallel_q8_weight g_schedule.parallel.down_weight
#define g_parallel_q8_x g_schedule.parallel.down_x
#define g_parallel_q8_out g_schedule.parallel.down_out
#define g_parallel_q8_weight_offset g_schedule.parallel.down_weight_offset
#define g_parallel_q8_x_offset g_schedule.parallel.down_x_offset
#define g_parallel_q8_out_offset g_schedule.parallel.down_out_offset
#define g_parallel_q8_args g_schedule.parallel.down_args
#define g_parallel_gate_up_pipeline g_schedule.parallel.gate_pipeline
#define g_parallel_gate_weight g_schedule.parallel.gate_weight
#define g_parallel_up_weight g_schedule.parallel.up_weight
#define g_parallel_gate_x g_schedule.parallel.gate_x
#define g_parallel_gate_out g_schedule.parallel.gate_out
#define g_parallel_up_out g_schedule.parallel.up_out
#define g_parallel_mid_out g_schedule.parallel.mid_out
#define g_parallel_gate_weight_offset g_schedule.parallel.gate_weight_offset
#define g_parallel_up_weight_offset g_schedule.parallel.up_weight_offset
#define g_parallel_gate_x_offset g_schedule.parallel.gate_x_offset
#define g_parallel_gate_out_offset g_schedule.parallel.gate_out_offset
#define g_parallel_up_out_offset g_schedule.parallel.up_out_offset
#define g_parallel_mid_out_offset g_schedule.parallel.mid_out_offset
#define g_parallel_gate_up_args g_schedule.parallel.gate_args
#define g_parallel_gate_up_clamp g_schedule.parallel.clamp
#define g_parallel_gate_up_nsg g_schedule.parallel.nsg
#define g_parallel_gate_up_nr0 g_schedule.parallel.nr0
#define g_parallel_gate_up_smem g_schedule.parallel.shared_bytes
#define g_parallel_ffn_mode g_schedule.parallel.mode
#define g_parallel_ffn_stage g_schedule.parallel.stage
#define g_parallel_q8_pending g_schedule.parallel.pending
#define g_parallel_q8_encoded g_schedule.parallel.encoded
#define g_parallel_split_active g_schedule.parallel.split
#define g_parallel_split_args g_schedule.parallel.split_args
#define g_parallel_split_ids g_schedule.parallel.ids
#define g_parallel_split_ids_offset g_schedule.parallel.ids_offset

/* Reset is deliberately idempotent. Closing the concurrent encoder preserves
 * work already encoded, while clearing every admission/reference field keeps
 * a failed FFN path from turning later ordinary dispatches concurrent. */
static void ds4_gpu_parallel_ffn_reset_state(BOOL close_encoder) {
    ds4_native_metal_parallel_reset(close_encoder);
}

void ds4_gpu_parallel_ffn_abort(void) {
    ds4_gpu_parallel_ffn_reset_state(YES);
}

/* Clang cleanup makes every early return from the large generic routed-MoE
 * wrapper abort an armed concurrent FFN without touching its many
 * established fallback branches. */
static void ds4_gpu_parallel_ffn_scope_cleanup(BOOL *armed) {
    if (armed && *armed && g_parallel_q8_pending) {
        ds4_gpu_parallel_ffn_abort();
    }
}

static int ds4_gpu_parallel_ffn_start_range(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        uint32_t              shared_lane_offset,
        uint32_t              shared_lane_count,
        const ds4_gpu_tensor *x,
        float                 clamp) {
    const ds4_metal_parallel_request request = {
        .gate = gate, .up = up, .mid = mid, .out = shared_out,
        .map = model_map, .size = model_size,
        .gate_offset = gate_offset, .up_offset = up_offset, .down_offset = down_offset,
        .model_dim = model_dim, .shared_dim = shared_dim, .x = x, .clamp = clamp,
        .lane_offset = shared_lane_offset, .lane_count = shared_lane_count,
    };
    return ds4_native_metal_parallel_start(&request);
}

int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp) {
    return ds4_gpu_parallel_ffn_start_range(
        gate, up, mid, shared_out, model_map, model_size,
        gate_offset, up_offset, down_offset, model_dim, shared_dim,
        0, shared_dim, x, clamp);
}

int ds4_gpu_parallel_ffn_start_sliced(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        uint32_t              shared_lane_offset,
        uint32_t              shared_lane_count,
        const ds4_gpu_tensor *x,
        float                 clamp) {
    return ds4_gpu_parallel_ffn_start_range(
        gate, up, mid, shared_out, model_map, model_size,
        gate_offset, up_offset, down_offset, model_dim, shared_dim,
        shared_lane_offset, shared_lane_count, x, clamp);
}

int ds4_gpu_parallel_ffn_start_split(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        const ds4_gpu_tensor *selected,
        uint32_t              tp_rank,
        uint32_t              tp_world,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              shift_q16) {
    const ds4_metal_parallel_request request = {
        .gate = gate, .up = up, .mid = mid, .out = shared_out,
        .map = model_map, .size = model_size,
        .gate_offset = gate_offset, .up_offset = up_offset, .down_offset = down_offset,
        .model_dim = model_dim, .shared_dim = shared_dim, .x = x, .clamp = clamp,
        .split = true, .selected = selected, .tp_rank = tp_rank, .tp_world = tp_world,
        .experts = n_expert, .used = n_expert_used, .shift_q16 = shift_q16,
    };
    return ds4_native_metal_parallel_start(&request);
}



static int ds4_gpu_parallel_q8_matvec_encode_pending(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        routed_mid) {
    return ds4_native_metal_parallel_first((__bridge void *)cb, (__bridge void *)routed_mid);
}

static int ds4_gpu_parallel_ffn_encode_second_stage(
        id<MTLCommandBuffer> cb) {
    return ds4_native_metal_parallel_second((__bridge void *)cb);
}

int ds4_gpu_parallel_ffn_finish(void) {
    return ds4_native_metal_parallel_finish();
}


static int ds4_gpu_stream_expert_cache_wait_inflight(const char *label) {
    const char *what = label ? label : "streaming expert cache in-flight";
    if (g_batch_cb && ds4_gpu_flush_commands() == 0) return 0;
    if ([g_pending_cbs count] != 0 &&
        ds4_gpu_wait_pending_command_buffers(what) == 0) {
        return 0;
    }
    return 1;
}



/*
 * Tensor-parallel gates.
 *
 * A TP gate is a mid-command-stream rendezvous with the peer machine: the
 * kernels ahead of the gate leave a partial block output in a slab slot,
 * the GPU signals g_tp_gpu_event, and the pre-encoded combine kernel waits
 * on g_tp_cpu_event.  A dedicated service thread bridges the two: it spins
 * until the GPU reaches the gate, runs the transport exchange (RDMA WRITE
 * plus flag poll, or a TCP write/read pair — behind the callback), and
 * CPU-signals the release.  On exchange failure the release is signaled
 * anyway so the GPU never deadlocks; the failure latches in g_tp_failed
 * and the eval aborts at the next command-buffer boundary.
 *
 * Gate sequence values increase monotonically per encoded gate.  Both ranks
 * encode the identical graph, so the values agree by construction and slots
 * never need resetting between tokens.
 */
/* ds4_gpu_tp_request is declared once in tp_control.h. */

/* Transport constants live once in tp_control.h. */

static id<MTLSharedEvent> g_tp_gpu_event;   /* GPU -> service thread */
static id<MTLSharedEvent> g_tp_cpu_event;   /* service thread -> GPU */
/* Batch (verify-block) gates run on their own sequence space and release
 * event: the row-gate seq feeds the RDMA pre-posted recv accounting, which
 * requires consecutive values, and a shared release event would make a
 * small batch value satisfy waits armed against the larger row seq. */
static id<MTLSharedEvent> g_tp_batch_gpu_event;
static id<MTLSharedEvent> g_tp_batch_cpu_event;

/* Batch flag values are tagged so a stale row-gate seq in the reused FFN
 * flag word can never satisfy a batch arrival spin (and vice versa). */
/* Expert-ownership split parameters for routed kernels. World 1 means TP is
 * not bound; world 2 assigns each rank one contiguous expert range. */
static int32_t g_tp_split_rank;
static int32_t g_tp_split_world = 1;


static int ds4_gpu_tp_world_is_two(void) {
    return g_tp_split_world == 2;
}

/* Return the contiguous routed-expert range backed by this process. Rank 1
 * owns the high range and receives any odd-count remainder. */
static void ds4_gpu_tp_expert_range(uint32_t n_total_expert,
                                    uint32_t *first_expert,
                                    uint32_t *n_expert) {
    *first_expert = 0;
    *n_expert = n_total_expert;
    if (g_tp_split_world != 2) return;

    const uint32_t low_experts = n_total_expert / 2u;
    if (g_tp_split_rank == 1) {
        *first_expert = low_experts;
        *n_expert = n_total_expert - low_experts;
    } else {
        *n_expert = low_experts;
    }
}

/* Attention head split for GLM batch prefill: each rank computes a
 * contiguous half of the heads in the qk-low / attention-lora /
 * value-project batch kernels; the caller zeroes the unowned head range
 * of the heads buffer and combines the attn-output partials over the
 * TP big-gate exchange. */
static int32_t g_tp_attn_head_split;

void ds4_gpu_tp_set_attn_head_split(int enabled) {
    g_tp_attn_head_split = enabled ? 1 : 0;
}

static void ds4_gpu_tp_attn_head_range(uint32_t n_head,
                                       uint32_t group,
                                       uint32_t *head_base,
                                       uint32_t *head_count) {
    *head_base = 0;
    *head_count = n_head;
    if (!g_tp_attn_head_split || g_tp_split_world != 2) return;
    const uint32_t half = n_head / 2u;
    if (half == 0u || (half % group) != 0u || (n_head % 2u) != 0u) return;
    *head_count = half;
    *head_base = g_tp_split_rank == 1 ? half : 0u;
}
/* Flag gates (DS4_TP_FLAG_GATES): the GPU publishes gate arrival by storing
 * the sequence number into a slab word instead of signaling the shared
 * event; the service thread spin-reads it from shared memory, which wakes
 * hundreds of microseconds earlier than signaledValue polling.  The
 * CPU->GPU release direction stays on the shared event. */

/* Poll gates (default with flag gates): the GPU waits for a gate release by
 * probing fresh cache lines at the head of a new command buffer instead of
 * parking on g_tp_cpu_event. See kernel_dsv4_tp_poll_release. */

static id<MTLBuffer> g_tp_poll_buffer;         /* release region + status words */
static volatile uint32_t *g_tp_poll_region;   /* CPU view of the release lines */
static volatile uint32_t *g_tp_poll_status;   /* per-slot poll result, GPU-written */
static int g_tp_poll_prev_valid;
static uint32_t g_tp_poll_prev_slot;
static uint64_t g_tp_poll_prev_seq;
/* Gate-time prefetch: the poll wait leaves the memory bus idle, so the next
 * phase's weights are streamed into the cache first (see the notebook). */
static ds4_metal_tp_prefetch_policy g_tp_native_prefetch;
#define g_tp_exchange_ewma_us (g_tp_native_prefetch.exchange_ewma_us)
#define g_tp_prefetch_plan (g_tp_native_prefetch.plan)
#define g_tp_prefetch_count (g_tp_native_prefetch.count)
#define g_tp_prefetch_map (g_tp_native_prefetch.map)
#define g_tp_prefetch_map_size (g_tp_native_prefetch.map_size)
#define g_tp_gate_prefetch (g_tp_native_prefetch.enabled)
        /* per gate kind, always on */
/* ds4_tp_prefetch_range lives once in tp_prefetch.h. */
 /* per gate kind */
/* Gate flag folded into the payload producer (see
 * kernel_dsv4_add2_f32_tp_flag_checked): the producer publishes the checked
 * flag for the sequence number the next gate encode will use, and the gate
 * encode skips its own flag kernel. */

static id<MTLBuffer> g_tp_fold_ctl;



static id<MTLBuffer> g_tp_prefetch_scratch;

static uint64_t g_tp_stat_poll_lines;      /* sum of poll hit lines (profile) */
static uint64_t g_tp_stat_poll_max_line;
static uint64_t g_tp_stat_poll_count;
static uint64_t g_tp_stat_encode_lead;     /* sum of (encoded seq - gate seq) at arrival */
static double g_tp_stat_release_ms;        /* time spent writing poll releases */
static double g_tp_stat_cbwait_ms;         /* flag seen -> payload verified in memory */
              /* slab offset of out slot 0 */
            /* partial vector bytes per slot */
static id<MTLBuffer> g_tp_check_buffer;    /* per-slot payload checksums */
static volatile uint32_t *g_tp_check_words;
static volatile uint8_t *g_tp_slab_cpu;   /* CPU view of the slab base */
/* Bound on gate command buffers in flight: the GPU only needs a few
 * segments of runway and a deep backlog costs command-buffer memory. */
static id<MTLBuffer> g_tp_slab_buffer;

static volatile uint32_t *g_tp_gpu_flags;   /* CPU view of the flag words */


static ds4_gpu_tp_exchange_fn g_tp_exchange_fn;
static ds4_gpu_tp_batch_exchange_fn g_tp_batch_exchange_fn;
static ds4_gpu_tp_big_exchange_fn g_tp_big_exchange_fn;

void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    g_tp_big_exchange_fn = fn;
}

static void *g_tp_exchange_ud;
static pthread_t g_tp_thread;
static int g_tp_thread_running;
static int g_tp_shutdown;
static int g_tp_failed_flag;
static pthread_mutex_t g_tp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tp_cond = PTHREAD_COND_INITIALIZER;
static ds4_gpu_tp_request g_tp_queue[DS4_GPU_TP_QUEUE];
static uint32_t g_tp_queue_head;
static uint32_t g_tp_queue_count;

static uint64_t g_tp_stat_gates;
static uint64_t g_tp_stat_batch_gates;      /* verify-block batch gates */
static double g_tp_stat_batch_gpu_wait_ms;   /* arrival spin (GPU compute + park) */
static double g_tp_stat_batch_exchange_ms;   /* exchange callback (peer wait + wire) */
static double g_tp_stat_batch_release_ms;    /* exchange end -> cpu event signaled */
static double g_tp_stat_gpu_wait_ms;
static double g_tp_stat_exchange_ms;
/* Gate constants live once in tp_control.h. */
static uint64_t g_tp_stat_gate_count[DS4_GPU_TP_GATES_PER_LAYER];
static double g_tp_stat_gate_gpu_wait_ms[DS4_GPU_TP_GATES_PER_LAYER];
static double g_tp_stat_gate_exchange_ms[DS4_GPU_TP_GATES_PER_LAYER];

/* GPU keep-alive (see kernel_dsv4_tp_keepalive): its own queue and thread,
 * alive exactly as long as the TP gate machinery. */
static id<MTLCommandQueue> g_tp_keepalive_queue;
static id<MTLBuffer> g_tp_keepalive_buffer;
static pthread_t g_tp_keepalive_thread;
static int g_tp_keepalive_running;

/* Nonzero while a verify block runs: the GPU is genuinely busy
 * there, so the keep-alive is a pure parasite (~2.3ms per 5-row block
 * measured against the single-machine verify). */
static volatile int g_tp_keepalive_paused;

void ds4_gpu_tp_keepalive_pause(int paused) {
    g_tp_keepalive_paused = paused;
}




/* Encode the touch dispatches of one gate's plan into the open batch. */
static uint64_t ds4_gpu_buffer_address(id<MTLBuffer> buffer, NSUInteger inner);

/* One dispatch of kernel_touch_u8_stride_table over up to DS4_TP_PREFETCH_MAX
 * model ranges (priority order, cut at `budget` bytes), 128 B per thread. */

static int ds4_gpu_tp_encode_gate_prefetch(uint32_t gate) {
    return ds4_native_metal_tp_prefetch_encode(gate);
}

static uint32_t ds4_gpu_tp_keepalive_tgs_from_env(void) {
    uint32_t ka_tgs = 1;
    const char *tgs_env = getenv("DS4_TP_KEEPALIVE_TGS");
    if (tgs_env) {
        int v = atoi(tgs_env);
        if (v > 0 && v <= 2048) ka_tgs = (uint32_t)v;
    }
    return ka_tgs;
}

static void *ds4_gpu_tp_keepalive_thread(void *arg) {
    (void)arg;
    /* Swept on the M5 Max pair: too few iterations lets clocks sag.  Current
     * TP split-resident Flash runs show 1.2M is a small Q4/Q2 decode win over
     * 800k, while two threadgroups waste work. */
    uint32_t iters = 1200000;
    const char *env = getenv("DS4_TP_KEEPALIVE_ITERS");
    if (env) iters = (uint32_t)atoi(env);
    /* One ALU-only threadgroup keeps the GPU from power-gating but does
     * not push the frequency governor; solo-vs-engine kernel gaps
     * (~1.7x) suggest decode runs well below max clocks. More TGs raise
     * apparent utilization without eating memory bandwidth. */
    uint32_t ka_tgs = ds4_gpu_tp_keepalive_tgs_from_env();
    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_tp_keepalive");
    if (!pipeline) {
        fprintf(stderr, "ds4: TP keep-alive pipeline missing\n");
        return NULL;
    }
    while (!g_tp_shutdown) {
        if (g_tp_keepalive_paused) {
            usleep(200);
            continue;
        }
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_tp_keepalive_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:g_tp_keepalive_buffer offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ka_tgs, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return NULL;
}

static void *ds4_gpu_tp_service_thread(void *arg) {
    (void)arg;
    const bool profile = getenv("DS4_TP_GATE_PROFILE") != NULL;
    while (1) {
        pthread_mutex_lock(&g_tp_mutex);
        while (g_tp_queue_count == 0 && !g_tp_shutdown)
            pthread_cond_wait(&g_tp_cond, &g_tp_mutex);
        if (g_tp_shutdown && g_tp_queue_count == 0) {
            pthread_mutex_unlock(&g_tp_mutex);
            break;
        }
        ds4_gpu_tp_request req = g_tp_queue[g_tp_queue_head];
        g_tp_queue_head = (g_tp_queue_head + 1) % DS4_GPU_TP_QUEUE;
        g_tp_queue_count--;
        pthread_mutex_unlock(&g_tp_mutex);

        /* Wait for the GPU to reach this gate.  Tight spin: gate arrival is
         * on the decode critical path and normally tens of microseconds
         * out; yielding here measurably delays the release wake-up. */
        const double t0 = profile ? ds4_gpu_now_ms() : 0.0;
        uint32_t spins = 0;
        if (req.big_bytes > 0) {
            /* Big gates always signal arrival through the batch shared
             * event (see ds4_gpu_tp_big_gate_kick): the event completion
             * semantics are what guarantee the bounce payload is visible
             * before the exchange reads it. */
            /* A big-gate exchange takes milliseconds, so a tight spin on
             * the shared event buys nothing; each signaledValue read is a
             * driver round trip.  Sleep-poll unless asked otherwise
             * (DS4_TP_BIG_GATE_SPIN_US=0 restores the tight spin). */
            static int64_t big_spin_us = -1;
            if (big_spin_us < 0)
                big_spin_us = (int64_t)ds4_gpu_env_u64("DS4_TP_BIG_GATE_SPIN_US", 20u, 0u, 5000u);
            while (g_tp_batch_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (big_spin_us > 0) usleep((useconds_t)big_spin_us);
                else if (++spins > (1u << 16)) sched_yield();
            }
        } else if (!req.event_arrival) {
            const uint32_t slot =
                req.layer * DS4_GPU_TP_GATES_PER_LAYER + req.gate;
            uint32_t want = (uint32_t)req.seq;
            if (req.rows > 0)
                want = DS4_TP_BATCH_FLAG_TAG | (uint32_t)req.seq;
            while (__atomic_load_n(&g_tp_gpu_flags[slot], __ATOMIC_ACQUIRE) != want) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 20)) {
                    sched_yield();
                    spins = 0;
                }
            }
        } else if (req.rows > 0) {
            while (g_tp_batch_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 16)) sched_yield();
            }
        } else {
            while (g_tp_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 16)) sched_yield();
            }
        }
        const double t1 = ds4_gpu_now_ms();
        if (profile) g_tp_stat_encode_lead += g_tp_seq - req.seq;
        if (req.poll && g_tp_check_words && g_tp_slab_cpu && req.rows == 0 &&
            req.big_bytes == 0) {
            /* The flag line may land before the payload lines: verify the
             * partial against the checksum the flag kernel published. */
            const uint32_t slot = req.layer * DS4_GPU_TP_GATES_PER_LAYER + req.gate;
            const uint32_t words = (uint32_t)(g_tp_vec_bytes / 4u);
            const volatile uint32_t *payload =
                (const volatile uint32_t *)(g_tp_slab_cpu + g_tp_out_off + (uint64_t)slot * g_tp_vec_bytes);
            const uint32_t mix = (uint32_t)req.seq * 0x9E3779B9u;
            uint64_t tries = 0;
            for (;;) {
                const uint32_t want = __atomic_load_n(&g_tp_check_words[slot], __ATOMIC_ACQUIRE) ^ mix;
                uint32_t sum = 0;
                for (uint32_t i = 0; i < words; i++) sum += payload[i];
                if (sum == want) break;
                if (++tries > 20000000ull || g_tp_shutdown) {
                    if (!g_tp_failed_flag)
                        fprintf(stderr, "ds4: TP gate payload never matched its checksum (seq %llu)\n",
                                (unsigned long long)req.seq);
                    g_tp_failed_flag = 1;
                    break;
                }
            }
            if (profile) g_tp_stat_cbwait_ms += ds4_gpu_now_ms() - t1;
        }
        int ok = 0;
        if (!g_tp_shutdown && !g_tp_failed_flag) {
            if (req.big_bytes > 0) {
                if (g_tp_big_exchange_fn)
                    ok = g_tp_big_exchange_fn(g_tp_exchange_ud, req.layer,
                                              req.seq, req.big_out,
                                              req.big_in, req.big_bytes);
            } else if (req.rows > 0) {
                const double bx0 = profile ? ds4_gpu_now_ms() : 0.0;
                if (g_tp_batch_exchange_fn)
                    ok = g_tp_batch_exchange_fn(g_tp_exchange_ud, req.layer,
                                                req.rows, req.seq);
                if (profile) {
                    g_tp_stat_batch_gates++;
                    g_tp_stat_batch_gpu_wait_ms += t1 - t0;
                    g_tp_stat_batch_exchange_ms += ds4_gpu_now_ms() - bx0;
                }
            } else if (g_tp_exchange_fn) {
                ok = g_tp_exchange_fn(g_tp_exchange_ud, req.layer, req.gate,
                                      req.seq);
            }
        }
        if (!ok && !g_tp_shutdown) {
            if (!g_tp_failed_flag)
                fprintf(stderr, "ds4: TP gate exchange failed (layer %u gate %u seq %llu)\n",
                        req.layer, req.gate, (unsigned long long)req.seq);
            g_tp_failed_flag = 1;
        }
        const double t_ex = ds4_gpu_now_ms();
        if (req.rows == 0 && req.big_bytes == 0 && req.gate < 2u) {
            /* Sized the gate-time prefetch: how long this rank typically
             * waits for the peer at this gate kind (flag seen -> exchange done). */
            const double ex_us = (t_ex - t1) * 1000.0;
            g_tp_exchange_ewma_us[req.gate] =
                g_tp_exchange_ewma_us[req.gate] == 0.0 ? ex_us :
                0.9 * g_tp_exchange_ewma_us[req.gate] + 0.1 * ex_us;
        }
        /* Release the GPU even on failure so end_commands can drain. */
        if (req.rows > 0) {
            const double br0 = profile ? ds4_gpu_now_ms() : 0.0;
            g_tp_batch_cpu_event.signaledValue = req.seq;
            if (profile) g_tp_stat_batch_release_ms += ds4_gpu_now_ms() - br0;
        } else if (req.poll) {
            /* The flag we just consumed was published by the command buffer
             * that ran the previous poll, so that poll's result is visible:
             * a timeout there means the GPU has already run on stale data. */
            const uint32_t slot = req.layer * DS4_GPU_TP_GATES_PER_LAYER + req.gate;
            if (profile && g_tp_poll_prev_valid) {
                const uint32_t hit = g_tp_poll_status[g_tp_poll_prev_slot * 2u];
                if (hit != 0xffffffffu) {
                    g_tp_stat_poll_lines += hit;
                    if (hit > g_tp_stat_poll_max_line) g_tp_stat_poll_max_line = hit;
                    g_tp_stat_poll_count++;
                }
            }
            if (g_tp_poll_prev_valid &&
                g_tp_poll_status[g_tp_poll_prev_slot * 2u] == 0xffffffffu &&
                !g_tp_shutdown) {
                if (!g_tp_failed_flag)
                    fprintf(stderr,
                            "ds4: TP poll gate timed out (seq %llu, slot %u)\n",
                            (unsigned long long)g_tp_poll_prev_seq,
                            g_tp_poll_prev_slot);
                g_tp_failed_flag = 1;
            }
            const uint32_t v = (uint32_t)req.seq;
            const double rel_t0 = profile ? ds4_gpu_now_ms() : 0.0;
            volatile uint32_t *region =
                g_tp_poll_region +
                (size_t)(req.seq % DS4_TP_POLL_RING) *
                    DS4_TP_POLL_LINES * (DS4_TP_POLL_LINE_BYTES / 4u);
            for (uint32_t j = 0; j < DS4_TP_POLL_LINES; j++) {
                __atomic_store_n(&region[j * (DS4_TP_POLL_LINE_BYTES / 4u)],
                                 v, __ATOMIC_RELAXED);
            }
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            if (profile) g_tp_stat_release_ms += ds4_gpu_now_ms() - rel_t0;
            g_tp_poll_prev_valid = 1;
            g_tp_poll_prev_slot = slot;
            g_tp_poll_prev_seq = req.seq;
        } else {
            g_tp_cpu_event.signaledValue = req.seq;
        }
        if (req.rows > 0 || !req.poll) g_tp_poll_prev_valid = 0;
        if (profile) {
            const double gpu_wait_ms = t1 - t0;
            const double exchange_ms = t_ex - t1;
            g_tp_stat_gpu_wait_ms += gpu_wait_ms;
            g_tp_stat_exchange_ms += exchange_ms;
            if (req.gate < DS4_GPU_TP_GATES_PER_LAYER) {
                g_tp_stat_gate_count[req.gate]++;
                g_tp_stat_gate_gpu_wait_ms[req.gate] += gpu_wait_ms;
                g_tp_stat_gate_exchange_ms[req.gate] += exchange_ms;
            }
            if (++g_tp_stat_gates % 860 == 0) {
                const double attn_count =
                    (double)(g_tp_stat_gate_count[DS4_GPU_TP_GATE_ATTN]
                                 ? g_tp_stat_gate_count[DS4_GPU_TP_GATE_ATTN] : 1u);
                const double ffn_count =
                    (double)(g_tp_stat_gate_count[DS4_GPU_TP_GATE_FFN]
                                 ? g_tp_stat_gate_count[DS4_GPU_TP_GATE_FFN] : 1u);
                fprintf(stderr,
                        "ds4: TP gates %llu: avg gpu-wait %.1f us, exchange %.1f us; "
                        "attn %.1f/%.1f us, ffn %.1f/%.1f us\n",
                        (unsigned long long)g_tp_stat_gates,
                        g_tp_stat_gpu_wait_ms / (double)g_tp_stat_gates * 1000.0,
                        g_tp_stat_exchange_ms / (double)g_tp_stat_gates * 1000.0,
                        g_tp_stat_gate_gpu_wait_ms[DS4_GPU_TP_GATE_ATTN] /
                            attn_count * 1000.0,
                        g_tp_stat_gate_exchange_ms[DS4_GPU_TP_GATE_ATTN] /
                            attn_count * 1000.0,
                        g_tp_stat_gate_gpu_wait_ms[DS4_GPU_TP_GATE_FFN] /
                            ffn_count * 1000.0,
                        g_tp_stat_gate_exchange_ms[DS4_GPU_TP_GATE_FFN] /
                            ffn_count * 1000.0);
                if (g_tp_stat_batch_gates)
                    fprintf(stderr, "ds4: TP batch gates: %llu, arrival wait %.1f us, exchange %.1f us, release %.1f us per gate\n",
                            (unsigned long long)g_tp_stat_batch_gates,
                            g_tp_stat_batch_gpu_wait_ms * 1e3 / (double)g_tp_stat_batch_gates,
                            g_tp_stat_batch_exchange_ms * 1e3 / (double)g_tp_stat_batch_gates,
                            g_tp_stat_batch_release_ms * 1e3 / (double)g_tp_stat_batch_gates);
                fprintf(stderr,
                        "ds4: TP gates: encode lead %.2f gates; verify %.1f us; release %.1f us; poll hit line avg %.1f max %llu over %llu\n",
                        (double)g_tp_stat_encode_lead / (double)g_tp_stat_gates,
                        g_tp_stat_cbwait_ms / (double)g_tp_stat_gates * 1000.0,
                        g_tp_stat_release_ms / (double)g_tp_stat_gates * 1000.0,
                        g_tp_stat_poll_count ?
                            (double)g_tp_stat_poll_lines / (double)g_tp_stat_poll_count : 0.0,
                        (unsigned long long)g_tp_stat_poll_max_line,
                        (unsigned long long)g_tp_stat_poll_count);
            }
        }
    }
    return NULL;
}

int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                    uint64_t out_off, uint64_t vec_bytes,
                    ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_tp_thread_running || rank > 1) return 0;
    g_tp_split_rank = (int32_t)rank;
    g_tp_split_world = 2;
    g_tp_slab_buffer = slab ? ds4_gpu_tensor_buffer(slab) : nil;
    g_tp_slab_buffer_off = slab ? ds4_gpu_tensor_offset(slab) : 0;
    g_tp_gpu_flags_off = gpu_flags_off;
    g_tp_out_off = out_off;
    g_tp_vec_bytes = vec_bytes;
    g_tp_slab_cpu = slab ? (volatile uint8_t *)ds4_gpu_tensor_contents(slab) : NULL;
    g_tp_check_buffer = [g_device newBufferWithLength:(NSUInteger)DS4_TP_POLL_MAX_SLOTS * sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
    if (g_tp_check_buffer) {
        memset([g_tp_check_buffer contents], 0, (size_t)DS4_TP_POLL_MAX_SLOTS * sizeof(uint32_t));
        g_tp_check_words = (volatile uint32_t *)[g_tp_check_buffer contents];
    } else {
        g_tp_check_words = NULL;
    }
    g_tp_gpu_flags = slab ?
        (volatile uint32_t *)((uint8_t *)ds4_gpu_tensor_contents(slab) + gpu_flags_off) : NULL;
    /* Flag arrival is the default: the slab-word publish detects in ~1-3us
     * where signaledValue polling costs 10-20, worth +2.3 t/s on the pair
     * (A/B 2026-07-06, byte-identical output).  DS4_TP_EVENT_GATES falls
     * back to the shared-event arrival path. */
    g_tp_flag_gates = g_tp_gpu_flags != NULL && getenv("DS4_TP_EVENT_GATES") == NULL;
    g_tp_poll_gates = false;
    g_tp_poll_buffer = nil;
    g_tp_poll_region = NULL;
    g_tp_poll_status = NULL;
    g_tp_poll_prev_valid = 0;
    g_tp_gate_prefetch = getenv("DS4_TP_DISABLE_GATE_PREFETCH") == NULL;
    memset(g_tp_prefetch_count, 0, sizeof(g_tp_prefetch_count));
    /* Default with flag gates: byte-identical to the event wait and about
     * 20 us cheaper per gate on the M5 Max pair (A/B 2026-09-01, README
     * ctx 128: 46.6-47.2 vs 43.6 t/s). DS4_TP_DISABLE_POLL_GATES restores
     * the shared-event wait for diagnosis. */
    if (g_tp_flag_gates && getenv("DS4_TP_DISABLE_POLL_GATES") == NULL) {
        const NSUInteger region_bytes =
            (NSUInteger)DS4_TP_POLL_LINES * DS4_TP_POLL_LINE_BYTES * DS4_TP_POLL_RING;
        const NSUInteger status_bytes =
            (NSUInteger)DS4_TP_POLL_MAX_SLOTS * 2u * sizeof(uint32_t);
        g_tp_poll_buffer =
            [g_device newBufferWithLength:region_bytes + status_bytes
                                  options:MTLResourceStorageModeShared];
        if (g_tp_poll_buffer) {
            memset([g_tp_poll_buffer contents], 0, region_bytes + status_bytes);
            g_tp_poll_region = (volatile uint32_t *)[g_tp_poll_buffer contents];
            g_tp_poll_status =
                (volatile uint32_t *)((uint8_t *)[g_tp_poll_buffer contents] + region_bytes);
            g_tp_poll_gates = true;
        } else {
            fprintf(stderr, "ds4: TP poll gate buffer failed; using event gates\n");
        }
    }
    g_tp_gpu_event = [g_device newSharedEvent];
    g_tp_cpu_event = [g_device newSharedEvent];
    g_tp_batch_gpu_event = [g_device newSharedEvent];
    g_tp_batch_cpu_event = [g_device newSharedEvent];
    if (!g_tp_gpu_event || !g_tp_cpu_event ||
        !g_tp_batch_gpu_event || !g_tp_batch_cpu_event) {
        fprintf(stderr, "ds4: failed to create TP shared events\n");
        return 0;
    }
    g_tp_exchange_fn = fn;
    g_tp_exchange_ud = ud;
    g_tp_seq = 0;
    g_tp_batch_seq = 0;
    g_tp_shutdown = 0;
    g_tp_failed_flag = 0;
    g_tp_queue_head = 0;
    g_tp_queue_count = 0;
    g_tp_stat_gates = 0;
    g_tp_stat_gpu_wait_ms = 0.0;
    g_tp_stat_exchange_ms = 0.0;
    memset(g_tp_stat_gate_count, 0, sizeof(g_tp_stat_gate_count));
    memset(g_tp_stat_gate_gpu_wait_ms, 0,
           sizeof(g_tp_stat_gate_gpu_wait_ms));
    memset(g_tp_stat_gate_exchange_ms, 0,
           sizeof(g_tp_stat_gate_exchange_ms));
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
    if (pthread_create(&g_tp_thread, &attr, ds4_gpu_tp_service_thread, NULL) != 0) {
        pthread_attr_destroy(&attr);
        fprintf(stderr, "ds4: failed to start TP gate service thread\n");
        return 0;
    }
    pthread_attr_destroy(&attr);
    g_tp_thread_running = 1;
    if (getenv("DS4_TP_NO_KEEPALIVE") == NULL) {
        uint32_t ka_tgs = ds4_gpu_tp_keepalive_tgs_from_env();
        g_tp_keepalive_queue = [g_device newCommandQueue];
        g_tp_keepalive_buffer = [g_device newBufferWithLength:(NSUInteger)ka_tgs * 256u * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (g_tp_keepalive_queue && g_tp_keepalive_buffer &&
            pthread_create(&g_tp_keepalive_thread, NULL,
                           ds4_gpu_tp_keepalive_thread, NULL) == 0) {
            g_tp_keepalive_running = 1;
        } else {
            fprintf(stderr, "ds4: TP keep-alive setup failed (continuing without)\n");
        }
    }
    return 1;
}

void ds4_gpu_tp_shutdown(void) {
    if (!g_tp_thread_running) return;
    pthread_mutex_lock(&g_tp_mutex);
    g_tp_shutdown = 1;
    pthread_cond_broadcast(&g_tp_cond);
    pthread_mutex_unlock(&g_tp_mutex);
    pthread_join(g_tp_thread, NULL);
    g_tp_thread_running = 0;
    if (g_tp_keepalive_running) {
        pthread_join(g_tp_keepalive_thread, NULL);
        g_tp_keepalive_running = 0;
        g_tp_keepalive_queue = nil;
        g_tp_keepalive_buffer = nil;
    }
    g_tp_exchange_fn = NULL;
    g_tp_batch_exchange_fn = NULL;
    g_tp_exchange_ud = NULL;
    g_tp_split_rank = 0;
    g_tp_split_world = 1;
    g_tp_session_batch_mode = 0;
    g_tp_poll_gates = false;
    g_tp_poll_buffer = nil;
    g_tp_poll_region = NULL;
    g_tp_poll_status = NULL;
}

void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    if (!g_tp_thread_running) return;
    g_tp_split_world = suspend ? 1 : 2;
}

/* Fold request for producers that write the TP partial with a generic matvec
 * (the attention output K-slice): the layer code registers (layer, gate)
 * right before the producer call; the producer takes the request when its
 * output is that slot and publishes the checked flag itself. */







/* Returns the slot when `out` (out_bytes long) is the requested partial. */
static int ds4_gpu_tp_flag_fold_take(const ds4_gpu_tensor *out, uint64_t out_bytes,
                                     uint32_t *slot_out, uint32_t *value_out) {
    return ds4_native_metal_tp_fold_take(out,out_bytes,slot_out,value_out);
}



void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    g_tp_batch_exchange_fn = fn;
}

/* Verify-block batch gate: same arrival/release machinery as the row gate
 * (the FFN flag word and event pair are reused — a decode gate and a batch
 * gate are never in flight together, and seq values stay globally unique),
 * but the service thread runs the multi-row exchange callback. */

/* Prefill batch gate: the service thread exchanges big_bytes directly
 * between two CPU-visible bounce buffers. The shared-event signal orders
 * all preceding GPU stores before the service thread reads the payload. */



/* Run one trivial command buffer so the process's first-submission costs
 * (command queue and GPU context setup, page-table validation of the
 * mapped model views) are paid at load time rather than inside the first
 * prefill.  TP sharding skips the load-time model-view warm, which used to
 * leave a 600-800 ms stall at the head of the first prompt on each rank. */
int ds4_gpu_warm_command_queue(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 1;
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv4_tp_keepalive");
    if (!pipeline) return 0;
    const double t0 = ds4_gpu_now_ms();
    @autoreleasepool {
        id<MTLBuffer> scratch = [g_device newBufferWithLength:256u * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (!scratch) return 0;
        memset(scratch.contents, 0, 256u * sizeof(float));
        id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const uint32_t iters = 1u;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBytes:&iters length:sizeof(iters) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return 0;
    }
    if (getenv("DS4_METAL_CB_TIMES"))
        fprintf(stderr, "ds4: command queue warm in %.1f ms\n", ds4_gpu_now_ms() - t0);
    ds4_gpu_queue_keepalive_start();
    return 1;
}

/* Main-queue idle keepalive.  Measured on M5 Max (macOS 26): once the
 * engine's command queue has been idle for about 3 s, the next command
 * buffer on it starts 600-800 ms late (the driver drops and rebuilds the
 * process's GPU mappings; 76-145 GB of no-copy model views here).  Work on
 * other queues does not prevent it.  One 1 KB kernel per second on this
 * queue keeps the mappings alive at no measurable cost.
 * DS4_METAL_DISABLE_QUEUE_KEEPALIVE=1 opts out. */
static pthread_t g_queue_keepalive_thread;
static volatile int g_queue_keepalive_running;
static volatile int g_queue_keepalive_stop;

static void *ds4_gpu_queue_keepalive_thread(void *arg) {
    (void)arg;
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv4_tp_keepalive");
    id<MTLBuffer> scratch = [g_device newBufferWithLength:256u * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    if (!pipeline || !scratch) return NULL;
    memset(scratch.contents, 0, 256u * sizeof(float));
    const uint32_t iters = 1u;
    uint64_t period_ms = ds4_gpu_env_u64("DS4_METAL_QUEUE_KEEPALIVE_MS", 1000u, 50u, 10000u);
    while (!g_queue_keepalive_stop) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:scratch offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
        }
        for (uint64_t slept = 0; slept < period_ms && !g_queue_keepalive_stop; slept += 50)
            usleep(50000);
    }
    return NULL;
}

static void ds4_gpu_queue_keepalive_start(void) {
    if (g_queue_keepalive_running) return;
    if (getenv("DS4_METAL_DISABLE_QUEUE_KEEPALIVE")) return;
    g_queue_keepalive_stop = 0;
    if (pthread_create(&g_queue_keepalive_thread, NULL, ds4_gpu_queue_keepalive_thread, NULL) == 0)
        g_queue_keepalive_running = 1;
}

static void ds4_gpu_queue_keepalive_stop_thread(void) {
    if (!g_queue_keepalive_running) return;
    g_queue_keepalive_stop = 1;
    pthread_join(g_queue_keepalive_thread, NULL);
    g_queue_keepalive_running = 0;
}

int ds4_gpu_tp_failed(void) {
    return g_tp_failed_flag;
}


static int ds4_gpu_signal_batch_and_wait_event(const char *label) {
    return ds4_native_metal_tp_signal_batch_wait(label);
}

int ds4_gpu_end_commands(void) {
    return ds4_native_metal_end_commands();
}

static int ds4_gpu_flash_attn_stage_profile_boundary(
        id<MTLCommandBuffer> __strong *cbp,
        const char           *mode,
        const char           *stage,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              n_keys,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              window,
        uint32_t              ratio,
        double               *stage_t0) {
    if (!cbp || !*cbp || !stage_t0 || !stage) return 0;
    if (ds4_gpu_end_commands() == 0) return 0;

    const double now_ms = ds4_gpu_now_ms();
    const char *filter = getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER");
    const int print_stage =
        !filter || !filter[0] ||
        strstr(stage, filter) != NULL ||
        (mode && strstr(mode, filter) != NULL);
    if (print_stage) {
        fprintf(stderr,
                "ds4: Metal FlashAttention prefill stage mode=%s tokens=%u comp=%u "
                "keys=%u heads=%u dim=%u window=%u ratio=%u %s=%.3f ms\n",
                mode ? mode : "unknown",
                n_tokens,
                n_comp,
                n_keys,
                n_head,
                head_dim,
                window,
                ratio,
                stage,
                now_ms - *stage_t0);
    }
    *stage_t0 = now_ms;

    if (ds4_gpu_begin_commands() == 0) return 0;
    int owned = 0;
    *cbp = ds4_gpu_command_buffer(&owned);
    return *cbp != nil && owned == 0;
}

int ds4_gpu_synchronize(void) {
    return ds4_native_metal_synchronize();
}

static void qwen4_nax_release_scratch(void);
static void qwen4_batch_release_scratch(void);

void ds4_gpu_cleanup(void) {
    ds4_metal_bootstrap_context context = ds4_metal_bootstrap_driver();
    if (!ds4_native_metal_bootstrap_begin_cleanup(&context)) return;
    @autoreleasepool { ds4_native_metal_bootstrap_cleanup(&context); }
}



static int ds4_gpu_quant_row_bytes(
        uint32_t  type,
        uint32_t  n_embd,
        uint64_t *row_bytes_out) {
    if (!row_bytes_out || n_embd == 0) return 0;
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 34u;
        return 1;
    case DS4_METAL_TENSOR_Q4_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 18u;
        return 1;
    case DS4_METAL_TENSOR_Q4_K:
        if ((n_embd % 256u) != 0) return 0;
        *row_bytes_out = ((uint64_t)n_embd / 256u) * 144u;
        return 1;
    default:
        return 0;
    }
}












int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
        uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    return ds4_native_metal_model_map_range(&g_model_views, g_ssd_streaming_mode,
        model_map, model_size, map_offset, map_size, max_tensor_bytes);
}

static __attribute__((unused)) int ds4_gpu_model_views_cover_spans(
        const void     *model_map,
        uint64_t        model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t        count);

int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size,
        const uint64_t *offsets, const uint64_t *sizes, uint32_t count,
        uint64_t max_tensor_bytes) {
    return ds4_native_metal_model_map_spans(&g_model_views, g_ssd_streaming_mode,
        model_map, model_size, offsets, sizes, count, max_tensor_bytes);
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size, 0);
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    return 1;
}

static __attribute__((unused)) int ds4_gpu_model_views_cover_spans(
        const void *model_map, uint64_t model_size, const uint64_t *offsets,
        const uint64_t *sizes, uint32_t count) {
    return ds4_native_metal_model_views_cover_spans(&g_model_views,
        model_map, model_size, offsets, sizes, count);
}


int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    (void)fd;
    (void)model_map;
    return 1;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    (void)model_map;
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    const uint32_t index = ds4_native_metal_model_views_find(&g_model_views,
        model_map, model_size, offset, len);
    if (index != UINT32_MAX) {
        *inner_offset = offset - g_model_views.entries[index].model_offset;
        return g_model_views.entries[index].buffer;
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    return nil;
}

/* Exact-view lifetimes live in the shared native/C header. */

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_impl(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset,
        ds4_gpu_exact_view_lifetime lifetime) {
    /* The Objective-C boundary checks driver/command collection availability;
     * native policy owns ranges, exact cache keys and retained references. */
    if (!g_device || (lifetime == DS4_GPU_EXACT_VIEW_TRANSIENT && !g_transient_buffers)) {
        fprintf(stderr, "ds4: Metal exact model range is outside the mapped model\n");
        return nil;
    }
    void *handle = ds4_metal_exact_cache_wrap(model_map, model_size, offset, len,
                                             inner_offset, lifetime);
    if (!handle) return nil;
    if (lifetime == DS4_GPU_EXACT_VIEW_CACHED)
        return (__bridge id<MTLBuffer>)handle; /* The native map retains it. */
    return (__bridge_transfer id<MTLBuffer>)handle;
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_CACHED);
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_transient(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_TRANSIENT);
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_owned(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_OWNED);
}

/* Production-linked exact-view lifetime oracle. The two fixture processes set
 * MIB=1 (evict) or MIB=0 (unlimited) before the static limit is first read. */
int ds4_gpu_test_exact_view_lifetimes(int expect_eviction) {
    const uint64_t model_bytes = 4ull * 1024 * 1024;
    const uint64_t view_bytes = 2ull * 1024 * 1024;
    const NSUInteger copied = 256;
    if (!ds4_gpu_init() || !ds4_gpu_synchronize()) return 0;
    uint8_t *model = mmap(NULL, (size_t)model_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) return 0;
    for (uint64_t i = 0; i < model_bytes; i++) model[i] = (uint8_t)(i * 17u + 3u);
    if (mprotect(model, (size_t)model_bytes, PROT_READ) != 0) {
        munmap(model, (size_t)model_bytes);
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        id<MTLBuffer> cached = nil, repeated = nil, shifted = nil;
        id<MTLBuffer> owned = nil, owned_again = nil, transient = nil;
        id<MTLBuffer> after = nil, cleared = nil, output = nil;
        id<MTLBlitCommandEncoder> blit = nil;
        uint64_t inner = 0, repeated_inner = 0, shifted_inner = 0;
        uint64_t owned_inner = 0, transient_inner = 0;
        const uint64_t offset = (uint64_t)sysconf(_SC_PAGESIZE) + 16u;
#define EXACT_REQUIRE(value) do { if (!(value)) { \
    fprintf(stderr, "ds4: exact-view lifetime test failed at %d: %s\n", __LINE__, #value); \
    goto done; \
} } while (0)
        ds4_gpu_model_buffer_cache_clear("exact-view fixture begin");
        EXACT_REQUIRE(ds4_gpu_begin_commands());
        cached = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset, view_bytes, &inner);
        repeated = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset, view_bytes, &repeated_inner);
        /* Different logical offsets normalize to the same four-part page key. */
        shifted = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset + 8u, view_bytes - 8u, &shifted_inner);
        EXACT_REQUIRE(cached && cached == repeated && cached == shifted);
        EXACT_REQUIRE(inner == 16u && repeated_inner == inner && shifted_inner == inner + 8u);
        owned = ds4_gpu_wrap_model_exact_range_owned(model, model_bytes, offset, view_bytes, &owned_inner);
        owned_again = ds4_gpu_wrap_model_exact_range_owned(model, model_bytes, offset, view_bytes, NULL);
        transient = ds4_gpu_wrap_model_exact_range_transient(model, model_bytes, offset, view_bytes, &transient_inner);
        EXACT_REQUIRE(owned && owned_again && transient);
        EXACT_REQUIRE(owned != cached && owned_again != owned && transient != cached && transient != owned);
        EXACT_REQUIRE(owned_inner == inner && transient_inner == inner);
        EXACT_REQUIRE([g_transient_buffers containsObject:transient]);
        EXACT_REQUIRE(![g_transient_buffers containsObject:owned] && ![g_transient_buffers containsObject:cached]);
        /* Insertion marks the cache over budget; it must not clear live views. */
        repeated = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset, view_bytes, NULL);
        EXACT_REQUIRE(repeated == cached);
        output = [g_device newBufferWithLength:3u * copied options:MTLResourceStorageModeShared];
        EXACT_REQUIRE(output);
        ds4_gpu_close_batch_encoder();
        blit = [g_batch_cb blitCommandEncoder];
        EXACT_REQUIRE(blit);
        [blit copyFromBuffer:cached sourceOffset:(NSUInteger)inner toBuffer:output destinationOffset:0 size:copied];
        [blit copyFromBuffer:owned sourceOffset:(NSUInteger)owned_inner toBuffer:output destinationOffset:copied size:copied];
        [blit copyFromBuffer:transient sourceOffset:(NSUInteger)transient_inner toBuffer:output destinationOffset:2u * copied size:copied];
        [blit endEncoding];
        blit = nil;
        g_batch_has_work = YES;
        EXACT_REQUIRE(ds4_gpu_end_commands());
        for (NSUInteger i = 0; i < 3; i++)
            EXACT_REQUIRE(!memcmp((uint8_t *)[output contents] + i * copied, model + offset, copied));
        EXACT_REQUIRE(![g_transient_buffers containsObject:transient]);
        /* Holding the old object prevents pointer reuse masking eviction. */
        after = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset, view_bytes, NULL);
        EXACT_REQUIRE(after && (expect_eviction ? after != cached : after == cached));
        ds4_gpu_model_buffer_cache_clear("exact-view fixture safe clear");
        cleared = ds4_gpu_wrap_model_exact_range(model, model_bytes, offset, view_bytes, NULL);
        EXACT_REQUIRE(cleared && cleared != after);
        ok = 1;
done:
        if (blit) [blit endEncoding];
        if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
        (void)ds4_gpu_synchronize();
        ds4_gpu_model_buffer_cache_clear("exact-view fixture retire");
        /* All buffers leave this scope before the no-copy backing is unmapped. */
#undef EXACT_REQUIRE
    }
    munmap(model, (size_t)model_bytes);
    if (ok) fprintf(stderr, "ds4: exact-view identity, owned/transient offset reads and safe %s eviction: PASS\n",
                    expect_eviction ? "bounded" : "unlimited");
    return ok;
}

uint32_t ds4_gpu_stream_expert_cache_configured_count(void){
    return ds4_native_expert_cache_configured_count(&g_expert_cache, g_ssd_streaming_mode != 0);
}

uint32_t ds4_gpu_stream_expert_cache_current_count(void){
    return ds4_native_expert_cache_current_count(&g_expert_cache, g_ssd_streaming_mode != 0);
}

uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes){
    return ds4_native_expert_cache_budget_for_expert_size(&g_expert_cache, g_ssd_streaming_mode != 0, gate_expert_bytes, down_expert_bytes);
}

static int ds4_gpu_stream_expert_cache_note_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes){
    return ds4_native_expert_cache_note_expert_size(&g_expert_cache, g_ssd_streaming_mode != 0, gate_expert_bytes, down_expert_bytes);
}

static uint32_t ds4_gpu_stream_expert_cache_requested_budget(void){
    return ds4_native_expert_cache_requested_budget(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static uint32_t ds4_gpu_stream_expert_cache_configured_budget(void){
    return ds4_native_expert_cache_configured_budget(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static uint32_t ds4_gpu_stream_expert_cache_effective_cap(
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected){
    return ds4_native_expert_cache_effective_cap(&g_expert_cache, g_ssd_streaming_mode != 0, layer, n_total_expert, n_selected);
}

static int ds4_gpu_stream_expert_timing_summary_enabled(void){
    return ds4_native_expert_timing_summary_enabled(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static uint32_t ds4_gpu_stream_expert_popcount(uint32_t mask) {
    return (uint32_t)__builtin_popcount(mask);
}

static int ds4_gpu_stream_expert_split_worthwhile(
        uint32_t resident_mask,
        uint32_t missing_mask) {
    if (resident_mask == 0 || missing_mask == 0) return 0;
    /*
     * The split path pays an extra command stage and a second routed-expert
     * bind.  It is worthwhile when several experts are missing and their SSD
     * reads can be hidden by resident expert work.  With one or two misses,
     * especially in large caches, a single unsplit routed pass is faster.
     */
    return ds4_gpu_stream_expert_popcount(missing_mask) >= 3u;
}

static void ds4_gpu_stream_expert_timing_note_selected(
        double sync_ms,
        double copy_ms,
        double bind_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_selected_calls++;
    g_stream_expert_timing_selected_sync_ms += sync_ms;
    g_stream_expert_timing_selected_copy_ms += copy_ms;
    g_stream_expert_timing_selected_read_ms += sync_ms + copy_ms;
    g_stream_expert_timing_selected_bind_ms += bind_ms;
}

static void ds4_gpu_stream_expert_timing_note_split(
        uint32_t resident_mask,
        uint32_t missing_mask,
        double   resident_ms,
        double   missing_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_split_layers++;
    g_stream_expert_timing_split_resident_experts +=
        ds4_gpu_stream_expert_popcount(resident_mask);
    g_stream_expert_timing_split_missing_experts +=
        ds4_gpu_stream_expert_popcount(missing_mask);
    g_stream_expert_timing_split_resident_ms += resident_ms;
    g_stream_expert_timing_split_missing_ms += missing_ms;
}

static void ds4_gpu_stream_expert_timing_note_split_missing_detail(
        double load_ms,
        double slot_ms,
        double prune_ms,
        double addr_ms,
        double wait_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_split_missing_load_ms += load_ms;
    g_stream_expert_timing_split_missing_slot_ms += slot_ms;
    g_stream_expert_timing_split_missing_prune_ms += prune_ms;
    g_stream_expert_timing_split_missing_addr_ms += addr_ms;
    g_stream_expert_timing_split_missing_wait_ms += wait_ms;
}





static void ds4_gpu_stream_expert_timing_note_reuse_scan(
        uint64_t entries,
        double   ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_reuse_scan_calls++;
    g_stream_expert_timing_reuse_scan_entries += entries;
    g_stream_expert_timing_reuse_scan_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_reuse_clear(double ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_reuse_clear_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_cache_class(
        uint32_t resident_mask,
        uint32_t missing_mask) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    const uint32_t resident = ds4_gpu_stream_expert_popcount(resident_mask);
    const uint32_t missing = ds4_gpu_stream_expert_popcount(missing_mask);
    if (missing == 0) {
        g_stream_expert_timing_cache_all_resident_layers++;
    } else if (resident == 0) {
        g_stream_expert_timing_cache_all_missing_layers++;
    } else {
        g_stream_expert_timing_cache_mixed_layers++;
    }
    g_stream_expert_timing_cache_resident_experts += resident;
    g_stream_expert_timing_cache_missing_experts += missing;
}



/* Actual ds4_gpu_stream_expert_pread_task is shared once in expert_load.h. */

/* Actual ds4_gpu_stream_expert_pending_load is shared once in expert_load.h. */

/* SAME actual pending/prefetch/pool state for C and native selection. ARC
 * initializes strong fields; SDK constants initialize real pthread records. */
static ds4_metal_expert_load_state g_expert_load = {
    .pool = { .mutex = PTHREAD_MUTEX_INITIALIZER,
              .start_cond = PTHREAD_COND_INITIALIZER,
              .done_cond = PTHREAD_COND_INITIALIZER }
};
#define g_stream_expert_pending_load (g_expert_load.pending)
#define g_glm_stream_selected_prefetch (g_expert_load.prefetch)

/* Actual ds4_gpu_glm_stream_selected_prefetch is shared once in expert_load.h. */



/* Actual ds4_gpu_stream_expert_pread_worker_args is shared once in expert_load.h. */






#define g_stream_expert_pread_pool_mutex (g_expert_load.pool.mutex)
#define g_stream_expert_pread_pool_start_cond (g_expert_load.pool.start_cond)
#define g_stream_expert_pread_pool_done_cond (g_expert_load.pool.done_cond)
#define g_stream_expert_pread_pool_threads (g_expert_load.pool.threads)
#define g_stream_expert_pread_pool_thread_count (g_expert_load.pool.thread_count)
#define g_stream_expert_pread_pool_active_workers (g_expert_load.pool.active_workers)
#define g_stream_expert_pread_pool_remaining_workers (g_expert_load.pool.remaining_workers)
#define g_stream_expert_pread_pool_n_tasks (g_expert_load.pool.n_tasks)
#define g_stream_expert_pread_pool_next_task (g_expert_load.pool.next_task)
#define g_stream_expert_pread_pool_generation (g_expert_load.pool.generation)
#define g_stream_expert_pread_pool_tasks (g_expert_load.pool.tasks)
#define g_stream_expert_pread_pool_initialized (g_expert_load.pool.initialized)
#define g_stream_expert_pread_pool_stopping (g_expert_load.pool.stopping)






/* Wait for gate/up tasks only; down tasks continue on the same pool. */
static int ds4_gpu_stream_expert_pending_gate_wait(void) {
return ds4_native_expert_pending_gate_wait();
}


static void ds4_gpu_stream_expert_pread_pool_shutdown(void) {
ds4_native_expert_pread_shutdown();
}



static uint32_t ds4_gpu_stream_expert_slab_locked_slot_count(void) {
return ds4_native_expert_locked_slots();
}



static void ds4_gpu_stream_expert_unlock_explicit_buffer(id<MTLBuffer> buffer){
    ds4_native_expert_unlock_explicit_buffer(&g_expert_cache, g_ssd_streaming_mode != 0, buffer);
}



/*
 * Large PRO caches otherwise create thousands of small shared Metal buffers.
 * Slabs keep the buffer object set small while locking pages only for slots
 * that actually hold a streamed expert.
 */


static int ds4_gpu_stream_expert_slab_slot_range(
        uint32_t slot,
        uint32_t *slab_index,
        uint64_t *slot_base){
    return ds4_native_expert_slab_slot_range(&g_expert_cache, g_ssd_streaming_mode != 0, slot, slab_index, slot_base);
}

static int ds4_gpu_stream_expert_slab_slot_for_buffer(
        id<MTLBuffer> buffer,
        NSUInteger    gate_inner,
        uint32_t     *slot_out){
    return ds4_native_expert_slab_slot_for_buffer(&g_expert_cache, g_ssd_streaming_mode != 0, buffer, gate_inner, slot_out);
}

static void ds4_gpu_stream_expert_slab_push_free_slot(uint32_t slot){
    ds4_native_expert_slab_push_free_slot(&g_expert_cache, g_ssd_streaming_mode != 0, slot);
}


static int ds4_gpu_stream_expert_slab_unlock_slot(uint32_t slot){
    return ds4_native_expert_slab_unlock_slot(&g_expert_cache, g_ssd_streaming_mode != 0, slot);
}



static uint64_t ds4_gpu_stream_expert_buffer_object_count(
        id<MTLBuffer> gate,
        id<MTLBuffer> up,
        id<MTLBuffer> down){
    return ds4_native_expert_buffer_object_count(&g_expert_cache, g_ssd_streaming_mode != 0, gate, up, down);
}

static int ds4_gpu_stream_expert_evict_dontneed_enabled(void){
    return ds4_native_expert_evict_dontneed_enabled(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_evict_dontneed_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len){
    ds4_native_expert_evict_dontneed_range(&g_expert_cache, g_ssd_streaming_mode != 0, model_map, model_size, offset, len);
}

static int ds4_gpu_stream_expert_split_requested(void) {
    return g_ssd_streaming_mode;
}

static uint32_t ds4_gpu_stream_expert_split_min_decode_tokens(void) {
    return 4;
}

static uint32_t ds4_gpu_stream_expert_split_min_cached(void) {
    uint32_t min_cached = 1024;
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget != 0 && budget < min_cached * 2u) {
        min_cached = budget / 2u;
    }
    return min_cached;
}

static int ds4_gpu_stream_expert_split_ready(void){
    return ds4_native_expert_split_ready(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_decay_route_hotness(void){
    ds4_native_expert_cache_decay_route_hotness(&g_expert_cache, g_ssd_streaming_mode != 0);
}

void ds4_gpu_stream_expert_cache_reset_route_hotness(void){
    ds4_native_expert_cache_reset_route_hotness(&g_expert_cache, g_ssd_streaming_mode != 0);
}

/* Experimental override; leave the established default unchanged. */
static uint32_t ds4_gpu_stream_expert_cache_decay_tokens(void){
    return ds4_native_expert_cache_decay_tokens(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_maybe_decay_route_hotness(void){
    ds4_native_expert_cache_maybe_decay_route_hotness(&g_expert_cache, g_ssd_streaming_mode != 0);
}

static void ds4_gpu_stream_expert_cache_note_route_hotness(
        uint32_t layer,
        uint32_t expert,
        uint32_t amount){
    ds4_native_expert_cache_note_route_hotness(&g_expert_cache, g_ssd_streaming_mode != 0, layer, expert, amount);
}

static void ds4_gpu_stream_expert_cache_note_selected_hotness(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected){
    ds4_native_expert_cache_note_selected_hotness(&g_expert_cache, g_ssd_streaming_mode != 0, layer, selected_ids, n_selected);
}

static void ds4_gpu_stream_expert_cache_note_frequency_hotness(
        uint32_t        layer,
        const uint32_t *frequency,
        uint32_t        n_total_expert){
    ds4_native_expert_cache_note_frequency_hotness(&g_expert_cache, g_ssd_streaming_mode != 0, layer, frequency, n_total_expert);
}

static void ds4_gpu_stream_expert_cache_note_token(uint32_t layer_index){
    ds4_native_expert_cache_note_token(&g_expert_cache, g_ssd_streaming_mode != 0, layer_index);
}

static int ds4_gpu_stream_expert_addr_table_requested(void) {
return ds4_native_expert_address_requested();
}



static int ds4_gpu_stream_prefill_batch_selected_addr_enabled(
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t gate_type,
        uint32_t down_type) {
return ds4_native_expert_prefill_enabled(n_tokens, n_total_expert, n_expert, gate_type, down_type);
}

void ds4_gpu_stream_expert_batch_set_ready(const ds4_gpu_tensor *selected) {
    g_stream_expert_batch_ready = selected;
}

int ds4_gpu_stream_expert_batch_supported(uint32_t rows, uint32_t total,
        uint32_t used, uint32_t gate_type, uint32_t down_type) {
    return ds4_gpu_stream_prefill_batch_selected_addr_enabled(rows, total, used,
        gate_type, down_type) && g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil;
}

static int ds4_gpu_glm_streaming_prefill_full_layer_active(void) {
    return g_glm_streaming_prefill_full_layer_runtime ||
           getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER") != NULL;
}

static int ds4_gpu_stream_full_expert_addr_table_requested(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") == NULL;
}

static uint64_t ds4_gpu_buffer_address(id<MTLBuffer> buffer, NSUInteger inner) {
    if (!buffer) return 0;
#if TARGET_OS_OSX
    if (@available(macOS 13.0, *)) {
        return (uint64_t)[buffer gpuAddress] + (uint64_t)inner;
    }
#endif
    return 0;
}

static int ds4_gpu_stream_compact_addr_ensure_buffers(uint32_t layer) {
    return ds4_native_table_stream_compact_addr_ensure_buffers(&g_expert_cache, layer);
}

static int ds4_gpu_stream_compact_addr_prepare(
        uint32_t layer,
        ds4_gpu_stream_expert_cache_entry * const entries[6],
        uint32_t n_entries,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        id<MTLBuffer> *selected_ids) {
    if (!entries || !gate_addrs || !up_addrs || !down_addrs || !selected_ids) return 0;
    if (!ds4_native_table_stream_compact_addr_prepare( &g_expert_cache, layer, entries, n_entries)) return 0;
    *gate_addrs=g_stream_compact_gate_addr_buffers[layer];
    *up_addrs=g_stream_compact_up_addr_buffers[layer];
    *down_addrs=g_stream_compact_down_addr_buffers[layer];
    *selected_ids=g_stream_compact_selected_buffers[layer];
    return 1;
}

static int ds4_gpu_stream_selected_ids_prepare(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected,
        id<MTLBuffer> *selected_buf,
        NSUInteger    *selected_off) {
    if (!selected_buf || !selected_off) return 0;
    if (!ds4_native_table_stream_selected_ids_prepare( &g_expert_cache, layer, selected_ids, n_selected)) return 0;
    *selected_buf=g_stream_selected_id_buffers[layer];
    *selected_off=0;
    return 1;
}

static int ds4_gpu_stream_expert_cache_ensure_addr_buffers(uint32_t layer) {
    return ds4_native_table_stream_expert_cache_ensure_addr_buffers(&g_expert_cache, layer);
}

static void ds4_gpu_stream_expert_cache_zero_addr_slot(uint32_t layer, uint32_t expert) {
    ds4_native_table_stream_expert_cache_zero_addr_slot(&g_expert_cache, layer, expert);
}

static int ds4_gpu_stream_expert_cache_set_addr_slot_raw(
        uint32_t      layer,
        uint32_t      expert,
        id<MTLBuffer> gate_buffer,
        NSUInteger    gate_inner,
        id<MTLBuffer> up_buffer,
        NSUInteger    up_inner,
        id<MTLBuffer> down_buffer,
        NSUInteger    down_inner) {
    return ds4_native_table_stream_expert_cache_set_addr_slot_raw(&g_expert_cache, layer, expert, gate_buffer, gate_inner, up_buffer, up_inner, down_buffer, down_inner);
}

static int ds4_gpu_stream_expert_cache_set_addr_slot(
        uint32_t      layer,
        uint32_t      expert,
        id<MTLBuffer> gate_buffer,
        NSUInteger    gate_inner,
        id<MTLBuffer> up_buffer,
        NSUInteger    up_inner,
        id<MTLBuffer> down_buffer,
        NSUInteger    down_inner) {
    if (!ds4_gpu_stream_expert_addr_table_requested()) return 1;
    return ds4_gpu_stream_expert_cache_set_addr_slot_raw(layer,
                                                         expert,
                                                         gate_buffer,
                                                         gate_inner,
                                                         up_buffer,
                                                         up_inner,
                                                         down_buffer,
                                                         down_inner);
}

static int ds4_gpu_stream_expert_cache_addr_buffers(
        uint32_t       layer,
        id<MTLBuffer> *gate,
        id<MTLBuffer> *up,
        id<MTLBuffer> *down) {
    if (!ds4_native_table_stream_expert_cache_addr_buffers( &g_expert_cache, layer)) return 0;
    if(gate)*gate=g_stream_expert_cache_gate_addr_buffers[layer];
    if(up)*up=g_stream_expert_cache_up_addr_buffers[layer];
    if(down)*down=g_stream_expert_cache_down_addr_buffers[layer];
    return 1;
}

static void ds4_gpu_stream_full_expert_addr_clear_layer(uint32_t layer) {
    ds4_native_table_stream_full_expert_addr_clear_layer(&g_expert_cache, layer);
}

static int ds4_gpu_stream_full_expert_addr_table_prepare(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        uint32_t       n_total_expert,
        uint64_t       gate_abs_offset,
        uint64_t       up_abs_offset,
        uint64_t       down_abs_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        ds4_gpu_stream_expert_cache_entry **entry_out) {
    ds4_gpu_stream_expert_cache_entry *entry = ds4_native_table_stream_full_expert_addr_table_prepare( &g_expert_cache, model_map, model_size, layer, n_total_expert, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes);
    if (!entry) return 0;
    if(gate_addrs)*gate_addrs=g_stream_expert_cache_gate_addr_buffers[layer];
    if(up_addrs)*up_addrs=g_stream_expert_cache_up_addr_buffers[layer];
    if(down_addrs)*down_addrs=g_stream_expert_cache_down_addr_buffers[layer];
    if(entry_out)*entry_out=entry;
    return 1;
}

static id<MTLBuffer> ds4_gpu_stream_expert_validate_status_buffer(void) {
    if (!g_device) return nil;
    if (g_stream_expert_validate_status_buffer) {
        return g_stream_expert_validate_status_buffer;
    }

    const NSUInteger bytes =
        (NSUInteger)DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS * sizeof(uint32_t);
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    if (!b) {
        fprintf(stderr, "ds4: Metal streaming expert validator allocation failed\n");
        return nil;
    }
    b.label = @"ds4_stream_expert_validate_status";
    memset([b contents], 0, bytes);
    g_stream_expert_validate_status_buffer = b;
    return b;
}

static int ds4_gpu_encode_stream_expert_cache_validate(
        id<MTLCommandBuffer> cb,
        const ds4_gpu_stream_expert_validate_args *args,
        id<MTLBuffer> selected,
        NSUInteger selected_off,
        id<MTLBuffer> gate_addrs,
        id<MTLBuffer> up_addrs,
        id<MTLBuffer> down_addrs,
        id<MTLBuffer> status) {
    if (!cb || !args || !selected || !gate_addrs || !up_addrs || !down_addrs ||
        !status || !g_moe_stream_expert_cache_validate_pipeline ||
        args->n_total_expert == 0 || args->n_total_expert > 384 ||
        args->n_expert == 0 || args->n_expert > 6) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_stream_expert_cache_validate_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:selected offset:selected_off atIndex:1];
    [enc setBuffer:gate_addrs offset:0 atIndex:2];
    [enc setBuffer:up_addrs offset:0 atIndex:3];
    [enc setBuffer:down_addrs offset:0 atIndex:4];
    [enc setBuffer:status offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_stream_expert_cache_validate_selected(
        const ds4_gpu_tensor *selected,
        id<MTLBuffer> gate_addrs,
        id<MTLBuffer> up_addrs,
        id<MTLBuffer> down_addrs,
        uint32_t n_total_expert,
        uint32_t n_expert,
        int32_t selected_ids[6],
        uint32_t *all_cached,
        uint32_t *miss_mask,
        uint32_t *invalid_mask) {
    if (!selected || !selected_ids || !all_cached || !miss_mask || !invalid_mask) {
        return 0;
    }
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> status = ds4_gpu_stream_expert_validate_status_buffer();
    if (!selectedbuf || !status) return 0;

    const NSUInteger status_bytes =
        (NSUInteger)DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS * sizeof(uint32_t);
    memset([status contents], 0, status_bytes);

    ds4_gpu_stream_expert_validate_args args = {
        .n_total_expert = n_total_expert,
        .n_expert = n_expert,
    };

    const int had_batch = g_batch_cb != nil;
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    if (!ds4_gpu_encode_stream_expert_cache_validate(cb,
                                                     &args,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     gate_addrs,
                                                     up_addrs,
                                                     down_addrs,
                                                     status)) {
        return 0;
    }

    if (had_batch) {
        if (ds4_gpu_end_commands() == 0) return 0;
    } else if (!ds4_gpu_finish_command_buffer(cb, owned,
                                              "streaming expert cache validator")) {
        return 0;
    }

    const uint32_t *words = (const uint32_t *)[status contents];
    *all_cached = words[0];
    *miss_mask = words[1];
    *invalid_mask = words[2];
    for (uint32_t i = 0; i < 6; i++) {
        selected_ids[i] = (int32_t)words[4 + i];
    }

    if (had_batch && ds4_gpu_begin_commands() == 0) return 0;
    return 1;
}

static void ds4_gpu_stream_expert_cache_clear_entry_internal(
        uint32_t                                  layer,
        uint32_t                                  expert,
        int                                       count_eviction,
        int                                       recycle_slab_slot,
        ds4_gpu_stream_expert_reusable_buffers   *reuse){
    ds4_native_expert_cache_clear_entry_internal(&g_expert_cache, g_ssd_streaming_mode != 0, layer, expert, count_eviction, recycle_slab_slot, reuse);
}

static void ds4_gpu_stream_expert_cache_clear_entry(
        uint32_t layer,
        uint32_t expert,
        int      count_eviction) {
    ds4_gpu_stream_expert_cache_clear_entry_internal(layer,
                                                     expert,
                                                     count_eviction,
                                                     1,
                                                     NULL);
}

/* Keep the surviving slab prefix warm when workspace growth reduces RAM
 * available for experts. Callers must drain GPU users before resizing. */
static int ds4_gpu_stream_expert_cache_trim(uint32_t experts){
    return ds4_native_expert_cache_trim(&g_expert_cache, g_ssd_streaming_mode != 0, experts);
}

static void ds4_gpu_stream_expert_cache_clear_all(int reset_stats){
    ds4_native_expert_cache_clear_all(&g_expert_cache, g_ssd_streaming_mode != 0, reset_stats);
}

static int ds4_gpu_stream_expert_cache_is_protected(
        uint32_t expert,
        const int32_t *protect_ids,
        uint32_t n_protect){
    return ds4_native_expert_cache_is_protected(&g_expert_cache, g_ssd_streaming_mode != 0, expert, protect_ids, n_protect);
}

static void ds4_gpu_stream_expert_cache_prune_layer(
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        const int32_t *protect_ids,
        uint32_t n_protect){
    ds4_native_expert_cache_prune_layer(&g_expert_cache, g_ssd_streaming_mode != 0, layer, n_total_expert, n_selected, protect_ids, n_protect);
}

static int ds4_gpu_stream_expert_cache_entry_protected(
        uint32_t layer,
        uint32_t expert,
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect){
    return ds4_native_expert_cache_entry_protected(&g_expert_cache, g_ssd_streaming_mode != 0, layer, expert, protect_layer, protect_ids, n_protect);
}

static int ds4_gpu_stream_expert_cache_entry_reusable(
        const ds4_gpu_stream_expert_cache_entry *e,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes){
    return ds4_native_expert_cache_entry_reusable(&g_expert_cache, g_ssd_streaming_mode != 0, e, gate_expert_bytes, down_expert_bytes);
}

static int ds4_gpu_stream_expert_cache_take_reusable(
        int                                     force_reuse,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuse){
    return ds4_native_expert_cache_take_reusable(&g_expert_cache, g_ssd_streaming_mode != 0, force_reuse, protect_layer, protect_ids, n_protect, gate_expert_bytes, down_expert_bytes, reuse);
}

static uint32_t ds4_gpu_stream_expert_cache_take_reusable_batch(
        uint32_t                                n_needed,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuses){
    return ds4_native_expert_cache_take_reusable_batch(&g_expert_cache, g_ssd_streaming_mode != 0, n_needed, protect_layer, protect_ids, n_protect, gate_expert_bytes, down_expert_bytes, reuses);
}


static uint32_t ds4_gpu_stream_expert_cache_release_mlock_margin(
        uint32_t       protect_layer,
        const int32_t *protect_ids,
        uint32_t       n_protect){
    return ds4_native_expert_cache_release_mlock_margin(&g_expert_cache, g_ssd_streaming_mode != 0, protect_layer, protect_ids, n_protect);
}



static void ds4_gpu_stream_expert_cache_prune_global(
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect){
    ds4_native_expert_cache_prune_global(&g_expert_cache, g_ssd_streaming_mode != 0, protect_layer, protect_ids, n_protect);
}

static int ds4_gpu_stream_expert_cache_entry_matches(
        const ds4_gpu_stream_expert_cache_entry *e,
        const void *model_map,
        uint64_t    model_size,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes){
    return ds4_native_expert_cache_entry_matches(&g_expert_cache, g_ssd_streaming_mode != 0, e, model_map, model_size, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes);
}

int ds4_gpu_stream_expert_cache_snapshot(const ds4_gpu_stream_expert_table *table,
        uint8_t *resident){
    return ds4_native_expert_cache_snapshot(&g_expert_cache, g_ssd_streaming_mode != 0, table, resident);
}

int ds4_gpu_stream_expert_cache_snapshot_selected(const ds4_gpu_stream_expert_table *table,
        const int32_t *selected, uint32_t n_selected, uint8_t *resident){
    return ds4_native_expert_cache_snapshot_selected(&g_expert_cache, g_ssd_streaming_mode != 0, table, selected, n_selected, resident);
}

static ds4_gpu_stream_expert_cache_entry *ds4_gpu_stream_expert_cache_peek(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes){
    return ds4_native_expert_cache_peek(&g_expert_cache, g_ssd_streaming_mode != 0, model_map, model_size, layer, expert, n_total_expert, n_selected, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes);
}

static ds4_gpu_stream_expert_cache_entry *
ds4_gpu_stream_expert_cache_install_loaded(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        uint32_t       expert,
        uint64_t       gate_abs_offset,
        uint64_t       up_abs_offset,
        uint64_t       down_abs_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer>  gate_buf,
        id<MTLBuffer>  up_buf,
        id<MTLBuffer>  down_buf,
        NSUInteger     gate_inner,
        NSUInteger     up_inner,
        NSUInteger     down_inner){
    return ds4_native_expert_cache_install_loaded(&g_expert_cache, g_ssd_streaming_mode != 0, model_map, model_size, layer, expert, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes, gate_buf, up_buf, down_buf, gate_inner, up_inner, down_inner);
}






static int ds4_gpu_stream_expert_pending_load_finish(
        ds4_gpu_stream_expert_cache_entry *entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED]) {
return ds4_native_expert_pending_finish(entries);
}

static void ds4_gpu_stream_expert_pending_load_clear(void) {
ds4_native_expert_pending_clear();
}

static int ds4_gpu_stream_expert_pending_load_matches(
        const void   *model_map,
        uint64_t      model_size,
        uint32_t      layer,
        const int32_t *selected_ids,
        uint32_t      n_total_expert,
        uint32_t      n_selected,
        uint64_t      gate_expert_bytes,
        uint64_t      down_expert_bytes) {
const ds4_gpu_stream_expert_table table = {
    .model_map = model_map, .model_size = model_size, .layer = layer,
    .n_total_expert = n_total_expert, .gate_offset = 0,
    .up_offset = 0, .down_offset = 0,
    .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes
};
return ds4_native_expert_pending_matches(&table, selected_ids, n_selected);
}



static int ds4_gpu_stream_expert_cache_load_selected_missing(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        const uint64_t *gate_abs_offsets,
        const uint64_t *up_abs_offsets,
        const uint64_t *down_abs_offsets,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        uint32_t       missing_mask,
        ds4_gpu_stream_expert_cache_entry **entries) {
const ds4_gpu_stream_expert_table table = {
    .model_map = model_map, .model_size = model_size, .layer = layer,
    .n_total_expert = n_total_expert, .gate_offset = 0,
    .up_offset = 0, .down_offset = 0,
    .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes
};
return ds4_native_expert_load_missing(&table, selected_ids, n_selected, gate_abs_offsets, up_abs_offsets, down_abs_offsets, missing_mask, entries);
}


static int ds4_gpu_glm_stream_selected_prefetch_take(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        int32_t    *selected_ids_out) {
const ds4_gpu_stream_expert_table table = {
    .model_map = model_map, .model_size = model_size, .layer = layer,
    .n_total_expert = n_total_expert, .gate_offset = gate_offset,
    .up_offset = up_offset, .down_offset = down_offset,
    .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes
};
return ds4_native_expert_prefetch_take(&table, n_selected, selected_ids_out);
}


static void ds4_gpu_stream_expert_cache_clear_layer(uint32_t layer){
    ds4_native_expert_cache_clear_layer(&g_expert_cache, g_ssd_streaming_mode != 0, layer);
}

static int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const ds4_gpu_tensor *selected,
        uint32_t       n_tokens,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        ds4_gpu_stream_expert_cache_entry **resources,
        uint32_t      *n_resources,
        uint32_t      *unique_out,
        id<MTLBuffer> *overflow_gate,
        id<MTLBuffer> *overflow_up,
        id<MTLBuffer> *overflow_down) {
/* Preserve caller-visible nil outputs and ARC writeback even on failure. */
if (overflow_gate) *overflow_gate = nil;
if (overflow_up) *overflow_up = nil;
if (overflow_down) *overflow_down = nil;
if (!gate_addrs || !up_addrs || !down_addrs || !resources || !n_resources ||
    !unique_out || !overflow_gate || !overflow_up || !overflow_down) return 0;
const ds4_gpu_stream_expert_table table = {
    .model_map = model_map, .model_size = model_size, .layer = layer,
    .n_total_expert = n_total_expert, .gate_offset = gate_offset,
    .up_offset = up_offset, .down_offset = down_offset,
    .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes
};
ds4_gpu_stream_expert_reusable_buffers addresses = {0}, overflow = {0};
const int ok = ds4_native_expert_prepare_batch(&table, selected, n_tokens, n_selected,
    &addresses, resources, n_resources, unique_out, &overflow);
*overflow_gate = overflow.gate_buffer;
*overflow_up = overflow.up_buffer;
*overflow_down = overflow.down_buffer;
if (ok) {
    *gate_addrs = addresses.gate_buffer;
    *up_addrs = addresses.up_buffer;
    *down_addrs = addresses.down_buffer;
}
return ok;
}




static uint32_t ds4_gpu_q4_expert_table_group_size(uint32_t n_total_expert) {
    return ds4_native_table_q4_expert_table_group_size(&g_expert_cache, n_total_expert);
}

static bool ds4_gpu_q4_table_queue_residency_requested(void) {
    return ds4_native_table_q4_table_queue_residency_requested(&g_expert_cache);
}

static bool ds4_gpu_q4_table_queue_residency_available(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        return g_queue && [g_queue respondsToSelector:@selector(addResidencySet:)];
    }
#endif
    return false;
}

static bool ds4_gpu_q4_non_streaming_opt_in_enabled(void) {
    return ds4_native_table_q4_non_streaming_opt_in_enabled(&g_expert_cache);
}

static bool ds4_gpu_q4_selected_paths_allowed(void) {
    return ds4_native_table_q4_selected_paths_allowed(&g_expert_cache);
}

int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return ds4_native_table_pro_q4_expert_table_auto_available(&g_expert_cache);
}

static bool ds4_gpu_q4_table_queue_residency_enabled(bool auto_queue_residency) {
    return ds4_native_table_q4_table_queue_residency_enabled(&g_expert_cache, auto_queue_residency);
}

static bool ds4_gpu_q4_table_model_residency_enabled(void) {
    return ds4_native_table_q4_table_model_residency_enabled(&g_expert_cache);
}

static bool ds4_gpu_pro_q4_expert_indirect_shape_supported(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    return ds4_native_table_pro_q4_expert_indirect_shape_supported(&g_expert_cache, n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}

static bool ds4_gpu_pro_q4_expert_table_auto_enabled(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    return ds4_native_table_pro_q4_expert_table_auto_enabled(&g_expert_cache, n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}

static bool ds4_gpu_pro_q4_expert_address_auto_enabled(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    return ds4_native_table_pro_q4_expert_address_auto_enabled(&g_expert_cache, n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}

static bool ds4_gpu_q4_table_bind_anchors_enabled(void) {
    return ds4_native_table_q4_table_bind_anchors_enabled(&g_expert_cache);
}

static int ds4_gpu_use_model_residency_set(id<MTLCommandBuffer> cb) {
    return ds4_native_table_use_model_residency_set(&g_expert_cache, cb);
}

static int ds4_gpu_bind_q4_expert_table_anchors(
        id<MTLComputeCommandEncoder> enc,
        DS4MetalQ4ExpertTable       *table,
        NSUInteger                   first_index,
        NSUInteger                   max_count) {
    return ds4_native_table_bind_q4_expert_table_anchors(&g_expert_cache, enc, table, first_index, max_count);
}





static id ds4_gpu_q4_expert_layer_residency_set(DS4MetalQ4ExpertTable *gate_table,
                                                DS4MetalQ4ExpertTable *up_table,
                                                DS4MetalQ4ExpertTable *down_table,
                                                bool auto_queue_residency) {
    return ds4_native_table_q4_expert_layer_residency_set(&g_expert_cache, gate_table, up_table, down_table, auto_queue_residency);
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_table(
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                tensor_offset,
        uint64_t                expert_bytes,
        uint32_t                n_total_expert,
        id<MTLArgumentEncoder>  encoder);

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_address_table(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    tensor_offset,
        uint64_t    expert_bytes,
        uint32_t    n_total_expert);

int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return ds4_native_table_preload_q4_expert_tables(&g_expert_cache, model_map, model_size, gate_offset, up_offset, down_offset, gate_expert_bytes, down_expert_bytes, n_total_expert);
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_table(
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                tensor_offset,
        uint64_t                expert_bytes,
        uint32_t                n_total_expert,
        id<MTLArgumentEncoder>  encoder) {
    return ds4_native_table_q4_expert_table(&g_expert_cache, model_map, model_size, tensor_offset, expert_bytes, n_total_expert, encoder);
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_address_table(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    tensor_offset,
        uint64_t    expert_bytes,
        uint32_t    n_total_expert) {
    return ds4_native_table_q4_expert_address_table(&g_expert_cache, model_map, model_size, tensor_offset, expert_bytes, n_total_expert);
}

static void ds4_gpu_use_q4_expert_table_resources(
        id<MTLCommandBuffer>        cb,
        id<MTLComputeCommandEncoder> enc,
        DS4MetalQ4ExpertTable       *table,
        bool                         queue_residency) {
    ds4_native_table_use_q4_expert_table_resources(&g_expert_cache, cb, enc, table, queue_residency);
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer score received undersized buffers\n");
            return 0;
        }

        if (n_head == 64 && head_dim == 128) {
            id<MTLComputePipelineState> direct_pipeline =
                ds4_gpu_hot_pipeline(g_dsv4_indexer_score_one_direct_pipeline,
                                        "kernel_dsv4_indexer_score_one_direct");
            if (!direct_pipeline) return 0;

            ds4_gpu_dsv4_indexer_scores_fused_args args = {
                .n_comp = n_comp,
                .n_tokens = 1,
                .n_head = n_head,
                .head_dim = head_dim,
                .pos0 = 0,
                .ratio = 4,
                .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
                .q_head_stride = (uint64_t)head_dim * sizeof(float),
                .weights_token_stride = (uint64_t)n_head * sizeof(float),
                .index_row_stride = (uint64_t)head_dim * sizeof(float),
                .score_token_stride = (uint64_t)n_comp * sizeof(float),
                .scale = scale,
            };

            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:direct_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
            [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
            [enc setThreadgroupMemoryLength:(128u + 4u) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_comp, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer direct score")) return 0;
            return 1;
        }

        const uint64_t head_score_bytes = (uint64_t)n_comp * n_head * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_head_scores_buffer,
                                             &g_indexer_head_scores_bytes,
                                             (NSUInteger)head_score_bytes,
                                             "ds4_indexer_head_scores")) {
            return 0;
        }

        ds4_gpu_q8_0_matvec_args dot_args =
            ds4_gpu_make_f32_mv_args(head_dim, n_comp, n_head);
        ds4_gpu_mv_dispatch dot_dispatch =
            ds4_gpu_make_plain_mv_dispatch(head_dim, 1);
        dot_args.nr0 = dot_dispatch.nr0;
        id<MTLComputePipelineState> dot_pipeline =
            ds4_gpu_get_mul_mv_pipeline(dot_dispatch.function_name, dot_dispatch.nsg);
        if (!dot_pipeline) return 0;
        ds4_gpu_dsv4_indexer_weighted_sum_args sum_args = {
            .ne00 = (int64_t)n_comp,
            .ne01 = 1,
            .ne02 = (int64_t)n_head,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * sizeof(float),
            .ne10 = (int64_t)n_head,
            .ne11 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)n_head * sizeof(float),
            .ne0 = (int64_t)n_comp,
            .ne1 = 1,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:dot_pipeline];
        [enc setBytes:&dot_args length:sizeof(dot_args) atIndex:0];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:3];
        if (dot_dispatch.smem) {
            [enc setThreadgroupMemoryLength:dot_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + (NSUInteger)dot_dispatch.nr0 - 1u) / (NSUInteger)dot_dispatch.nr0,
                                              n_head,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dot_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_weighted_sum_pipeline];
        [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer score")) return 0;
    }

    return 1;
}

static int ds4_gpu_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer prefill scores received undersized buffers\n");
            return 0;
        }
        if (head_dim != 128) {
            fprintf(stderr, "ds4: Metal fused DS4 indexer scores expect 128-wide rows\n");
            return 0;
        }
        /*
         * The NAX/TensorOps score builder is a prefill-only win.  At small
         * batches and in one-token decode the setup cost is not amortized, so
         * those paths keep the older direct/tiled score kernels.
         */
        const bool use_nax = ds4_gpu_mpp_available() && n_tokens >= 16u;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(
            use_nax ? "kernel_dsv4_indexer_scores_nax" :
            (g_quality_mode ? "kernel_dsv4_indexer_scores_tiled_f32"
                            : "kernel_dsv4_indexer_scores_tiled"));
        if (!pipeline) return 0;

        ds4_gpu_dsv4_indexer_scores_fused_args args = {
            .n_comp = n_comp,
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .pos0 = pos0,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
            .q_head_stride = (uint64_t)head_dim * sizeof(float),
            .weights_token_stride = (uint64_t)n_head * sizeof(float),
            .index_row_stride = (uint64_t)head_dim * sizeof(float),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        if (use_nax) {
            const NSUInteger q_shared = 2u * 32u * 32u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 32u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 15u) / 16u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (g_quality_mode) {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared + dot_shared) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer prefill scores")) return 0;
    }

    return 1;
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 pos0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}


int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0) return 0;
    if (ds4_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(logits) < (uint64_t)n_vocab * sizeof(float)) {
        fprintf(stderr, "ds4: Metal graph argmax received undersized buffers\n");
        return 0;
    }

    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1, 1);
}


static int ds4_gpu_matmul_q8_0_legacy_tensor(ds4_gpu_tensor *out,
        const void *map, uint64_t size, uint64_t offset, uint64_t width,
        uint64_t outputs, const ds4_gpu_tensor *x, uint64_t rows,
        bool prefer_decode_mpp, bool prefill_default) {
    return ds4_native_metal_q8_legacy(out, map, size, offset, width, outputs,
                                     x, rows, prefer_decode_mpp, prefill_default);
}









int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}





/* Decode-only fusion of the router logits matvec with the shared-expert
 * gate/up SwiGLU: one dispatch instead of two on the same normalized FFN
 * input.  Bit-exact by construction (see src/metal/dense.metal).  Returns 1 on
 * success, 0 when the shape is unsupported (caller falls back), -1 on
 * error. */
















/* Quad variant of the paired compressor projection: the attention compressor
 * and indexer compressor pairs share the input activation and F16 matvec
 * shape, so one dispatch covers all four matrices.  Bit-exact by
 * construction (see src/metal/dense.metal).  Returns 1 when the fused dispatch
 * ran, 0 when the caller should use the separate paths, -1 on error. */

/* Decode-only emit-path fusion: finalize the freshly pooled attention and
 * indexer compressor rows (norm + rope + fp8/commit + qat) in one dispatch
 * instead of seven.  Bit-exact vs the separate dispatches; see the kernel
 * comment.  Returns 1 when the fused dispatch ran, 0 to fall back. */

/* Decode-only fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch (both read the same normalized input).
 * Bit-exact vs the separate dispatches; see the kernel comment.  Returns
 * 1 when the fused dispatch ran, 0 when the caller must use the separate
 * paths, -1 on error. */







/* Decode-only triple fusion: q/kv RMS norm + KV RoPE tail + FP8/raw store.
 * Same arithmetic, order and rounding as the three separate dispatches (see
 * src/metal/norm.metal); gated and verified against full-vocabulary logits.
 * Returns 1 on success, 0 on unsupported shape (caller falls back). */
/* Deferred kv norm/RoPE/FP8 task: the q/kv norm dispatch runs its q task
 * only and the kv task is folded into the KV staging kernel of the same
 * layer (ds4_gpu_kv_norm_task_flush runs it standalone if no staging
 * consumed it). */
#define g_kv_task g_schedule.kv
#define g_qkv_norm_defer_kv g_schedule.defer_kv

void ds4_gpu_dsv4_qkv_norm_defer_kv_next(void) {
    ds4_native_metal_defer_kv_next();
}

int ds4_gpu_kv_norm_task_pending(void) {
    return g_kv_task.pending;
}

/* Run a deferred kv task standalone (the full q/kv norm kernel; the q task
 * recomputes identical values).  Used when no KV staging consumed it. */
int ds4_gpu_kv_norm_task_flush(void) {
    return ds4_native_metal_kv_flush();
}

/* Run the deferred kv task in a concurrent encoder left open for the
 * caller's next dispatch (the q_b projection), so its ~9 us of serial work
 * hides under that stream.  End closes the section. */
int ds4_gpu_kv_norm_task_begin_concurrent(void) {
    return ds4_native_metal_kv_begin();
}

void ds4_gpu_kv_norm_task_end_concurrent(void) {
    ds4_native_metal_kv_end();
}

int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint64_t              raw_cap,
        uint32_t              raw_row,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps) {
    const ds4_metal_fused_kv_request request = {
        .q_out = q_out, .q = q, .map = model_map, .size = model_size,
        .q_weight_offset = q_weight_offset, .q_n = q_n,
        .kv_out = kv_out, .kv = kv, .kv_weight_offset = kv_weight_offset, .kv_n = kv_n,
        .raw = raw_cache, .raw_cap = raw_cap, .raw_row = raw_row,
        .n_rot = n_rot, .pos0 = pos0, .n_ctx_orig = n_ctx_orig,
        .freq_base = freq_base, .freq_scale = freq_scale, .ext_factor = ext_factor,
        .attn_factor = attn_factor, .beta_fast = beta_fast, .beta_slow = beta_slow, .eps = eps,
    };
    return ds4_native_metal_fused_kv(&request);
}










static int ds4_gpu_encode_set_rows_f32_i32(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *dst,
        id<MTLBuffer>        srcbuf,
        NSUInteger           src_off,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             dst_rows,
        uint32_t             width) {
    return ds4_native_metal_set_rows((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)srcbuf, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)ds4_gpu_tensor_buffer(dst), ds4_gpu_tensor_offset(dst), ds4_gpu_tensor_bytes(dst)},
        rows, n_rows, dst_rows, width);
}

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    return ds4_native_metal_add_1d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)a, a_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)b, b_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)out, out_off, 0}, n);
}


/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
/* The fused KV RoPE/FP8 kernel replicates the affine decode RoPE specialisation,
 * so it is only valid where that specialisation is the path in use. */

/* Decode-only fusion: one dispatch does the KV RoPE tail and the FP8/raw
 * finalizer that previously cost two. Same arithmetic, same order; gated and
 * verified against full-vocabulary logits. */



/* ds4_gpu_cpy_tile_rows_args lives once in compressor.h. */

/* dst[t*width + e] = ape[((pos0 + t) % ratio)*width + e] for t < n_tokens,
 * converting f16 -> f32 when ape_type == 1.  One dispatch for the whole
 * batch (the previous per-segment copies were n_tokens/ratio encoders). */

























static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value) {
    if (!cb || !buf || width == 0 || rows == 0 || (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth_max = g_unary_fill_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}















static NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    return ds4_native_metal_copy_3d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0},
        cols, rows, planes, sizeof(float), src_row_stride, src_plane_stride,
        dst_row_stride, dst_plane_stride, DS4_PRIM_COPY_F32);
}

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    return ds4_native_metal_copy_3d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0},
        cols, rows, planes, src_col_stride, src_row_stride, src_plane_stride,
        dst_row_stride, dst_plane_stride, DS4_PRIM_COPY_F32);
}

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    return ds4_native_metal_copy_1d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0}, n, DS4_PRIM_COPY_F32_F16, false);
}


static int ds4_gpu_encode_cpy_f32_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    return ds4_native_metal_copy_3d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0},
        cols, rows, planes, sizeof(float), src_row_stride, src_plane_stride,
        dst_row_stride, dst_plane_stride, DS4_PRIM_COPY_F32_F16);
}

static int ds4_gpu_encode_cpy_f16_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    return ds4_native_metal_copy_3d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0},
        cols, rows, planes, sizeof(uint16_t), src_row_stride, src_plane_stride,
        dst_row_stride, dst_plane_stride, DS4_PRIM_COPY_F16);
}

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    return ds4_native_metal_copy_1d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0}, n, DS4_PRIM_COPY_F16_F32, false);
}

static int ds4_gpu_encode_copy_to_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        bool                 src_is_f16,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    return ds4_native_metal_copy_1d((__bridge void *)cb,
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)src, src_off, 0},
        (ds4_metal_attention_view){(ds4_metal_model_buffer *)(__bridge void *)dst, dst_off, 0}, n,
        src_is_f16 ? DS4_PRIM_COPY_F16 : DS4_PRIM_COPY_F32_F16, true);
}



static int ds4_gpu_encode_fill_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             n,
        float                value) {
    if (!cb || !buf || n == 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(n, 1, 0, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth = (NSUInteger)n;
    const NSUInteger max_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}


/* Rectangular causal/window mask for raw prefill: q covers rows
 * [q_row0, q_row0 + n_q) of an n_kv-token block whose keys all live in
 * rows [0, n_kv).  The square prefill case is q_row0 == 0, n_q == n_kv. */

static void ds4_gpu_fill_glm_prefill_mask(
        uint16_t *mask,
        uint32_t  pos0,
        uint32_t  n_tokens,
        uint32_t  cache_len) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * cache_len;
        for (uint32_t k = 0; k < cache_len; k++) {
            row[k] = k <= qpos ? 0u : neg_inf_half;
        }
    }
}

static id<MTLBuffer> ds4_gpu_glm_prefill_mask_buffer(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        NSUInteger mask_bytes) {
    const int same_shape =
        g_glm_flash_attn_mask_valid &&
        g_glm_flash_attn_mask_buffer &&
        g_glm_flash_attn_mask_bytes >= mask_bytes &&
        g_glm_flash_attn_mask_pos0 == pos0 &&
        g_glm_flash_attn_mask_tokens == n_tokens &&
        g_glm_flash_attn_mask_cache_len == cache_len;
    if (same_shape) return g_glm_flash_attn_mask_buffer;

    if (g_glm_flash_attn_mask_buffer) {
        [g_transient_buffers addObject:g_glm_flash_attn_mask_buffer];
        g_glm_flash_attn_mask_buffer = nil;
    }
    g_glm_flash_attn_mask_bytes = 0;
    g_glm_flash_attn_mask_valid = 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_glm_flash_attn_mask_buffer,
                                        &g_glm_flash_attn_mask_bytes,
                                        mask_bytes,
                                        "ds4_glm_flash_attn_mask")) {
        return nil;
    }

    ds4_gpu_fill_glm_prefill_mask((uint16_t *)[g_glm_flash_attn_mask_buffer contents],
                                  pos0,
                                  n_tokens,
                                  cache_len);
    g_glm_flash_attn_mask_pos0 = pos0;
    g_glm_flash_attn_mask_tokens = n_tokens;
    g_glm_flash_attn_mask_cache_len = cache_len;
    g_glm_flash_attn_mask_valid = 1;
    return g_glm_flash_attn_mask_buffer;
}








typedef struct {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
} ds4_gpu_directional_steering_project_args;

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_directional_steering_project_f32");
        if (!pipeline) return 0;

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> dbuf = ds4_gpu_tensor_buffer(directions);
        const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
        const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
        if (!xbuf || !dbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(directions) < dir_bytes) {
            fprintf(stderr, "ds4: Metal directional steering received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        while (nth > width && nth > 1u) nth >>= 1;
        if (nth == 0) nth = 1;

        ds4_gpu_directional_steering_project_args args = {
            .width = width,
            .rows = rows,
            .layer = layer,
            .n_threads = (uint32_t)nth,
            .scale = scale,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:dbuf offset:ds4_gpu_tensor_offset(directions) atIndex:2];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "directional steering")) return 0;
    }

    return 1;
}

static NSUInteger ds4_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}

static int ds4_gpu_encode_unary_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer>               src,
        NSUInteger                  src_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        uint32_t                    width,
        uint32_t                    rows,
        int                         c4,
        float                       min,
        float                       max) {
    if (!cb || !pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if (c4 && (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, c4, 0.0f, 0.0f);
    args.min = min;
    args.max = max;

    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = ds4_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows) {
    const uint64_t lhs_row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rhs_row_bytes = sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = lhs_row_bytes,
        .nb02 = (uint64_t)rows * lhs_row_bytes,
        .nb03 = (uint64_t)rows * lhs_row_bytes,
        .ne10 = 1,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = (uint64_t)rows * rhs_row_bytes,
        .nb13 = (uint64_t)rows * rhs_row_bytes,
        .ne0 = (int32_t)width,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = lhs_row_bytes,
        .nb2 = (uint64_t)rows * lhs_row_bytes,
        .nb3 = (uint64_t)rows * lhs_row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (ds4_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
    };
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return ds4_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (ds4_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
    };
}

static const char *ds4_gpu_metal_tensor_type_name(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return "iq2_xxs";
    case DS4_METAL_TENSOR_Q2_K:    return "q2_k";
    case DS4_METAL_TENSOR_Q4_K:    return "q4_k";
    case DS4_METAL_TENSOR_Q5_K:    return "q5_k";
    case DS4_METAL_TENSOR_Q6_K:    return "q6_k";
    case DS4_METAL_TENSOR_MXFP4:   return "mxfp4";
    default:                       return "unknown";
    }
}

static const char *ds4_gpu_trim_env_value(const char *env, size_t *len_out) {
    if (len_out) *len_out = 0;
    if (!env) return NULL;

    while (isspace((unsigned char)*env)) env++;
    size_t n = strlen(env);
    while (n > 0 && isspace((unsigned char)env[n - 1])) n--;
    if (len_out) *len_out = n;
    return env;
}

static bool ds4_gpu_profile_layer_value_match(const char *env, uint32_t layer_index) {
    size_t env_len = 0;
    env = ds4_gpu_trim_env_value(env, &env_len);
    if (!env || env_len == 0) return true;
    if (ds4_gpu_env_value_eq(env, env_len, "all")) return true;

    const char *p = env;
    const char *end_env = env + env_len;
    while (p < end_env) {
        while (p < end_env && (*p == ' ' || *p == '\t' || *p == ',')) p++;
        if (p >= end_env) break;

        char *end = NULL;
        const unsigned long first = strtoul(p, &end, 10);
        if (end == p || end > end_env || first > UINT32_MAX) return false;

        unsigned long last = first;
        p = end;
        if (p < end_env && *p == '-') {
            p++;
            last = strtoul(p, &end, 10);
            if (end == p || end > end_env || last > UINT32_MAX) return false;
            p = end;
        }

        if (first <= layer_index && layer_index <= last) return true;
        while (p < end_env && (*p == ' ' || *p == '\t')) p++;
        if (p < end_env && *p != ',') return false;
    }
    return false;
}

static bool ds4_gpu_stage_profile_enabled_for_layer(const char *flag_env_name,
                                                    const char *layer_env_name,
                                                    uint32_t    layer_index) {
    size_t flag_len = 0;
    const char *flag = ds4_gpu_trim_env_value(getenv(flag_env_name), &flag_len);
    if (!flag) return false;

    size_t layer_len = 0;
    const char *layer = ds4_gpu_trim_env_value(getenv(layer_env_name), &layer_len);
    const bool has_layer_filter = layer && layer_len != 0;

    if (flag_len != 0) {
        if (ds4_gpu_env_value_eq(flag, flag_len, "0") ||
            ds4_gpu_env_value_eq(flag, flag_len, "false") ||
            ds4_gpu_env_value_eq(flag, flag_len, "no") ||
            ds4_gpu_env_value_eq(flag, flag_len, "off")) {
            return false;
        }
        if (!has_layer_filter &&
            !ds4_gpu_env_value_eq(flag, flag_len, "1") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "true") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "yes") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "on") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "all")) {
            return ds4_gpu_profile_layer_value_match(flag, layer_index);
        }
    }

    return ds4_gpu_profile_layer_value_match(layer, layer_index);
}

/* TensorOps routed-MoE prefill uses bits 0/1/2 for gate/up/down. Unsupported
 * tensor types keep their established kernels. --quality and the global
 * Metal4 comparison switch retain the reference path. */

static id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
    case DS4_METAL_TENSOR_Q8_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f32", false);
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
    case DS4_METAL_TENSOR_Q5_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f32", false);
    case DS4_METAL_TENSOR_Q6_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f32", false);
    case DS4_METAL_TENSOR_MXFP4:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_mxfp4_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_addr_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q4_K_f32", false);
    case DS4_METAL_TENSOR_MXFP4:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_mxfp4_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f16", false);
    case DS4_METAL_TENSOR_Q8_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f16", false);
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    case DS4_METAL_TENSOR_Q5_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f16", false);
    case DS4_METAL_TENSOR_Q6_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f16", false);
    case DS4_METAL_TENSOR_MXFP4:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_mxfp4_f16", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_addr_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q4_K_f16", false);
    case DS4_METAL_TENSOR_MXFP4:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_mxfp4_f16", false);
    default:
        return nil;
    }
}

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}


static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes =
        (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
    const uint64_t pair_rows =
        (uint64_t)(uint32_t)mm_args->ne20 * (uint32_t)mm_args->ne21;
    const uint64_t work_cap =
        (pair_rows + 31u * (uint32_t)mm_args->ne02 + 31u) / 32u;
    const NSUInteger work_item_bytes = 2u * sizeof(uint32_t);
    if (work_cap > (NSUIntegerMax - 8u) / work_item_bytes ||
        work_offset > NSUIntegerMax - 8u -
                          (NSUInteger)work_cap * work_item_bytes) {
        return 0;
    }
    const NSUInteger total_bytes =
        work_offset + 8u + (NSUInteger)work_cap * work_item_bytes;
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         total_bytes,
                                         "ds4_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:4];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}


static int ds4_gpu_encode_mul_mm_id_mapped_tile_resources(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t                    resource_count,
        uint32_t                    resource_kind,
        id<MTLBuffer>               overflow_resource,
        uint32_t                    tile_rows) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0 ||
        (tile_rows != 32u && tile_rows != 64u)) {
        return 0;
    }
    /* The simdgroup and TensorOps kernels share the 32-token work map. */
    const bool use_resource_hints =
        getenv("DS4_METAL_MOE_MM_ID_USE_RESOURCES") != NULL &&
        getenv("DS4_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES") == NULL;

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) {
        return 0;
    }
    const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
    const uint64_t pair_rows =
        (uint64_t)(uint32_t)mm_args->ne20 * (uint32_t)mm_args->ne21;
    const uint64_t work_cap =
        (pair_rows + 31u * (uint32_t)mm_args->ne02 + 31u) / 32u;
    const NSUInteger work_item_bytes = 2u * sizeof(uint32_t);
    if (work_cap > NSUIntegerMax ||
        work_offset > NSUIntegerMax - 8u ||
        (NSUInteger)work_cap >
            (NSUIntegerMax - work_offset - 8u) / work_item_bytes ||
        g_moe_id_map_bytes <
            work_offset + 8u + (NSUInteger)work_cap * work_item_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:6];
    for (uint32_t i = 0; resources && i < resource_count; i++) {
        ds4_gpu_stream_expert_cache_entry *entry = resources[i];
        id<MTLBuffer> b = resource_kind == 0 ? entry->gate_buffer :
                          resource_kind == 1 ? entry->up_buffer : entry->down_buffer;
        [enc useResource:b usage:MTLResourceUsageRead];
    }
    if (overflow_resource) [enc useResource:overflow_resource usage:MTLResourceUsageRead];
    if (use_resource_hints) {
        [enc useResource:src0 usage:MTLResourceUsageRead];
    }
    [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)work_cap,
                                          ((NSUInteger)mm_args->ne0 + tile_rows - 1u) / tile_rows,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mm_id_args *args,
        id<MTLBuffer> src0, NSUInteger src0_off,
        id<MTLBuffer> src1, NSUInteger src1_off,
        id<MTLBuffer> dst, NSUInteger dst_off, NSUInteger threadgroup_bytes) {
    return ds4_gpu_encode_mul_mm_id_mapped_tile_resources(
        cb, pipeline, args, src0, src0_off, src1, src1_off, dst, dst_off,
        threadgroup_bytes, NULL, 0, 0, nil, 64);
}

static int ds4_gpu_encode_mul_mm_id_addr_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t                    resource_count,
        uint32_t                    resource_kind,
        id<MTLBuffer>               overflow_resource) {
    if (!cb || !mm_pipeline || !mm_args || !src0_addrs || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tile_n = 32u;
    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes =
        (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0_addrs offset:0 atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc useResource:src0_addrs usage:MTLResourceUsageRead];
    for (uint32_t i = 0; resources && i < resource_count; i++) {
        ds4_gpu_stream_expert_cache_entry *entry = resources[i];
        if (!entry) continue;
        id<MTLBuffer> b =
            resource_kind == 0 ? entry->gate_buffer :
            resource_kind == 1 ? entry->up_buffer :
                                 entry->down_buffer;
        if (b) [enc useResource:b usage:MTLResourceUsageRead];
    }
    if (overflow_resource) {
        [enc useResource:overflow_resource usage:MTLResourceUsageRead];
    }
    [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + tile_n - 1u) / tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}


static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    return ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                  mm_pipeline,
                                                  mm_args,
                                                  src0,
                                                  src0_off,
                                                  src1,
                                                  src1_off,
                                                  dst,
                                                  dst_off,
                                                  8192u);
}



static int ds4_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        .write_clamped = getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ? 1u : 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_sum6(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum6_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 6u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum6_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum6_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_sum8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum8_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 8u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum8_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 6 &&
        ds4_gpu_encode_moe_sum6(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    if (n_expert == 8 &&
        ds4_gpu_encode_moe_sum8(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}





static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}



int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !kv_raw || !model_map ||
        n_tokens == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 ||
        kv_lora_dim > kv_raw_dim || (kv_lora_dim & 3u) != 0 ||
        !isfinite(eps) || eps < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
        if (!rawbuf || !outbuf ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal GLM KV RMS norm received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM KV RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        const bool exact_decode_weight_view =
            n_tokens == 1u &&
            weight_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_DISABLE_DECODE_NORM_EXACT_VIEWS") == NULL;
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = exact_decode_weight_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset,
                                           weight_bytes,
                                           &weight_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &weight_inner);
        if (!weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_kv_lora_rms_norm_pipeline,
                                 "kernel_glm_kv_lora_rms_norm");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_kv_lora_rms_norm_args args = {
            .n_tokens = n_tokens,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .eps = eps,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(kv_lora_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM KV RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !kv_norm || !model_map ||
        n_tokens == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || n_head == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_norm);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM k_b projection received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t kv_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        if (!outbuf || !kvbuf ||
            ds4_gpu_tensor_bytes(kv_norm) < kv_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal GLM k_b projection received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM k_b projection range is outside the mapped model\n");
            return 0;
        }
        const NSUInteger q_blocks = ((NSUInteger)qk_nope + 31u) / 32u;
        if (q_blocks > 8u) {
            fprintf(stderr, "ds4: Metal GLM k_b projection q width is too large for the tiled kernel\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_k_b_project_pipeline,
                                 "kernel_glm_k_b_project_q8_0");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_k_b_project_args args = {
            .n_tokens = n_tokens,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .n_head = n_head,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .pad1 = 0,
            .pad2 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv_norm) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(NSUInteger)kv_lora_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(32, q_blocks, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM k_b projection")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    return ds4_gpu_glm_k_b_project_typed_tensor(out,
                                                kv_norm,
                                                model_map,
                                                model_size,
                                                weight_offset,
                                                DS4_METAL_TENSOR_Q8_0,
                                                n_tokens,
                                                kv_lora_dim,
                                                qk_nope,
                                                n_head);
}

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_lora_cache || !k_rope_cache || !kv_norm || !kv_raw ||
        n_tokens == 0 || cache_cap == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> kvnormbuf = ds4_gpu_tensor_buffer(kv_norm);
        id<MTLBuffer> kvrawbuf = ds4_gpu_tensor_buffer(kv_raw);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t kv_norm_bytes =
            (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t kv_raw_bytes =
            (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        if (!kvcachebuf || !ropecachebuf || !kvnormbuf || !kvrawbuf ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(kv_norm) < kv_norm_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < kv_raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM compact KV store received undersized buffers\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_store_compact_kv_pipeline,
                                 "kernel_glm_store_compact_kv");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_store_compact_kv_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_rope = qk_rope,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad1 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvnormbuf offset:ds4_gpu_tensor_offset(kv_norm) atIndex:1];
        [enc setBuffer:kvrawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM compact KV store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_lora_cache || !k_rope_cache || !kv_raw ||
        !model_map || n_tokens == 0 || cache_cap == 0 ||
        q_n == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 ||
        (q_n & 3u) != 0 || (kv_lora_dim & 3u) != 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> kvrawbuf = ds4_gpu_tensor_buffer(kv_raw);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t kv_raw_bytes =
            (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        if (!qbuf || !qoutbuf || !kvcachebuf || !ropecachebuf || !kvrawbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * n_tokens ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * n_tokens ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < kv_raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM fused q/kv norm compact store received undersized buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_weight_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM fused q/kv norm compact store weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_weight_inner = 0;
        id<MTLBuffer> q_weightbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                              q_weight_offset, q_row_bytes,
                                                              &q_weight_inner);
        if (!q_weightbuf) return 0;
        uint64_t kv_weight_inner = 0;
        id<MTLBuffer> kv_weightbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                               kv_weight_offset, kv_weight_bytes,
                                                               &kv_weight_inner);
        if (!kv_weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_qkv_norm_store_compact_kv_pipeline,
                                 "kernel_glm_qkv_norm_store_compact_kv");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_qkv_norm_store_compact_kv_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .q_n = q_n,
            .q_n4 = q_n / 4u,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .kv_lora_n4 = kv_lora_dim / 4u,
            .qk_rope = qk_rope,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .eps = eps,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_weightbuf offset:(NSUInteger)q_weight_inner atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvrawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:4];
        [enc setBuffer:kv_weightbuf offset:(NSUInteger)kv_weight_inner atIndex:5];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:6];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:7];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 3, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM fused q/kv norm compact store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!indexer_key_cache || !raw_k || !model_map ||
        n_tokens == 0 || cache_cap == 0 ||
        head_dim == 0 || rot_dim == 0 ||
        rot_dim > head_dim || (rot_dim & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> cachebuf = ds4_gpu_tensor_buffer(indexer_key_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_k);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t cache_bytes =
            (uint64_t)cache_cap * head_dim * cache_elem_bytes;
        const uint64_t raw_bytes =
            (uint64_t)n_tokens * head_dim * sizeof(float);
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
        if (!cachebuf || !rawbuf ||
            ds4_gpu_tensor_bytes(indexer_key_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(raw_k) < raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexer K store received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || norm_bytes > model_size - weight_offset ||
            bias_offset > model_size || norm_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM indexer K norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        uint64_t bias_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight_offset, norm_bytes,
                                     &weight_inner);
        id<MTLBuffer> biasbuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     bias_offset, norm_bytes,
                                     &bias_inner);
        if (!weightbuf || !biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_store_indexer_k_pipeline,
                                 "kernel_glm_store_indexer_k");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_store_indexer_k_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .head_dim = head_dim,
            .rot_dim = rot_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .pad1 = 0.0f,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(head_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_k) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:3];
        [enc setBuffer:cachebuf offset:ds4_gpu_tensor_offset(indexer_key_cache) atIndex:4];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexer K store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm53_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool_cache,
        ds4_gpu_tensor       *tail_k,
        ds4_gpu_tensor       *tail_gate,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              norm_bias_offset,
        uint64_t              ape_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        float                 eps,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!pool_cache || !tail_k || !tail_gate || !raw_k || !gate || !model_map ||
        n_tokens == 0 || cache_cap == 0 || head_dim == 0 || pool_size != 4u ||
        head_dim > 1024u || !isfinite(eps) || eps <= 0.0f ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> poolbuf = ds4_gpu_tensor_buffer(pool_cache);
        id<MTLBuffer> tailkbuf = ds4_gpu_tensor_buffer(tail_k);
        id<MTLBuffer> tailgatebuf = ds4_gpu_tensor_buffer(tail_gate);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_k);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        const uint64_t pool_cap =
            ((uint64_t)cache_cap + pool_size - 1u) / pool_size;
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t pool_bytes = pool_cap * head_dim * cache_elem_bytes;
        const uint64_t tail_bytes = (uint64_t)pool_size * head_dim * sizeof(float);
        const uint64_t input_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t ape_bytes =
            (uint64_t)pool_size * head_dim * sizeof(uint16_t);
        if (!poolbuf || !tailkbuf || !tailgatebuf || !rawbuf || !gatebuf ||
            ds4_gpu_tensor_bytes(pool_cache) < pool_bytes ||
            ds4_gpu_tensor_bytes(tail_k) < tail_bytes ||
            ds4_gpu_tensor_bytes(tail_gate) < tail_bytes ||
            ds4_gpu_tensor_bytes(raw_k) < input_bytes ||
            ds4_gpu_tensor_bytes(gate) < input_bytes) {
            fprintf(stderr, "ds4: Metal GLM-5.3 indexer pool received undersized buffers\n");
            return 0;
        }
        if (norm_weight_offset > model_size ||
            norm_bytes > model_size - norm_weight_offset ||
            norm_bias_offset > model_size ||
            norm_bytes > model_size - norm_bias_offset ||
            ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal GLM-5.3 indexer pool model range is invalid\n");
            return 0;
        }

        uint64_t norm_weight_inner = 0;
        uint64_t norm_bias_inner = 0;
        uint64_t ape_inner = 0;
        id<MTLBuffer> norm_weightbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, norm_weight_offset, norm_bytes, &norm_weight_inner);
        id<MTLBuffer> norm_biasbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, norm_bias_offset, norm_bytes, &norm_bias_inner);
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(
            model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!norm_weightbuf || !norm_biasbuf || !apebuf) return 0;

        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_glm53_indexer_pool_update_pipeline,
            "kernel_glm53_indexer_pool_update");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:norm_weightbuf offset:(NSUInteger)norm_weight_inner atIndex:3];
        [enc setBuffer:norm_biasbuf offset:(NSUInteger)norm_bias_inner atIndex:4];
        [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:5];
        [enc setBuffer:poolbuf offset:ds4_gpu_tensor_offset(pool_cache) atIndex:6];
        [enc setBuffer:tailkbuf offset:ds4_gpu_tensor_offset(tail_k) atIndex:7];
        [enc setBuffer:tailgatebuf offset:ds4_gpu_tensor_offset(tail_gate) atIndex:8];
        [enc setThreadgroupMemoryLength:
            ((NSUInteger)pool_size * head_dim + 2u * pool_size) * sizeof(float)
                                  atIndex:0];

        uint32_t done = 0;
        const uint32_t leading = pos0 % pool_size;
        if (leading != 0u) {
            uint32_t chunk = pool_size - leading;
            if (chunk > n_tokens) chunk = n_tokens;
            ds4_gpu_glm53_indexer_pool_update_args args = {
                .pos0 = pos0,
                .n_tokens = chunk,
                .cache_cap = cache_cap,
                .head_dim = head_dim,
                .pool_size = pool_size,
                .cache_f16 = cache_f16 ? 1u : 0u,
                .eps = eps,
                .pad0 = 0,
            };
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_k) atIndex:1];
            [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
            done = chunk;
        }
        if (done < n_tokens) {
            const uint32_t chunk = n_tokens - done;
            ds4_gpu_glm53_indexer_pool_update_args args = {
                .pos0 = pos0 + done,
                .n_tokens = chunk,
                .cache_cap = cache_cap,
                .head_dim = head_dim,
                .pool_size = pool_size,
                .cache_f16 = cache_f16 ? 1u : 0u,
                .eps = eps,
                .pad0 = 0,
            };
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:rawbuf
                    offset:ds4_gpu_tensor_offset(raw_k) +
                           (NSUInteger)((uint64_t)done * head_dim * sizeof(float))
                  atIndex:1];
            [enc setBuffer:gatebuf
                    offset:ds4_gpu_tensor_offset(gate) +
                           (NSUInteger)((uint64_t)done * head_dim * sizeof(float))
                  atIndex:2];
            const uint32_t groups = (chunk + pool_size - 1u) / pool_size;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 indexer pool update")) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_glm53_expand_pool_selection_tensor(
        ds4_gpu_tensor       *raw_selected,
        const ds4_gpu_tensor *pool_selected,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              selected_pools,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_selected || !pool_selected || n_tokens == 0 ||
        selected_pools == 0 || index_topk == 0 || pool_size == 0 ||
        selected_pools > index_topk / pool_size ||
        output_width < index_topk || output_width < index_topk + pool_size - 1u) {
        return 0;
    }
    @autoreleasepool {
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_selected);
        id<MTLBuffer> poolbuf = ds4_gpu_tensor_buffer(pool_selected);
        const uint64_t raw_bytes =
            (uint64_t)n_tokens * output_width * sizeof(uint32_t);
        const uint64_t pool_bytes =
            (uint64_t)n_tokens * selected_pools * sizeof(uint32_t);
        if (!rawbuf || !poolbuf ||
            ds4_gpu_tensor_bytes(raw_selected) < raw_bytes ||
            ds4_gpu_tensor_bytes(pool_selected) < pool_bytes) {
            fprintf(stderr, "ds4: Metal GLM-5.3 pool expansion received undersized buffers\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
            g_glm53_expand_pool_selection_pipeline,
            "kernel_glm53_expand_pool_selection");
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_glm53_expand_pool_selection_args args = {
            .n_tokens = n_tokens,
            .pos0 = pos0,
            .selected_pools = selected_pools,
            .index_topk = index_topk,
            .pool_size = pool_size,
            .output_width = output_width,
        };
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:poolbuf offset:ds4_gpu_tensor_offset(pool_selected) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_selected) atIndex:2];
        const NSUInteger total = (NSUInteger)n_tokens * output_width;
        const NSUInteger nth = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 pool selection expansion")) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        n_tokens == 0 || cache_cap == 0 || n_head == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        qk_dim < qk_nope || (qk_rope & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0 ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> knbuf = ds4_gpu_tensor_buffer(k_nope);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(value);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t cache_value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t kn_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        const uint64_t value_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        if (!keybuf || !valbuf || !rawbuf || !knbuf || !vbuf ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_value_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(k_nope) < kn_bytes ||
            ds4_gpu_tensor_bytes(value) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM KV cache builder received undersized buffers\n");
            return 0;
        }

        const bool decode_group4 =
            n_tokens == 1u &&
            n_head >= 4u &&
            ds4_gpu_env_bool("DS4_METAL_DISABLE_GLM_DECODE_KV_GROUP4") <= 0;
        id<MTLComputePipelineState> pipeline =
            decode_group4 ?
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_decode_group4_pipeline,
                                 "kernel_glm_build_kv_cache_decode_group4") :
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_pipeline,
                                 "kernel_glm_build_kv_cache");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_build_kv_cache_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0u,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:knbuf offset:ds4_gpu_tensor_offset(k_nope) atIndex:2];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(value) atIndex:3];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:4];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:5];
        const NSUInteger group_y = decode_group4 ?
            ((NSUInteger)n_head + 3u) / 4u :
            (NSUInteger)n_head;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, group_y, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM KV cache build")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        pos0 != 0 || n_tokens == 0 || cache_cap == 0 || n_head == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        qk_dim < qk_nope || (qk_rope & 1u) != 0 ||
        n_tokens > cache_cap ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> knbuf = ds4_gpu_tensor_buffer(k_nope);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(value);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t cache_value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t kn_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        const uint64_t value_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        if (!keybuf || !valbuf || !rawbuf || !knbuf || !vbuf ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_value_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(k_nope) < kn_bytes ||
            ds4_gpu_tensor_bytes(value) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM staged KV cache builder received undersized buffers\n");
            return 0;
        }

        const NSUInteger q_row_bytes_f16 = (NSUInteger)qk_dim * sizeof(uint16_t);
        const NSUInteger v_row_bytes_f16 = (NSUInteger)value_dim * sizeof(uint16_t);
        const NSUInteger key_f16_offset = 0;
        const NSUInteger key_f16_bytes =
            (NSUInteger)n_tokens * (NSUInteger)n_head * q_row_bytes_f16;
        const NSUInteger value_f16_offset = key_f16_bytes;
        const NSUInteger value_f16_bytes =
            (NSUInteger)n_tokens * (NSUInteger)n_head * v_row_bytes_f16;
        const NSUInteger kv_f16_bytes = key_f16_bytes + value_f16_bytes;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                           &g_flash_attn_kv_bytes,
                                           kv_f16_bytes,
                                           "ds4_glm_flash_attn_kv_f16")) {
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_flash_pipeline,
                                 "kernel_glm_build_kv_cache_flash");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_build_kv_cache_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0u,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:knbuf offset:ds4_gpu_tensor_offset(k_nope) atIndex:2];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(value) atIndex:3];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:4];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:5];
        [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:6];
        [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM staged KV cache build")) return 0;
    }

    return 1;
}

static int ds4_gpu_glm_attention_flash_tensor_impl(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16,
        int                   kv_pre_staged) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim != 256u || value_dim != 256u ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0 ||
        cache_len > ds4_gpu_glm_flash_attention_max_cache_len()) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        const uint64_t heads_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        if (!headsbuf || !qbuf || !keybuf || !valbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM FlashAttention received undersized buffers\n");
            return 0;
        }
        const uint64_t key_elems = (uint64_t)cache_len * n_head * qk_dim;
        const uint64_t value_elems = (uint64_t)cache_len * n_head * value_dim;
        if (key_elems > UINT32_MAX || value_elems > UINT32_MAX) {
            return 0;
        }

        const uint32_t nqptg = 8;
        const uint32_t ncpsg = 64;
        const uint32_t nsg = 4;
        const bool has_kvpad = (cache_len % ncpsg) != 0;
        const bool bc_mask = (n_tokens % nqptg) != 0;
        const NSUInteger q_row_bytes = (NSUInteger)qk_dim * sizeof(float);
        const NSUInteger q_row_bytes_f16 = (NSUInteger)qk_dim * sizeof(uint16_t);
        const NSUInteger v_row_bytes = (NSUInteger)value_dim * sizeof(float);
        const NSUInteger v_row_bytes_f16 = (NSUInteger)value_dim * sizeof(uint16_t);
        const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)cache_len * sizeof(uint16_t);
        const NSUInteger key_f16_offset = 0;
        const NSUInteger key_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head * q_row_bytes_f16;
        const NSUInteger value_f16_offset = key_f16_bytes;
        const NSUInteger value_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head * v_row_bytes_f16;
        const NSUInteger kv_f16_bytes = key_f16_bytes + value_f16_bytes;
        const NSUInteger pad_bytes = has_kvpad
            ? (NSUInteger)ncpsg * ((NSUInteger)n_head * (q_row_bytes_f16 + v_row_bytes_f16) +
                                   (NSUInteger)n_tokens * sizeof(uint16_t))
            : 1u;
        const NSUInteger nblk0 = ((NSUInteger)cache_len + ncpsg - 1u) / ncpsg;
        const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
        const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

        id<MTLBuffer> mask_buffer =
            ds4_gpu_glm_prefill_mask_buffer(pos0, n_tokens, cache_len, mask_bytes);
        if (!mask_buffer) return 0;
        if (kv_pre_staged) {
            if (!g_flash_attn_kv_buffer || g_flash_attn_kv_bytes < kv_f16_bytes) {
                fprintf(stderr, "ds4: GLM staged FlashAttention KV scratch is missing\n");
                return 0;
            }
        } else if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                                  &g_flash_attn_kv_bytes,
                                                  kv_f16_bytes,
                                                  "ds4_glm_flash_attn_kv_f16")) {
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                             &g_flash_attn_pad_bytes,
                                             pad_bytes,
                                             "ds4_glm_flash_attn_pad") ||
            !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                             &g_flash_attn_blk_bytes,
                                             blk_bytes,
                                             "ds4_glm_flash_attn_blk")) {
            return 0;
        }

        id<MTLComputePipelineState> pad_pipeline = nil;
        if (has_kvpad) {
            pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
            if (!pad_pipeline) return 0;
        }
        id<MTLComputePipelineState> blk_pipeline =
            ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
        id<MTLComputePipelineState> attn_pipeline =
            ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk256_dv256",
                                              true, false, false, false, has_kvpad, bc_mask,
                                              (int32_t)qk_dim,
                                              (int32_t)value_dim,
                                              (int32_t)nsg);
        if (!blk_pipeline || !attn_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!kv_pre_staged) {
            const bool copied = cache_f16 ?
                (ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16)) :
                (ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * q_row_bytes,
                                               q_row_bytes,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * v_row_bytes,
                                               v_row_bytes,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16));
            if (!copied) {
                return 0;
            }
        }

        if (has_kvpad) {
            ds4_gpu_flash_attn_pad_args pad_args = {
                .ne11 = (int32_t)cache_len,
                .ne_12_2 = (int32_t)n_head,
                .ne_12_3 = 1,
                .nb11 = q_row_bytes_f16,
                .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
                .nb13 = (uint64_t)cache_len * (uint64_t)n_head * q_row_bytes_f16,
                .nb21 = v_row_bytes_f16,
                .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
                .nb23 = (uint64_t)cache_len * (uint64_t)n_head * v_row_bytes_f16,
                .ne31 = (int32_t)n_tokens,
                .ne32 = 1,
                .ne33 = 1,
                .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
                .nb32 = mask_bytes,
                .nb33 = mask_bytes,
            };

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pad_pipeline];
            [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
            [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:1];
            [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:2];
            [enc setBuffer:mask_buffer offset:0 atIndex:3];
            [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        ds4_gpu_flash_attn_blk_args blk_args = {
            .ne01 = (int32_t)n_tokens,
            .ne30 = (int32_t)cache_len,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:mask_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        ds4_gpu_flash_attn_vec_args args = {
            .ne01 = (int32_t)n_tokens,
            .ne02 = (int32_t)n_head,
            .ne03 = 1,
            .nb01 = (uint64_t)n_head * q_row_bytes,
            .nb02 = q_row_bytes,
            .nb03 = (uint64_t)n_tokens * n_head * q_row_bytes,
            .ne11 = (int32_t)cache_len,
            .ne_12_2 = (int32_t)n_head,
            .ne_12_3 = 1,
            .ns10 = (int32_t)qk_dim,
            .nb11 = q_row_bytes_f16,
            .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
            .nb13 = (uint64_t)cache_len * (uint64_t)n_head * q_row_bytes_f16,
            .ns20 = (int32_t)value_dim,
            .nb21 = v_row_bytes_f16,
            .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
            .nb23 = (uint64_t)cache_len * (uint64_t)n_head * v_row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
            .ne1 = (int32_t)n_head,
            .ne2 = (int32_t)n_tokens,
            .ne3 = 1,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .max_bias = 0.0f,
            .m0 = 0.0f,
            .m1 = 0.0f,
            .n_head_log2 = 0,
            .logit_softcap = 0.0f,
        };

        const NSUInteger padded_v = ds4_gpu_align_up_ns(value_dim, 64u);
        const NSUInteger shared_elems = (NSUInteger)nqptg *
            ((NSUInteger)qk_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
        const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:2];
        [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:3];
        [enc setBuffer:mask_buffer offset:0 atIndex:4];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:5];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM FlashAttention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    return ds4_gpu_glm_attention_flash_tensor_impl(heads,
                                                   q,
                                                   key_cache,
                                                   value_cache,
                                                   pos0,
                                                   n_tokens,
                                                   cache_len,
                                                   cache_cap,
                                                   n_head,
                                                   qk_dim,
                                                   value_dim,
                                                   cache_f16,
                                                   0);
}

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    if (pos0 != 0 || n_tokens != cache_len) return 0;
    return ds4_gpu_glm_attention_flash_tensor_impl(heads,
                                                   q,
                                                   key_cache,
                                                   value_cache,
                                                   pos0,
                                                   n_tokens,
                                                   cache_len,
                                                   cache_cap,
                                                   n_head,
                                                   qk_dim,
                                                   value_dim,
                                                   cache_f16,
                                                   1);
}

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim == 0 || value_dim == 0 ||
        (qk_dim & 3u) != 0 ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0 ||
        cache_len > ds4_gpu_glm_full_attention_max_cache_len()) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        const uint64_t heads_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        if (!headsbuf || !qbuf || !keybuf || !valbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM attention received undersized buffers\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_attention_full_pipeline,
                                 "kernel_glm_attention_full");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint32_t full_attn_mode = 2u;
        ds4_gpu_glm_attention_full_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_len = cache_len,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .qk_dim = qk_dim,
            .value_dim = value_dim,
            .pad0 = full_attn_mode,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad1 = 0u,
            .pad2 = 0u,
            .scale = 1.0f / sqrtf((float)qk_dim),
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:4];
        [enc setThreadgroupMemoryLength:ds4_gpu_align_up_ns(
                (256u + (NSUInteger)cache_len) * sizeof(float), 16u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM full attention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || n_selected == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!selectedbuf || ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM selected range received undersized buffer\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_fill_selected_range_pipeline,
                                 "kernel_glm_fill_selected_range");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_fill_selected_range_args args = {
            .n_selected = n_selected,
        };
        const NSUInteger nth = 256u;
        const NSUInteger n_groups = ((NSUInteger)n_selected + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM selected range")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || n_tokens == 0 || n_selected == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t total = (uint64_t)n_tokens * n_selected;
        if (n_tokens != 0 && total / n_tokens != n_selected) return 0;
        if (total > UINT64_MAX / sizeof(uint32_t)) return 0;
        const uint64_t selected_bytes = total * sizeof(uint32_t);
        if (!selectedbuf || ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM selected range batch received undersized buffer\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_fill_selected_range_batch_pipeline,
                                 "kernel_glm_fill_selected_range_batch");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_fill_selected_range_batch_args args = {
            .n_tokens = n_tokens,
            .pos0 = pos0,
            .n_selected = n_selected,
            .pad_row = pad_row,
        };
        const NSUInteger nth = 256u;
        const NSUInteger n_groups = ((NSUInteger)total + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM selected range batch")) return 0;
    }

    return 1;
}







int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!qk_low || !q || !model_map ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t out_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        if (!outbuf || !qbuf ||
            ds4_gpu_tensor_bytes(qk_low) < out_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_glm52 =
            n_head == 64u &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            row_bytes == 204u &&
            weight_type == DS4_METAL_TENSOR_Q8_0;
        /* Coalesced simdgroup variant: lanes split the 192-dot so weight
         * reads coalesce, and 2048 threadgroups replace 64. The thread-
         * per-row kernels measured ~7.5x off the weight-bandwidth floor
         * (7.2ms of the decode token by skip-ablation). Covers the GLM 5.2
         * shape for both Q8_0 k_b and the DenseQ4 GGUF's Q4_0 k_b (the Q8
         * fast path above never engaged there). */
        const int use_glm52_sg =
            (n_head == 32u || n_head == 64u) &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            ((weight_type == DS4_METAL_TENSOR_Q8_0 && row_bytes == 204u) ||
             (weight_type == DS4_METAL_TENSOR_Q4_0 && row_bytes == 108u)) &&
            getenv("DS4_METAL_DISABLE_GLM_QKLOW_SG") == NULL &&
            ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_sg_pipeline,
                                 "kernel_glm_qk_lowrank_q8_0_glm52_sg") != nil;
        if (getenv("DS4_METAL_GLM_QKLOW_DEBUG")) {
            static int printed = 0;
            if (!printed) {
                printed = 1;
                fprintf(stderr, "ds4: qk_lowrank decode path: use_glm52=%d sg=%d n_head=%u kv=%u nope=%u dim=%u rb=%llu type=%u\n",
                        use_glm52, use_glm52_sg, n_head, kv_lora_dim, qk_nope, qk_dim,
                        (unsigned long long)row_bytes, weight_type);
            }
        }
        id<MTLComputePipelineState> pipeline =
            use_glm52_sg ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_sg_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_glm52_sg") :
            use_glm52 ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_glm52") :
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_qk_lowrank_args args = {
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_dim = qk_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .pad1 = 0,
            .pad2 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:3];
        if (use_glm52_sg) {
            /* 8 simdgroups x 2 rows per threadgroup: (64, 512/16) grid. */
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,
                                                  (NSUInteger)(kv_lora_dim / 16u),
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        } else {
            if (use_glm52) {
                [enc setThreadgroupMemoryLength:192u * sizeof(float) atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM qk lowrank")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_tensor(qk_low,
                                               q,
                                               model_map,
                                               model_size,
                                               weight_offset,
                                               DS4_METAL_TENSOR_Q8_0,
                                               n_head,
                                               kv_lora_dim,
                                               qk_nope,
                                               qk_dim);
}

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!qk_low || !q || !model_map ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t out_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        if (!outbuf || !qbuf ||
            ds4_gpu_tensor_bytes(qk_low) < out_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_glm52_t4 =
            n_tokens >= 4u &&
            n_head == 64u &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            row_bytes == 204u &&
            weight_type == DS4_METAL_TENSOR_Q8_0;
        id<MTLComputePipelineState> pipeline =
            use_glm52_t4 ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_batch_glm52_t4_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_batch_glm52_t4") :
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_batch_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_batch");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint32_t head_base = 0;
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &head_base, &head_count);
        ds4_gpu_glm_qk_lowrank_batch_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_dim = qk_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .head_base = head_base,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:3];
        if (use_glm52_t4) {
            [enc setThreadgroupMemoryLength:4u * 192u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  ((NSUInteger)n_tokens + 3u) / 4u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch qk lowrank")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_batch_tensor(qk_low,
                                                     q,
                                                     model_map,
                                                     model_size,
                                                     weight_offset,
                                                     DS4_METAL_TENSOR_Q8_0,
                                                     n_tokens,
                                                     n_head,
                                                     kv_lora_dim,
                                                     qk_nope,
                                                     qk_dim);
}

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !lora || !model_map ||
        n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || value_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, kv_lora_dim, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM batch value project received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t heads_bytes =
            (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        if (!headsbuf || !lorabuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(lora) < lora_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch value project received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM batch value project range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_mma =
            n_head == 64u &&
            kv_lora_dim == 512u &&
            value_dim == 256u &&
            row_bytes == 544u &&
            weight_type == DS4_METAL_TENSOR_Q8_0 &&
            n_tokens >= 32u;
        id<MTLComputePipelineState> pipeline =
            use_mma ?
                ds4_gpu_hot_pipeline(g_glm_value_project_q8_0_batch_heads_mma_pipeline,
                                     "kernel_glm_value_project_q8_0_batch_heads_mma") :
                ds4_gpu_hot_pipeline(g_glm_value_project_q8_0_batch_heads_pipeline,
                                     "kernel_glm_value_project_q8_0_batch_heads");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint32_t head_base = 0;
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &head_base, &head_count);
        ds4_gpu_glm_qk_lowrank_batch_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = 0,
            .qk_dim = value_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .head_base = head_base,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora) atIndex:2];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:3];
        if (use_mma) {
            [enc setThreadgroupMemoryLength:16u * 1024u atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tokens + 31u) / 32u,
                                                  ((NSUInteger)value_dim + 63u) / 64u,
                                                  (NSUInteger)head_count)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setThreadgroupMemoryLength:(NSUInteger)kv_lora_dim * sizeof(float)
                                    atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch value project")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    return ds4_gpu_glm_value_project_typed_batch_heads_tensor(heads,
                                                              lora,
                                                              model_map,
                                                              model_size,
                                                              weight_offset,
                                                              DS4_METAL_TENSOR_Q8_0,
                                                              n_tokens,
                                                              n_head,
                                                              kv_lora_dim,
                                                              value_dim);
}

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 || qk_dim < qk_nope ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM indexed attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes = (uint64_t)n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!headsbuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM indexed attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_decode_pipeline,
                                 "kernel_glm_attention_indexed_decode");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        ds4_gpu_glm_attention_indexed_decode_args args = {
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .value_type = value_weight_type,
        };
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:5];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:7];
        const NSUInteger scratch_floats =
            256u + (NSUInteger)n_selected + (NSUInteger)kv_lora_dim;
        [enc setThreadgroupMemoryLength:ds4_gpu_align_up_ns(
                scratch_floats * sizeof(float), 16u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed attention decode")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_decode_typed_tensor(heads,
                                                             q,
                                                             qk_low,
                                                             kv_lora_cache,
                                                             k_rope_cache,
                                                             model_map,
                                                             model_size,
                                                             value_weight_offset,
                                                             DS4_METAL_TENSOR_Q8_0,
                                                             selected,
                                                             n_selected,
                                                             cache_cap,
                                                             cache_f16,
                                                             n_head,
                                                             kv_lora_dim,
                                                             qk_nope,
                                                             qk_rope,
                                                             value_dim,
                                                             n_ctx_orig,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             beta_fast,
                                                             beta_slow);
}

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    if (!heads || !partial_lora || !partial_ms || !q || !qk_low ||
        !kv_lora_cache || !k_rope_cache || !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || (n_head % 8u) != 0 ||
        kv_lora_dim != 512u ||
        qk_nope == 0 || qk_rope != 64u ||
        value_dim == 0 || qk_dim < qk_nope ||
        block_rows == 0u || needed_blocks == 0u ||
        n_blocks < needed_blocks || n_blocks > 64u ||
        !cache_f16 ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> partial_lorabuf = ds4_gpu_tensor_buffer(partial_lora);
        id<MTLBuffer> partial_msbuf = ds4_gpu_tensor_buffer(partial_ms);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = sizeof(uint16_t);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes = (uint64_t)n_head * value_dim * sizeof(float);
        const uint64_t partial_lora_bytes =
            (uint64_t)n_blocks * n_head * kv_lora_dim * sizeof(float);
        const uint64_t partial_ms_bytes =
            (uint64_t)n_blocks * n_head * 2u * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!headsbuf || !partial_lorabuf || !partial_msbuf ||
            !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(partial_lora) < partial_lora_bytes ||
            ds4_gpu_tensor_bytes(partial_ms) < partial_ms_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        const bool use_valid_fullheads =
            selected_rows_valid && (n_head % 8u) == 0u;
        id<MTLComputePipelineState> partial_pipeline =
            ds4_gpu_hot_pipeline(use_valid_fullheads ?
                                     g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline :
                                     g_glm_attention_indexed_decode_split_group8_partial_pipeline,
                                 use_valid_fullheads ?
                                     "kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads" :
                                     "kernel_glm_attention_indexed_decode_split_group8_partial");
        const bool use_reduce16 =
            n_blocks == 16u && block_rows == 128u &&
            n_selected == 2048u && value_dim == 256u;
        id<MTLComputePipelineState> reduce_pipeline =
            ds4_gpu_hot_pipeline(use_reduce16 ?
                                     g_glm_attention_indexed_decode_split_group8_reduce16_pipeline :
                                     g_glm_attention_indexed_decode_split_group8_reduce_pipeline,
                                 use_reduce16 ?
                                     "kernel_glm_attention_indexed_decode_split_group8_reduce16" :
                                     "kernel_glm_attention_indexed_decode_split_group8_reduce");
        if (!partial_pipeline || !reduce_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_decode_split_args args = {
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = 1u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .block_rows = block_rows,
            .n_blocks = n_blocks,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .value_type = value_weight_type,
        };
        const NSUInteger stage_rows = 16u;
        const NSUInteger kv_vecs = (NSUInteger)kv_lora_dim / 4u;
        const NSUInteger rope_vecs = (NSUInteger)qk_rope / 4u;
        const NSUInteger partial_scratch_bytes =
            stage_rows * kv_vecs * sizeof(uint16_t) * 4u +
            stage_rows * rope_vecs * sizeof(float) * 4u;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:partial_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
        [enc setBuffer:partial_lorabuf offset:ds4_gpu_tensor_offset(partial_lora) atIndex:6];
        [enc setBuffer:partial_msbuf offset:ds4_gpu_tensor_offset(partial_ms) atIndex:7];
        [enc setThreadgroupMemoryLength:partial_scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head / 8u, (NSUInteger)n_blocks, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        const NSUInteger reduce_scratch_floats = 256u + 64u + (NSUInteger)kv_lora_dim;
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:reduce_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:partial_lorabuf offset:ds4_gpu_tensor_offset(partial_lora) atIndex:1];
        [enc setBuffer:partial_msbuf offset:ds4_gpu_tensor_offset(partial_ms) atIndex:2];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:3];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:4];
        [enc setThreadgroupMemoryLength:reduce_scratch_floats * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM split grouped indexed attention decode")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(heads,
                                                                          partial_lora,
                                                                          partial_ms,
                                                                          q,
                                                                          qk_low,
                                                                          kv_lora_cache,
                                                                          k_rope_cache,
                                                                          model_map,
                                                                          model_size,
                                                                          value_weight_offset,
                                                                          DS4_METAL_TENSOR_Q8_0,
                                                                          selected,
                                                                          n_selected,
                                                                          selected_rows_valid,
                                                                          cache_cap,
                                                                          cache_f16,
                                                                          n_head,
                                                                          kv_lora_dim,
                                                                          qk_nope,
                                                                          qk_rope,
                                                                          value_dim,
                                                                          n_ctx_orig,
                                                                          block_rows,
                                                                          n_blocks,
                                                                          freq_base,
                                                                          freq_scale,
                                                                          ext_factor,
                                                                          attn_factor,
                                                                          beta_fast,
                                                                          beta_slow);
}

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 || qk_dim < qk_nope ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes =
            (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes =
            (uint64_t)n_tokens * n_selected * sizeof(uint32_t);
        if (!headsbuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        const uint64_t grouped_attn = (n_tokens >= 128u) ? 8u : 2u;
        const NSUInteger q2_bit_words = ((NSUInteger)cache_cap + 31u) / 32u;
        const NSUInteger q2_scratch_bytes =
            q2_bit_words * sizeof(uint32_t) +
            4u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
            8u * (NSUInteger)kv_lora_dim * sizeof(float);
        const NSUInteger max_tg_mem = [g_device maxThreadgroupMemoryLength];
        const bool q2_fits = max_tg_mem == 0 || q2_scratch_bytes <= max_tg_mem;
        const bool use_q2_group4 = grouped_attn == 4u && n_tokens >= 2u && q2_fits;
        const bool use_group8 =
            grouped_attn == 8u ||
            (grouped_attn == 4u && !use_q2_group4);
        const bool use_group2 = grouped_attn == 1u || grouped_attn == 2u;
        id<MTLComputePipelineState> pipeline = use_q2_group4 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_q2_group4_pipeline,
                                 "kernel_glm_attention_indexed_batch_q2_group4") :
            (use_group8 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group8_pipeline,
                                 "kernel_glm_attention_indexed_batch_group8") :
            (use_group2 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group2_pipeline,
                                 "kernel_glm_attention_indexed_batch_group2") :
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_pipeline,
                                 "kernel_glm_attention_indexed_batch")));
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .value_type = value_weight_type,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        const NSUInteger scratch_bytes = use_q2_group4 ?
            q2_scratch_bytes :
            (use_group8 ?
             (8u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
              8u * (NSUInteger)kv_lora_dim * sizeof(float)) :
            ((use_group2 ?
             (512u + 2u * (NSUInteger)n_selected + 2u * (NSUInteger)kv_lora_dim) :
             (256u + (NSUInteger)n_selected + (NSUInteger)kv_lora_dim)) * sizeof(float)));

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:5];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:7];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        if (use_q2_group4) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 3u) / 4u,
                                                  ((NSUInteger)n_tokens + 1u) / 2u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else if (use_group8) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 7u) / 8u,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        } else if (use_group2) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 1u) / 2u,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)n_tokens, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed batch attention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_typed_tensor(heads,
                                                            q,
                                                            qk_low,
                                                            kv_lora_cache,
                                                            k_rope_cache,
                                                            model_map,
                                                            model_size,
                                                            value_weight_offset,
                                                            DS4_METAL_TENSOR_Q8_0,
                                                            selected,
                                                            n_tokens,
                                                            n_selected,
                                                            cache_cap,
                                                            cache_f16,
                                                            n_head,
                                                            kv_lora_dim,
                                                            qk_nope,
                                                            qk_rope,
                                                            value_dim,
                                                            n_ctx_orig,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            beta_fast,
                                                            beta_slow);
}


static int ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  selected_rows_valid) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache || !selected ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || (qk_rope & 1u) != 0 ||
        qk_dim < qk_nope ||
        (qk_rope != 0 && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0 && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora_out);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes =
            (uint64_t)n_tokens * n_selected * sizeof(uint32_t);
        if (!lorabuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(lora_out) < lora_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention-lora received undersized buffers\n");
            return 0;
        }

        const bool use_vec_lora =
            cache_f16 && kv_lora_dim == 512u &&
            (qk_rope == 0u || qk_rope == 64u);
        const bool full_head_groups = (n_head % 8u) == 0u;
        id<MTLComputePipelineState> pipeline = nil;
        if (use_vec_lora && selected_rows_valid && full_head_groups) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads");
        } else if (use_vec_lora && selected_rows_valid) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_valid");
        } else if (use_vec_lora) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec");
        } else {
            pipeline = ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group8_pipeline,
                                            "kernel_glm_attention_indexed_batch_group8");
        }
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = kv_lora_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = 0,
            .value_type = 1u,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &args.head_base, &head_count);
        const NSUInteger scratch_bytes = use_vec_lora ?
            (16u * ((NSUInteger)kv_lora_dim / 4u) * sizeof(uint16_t) * 4u +
             16u * ((NSUInteger)qk_rope / 4u) * sizeof(float) * 4u) :
            (8u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
             8u * (NSUInteger)kv_lora_dim * sizeof(float));

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        if (use_vec_lora) {
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:6];
        } else {
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:5];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:7];
        }
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)head_count + 7u) / 8u,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed batch attention-lora")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        pos0 > n_selected || n_tokens > n_selected - pos0 ||
        n_head == 0 || kv_lora_dim != 512u ||
        qk_nope == 0 || (qk_rope != 0u && qk_rope != 64u) ||
        qk_dim < qk_nope ||
        (qk_rope != 0u && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0u && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora_out);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        const uint64_t cache_elem_bytes =
            cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        if (!lorabuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf ||
            ds4_gpu_tensor_bytes(lora_out) < lora_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes) {
            fprintf(stderr, "ds4: Metal GLM causal batch attention-lora received undersized buffers\n");
            return 0;
        }

        const bool full_head_groups = (n_head % 8u) == 0u;
        id<MTLComputePipelineState> pipeline = full_head_groups ?
            ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads") :
            ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_causal");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = kv_lora_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = 0,
            .value_type = 1u,
            .pos0 = pos0,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &args.head_base, &head_count);
        const NSUInteger scratch_bytes =
            16u * ((NSUInteger)kv_lora_dim / 4u) * sizeof(uint16_t) * 4u +
            16u * ((NSUInteger)qk_rope / 4u) * sizeof(float) * 4u;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:5];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)head_count + 7u) / 8u,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM causal indexed batch attention-lora")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(lora_out,
                                                                 q,
                                                                 qk_low,
                                                                 kv_lora_cache,
                                                                 k_rope_cache,
                                                                 selected,
                                                                 n_tokens,
                                                                 n_selected,
                                                                 cache_cap,
                                                                 cache_f16,
                                                                 n_head,
                                                                 kv_lora_dim,
                                                                 qk_nope,
                                                                 qk_rope,
                                                                 n_ctx_orig,
                                                                 freq_base,
                                                                 freq_scale,
                                                                 ext_factor,
                                                                 attn_factor,
                                                                 beta_fast,
                                                                 beta_slow,
                                                                 false);
}

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(lora_out,
                                                                 q,
                                                                 qk_low,
                                                                 kv_lora_cache,
                                                                 k_rope_cache,
                                                                 selected,
                                                                 n_tokens,
                                                                 n_selected,
                                                                 cache_cap,
                                                                 cache_f16,
                                                                 n_head,
                                                                 kv_lora_dim,
                                                                 qk_nope,
                                                                 qk_rope,
                                                                 n_ctx_orig,
                                                                 freq_base,
                                                                 freq_scale,
                                                                 ext_factor,
                                                                 attn_factor,
                                                                 beta_fast,
                                                                 beta_slow,
                                                                 true);
}

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert > 512u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal GLM router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM router bias range is outside the mapped model\n");
            return 0;
        }
        const bool exact_bias_view =
            bias_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_DISABLE_DECODE_ROUTER_BIAS_EXACT_VIEWS") == NULL;
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = exact_bias_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           bias_offset,
                                           bias_bytes,
                                           &bias_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     bias_offset,
                                     bias_bytes,
                                     &bias_inner);
        if (!biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_router_select_one_pipeline,
                                 "kernel_glm_router_select_one");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:5];
        const NSUInteger router_threads = n_expert > 256u ? 512u : 256u;
        [enc setThreadgroupMemoryLength:router_threads *
             (sizeof(float) + sizeof(int32_t)) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(router_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM router select")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 ||
        n_expert == 0 || n_expert > 512u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(float);
        const uint64_t probs_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < logits_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes ||
            ds4_gpu_tensor_bytes(probs) < probs_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM batch router bias range is outside the mapped model\n");
            return 0;
        }
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         bias_offset, bias_bytes,
                                                         &bias_inner);
        if (!biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_router_select_one_pipeline,
                                 "kernel_glm_router_select_one");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:5];
        const NSUInteger router_threads = n_expert > 256u ? 512u : 256u;
        [enc setThreadgroupMemoryLength:router_threads *
             (sizeof(float) + sizeof(int32_t)) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(router_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch router select")) return 0;
    }

    return 1;
}

static bool ds4_gpu_glm_gate_pair_type_supported(
        uint32_t gate_type,
        uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == DS4_METAL_TENSOR_Q2_K ||
            gate_type == DS4_METAL_TENSOR_Q4_K ||
            gate_type == DS4_METAL_TENSOR_Q5_K);
}

static bool ds4_gpu_glm_down_type_supported(uint32_t down_type) {
    return down_type == DS4_METAL_TENSOR_Q2_K ||
           down_type == DS4_METAL_TENSOR_Q4_K ||
           down_type == DS4_METAL_TENSOR_Q5_K ||
           down_type == DS4_METAL_TENSOR_Q6_K;
}

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    /* TP sharding: only the owned contiguous expert range is mapped,
     * so bind from the owned base, validate only its bytes, and tell the
     * kernels the first expert id present at that base. */
    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    /* Streaming address tables point at individually cached experts by their
     * global id. Keep the source offsets global; ownership is enforced by the
     * kernels before they dereference an address-table slot. */
    const bool streaming_tp =
        g_ssd_streaming_mode && g_tp_split_world == 2;
    if (streaming_tp) {
        first_expert = 0;
        n_bind_expert = n_total_expert;
    }
    const int32_t tp_expert_base_host = (int32_t)first_expert;
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;

    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed MoE received inconsistent expert strides\n");
        return 0;
    }
    ds4_gpu_stream_expert_cache_note_token(layer_index);
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal GLM routed MoE tensor range is outside the mapped model\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal GLM routed MoE received undersized activation buffers\n");
            return 0;
        }

        const BOOL gate_pair_q2 = gate_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q5 = gate_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL down_scalar_q4 = down_type == DS4_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == DS4_METAL_TENSOR_Q6_K;
        const BOOL down_simd = down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL stream_addr_q2 =
            gate_pair_q2 && down_scalar_q2 &&
            g_glm_q2_k_addr_pair_swiglu2_f32_pipeline != nil &&
            g_glm_q2_k_addr_down_f32_pipeline != nil;
        const BOOL stream_addr_q4 =
            !gate_pair_q2 && !gate_pair_q5 && down_scalar_q4 &&
            g_glm_q4_k_addr_pair_swiglu_f32_pipeline != nil &&
            g_glm_q4_k_addr_down_f32_pipeline != nil;
        BOOL use_stream_expert_addr_table =
            g_ssd_streaming_mode &&
            !force_resident &&
            getenv("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE") == NULL &&
            (stream_addr_q2 || stream_addr_q4) &&
            layer_index < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
            n_total_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            n_expert <= 8u &&
            ds4_gpu_stream_expert_cache_configured_budget() >= n_expert &&
            ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                         down_expert_bytes) &&
            ds4_gpu_stream_expert_cache_effective_cap(layer_index,
                                                      n_total_expert,
                                                      n_expert) != 0;
        int32_t stream_selected_ids[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        ds4_gpu_stream_expert_cache_entry *stream_entries[8] = {
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
        };
        uint64_t stream_gate_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t stream_up_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t stream_down_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t stream_missing_mask = 0;
        uint32_t stream_entry_count = 0;
        uint32_t stream_resident_mask = 0;
        BOOL use_stream_split_deferred = false;
        id<MTLBuffer> stream_gate_addr_buf = nil;
        id<MTLBuffer> stream_up_addr_buf = nil;
        id<MTLBuffer> stream_down_addr_buf = nil;
        const int glm_stream_timing =
            ds4_gpu_stream_expert_timing_summary_enabled();

        if (use_stream_expert_addr_table) {
            const int had_batch = g_batch_cb != nil;
            int stream_ok = 1;
            const int have_prefetched_selected =
                ds4_gpu_glm_stream_selected_prefetch_take(model_map,
                                                          model_size,
                                                          layer_index,
                                                          n_total_expert,
                                                          n_expert,
                                                          gate_offset,
                                                          up_offset,
                                                          down_offset,
                                                          gate_expert_bytes,
                                                          down_expert_bytes,
                                                          stream_selected_ids);
            if (have_prefetched_selected) {
                if (had_batch && g_batch_has_work &&
                    g_stream_expert_pending_load.active &&
                    ds4_gpu_flush_commands() == 0) {
                    stream_ok = 0;
                }
            } else {
                if (had_batch && ds4_gpu_end_commands() == 0) {
                    stream_ok = 0;
                }
                if (stream_ok &&
                    ds4_gpu_tensor_read(selected,
                                        0,
                                        stream_selected_ids,
                                        (uint64_t)n_expert * sizeof(stream_selected_ids[0])) == 0) {
                    stream_ok = 0;
                }
            }
            for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                if (stream_selected_ids[i] < 0 ||
                    (uint32_t)stream_selected_ids[i] >= n_total_expert) {
                    fprintf(stderr,
                            "ds4: Metal GLM routed MoE selected expert id %d is outside 0..%u\n",
                            stream_selected_ids[i],
                            n_total_expert);
                    stream_ok = 0;
                }
            }
            if (stream_ok) {
                ds4_gpu_stream_expert_cache_note_selected_hotness(layer_index,
                                                                  stream_selected_ids,
                                                                  n_expert);
                if (!ds4_gpu_moe_selected_hotlist_record(layer_index,
                                                         stream_selected_ids,
                                                         n_expert,
                                                         n_total_expert)) {
                    stream_ok = 0;
                }
            }
            if (stream_ok) {
                g_glm_stream_expert_addr_table_building++;
                for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                    const uint64_t expert_id = (uint64_t)(uint32_t)stream_selected_ids[i];
                    if (expert_id > UINT64_MAX / gate_expert_bytes ||
                        expert_id > UINT64_MAX / down_expert_bytes) {
                        fprintf(stderr, "ds4: Metal GLM routed MoE selected expert offset overflow\n");
                        stream_ok = 0;
                        break;
                    }
                    const uint64_t gate_rel = expert_id * gate_expert_bytes;
                    const uint64_t down_rel = expert_id * down_expert_bytes;
                    if (gate_rel > UINT64_MAX - gate_offset ||
                        gate_rel > UINT64_MAX - up_offset ||
                        down_rel > UINT64_MAX - down_offset) {
                        fprintf(stderr, "ds4: Metal GLM routed MoE selected expert offset overflow\n");
                        stream_ok = 0;
                        break;
                    }
                    stream_gate_abs_offsets[i] = gate_offset + gate_rel;
                    stream_up_abs_offsets[i] = up_offset + gate_rel;
                    stream_down_abs_offsets[i] = down_offset + down_rel;
                    stream_entries[i] =
                        ds4_gpu_stream_expert_cache_peek(model_map,
                                                         model_size,
                                                         layer_index,
                                                         (uint32_t)stream_selected_ids[i],
                                                         n_total_expert,
                                                         n_expert,
                                                         stream_gate_abs_offsets[i],
                                                         stream_up_abs_offsets[i],
                                                         stream_down_abs_offsets[i],
                                                         gate_expert_bytes,
                                                         down_expert_bytes);
                    if (!stream_entries[i]) {
                        stream_missing_mask |= 1u << i;
                    } else {
                        stream_resident_mask |= 1u << i;
                    }
                }
                if (stream_ok && glm_stream_timing) {
                    ds4_gpu_stream_expert_timing_note_cache_class(
                            stream_resident_mask,
                            stream_missing_mask);
                }
                use_stream_split_deferred =
                    stream_ok &&
                    getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_SPLIT") == NULL &&
                    stream_missing_mask != 0 &&
                    stream_resident_mask != 0 &&
                    ds4_gpu_stream_expert_split_worthwhile(stream_resident_mask,
                                                           stream_missing_mask) &&
                    ds4_gpu_stream_expert_split_ready() &&
                    ((gate_pair_q2 &&
                      g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline != nil) ||
                     (!gate_pair_q2 && !gate_pair_q5 &&
                      g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline != nil));
                if (use_stream_split_deferred) {
                    const ds4_gpu_stream_expert_table table = {
                        .model_map = model_map,
                        .model_size = model_size,
                        .layer = layer_index,
                        .n_total_expert = n_total_expert,
                        .gate_offset = gate_offset,
                        .up_offset = up_offset,
                        .down_offset = down_offset,
                        .gate_expert_bytes = gate_expert_bytes,
                        .down_expert_bytes = down_expert_bytes,
                    };
                    if (!ds4_gpu_stream_expert_cache_begin_selected_load(
                                &table,
                                stream_selected_ids,
                                n_expert)) {
                        stream_ok = 0;
                    }
                }
                if (stream_ok && stream_missing_mask != 0 &&
                    !use_stream_split_deferred &&
                    !ds4_gpu_stream_expert_cache_load_selected_missing(
                            model_map,
                            model_size,
                            layer_index,
                            stream_selected_ids,
                            n_total_expert,
                            n_expert,
                            stream_gate_abs_offsets,
                            stream_up_abs_offsets,
                            stream_down_abs_offsets,
                            gate_expert_bytes,
                            down_expert_bytes,
                            stream_missing_mask,
                            stream_entries)) {
                    fprintf(stderr,
                            "ds4: Metal GLM streaming expert cache failed to load "
                            "layer=%u missing=0x%x budget=%u\n",
                            layer_index,
                            stream_missing_mask,
                            ds4_gpu_stream_expert_cache_configured_budget());
                    stream_ok = 0;
                }
                for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                    ds4_gpu_stream_expert_cache_entry *entry = stream_entries[i];
                    if (!entry) {
                        if (use_stream_split_deferred &&
                            (stream_missing_mask & (1u << i)) != 0) {
                            continue;
                        }
                        stream_ok = 0;
                        break;
                    }
                    if (!ds4_gpu_stream_expert_cache_set_addr_slot(
                                layer_index,
                                (uint32_t)stream_selected_ids[i],
                                entry->gate_buffer,
                                entry->gate_inner,
                                entry->up_buffer,
                                entry->up_inner,
                                entry->down_buffer,
                                entry->down_inner)) {
                        stream_ok = 0;
                        break;
                    }
                    stream_entry_count++;
                }
                if (stream_ok &&
                    !ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                              &stream_gate_addr_buf,
                                                              &stream_up_addr_buf,
                                                              &stream_down_addr_buf)) {
                    stream_ok = 0;
                }
                g_glm_stream_expert_addr_table_building--;
            }
            if (!have_prefetched_selected && had_batch &&
                ds4_gpu_begin_commands() == 0) {
                stream_ok = 0;
            }
            if (!stream_ok) return 0;
            ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                    n_total_expert,
                                                    n_expert,
                                                    stream_selected_ids,
                                                    n_expert);
            ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                     stream_selected_ids,
                                                     n_expert);
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;
        if (!use_stream_expert_addr_table) {
            gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               gate_offset, gate_tensor_bytes,
                                               &gate_inner);
            upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                             up_offset, up_tensor_bytes,
                                             &up_inner);
            downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               down_offset, down_tensor_bytes,
                                               &down_inner);
            if (!gatebuf || !upbuf || !downbuf) return 0;
        }

        id<MTLComputePipelineState> pair_pipeline =
            (use_stream_split_deferred ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32_masked") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32_masked")) :
             use_stream_expert_addr_table ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32")) :
             gate_pair_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q2_K_pair_swiglu_f32") :
             gate_pair_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q5_K_pair_swiglu_f32") :
             ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                  "kernel_glm_q4_K_pair_swiglu2_f32"));
        id<MTLComputePipelineState> down_pipeline =
            (use_stream_expert_addr_table ?
             (down_scalar_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_down_f32_pipeline,
                                   "kernel_glm_q2_K_addr_down_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_down_f32_pipeline,
                                   "kernel_glm_q4_K_addr_down_f32")) :
             down_scalar_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_down_f32_pipeline,
                                  "kernel_glm_q2_K_down_f32") :
             down_scalar_q4 ?
             ds4_gpu_hot_pipeline(g_glm_q4_k_down_f32_pipeline,
                                  "kernel_glm_q4_K_down_f32") :
             down_simd_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                  "kernel_glm_q5_K_down_f32") :
             ds4_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                  "kernel_glm_q6_K_down_f32"));
        if (!pair_pipeline || !down_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile =
            g_batch_cb != nil &&
            ds4_gpu_stage_profile_enabled_for_layer(
                    "DS4_METAL_GLM_MOE_ONE_STAGE_PROFILE",
                    "DS4_METAL_GLM_MOE_ONE_STAGE_PROFILE_LAYER",
                    layer_index);
        const char *glm_moe_stage_filter =
            getenv("DS4_METAL_GLM_MOE_STAGE_PROFILE_FILTER");
        const char *glm_pair_path =
            use_stream_expert_addr_table ?
            (gate_pair_q2 ? "q2_stream_addr_swiglu" :
                            "q4_stream_addr_swiglu") :
            gate_pair_q2 ? "q2_scalar_swiglu" :
            gate_pair_q5 ? "q5_pair_simd_swiglu" : "q4_pair2_simd_swiglu";
        const char *glm_down_path =
            use_stream_expert_addr_table ?
            (down_scalar_q2 ? "q2_stream_addr_down" :
                              "q4_stream_addr_down_simd") :
            down_scalar_q2 ? "q2_down_simd" :
            down_scalar_q4 ? "q4_down_simd" :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }
        int ok = 1;
#define DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM routed MoE one stage layer=%u tokens=1 experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ds4_gpu_glm_routed_moe_args args = {
            .tp_rank = g_tp_split_rank,
            .tp_world = g_tp_split_world,
            .tp_expert_base = tp_expert_base_host,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = down_type,
            .swiglu_clamp = swiglu_clamp,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (use_stream_expert_addr_table ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 7u) / 8u)) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            use_stream_expert_addr_table ? (NSUInteger)((expert_mid_dim + 3u) / 4u) :
            (NSUInteger)((expert_mid_dim + 1u) / 2u);
        const NSUInteger pair_threadgroup_bytes = 0u;
        const NSUInteger pair_threads = 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (NSUInteger)((out_dim + 7u) / 8u) :
            down_simd_q4 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            (down_scalar_q2 || down_simd) ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            (down_scalar_q2 || down_simd) ? 64u : 256u;
        if (use_stream_expert_addr_table &&
            !ds4_gpu_stream_expert_cache_mark_entries_inflight(
                    stream_entries,
                    use_stream_split_deferred ? n_expert : stream_entry_count,
                    use_stream_split_deferred ? stream_resident_mask : 0)) {
            return 0;
        }

        const int glm_stream_split_timing =
            use_stream_split_deferred && glm_stream_timing;
        double glm_stream_split_t0 =
            glm_stream_split_timing ? ds4_gpu_now_ms() : 0.0;
        double glm_stream_split_resident_ms = 0.0;
        double glm_stream_split_missing_load_ms = 0.0;
        double glm_stream_split_missing_wait_ms = 0.0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        if (use_stream_split_deferred) {
            [enc setBytes:&stream_resident_mask length:sizeof(stream_resident_mask) atIndex:1];
            [enc setBuffer:stream_gate_addr_buf offset:0u atIndex:2];
            [enc setBuffer:stream_up_addr_buf offset:0u atIndex:3];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:6];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:7];
        } else {
            [enc setBuffer:use_stream_expert_addr_table ? stream_gate_addr_buf : gatebuf
                    offset:use_stream_expert_addr_table ? 0u : (NSUInteger)gate_inner
                   atIndex:1];
            [enc setBuffer:use_stream_expert_addr_table ? stream_up_addr_buf : upbuf
                    offset:use_stream_expert_addr_table ? 0u : (NSUInteger)up_inner
                   atIndex:2];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        }
        if (use_stream_expert_addr_table) {
            const uint32_t use_count =
                use_stream_split_deferred ? n_expert : stream_entry_count;
            const uint32_t use_mask =
                use_stream_split_deferred ? stream_resident_mask : 0;
            for (uint32_t i = 0; i < use_count; i++) {
                if (use_mask != 0 && (use_mask & (1u << i)) == 0) continue;
                [enc useResource:stream_entries[i]->gate_buffer usage:MTLResourceUsageRead];
                [enc useResource:stream_entries[i]->up_buffer usage:MTLResourceUsageRead];
            }
        }
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE("pair");

        if (ok && use_stream_split_deferred) {
            id<MTLCommandBuffer> resident_cb = nil;
            const int resident_owned = owned;
            if (owned) {
                resident_cb = cb;
                [resident_cb commit];
                cb = nil;
            } else {
                ok = ds4_gpu_flush_commands();
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_resident_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
            }
            if (ok) {
                ok = ds4_gpu_stream_expert_pending_load_finish(stream_entries);
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_missing_load_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
            }
            if (ok) {
                ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                        n_total_expert,
                                                        n_expert,
                                                        stream_selected_ids,
                                                        n_expert);
                ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                         stream_selected_ids,
                                                         n_expert);
                ok = ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                              &stream_gate_addr_buf,
                                                              &stream_up_addr_buf,
                                                              &stream_down_addr_buf);
            }
            if (ok) {
                if (resident_owned) {
                    ok = ds4_gpu_wait_command_buffer(
                            resident_cb,
                            "GLM streaming split resident pair");
                    ds4_gpu_stream_expert_cache_note_owned_completed();
                } else {
                    ok = ds4_gpu_wait_pending_command_buffers(
                            "GLM streaming split resident pair");
                }
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_missing_wait_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
                ds4_gpu_stream_expert_timing_note_split(
                        stream_resident_mask,
                        stream_missing_mask,
                        glm_stream_split_resident_ms,
                        glm_stream_split_missing_load_ms +
                        glm_stream_split_missing_wait_ms);
                ds4_gpu_stream_expert_timing_note_split_missing_detail(
                        glm_stream_split_missing_load_ms,
                        0.0,
                        0.0,
                        0.0,
                        glm_stream_split_missing_wait_ms);
            }
            if (ok &&
                !ds4_gpu_stream_expert_cache_mark_entries_inflight(stream_entries,
                                                                   n_expert,
                                                                   stream_missing_mask)) {
                ok = 0;
            }
            if (ok) {
                cb = ds4_gpu_command_buffer(&owned);
                if (!cb) ok = 0;
            }
            if (ok) {
                enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pair_pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBytes:&stream_missing_mask length:sizeof(stream_missing_mask) atIndex:1];
                [enc setBuffer:stream_gate_addr_buf offset:0u atIndex:2];
                [enc setBuffer:stream_up_addr_buf offset:0u atIndex:3];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
                [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
                [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:6];
                [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:7];
                for (uint32_t i = 0; i < n_expert; i++) {
                    if ((stream_missing_mask & (1u << i)) == 0) continue;
                    [enc useResource:stream_entries[i]->gate_buffer usage:MTLResourceUsageRead];
                    [enc useResource:stream_entries[i]->up_buffer usage:MTLResourceUsageRead];
                }
                if (pair_threadgroup_bytes != 0u) {
                    [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
                }
                [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                                      (NSUInteger)n_expert,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);
            }
        }

        if (!ok) return 0;

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_down_addr_buf : downbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)down_inner
               atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        if (use_stream_expert_addr_table) {
            const uint32_t use_count =
                use_stream_split_deferred ? n_expert : stream_entry_count;
            for (uint32_t i = 0; i < use_count; i++) {
                [enc useResource:stream_entries[i]->down_buffer usage:MTLResourceUsageRead];
            }
        }
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE("down");

        if (!ok) return 0;
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM routed MoE")) return 0;
#undef DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE
    }

    return 1;
}

static bool ds4_gpu_glm_grouped_moe_fast_default(void) {
    return !g_quality_mode;
}

static bool ds4_gpu_glm_routed_moe_batch_grouped_available(
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint32_t n_expert,
        uint32_t n_tokens) {
    if (n_tokens < 32u ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type) ||
        ds4_gpu_mul_mm_id_map0_name(n_expert) == NULL) {
        return false;
    }
    const bool fast_default = ds4_gpu_glm_grouped_moe_fast_default();
    if (!fast_default) {
        return false;
    }
    if (n_tokens < 96u) return false;

    return ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert)) != nil &&
           ds4_gpu_routed_mm_pipeline(gate_type) != nil &&
           ds4_gpu_routed_mm_pipeline(up_type) != nil &&
           ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) != nil;
}

static bool ds4_gpu_glm_grouped_moe_layer_enabled(uint32_t layer_index) {
    (void)layer_index;
    return true;
}

static int ds4_gpu_glm_routed_moe_batch_grouped_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens) {
    if (!ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                         up_type,
                                                         down_type,
                                                         n_expert,
                                                         n_tokens)) {
        return 0;
    }
    if (n_expert > UINT32_MAX / n_tokens ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        return 0;
    }

    const uint32_t pair_rows = n_tokens * n_expert;
    if ((uint64_t)pair_rows > UINT64_MAX / expert_mid_dim ||
        (uint64_t)pair_rows > UINT64_MAX / out_dim ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        return 0;
    }

    const bool mid_f16 = true;
    const NSUInteger mm_id_threadgroup_bytes = 8192u;
    const uint64_t compact_mid_values = (uint64_t)pair_rows * expert_mid_dim;
    const uint64_t down_values = (uint64_t)pair_rows * out_dim;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (compact_mid_values > UINT64_MAX / sizeof(float) ||
        compact_mid_values > UINT64_MAX / (mid_f16 ? sizeof(uint16_t) : sizeof(float)) ||
        down_values > UINT64_MAX / sizeof(float) ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t gate_scratch_bytes = compact_mid_values * sizeof(float);
    const uint64_t mid_bytes = compact_mid_values * (mid_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t down_scratch_bytes = down_values * sizeof(float);
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_bytes = selected_values * sizeof(int32_t);
    const uint64_t weights_bytes = selected_values * sizeof(float);
    if (gate_scratch_bytes > UINT64_MAX - gate_scratch_bytes ||
        gate_scratch_bytes > NSUIntegerMax ||
        gate_scratch_bytes * 2ull > NSUIntegerMax ||
        down_scratch_bytes > NSUIntegerMax) {
        return 0;
    }

    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM grouped routed MoE received undersized activation buffers\n");
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_moe_gate_scratch_buffer,
                                           &g_moe_gate_scratch_bytes,
                                           (NSUInteger)(gate_scratch_bytes * 2ull),
                                           "ds4_glm_moe_gate_up_scratch")) {
            return 0;
        }
        if (n_expert > 1 &&
            !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                           &g_moe_down_scratch_bytes,
                                           (NSUInteger)down_scratch_bytes,
                                           "ds4_glm_moe_down_scratch")) {
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         gate_offset, gate_tensor_bytes,
                                                         &gate_inner);
        id<MTLBuffer> upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       up_offset, up_tensor_bytes,
                                                       &up_inner);
        id<MTLBuffer> downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         down_offset, down_tensor_bytes,
                                                         &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        id<MTLComputePipelineState> map_pipeline =
            ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
        id<MTLComputePipelineState> gate_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
        id<MTLComputePipelineState> up_pipeline = ds4_gpu_routed_mm_pipeline(up_type);
        id<MTLComputePipelineState> down_pipeline =
            ds4_gpu_routed_mm_f16_rhs_pipeline(down_type);
        if (!map_pipeline || !gate_pipeline || !up_pipeline || !down_pipeline) {
            return 0;
        }

        ds4_gpu_mul_mm_id_map_args map_args =
            ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args gate_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args up_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          up_row_bytes, up_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args down_args =
            ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                    down_row_bytes, down_expert_bytes,
                                                    n_expert, n_expert, n_tokens,
                                                    mid_f16 ? sizeof(uint16_t) : sizeof(float));
        gate_args.tp_rank = g_tp_split_rank;
        gate_args.tp_world = g_tp_split_world;
        gate_args.tp_expert_base = (int32_t)first_expert;
        up_args.tp_rank = g_tp_split_rank;
        up_args.tp_world = g_tp_split_world;
        up_args.tp_expert_base = (int32_t)first_expert;
        down_args.tp_rank = g_tp_split_rank;
        down_args.tp_world = g_tp_split_world;
        down_args.tp_expert_base = (int32_t)first_expert;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }

        int ok = 1;
#define DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM grouped routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                           map_pipeline,
                                           &map_args,
                                           &gate_args,
                                           selectedbuf,
                                           ds4_gpu_tensor_offset(selected));
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("map");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       gate_pipeline,
                                                       &gate_args,
                                                       gatebuf,
                                                       (NSUInteger)gate_inner,
                                                       xbuf,
                                                       ds4_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       0,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("gate");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       up_pipeline,
                                                       &up_args,
                                                       upbuf,
                                                       (NSUInteger)up_inner,
                                                       xbuf,
                                                       ds4_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       (NSUInteger)gate_scratch_bytes,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("up");
        if (ok) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                   g_moe_gate_scratch_buffer,
                                                   0,
                                                   g_moe_gate_scratch_buffer,
                                                   (NSUInteger)gate_scratch_bytes,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   ds4_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   pair_rows,
                                                   swiglu_clamp,
                                                   mid_f16);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : g_moe_down_scratch_buffer;
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) : 0;
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_pipeline,
                                                       &down_args,
                                                       downbuf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("down");
        if (ok && n_expert > 1) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                 down_dst,
                                                 down_dst_off,
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 out_dim,
                                                 n_expert,
                                                 n_tokens);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM grouped routed batch MoE")) {
            return 0;
        }
#undef DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE
    }

    return 1;
}

static int ds4_gpu_glm_routed_moe_batch_grouped_addr_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        id<MTLBuffer>          gate_addrs,
        id<MTLBuffer>          up_addrs,
        id<MTLBuffer>          down_addrs,
        id<MTLBuffer>          overflow_gate,
        id<MTLBuffer>          overflow_up,
        id<MTLBuffer>          overflow_down,
        uint32_t               gate_type,
        uint32_t               up_type,
        uint32_t               down_type,
        uint64_t               gate_expert_bytes,
        uint64_t               gate_row_bytes,
        uint64_t               up_expert_bytes,
        uint64_t               up_row_bytes,
        uint64_t               down_expert_bytes,
        uint64_t               down_row_bytes,
        uint32_t               expert_in_dim,
        uint32_t               expert_mid_dim,
        uint32_t               out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t               n_total_expert,
        uint32_t               n_expert,
        float                  swiglu_clamp,
        uint32_t               layer_index,
        const ds4_gpu_tensor *x,
        uint32_t               n_tokens,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t               resource_count) {
    if (n_expert > UINT32_MAX / n_tokens ||
        !gate_addrs || !up_addrs || !down_addrs ||
        !ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                        up_type,
                                                        down_type,
                                                        n_expert,
                                                        n_tokens)) {
        return 0;
    }

    const uint32_t pair_rows = n_tokens * n_expert;
    if ((uint64_t)pair_rows > UINT64_MAX / expert_mid_dim ||
        (uint64_t)pair_rows > UINT64_MAX / out_dim ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        return 0;
    }

    const bool mid_f16 = true;
    const NSUInteger mm_id_threadgroup_bytes = 8192u;
    const uint64_t compact_mid_values = (uint64_t)pair_rows * expert_mid_dim;
    const uint64_t down_values = (uint64_t)pair_rows * out_dim;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (compact_mid_values > UINT64_MAX / sizeof(float) ||
        compact_mid_values > UINT64_MAX / sizeof(uint16_t) ||
        down_values > UINT64_MAX / sizeof(float) ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t gate_scratch_bytes = compact_mid_values * sizeof(float);
    const uint64_t mid_bytes = compact_mid_values * sizeof(uint16_t);
    const uint64_t down_scratch_bytes = down_values * sizeof(float);
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_bytes = selected_values * sizeof(int32_t);
    const uint64_t weights_bytes = selected_values * sizeof(float);
    if (gate_scratch_bytes > UINT64_MAX - gate_scratch_bytes ||
        gate_scratch_bytes > NSUIntegerMax ||
        gate_scratch_bytes * 2ull > NSUIntegerMax ||
        down_scratch_bytes > NSUIntegerMax) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM grouped-address routed MoE received undersized activation buffers\n");
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_moe_gate_scratch_buffer,
                                           &g_moe_gate_scratch_bytes,
                                           (NSUInteger)(gate_scratch_bytes * 2ull),
                                           "ds4_glm_moe_gate_up_scratch")) {
            return 0;
        }
        if (n_expert > 1 &&
            !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                           &g_moe_down_scratch_bytes,
                                           (NSUInteger)down_scratch_bytes,
                                           "ds4_glm_moe_down_scratch")) {
            return 0;
        }

        id<MTLComputePipelineState> map_pipeline =
            ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
        id<MTLComputePipelineState> gate_pipeline =
            ds4_gpu_routed_mm_addr_pipeline(gate_type);
        id<MTLComputePipelineState> up_pipeline =
            ds4_gpu_routed_mm_addr_pipeline(up_type);
        id<MTLComputePipelineState> down_pipeline =
            ds4_gpu_routed_mm_addr_f16_rhs_pipeline(down_type);
        if (!map_pipeline || !gate_pipeline || !up_pipeline || !down_pipeline) {
            return 0;
        }

        ds4_gpu_mul_mm_id_map_args map_args =
            ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args gate_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args up_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          up_row_bytes, up_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args down_args =
            ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                    down_row_bytes, down_expert_bytes,
                                                    n_expert, n_expert, n_tokens,
                                                    sizeof(uint16_t));
        if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(resources,
                                                               resource_count,
                                                               0)) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }

        int ok = 1;
#define DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM grouped-address routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s mid=f16 %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                           map_pipeline,
                                           &map_args,
                                           &gate_args,
                                           selectedbuf,
                                           ds4_gpu_tensor_offset(selected));
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("map");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           gate_pipeline,
                                                           &gate_args,
                                                           gate_addrs,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           g_moe_gate_scratch_buffer,
                                                           0,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           0,
                                                           overflow_gate);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("gate");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           up_pipeline,
                                                           &up_args,
                                                           up_addrs,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           g_moe_gate_scratch_buffer,
                                                           (NSUInteger)gate_scratch_bytes,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           1,
                                                           overflow_up);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("up");
        if (ok) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                   g_moe_gate_scratch_buffer,
                                                   0,
                                                   g_moe_gate_scratch_buffer,
                                                   (NSUInteger)gate_scratch_bytes,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   ds4_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   pair_rows,
                                                   swiglu_clamp,
                                                   mid_f16);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : g_moe_down_scratch_buffer;
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) : 0;
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           down_pipeline,
                                                           &down_args,
                                                           down_addrs,
                                                           midbuf,
                                                           ds4_gpu_tensor_offset(mid),
                                                           down_dst,
                                                           down_dst_off,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           2,
                                                           overflow_down);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("down");
        if (ok && n_expert > 1) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                 down_dst,
                                                 down_dst_off,
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 out_dim,
                                                 n_expert,
                                                 n_tokens);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                           "GLM grouped-address routed batch MoE")) {
            return 0;
        }
#undef DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE
    }

    return 1;
}

static int ds4_gpu_glm_routed_moe_batch_tensor_impl(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    allow_grouped,
        bool                    force_scalar_q4_pair) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    const uint64_t per_token_mid = (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t full_gate_tensor_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_up_tensor_bytes =
        (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t full_down_tensor_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE received inconsistent expert strides\n");
        return 0;
    }
    if (gate_offset > model_size || full_gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || full_up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || full_down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE tensor range is outside the mapped model\n");
        return 0;
    }

    if (allow_grouped &&
        (!g_ssd_streaming_mode ||
         ds4_gpu_glm_streaming_prefill_full_layer_active()) &&
        ds4_gpu_glm_grouped_moe_layer_enabled(layer_index) &&
        ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                       up_type,
                                                       down_type,
                                                       n_expert,
                                                       n_tokens)) {
        return ds4_gpu_glm_routed_moe_batch_grouped_tensor(out,
                                                           mid,
                                                           model_map,
                                                           model_size,
                                                           gate_offset,
                                                           up_offset,
                                                           down_offset,
                                                           gate_type,
                                                           up_type,
                                                           down_type,
                                                           gate_expert_bytes,
                                                           gate_row_bytes,
                                                           up_expert_bytes,
                                                           up_row_bytes,
                                                           down_expert_bytes,
                                                           down_row_bytes,
                                                           expert_in_dim,
                                                           expert_mid_dim,
                                                           out_dim,
                                                           selected,
                                                           weights,
                                                           n_total_expert,
                                                           n_expert,
                                                           swiglu_clamp,
                                                           layer_index,
                                                           x,
                                                           n_tokens);
    }

    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    /* Streaming address tables are indexed by global expert id. As in the
     * one-token path, keep model offsets global and let the kernels enforce
     * rank ownership before dereferencing the selected address. */
    const bool streaming_tp =
        g_ssd_streaming_mode && g_tp_split_world == 2;
    if (streaming_tp) {
        first_expert = 0;
        n_bind_expert = n_total_expert;
    }
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
        const uint64_t out_values = (uint64_t)n_tokens * out_dim;
        const uint64_t mid_values =
            (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
        const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
        if (x_values > UINT64_MAX / sizeof(float) ||
            out_values > UINT64_MAX / sizeof(float) ||
            mid_values > UINT64_MAX / sizeof(float) ||
            selected_values > UINT64_MAX / sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal GLM routed batch MoE activation byte size overflow\n");
            return 0;
        }
        const uint64_t x_bytes = x_values * sizeof(float);
        const uint64_t mid_bytes = mid_values * sizeof(float);
        const uint64_t out_bytes = out_values * sizeof(float);
        const uint64_t selected_bytes = selected_values * sizeof(int32_t);
        const uint64_t weights_bytes = selected_values * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM routed batch MoE received undersized activation buffers\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;
        id<MTLBuffer> stream_gate_addr_buf = nil;
        id<MTLBuffer> stream_up_addr_buf = nil;
        id<MTLBuffer> stream_down_addr_buf = nil;
        id<MTLBuffer> stream_overflow_gate = nil;
        id<MTLBuffer> stream_overflow_up = nil;
        id<MTLBuffer> stream_overflow_down = nil;
        ds4_gpu_stream_expert_cache_entry
            *stream_resources[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = { NULL };
        uint32_t stream_resource_count = 0;
        uint32_t stream_unique = 0;

        const BOOL gate_pair_q2 = gate_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q5 = gate_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL down_scalar_q4 = down_type == DS4_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == DS4_METAL_TENSOR_Q6_K;
        const BOOL down_simd = down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL stream_addr_q2 =
            gate_pair_q2 && down_scalar_q2 &&
            g_glm_q2_k_addr_pair_swiglu2_f32_pipeline != nil &&
            g_glm_q2_k_addr_down_f32_pipeline != nil;
        const BOOL stream_addr_q4 =
            !gate_pair_q2 && !gate_pair_q5 && down_scalar_q4 &&
            g_glm_q4_k_addr_pair_swiglu_f32_pipeline != nil &&
            g_glm_q4_k_addr_down_f32_pipeline != nil;
        BOOL use_stream_expert_addr_table =
            g_ssd_streaming_mode &&
            !ds4_gpu_glm_streaming_prefill_full_layer_active() &&
            n_tokens > 1 &&
            (stream_addr_q2 || stream_addr_q4) &&
            layer_index < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
            n_total_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            n_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED &&
            ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                         down_expert_bytes) &&
            ds4_gpu_stream_expert_cache_effective_cap(layer_index,
                                                      n_total_expert,
                                                      n_expert) != 0;
        const BOOL use_stream_grouped_addr_table =
            use_stream_expert_addr_table &&
            allow_grouped &&
            getenv("DS4_METAL_GLM_DISABLE_STREAMING_GROUPED_ADDR_PREFILL") == NULL &&
            ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                           up_type,
                                                           down_type,
                                                           n_expert,
                                                           n_tokens);
        const BOOL enable_q4_pair4 = true;
        const BOOL q4_scalar_pair = false;
        const BOOL q4_pair2 =
            !use_stream_expert_addr_table &&
            !gate_pair_q5 && !q4_scalar_pair &&
            (force_scalar_q4_pair || !enable_q4_pair4);
        id<MTLComputePipelineState> pair_pipeline =
            use_stream_expert_addr_table ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32")) :
            gate_pair_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q2_K_pair_swiglu_f32") :
            gate_pair_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q5_K_pair_swiglu_f32") :
            (q4_scalar_pair ?
             ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q4_K_pair_swiglu_f32") :
             q4_pair2 ?
              ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu4_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu4_f32"));
        id<MTLComputePipelineState> down_pipeline =
            use_stream_expert_addr_table ?
             (down_scalar_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_down_f32_pipeline,
                                   "kernel_glm_q2_K_addr_down_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_down_f32_pipeline,
                                   "kernel_glm_q4_K_addr_down_f32")) :
            down_scalar_q2 ?
            ds4_gpu_hot_pipeline(g_glm_q2_k_down_f32_pipeline,
                                 "kernel_glm_q2_K_down_f32") :
            down_scalar_q4 ?
            ds4_gpu_hot_pipeline(g_glm_q4_k_down_f32_pipeline,
                                 "kernel_glm_q4_K_down_f32") :
            down_simd_q5 ?
            ds4_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                 "kernel_glm_q5_K_down_f32") :
            ds4_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                 "kernel_glm_q6_K_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        if (use_stream_expert_addr_table) {
            const int had_batch = g_batch_cb != nil;
            if (had_batch && ds4_gpu_end_commands() == 0) {
                return 0;
            }
            if (!ds4_gpu_stream_expert_cache_prepare_selected_batch(
                        model_map,
                        model_size,
                        layer_index,
                        selected,
                        n_tokens,
                        n_total_expert,
                        n_expert,
                        gate_offset,
                        up_offset,
                        down_offset,
                        gate_expert_bytes,
                        down_expert_bytes,
                        &stream_gate_addr_buf,
                        &stream_up_addr_buf,
                        &stream_down_addr_buf,
                        stream_resources,
                        &stream_resource_count,
                        &stream_unique,
                        &stream_overflow_gate,
                        &stream_overflow_up,
                        &stream_overflow_down)) {
                return 0;
            }
            if (stream_unique == 0) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                return 0;
            }
            if (had_batch && ds4_gpu_begin_commands() == 0) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                return 0;
            }
            if (use_stream_grouped_addr_table) {
                return ds4_gpu_glm_routed_moe_batch_grouped_addr_tensor(
                        out,
                        mid,
                        stream_gate_addr_buf,
                        stream_up_addr_buf,
                        stream_down_addr_buf,
                        stream_overflow_gate,
                        stream_overflow_up,
                        stream_overflow_down,
                        gate_type,
                        up_type,
                        down_type,
                        gate_expert_bytes,
                        gate_row_bytes,
                        up_expert_bytes,
                        up_row_bytes,
                        down_expert_bytes,
                        down_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        out_dim,
                        selected,
                        weights,
                        n_total_expert,
                        n_expert,
                        swiglu_clamp,
                        layer_index,
                        x,
                        n_tokens,
                        stream_resources,
                        stream_resource_count);
            }
        } else {
            gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               gate_offset, gate_tensor_bytes,
                                               &gate_inner);
            upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                             up_offset, up_tensor_bytes,
                                             &up_inner);
            downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               down_offset, down_tensor_bytes,
                                               &down_inner);
            if (!gatebuf || !upbuf || !downbuf) return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        const char *glm_pair_path = use_stream_expert_addr_table ?
                                    (gate_pair_q2 ? "q2_stream_addr_swiglu" :
                                                    "q4_stream_addr_swiglu") :
                                    gate_pair_q2 ? "q2_scalar_swiglu" :
                                    gate_pair_q5 ? "q5_pair_simd_swiglu" :
                                    (q4_scalar_pair ? "q4_scalar_swiglu" :
                                    (q4_pair2 ? "q4_pair2_simd_swiglu" :
                                                "q4_pair4_simd_swiglu"));
        const char *glm_down_path =
            use_stream_expert_addr_table ?
            (down_scalar_q2 ? "q2_stream_addr_down" : "q4_stream_addr_down_simd") :
            down_scalar_q2 ? "q2_down_scalar" :
            down_scalar_q4 ? "q4_down_simd" :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }
        int ok = 1;
#define DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM routed MoE batch stage layer=%u tokens=%u experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ds4_gpu_glm_routed_moe_args args = {
            .tp_rank = g_tp_split_rank,
            .tp_world = g_tp_split_world,
            .tp_expert_base = (int32_t)first_expert,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = n_tokens,
            .mid_token_stride = mid_token_stride,
            .down_type = down_type,
            .swiglu_clamp = swiglu_clamp,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (use_stream_expert_addr_table ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 7u) / 8u)) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            use_stream_expert_addr_table ? (NSUInteger)((expert_mid_dim + 3u) / 4u) :
            q4_scalar_pair ? (NSUInteger)expert_mid_dim :
            q4_pair2 ? (NSUInteger)((expert_mid_dim + 1u) / 2u) :
            (NSUInteger)((expert_mid_dim + 7u) / 8u);
        const NSUInteger pair_threadgroup_bytes =
            q4_scalar_pair ? 512u * sizeof(float) : 0u;
        const NSUInteger pair_threads =
            q4_scalar_pair ? 256u : 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (NSUInteger)((out_dim + 7u) / 8u) :
            down_simd_q4 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            down_simd ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            down_simd ? 64u : 256u;
        if (use_stream_expert_addr_table &&
            !ds4_gpu_stream_expert_cache_mark_entries_inflight(stream_resources,
                                                               stream_resource_count,
                                                               0)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_gate_addr_buf : gatebuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)gate_inner
               atIndex:1];
        [enc setBuffer:use_stream_expert_addr_table ? stream_up_addr_buf : upbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)up_inner
               atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        if (use_stream_expert_addr_table) {
            for (uint32_t i = 0; i < stream_resource_count; i++) {
                [enc useResource:stream_resources[i]->gate_buffer usage:MTLResourceUsageRead];
                [enc useResource:stream_resources[i]->up_buffer usage:MTLResourceUsageRead];
            }
            if (stream_overflow_gate) [enc useResource:stream_overflow_gate usage:MTLResourceUsageRead];
            if (stream_overflow_up) [enc useResource:stream_overflow_up usage:MTLResourceUsageRead];
        }
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              (NSUInteger)n_tokens)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE("pair");

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_down_addr_buf : downbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)down_inner
               atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        if (use_stream_expert_addr_table) {
            for (uint32_t i = 0; i < stream_resource_count; i++) {
                [enc useResource:stream_resources[i]->down_buffer usage:MTLResourceUsageRead];
            }
            if (stream_overflow_down) [enc useResource:stream_overflow_down usage:MTLResourceUsageRead];
        }
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE("down");

        if (!ok) return 0;
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM routed batch MoE")) return 0;
#undef DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE
    }

    return 1;
}

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident) {
    (void)force_resident;
    return ds4_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    swiglu_clamp,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    true,
                                                    false);
}

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    return ds4_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    swiglu_clamp,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    false,
                                                    false);
}














/* Release decode fused HC pre-sublayer operation.  The graph driver owns the
 * optional reference fallback so this function stays a direct fused dispatch. */

/* HC-pre plus the immediately following weighted RMSNorm, specialized for
 * DS4's HC=4 shape. Both decode and batched prefill use this implementation;
 * the kernel preserves their established single-row and batched scale formulas. */

/* ds4_gpu_hc_norm_mix_args lives once in hc_control.h. */


/* Fused decode HC-pre: unweighted RMSNorm over the flattened HC row plus the
 * F16 HC-mix matvec in one dispatch.  The kernel reproduces the standalone
 * kernels' reduction trees bit-exactly (see src/metal/dsv4_hc.metal), so this is
 * purely a dispatch-count optimization. */






/* Expand of the SUM of two block vectors — the TP attention combine folded
 * into the HC expand (the kernel's has_add path adds them element-wise
 * before the post/comb mixing, canonical rank order preserved by argument
 * position). */









/* Kimi Delta Attention primitive shared with the GLM-5.3 graph. */
static int glm53_gpu_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int glm53_gpu_tensor_has(
        const ds4_gpu_tensor *tensor,
        uint64_t              elements,
        uint64_t              element_size) {
    uint64_t bytes = 0;
    return tensor &&
           glm53_gpu_mul_u64(elements, element_size, &bytes) &&
           ds4_gpu_tensor_buffer(tensor) != nil &&
           ds4_gpu_tensor_bytes(tensor) >= bytes;
}

static id<MTLBuffer> glm53_gpu_weight_buffer(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    bytes,
        uint64_t   *inner_offset,
        const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) {
        fprintf(stderr, "ds4: GLM-5.3 %s range is outside the mapped model\n",
                label ? label : "weight");
        return nil;
    }
    return ds4_gpu_wrap_model_range(model_map, model_size, offset, bytes,
                                    inner_offset);
}

typedef ds4_metal_embedding_bf16_args glm53_gpu_bf16_matmul_args;


static uint32_t glm53_gpu_bf16_mv_nsg(void) {
    return ds4_gpu_device_name_contains("M3 Ultra") &&
           getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE") == NULL &&
           getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_BF16_NSG4") == NULL
               ? 4u : 8u;
}


int ds4_gpu_glm53_matmul_bf16_qkv(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!ds4_gpu_device_name_contains("M3 Ultra")) return 0;
    uint64_t weights = 0;
    if (in_dim == 0 || out_dim == 0 ||
        !glm53_gpu_mul_u64(in_dim, out_dim, &weights) ||
        !glm53_gpu_tensor_has(x, in_dim, sizeof(float)) ||
        !glm53_gpu_tensor_has(out_q, out_dim, sizeof(float)) ||
        !glm53_gpu_tensor_has(out_k, out_dim, sizeof(float)) ||
        !glm53_gpu_tensor_has(out_v, out_dim, sizeof(float))) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t weight_bytes = weights * sizeof(uint16_t);
        uint64_t inner_q = 0, inner_k = 0, inner_v = 0;
        id<MTLBuffer> weight_q = glm53_gpu_weight_buffer(
            model_map, model_size, weight_q_offset, weight_bytes,
            &inner_q, "BF16 Q matrix");
        id<MTLBuffer> weight_k = glm53_gpu_weight_buffer(
            model_map, model_size, weight_k_offset, weight_bytes,
            &inner_k, "BF16 K matrix");
        id<MTLBuffer> weight_v = glm53_gpu_weight_buffer(
            model_map, model_size, weight_v_offset, weight_bytes,
            &inner_v, "BF16 V matrix");
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_mul_mv_bf16_f32_qkv");
        if (!weight_q || !weight_k || !weight_v || !pipeline) return 0;

        const uint32_t nsg = glm53_gpu_bf16_mv_nsg();
        glm53_gpu_bf16_matmul_args args = {
            .in_dim = in_dim,
            .out_dim = out_dim,
            .n_rows = 1u,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weight_q offset:(NSUInteger)inner_q atIndex:1];
        [enc setBuffer:weight_k offset:(NSUInteger)inner_k atIndex:2];
        [enc setBuffer:weight_v offset:(NSUInteger)inner_v atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(x)
                offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:ds4_gpu_tensor_buffer(out_q)
                offset:ds4_gpu_tensor_offset(out_q) atIndex:5];
        [enc setBuffer:ds4_gpu_tensor_buffer(out_k)
                offset:ds4_gpu_tensor_offset(out_k) atIndex:6];
        [enc setBuffer:ds4_gpu_tensor_buffer(out_v)
                offset:ds4_gpu_tensor_offset(out_v) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((out_dim + nsg - 1u) / nsg,
                                              1u, 3u)
            threadsPerThreadgroup:MTLSizeMake(32u * nsg, 1u, 1u)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned,
                                              "GLM-5.3 BF16 QKV matmul");
    }
}

typedef struct {
    uint32_t width;
    uint32_t rows;
    float eps;
} glm53_vision_rows_args;

typedef struct {
    uint32_t rows;
    uint32_t grid_h;
    uint32_t grid_w;
    float eps;
} glm53_vision_qkv_args;

typedef struct {
    uint32_t rows;
    float scale;
} glm53_vision_attention_args;

typedef struct {
    uint32_t dst_row;
    uint32_t image_row;
    uint32_t rows;
    uint32_t total_rows;
    uint32_t width;
    uint32_t hc;
} glm53_vision_scatter_args;

static int glm53_vision_dispatch_rows(
        const char           *kernel,
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              width,
        uint32_t              rows,
        float                 eps,
        const char           *label) {
    uint64_t inner = 0;
    id<MTLBuffer> weight = glm53_gpu_weight_buffer(
            model_map, model_size, weight_offset,
            (uint64_t)width * sizeof(uint16_t), &inner, label);
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(kernel);
    if (!weight || !pipeline) return 0;
    glm53_vision_rows_args args = { width, rows, eps };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc setBuffer:weight offset:(NSUInteger)inner atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, label);
}

static int glm53_vision_dispatch_bias(
        const char           *kernel,
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *residual,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        uint32_t              width,
        uint32_t              rows,
        const char           *label) {
    uint64_t inner = 0;
    id<MTLBuffer> bias = glm53_gpu_weight_buffer(
            model_map, model_size, bias_offset,
            (uint64_t)width * sizeof(uint16_t), &inner, label);
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(kernel);
    if (!bias || !pipeline) return 0;
    glm53_vision_rows_args args = { width, rows, 0.0f };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc setBuffer:bias offset:(NSUInteger)inner atIndex:2];
    if (residual) {
        [enc setBuffer:ds4_gpu_tensor_buffer(residual)
                offset:ds4_gpu_tensor_offset(residual) atIndex:3];
    }
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)width, 256u), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, label);
}

static int glm53_vision_dispatch_qkv(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        const ds4_gpu_tensor *qkv,
        const void           *model_map,
        uint64_t              model_size,
        const ds4_glm53_vision_layer_weights *w,
        uint32_t              rows,
        uint32_t              grid_h,
        uint32_t              grid_w) {
    uint64_t bias_inner = 0, q_inner = 0, k_inner = 0;
    id<MTLBuffer> bias = glm53_gpu_weight_buffer(
            model_map, model_size, w->qkv_bias, 3072u * sizeof(uint16_t),
            &bias_inner, "vision QKV bias");
    id<MTLBuffer> qw = glm53_gpu_weight_buffer(
            model_map, model_size, w->q_norm, 64u * sizeof(uint16_t),
            &q_inner, "vision Q norm");
    id<MTLBuffer> kw = glm53_gpu_weight_buffer(
            model_map, model_size, w->k_norm, 64u * sizeof(uint16_t),
            &k_inner, "vision K norm");
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_qkv_rope");
    if (!bias || !qw || !kw || !pipeline) return 0;
    glm53_vision_qkv_args args = { rows, grid_h, grid_w, 1.0e-5f };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(qkv)
            offset:ds4_gpu_tensor_offset(qkv) atIndex:1];
    [enc setBuffer:bias offset:(NSUInteger)bias_inner atIndex:2];
    [enc setBuffer:qw offset:(NSUInteger)q_inner atIndex:3];
    [enc setBuffer:kw offset:(NSUInteger)k_inner atIndex:4];
    [enc setBuffer:ds4_gpu_tensor_buffer(q)
            offset:ds4_gpu_tensor_offset(q) atIndex:5];
    [enc setBuffer:ds4_gpu_tensor_buffer(k)
            offset:ds4_gpu_tensor_offset(k) atIndex:6];
    [enc setBuffer:ds4_gpu_tensor_buffer(v)
            offset:ds4_gpu_tensor_offset(v) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 16u, 1)
         threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision QKV");
}

static int glm53_vision_dispatch_attention(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              rows) {
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_attention");
    if (!pipeline) return 0;
    glm53_vision_attention_args args = { rows, 0.125f };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(q)
            offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(k)
            offset:ds4_gpu_tensor_offset(k) atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(v)
            offset:ds4_gpu_tensor_offset(v) atIndex:3];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 16u, 1)
         threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision attention");
}

static int glm53_vision_dispatch_swiglu_bias(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_bias_offset,
        uint64_t              up_bias_offset,
        uint32_t              width,
        uint32_t              rows) {
    uint64_t gate_inner = 0, up_inner = 0;
    id<MTLBuffer> gate_bias = glm53_gpu_weight_buffer(
            model_map, model_size, gate_bias_offset,
            (uint64_t)width * sizeof(uint16_t), &gate_inner,
            "vision gate bias");
    id<MTLBuffer> up_bias = glm53_gpu_weight_buffer(
            model_map, model_size, up_bias_offset,
            (uint64_t)width * sizeof(uint16_t), &up_inner,
            "vision up bias");
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_swiglu_bias");
    if (!gate_bias || !up_bias || !pipeline) return 0;
    glm53_vision_rows_args args = { width, rows, 0.0f };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(gate)
            offset:ds4_gpu_tensor_offset(gate) atIndex:1];
    [enc setBuffer:gate_bias offset:(NSUInteger)gate_inner atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(up)
            offset:ds4_gpu_tensor_offset(up) atIndex:3];
    [enc setBuffer:up_bias offset:(NSUInteger)up_inner atIndex:4];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)width, 256u), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision SwiGLU");
}

static int glm53_vision_dispatch_reorder(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t              rows) {
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_downsample_reorder");
    if (!pipeline) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(4096u, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision downsample reorder");
}

static int glm53_vision_dispatch_layernorm_gelu(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              rows) {
    uint64_t weight_inner = 0, bias_inner = 0;
    id<MTLBuffer> weight = glm53_gpu_weight_buffer(
            model_map, model_size, weight_offset, 4096u * sizeof(uint16_t),
            &weight_inner, "vision merger norm");
    id<MTLBuffer> bias = glm53_gpu_weight_buffer(
            model_map, model_size, bias_offset, 4096u * sizeof(uint16_t),
            &bias_inner, "vision merger norm bias");
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_layernorm_gelu");
    if (!weight || !bias || !pipeline) return 0;
    glm53_vision_rows_args args = { 4096u, rows, 1.0e-5f };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc setBuffer:weight offset:(NSUInteger)weight_inner atIndex:2];
    [enc setBuffer:bias offset:(NSUInteger)bias_inner atIndex:3];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:4];
    [enc setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision merger norm");
}

/* Vision graph allocation and the complete transformer/merger flow live in
 * engine/glm53/vision.zig. These functions are actual encoder leaves. */
#include "../engine/glm53/vision_api.h"

int ds4_native_glm53_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
        ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, const void *map,
        uint64_t size, const ds4_glm53_vision_layer_weights *weights,
        uint32_t rows, uint32_t grid_height, uint32_t grid_width) {
    @autoreleasepool {
        return glm53_vision_dispatch_qkv(q, k, v, qkv, map, size, weights,
                                         rows, grid_height, grid_width);
    }
}
int ds4_native_glm53_vision_swiglu_bias(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        const void *map, uint64_t size, uint64_t gate_bias, uint64_t up_bias,
        uint32_t width, uint32_t rows) {
    @autoreleasepool {
        return glm53_vision_dispatch_swiglu_bias(out, gate, up, map, size,
                                                gate_bias, up_bias, width, rows);
    }
}
int ds4_native_glm53_vision_reorder(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, uint32_t rows) {
    @autoreleasepool {
        return glm53_vision_dispatch_reorder(out, input, rows);
    }
}
int ds4_native_glm53_vision_merger_norm(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *input, const void *map, uint64_t size,
        uint64_t weight, uint64_t bias, uint32_t rows) {
    @autoreleasepool {
        return glm53_vision_dispatch_layernorm_gelu(out, input, map, size,
                                                   weight, bias, rows);
    }
}

typedef struct {
    uint32_t width;
    uint32_t rows;
} deepseek4_vision_rows_args;

typedef struct {
    uint32_t rows;
    uint32_t grid_width;
} deepseek4_vision_qkv_args;

typedef struct {
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t output_rows;
} deepseek4_vision_align_args;

static int deepseek4_vision_dispatch_qkv(
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        const ds4_gpu_tensor *qkv,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        uint32_t              rows,
        uint32_t              grid_width) {
    uint64_t bias_inner = 0;
    id<MTLBuffer> bias = glm53_gpu_weight_buffer(
            model_map, model_size, bias_offset, 3072u * sizeof(uint16_t),
            &bias_inner, "DeepSeek vision QKV bias");
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_deepseek4_vision_qkv_rope");
    if (!bias || !pipeline) return 0;
    deepseek4_vision_qkv_args args = { rows, grid_width };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(qkv)
            offset:ds4_gpu_tensor_offset(qkv) atIndex:1];
    [enc setBuffer:bias offset:(NSUInteger)bias_inner atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(q)
            offset:ds4_gpu_tensor_offset(q) atIndex:3];
    [enc setBuffer:ds4_gpu_tensor_buffer(k)
            offset:ds4_gpu_tensor_offset(k) atIndex:4];
    [enc setBuffer:ds4_gpu_tensor_buffer(v)
            offset:ds4_gpu_tensor_offset(v) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 16u, 1)
         threadsPerThreadgroup:MTLSizeMake(32u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned,
                                          "DeepSeek vision QKV");
}

static int deepseek4_vision_dispatch_two_rows(
        const char           *kernel,
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t              width,
        uint32_t              rows,
        const char           *label) {
    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(kernel);
    if (!pipeline) return 0;
    deepseek4_vision_rows_args args = { width, rows };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)width, 256u), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, label);
}

static int deepseek4_vision_dispatch_round(
        ds4_gpu_tensor *x,
        uint32_t width,
        uint32_t rows) {
    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline("kernel_deepseek4_vision_round_bf16");
    if (!pipeline) return 0;
    deepseek4_vision_rows_args args = { width, rows };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)width, 256u), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned,
                                          "DeepSeek vision BF16 round");
}

static int deepseek4_vision_dispatch_aligner_reorder(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t              grid_height,
        uint32_t              grid_width,
        uint32_t              output_rows) {
    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline("kernel_deepseek4_vision_aligner_reorder");
    if (!pipeline) return 0;
    deepseek4_vision_align_args args = {
        grid_height, grid_width, output_rows
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(9216u, output_rows, 1)
         threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned,
                                          "DeepSeek vision aligner reorder");
}

static int deepseek4_vision_dispatch_gelu_bias(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        uint32_t              rows,
        uint32_t              width) {
    uint64_t bias_inner = 0;
    id<MTLBuffer> bias = glm53_gpu_weight_buffer(
            model_map, model_size, bias_offset, (uint64_t)width * sizeof(uint16_t),
            &bias_inner, "DeepSeek vision aligner bias");
    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline("kernel_deepseek4_vision_gelu_bias");
    if (!bias || !pipeline) return 0;
    deepseek4_vision_rows_args args = { width, rows };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(x)
            offset:ds4_gpu_tensor_offset(x) atIndex:1];
    [enc setBuffer:bias offset:(NSUInteger)bias_inner atIndex:2];
    [enc setBuffer:ds4_gpu_tensor_buffer(out)
            offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
         threadsPerThreadgroup:MTLSizeMake(256u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned,
                                          "DeepSeek vision aligner GELU");
}

#include "vision_bridge.inc.m"

int ds4_gpu_glm53_scatter_image_hc(
        ds4_gpu_tensor       *hc,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc) {
    if (!hc || !image || rows == 0 || n_embd == 0 || n_hc == 0 ||
        dst_row > total_rows || rows > total_rows - dst_row ||
        !ds4_gpu_commands_active()) return 0;
    id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_vision_scatter_hc");
    if (!pipeline) return 0;
    glm53_vision_scatter_args args = {
        dst_row, image_row, rows, total_rows, n_embd, n_hc
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = cb ? ds4_gpu_compute_encoder(cb) : nil;
    if (!enc) return 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ds4_gpu_tensor_buffer(hc)
            offset:ds4_gpu_tensor_offset(hc) atIndex:1];
    [enc setBuffer:ds4_gpu_tensor_buffer(image)
            offset:ds4_gpu_tensor_offset(image) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n_embd, rows * n_hc, 1)
         threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)n_embd, 256u), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return ds4_gpu_finish_command_buffer(cb, owned, "GLM-5.3 vision scatter");
}

typedef struct {
    uint32_t n_heads;
    uint32_t n_rows;
    float lower_bound;
    float norm_eps;
} glm53_gpu_kda_args;

int ds4_gpu_glm53_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps) {
    enum { GLM53_KDA_DIM = 128, GLM53_KDA_HISTORY = 3 };
    if (!g_initialized && !ds4_gpu_init()) return 0;
    uint64_t projection = 0, activation_elements = 0;
    uint64_t conv_elements = 0, state_elements = 0;
    if (n_heads == 0 || n_rows == 0 || gate_lower_bound >= 0.0f ||
        !glm53_gpu_mul_u64(n_heads, GLM53_KDA_DIM, &projection) ||
        !glm53_gpu_mul_u64(projection, n_rows, &activation_elements) ||
        !glm53_gpu_mul_u64(activation_elements,
                        3u * GLM53_KDA_HISTORY, &conv_elements) ||
        !glm53_gpu_mul_u64(activation_elements, GLM53_KDA_DIM, &state_elements) ||
        !glm53_gpu_tensor_has(q, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(k, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(v, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(raw_gate, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(output_gate, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(raw_beta,
                           (uint64_t)n_rows * n_heads, sizeof(float)) ||
        !glm53_gpu_tensor_has(out, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(conv_state, conv_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(recurrent_state, state_elements, sizeof(float))) {
        fprintf(stderr, "ds4: GLM-5.3 KDA decode received invalid buffers\n");
        return 0;
    }

    uint64_t conv_bytes = 0, dt_bytes = 0;
    if (!glm53_gpu_mul_u64(projection, 4u * sizeof(float), &conv_bytes) ||
        !glm53_gpu_mul_u64(projection, sizeof(float), &dt_bytes)) {
        return 0;
    }
    const uint64_t a_log_bytes = (uint64_t)n_heads * sizeof(float);
    const uint64_t norm_bytes = GLM53_KDA_DIM * sizeof(float);

    @autoreleasepool {
        uint64_t qw_inner = 0, kw_inner = 0, vw_inner = 0;
        uint64_t a_inner = 0, dt_inner = 0, norm_inner = 0;
        id<MTLBuffer> qw = glm53_gpu_weight_buffer(
            model_map, model_size, q_conv_offset, conv_bytes,
            &qw_inner, "KDA Q convolution");
        id<MTLBuffer> kw = glm53_gpu_weight_buffer(
            model_map, model_size, k_conv_offset, conv_bytes,
            &kw_inner, "KDA K convolution");
        id<MTLBuffer> vw = glm53_gpu_weight_buffer(
            model_map, model_size, v_conv_offset, conv_bytes,
            &vw_inner, "KDA V convolution");
        id<MTLBuffer> a_log = glm53_gpu_weight_buffer(
            model_map, model_size, a_log_offset, a_log_bytes,
            &a_inner, "KDA A_log");
        id<MTLBuffer> dt_bias = glm53_gpu_weight_buffer(
            model_map, model_size, dt_bias_offset, dt_bytes,
            &dt_inner, "KDA dt bias");
        id<MTLBuffer> output_norm = glm53_gpu_weight_buffer(
            model_map, model_size, output_norm_offset, norm_bytes,
            &norm_inner, "KDA output norm");
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_kda_decode");
        if (!qw || !kw || !vw || !a_log || !dt_bias || !output_norm ||
            !pipeline) {
            return 0;
        }

        glm53_gpu_kda_args args = {
            .n_heads = n_heads,
            .n_rows = n_rows,
            .lower_bound = gate_lower_bound,
            .norm_eps = norm_eps,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(q)
                offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(k)
                offset:ds4_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(v)
                offset:ds4_gpu_tensor_offset(v) atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(raw_gate)
                offset:ds4_gpu_tensor_offset(raw_gate) atIndex:4];
        [enc setBuffer:ds4_gpu_tensor_buffer(raw_beta)
                offset:ds4_gpu_tensor_offset(raw_beta) atIndex:5];
        [enc setBuffer:ds4_gpu_tensor_buffer(output_gate)
                offset:ds4_gpu_tensor_offset(output_gate) atIndex:6];
        [enc setBuffer:qw offset:(NSUInteger)qw_inner atIndex:7];
        [enc setBuffer:kw offset:(NSUInteger)kw_inner atIndex:8];
        [enc setBuffer:vw offset:(NSUInteger)vw_inner atIndex:9];
        [enc setBuffer:a_log offset:(NSUInteger)a_inner atIndex:10];
        [enc setBuffer:dt_bias offset:(NSUInteger)dt_inner atIndex:11];
        [enc setBuffer:output_norm offset:(NSUInteger)norm_inner atIndex:12];
        [enc setBuffer:ds4_gpu_tensor_buffer(conv_state)
                offset:ds4_gpu_tensor_offset(conv_state) atIndex:13];
        [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_state)
                offset:ds4_gpu_tensor_offset(recurrent_state) atIndex:14];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) atIndex:15];
        [enc setThreadgroupMemoryLength:656u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, n_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(
            cb, owned, "GLM-5.3 fused KDA decode");
    }
}

int ds4_gpu_glm53_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps) {
    enum { GLM53_KDA_DIM = 128, GLM53_KDA_HISTORY = 3 };
    if (!g_initialized && !ds4_gpu_init()) return 0;
    uint64_t projection = 0, activation_elements = 0;
    uint64_t conv_elements = 0, state_elements = 0;
    if (n_heads == 0 || n_tokens == 0 || gate_lower_bound >= 0.0f ||
        !glm53_gpu_mul_u64(n_heads, GLM53_KDA_DIM, &projection) ||
        !glm53_gpu_mul_u64(projection, n_tokens, &activation_elements) ||
        !glm53_gpu_mul_u64(projection,
                        3u * GLM53_KDA_HISTORY, &conv_elements) ||
        !glm53_gpu_mul_u64(projection, GLM53_KDA_DIM, &state_elements) ||
        !glm53_gpu_tensor_has(q, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(k, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(v, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(raw_gate, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(output_gate, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(raw_beta,
                           (uint64_t)n_tokens * n_heads, sizeof(float)) ||
        !glm53_gpu_tensor_has(out, activation_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(conv_state, conv_elements, sizeof(float)) ||
        !glm53_gpu_tensor_has(recurrent_state, state_elements, sizeof(float))) {
        fprintf(stderr, "ds4: GLM-5.3 KDA prefill received invalid buffers\n");
        return 0;
    }

    uint64_t conv_bytes = 0, dt_bytes = 0;
    if (!glm53_gpu_mul_u64(projection, 4u * sizeof(float), &conv_bytes) ||
        !glm53_gpu_mul_u64(projection, sizeof(float), &dt_bytes)) {
        return 0;
    }
    const uint64_t a_log_bytes = (uint64_t)n_heads * sizeof(float);
    const uint64_t norm_bytes = GLM53_KDA_DIM * sizeof(float);

    @autoreleasepool {
        uint64_t qw_inner = 0, kw_inner = 0, vw_inner = 0;
        uint64_t a_inner = 0, dt_inner = 0, norm_inner = 0;
        id<MTLBuffer> qw = glm53_gpu_weight_buffer(
            model_map, model_size, q_conv_offset, conv_bytes,
            &qw_inner, "KDA Q convolution");
        id<MTLBuffer> kw = glm53_gpu_weight_buffer(
            model_map, model_size, k_conv_offset, conv_bytes,
            &kw_inner, "KDA K convolution");
        id<MTLBuffer> vw = glm53_gpu_weight_buffer(
            model_map, model_size, v_conv_offset, conv_bytes,
            &vw_inner, "KDA V convolution");
        id<MTLBuffer> a_log = glm53_gpu_weight_buffer(
            model_map, model_size, a_log_offset, a_log_bytes,
            &a_inner, "KDA A_log");
        id<MTLBuffer> dt_bias = glm53_gpu_weight_buffer(
            model_map, model_size, dt_bias_offset, dt_bytes,
            &dt_inner, "KDA dt bias");
        id<MTLBuffer> output_norm = glm53_gpu_weight_buffer(
            model_map, model_size, output_norm_offset, norm_bytes,
            &norm_inner, "KDA output norm");
        id<MTLComputePipelineState> prep_pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_kda_prefill_prepare");
        id<MTLComputePipelineState> recurrence_pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_kda_prefill_recurrence");
        id<MTLComputePipelineState> output_pipeline =
            ds4_gpu_get_pipeline("kernel_glm53_kda_prefill_output");
        if (!qw || !kw || !vw || !a_log || !dt_bias || !output_norm ||
            !prep_pipeline || !recurrence_pipeline || !output_pipeline) {
            return 0;
        }

        glm53_gpu_kda_args args = {
            .n_heads = n_heads,
            .n_rows = n_tokens,
            .lower_bound = gate_lower_bound,
            .norm_eps = norm_eps,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);

        [enc setComputePipelineState:prep_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(q)
                offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(k)
                offset:ds4_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(v)
                offset:ds4_gpu_tensor_offset(v) atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(raw_gate)
                offset:ds4_gpu_tensor_offset(raw_gate) atIndex:4];
        [enc setBuffer:qw offset:(NSUInteger)qw_inner atIndex:5];
        [enc setBuffer:kw offset:(NSUInteger)kw_inner atIndex:6];
        [enc setBuffer:vw offset:(NSUInteger)vw_inner atIndex:7];
        [enc setBuffer:a_log offset:(NSUInteger)a_inner atIndex:8];
        [enc setBuffer:dt_bias offset:(NSUInteger)dt_inner atIndex:9];
        [enc setBuffer:ds4_gpu_tensor_buffer(conv_state)
                offset:ds4_gpu_tensor_offset(conv_state) atIndex:10];
        [enc setThreadgroupMemoryLength:264u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:recurrence_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(q)
                offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(k)
                offset:ds4_gpu_tensor_offset(k) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(v)
                offset:ds4_gpu_tensor_offset(v) atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(raw_gate)
                offset:ds4_gpu_tensor_offset(raw_gate) atIndex:4];
        [enc setBuffer:ds4_gpu_tensor_buffer(raw_beta)
                offset:ds4_gpu_tensor_offset(raw_beta) atIndex:5];
        [enc setBuffer:ds4_gpu_tensor_buffer(recurrent_state)
                offset:ds4_gpu_tensor_offset(recurrent_state) atIndex:6];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 32, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:output_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(output_gate)
                offset:ds4_gpu_tensor_offset(output_gate) atIndex:2];
        [enc setBuffer:output_norm offset:(NSUInteger)norm_inner atIndex:3];
        [enc setThreadgroupMemoryLength:4u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(
            cb, owned, "GLM-5.3 KDA layer-major prefill");
    }
}

static bool dsv41_tensor_has_floats(const ds4_gpu_tensor *tensor, uint64_t count) {
    return tensor && count <= UINT64_MAX / sizeof(float) &&
           ds4_gpu_tensor_bytes(tensor) >= count * sizeof(float);
}

/* Frequency rounding errors accumulate into phase errors at long contexts.
 * Preserve the reference's pow/reciprocal and YaRN operation order here. */
#pragma float_control(precise, on, push)
#pragma float_control(pop)













void ds4_gpu_set_glm_mtp_verify_mode(bool enabled) {
    (void)enabled;
}

/* Qwen3.8-Flash-Next kernels (src/metal/qwen4.metal): one dispatch per wrapper,
 * weights bound from the model map by offset, transients as ds4_gpu_tensors.
 * Every wrapper joins the active command batch. */

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger    off;
} qwen4_bind;

static bool qwen4_bind_tensor(qwen4_bind *b, const ds4_gpu_tensor *t, uint64_t min_bytes, const char *what) {
    if (!t || !ds4_gpu_tensor_buffer(t) || ds4_gpu_tensor_bytes(t) < min_bytes) {
        fprintf(stderr, "ds4: Qwen3.8 %s buffer is missing or undersized (%" PRIu64 " < %" PRIu64 ")\n",
                what, t ? ds4_gpu_tensor_bytes(t) : 0, min_bytes);
        return false;
    }
    b->buf = ds4_gpu_tensor_buffer(t);
    b->off = ds4_gpu_tensor_offset(t);
    return true;
}

static bool qwen4_bind_weight(qwen4_bind *b, const void *map, uint64_t size,
                              uint64_t offset, uint64_t bytes, const char *what) {
    uint64_t inner = 0;
    if (!map || offset > size || bytes > size - offset) {
        fprintf(stderr, "ds4: Qwen3.8 %s range is outside the mapped model\n", what);
        return false;
    }
    b->buf = ds4_gpu_wrap_model_range(map, size, offset, bytes, &inner);
    b->off = (NSUInteger)inner;
    return b->buf != nil;
}

enum {
    QWEN4_K_HC_NORM_F16 = 0,
    QWEN4_K_HC_NORM_F32,
    QWEN4_K_HC_NORM_Q8,
    QWEN4_K_HC_NORM_REUSE_F16,
    QWEN4_K_HC_NORM_REUSE_F32,
    QWEN4_K_HC_NORM_REUSE_Q8,
    QWEN4_K_HC_GATE_MIX_F16,
    QWEN4_K_HC_GATE_MIX_F16_PF,
    QWEN4_K_HC_GATE_MIX_F32,
    QWEN4_K_HC_GATE_MIX_Q8,
    QWEN4_K_HC_GATE_MIX_PAIR_F16,
    QWEN4_K_HC_GATE_MIX_PAIR_F16_PF,
    QWEN4_K_HC_GATE_MIX_PAIR_F32,
    QWEN4_K_HC_GATE_MIX_PAIR_Q8,
    QWEN4_K_MULTI_GEMV,
    QWEN4_K_HC_COMBINE,
    QWEN4_K_CONV_STREAM,
    QWEN4_K_GDN_PREP,
    QWEN4_K_GDN_SCAN,
    QWEN4_K_GDN_SCAN_R4,
    QWEN4_K_GDN_OUT,
    QWEN4_K_PLE_GATE,
    QWEN4_K_PLE_CONV,
    QWEN4_K_ROUTER,
    QWEN4_K_ATTN_PREP,
    QWEN4_K_IDX_BLOCK_KEY,
    QWEN4_K_IDX_SCORE,
    QWEN4_K_IDX_SCORE_VEC,
    QWEN4_K_IDX_SELECT,
    QWEN4_K_IDX_SELECT_PRE,
    QWEN4_K_IDX_SCORE_MM,
    QWEN4_K_IDX_EXPAND,
    QWEN4_K_ATTN_DECODE_NPT8,
    QWEN4_K_ATTN_DECODE_NPT4,
    QWEN4_K_ATTN_DECODE_NPT1,
    QWEN4_K_ATTN_MERGE_NPT8,
    QWEN4_K_ATTN_MERGE_NPT4,
    QWEN4_K_ATTN_MERGE_NPT1,
    QWEN4_K_ATTN_MERGE_WIDE_NPT8,
    QWEN4_K_ATTN_MERGE_WIDE_NPT4,
    QWEN4_K_ATTN_MM,
    QWEN4_K_MOE_MID,
    QWEN4_K_MOE_MID_Q4K,
    QWEN4_K_MOE_MID_Q4K_NR1,
    QWEN4_K_MOE_DOWN,
    QWEN4_K_MOE_DOWN_MXFP4_PF,
    QWEN4_K_MOE_MID_Q4K_GROUPED,
    QWEN4_K_MOE_DOWN_MXFP4_GROUPED,
    QWEN4_K_MOE_REDUCE,
    QWEN4_K_HC_COMBINE_NORM,
    QWEN4_K_ARGMAX,
    QWEN4_K_MTP_STAGE,
    QWEN4_K_MTP_COMBINE,
    QWEN4_K_GDN_FRONT,
    QWEN4_K_MOE_BUILD_LISTS,
    QWEN4_K_MOE_MM_MID,
    QWEN4_K_MOE_MM_MID_NT1,
    QWEN4_K_MOE_MM_MID_NT2,
    QWEN4_K_MOE_MM_DOWN,
    QWEN4_K_MOE_MM_DOWN_NT1,
    QWEN4_K_MOE_MM_DOWN_NT2,
    QWEN4_K_MOE_MM_DOWN_NT8,
    QWEN4_K_MOE_MM_MID_NT8,
    QWEN4_K_MOE_MM_MID_NAX,
    QWEN4_K_MOE_MM_DOWN_NAX,
    QWEN4_K_MOE_MM_MID_NAX64,
    QWEN4_K_MOE_MM_DOWN_NAX64,
    QWEN4_K_MOE_MM_MID_NAXF,
    QWEN4_K_MOE_MM_DOWN_NAXF,
    QWEN4_K_MOE_MM_MID_NAXF64,
    QWEN4_K_MOE_MM_DOWN_NAXF64,
    QWEN4_K_MOE_MM_MID_NAXC,
    QWEN4_K_MOE_MM_DOWN_NAXC,
    QWEN4_K_MOE_MM_MID_NAXC64,
    QWEN4_K_MOE_MM_DOWN_NAXC64,
    QWEN4_K_ROWS_F32_TO_F16,
    QWEN4_K_DENSE_MM,
    QWEN4_K_DENSE_MM_REDUCE,
    QWEN4_K_BATCH_MM_Q8_T1,
    QWEN4_K_BATCH_MM_Q8_T2,
    QWEN4_K_GDN_SCAN_ROWS,
    QWEN4_K_CONV_STREAM_ROWS,
    QWEN4_K_GDN_SCAN_ROWS2,
    QWEN4_K_CONV_STREAM_ROWS2,
    QWEN4_K_ATTN_PREP_ROWS,
    QWEN4_K_IDX_BLOCK_KEY_ROWS,
    QWEN4_K_IDX_SCORE_ROWS,
    QWEN4_K_IDX_SELECT_ROWS,
    QWEN4_K_IDX_EXPAND_ROWS,
    QWEN4_K_ATTN_DECODE_ROWS_NPT8,
    QWEN4_K_ATTN_DECODE_ROWS_NPT4,
    QWEN4_K_ATTN_MERGE_ROWS_NPT8,
    QWEN4_K_ATTN_MERGE_ROWS_NPT4,
    QWEN4_K_HC_LO_ACT,
    QWEN4_K_HC_MIX_ROWS,
    QWEN4_K_VIS_PATCH_FINISH,
    QWEN4_K_VIS_LAYERNORM,
    QWEN4_K_VIS_QKV_ROPE,
    QWEN4_K_VIS_ATTENTION,
    QWEN4_K_VIS_BIAS_RESIDUAL,
    QWEN4_K_VIS_BIAS_ACT,
    QWEN4_K_COUNT,
};

static const char *const qwen4_kernel_names[QWEN4_K_COUNT] = {
    "kernel_qwen4_hc_norm_f16",
    "kernel_qwen4_hc_norm_f32",
    "kernel_qwen4_hc_norm_q8",
    "kernel_qwen4_hc_norm_reuse_f16",
    "kernel_qwen4_hc_norm_reuse_f32",
    "kernel_qwen4_hc_norm_reuse_q8",
    "kernel_qwen4_hc_gate_mix_f16",
    "kernel_qwen4_hc_gate_mix_f16_pf",
    "kernel_qwen4_hc_gate_mix_f32",
    "kernel_qwen4_hc_gate_mix_q8",
    "kernel_qwen4_hc_gate_mix_pair_f16",
    "kernel_qwen4_hc_gate_mix_pair_f16_pf",
    "kernel_qwen4_hc_gate_mix_pair_f32",
    "kernel_qwen4_hc_gate_mix_pair_q8",
    "kernel_qwen4_multi_gemv",
    "kernel_qwen4_hc_combine",
    "kernel_qwen4_conv_stream",
    "kernel_qwen4_gdn_prep",
    "kernel_qwen4_gdn_scan",
    "kernel_qwen4_gdn_scan_r4",
    "kernel_qwen4_gdn_out",
    "kernel_qwen4_ple_gate",
    "kernel_qwen4_ple_conv",
    "kernel_qwen4_router_topk",
    "kernel_qwen4_attn_prep",
    "kernel_qwen4_idx_block_key",
    "kernel_qwen4_idx_score",
    "kernel_qwen4_idx_score_vec",
    "kernel_qwen4_idx_select",
    "kernel_qwen4_idx_select_pre",
    "kernel_qwen4_idx_score_mm",
    "kernel_qwen4_idx_expand",
    "kernel_qwen4_attn_decode_npt8",
    "kernel_qwen4_attn_decode_npt4",
    "kernel_qwen4_attn_decode_npt1",
    "kernel_qwen4_attn_merge_npt8",
    "kernel_qwen4_attn_merge_npt4",
    "kernel_qwen4_attn_merge_npt1",
    "kernel_qwen4_attn_merge_wide_npt8",
    "kernel_qwen4_attn_merge_wide_npt4",
    "kernel_qwen4_attn_mm",
    "kernel_qwen4_moe_mid",
    "kernel_qwen4_moe_mid_q4k",
    "kernel_qwen4_moe_mid_q4k_nr1",
    "kernel_qwen4_moe_down",
    "kernel_qwen4_moe_down_mxfp4_pf",
    "kernel_qwen4_moe_mid_q4k_grouped",
    "kernel_qwen4_moe_down_mxfp4_grouped",
    "kernel_qwen4_moe_reduce",
    "kernel_qwen4_hc_combine_norm_f16",
    "kernel_qwen4_argmax",
    "kernel_qwen4_mtp_stage",
    "kernel_qwen4_mtp_combine",
    "kernel_qwen4_gdn_front",
    "kernel_qwen4_moe_build_lists",
    "kernel_qwen4_moe_mm_mid",
    "kernel_qwen4_moe_mm_mid_nt1",
    "kernel_qwen4_moe_mm_mid_nt2",
    "kernel_qwen4_moe_mm_down",
    "kernel_qwen4_moe_mm_down_nt1",
    "kernel_qwen4_moe_mm_down_nt2",
    "kernel_qwen4_moe_mm_down_nt8",
    "kernel_qwen4_moe_mm_mid_nt8",
    "kernel_qwen4_moe_mm_mid_nax",
    "kernel_qwen4_moe_mm_down_nax",
    "kernel_qwen4_moe_mm_mid_nax64",
    "kernel_qwen4_moe_mm_down_nax64",
    "kernel_qwen4_moe_mm_mid_naxf",
    "kernel_qwen4_moe_mm_down_naxf",
    "kernel_qwen4_moe_mm_mid_naxf64",
    "kernel_qwen4_moe_mm_down_naxf64",
    "kernel_qwen4_moe_mm_mid_naxc",
    "kernel_qwen4_moe_mm_down_naxc",
    "kernel_qwen4_moe_mm_mid_naxc64",
    "kernel_qwen4_moe_mm_down_naxc64",
    "kernel_qwen4_rows_f32_to_f16",
    "kernel_qwen4_dense_mm",
    "kernel_qwen4_dense_mm_reduce",
    "kernel_qwen4_batch_mm_q8_t1",
    "kernel_qwen4_batch_mm_q8_t2",
    "kernel_qwen4_gdn_scan_rows",
    "kernel_qwen4_conv_stream_rows",
    "kernel_qwen4_gdn_scan_rows2",
    "kernel_qwen4_conv_stream_rows2",
    "kernel_qwen4_attn_prep_rows",
    "kernel_qwen4_idx_block_key_rows",
    "kernel_qwen4_idx_score_rows",
    "kernel_qwen4_idx_select_rows",
    "kernel_qwen4_idx_expand_rows",
    "kernel_qwen4_attn_decode_rows_npt8",
    "kernel_qwen4_attn_decode_rows_npt4",
    "kernel_qwen4_attn_merge_rows_npt8",
    "kernel_qwen4_attn_merge_rows_npt4",
    "kernel_qwen4_hc_lo_act",
    "kernel_qwen4_hc_mix_rows",
    "kernel_qwen4_vis_patch_finish",
    "kernel_qwen4_vis_layernorm",
    "kernel_qwen4_vis_qkv_rope",
    "kernel_qwen4_vis_attention",
    "kernel_qwen4_vis_bias_residual",
    "kernel_qwen4_vis_bias_act",
};

typedef struct {
    uint32_t n_tokens, n_slots, n_out, in_dim, out_rows, weight_type, row_bytes, list_cap;
    uint64_t expert_bytes;
    uint32_t n_expert, tiles_per_launch, tail_base, expert_major;
} qwen4_moe_mm_args;

/* Expert-major tile order (DS4_QWEN4_MOE_MM_ORDER=0 restores the tile-major
 * grid): consecutive threadgroups share a token tile and one expert's rows
 * stay cache-resident across its tiles. */
static uint32_t qwen4_moe_mm_expert_major(void) {
    const int override = ds4_gpu_env_bool("DS4_QWEN4_MOE_MM_ORDER");
    return override >= 0 ? (uint32_t)override : ds4_gpu_device_is_m5_apple_silicon() ? 1u : 0u;
}

static MTLSize qwen4_moe_mm_grid(uint32_t row_blocks, uint32_t n_expert, uint32_t tiles, uint32_t expert_major) {
    return expert_major ? MTLSizeMake((NSUInteger)row_blocks * tiles, n_expert, 1)
                        : MTLSizeMake(row_blocks, n_expert, tiles);
}

static id<MTLComputePipelineState> g_qwen4_pipelines[QWEN4_K_COUNT];
#define QWEN4_ATTN_NSG 4
#define QWEN4_ATTN_MAX_SPLITS 64
#define QWEN4_ATTN_ROWS_MAX 64     /* decode batch rows the attention rows kernels take */

typedef struct {
    uint32_t n_tokens, n_slots, in_dim, out_rows, weight_type, row_bytes;
    uint64_t expert_bytes;
    uint32_t has_shared, shared_type, shared_row_bytes, n_total_expert;
    uint32_t list_cap, pad0;
} qwen4_moe_args;

static bool qwen4_moe_mv_specialize(uint32_t type) {
    /* Constant quantization and logical width remove the generic decode
     * branches. Keep the original per-lane reduction order and padded stride.
     * M3 Ultra uses low-bit and MXFP4 down rows; M5 uses MXFP4 down rows. */
    const int override = ds4_gpu_env_bool("DS4_QWEN4_MOE_MV_SPECIALIZE");
    return override >= 0 ? override != 0 :
        ((type == 16u || type == 10u || type == 39u) && ds4_gpu_device_name_contains("M3 Ultra")) ||
        (type == 39u && ds4_gpu_device_is_m5_apple_silicon());
}

static uint32_t qwen4_moe_mv_rows(void) {
    return (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_MOE_MV_NR", 1u, 1u, 4u);
}

static uint32_t qwen4_moe_mv_groups(uint32_t type) {
    /* Q2_K and MXFP4 use sixteen groups on M3 Ultra; MXFP4 also does on M5.
     * Each group keeps its own rows and unchanged per-lane reduction order. */
    const uint32_t default_nsg =
        ((type == 10u || type == 39u) && ds4_gpu_device_name_contains("M3 Ultra")) ||
        (type == 39u && ds4_gpu_device_is_m5_apple_silicon()) ? 16u : 8u;
    return (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_MOE_MV_NSG", default_nsg, 1u, 16u);
}

/* resident: buffers the kernel reaches through GPU addresses rather than
 * bindings (the decode batch's per-session caches); they are made resident
 * for this dispatch. */
static int qwen4_dispatch_resident(int kernel, const void *args, size_t args_len,
                                   const qwen4_bind *binds, int n_binds,
                                   MTLSize grid, MTLSize tg, NSUInteger tg_mem,
                                   const qwen4_bind *resident, uint32_t n_resident) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        if (kernel == QWEN4_K_MOE_MID || kernel == QWEN4_K_MOE_DOWN || kernel == QWEN4_K_MOE_DOWN_MXFP4_PF) {
            const qwen4_moe_args *a = args;
            const bool specialize = qwen4_moe_mv_specialize(a->weight_type);
            const uint32_t values[] = {a->weight_type, a->shared_type, specialize ? a->in_dim : 0u, specialize ? qwen4_moe_mv_rows() : 0u};
            NSString *key = [NSString stringWithFormat:@"%s_type=%u_shared=%u_dim=%u_rows=%u",
                             qwen4_kernel_names[kernel], values[0], values[1], values[2], values[3]];
            pipeline = [g_pipeline_cache objectForKey:key];
            if (!pipeline) {
                MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
                for (uint32_t i = 0; specialize && i < 4u; i++)
                    [constants setConstantValue:&values[i] type:MTLDataTypeUInt atIndex:901u+i];
                NSError *error = nil;
                id<MTLFunction> fn = [g_library newFunctionWithName:
                    [NSString stringWithUTF8String:qwen4_kernel_names[kernel]]
                    constantValues:constants error:&error];
                if (fn) pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
                if (!pipeline) {
                    fprintf(stderr, "ds4: Qwen MoE decode pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
                    return 0;
                }
                [g_pipeline_cache setObject:pipeline forKey:key];
            }
        } else if (kernel >= QWEN4_K_MOE_MM_MID && kernel <= QWEN4_K_MOE_MM_DOWN_NAXC64) {
            /* Keep each quantization's dequantizer constant through the K
             * loop. M3 Ultra and M5 have balanced full-model measurements;
             * other devices can opt in, and zero restores the generic kernel. */
            const int override = ds4_gpu_env_bool("DS4_QWEN4_MOE_MM_SPECIALIZE");
            const bool specialize = override >= 0 ? override != 0 :
                ds4_gpu_device_name_contains("M3 Ultra") ||
                ds4_gpu_device_is_m5_apple_silicon();
            const uint32_t type = specialize ?
                ((const qwen4_moe_mm_args *)args)->weight_type : 0u;
            if (type >= 40u) return 0;
            const uint32_t tail_base = ((const qwen4_moe_mm_args *)args)->tail_base;
            NSString *key = [NSString stringWithFormat:@"%s_type=%u_tail=%u",
                             qwen4_kernel_names[kernel], type, tail_base];
            pipeline = [g_pipeline_cache objectForKey:key];
            if (!pipeline) {
                MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
                [constants setConstantValue:&type type:MTLDataTypeUInt atIndex:900];
                [constants setConstantValue:&tail_base type:MTLDataTypeUInt atIndex:905];
                NSError *error = nil;
                id<MTLFunction> fn = [g_library newFunctionWithName:
                    [NSString stringWithUTF8String:qwen4_kernel_names[kernel]]
                    constantValues:constants error:&error];
                if (fn) pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
                if (!pipeline) {
                    fprintf(stderr, "ds4: Qwen MoE pipeline failed: %s\n",
                            [[error localizedDescription] UTF8String]);
                    return 0;
                }
                [g_pipeline_cache setObject:pipeline forKey:key];
            }
        } else {
            if (!g_qwen4_pipelines[kernel]) {
                g_qwen4_pipelines[kernel] = ds4_gpu_get_pipeline(qwen4_kernel_names[kernel]);
                if (!g_qwen4_pipelines[kernel]) {
                    static int source_skew_note;
                    if (!source_skew_note) {
                        source_skew_note = 1;
                        fprintf(stderr,
                            "ds4: Qwen3.8 kernel '%s' is absent from the compiled Metal sources.\n"
                            "ds4: Metal sources are loaded from ./metal at startup, so a binary newer\n"
                            "ds4: than the working directory's src/metal/ files (e.g. run from an older\n"
                            "ds4: checkout) produces this error. Run from the matching checkout, update\n"
                            "ds4: src/metal/, or point DS4_METAL_QWEN4_SOURCE at a current qwen4.metal.\n",
                            qwen4_kernel_names[kernel]);
                    }
                    return 0;
                }
            }
            pipeline = g_qwen4_pipelines[kernel];
        }
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:args length:args_len atIndex:0];
        for (int i = 0; i < n_binds; i++) {
            [enc setBuffer:binds[i].buf offset:binds[i].off atIndex:(NSUInteger)(i + 1)];
        }
        if (tg_mem) [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
        if (n_resident) {
            id<MTLResource> res[QWEN4_ATTN_ROWS_MAX * 5u];
            if (n_resident > sizeof(res) / sizeof(res[0])) return 0;
            for (uint32_t i = 0; i < n_resident; i++) res[i] = resident[i].buf;
            [enc useResources:res count:n_resident usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, qwen4_kernel_names[kernel]);
    }
}

static int qwen4_dispatch(int kernel, const void *args, size_t args_len,
                          const qwen4_bind *binds, int n_binds,
                          MTLSize grid, MTLSize tg, NSUInteger tg_mem) {
    return qwen4_dispatch_resident(kernel, args, args_len, binds, n_binds, grid, tg, tg_mem, NULL, 0u);
}

/* bytes of one hc mixer row of n elements: f16, f32 or q8_0 */
static uint64_t qwen4_hc_row_bytes(uint32_t weight_type, uint64_t n) {
    return weight_type == 1u ? n * 2u : weight_type == 0u ? n * 4u : weight_type == 8u ? (n / 32u) * 34u : 0u;
}

static int qwen4_hc_kernel(uint32_t weight_type, int k16, int k32, int kq8) {
    return weight_type == 1u ? k16 : weight_type == 0u ? k32 : kq8;
}

static uint32_t qwen4_expert_row_bytes(uint32_t weight_type, uint32_t in_dim) {
    if (in_dim == 0 || (in_dim % 32u) != 0) return 0;
    switch (weight_type) {
    case 8u:  return (in_dim / 32u) * 34u;   /* q8_0 */
    case 12u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 144u;   /* q4_K */
    case 10u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 84u;    /* q2_K */
    case 16u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 66u;    /* iq2_xxs */
    case 39u: return (in_dim / 32u) * 17u;   /* mxfp4 */
    case 2u:  return (in_dim / 32u) * 18u;   /* q4_0 */
    case 1u:  return in_dim * 2u;            /* f16 */
    case 30u: return in_dim * 2u;            /* bf16 */
    case 0u:  return in_dim * 4u;            /* f32 */
    default:  return 0;
    }
}

int ds4_gpu_qwen4_decode_fusions_enabled(void) {
    const int override = ds4_gpu_env_bool("DS4_QWEN4_DECODE_FUSIONS");
    return override >= 0 ? override : ds4_gpu_device_name_contains("M3 Ultra");
}

int ds4_gpu_qwen4_hc_combine_norm_tensor(
        ds4_gpu_tensor *next_R, const ds4_gpu_tensor *blk, const ds4_gpu_tensor *old_inj,
        ds4_gpu_tensor *xn, ds4_gpu_tensor *inj_part, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t gamma_offset, uint64_t inject_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_inject, float eps) {
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    const uint64_t wrow = qwen4_hc_row_bytes(weight_type, dim);
    struct { uint32_t n_tokens, n_embd, n_hc, n_inject; float eps; uint32_t pad0, pad1, pad2; } args =
        { n_tokens, n_embd, n_hc, n_inject, eps, 0, 0, 0 };
    qwen4_bind b[8];
    if (n_tokens != 1u || weight_type != 1u || n_inject != n_hc) return 0;
    if (ds4_gpu_tensor_buffer(next_R) == ds4_gpu_tensor_buffer(R) ||
        ds4_gpu_tensor_buffer(inj_part) == ds4_gpu_tensor_buffer(old_inj)) return 0;
    if (n_tokens == 0 || n_embd == 0 || n_hc == 0 || n_hc > 8 || n_inject > 4 || wrow == 0 || (dim % 32) != 0 ||
        !qwen4_bind_tensor(&b[0], R, n_tokens * dim * sizeof(float), "hc norm input") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, gamma_offset, dim * sizeof(float), "hc norm gamma") ||
        !qwen4_bind_tensor(&b[3], xn, n_tokens * dim * sizeof(float), "hc norm output")) {
        return 0;
    }
    if (n_inject) {
        if (!qwen4_bind_weight(&b[2], model_map, model_size, inject_offset, (uint64_t)n_inject * wrow, "hc inject") ||
            !qwen4_bind_tensor(&b[4], inj_part,
                               (uint64_t)n_tokens * n_hc * DS4_QWEN4_HC_CHUNKS * n_inject * sizeof(float),
                               "hc inject partials")) {
            return 0;
        }
    } else {
        b[2] = b[1];
        b[4] = b[3];
    }
    if (!qwen4_bind_tensor(&b[5], next_R, dim * sizeof(float), "combine norm next R") ||
        !qwen4_bind_tensor(&b[6], blk, n_embd * sizeof(float), "combine norm block") ||
        !qwen4_bind_tensor(&b[7], old_inj, n_hc * DS4_QWEN4_HC_CHUNKS * n_hc * sizeof(float), "combine norm old inject")) return 0;
    return qwen4_dispatch(QWEN4_K_HC_COMBINE_NORM, &args, sizeof(args), b, 8,
        MTLSizeMake(n_hc * DS4_QWEN4_HC_CHUNKS, 1, 1), MTLSizeMake(128, 1, 1), 0);
}

int ds4_gpu_qwen4_hc_norm_tensor(
        ds4_gpu_tensor *xn, ds4_gpu_tensor *inj_part, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t gamma_offset, uint64_t inject_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_inject, float eps) {
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    const uint64_t wrow = qwen4_hc_row_bytes(weight_type, dim);
    struct { uint32_t n_tokens, n_embd, n_hc, n_inject; float eps; uint32_t pad0, pad1, pad2; } args =
        { n_tokens, n_embd, n_hc, n_inject, eps, 0, 0, 0 };
    qwen4_bind b[5];
    if (n_tokens == 0 || n_embd == 0 || n_hc == 0 || n_hc > 8 || n_inject > 4 || wrow == 0 || (dim % 32) != 0 ||
        !qwen4_bind_tensor(&b[0], R, n_tokens * dim * sizeof(float), "hc norm input") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, gamma_offset, dim * sizeof(float), "hc norm gamma") ||
        !qwen4_bind_tensor(&b[3], xn, n_tokens * dim * sizeof(float), "hc norm output")) {
        return 0;
    }
    if (n_inject) {
        if (!qwen4_bind_weight(&b[2], model_map, model_size, inject_offset, (uint64_t)n_inject * wrow, "hc inject") ||
            !qwen4_bind_tensor(&b[4], inj_part,
                               (uint64_t)n_tokens * n_hc * DS4_QWEN4_HC_CHUNKS * n_inject * sizeof(float),
                               "hc inject partials")) {
            return 0;
        }
    } else {
        b[2] = b[1];
        b[4] = b[3];
    }
    /* Reuse wins at the measured large prefill shape. Decode and MTP keep
     * the original chunk parallelism, including with an explicit override. */
    bool reuse = false;
    if (n_tokens > 2u) {
        const char *reuse_env = getenv("DS4_QWEN4_HC_NORM_REUSE");
        if (reuse_env != NULL && strcmp(reuse_env, "1") == 0) {
            reuse = true;
        } else if (reuse_env == NULL || strcmp(reuse_env, "0") != 0) {
            reuse = n_tokens >= 8192u && n_embd == 2560u && n_hc == 4u && n_inject == 4u &&
                    (ds4_gpu_device_name_contains("M3 Ultra") || ds4_gpu_device_is_m5_apple_silicon());
        }
    }
    const int kernel = reuse
        ? qwen4_hc_kernel(weight_type, QWEN4_K_HC_NORM_REUSE_F16, QWEN4_K_HC_NORM_REUSE_F32, QWEN4_K_HC_NORM_REUSE_Q8)
        : qwen4_hc_kernel(weight_type, QWEN4_K_HC_NORM_F16, QWEN4_K_HC_NORM_F32, QWEN4_K_HC_NORM_Q8);
    return qwen4_dispatch(kernel, &args, sizeof(args), b, 5,
                          MTLSizeMake(reuse ? n_hc : n_hc * DS4_QWEN4_HC_CHUNKS, n_tokens, 1), MTLSizeMake(128, 1, 1), 0);
}

int ds4_gpu_qwen4_hc_gate_mix_tensor(
        ds4_gpu_tensor *mixed, const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo,
        const void *model_map, uint64_t model_size, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_rank) {
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    const uint64_t wrow = qwen4_hc_row_bytes(weight_type, n_rank);
    struct { uint32_t n_tokens, n_embd, n_hc, n_rank; } args = { n_tokens, n_embd, n_hc, n_rank };
    qwen4_bind b[4];
    if (n_tokens == 0 || n_hc != 4 || (n_rank % 4) != 0 || wrow == 0 || (weight_type == 8u && (n_rank % 32) != 0) ||
        !qwen4_bind_tensor(&b[0], xn, n_tokens * dim * sizeof(float), "hc gate input") ||
        !qwen4_bind_tensor(&b[1], lo, (uint64_t)n_tokens * n_rank * sizeof(float), "hc gate lowrank") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, up_offset, dim * wrow, "hc up") ||
        !qwen4_bind_tensor(&b[3], mixed, (uint64_t)n_tokens * n_embd * sizeof(float), "hc mixed")) {
        return 0;
    }
    const bool pair = n_tokens == 2u && getenv("DS4_QWEN4_NO_HC_PAIR") == NULL;
    /* Register-prefetched F16 rows (same lane order and rounding, pinned
     * against the plain kernel by tests/test_qwen4_kernels.c); M5 default. */
    const int prefetch_override = ds4_gpu_env_bool("DS4_QWEN4_HC_MIX_PREFETCH");
    const bool prefetch = weight_type == 1u &&
        (prefetch_override >= 0 ? prefetch_override > 0 : ds4_gpu_device_is_m5_apple_silicon());
    const int kernel = pair ? (prefetch ? QWEN4_K_HC_GATE_MIX_PAIR_F16_PF
                                        : qwen4_hc_kernel(weight_type, QWEN4_K_HC_GATE_MIX_PAIR_F16,
                                              QWEN4_K_HC_GATE_MIX_PAIR_F32, QWEN4_K_HC_GATE_MIX_PAIR_Q8))
                            : prefetch ? QWEN4_K_HC_GATE_MIX_F16_PF
                            : qwen4_hc_kernel(weight_type, QWEN4_K_HC_GATE_MIX_F16, QWEN4_K_HC_GATE_MIX_F32,
                                       QWEN4_K_HC_GATE_MIX_Q8);
    /* More independent output rows share the activated inputs in MTP.
     * Keep the per-row lane mapping and reduction order unchanged. */
    const uint32_t default_nsg = n_embd == 2560u && n_rank == 320u &&
        ds4_gpu_device_name_contains("M3 Ultra") ? 16u : 4u;
    const uint32_t nsg = pair ?
        (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_HC_PAIR_NSG", default_nsg, 1u, 16u) : 4u;
    return qwen4_dispatch(kernel, &args, sizeof(args), b, 4,
                          MTLSizeMake((n_embd + nsg - 1u) / nsg, pair ? 1u : n_tokens, 1), MTLSizeMake(nsg * 32u, 1, 1),
                          pair ? (NSUInteger)n_rank * 2u * sizeof(float) : 0u);
}

int ds4_gpu_qwen4_hc_combine_tensor(
        ds4_gpu_tensor *R, const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    struct { uint32_t n_tokens, n_embd, n_hc, pad0; } args = { n_tokens, n_embd, n_hc, 0 };
    qwen4_bind b[3];
    if (n_tokens == 0 || n_hc == 0 || n_hc > 8 ||
        !qwen4_bind_tensor(&b[0], R, (uint64_t)n_tokens * n_embd * n_hc * sizeof(float), "hc combine residual") ||
        !qwen4_bind_tensor(&b[1], out, (uint64_t)n_tokens * n_embd * sizeof(float), "hc combine block output") ||
        !qwen4_bind_tensor(&b[2], inj, (uint64_t)n_tokens * n_hc * DS4_QWEN4_HC_CHUNKS * n_hc * sizeof(float),
                           "hc combine inject")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_HC_COMBINE, &args, sizeof(args), b, 3,
                          MTLSizeMake((n_embd + 255) / 256, n_tokens, 1), MTLSizeMake(256, 1, 1), 0);
}

/* Diagnostic rollback is read per dispatch, including within one loaded engine. */
static bool qwen4_prefill_reuse(uint32_t n_tokens) {
    if (n_tokens <= 8u) return false;
    const int override = ds4_gpu_env_bool("DS4_QWEN4_PREFILL_REUSE");
    return override >= 0 ? override > 0 :
        n_tokens >= 1024u && ds4_gpu_device_name_contains("M3 Ultra");
}

int ds4_gpu_qwen4_conv_stream_tensor(
        ds4_gpu_tensor *x, ds4_gpu_tensor *state,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_channels, uint32_t conv_kernel, bool apply_silu) {
    struct { uint32_t n_tokens, n_channels, conv_kernel, apply_silu; } args =
        { n_tokens, n_channels, conv_kernel, apply_silu ? 1u : 0u };
    qwen4_bind b[3];
    if (n_tokens == 0 || n_channels == 0 || conv_kernel < 2 || conv_kernel > 4 ||
        !qwen4_bind_tensor(&b[0], x, (uint64_t)n_tokens * n_channels * sizeof(float), "conv input") ||
        !qwen4_bind_tensor(&b[1], state, (uint64_t)(conv_kernel - 1) * n_channels * sizeof(float), "conv state") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, weight_offset,
                           (uint64_t)n_channels * conv_kernel * sizeof(float), "conv weight")) {
        return 0;
    }
    const uint64_t blocks = ((uint64_t)n_tokens + 63u) / 64u;
    const uint64_t total = blocks * (conv_kernel - 1u) * n_channels;
    if (total <= (1u << 28) && qwen4_prefill_reuse(n_tokens)) {
        if (!ds4_gpu_ensure_scratch_buffer(&g_qwen4_conv_halo_buffer, &g_qwen4_conv_halo_bytes,
                                          (NSUInteger)(total * 4u), "convolution block history")) return 0;
        const char *names[] = {"kernel_qwen4_conv_halo", "kernel_qwen4_conv_blocked"};
        for (int stage = 0; stage < 2; stage++) {
            id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(names[stage]);
            if (!pipeline) return 0;
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:b[0].buf offset:b[0].off atIndex:1];
            [enc setBuffer:b[1].buf offset:b[1].off atIndex:2];
            if (stage) [enc setBuffer:b[2].buf offset:b[2].off atIndex:3];
            [enc setBuffer:g_qwen4_conv_halo_buffer offset:0 atIndex:stage ? 4 : 3];
            const uint64_t work = stage ? blocks * n_channels : total;
            [enc dispatchThreadgroups:MTLSizeMake((work + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            if (!ds4_gpu_finish_command_buffer(cb, owned, names[stage])) return 0;
        }
        return 1;
    }
    return qwen4_dispatch(QWEN4_K_CONV_STREAM, &args, sizeof(args), b, 3,
                          MTLSizeMake((n_channels + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_gdn_prep_tensor(
        ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *model_map, uint64_t model_size, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim) {
    const uint64_t conv_dim = (uint64_t)(2u * n_k_head + n_v_head) * head_dim;
    struct { uint32_t n_tokens, n_k_head, n_v_head, head_dim; } args = { n_tokens, n_k_head, n_v_head, head_dim };
    qwen4_bind bd[5];
    if (n_tokens == 0 || head_dim < 32 || head_dim > 128 || (head_dim % 32) != 0 ||
        !qwen4_bind_tensor(&bd[0], qkv, n_tokens * conv_dim * sizeof(float), "gdn qkv") ||
        !qwen4_bind_tensor(&bd[1], a, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn alpha") ||
        !qwen4_bind_tensor(&bd[2], b, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn beta") ||
        !qwen4_bind_weight(&bd[3], model_map, model_size, ssm_a_offset, (uint64_t)n_v_head * sizeof(float), "ssm_a") ||
        !qwen4_bind_weight(&bd[4], model_map, model_size, dt_bias_offset, (uint64_t)n_v_head * sizeof(float),
                           "dt_bias")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_GDN_PREP, &args, sizeof(args), bd, 5,
                          MTLSizeMake(n_k_head, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}

int ds4_gpu_qwen4_gdn_scan_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        ds4_gpu_tensor *snap_state, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state, uint32_t snap2_tok) {
    const uint64_t conv_dim = (uint64_t)(2u * n_k_head + n_v_head) * head_dim;
    struct { uint32_t n_tokens, n_k_head, n_v_head, head_dim, snap_tok, snap2_tok, pad1, pad2; } args =
        { n_tokens, n_k_head, n_v_head, head_dim,
          snap_state ? snap_tok : UINT32_MAX, snap2_state ? snap2_tok : UINT32_MAX, 0, 0 };
    const uint64_t state_bytes = (uint64_t)n_v_head * head_dim * head_dim * sizeof(float);
    qwen4_bind bd[7];
    if (n_tokens == 0 || head_dim < 32 || head_dim > 128 || (head_dim % 32) != 0 || n_k_head == 0 ||
        !qwen4_bind_tensor(&bd[0], qkv, n_tokens * conv_dim * sizeof(float), "gdn scan qkv") ||
        !qwen4_bind_tensor(&bd[1], a, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn scan g") ||
        !qwen4_bind_tensor(&bd[2], b, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn scan beta") ||
        !qwen4_bind_tensor(&bd[3], state, state_bytes, "gdn state") ||
        !qwen4_bind_tensor(&bd[4], out, (uint64_t)n_tokens * n_v_head * head_dim * sizeof(float), "gdn scan output")) {
        return 0;
    }
    if (snap_state) {
        if (!qwen4_bind_tensor(&bd[5], snap_state, state_bytes, "gdn snapshot")) return 0;
    } else {
        bd[5] = bd[3];
    }
    if (snap2_state) {
        if (!qwen4_bind_tensor(&bd[6], snap2_state, state_bytes, "gdn snapshot 2")) return 0;
    } else {
        bd[6] = bd[3];
    }
    const bool decode_r4 = (n_tokens <= 2u || (n_tokens == 3u && g_qwen4_verify_rows_exact)) &&
        getenv("DS4_QWEN4_NO_GDN_R4") == NULL;
    if (head_dim == 128u && (n_tokens > 8u || decode_r4)) {
        /* SIMD groups own independent value rows, so changing their grouping
         * preserves the recurrent arithmetic. M3 Ultra prefers eight groups
         * for verification and four at the measured large-prefill shape. */
        const uint32_t default_nsg = n_tokens == 2u && ds4_gpu_device_name_contains("M3 Ultra") ? 8u : 1u;
        const uint32_t prefill_default_nsg =
            n_tokens >= 8192u && n_k_head == 16u && n_v_head == 48u &&
            ds4_gpu_device_name_contains("M3 Ultra") ? 4u : 1u;
        const uint32_t nsg = n_tokens <= 2u ?
            (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_GDN_NSG", default_nsg, 1u, 8u) :
            (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_GDN_PREFILL_NSG", prefill_default_nsg, 1u, 8u);
        return qwen4_dispatch(QWEN4_K_GDN_SCAN_R4, &args, sizeof(args), bd, 7,
                              MTLSizeMake((head_dim + 4u * nsg - 1u) / (4u * nsg), n_v_head, 1),
                              MTLSizeMake(32u * nsg, 1, 1), 0);
    }
    return qwen4_dispatch(QWEN4_K_GDN_SCAN, &args, sizeof(args), bd, 7,
                          MTLSizeMake(head_dim, n_v_head, 1), MTLSizeMake(32, 1, 1), 0);
}

static ds4_gpu_tensor *g_qwen4_gdn_slots;

/* One decode step for every row of a batch against its own slot of the state
 * pool.  The grid keeps the per-row kernel's shape and simd-group grouping, so
 * each row's arithmetic is unchanged and the result is bit-identical; the rows
 * merely share a dispatch long enough to reach steady state. */
int ds4_gpu_qwen4_gdn_scan_rows_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state_pool,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const uint32_t *slots, uint32_t n_rows,
        uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t state_stride, uint32_t qkv_stride, uint32_t out_stride) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (head_dim != 128u || n_rows == 0u || n_rows > 64u) return 0;

    struct { uint32_t n_rows, n_k_head, n_v_head, head_dim, state_stride, qkv_stride, out_stride, pad0; } args =
        { n_rows, n_k_head, n_v_head, head_dim, state_stride, qkv_stride, out_stride, 0 };

    const uint64_t slot_bytes = (uint64_t)n_rows * sizeof(uint32_t);
    if (!g_qwen4_gdn_slots) {
        g_qwen4_gdn_slots = ds4_gpu_tensor_alloc(64u * sizeof(uint32_t));
        if (!g_qwen4_gdn_slots) return 0;
    }
    if (ds4_gpu_tensor_write(g_qwen4_gdn_slots, 0, slots, slot_bytes) == 0) return 0;

    qwen4_bind b[6];
    if (!qwen4_bind_tensor(&b[0], qkv, (uint64_t)n_rows * qkv_stride * sizeof(float), "gdn rows qkv") ||
        !qwen4_bind_tensor(&b[1], ga, (uint64_t)n_rows * n_v_head * sizeof(float), "gdn rows decay") ||
        !qwen4_bind_tensor(&b[2], gb, (uint64_t)n_rows * n_v_head * sizeof(float), "gdn rows beta") ||
        !qwen4_bind_tensor(&b[3], state_pool, 0, "gdn rows state pool") ||
        !qwen4_bind_tensor(&b[4], out, (uint64_t)n_rows * out_stride * sizeof(float), "gdn rows output") ||
        !qwen4_bind_tensor(&b[5], g_qwen4_gdn_slots, slot_bytes, "gdn rows slots")) {
        return 0;
    }
    /* Simd groups own independent value rows, so their grouping changes the
     * dispatch shape without touching a row's recurrent arithmetic. */
    const uint32_t nsg = 1u;
    return qwen4_dispatch(QWEN4_K_GDN_SCAN_ROWS, &args, sizeof(args), b, 6,
                          MTLSizeMake((head_dim + 4u * nsg - 1u) / (4u * nsg), n_v_head, n_rows),
                          MTLSizeMake(32u * nsg, 1, 1), 0);
}

/* One decode token per batch row, each against its own slot of the history
 * pool; see kernel_qwen4_conv_stream_rows. */
int ds4_gpu_qwen4_conv_stream_rows_tensor(
        ds4_gpu_tensor *x, ds4_gpu_tensor *hist_pool,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const uint32_t *slots, uint32_t n_rows, uint32_t n_channels,
        uint32_t conv_kernel, uint32_t state_stride, uint32_t x_stride, int apply_silu) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_rows == 0u || n_rows > 64u || conv_kernel < 2u || conv_kernel > 4u) return 0;

    struct { uint32_t n_rows, n_channels, conv_kernel, apply_silu, state_stride, x_stride, pad0, pad1; } args =
        { n_rows, n_channels, conv_kernel, apply_silu ? 1u : 0u, state_stride, x_stride, 0, 0 };

    const uint64_t slot_bytes = (uint64_t)n_rows * sizeof(uint32_t);
    if (!g_qwen4_gdn_slots) {
        g_qwen4_gdn_slots = ds4_gpu_tensor_alloc(64u * sizeof(uint32_t));
        if (!g_qwen4_gdn_slots) return 0;
    }
    if (ds4_gpu_tensor_write(g_qwen4_gdn_slots, 0, slots, slot_bytes) == 0) return 0;

    qwen4_bind b[4];
    if (!qwen4_bind_tensor(&b[0], x, (uint64_t)n_rows * x_stride * sizeof(float), "conv rows x") ||
        !qwen4_bind_tensor(&b[1], hist_pool, 0, "conv rows history pool") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, weight_offset,
                           (uint64_t)n_channels * conv_kernel * sizeof(float), "conv rows weight") ||
        !qwen4_bind_tensor(&b[3], g_qwen4_gdn_slots, slot_bytes, "conv rows slots")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_CONV_STREAM_ROWS, &args, sizeof(args), b, 4,
                          MTLSizeMake((n_channels + 255u) / 256u, n_rows, 1), MTLSizeMake(256, 1, 1), 0);
}

/* ---- GDN rows by address: one or two tokens per session ---------------- */

static uint64_t qwen4_tensor_address(const ds4_gpu_tensor *t);

typedef struct {
    uint64_t state, hist, snap_state, snap_hist;
    uint32_t row0, n_tok, pad0, pad1;
} qwen4_gdn_row_entry;

int ds4_gpu_qwen4_gdn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
                                 const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!rows || n_rows == 0u || n_rows > QWEN4_ATTN_ROWS_MAX) return 0;
    qwen4_gdn_row_entry e[QWEN4_ATTN_ROWS_MAX];
    for (uint32_t i = 0; i < n_rows; i++) {
        e[i].state = qwen4_tensor_address(rows[i].state);
        e[i].hist = qwen4_tensor_address(rows[i].hist);
        e[i].snap_state = qwen4_tensor_address(rows[i].snap_state);
        e[i].snap_hist = qwen4_tensor_address(rows[i].snap_hist);
        if (!e[i].state || !e[i].hist || (rows[i].snap_state && !e[i].snap_state) ||
            (rows[i].snap_hist && !e[i].snap_hist) || rows[i].n_tok == 0u || rows[i].n_tok > 2u) {
            return 0;
        }
        e[i].row0 = rows[i].row0;
        e[i].n_tok = rows[i].n_tok;
        e[i].pad0 = e[i].pad1 = 0u;
    }
    return ds4_gpu_tensor_write(table, entry0 * sizeof(e[0]), e, (uint64_t)n_rows * sizeof(e[0]));
}

static uint32_t qwen4_gdn_rows_resident(qwen4_bind *out, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_tensor *ts[4] = { rows[i].state, rows[i].hist, rows[i].snap_state, rows[i].snap_hist };
        for (int j = 0; j < 4; j++) {
            if (!ts[j]) continue;
            out[n].buf = ds4_gpu_tensor_buffer(ts[j]);
            out[n].off = 0;
            n++;
        }
    }
    return n;
}

static bool qwen4_bind_gdn_rows(qwen4_bind *b, const ds4_gpu_tensor *table, uint64_t entry0, uint32_t n_rows) {
    const uint64_t off = entry0 * sizeof(qwen4_gdn_row_entry);
    if (!table || !ds4_gpu_tensor_buffer(table) ||
        ds4_gpu_tensor_bytes(table) < off + (uint64_t)n_rows * sizeof(qwen4_gdn_row_entry)) {
        fprintf(stderr, "ds4: Qwen3.8 GDN row table is missing or undersized\n");
        return false;
    }
    b->buf = ds4_gpu_tensor_buffer(table);
    b->off = ds4_gpu_tensor_offset(table) + (NSUInteger)off;
    return true;
}

int ds4_gpu_qwen4_gdn_scan_rows2_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *ga, const ds4_gpu_tensor *gb,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t qkv_stride, uint32_t out_stride) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (head_dim != 128u || n_rows == 0u || n_rows > QWEN4_ATTN_ROWS_MAX) return 0;
    struct { uint32_t n_rows, n_k_head, n_v_head, head_dim, state_stride, qkv_stride, out_stride, pad0; } args =
        { n_rows, n_k_head, n_v_head, head_dim, 0u, qkv_stride, out_stride, 0 };
    qwen4_bind b[5], res[QWEN4_ATTN_ROWS_MAX * 4u];
    if (!qwen4_bind_tensor(&b[0], qkv, (uint64_t)n_batch_rows * qkv_stride * sizeof(float), "gdn rows qkv") ||
        !qwen4_bind_tensor(&b[1], ga, (uint64_t)n_batch_rows * n_v_head * sizeof(float), "gdn rows decay") ||
        !qwen4_bind_tensor(&b[2], gb, (uint64_t)n_batch_rows * n_v_head * sizeof(float), "gdn rows beta") ||
        !qwen4_bind_tensor(&b[3], out, (uint64_t)n_batch_rows * out_stride * sizeof(float), "gdn rows output") ||
        !qwen4_bind_gdn_rows(&b[4], table, entry0, n_rows)) {
        return 0;
    }
    const uint32_t nsg = 1u;
    return qwen4_dispatch_resident(QWEN4_K_GDN_SCAN_ROWS2, &args, sizeof(args), b, 5,
                                   MTLSizeMake((head_dim + 4u * nsg - 1u) / (4u * nsg), n_v_head, n_rows),
                                   MTLSizeMake(32u * nsg, 1, 1), 0, res, qwen4_gdn_rows_resident(res, rows, n_rows));
}

int ds4_gpu_qwen4_conv_stream_rows2_tensor(
        ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_gdn_row *rows, uint32_t n_rows,
        uint32_t n_batch_rows, uint32_t n_channels, uint32_t conv_kernel, uint32_t x_stride, int apply_silu) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_rows == 0u || n_rows > QWEN4_ATTN_ROWS_MAX || conv_kernel < 2u || conv_kernel > 4u) return 0;
    struct { uint32_t n_rows, n_channels, conv_kernel, apply_silu, state_stride, x_stride, pad0, pad1; } args =
        { n_rows, n_channels, conv_kernel, apply_silu ? 1u : 0u, 0u, x_stride, 0, 0 };
    qwen4_bind b[3], res[QWEN4_ATTN_ROWS_MAX * 4u];
    if (!qwen4_bind_tensor(&b[0], x, (uint64_t)n_batch_rows * x_stride * sizeof(float), "conv rows x") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, weight_offset,
                           (uint64_t)n_channels * conv_kernel * sizeof(float), "conv rows weight") ||
        !qwen4_bind_gdn_rows(&b[2], table, entry0, n_rows)) {
        return 0;
    }
    return qwen4_dispatch_resident(QWEN4_K_CONV_STREAM_ROWS2, &args, sizeof(args), b, 3,
                                   MTLSizeMake((n_channels + 255u) / 256u, n_rows, 1), MTLSizeMake(256, 1, 1), 0,
                                   res, qwen4_gdn_rows_resident(res, rows, n_rows));
}

int ds4_gpu_qwen4_gdn_out_tensor(
        ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, float eps) {
    struct { uint32_t n_tokens, n_head, head_dim; float eps; } args = { n_tokens, n_head, head_dim, eps };
    qwen4_bind b[3];
    const uint64_t bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    if (n_tokens == 0 || head_dim < 32 || (head_dim % 32) != 0 ||
        !qwen4_bind_tensor(&b[0], o, bytes, "gdn out") ||
        !qwen4_bind_tensor(&b[1], z, bytes, "gdn gate") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, weight_offset, (uint64_t)head_dim * sizeof(float),
                           "ssm_norm")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_GDN_OUT, &args, sizeof(args), b, 3,
                          MTLSizeMake(n_head, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}

int ds4_gpu_qwen4_ple_gate_tensor(
        ds4_gpu_tensor *gated, ds4_gpu_tensor *normed, const ds4_gpu_tensor *R,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const void *model_map, uint64_t model_size,
        uint64_t g_key_offset, uint64_t g_query_offset, uint64_t g_conv_offset,
        uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    struct { uint32_t n_tokens, n_embd, n_hc; float eps; } args = { n_tokens, n_embd, n_hc, eps };
    qwen4_bind b[8];
    if (n_tokens == 0 ||
        !qwen4_bind_tensor(&b[0], R, n_tokens * dim * sizeof(float), "ple residual") ||
        !qwen4_bind_tensor(&b[1], key, n_tokens * dim * sizeof(float), "ple key") ||
        !qwen4_bind_tensor(&b[2], value, (uint64_t)n_tokens * n_embd * sizeof(float), "ple value") ||
        !qwen4_bind_weight(&b[3], model_map, model_size, g_key_offset, dim * sizeof(float), "ple norm_key") ||
        !qwen4_bind_weight(&b[4], model_map, model_size, g_query_offset, dim * sizeof(float), "ple norm_query") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, g_conv_offset, dim * sizeof(float), "ple norm_conv") ||
        !qwen4_bind_tensor(&b[6], gated, n_tokens * dim * sizeof(float), "ple gated") ||
        !qwen4_bind_tensor(&b[7], normed, n_tokens * dim * sizeof(float), "ple normed")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_PLE_GATE, &args, sizeof(args), b, 8,
                          MTLSizeMake(n_tokens, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_ple_conv_tensor(
        ds4_gpu_tensor *R, const ds4_gpu_tensor *gated, const ds4_gpu_tensor *normed,
        ds4_gpu_tensor *history, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_channels, uint32_t conv_kernel, uint32_t dilation,
        ds4_gpu_tensor *snap_history, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_history, uint32_t snap2_tok) {
    const uint32_t wsize = weight_type == 1u ? 2u : weight_type == 0u ? 4u : 0u;
    struct { uint32_t n_tokens, n_channels, conv_kernel, dilation, weight_f16, snap_tok, snap2_tok, pad2; } args =
        { n_tokens, n_channels, conv_kernel, dilation, weight_type == 1u ? 1u : 0u,
          snap_history ? snap_tok : UINT32_MAX, snap2_history ? snap2_tok : UINT32_MAX, 0 };
    if (wsize == 0) return 0;
    const uint64_t rows = (uint64_t)n_tokens * n_channels * sizeof(float);
    const uint32_t hist = (conv_kernel - 1u) * dilation;
    qwen4_bind b[7];
    if (n_tokens == 0 || conv_kernel < 2 || conv_kernel > 4 || dilation == 0 || hist > 9 ||
        !qwen4_bind_tensor(&b[0], R, rows, "ple residual") ||
        !qwen4_bind_tensor(&b[1], gated, rows, "ple gated") ||
        !qwen4_bind_tensor(&b[2], normed, rows, "ple normed") ||
        !qwen4_bind_tensor(&b[3], history, (uint64_t)hist * n_channels * sizeof(float), "ple history") ||
        !qwen4_bind_weight(&b[4], model_map, model_size, weight_offset,
                           (uint64_t)n_channels * conv_kernel * wsize, "ple conv weight")) {
        return 0;
    }
    if (snap_history) {
        if (!qwen4_bind_tensor(&b[5], snap_history,
                               (uint64_t)(conv_kernel - 1) * dilation * n_channels * sizeof(float), "ple snapshot")) {
            return 0;
        }
    } else {
        b[5] = b[3];
    }
    if (snap2_history) {
        if (!qwen4_bind_tensor(&b[6], snap2_history,
                               (uint64_t)(conv_kernel - 1) * dilation * n_channels * sizeof(float), "ple snapshot 2")) {
            return 0;
        }
    } else {
        b[6] = b[3];
    }
    return qwen4_dispatch(QWEN4_K_PLE_CONV, &args, sizeof(args), b, 7,
                          MTLSizeMake((n_channels + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_router_topk_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint32_t gate_type, uint32_t in_dim, ds4_gpu_tensor *shared_gate,
        uint32_t n_tokens, uint32_t n_expert, uint32_t n_used) {
    const uint32_t gate_bytes = in_dim ? qwen4_expert_row_bytes(gate_type, in_dim) : 0u;
    struct { uint32_t n_tokens, n_expert, n_used, gate_type, in_dim, pad0, pad1, pad2; } args =
        { n_tokens, n_expert, n_used, gate_type, in_dim, 0, 0, 0 };
    qwen4_bind b[6];
    if (n_tokens == 0 || n_expert == 0 || n_expert > 512 || n_used == 0 || n_used > n_expert || n_used > 16 ||
        (in_dim && gate_bytes == 0) ||
        !qwen4_bind_tensor(&b[0], logits, (uint64_t)n_tokens * n_expert * sizeof(float), "router logits") ||
        !qwen4_bind_tensor(&b[1], selected, (uint64_t)n_tokens * n_used * sizeof(int32_t), "router selected") ||
        !qwen4_bind_tensor(&b[2], weights, (uint64_t)n_tokens * n_used * sizeof(float), "router weights")) {
        return 0;
    }
    if (in_dim) {
        if (!qwen4_bind_tensor(&b[3], x, (uint64_t)n_tokens * in_dim * sizeof(float), "router input") ||
            !qwen4_bind_weight(&b[4], model_map, model_size, gate_offset, gate_bytes, "shared gate row") ||
            !qwen4_bind_tensor(&b[5], shared_gate, (uint64_t)n_tokens * sizeof(float), "shared gate logit")) {
            return 0;
        }
    } else {
        b[3] = b[0];
        b[4] = b[0];
        b[5] = b[2];
    }
    return qwen4_dispatch(QWEN4_K_ROUTER, &args, sizeof(args), b, 6,
                          MTLSizeMake(n_tokens, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

static float g_qwen4_rope_freq[32];
static float g_qwen4_rope_mscale = 1.0f;
static bool g_qwen4_rope_set = false;

void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n_pairs, float mscale) {
    memset(g_qwen4_rope_freq, 0, sizeof(g_qwen4_rope_freq));
    for (uint32_t i = 0; freq && i < n_pairs && i < 32u; i++) g_qwen4_rope_freq[i] = freq[i];
    g_qwen4_rope_mscale = freq ? mscale : 1.0f;
    g_qwen4_rope_set = freq != NULL;
}

static void qwen4_rope_fill(float *freq, float *mscale, uint32_t n_rot, float base) {
    for (uint32_t i = 0; i < 32u; i++) {
        freq[i] = g_qwen4_rope_set ? g_qwen4_rope_freq[i] :
                  i < n_rot / 2u ? powf(base, -2.0f * (float)i / (float)n_rot) : 0.0f;
    }
    *mscale = g_qwen4_rope_set ? g_qwen4_rope_mscale : 1.0f;
}

int ds4_gpu_qwen4_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        ds4_gpu_tensor *iq_out, ds4_gpu_tensor *ik_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset, uint64_t g_iq_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_idx_head, uint32_t idx_dim, uint32_t pos0, uint32_t cache_cap,
        float rope_base, float eps) {
    struct {
        uint32_t n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap;
        float rope_base, eps; uint32_t pad0; float rope_mscale; float rope_freq[32];
    } args = { n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap,
               rope_base, eps, 0, 1.0f, { 0 } };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * n_head_kv * head_dim * sizeof(float);
    const uint64_t iq_bytes = (uint64_t)n_tokens * n_idx_head * idx_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * 2u;
    qwen4_bind b[15];
    if (n_tokens == 0 || head_dim < 32 || head_dim > 256 || (head_dim % 32) != 0 ||
        idx_dim < 32 || idx_dim > 128 || (idx_dim % 32) != 0 || n_rot > 64 || (n_rot % 2) != 0 ||
        n_rot > idx_dim || (uint64_t)pos0 + n_tokens > cache_cap || n_head_kv == 0 || (n_head % n_head_kv) != 0 ||
        !qwen4_bind_tensor(&b[0], qg, 2u * q_bytes, "attn q/gate projection") ||
        !qwen4_bind_tensor(&b[1], kproj, kv_bytes, "attn k projection") ||
        !qwen4_bind_tensor(&b[2], vproj, kv_bytes, "attn v projection") ||
        !qwen4_bind_tensor(&b[3], iq, iq_bytes, "indexer q projection") ||
        !qwen4_bind_tensor(&b[4], ik, (uint64_t)n_tokens * idx_dim * sizeof(float), "indexer k projection") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, g_q_offset, (uint64_t)head_dim * sizeof(float),
                           "attn q_norm") ||
        !qwen4_bind_weight(&b[6], model_map, model_size, g_k_offset, (uint64_t)head_dim * sizeof(float),
                           "attn k_norm") ||
        !qwen4_bind_weight(&b[7], model_map, model_size, g_iq_offset, (uint64_t)idx_dim * sizeof(float),
                           "indexer q_norm") ||
        !qwen4_bind_tensor(&b[8], q_out, q_bytes, "attn q") ||
        !qwen4_bind_tensor(&b[9], gate_out, q_bytes, "attn gate") ||
        !qwen4_bind_tensor(&b[10], k_cache, cache_bytes, "k cache") ||
        !qwen4_bind_tensor(&b[11], v_cache, cache_bytes, "v cache") ||
        !qwen4_bind_tensor(&b[12], iq_out, iq_bytes, "indexer q") ||
        !qwen4_bind_tensor(&b[13], ik_cache, (uint64_t)cache_cap * idx_dim * sizeof(float), "indexer k cache") ||
        !qwen4_bind_tensor(&b[14], pos3, (uint64_t)cache_cap * 16u, "rope positions")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_ATTN_PREP, &args, sizeof(args), b, 15,
                          MTLSizeMake(n_head + n_head_kv + n_idx_head + 1u, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}

int ds4_gpu_qwen4_idx_block_key_tensor(
        ds4_gpu_tensor *block_key, const ds4_gpu_tensor *ik_cache, const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size, uint64_t g_ik_offset,
        uint32_t block0, uint32_t n_blocks, uint32_t ratio, uint32_t idx_dim, uint32_t n_rot,
        float rope_base, float eps) {
    struct { uint32_t block0, n_blocks, ratio, idx_dim, n_rot; float rope_base, eps; uint32_t pad0;
             float rope_mscale; float rope_freq[32]; } args =
        { block0, n_blocks, ratio, idx_dim, n_rot, rope_base, eps, 0, 1.0f, { 0 } };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    const uint64_t n_keys = ((uint64_t)block0 + n_blocks) * ratio;
    qwen4_bind b[4];
    if (n_blocks == 0 || ratio == 0 || idx_dim < 32 || idx_dim > 128 || (idx_dim % 32) != 0 || n_rot > idx_dim ||
        !qwen4_bind_tensor(&b[0], ik_cache, n_keys * idx_dim * sizeof(float), "indexer k cache") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, g_ik_offset, (uint64_t)idx_dim * sizeof(float),
                           "indexer k_norm") ||
        !qwen4_bind_tensor(&b[2], pos3, n_keys * 16u, "rope positions") ||
        !qwen4_bind_tensor(&b[3], block_key, ((uint64_t)block0 + n_blocks) * idx_dim * 2u, "block keys")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_IDX_BLOCK_KEY, &args, sizeof(args), b, 4,
                          MTLSizeMake(n_blocks, 1, 1), MTLSizeMake(32, 1, 1), 0);
}

int ds4_gpu_qwen4_idx_score_tensor(
        ds4_gpu_tensor *score, ds4_gpu_tensor *tile_max, const ds4_gpu_tensor *iq, const ds4_gpu_tensor *block_key,
        uint32_t n_tokens, uint32_t n_blocks, uint32_t n_idx_head, uint32_t idx_dim,
        uint32_t pos0, uint32_t ratio) {
    struct { uint32_t n_tokens, n_blocks, n_idx_head, idx_dim, pos0, ratio, pad0, pad1; } args =
        { n_tokens, n_blocks, n_idx_head, idx_dim, pos0, ratio, 0, 0 };
    qwen4_bind b[4];
    if (n_tokens == 0 || n_blocks == 0 || ratio == 0 ||
        !qwen4_bind_tensor(&b[0], iq, (uint64_t)n_tokens * n_idx_head * idx_dim * sizeof(float), "indexer q") ||
        !qwen4_bind_tensor(&b[1], block_key, (uint64_t)n_blocks * idx_dim * 2u, "block keys") ||
        !qwen4_bind_tensor(&b[2], score, (uint64_t)n_tokens * n_blocks * sizeof(float), "indexer scores")) {
        return 0;
    }
    if (n_tokens > 2u && n_idx_head == 4u && idx_dim == 128u) {
        return qwen4_dispatch(QWEN4_K_IDX_SCORE_MM, &args, sizeof(args), b, 3,
                              MTLSizeMake((n_blocks + 63) / 64, (n_tokens + 15) / 16, 1), MTLSizeMake(128, 1, 1), 0);
    }
    /* staged queries and vector key loads, plus the tile maxima the
     * prefiltered selector needs; measured on M5, other devices keep the
     * scalar scorer */
    const int vec_override = ds4_gpu_env_bool("DS4_QWEN4_IDX_SCORE_VEC");
    const bool vec = tile_max && n_idx_head * idx_dim <= 512u && (idx_dim & 3u) == 0u &&
        (vec_override >= 0 ? vec_override != 0 : ds4_gpu_device_is_m5_apple_silicon());
    if (vec) {
        if (!qwen4_bind_tensor(&b[3], tile_max, (uint64_t)n_tokens * ((n_blocks + 7u) / 8u) * sizeof(uint32_t),
                               "indexer tile maxima")) return 0;
        return qwen4_dispatch(QWEN4_K_IDX_SCORE_VEC, &args, sizeof(args), b, 4,
                              MTLSizeMake((n_blocks + 127) / 128, n_tokens, 1), MTLSizeMake(128, 1, 1), 0);
    }
    return qwen4_dispatch(QWEN4_K_IDX_SCORE, &args, sizeof(args), b, 3,
                          MTLSizeMake((n_blocks + 127) / 128, n_tokens, 1), MTLSizeMake(128, 1, 1), 0);
}

int ds4_gpu_qwen4_idx_select_tensor(
        ds4_gpu_tensor *sel, const ds4_gpu_tensor *score, const ds4_gpu_tensor *tile_max,
        uint32_t n_blocks, uint32_t n_tokens, uint32_t top_k) {
    struct { uint32_t n_tokens, n_blocks, top_k, pad0; } args = { n_tokens, n_blocks, top_k, 0 };
    qwen4_bind b[3];
    if (n_tokens == 0 || top_k == 0 || top_k > n_blocks ||
        !qwen4_bind_tensor(&b[0], score, (uint64_t)n_tokens * n_blocks * sizeof(float), "indexer scores") ||
        !qwen4_bind_tensor(&b[2], sel, (uint64_t)n_tokens * top_k * sizeof(int32_t), "selected blocks")) {
        return 0;
    }
    /* Long rows: bound the k-th score by the k-th tile maximum and select
     * over the surviving tiles in threadgroup memory (exact; see the kernel).
     * The scorer that emits tile maxima only runs for decode batches. */
    const int pre_override = ds4_gpu_env_bool("DS4_QWEN4_IDX_PREFILTER");
    /* Scalar/MM scorers do not populate tile maxima. An independent
     * scorer override must also disable their consumer. */
    const int vec_override = ds4_gpu_env_bool("DS4_QWEN4_IDX_SCORE_VEC");
    const bool vec = vec_override >= 0 ? vec_override != 0 : ds4_gpu_device_is_m5_apple_silicon();
    const bool pre = tile_max && vec && n_tokens <= 2u && n_blocks > 8u * top_k &&
        (pre_override >= 0 ? pre_override != 0 : ds4_gpu_device_is_m5_apple_silicon());
    if (pre) {
        if (!qwen4_bind_tensor(&b[1], tile_max, (uint64_t)n_tokens * ((n_blocks + 7u) / 8u) * sizeof(uint32_t),
                               "indexer tile maxima")) return 0;
        return qwen4_dispatch(QWEN4_K_IDX_SELECT_PRE, &args, sizeof(args), b, 3,
                              MTLSizeMake(n_tokens, 1, 1), MTLSizeMake(1024, 1, 1), 0);
    }
    b[1] = b[2];
    return qwen4_dispatch(QWEN4_K_IDX_SELECT, &args, sizeof(args), b, 2,
                          MTLSizeMake(n_tokens, 1, 1), MTLSizeMake(1024, 1, 1), 0);
}

int ds4_gpu_qwen4_idx_expand_tensor(
        ds4_gpu_tensor *sel_tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel_blocks,
        uint32_t n_tokens, uint32_t n_sel_blocks, uint32_t ratio, uint32_t pos0, uint32_t sel_stride) {
    struct { uint32_t n_tokens, n_sel_blocks, ratio, pos0, sel_stride, pad0, pad1, pad2; } args =
        { n_tokens, n_sel_blocks, ratio, pos0, sel_stride, 0, 0, 0 };
    qwen4_bind b[3];
    if (n_tokens == 0 || ratio == 0 || sel_stride < n_sel_blocks * ratio + ratio - 1u ||
        !qwen4_bind_tensor(&b[0], sel_blocks, (uint64_t)n_tokens * n_sel_blocks * sizeof(int32_t), "selected blocks") ||
        !qwen4_bind_tensor(&b[1], sel_tokens, (uint64_t)n_tokens * sel_stride * sizeof(int32_t), "selected tokens") ||
        !qwen4_bind_tensor(&b[2], n_sel, (uint64_t)n_tokens * sizeof(uint32_t), "selected counts")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_IDX_EXPAND, &args, sizeof(args), b, 3,
                          MTLSizeMake(n_tokens, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

/* keys per split of the partial-softmax decode path (DS4_QWEN4_ATTN_SPLIT_KEYS overrides) */
/* Read per call, not cached, so an A/B harness can switch it per step. */
static uint32_t qwen4_attn_split_keys(void) {
    const char *env = getenv("DS4_QWEN4_ATTN_SPLIT_KEYS");
    const int v = env ? atoi(env) : 0;
    return v > 0 ? (uint32_t)v : 32u;
}

uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t n_tokens, uint32_t n_head, uint32_t head_dim) {
    return (uint64_t)n_tokens * n_head * QWEN4_ATTN_MAX_SPLITS * (2u + head_dim);
}

int ds4_gpu_qwen4_attn_decode_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel, ds4_gpu_tensor *part,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t pos0, bool use_sel, uint32_t sel_stride, float scale) {
    const uint32_t n_keys = use_sel ? sel_stride : pos0 + n_tokens;
    uint32_t n_splits = 1;
    if (part) {
        n_splits = (n_keys + qwen4_attn_split_keys() - 1) / qwen4_attn_split_keys();
        if (n_splits < 1) n_splits = 1;
        if (n_splits > QWEN4_ATTN_MAX_SPLITS) n_splits = QWEN4_ATTN_MAX_SPLITS;
    }
    const uint32_t keys_per_split = (n_keys + n_splits - 1) / n_splits;
    struct { uint32_t n_tokens, n_head, n_head_kv, head_dim, pos0, use_sel, sel_stride; float scale;
             uint32_t n_splits, keys_per_split, pad0, pad1; } args =
        { n_tokens, n_head, n_head_kv, head_dim, pos0, use_sel ? 1u : 0u, sel_stride, scale,
          n_splits, keys_per_split, 0, 0 };
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)(pos0 + n_tokens) * n_head_kv * head_dim * 2u;
    const uint64_t part_bytes = (uint64_t)n_tokens * n_head * n_splits * (2u + head_dim) * sizeof(float);
    qwen4_bind b[8];
    if (n_tokens == 0 || n_head_kv == 0 || (n_head % n_head_kv) != 0 || n_head / n_head_kv > 12 ||
        (head_dim != 32 && head_dim != 128 && head_dim != 256) ||
        !qwen4_bind_tensor(&b[0], q, q_bytes, "attn q") ||
        !qwen4_bind_tensor(&b[1], gate, q_bytes, "attn gate") ||
        !qwen4_bind_tensor(&b[2], k_cache, cache_bytes, "k cache") ||
        !qwen4_bind_tensor(&b[3], v_cache, cache_bytes, "v cache") ||
        !qwen4_bind_tensor(&b[6], out, q_bytes, "attn out")) {
        return 0;
    }
    if (use_sel) {
        if (!qwen4_bind_tensor(&b[4], sel_tokens, (uint64_t)n_tokens * sel_stride * sizeof(int32_t),
                               "selected tokens") ||
            !qwen4_bind_tensor(&b[5], n_sel, (uint64_t)n_tokens * sizeof(uint32_t), "selected counts")) {
            return 0;
        }
    } else {
        b[4] = b[0];
        b[5] = b[0];
    }
    if (n_splits > 1) {
        if (!qwen4_bind_tensor(&b[7], part, part_bytes, "attn partials")) return 0;
    } else {
        b[7] = b[6];
    }
    /* Keep short prefill tails on the decode arithmetic. Rounding queries
     * to half for just a few rows can flip nearly tied expert selections
     * relative to one-token continuation; large prefills retain matrix tiles. */
    if (n_splits == 1 && n_tokens > 8u && head_dim == 256u && n_head / n_head_kv <= 16u &&
        getenv("DS4_QWEN4_NO_ATTN_MM") == NULL) {
        return qwen4_dispatch(QWEN4_K_ATTN_MM, &args, sizeof(args), b, 7,
                              MTLSizeMake(n_head_kv, n_tokens, 1), MTLSizeMake(128, 1, 1), 0);
    }
    const int kd = head_dim == 256u ? QWEN4_K_ATTN_DECODE_NPT8 :
                   head_dim == 128u ? QWEN4_K_ATTN_DECODE_NPT4 : QWEN4_K_ATTN_DECODE_NPT1;
    const int km = head_dim == 256u ? QWEN4_K_ATTN_MERGE_NPT8 :
                   head_dim == 128u ? QWEN4_K_ATTN_MERGE_NPT4 : QWEN4_K_ATTN_MERGE_NPT1;
    if (!qwen4_dispatch(kd, &args, sizeof(args), b, 8,
                        MTLSizeMake(n_splits, n_head_kv, n_tokens), MTLSizeMake(32 * QWEN4_ATTN_NSG, 1, 1), 0)) {
        return 0;
    }
    if (n_splits == 1) return 1;
    qwen4_bind mb[3] = { b[7], b[1], b[6] };
    /* One thread per dim merges the same split chain with eight times the
     * threads; measured on M5, other devices keep the simdgroup merge. */
    const int wide_override = ds4_gpu_env_bool("DS4_QWEN4_ATTN_MERGE_WIDE");
    const bool wide = head_dim >= 128u &&
        (wide_override >= 0 ? wide_override != 0 : ds4_gpu_device_is_m5_apple_silicon());
    if (wide) {
        return qwen4_dispatch(head_dim == 256u ? QWEN4_K_ATTN_MERGE_WIDE_NPT8 : QWEN4_K_ATTN_MERGE_WIDE_NPT4,
                              &args, sizeof(args), mb, 3,
                              MTLSizeMake(n_head, n_tokens, 1), MTLSizeMake(head_dim, 1, 1), 0);
    }
    return qwen4_dispatch(km, &args, sizeof(args), mb, 3,
                          MTLSizeMake(n_head, n_tokens, 1), MTLSizeMake(32, 1, 1), 0);
}

/* ---- decode batch: attention over rows ----
 *
 * The caches stay per session, so the rows kernels find each row's through
 * a table of GPU addresses (one entry per row, staged per layer) and the
 * dispatch makes those buffers resident.  Every kernel keeps the single-row
 * arithmetic, so a row's outputs match the per-row dispatches bit for bit. */

typedef struct {
    uint64_t k_cache, v_cache, ik_cache, block_key, pos3;
    uint32_t pos, n_blocks, use_sel, pad0;
} qwen4_attn_row_entry;

static uint64_t qwen4_tensor_address(const ds4_gpu_tensor *t) {
    return t ? ds4_gpu_buffer_address(ds4_gpu_tensor_buffer(t), ds4_gpu_tensor_offset(t)) : 0u;
}

int ds4_gpu_qwen4_attn_rows_stage(ds4_gpu_tensor *table, uint64_t entry0,
                                  const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows, uint32_t ratio) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!rows || n_rows == 0u || n_rows > QWEN4_ATTN_ROWS_MAX || ratio == 0u) return 0;
    qwen4_attn_row_entry e[QWEN4_ATTN_ROWS_MAX];
    for (uint32_t i = 0; i < n_rows; i++) {
        e[i].k_cache = qwen4_tensor_address(rows[i].k_cache);
        e[i].v_cache = qwen4_tensor_address(rows[i].v_cache);
        e[i].ik_cache = qwen4_tensor_address(rows[i].ik_cache);
        e[i].block_key = qwen4_tensor_address(rows[i].block_key);
        e[i].pos3 = qwen4_tensor_address(rows[i].pos3);
        if (!e[i].k_cache || !e[i].v_cache || !e[i].ik_cache || !e[i].block_key || !e[i].pos3) {
            fprintf(stderr, "ds4: Qwen3.8 attention rows need GPU buffer addresses (macOS 13+)\n");
            return 0;
        }
        e[i].pos = rows[i].pos;
        e[i].n_blocks = (rows[i].pos + 1u) / ratio;
        e[i].use_sel = rows[i].use_sel ? 1u : 0u;
        e[i].pad0 = 0u;
    }
    return ds4_gpu_tensor_write(table, entry0 * sizeof(e[0]), e, (uint64_t)n_rows * sizeof(e[0]));
}

static bool qwen4_bind_rows(qwen4_bind *b, const ds4_gpu_tensor *table, uint64_t entry0, uint32_t n_rows) {
    const uint64_t off = entry0 * sizeof(qwen4_attn_row_entry);
    if (!table || !ds4_gpu_tensor_buffer(table) ||
        ds4_gpu_tensor_bytes(table) < off + (uint64_t)n_rows * sizeof(qwen4_attn_row_entry)) {
        fprintf(stderr, "ds4: Qwen3.8 attention row table is missing or undersized\n");
        return false;
    }
    b->buf = ds4_gpu_tensor_buffer(table);
    b->off = ds4_gpu_tensor_offset(table) + (NSUInteger)off;
    return true;
}

static uint32_t qwen4_rows_resident(qwen4_bind *out, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_tensor *ts[5] = { rows[i].k_cache, rows[i].v_cache, rows[i].ik_cache,
                                        rows[i].block_key, rows[i].pos3 };
        for (int j = 0; j < 5; j++) { out[n].buf = ds4_gpu_tensor_buffer(ts[j]); out[n].off = 0; n++; }
    }
    return n;
}

int ds4_gpu_qwen4_attn_prep_rows_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *iq_out,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *model_map, uint64_t model_size,
        uint64_t g_q_offset, uint64_t g_k_offset, uint64_t g_iq_offset,
        uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_idx_head, uint32_t idx_dim, float rope_base, float eps) {
    /* The caller checks each row's position against its own cache; the
     * kernel does not read cache_cap. */
    struct {
        uint32_t n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap;
        float rope_base, eps; uint32_t pad0; float rope_mscale; float rope_freq[32];
    } args = { n_rows, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, 0u, 0u,
               rope_base, eps, 0, 1.0f, { 0 } };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    const uint64_t q_bytes = (uint64_t)n_rows * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_rows * n_head_kv * head_dim * sizeof(float);
    const uint64_t iq_bytes = (uint64_t)n_rows * n_idx_head * idx_dim * sizeof(float);
    qwen4_bind b[12], res[QWEN4_ATTN_ROWS_MAX * 5u];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || head_dim < 32 || head_dim > 256 || (head_dim % 32) != 0 ||
        idx_dim < 32 || idx_dim > 128 || (idx_dim % 32) != 0 || n_rot > 64 || (n_rot % 2) != 0 ||
        n_rot > idx_dim || n_head_kv == 0 || (n_head % n_head_kv) != 0) {
        return 0;
    }
    if (!qwen4_bind_tensor(&b[0], qg, 2u * q_bytes, "attn rows q/gate projection") ||
        !qwen4_bind_tensor(&b[1], kproj, kv_bytes, "attn rows k projection") ||
        !qwen4_bind_tensor(&b[2], vproj, kv_bytes, "attn rows v projection") ||
        !qwen4_bind_tensor(&b[3], iq, iq_bytes, "indexer rows q projection") ||
        !qwen4_bind_tensor(&b[4], ik, (uint64_t)n_rows * idx_dim * sizeof(float), "indexer rows k projection") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, g_q_offset, (uint64_t)head_dim * sizeof(float), "attn q_norm") ||
        !qwen4_bind_weight(&b[6], model_map, model_size, g_k_offset, (uint64_t)head_dim * sizeof(float), "attn k_norm") ||
        !qwen4_bind_weight(&b[7], model_map, model_size, g_iq_offset, (uint64_t)idx_dim * sizeof(float), "indexer q_norm") ||
        !qwen4_bind_tensor(&b[8], q_out, q_bytes, "attn rows q") ||
        !qwen4_bind_tensor(&b[9], gate_out, q_bytes, "attn rows gate") ||
        !qwen4_bind_tensor(&b[10], iq_out, iq_bytes, "indexer rows q") ||
        !qwen4_bind_rows(&b[11], table, entry0, n_rows)) {
        return 0;
    }
    return qwen4_dispatch_resident(QWEN4_K_ATTN_PREP_ROWS, &args, sizeof(args), b, 12,
                                   MTLSizeMake(n_head + n_head_kv + n_idx_head + 1u, n_rows, 1), MTLSizeMake(32, 1, 1), 0,
                                   res, qwen4_rows_resident(res, rows, n_rows));
}

int ds4_gpu_qwen4_idx_block_key_rows_tensor(
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        const void *model_map, uint64_t model_size, uint64_t g_ik_offset,
        uint32_t ratio, uint32_t idx_dim, uint32_t n_rot, float rope_base, float eps) {
    struct { uint32_t block0, n_blocks, ratio, idx_dim, n_rot; float rope_base, eps; uint32_t pad0;
             float rope_mscale; float rope_freq[32]; } args =
        { 0u, n_rows, ratio, idx_dim, n_rot, rope_base, eps, 0, 1.0f, { 0 } };
    qwen4_rope_fill(args.rope_freq, &args.rope_mscale, n_rot, rope_base);
    qwen4_bind b[2], res[QWEN4_ATTN_ROWS_MAX * 5u];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || ratio == 0 || idx_dim < 32 || idx_dim > 128 ||
        (idx_dim % 32) != 0 || n_rot > idx_dim ||
        !qwen4_bind_weight(&b[0], model_map, model_size, g_ik_offset, (uint64_t)idx_dim * sizeof(float), "indexer k_norm") ||
        !qwen4_bind_rows(&b[1], table, entry0, n_rows)) {
        return 0;
    }
    return qwen4_dispatch_resident(QWEN4_K_IDX_BLOCK_KEY_ROWS, &args, sizeof(args), b, 2,
                                   MTLSizeMake(n_rows, 1, 1), MTLSizeMake(32, 1, 1), 0,
                                   res, qwen4_rows_resident(res, rows, n_rows));
}

int ds4_gpu_qwen4_idx_score_rows_tensor(
        ds4_gpu_tensor *score, ds4_gpu_tensor *tile_max, const ds4_gpu_tensor *iq,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t n_idx_head, uint32_t idx_dim, uint32_t ratio) {
    struct { uint32_t n_tokens, n_blocks, n_idx_head, idx_dim, pos0, ratio, pad0, pad1; } args =
        { n_rows, n_block_stride, n_idx_head, idx_dim, 0u, ratio, 0, 0 };
    const uint32_t n_tiles = (n_block_stride + 7u) / 8u;
    qwen4_bind b[4], res[QWEN4_ATTN_ROWS_MAX * 5u];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || n_block_stride == 0 || ratio == 0 ||
        n_idx_head * idx_dim > 512u || (idx_dim & 3u) != 0u ||
        !qwen4_bind_tensor(&b[0], iq, (uint64_t)n_rows * n_idx_head * idx_dim * sizeof(float), "indexer rows q") ||
        !qwen4_bind_tensor(&b[1], score, (uint64_t)n_rows * n_block_stride * sizeof(float), "indexer rows scores") ||
        !qwen4_bind_tensor(&b[2], tile_max, (uint64_t)n_rows * n_tiles * sizeof(uint32_t), "indexer rows tile maxima") ||
        !qwen4_bind_rows(&b[3], table, entry0, n_rows)) {
        return 0;
    }
    return qwen4_dispatch_resident(QWEN4_K_IDX_SCORE_ROWS, &args, sizeof(args), b, 4,
                                   MTLSizeMake((n_block_stride + 127) / 128, n_rows, 1), MTLSizeMake(128, 1, 1), 0,
                                   res, qwen4_rows_resident(res, rows, n_rows));
}

int ds4_gpu_qwen4_idx_select_rows_tensor(
        ds4_gpu_tensor *sel, const ds4_gpu_tensor *score, const ds4_gpu_tensor *tile_max,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_block_stride, uint32_t top_k) {
    struct { uint32_t n_tokens, n_blocks, top_k, pad0; } args = { n_rows, n_block_stride, top_k, 0 };
    const uint32_t n_tiles = (n_block_stride + 7u) / 8u;
    qwen4_bind b[4];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || top_k == 0 || top_k > n_block_stride ||
        !qwen4_bind_tensor(&b[0], score, (uint64_t)n_rows * n_block_stride * sizeof(float), "indexer rows scores") ||
        !qwen4_bind_tensor(&b[1], tile_max, (uint64_t)n_rows * n_tiles * sizeof(uint32_t), "indexer rows tile maxima") ||
        !qwen4_bind_tensor(&b[2], sel, (uint64_t)n_rows * top_k * sizeof(int32_t), "rows selected blocks") ||
        !qwen4_bind_rows(&b[3], table, entry0, n_rows)) {
        return 0;
    }
    (void)rows;
    return qwen4_dispatch(QWEN4_K_IDX_SELECT_ROWS, &args, sizeof(args), b, 4,
                          MTLSizeMake(n_rows, 1, 1), MTLSizeMake(1024, 1, 1), 0);
}

int ds4_gpu_qwen4_idx_expand_rows_tensor(
        ds4_gpu_tensor *sel_tokens, ds4_gpu_tensor *n_sel, const ds4_gpu_tensor *sel_blocks,
        const ds4_gpu_tensor *table, uint64_t entry0, uint32_t n_rows,
        uint32_t n_sel_blocks, uint32_t ratio, uint32_t sel_stride) {
    struct { uint32_t n_tokens, n_sel_blocks, ratio, pos0, sel_stride, pad0, pad1, pad2; } args =
        { n_rows, n_sel_blocks, ratio, 0u, sel_stride, 0, 0, 0 };
    qwen4_bind b[4];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || ratio == 0 || sel_stride < n_sel_blocks * ratio + ratio - 1u ||
        !qwen4_bind_tensor(&b[0], sel_blocks, (uint64_t)n_rows * n_sel_blocks * sizeof(int32_t), "rows selected blocks") ||
        !qwen4_bind_tensor(&b[1], sel_tokens, (uint64_t)n_rows * sel_stride * sizeof(int32_t), "rows selected tokens") ||
        !qwen4_bind_tensor(&b[2], n_sel, (uint64_t)n_rows * sizeof(uint32_t), "rows selected counts") ||
        !qwen4_bind_rows(&b[3], table, entry0, n_rows)) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_IDX_EXPAND_ROWS, &args, sizeof(args), b, 4,
                          MTLSizeMake(n_rows, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_attn_decode_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel, ds4_gpu_tensor *part,
        const ds4_gpu_tensor *table, uint64_t entry0, const ds4_gpu_qwen4_attn_row *rows, uint32_t n_rows,
        uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t sel_stride, float scale) {
    /* Each row takes the splits the single-row dispatch gives its key count
     * (see the kernel); the partials use the split cap as their stride. */
    struct { uint32_t n_tokens, n_head, n_head_kv, head_dim, pos0, use_sel, sel_stride; float scale;
             uint32_t n_splits, keys_per_split, pad0, pad1; } args =
        { n_rows, n_head, n_head_kv, head_dim, 0u, 0u, sel_stride, scale,
          QWEN4_ATTN_MAX_SPLITS, qwen4_attn_split_keys(), 0, 0 };
    const uint64_t q_bytes = (uint64_t)n_rows * n_head * head_dim * sizeof(float);
    const uint64_t part_bytes = ds4_gpu_qwen4_attn_part_floats(n_rows, n_head, head_dim) * sizeof(float);
    qwen4_bind b[7], res[QWEN4_ATTN_ROWS_MAX * 5u];
    if (n_rows == 0 || n_rows > QWEN4_ATTN_ROWS_MAX || n_head_kv == 0 || (n_head % n_head_kv) != 0 ||
        n_head / n_head_kv > 12 || (head_dim != 128 && head_dim != 256) ||
        !qwen4_bind_tensor(&b[0], q, q_bytes, "attn rows q") ||
        !qwen4_bind_tensor(&b[1], gate, q_bytes, "attn rows gate") ||
        !qwen4_bind_tensor(&b[2], sel_tokens, (uint64_t)n_rows * sel_stride * sizeof(int32_t), "rows selected tokens") ||
        !qwen4_bind_tensor(&b[3], n_sel, (uint64_t)n_rows * sizeof(uint32_t), "rows selected counts") ||
        !qwen4_bind_tensor(&b[4], out, q_bytes, "attn rows out") ||
        !qwen4_bind_tensor(&b[5], part, part_bytes, "attn rows partials") ||
        !qwen4_bind_rows(&b[6], table, entry0, n_rows)) {
        return 0;
    }
    const bool npt8 = head_dim == 256u;
    if (!qwen4_dispatch_resident(npt8 ? QWEN4_K_ATTN_DECODE_ROWS_NPT8 : QWEN4_K_ATTN_DECODE_ROWS_NPT4,
                                 &args, sizeof(args), b, 7,
                                 MTLSizeMake(QWEN4_ATTN_MAX_SPLITS, n_head_kv, n_rows),
                                 MTLSizeMake(32 * QWEN4_ATTN_NSG, 1, 1), 0,
                                 res, qwen4_rows_resident(res, rows, n_rows))) {
        return 0;
    }
    qwen4_bind mb[4] = { b[5], b[1], b[4], b[6] };
    return qwen4_dispatch(npt8 ? QWEN4_K_ATTN_MERGE_ROWS_NPT8 : QWEN4_K_ATTN_MERGE_ROWS_NPT4,
                          &args, sizeof(args), mb, 4,
                          MTLSizeMake(n_head, n_rows, 1), MTLSizeMake(head_dim, 1, 1), 0);
}

int ds4_gpu_qwen4_moe_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint32_t shared_type) {
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, in_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * ff_dim;
    const bool has_shared = shared_type != UINT32_MAX;
    const uint32_t sh_row_bytes = has_shared ? qwen4_expert_row_bytes(shared_type, in_dim) : 0u;
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    const uint64_t experts_bytes = expert_bytes * n_total_expert;
    const uint64_t shared_bytes = (uint64_t)sh_row_bytes * ff_dim;
    qwen4_moe_args args = { n_tokens, n_slots, in_dim, ff_dim, weight_type, row_bytes, expert_bytes,
                            has_shared ? 1u : 0u, has_shared ? shared_type : 0u, sh_row_bytes, n_total_expert, 0u, 0u };
    qwen4_bind b[7];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || ff_dim == 0 || (has_shared && sh_row_bytes == 0) ||
        !qwen4_bind_weight(&b[0], model_map, model_size, gate_offset, experts_bytes, "moe gate experts") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, up_offset, experts_bytes, "moe up experts") ||
        !qwen4_bind_tensor(&b[2], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[3], x, (uint64_t)n_tokens * in_dim * sizeof(float), "moe input") ||
        !qwen4_bind_tensor(&b[4], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid")) {
        return 0;
    }
    if (has_shared) {
        if (!qwen4_bind_weight(&b[5], model_map, model_size, shared_gate_offset, shared_bytes, "shared gate") ||
            !qwen4_bind_weight(&b[6], model_map, model_size, shared_up_offset, shared_bytes, "shared up")) {
            return 0;
        }
    } else {
        b[5] = b[0];
        b[6] = b[1];
    }
    const bool q4k = weight_type == 12u && getenv("DS4_QWEN4_NO_Q4K_MID") == NULL;
    const bool m3_ultra = q4k && ds4_gpu_device_name_contains("M3 Ultra");
    /* One row per SIMD group improves both single-token decode and the
     * two-token MTP verifier on M3 Ultra without changing dot-product order.
     * M5 measured the single-token case with four groups per threadgroup and
     * the two-row MTP passes with four. */
    const bool m5_single = q4k && (n_tokens <= 2u || (n_tokens == 3u && g_qwen4_verify_rows_exact)) &&
        ds4_gpu_device_is_m5_apple_silicon();
    const uint32_t default_nr = (m3_ultra && n_tokens <= 2u) || m5_single ? 1u : 2u;
    const bool specialize = !q4k && qwen4_moe_mv_specialize(weight_type);
    const uint64_t nr_env = q4k ?
        ds4_gpu_env_u64("DS4_QWEN4_Q4K_MID_NR", default_nr, 1u, UINT64_MAX) :
        (specialize ? qwen4_moe_mv_rows() : 2u);
    const uint32_t nr = nr_env >= 1u && nr_env <= (q4k ? 2u : 4u) ? (uint32_t)nr_env : default_nr;
    /* NR2 without an NSG override restores the former ordered dispatch. */
    const uint32_t default_nsg = nr != 1u ? 2u : m3_ultra ? 8u : m5_single ? 4u : 2u;
    const uint32_t nsg = q4k ?
        (uint32_t)ds4_gpu_env_u64("DS4_QWEN4_Q4K_MID_NSG", default_nsg, 1u, 8u) :
        (specialize ? qwen4_moe_mv_groups(weight_type) : 4u);
    const uint32_t rows_per_tg = nr * nsg;
    const int kernel = !q4k ? QWEN4_K_MOE_MID : nr == 1u ? QWEN4_K_MOE_MID_Q4K_NR1 : QWEN4_K_MOE_MID_Q4K;
    return qwen4_dispatch(kernel, &args, sizeof(args), b, 7,
                          MTLSizeMake((ff_dim + rows_per_tg - 1) / rows_per_tg, n_out, n_tokens),
                          MTLSizeMake(32u * nsg, 1, 1), 0);
}

int ds4_gpu_qwen4_moe_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t ff_dim, uint32_t out_dim,
        uint64_t shared_down_offset, uint32_t shared_type) {
    const uint32_t weight_dim = weight_type == 10u ? (ff_dim + 255u) / 256u * 256u : ff_dim;
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, weight_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * out_dim;
    const bool has_shared = shared_type != UINT32_MAX;
    const uint32_t sh_row_bytes = has_shared ? qwen4_expert_row_bytes(shared_type, ff_dim) : 0u;
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    const uint64_t experts_bytes = expert_bytes * n_total_expert;
    const uint64_t shared_bytes = (uint64_t)sh_row_bytes * out_dim;
    qwen4_moe_args args = { n_tokens, n_slots, ff_dim, out_dim, weight_type, row_bytes, expert_bytes,
                            has_shared ? 1u : 0u, has_shared ? shared_type : 0u, sh_row_bytes, 0u, 0u, 0u };
    qwen4_bind b[5];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || (ff_dim % 32u) != 0 ||
        out_dim == 0 || (has_shared && sh_row_bytes == 0) ||
        !qwen4_bind_weight(&b[0], model_map, model_size, down_offset, experts_bytes, "moe down experts") ||
        !qwen4_bind_tensor(&b[1], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[2], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[3], part, (uint64_t)n_tokens * n_out * out_dim * sizeof(float), "moe partial")) {
        return 0;
    }
    if (has_shared) {
        if (!qwen4_bind_weight(&b[4], model_map, model_size, shared_down_offset, shared_bytes, "shared down")) return 0;
    } else {
        b[4] = b[0];
    }
    const uint32_t nsg = qwen4_moe_mv_specialize(weight_type) ? qwen4_moe_mv_groups(weight_type) : 4u;
    const uint32_t rows_per_tg = nsg * (qwen4_moe_mv_specialize(weight_type) ? qwen4_moe_mv_rows() : 2u);
    /* MXFP4 rows with four blocks per lane requested ahead (same lane map and
     * chain order, byte-identical); M5 default, DS4_QWEN4_MOE_DOWN_PREFETCH=0/1
     * overrides on any device. */
    const int prefetch_override = ds4_gpu_env_bool("DS4_QWEN4_MOE_DOWN_PREFETCH");
    const bool prefetch = weight_type == 39u && (ff_dim % 32u) == 0 &&
        (prefetch_override >= 0 ? prefetch_override > 0 : ds4_gpu_device_is_m5_apple_silicon());
    return qwen4_dispatch(prefetch ? QWEN4_K_MOE_DOWN_MXFP4_PF : QWEN4_K_MOE_DOWN, &args, sizeof(args), b, 5,
                          MTLSizeMake((out_dim + rows_per_tg - 1) / rows_per_tg, n_out, n_tokens),
                          MTLSizeMake(32u * nsg, 1, 1), 0);
}

/* Decode batch over its distinct experts (kernel_qwen4_moe_mid_q4k_grouped):
 * Q4_K gate/up, no shared slot, per-expert pair lists from
 * ds4_gpu_qwen4_moe_build_lists_tensor.  Same grid as the per-token kernel;
 * the pairs that do not own their expert return. */
int ds4_gpu_qwen4_moe_mid_grouped_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t list_cap,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t in_dim, uint32_t ff_dim) {
    if (weight_type != 12u || (in_dim % 256u) != 0) return 0;
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, in_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * ff_dim;
    qwen4_moe_args args = { n_tokens, n_slots, in_dim, ff_dim, weight_type, row_bytes, expert_bytes,
                            0u, 0u, 0u, n_total_expert, list_cap, 0u };
    qwen4_bind b[7];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || ff_dim == 0 || list_cap == 0 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, gate_offset, expert_bytes * n_total_expert, "moe gate experts") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, up_offset, expert_bytes * n_total_expert, "moe up experts") ||
        !qwen4_bind_tensor(&b[2], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[3], x, (uint64_t)n_tokens * in_dim * sizeof(float), "moe input") ||
        !qwen4_bind_tensor(&b[4], mid, (uint64_t)n_tokens * n_slots * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[5], lists, (uint64_t)n_total_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[6], counts, (uint64_t)n_total_expert * sizeof(int32_t), "moe counts")) {
        return 0;
    }
    const uint32_t nsg = 2u, rows_per_tg = 2u * nsg;
    return qwen4_dispatch(QWEN4_K_MOE_MID_Q4K_GROUPED, &args, sizeof(args), b, 7,
                          MTLSizeMake((ff_dim + rows_per_tg - 1) / rows_per_tg, n_slots, n_tokens),
                          MTLSizeMake(32u * nsg, 1, 1), 0);
}

/* The MXFP4 down rows of the same batch, on the same lists. */
int ds4_gpu_qwen4_moe_down_grouped_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, uint32_t list_cap,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_total_expert, uint32_t n_tokens, uint32_t n_slots,
        uint32_t ff_dim, uint32_t out_dim) {
    if (weight_type != 39u || (ff_dim % 32u) != 0) return 0;
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, ff_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * out_dim;
    qwen4_moe_args args = { n_tokens, n_slots, ff_dim, out_dim, weight_type, row_bytes, expert_bytes,
                            0u, 0u, 0u, n_total_expert, list_cap, 0u };
    qwen4_bind b[6];
    if (n_tokens == 0 || n_slots == 0 || row_bytes == 0 || out_dim == 0 || list_cap == 0 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, down_offset, expert_bytes * n_total_expert, "moe down experts") ||
        !qwen4_bind_tensor(&b[1], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[2], mid, (uint64_t)n_tokens * n_slots * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[3], part, (uint64_t)n_tokens * n_slots * out_dim * sizeof(float), "moe partial") ||
        !qwen4_bind_tensor(&b[4], lists, (uint64_t)n_total_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[5], counts, (uint64_t)n_total_expert * sizeof(int32_t), "moe counts")) {
        return 0;
    }
    const uint32_t nsg = 4u, rows_per_tg = 2u * nsg;
    return qwen4_dispatch(QWEN4_K_MOE_DOWN_MXFP4_GROUPED, &args, sizeof(args), b, 6,
                          MTLSizeMake((out_dim + rows_per_tg - 1) / rows_per_tg, n_slots, n_tokens),
                          MTLSizeMake(32u * nsg, 1, 1), 0);
}

int ds4_gpu_qwen4_moe_reduce_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *shared_gate, const ds4_gpu_tensor *shared, ds4_gpu_tensor *R, const ds4_gpu_tensor *inj,
        uint32_t n_tokens, uint32_t n_slots, uint32_t part_stride, uint32_t dim, uint32_t n_hc) {
    const uint32_t shared_src = !shared_gate ? 0u : shared ? 2u : 1u;
    struct { uint32_t n_tokens, n_slots, dim, shared_src, n_hc, part_stride, pad1, pad2; } args =
        { n_tokens, n_slots, dim, shared_src, R ? n_hc : 0u, part_stride, 0, 0 };
    qwen4_bind b[7];
    if (n_tokens == 0 || n_slots == 0 || dim == 0 || (R && n_hc == 0) || n_hc > 8 || part_stride < n_slots ||
        (shared_src == 1u && part_stride < n_slots + 1u) ||
        !qwen4_bind_tensor(&b[0], part, (uint64_t)n_tokens * part_stride * dim * sizeof(float), "moe partial") ||
        !qwen4_bind_tensor(&b[1], weights, (uint64_t)n_tokens * n_slots * sizeof(float), "moe weights") ||
        !qwen4_bind_tensor(&b[3], out, (uint64_t)n_tokens * dim * sizeof(float), "moe out")) {
        return 0;
    }
    if (shared_gate) {
        if (!qwen4_bind_tensor(&b[2], shared_gate, (uint64_t)n_tokens * sizeof(float), "shared gate")) return 0;
    } else {
        b[2] = b[1];
    }
    if (R) {
        if (!qwen4_bind_tensor(&b[4], R, (uint64_t)n_tokens * dim * n_hc * sizeof(float), "moe residual") ||
            !qwen4_bind_tensor(&b[5], inj, (uint64_t)n_tokens * n_hc * DS4_QWEN4_HC_CHUNKS * n_hc * sizeof(float),
                               "moe inject")) {
            return 0;
        }
    } else {
        b[4] = b[3];
        b[5] = b[1];
    }
    if (shared_src == 2u) {
        if (!qwen4_bind_tensor(&b[6], shared, (uint64_t)n_tokens * dim * sizeof(float), "shared expert output")) {
            return 0;
        }
    } else {
        b[6] = b[0];
    }
    return qwen4_dispatch(QWEN4_K_MOE_REDUCE, &args, sizeof(args), b, 7,
                          MTLSizeMake((dim + 255) / 256, n_tokens, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_moe_build_lists_tensor(
        ds4_gpu_tensor *lists, ds4_gpu_tensor *counts, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_slots, uint32_t n_expert, uint32_t list_cap) {
    qwen4_moe_mm_args args = { n_tokens, n_slots, 0, 0, 0, 0, 0, list_cap, 0, n_expert, 0, 0, 0 };
    qwen4_bind b[3];
    if (n_tokens == 0 || n_slots == 0 || n_expert == 0 || n_expert > 512 || list_cap == 0 ||
        !qwen4_bind_tensor(&b[0], selected, (uint64_t)n_tokens * n_slots * sizeof(int32_t), "moe selected") ||
        !qwen4_bind_tensor(&b[1], lists, (uint64_t)n_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[2], counts, (uint64_t)n_expert * sizeof(int32_t), "moe counts")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_MOE_BUILD_LISTS, &args, sizeof(args), b, 3,
                          MTLSizeMake(1, 1, 1), MTLSizeMake(512, 1, 1), 0);
}

static uint32_t qwen4_moe_mm_tiles(uint32_t n_tokens, bool mid) {
    uint32_t tiles = (n_tokens + 31u) / 32u;
    /* Spread large routed batches over more independent tiles on M3 Ultra
     * (gate/up and down) and on M5 (gate/up; the down tiles measured flat).
     * Each tile keeps the same K loop and accumulation order. */
    const bool spread = ds4_gpu_device_name_contains("M3 Ultra") ||
                        (mid && ds4_gpu_device_is_m5_apple_silicon());
    const uint32_t default_cap = spread ?
        (n_tokens >= 8192u ? 32u : n_tokens >= 4096u ? 16u : 8u) : 8u;
    const uint32_t cap = (uint32_t)ds4_gpu_env_u64(
        mid ? "DS4_QWEN4_MOE_MID_TILES" : "DS4_QWEN4_MOE_DOWN_TILES", default_cap, 1u, 32u);
    return tiles > cap ? cap : tiles;
}

static uint32_t qwen4_moe_mm_nt(uint32_t n_tokens, uint32_t type, const char *env_name) {
    /* Low-bit experts receive few tokens in short prefills. Smaller token
     * tiles avoid unused matrix products while retaining the same K order.
     * Keep 32-token tiles for larger batches and unmeasured devices/types. */
    uint32_t default_nt = 4u;
    if ((type == 16u || type == 10u) && ds4_gpu_device_name_contains("M3 Ultra")) {
        if (n_tokens <= 512u) default_nt = 1u;
        else if (n_tokens <= 1024u) default_nt = 2u;
    }
    const bool down = strstr(env_name, "DOWN") != NULL;
    /* Reuse each decoded down-weight tile across 64 tokens. The K loop and
     * partial-tail accumulation stay unchanged; short batches keep small tiles. */
    if (down && type == 10u && n_tokens >= 8192u && qwen4_prefill_reuse(n_tokens)) default_nt = 8u;
    /* M5: 64-token gate/up and down tiles for the Q4_K / MXFP4 pack at
     * chunk-sized batches (same tile arithmetic; remainders take the
     * 8/16/32-token kernels).  Measured on M5 Max, see
     * speed-bench/qwen38-m5-round4.md. */
    if ((type == 12u || type == 39u) && n_tokens >= 4096u && ds4_gpu_device_is_m5_apple_silicon()) default_nt = 8u;
    const uint32_t nt = (uint32_t)ds4_gpu_env_u64(env_name, default_nt, 1u, 8u);
    return nt == 1u || nt == 2u || nt == 4u || nt == 8u ? nt : default_nt;
}

/* Routed tiles on the Metal 4 tensor ops (drift class: the cooperative
 * matmul's accumulation order differs from the simdgroup tiles; needs the
 * tensor API).  DS4_QWEN4_MOE_MM_NAX: 0 simdgroup tiles; 1 tensor-op
 * tiles of 32 tokens; 2 tensor-op tiles of 64 tokens; 3/4 the same widths
 * with the activation operand kept at full float precision (x gathered from
 * the float tensor, no half rounding of x or of the mid hand-off, no
 * conversion pass); 5/6 compensated tiles (64/32 tokens), which stage the
 * half rounding residual of the operand beside it and run the tensor op on
 * both, reaching the 3/4 accuracy while keeping the fp16 tensor-op rate.
 * Unset selects 2 (the 64-token half tiles) for every tensor tier on
 * devices with the tensor API: its operand rounding matches the simdgroup
 * tiles and its fixture NLL sits within noise of them, so the compensated
 * level stays an env-selectable accuracy option.  0 forces the simdgroup
 * tiles.  Returns the token tile width, 0 for the simdgroup path. */
static long qwen4_moe_mm_nax_level(uint32_t type) {
    const char *v = getenv("DS4_QWEN4_MOE_MM_NAX");
    /* default 2 for every tensor tier: level 2 matches the simdgroup tiles'
     * operand rounding and differs only by accumulation order, and its
     * fixture NLL is within noise of them on both packs (Q4 0.20503 vs
     * 0.20505, Q2 0.30418 vs 0.30346), while it carries the full prefill
     * gain (+48/+51% Q4, +29/+36% Q2).  The compensated level 5 keeps the
     * best absolute NLL and stays one env variable away. */
    if (!v || !v[0]) return (type == 12u || type == 39u || type == 16u || type == 10u) ? 2 : 0;
    return strtol(v, NULL, 10);
}
static uint32_t qwen4_moe_mm_nax(uint32_t type) {
    if (!ds4_gpu_mpp_available()) return 0;
    if (!(type == 12u || type == 39u || type == 16u || type == 10u)) return 0;
    const long n = qwen4_moe_mm_nax_level(type);
    if (n <= 0) return 0;
    return (n == 1 || n == 4 || n == 6) ? 32u : 64u;
}
/* float-activation tensor tiles (levels 3 and 4) */
static bool qwen4_moe_mm_nax_fx(uint32_t type) {
    const long n = qwen4_moe_mm_nax_level(type);
    return n == 3 || n == 4;
}
/* compensated tensor tiles (levels 5 and 6): the fast fp16 tensor kernels run
 * twice per K step, on the half-rounded operand and on its half residual, so
 * the activation operand enters at ~2^-22 relative instead of 2^-11 */
static bool qwen4_moe_mm_nax_comp(uint32_t type) {
    const long n = qwen4_moe_mm_nax_level(type);
    return n == 5 || n == 6;
}

static ds4_gpu_tensor *g_qwen4_nax_half_x, *g_qwen4_nax_half_mid, *g_qwen4_nax_half_midr;
static uint64_t g_qwen4_nax_half_x_bytes, g_qwen4_nax_half_mid_bytes, g_qwen4_nax_half_midr_bytes;
static const ds4_gpu_tensor *g_qwen4_nax_half_mid_for;
static uint64_t g_qwen4_nax_half_mid_count;
static void qwen4_nax_release_scratch(void) {
    if (g_qwen4_nax_half_x) ds4_gpu_tensor_free(g_qwen4_nax_half_x);
    if (g_qwen4_nax_half_mid) ds4_gpu_tensor_free(g_qwen4_nax_half_mid);
    if (g_qwen4_nax_half_midr) ds4_gpu_tensor_free(g_qwen4_nax_half_midr);
    g_qwen4_nax_half_x = g_qwen4_nax_half_mid = g_qwen4_nax_half_midr = NULL;
    g_qwen4_nax_half_x_bytes = g_qwen4_nax_half_mid_bytes = g_qwen4_nax_half_midr_bytes = 0;
    g_qwen4_nax_half_mid_for = NULL;
    g_qwen4_nax_half_mid_count = 0;
}
static ds4_gpu_tensor *qwen4_nax_scratch(ds4_gpu_tensor **slot, uint64_t *slot_bytes, uint64_t bytes) {
    if (!*slot || *slot_bytes < bytes) {
        if (*slot) ds4_gpu_tensor_free(*slot);
        *slot = ds4_gpu_tensor_alloc(bytes);
        *slot_bytes = *slot ? bytes : 0;
    }
    return *slot;
}
static ds4_gpu_tensor *qwen4_nax_half_operand(ds4_gpu_tensor **slot, uint64_t *slot_bytes,
                                              const ds4_gpu_tensor *src, uint64_t count) {
    const uint64_t bytes = count * sizeof(uint16_t);
    if (count == 0 || (count % 4u) != 0) return NULL;
    ds4_gpu_tensor *dst = qwen4_nax_scratch(slot, slot_bytes, bytes);
    if (!dst) return NULL;
    struct { uint32_t n4; } args = { (uint32_t)(count / 4u) };
    qwen4_bind b[2];
    if (!qwen4_bind_tensor(&b[0], src, count * sizeof(float), "nax half operand source") ||
        !qwen4_bind_tensor(&b[1], dst, bytes, "nax half operand")) return NULL;
    const uint32_t n_threads = (args.n4 + 3u) / 4u;   /* 16 values per thread */
    if (!qwen4_dispatch(QWEN4_K_ROWS_F32_TO_F16, &args, sizeof(args), b, 2,
                        MTLSizeMake((n_threads + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1), 0)) return NULL;
    return dst;
}

static bool qwen4_moe_mm_tails(uint32_t type, uint32_t nt) {
    /* Remainder tiles measured on M3 Ultra for the low-bit experts and on M5
     * for Q4_K gate/up with MXFP4 down. */
    const int override = ds4_gpu_env_bool("DS4_QWEN4_MOE_TAILS");
    return nt > 1u && (override >= 0 ? override != 0 :
        ((type == 16u || type == 10u) && ds4_gpu_device_name_contains("M3 Ultra")) ||
        ((type == 12u || type == 39u) && ds4_gpu_device_is_m5_apple_silicon()));
}

int ds4_gpu_qwen4_moe_mm_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap) {
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, in_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * ff_dim;
    const uint32_t tiles = qwen4_moe_mm_tiles(n_tokens, true);
    const uint32_t nt = qwen4_moe_mm_nt(n_tokens, weight_type, "DS4_QWEN4_MOE_MID_NT");
    const int kernel = nt == 1u ? QWEN4_K_MOE_MM_MID_NT1 :
                       nt == 2u ? QWEN4_K_MOE_MM_MID_NT2 :
                       nt == 8u ? QWEN4_K_MOE_MM_MID_NT8 : QWEN4_K_MOE_MM_MID;
    qwen4_moe_mm_args args = { n_tokens, n_slots, n_out, in_dim, ff_dim, weight_type, row_bytes, list_cap,
                               expert_bytes, n_expert, tiles, 0, qwen4_moe_mm_expert_major() };
    const bool tails = qwen4_moe_mm_tails(weight_type, nt);
    if (tails) args.tail_base = nt * 8u;
    qwen4_bind b[8];
    if (n_tokens == 0 || n_slots == 0 || n_out < n_slots || row_bytes == 0 ||
        (weight_type != 8u && weight_type != 39u && weight_type != 12u && weight_type != 10u && weight_type != 16u && weight_type != 2u) ||
        (in_dim % 64) != 0 || ff_dim == 0 || n_expert == 0 || n_expert > 512 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, gate_offset, expert_bytes * n_expert, "moe gate experts") ||
        !qwen4_bind_weight(&b[1], model_map, model_size, up_offset, expert_bytes * n_expert, "moe up experts") ||
        !qwen4_bind_tensor(&b[2], lists, (uint64_t)n_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[3], counts, (uint64_t)n_expert * sizeof(int32_t), "moe counts") ||
        !qwen4_bind_tensor(&b[4], x, (uint64_t)n_tokens * in_dim * sizeof(float), "moe input") ||
        !qwen4_bind_tensor(&b[5], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid")) {
        return 0;
    }
    const uint32_t nax = qwen4_moe_mm_nax(weight_type);
    const bool nax_fx = qwen4_moe_mm_nax_fx(weight_type);
    const bool nax_comp = qwen4_moe_mm_nax_comp(weight_type);
    if (nax) {
        args.tail_base = 0;
        const uint64_t mid_count = (uint64_t)n_tokens * n_out * ff_dim;
        if (nax_comp) {
            /* compensated tiles: x stays bound at b[4] (float) and the tiles
             * write both the half copy of mid and its half residual */
            ds4_gpu_tensor *mh = qwen4_nax_scratch(&g_qwen4_nax_half_mid, &g_qwen4_nax_half_mid_bytes, mid_count * sizeof(uint16_t));
            ds4_gpu_tensor *mr = qwen4_nax_scratch(&g_qwen4_nax_half_midr, &g_qwen4_nax_half_midr_bytes, mid_count * sizeof(uint16_t));
            if (!mh || !mr || !qwen4_bind_tensor(&b[6], mh, mid_count * sizeof(uint16_t), "moe mid half") ||
                !qwen4_bind_tensor(&b[7], mr, mid_count * sizeof(uint16_t), "moe mid residual")) return 0;
            g_qwen4_nax_half_mid_for = mid;
            g_qwen4_nax_half_mid_count = mid_count;
        } else if (nax_fx) {
            /* float-activation tiles: x stays bound at b[4]; they write no
             * half copy of mid, so b[6] is a never-read placeholder */
            if (!qwen4_bind_tensor(&b[6], mid, mid_count * sizeof(uint16_t), "moe mid")) return 0;
        } else {
            ds4_gpu_tensor *xh = qwen4_nax_half_operand(&g_qwen4_nax_half_x, &g_qwen4_nax_half_x_bytes, x, (uint64_t)n_tokens * in_dim);
            ds4_gpu_tensor *mh = qwen4_nax_scratch(&g_qwen4_nax_half_mid, &g_qwen4_nax_half_mid_bytes, mid_count * sizeof(uint16_t));
            if (!xh || !mh || !qwen4_bind_tensor(&b[4], xh, (uint64_t)n_tokens * in_dim * sizeof(uint16_t), "moe input half") ||
                !qwen4_bind_tensor(&b[6], mh, mid_count * sizeof(uint16_t), "moe mid half")) return 0;
            g_qwen4_nax_half_mid_for = mid;
            g_qwen4_nax_half_mid_count = mid_count;
        }
        const bool nax_tails = nax == 64u && qwen4_moe_mm_tails(weight_type, 8u);
        if (nax_tails) args.tail_base = 64u;
        const int nax_mid = nax_comp ? (nax == 64u ? QWEN4_K_MOE_MM_MID_NAXC64 : QWEN4_K_MOE_MM_MID_NAXC)
                          : nax_fx ? (nax == 64u ? QWEN4_K_MOE_MM_MID_NAXF64 : QWEN4_K_MOE_MM_MID_NAXF)
                          : (nax == 64u ? QWEN4_K_MOE_MM_MID_NAX64 : QWEN4_K_MOE_MM_MID_NAX);
        const int nax_mid_tail = nax_comp ? QWEN4_K_MOE_MM_MID_NAXC : nax_fx ? QWEN4_K_MOE_MM_MID_NAXF : QWEN4_K_MOE_MM_MID_NAX;
        if (!qwen4_dispatch(nax_mid, &args, sizeof(args), b, nax_comp ? 8 : 7,
                            qwen4_moe_mm_grid((ff_dim + 63u) / 64u, n_expert, tiles, args.expert_major),
                            MTLSizeMake(128, 1, 1), nax == 64u ? 16384u : nax_fx || nax_comp ? 12288u : 10240u)) return 0;
        if (!nax_tails) return 1;
        args.tiles_per_launch = 1;
        return qwen4_dispatch(nax_mid_tail, &args, sizeof(args), b, nax_comp ? 8 : 7,
                              qwen4_moe_mm_grid((ff_dim + 63u) / 64u, n_expert, 1, args.expert_major),
                              MTLSizeMake(128, 1, 1), nax_fx || nax_comp ? 12288u : 10240u);
    }
    if (!qwen4_dispatch(kernel, &args, sizeof(args), b, 6,
                          qwen4_moe_mm_grid((ff_dim + 31u) / 32u, n_expert, tiles, args.expert_major),
                          MTLSizeMake(128, 1, 1), 0)) return 0;
    if (tails) {
        const MTLSize tail_grid = MTLSizeMake((ff_dim + 31u) / 32u, n_expert, 1);
        if (!qwen4_dispatch(QWEN4_K_MOE_MM_MID_NT1, &args, sizeof(args), b, 6, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 2u && !qwen4_dispatch(QWEN4_K_MOE_MM_MID_NT2, &args, sizeof(args), b, 6, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 4u && !qwen4_dispatch(QWEN4_K_MOE_MM_MID, &args, sizeof(args), b, 6, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
    }
    return 1;
}

int ds4_gpu_qwen4_moe_mm_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t weight_type, uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
        uint32_t ff_dim, uint32_t out_dim, uint32_t list_cap) {
    const uint32_t weight_dim = weight_type == 10u ? (ff_dim + 255u) / 256u * 256u : ff_dim;
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, weight_dim);
    const uint64_t expert_bytes = (uint64_t)row_bytes * out_dim;
    const uint32_t tiles = qwen4_moe_mm_tiles(n_tokens, false);
    const uint32_t nt = qwen4_moe_mm_nt(n_tokens, weight_type, "DS4_QWEN4_MOE_DOWN_NT");
    const int kernel = nt == 1u ? QWEN4_K_MOE_MM_DOWN_NT1 :
                       nt == 2u ? QWEN4_K_MOE_MM_DOWN_NT2 :
                       nt == 8u ? QWEN4_K_MOE_MM_DOWN_NT8 : QWEN4_K_MOE_MM_DOWN;
    qwen4_moe_mm_args args = { n_tokens, n_slots, n_out, ff_dim, out_dim, weight_type, row_bytes, list_cap,
                               expert_bytes, n_expert, tiles, 0, qwen4_moe_mm_expert_major() };
    const bool tails = qwen4_moe_mm_tails(weight_type, nt);
    if (tails) args.tail_base = nt * 8u;
    qwen4_bind b[6];
    if (n_tokens == 0 || n_slots == 0 || n_out < n_slots || row_bytes == 0 ||
        (weight_type != 8u && weight_type != 39u && weight_type != 12u && weight_type != 10u && weight_type != 16u && weight_type != 2u) ||
        (ff_dim % 64) != 0 || out_dim == 0 || n_expert == 0 || n_expert > 512 ||
        !qwen4_bind_weight(&b[0], model_map, model_size, down_offset, expert_bytes * n_expert, "moe down experts") ||
        !qwen4_bind_tensor(&b[1], lists, (uint64_t)n_expert * list_cap * sizeof(int32_t), "moe lists") ||
        !qwen4_bind_tensor(&b[2], counts, (uint64_t)n_expert * sizeof(int32_t), "moe counts") ||
        !qwen4_bind_tensor(&b[3], mid, (uint64_t)n_tokens * n_out * ff_dim * sizeof(float), "moe mid") ||
        !qwen4_bind_tensor(&b[4], part, (uint64_t)n_tokens * n_out * out_dim * sizeof(float), "moe partial")) {
        return 0;
    }
    const uint32_t nax = qwen4_moe_mm_nax(weight_type);
    const bool nax_fx = qwen4_moe_mm_nax_fx(weight_type);
    bool nax_comp = qwen4_moe_mm_nax_comp(weight_type);
    if (nax) {
        args.tail_base = 0;
        const uint64_t mid_count = (uint64_t)n_tokens * n_out * ff_dim;
        if (nax_comp) {
            /* the compensated tiles read the half copy of mid and its
             * residual; both must come from the mid tiles' matching launch,
             * otherwise fall back to the uncompensated tiles */
            if (g_qwen4_nax_half_mid_for == mid && g_qwen4_nax_half_mid_count == mid_count &&
                g_qwen4_nax_half_mid && g_qwen4_nax_half_midr) {
                if (!qwen4_bind_tensor(&b[3], g_qwen4_nax_half_mid, mid_count * sizeof(uint16_t), "moe mid half") ||
                    !qwen4_bind_tensor(&b[5], g_qwen4_nax_half_midr, mid_count * sizeof(uint16_t), "moe mid residual")) return 0;
                g_qwen4_nax_half_mid_for = NULL;
                g_qwen4_nax_half_mid_count = 0;
            } else {
                nax_comp = false;
            }
        }
        if (!nax_comp && !nax_fx) {
            /* the half tiles read the mid tiles' half copy of mid */
            ds4_gpu_tensor *mh = g_qwen4_nax_half_mid_for == mid && g_qwen4_nax_half_mid_count == mid_count && g_qwen4_nax_half_mid
                                 ? g_qwen4_nax_half_mid
                                 : qwen4_nax_half_operand(&g_qwen4_nax_half_mid, &g_qwen4_nax_half_mid_bytes, mid, mid_count);
            g_qwen4_nax_half_mid_for = NULL;
            g_qwen4_nax_half_mid_count = 0;
            if (!mh || !qwen4_bind_tensor(&b[3], mh, mid_count * sizeof(uint16_t), "moe mid half")) return 0;
        }
        const bool nax_tails = nax == 64u && qwen4_moe_mm_tails(weight_type, 8u);
        if (nax_tails) args.tail_base = 64u;
        const int nax_down = nax_comp ? (nax == 64u ? QWEN4_K_MOE_MM_DOWN_NAXC64 : QWEN4_K_MOE_MM_DOWN_NAXC)
                           : nax_fx ? (nax == 64u ? QWEN4_K_MOE_MM_DOWN_NAXF64 : QWEN4_K_MOE_MM_DOWN_NAXF)
                           : (nax == 64u ? QWEN4_K_MOE_MM_DOWN_NAX64 : QWEN4_K_MOE_MM_DOWN_NAX);
        const int nax_down_tail = nax_comp ? QWEN4_K_MOE_MM_DOWN_NAXC : nax_fx ? QWEN4_K_MOE_MM_DOWN_NAXF : QWEN4_K_MOE_MM_DOWN_NAX;
        if (!qwen4_dispatch(nax_down, &args, sizeof(args), b, nax_comp ? 6 : 5,
                            qwen4_moe_mm_grid((out_dim + 63u) / 64u, n_expert, tiles, args.expert_major),
                            MTLSizeMake(128, 1, 1), nax == 64u ? 16384u : 8192u)) return 0;
        if (!nax_tails) return 1;
        args.tiles_per_launch = 1;
        return qwen4_dispatch(nax_down_tail, &args, sizeof(args), b, nax_comp ? 6 : 5,
                              qwen4_moe_mm_grid((out_dim + 63u) / 64u, n_expert, 1, args.expert_major),
                              MTLSizeMake(128, 1, 1), 8192u);
    }
    if (!qwen4_dispatch(kernel, &args, sizeof(args), b, 5,
                          qwen4_moe_mm_grid((out_dim + 31u) / 32u, n_expert, tiles, args.expert_major),
                          MTLSizeMake(128, 1, 1), 0)) return 0;
    if (tails) {
        const MTLSize tail_grid = MTLSizeMake((out_dim + 31u) / 32u, n_expert, 1);
        if (!qwen4_dispatch(QWEN4_K_MOE_MM_DOWN_NT1, &args, sizeof(args), b, 5, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 2u && !qwen4_dispatch(QWEN4_K_MOE_MM_DOWN_NT2, &args, sizeof(args), b, 5, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
        if (nt > 4u && !qwen4_dispatch(QWEN4_K_MOE_MM_DOWN, &args, sizeof(args), b, 5, tail_grid, MTLSizeMake(128, 1, 1), 0)) return 0;
    }
    return 1;
}

int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
                                             uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 || !out_dim) return 0;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(w);
        if (!xbuf || !outbuf || !wbuf ||
            ds4_gpu_tensor_bytes(x) < (uint64_t)in_dim * sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < (uint64_t)out_dim * sizeof(float) ||
            ds4_gpu_tensor_bytes(w) < (uint64_t)out_dim * row_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor-weight matvec received undersized buffers\n");
            return 0;
        }
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) mv_dispatch.nsg = 8;
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(w) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor-weight matvec") ? 1 : 0;
    }
}

int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out_idx, ds4_gpu_tensor *scratch,
                               const ds4_gpu_tensor *logits, uint32_t n_vocab) {
    if (!n_vocab || n_vocab > INT32_MAX) return 0;
    const uint32_t chunks = (n_vocab + 4095u) / 4096u;
    struct { uint32_t n, finish; } args = {n_vocab, 0};
    qwen4_bind b[3];
    if (!qwen4_bind_tensor(&b[0], logits, (uint64_t)n_vocab * sizeof(float), "argmax logits") ||
        !qwen4_bind_tensor(&b[1], scratch, (uint64_t)chunks * 8u, "argmax partials") ||
        !qwen4_bind_tensor(&b[2], out_idx, sizeof(int32_t), "argmax index")) return 0;
    if (!qwen4_dispatch(QWEN4_K_ARGMAX, &args, sizeof(args), b, 3,
                       MTLSizeMake(chunks, 1, 1), MTLSizeMake(256, 1, 1), 0)) return 0;
    args.n = chunks;
    args.finish = 1;
    return qwen4_dispatch(QWEN4_K_ARGMAX, &args, sizeof(args), b, 3,
                          MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_mtp_stage_tensor(
        ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *R,
        const void *model_map, uint64_t model_size, uint64_t g_e_offset, uint64_t g_h_offset,
        uint32_t n_embd, uint32_t n_hc, float eps) {
    struct { uint32_t n_embd, n_hc, pad0; float eps; } args = { n_embd, n_hc, 0u, eps };
    qwen4_bind b[5];
    if (n_embd == 0 || n_hc == 0 ||
        !qwen4_bind_tensor(&b[0], e, (uint64_t)n_embd * sizeof(float), "mtp embedding") ||
        !qwen4_bind_tensor(&b[1], R, (uint64_t)n_embd * n_hc * sizeof(float), "mtp streams") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, g_e_offset, (uint64_t)n_embd * sizeof(float), "mtp enorm") ||
        !qwen4_bind_weight(&b[3], model_map, model_size, g_h_offset, (uint64_t)n_embd * n_hc * sizeof(float),
                           "mtp hnorm") ||
        !qwen4_bind_tensor(&b[4], cat, (uint64_t)(n_hc + 1u) * 2u * n_embd * sizeof(float), "mtp concat")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_MTP_STAGE, &args, sizeof(args), b, 5,
                          MTLSizeMake(n_hc + 1u, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_mtp_combine_tensor(
        ds4_gpu_tensor *R_out, const ds4_gpu_tensor *proj, uint32_t n_embd, uint32_t n_hc) {
    struct { uint32_t n_embd, n_hc; } args = { n_embd, n_hc };
    qwen4_bind b[2];
    if (n_embd == 0 || n_hc == 0 ||
        !qwen4_bind_tensor(&b[0], proj, (uint64_t)(n_hc + 1u) * n_embd * sizeof(float), "mtp projection") ||
        !qwen4_bind_tensor(&b[1], R_out, (uint64_t)n_embd * n_hc * sizeof(float), "mtp residual")) {
        return 0;
    }
    const uint32_t n = n_embd * n_hc;
    return qwen4_dispatch(QWEN4_K_MTP_COMBINE, &args, sizeof(args), b, 2,
                          MTLSizeMake((n + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_gdn_front_tensor(
        ds4_gpu_tensor *qkv, ds4_gpu_tensor *state, const ds4_gpu_tensor *mixed,
        ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *model_map, uint64_t model_size, uint64_t conv_offset,
        uint64_t alpha_offset, uint64_t beta_offset, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_k_head, uint32_t n_v_head, uint32_t head_dim,
        uint32_t conv_kernel, uint32_t in_dim, ds4_gpu_tensor *snap_state, uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state, uint32_t snap2_tok) {
    const uint32_t row_bytes = qwen4_expert_row_bytes(weight_type, in_dim);
    struct { uint32_t n_tokens, n_k_head, n_v_head, head_dim, conv_kernel, weight_type, in_dim, row_bytes,
                      snap_tok, snap2_tok, pad1, pad2; } args =
        { n_tokens, n_k_head, n_v_head, head_dim, conv_kernel, weight_type, in_dim, row_bytes,
          snap_state ? snap_tok : UINT32_MAX, snap2_state ? snap2_tok : UINT32_MAX, 0, 0 };
    const uint64_t conv_dim = 2ull * n_k_head * head_dim + (uint64_t)n_v_head * head_dim;
    const uint64_t state_bytes = (uint64_t)(conv_kernel - 1) * conv_dim * sizeof(float);
    const uint64_t proj_bytes = (uint64_t)n_v_head * row_bytes;
    const uint64_t vhead_bytes = (uint64_t)n_v_head * sizeof(float);
    qwen4_bind b[12];
    if (n_tokens == 0 || n_k_head == 0 || head_dim % 32 != 0 || n_v_head % n_k_head != 0 ||
        conv_kernel < 2 || conv_kernel > 4 || row_bytes == 0 ||
        !qwen4_bind_tensor(&b[0], qkv, (uint64_t)n_tokens * conv_dim * sizeof(float), "gdn front qkv") ||
        !qwen4_bind_tensor(&b[1], state, state_bytes, "gdn front conv state") ||
        !qwen4_bind_weight(&b[2], model_map, model_size, conv_offset, conv_dim * conv_kernel * sizeof(float),
                           "gdn front conv weight") ||
        !qwen4_bind_tensor(&b[3], mixed, (uint64_t)n_tokens * in_dim * sizeof(float), "gdn front input") ||
        !qwen4_bind_weight(&b[4], model_map, model_size, alpha_offset, proj_bytes, "gdn front alpha") ||
        !qwen4_bind_weight(&b[5], model_map, model_size, beta_offset, proj_bytes, "gdn front beta") ||
        !qwen4_bind_weight(&b[6], model_map, model_size, ssm_a_offset, vhead_bytes, "gdn front ssm_a") ||
        !qwen4_bind_weight(&b[7], model_map, model_size, dt_bias_offset, vhead_bytes, "gdn front dt_bias") ||
        !qwen4_bind_tensor(&b[8], ga, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn front decay") ||
        !qwen4_bind_tensor(&b[9], gb, (uint64_t)n_tokens * n_v_head * sizeof(float), "gdn front beta out")) {
        return 0;
    }
    if (snap_state) {
        if (!qwen4_bind_tensor(&b[10], snap_state, state_bytes, "gdn front snapshot")) return 0;
    } else {
        b[10] = b[1];
    }
    if (snap2_state) {
        if (!qwen4_bind_tensor(&b[11], snap2_state, state_bytes, "gdn front snapshot 2")) return 0;
    } else {
        b[11] = b[1];
    }
    /* One threadgroup per key head: simdgroups 0/1 normalize, the rest take
     * the alpha/beta rows, and every channel's conv stays on one thread, so
     * the thread count only spreads the conv wider.  M5 measured 1024. */
    const uint64_t nth = ds4_gpu_env_u64("DS4_QWEN4_GDN_FRONT_THREADS",
                                         ds4_gpu_device_is_m5_apple_silicon() ? 1024u : 256u, 96u, 1024u);
    return qwen4_dispatch(QWEN4_K_GDN_FRONT, &args, sizeof(args), b, 12,
                          MTLSizeMake(n_k_head, 1, 1), MTLSizeMake((NSUInteger)(nth / 32u * 32u), 1, 1), 0);
}

int ds4_gpu_qwen4_multi_gemv_tensor(
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t in_dim, uint32_t n_out,
        ds4_gpu_tensor *const *outs, const void *model_map, uint64_t model_size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *out_rows) {
    struct { uint32_t n_tokens, in_dim, n_out, pad0, rows[4], types[4], row_bytes[4]; } args = {0};
    qwen4_bind b[9];
    if (!x || n_tokens == 0 || in_dim == 0 || (in_dim % 32) != 0 || n_out == 0 || n_out > 4 ||
        !qwen4_bind_tensor(&b[0], x, (uint64_t)n_tokens * in_dim * sizeof(float), "multi gemv input")) {
        return 0;
    }
    args.n_tokens = n_tokens;
    args.in_dim = in_dim;
    args.n_out = n_out;
    uint32_t total = 0;
    for (uint32_t i = 0; i < 4; i++) {
        if (i < n_out) {
            const uint32_t rb = qwen4_expert_row_bytes(types[i], in_dim);
            if (rb == 0 || out_rows[i] == 0 ||
                !qwen4_bind_weight(&b[1 + i], model_map, model_size, offsets[i], (uint64_t)rb * out_rows[i],
                                   "multi gemv weight") ||
                !qwen4_bind_tensor(&b[5 + i], outs[i], (uint64_t)n_tokens * out_rows[i] * sizeof(float),
                                   "multi gemv output")) {
                return 0;
            }
            args.rows[i] = out_rows[i];
            args.types[i] = types[i];
            args.row_bytes[i] = rb;
            total += out_rows[i];
        } else {
            b[1 + i] = b[1];
            b[5 + i] = b[5];
        }
    }
    return qwen4_dispatch(QWEN4_K_MULTI_GEMV, &args, sizeof(args), b, 9,
                          MTLSizeMake((total + 7) / 8, n_tokens, 1), MTLSizeMake(128, 1, 1), 0);
}

static ds4_gpu_tensor *g_qwen4_dense_mm_partials;
static uint64_t g_qwen4_dense_mm_partials_bytes;

static void qwen4_batch_release_scratch(void) {
    ds4_gpu_tensor_free(g_qwen4_dense_mm_partials);
    ds4_gpu_tensor_free(g_qwen4_gdn_slots);
    g_qwen4_dense_mm_partials = g_qwen4_gdn_slots = NULL;
    g_qwen4_dense_mm_partials_bytes = 0;
    for (unsigned i = 0; i < QWEN4_K_COUNT; i++) g_qwen4_pipelines[i] = nil;
}

/* Scratch for the k-split planes, allocated once and generously: the
 * command buffer holds unretained references, so a scratch replaced while
 * earlier dispatches of the same buffer still point at it is read after its
 * release.  Growing it per request was what made wide splits look wrong. */
static bool qwen4_dense_mm_partials_ensure(uint64_t bytes) {
    if (g_qwen4_dense_mm_partials && g_qwen4_dense_mm_partials_bytes >= bytes) return true;
    if (g_qwen4_dense_mm_partials) {
        fprintf(stderr, "ds4: Qwen3.8 k-split scratch of %.1f MiB cannot grow to %.1f MiB mid-batch\n",
                ds4_gpu_mib(g_qwen4_dense_mm_partials_bytes), ds4_gpu_mib(bytes));
        return false;
    }
    const uint64_t grow = bytes > (64u << 20) ? bytes : (64u << 20);
    g_qwen4_dense_mm_partials = ds4_gpu_tensor_alloc(grow);
    if (!g_qwen4_dense_mm_partials) return false;
    g_qwen4_dense_mm_partials_bytes = grow;
    return true;
}

int ds4_gpu_qwen4_dense_mm_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t weight_type,
        uint32_t n_tokens, uint32_t in_dim, uint32_t out_rows) {
    const uint32_t row_bytes = weight_type == 0u ? in_dim * 4u : weight_type == 1u ? in_dim * 2u :
                               qwen4_expert_row_bytes(weight_type, in_dim);
    struct { uint32_t n_tokens, in_dim, out_rows, weight_type, row_bytes, n_split, pad1, pad2; } args =
        { n_tokens, in_dim, out_rows, weight_type, row_bytes, 1, 0, 0 };
    qwen4_bind b[3];
    if (n_tokens == 0 || row_bytes == 0 || (weight_type != 0u && weight_type != 1u && weight_type != 8u) ||
        (weight_type == 8u && (in_dim % 32) != 0) || (in_dim % 8) != 0 || out_rows == 0) {
        fprintf(stderr, "ds4: Qwen3.8 dense mm rejected type %u in %u out %u tokens %u\n",
                weight_type, in_dim, out_rows, n_tokens);
        return 0;
    }
    if (!qwen4_bind_weight(&b[0], model_map, model_size, weight_offset, (uint64_t)row_bytes * out_rows,
                           "dense mm weight") ||
        !qwen4_bind_tensor(&b[1], x, (uint64_t)n_tokens * in_dim * sizeof(float), "dense mm input") ||
        !qwen4_bind_tensor(&b[2], out, (uint64_t)n_tokens * out_rows * sizeof(float), "dense mm output")) {
        return 0;
    }
    /* A projection whose output is narrow gives this grid only a few
     * threadgroups when the batch is a decode step; split k until the
     * machine has work, then sum the planes.  A prefill chunk's token tiles
     * already fill it, and its planes would not fit the scratch anyway.
     * The split is chosen so the partials stay small and every threadgroup
     * still walks enough k to amortize its staging. */
    const uint32_t tiles = (out_rows + 31u) / 32u;
    const uint32_t threadgroups = tiles * ((n_tokens + 31u) / 32u);
    uint32_t n_split = 1u;
    if (!ds4_gpu_env_u64("DS4_QWEN4_NO_DENSE_MM_KSPLIT", 0u, 0u, 1u)) {
        const uint32_t nk = (in_dim + 31u) / 32u;
        const uint32_t target = 128u;
        const uint32_t want = threadgroups >= target ? 1u : (target + threadgroups - 1u) / threadgroups;
        n_split = want > nk / 4u ? (nk / 4u ? nk / 4u : 1u) : want;
        if (n_split > 64u) n_split = 64u;
    }
    args.n_split = n_split;
    if (n_split <= 1u) {
        return qwen4_dispatch(QWEN4_K_DENSE_MM, &args, sizeof(args), b, 3,
                              MTLSizeMake(tiles, (n_tokens + 31) / 32, 1), MTLSizeMake(128, 1, 1), 0);
    }

    const uint64_t plane = (uint64_t)n_tokens * out_rows * sizeof(float);
    if (!qwen4_dense_mm_partials_ensure(plane * n_split)) return 0;
    qwen4_bind bp[3];
    bp[0] = b[0];
    bp[1] = b[1];
    if (!qwen4_bind_tensor(&bp[2], g_qwen4_dense_mm_partials, plane * n_split, "dense mm partials")) {
        return 0;
    }
    if (!qwen4_dispatch(QWEN4_K_DENSE_MM, &args, sizeof(args), bp, 3,
                        MTLSizeMake(tiles, (n_tokens + 31) / 32, n_split), MTLSizeMake(128, 1, 1), 0)) {
        return 0;
    }
    qwen4_bind br[2];
    br[0] = bp[2];
    br[1] = b[2];
    const uint64_t n = (uint64_t)n_tokens * out_rows;
    return qwen4_dispatch(QWEN4_K_DENSE_MM_REDUCE, &args, sizeof(args), br, 2,
                          MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1), MTLSizeMake(256, 1, 1), 0);
}

/* Decode batch of 8 or 16 rows against a Q8 matrix (kernel_qwen4_batch_mm_q8):
 * the k-split gives narrow projections enough simdgroups, the partial planes
 * are summed by kernel_qwen4_dense_mm_reduce. */
int ds4_gpu_qwen4_batch_mm_q8_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t in_dim, uint32_t out_rows) {
    if ((n_tokens != 8u && n_tokens != 16u) || (in_dim % 32u) != 0 || (out_rows % 32u) != 0 || in_dim == 0) return 0;
    const uint32_t row_bytes = (in_dim / 32u) * 34u;
    const uint32_t nk = in_dim / 32u;
    const uint32_t simdgroups = out_rows / 32u;
    /* enough simdgroups to cover the machine several times over, no split
     * shorter than four blocks */
    const uint32_t target = 640u;
    uint32_t n_split = simdgroups >= target ? 1u : (target + simdgroups - 1u) / simdgroups;
    if (n_split > nk / 4u) n_split = nk / 4u ? nk / 4u : 1u;
    if (n_split > 16u) n_split = 16u;
    struct { uint32_t n_tokens, in_dim, out_rows, weight_type, row_bytes, n_split, pad1, pad2; } args =
        { n_tokens, in_dim, out_rows, 8u, row_bytes, n_split, 0, 0 };
    qwen4_bind b[3];
    if (!qwen4_bind_weight(&b[0], model_map, model_size, weight_offset, (uint64_t)row_bytes * out_rows, "batch mm weight") ||
        !qwen4_bind_tensor(&b[1], x, (uint64_t)n_tokens * in_dim * sizeof(float), "batch mm input") ||
        !qwen4_bind_tensor(&b[2], out, (uint64_t)n_tokens * out_rows * sizeof(float), "batch mm output")) {
        return 0;
    }
    const MTLSize grid = MTLSizeMake((out_rows + 127u) / 128u, 1, n_split);
    const int kernel = n_tokens == 16u ? QWEN4_K_BATCH_MM_Q8_T2 : QWEN4_K_BATCH_MM_Q8_T1;
    if (n_split <= 1u) {
        return qwen4_dispatch(kernel, &args, sizeof(args), b, 3, grid, MTLSizeMake(128, 1, 1), 0);
    }
    const uint64_t plane = (uint64_t)n_tokens * out_rows * sizeof(float);
    if (!qwen4_dense_mm_partials_ensure(plane * n_split)) return 0;
    qwen4_bind bp[3] = { b[0], b[1] };
    if (!qwen4_bind_tensor(&bp[2], g_qwen4_dense_mm_partials, plane * n_split, "batch mm partials")) return 0;
    if (!qwen4_dispatch(kernel, &args, sizeof(args), bp, 3, grid, MTLSizeMake(128, 1, 1), 0)) return 0;
    qwen4_bind br[2] = { bp[2], b[2] };
    const uint64_t n = (uint64_t)n_tokens * out_rows;
    return qwen4_dispatch(QWEN4_K_DENSE_MM_REDUCE, &args, sizeof(args), br, 2,
                          MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1), MTLSizeMake(256, 1, 1), 0);
}

/* Vision tower (src/metal/qwen4_vision.metal). */
typedef struct { uint32_t rows, width, grid_w, n_head, head_dim, mode; float eps, scale; } qwen4_vis_args;

static int qwen4_vis_rows(int kernel, const qwen4_vis_args *a, qwen4_bind *b, int n_bind, uint32_t rows, uint32_t y) {
    const int ok = qwen4_dispatch(kernel, a, sizeof(*a), b, n_bind, MTLSizeMake(rows, y, 1),
                                  MTLSizeMake(kernel == QWEN4_K_VIS_ATTENTION ? 32 : 256, 1, 1), 0);
    if (!ok) fprintf(stderr, "ds4: Qwen3.8 vision kernel %s failed\n", qwen4_kernel_names[kernel]);
    return ok;
}

static int qwen4_vis_elems(int kernel, const qwen4_vis_args *a, qwen4_bind *b, int n_bind) {
    const uint64_t n = (uint64_t)a->rows * a->width;
    const int ok = qwen4_dispatch(kernel, a, sizeof(*a), b, n_bind, MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1),
                                  MTLSizeMake(256, 1, 1), 0);
    if (!ok) fprintf(stderr, "ds4: Qwen3.8 vision kernel %s failed\n", qwen4_kernel_names[kernel]);
    return ok;
}

static bool qwen4_vis_w(qwen4_bind *b, const void *map, uint64_t size, uint64_t off, uint64_t n, const char *what) {
    return qwen4_bind_weight(b, map, size, off, n * sizeof(float), what);
}

static bool qwen4_vis_t(qwen4_bind *b, const ds4_gpu_tensor *t, uint64_t n, const char *what) {
    return qwen4_bind_tensor(b, t, n * sizeof(float), what);
}

/* Native vision owns the workspace, 27-block loop and merger. Keep only
 * primitive shader packets here, where Metal buffers and pipeline IDs live. */
#include "../engine/qwen4/vision_api.h"

int ds4_native_qwen4_vision_patch(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *first, const ds4_gpu_tensor *second,
        const ds4_gpu_tensor *position, const void *map, uint64_t size,
        uint64_t bias, uint32_t rows, uint32_t width) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width };
        qwen4_bind bindings[5];
        const uint64_t count = (uint64_t)rows * width;
        return qwen4_vis_t(&bindings[0], first, count, "vision patch a0") &&
               qwen4_vis_t(&bindings[1], second, count, "vision patch a1") &&
               qwen4_vis_w(&bindings[2], map, size, bias, width, "vision patch bias") &&
               qwen4_vis_t(&bindings[3], position, count, "vision pos") &&
               qwen4_vis_t(&bindings[4], out, count, "vision x") &&
               qwen4_vis_elems(QWEN4_K_VIS_PATCH_FINISH, &args, bindings, 5);
    }
}
int ds4_native_qwen4_vision_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
        const void *map, uint64_t size, uint64_t weight, uint64_t bias,
        uint32_t rows, uint32_t width, float epsilon) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width, .eps = epsilon };
        qwen4_bind bindings[4];
        const uint64_t count = (uint64_t)rows * width;
        return qwen4_vis_t(&bindings[0], input, count, "vision norm input") &&
               qwen4_vis_w(&bindings[1], map, size, weight, width, "vision norm weight") &&
               qwen4_vis_w(&bindings[2], map, size, bias, width, "vision norm bias") &&
               qwen4_vis_t(&bindings[3], out, count, "vision norm output") &&
               qwen4_vis_rows(QWEN4_K_VIS_LAYERNORM, &args, bindings, 4, rows, 1);
    }
}
int ds4_native_qwen4_vision_qkv(ds4_gpu_tensor *q, ds4_gpu_tensor *k,
        ds4_gpu_tensor *v, const ds4_gpu_tensor *qkv, const void *map,
        uint64_t size, uint64_t bias, uint32_t rows, uint32_t width,
        uint32_t grid_width, uint32_t heads, uint32_t head_width) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width,
            .grid_w = grid_width, .n_head = heads, .head_dim = head_width };
        qwen4_bind bindings[5];
        const uint64_t count = (uint64_t)rows * width;
        return qwen4_vis_t(&bindings[0], qkv, 3u * count, "vision qkv") &&
               qwen4_vis_w(&bindings[1], map, size, bias, 3u * width, "vision qkv bias") &&
               qwen4_vis_t(&bindings[2], q, count, "vision q") &&
               qwen4_vis_t(&bindings[3], k, count, "vision k") &&
               qwen4_vis_t(&bindings[4], v, count, "vision v") &&
               qwen4_vis_rows(QWEN4_K_VIS_QKV_ROPE, &args, bindings, 5, rows, 1);
    }
}
int ds4_native_qwen4_vision_attention(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
        uint32_t rows, uint32_t width, uint32_t heads, uint32_t head_width,
        float scale) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width,
            .n_head = heads, .head_dim = head_width, .scale = scale };
        qwen4_bind bindings[4];
        const uint64_t count = (uint64_t)rows * width;
        return qwen4_vis_t(&bindings[0], q, count, "vision q") &&
               qwen4_vis_t(&bindings[1], k, count, "vision k") &&
               qwen4_vis_t(&bindings[2], v, count, "vision v") &&
               qwen4_vis_t(&bindings[3], out, count, "vision attention") &&
               qwen4_vis_rows(QWEN4_K_VIS_ATTENTION, &args, bindings, 4, rows, heads);
    }
}
int ds4_native_qwen4_vision_residual(ds4_gpu_tensor *residual,
        const ds4_gpu_tensor *input, const void *map, uint64_t size,
        uint64_t bias, uint32_t rows, uint32_t width) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width };
        qwen4_bind bindings[3];
        const uint64_t count = (uint64_t)rows * width;
        return qwen4_vis_t(&bindings[0], residual, count, "vision residual") &&
               qwen4_vis_t(&bindings[1], input, count, "vision residual input") &&
               qwen4_vis_w(&bindings[2], map, size, bias, width, "vision residual bias") &&
               qwen4_vis_elems(QWEN4_K_VIS_BIAS_RESIDUAL, &args, bindings, 3);
    }
}
int ds4_native_qwen4_vision_bias_activation(ds4_gpu_tensor *tensor,
        const void *map, uint64_t size, uint64_t bias,
        uint32_t rows, uint32_t width, uint32_t mode) {
    @autoreleasepool {
        qwen4_vis_args args = { .rows = rows, .width = width, .mode = mode };
        qwen4_bind bindings[2];
        return qwen4_vis_t(&bindings[0], tensor, (uint64_t)rows * width, "vision activation") &&
               qwen4_vis_w(&bindings[1], map, size, bias, width, "vision activation bias") &&
               qwen4_vis_elems(QWEN4_K_VIS_BIAS_ACT, &args, bindings, 2);
    }
}

int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *lo_act, const ds4_gpu_tensor *lo,
                                   uint32_t n_tokens, uint32_t n_hc, uint32_t n_rank) {
    struct { uint32_t n_tokens, n_embd, n_hc, n_rank; } args = { n_tokens, 0, n_hc, n_rank };
    qwen4_bind b[2];
    const uint64_t n = (uint64_t)n_tokens * n_rank;
    if (n == 0 || n_hc == 0 ||
        !qwen4_bind_tensor(&b[0], lo, n * sizeof(float), "hc lo") ||
        !qwen4_bind_tensor(&b[1], lo_act, n * sizeof(float), "hc lo act")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_HC_LO_ACT, &args, sizeof(args), b, 2,
                          MTLSizeMake((n + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *mixed, const ds4_gpu_tensor *u, const ds4_gpu_tensor *xn,
                                     uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    struct { uint32_t n_tokens, n_embd, n_hc, n_rank; } args = { n_tokens, n_embd, n_hc, 0 };
    qwen4_bind b[3];
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    if (n_tokens == 0 || n_embd == 0 || n_hc == 0 ||
        !qwen4_bind_tensor(&b[0], u, (uint64_t)n_tokens * dim * sizeof(float), "hc u") ||
        !qwen4_bind_tensor(&b[1], xn, (uint64_t)n_tokens * dim * sizeof(float), "hc xn") ||
        !qwen4_bind_tensor(&b[2], mixed, (uint64_t)n_tokens * n_embd * sizeof(float), "hc mixed")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_HC_MIX_ROWS, &args, sizeof(args), b, 3,
                          MTLSizeMake((n_embd + 255) / 256, n_tokens, 1), MTLSizeMake(256, 1, 1), 0);
}

#include "resources.inc.m"

#include "bridge.inc.m"

#include "model_views.inc.m"

#include "exact_cache.inc.m"

#include "index_selection.inc.m"

#include "dense_matmul.inc.m"

#include "attention_decode.inc.m"

#include "attention_batch.inc.m"

#include "routed_batch.inc.m"

#include "routed_single.inc.m"

#include "expert_cache.inc.m"

#include "expert_tables.inc.m"

#include "expert_load.inc.m"

#include "attention_output.inc.m"

#include "rope.inc.m"


#include "compressor.inc.m"


#include "scheduling.inc.m"


#include "embedding.inc.m"


#include "hc_control.inc.m"


#include "router_shared.inc.m"


#include "router.inc.m"


#include "supported_primitives.inc.m"


#include "tp_control.inc.m"

#include "primitives.inc.m"


#include "tp_prefetch.inc.m"

/* Test-only observation of the real two-level scheduler. No fabricated state:
 * public start owns admission/resources, and public finish owns retirement. */
int ds4_gpu_test_parallel_ffn_stages(const ds4_gpu_tensor *routed_mid) {
    if (!g_batch_cb || !routed_mid) return 0;
    id<MTLBuffer> buffer = ds4_gpu_tensor_buffer(routed_mid);
    return buffer && ds4_gpu_parallel_q8_matvec_encode_pending(g_batch_cb, buffer) &&
           ds4_gpu_parallel_ffn_encode_second_stage(g_batch_cb);
}

/* Native resource lifecycle policy shares these ARC driver leaves. */
#include "model_lifecycle.inc.m"

#include "bootstrap.inc.m"
