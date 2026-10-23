#ifndef DS4_METAL_SCHEDULING_H
#define DS4_METAL_SCHEDULING_H
#include "attention_decode.h"
#include "dense_matmul.h"
#include "routed_batch.h"
#include <stdbool.h>
#include <stddef.h>
#ifdef __OBJC__
#define DS4_COMMAND __strong id<MTLCommandBuffer>
#define DS4_ENCODER __strong id<MTLComputeCommandEncoder>
#define DS4_BUFFER __strong id<MTLBuffer>
#define DS4_PIPELINE __strong id<MTLComputePipelineState>
#define DS4_PENDING __strong NSMutableArray<id<MTLCommandBuffer>> *
#else
typedef void *ds4_metal_resource_ref;
#define DS4_COMMAND ds4_metal_resource_ref
#define DS4_ENCODER ds4_metal_resource_ref
#define DS4_BUFFER ds4_metal_resource_ref
#define DS4_PIPELINE ds4_metal_resource_ref
#define DS4_PENDING ds4_metal_resource_ref
#endif

/* The actual process command state, not a snapshot. Native code owns scalar
 * policy. Only the ARC leaves assign or retire its strong object fields. */
typedef struct {
    DS4_COMMAND batch;
    DS4_ENCODER encoder;
    bool concurrent, has_work;
    DS4_PENDING pending;
    double created_ms;
    bool static_head_norm;
} ds4_metal_commands;

/* GPU-decided complementary shared-expert lanes; the host binds full weight
 * ranges and IDs, while each rank's shader trims its own lane range. */
typedef struct {
    int32_t tp_rank, tp_world, n_expert, n_expert_used;
    int32_t shared_dim, lane_granule, shift_q16;
} ds4_metal_shared_split_args;
typedef struct {
    DS4_PIPELINE down_pipeline, gate_pipeline;
    DS4_BUFFER down_weight, down_x, down_out;
    DS4_BUFFER gate_weight, up_weight, gate_x, gate_out, up_out, mid_out, ids;
    uint64_t down_weight_offset, down_x_offset, down_out_offset;
    uint64_t gate_weight_offset, up_weight_offset, gate_x_offset;
    uint64_t gate_out_offset, up_out_offset, mid_out_offset, ids_offset;
    ds4_metal_mv_args down_args, gate_args;
    float clamp;
    uint64_t nsg, nr0, shared_bytes;
    int mode, stage; /* mode2: gate/up followed by down */
    bool pending, encoded, split;
    ds4_metal_shared_split_args split_args;
} ds4_metal_parallel_ffn;
typedef struct {
    int32_t head_dim, n_rot, raw_row;
} ds4_metal_kv_store_args;
typedef struct {
    int pending;
    DS4_BUFFER kv, kv_out, kv_w, q, q_out, q_w, raw;
    uint64_t kv_off, kv_out_off, kv_w_off, q_off, q_out_off, q_w_off, raw_off;
    ds4_metal_qkv_rms_args nargs;
    ds4_metal_attention_rope_args rope;
    ds4_metal_kv_store_args store;
    uint64_t norm_threads;
} ds4_metal_kv_task;
typedef struct {
    ds4_metal_commands commands;
    ds4_metal_parallel_ffn parallel;
    ds4_metal_kv_task kv;
    int defer_kv;
} ds4_metal_scheduling;
#undef DS4_COMMAND
#undef DS4_ENCODER
#undef DS4_BUFFER
#undef DS4_PIPELINE
#undef DS4_PENDING

/* Fixed ARC/driver actions. Every returned opaque object is +1; drop balances
 * it. Installing a batch/encoder/task retains independently before that drop. */
ds4_metal_scheduling *ds4_metal_scheduling_state(void);
void *ds4_metal_schedule_retain(void *);
void ds4_metal_schedule_drop(void *);
void *ds4_metal_schedule_new_command(bool unretained);
void ds4_metal_schedule_set_batch(void *);
void ds4_metal_schedule_set_encoder(void *);
void *ds4_metal_schedule_new_encoder(void *, bool concurrent, bool timeline, uintptr_t caller);
void ds4_metal_schedule_end_encoder(void *);
bool ds4_metal_schedule_is_concurrent(void *);
bool ds4_metal_schedule_timeline_active(void);
void ds4_metal_schedule_timeline_attach(void *);
void ds4_metal_schedule_commit(void *);
uint64_t ds4_metal_schedule_pending_count(void);
void *ds4_metal_schedule_pending_at(uint64_t);
void ds4_metal_schedule_pending_append(void *);
void ds4_metal_schedule_pending_clear(void);
int ds4_metal_schedule_wait(void *, const char *);
void ds4_metal_schedule_invalidate_maps(void);
void ds4_metal_schedule_clear_transients(bool profile);
void ds4_metal_schedule_evict(const char *);
void ds4_metal_schedule_finish_trace(void *, const char *, double commit_ms,
    double created_ms, uintptr_t caller);
enum { DS4_SCHEDULE_OWNED_CREATED, DS4_SCHEDULE_OWNED_DONE,
    DS4_SCHEDULE_BATCH_CREATED, DS4_SCHEDULE_BATCH_COMMITTED,
    DS4_SCHEDULE_PENDING_DONE, DS4_SCHEDULE_BATCH_TO_OWNED };
void ds4_metal_schedule_cache_event(uint32_t);
enum { DS4_SCHEDULE_STREAMING = 1, DS4_SCHEDULE_PRE_M5 = 2 };
uint32_t ds4_metal_schedule_facts(void);
uint64_t ds4_metal_schedule_pipeline_threads(const ds4_metal_hc_pipeline *);
void ds4_metal_schedule_parallel_install(const ds4_metal_attention_view *buffers,
    const ds4_metal_hc_pipeline *gate, const ds4_metal_hc_pipeline *down);
void ds4_metal_schedule_parallel_retire(void);
void ds4_metal_schedule_parallel_gate(void *encoder, uint64_t groups);
void ds4_metal_schedule_parallel_down(void *encoder, uint64_t groups);
void ds4_metal_schedule_parallel_barrier(void *encoder, void *routed_mid);
void ds4_metal_schedule_kv_install(const ds4_metal_attention_view *buffers);
typedef struct {
    ds4_metal_qkv_rms_args norm;
    ds4_metal_attention_rope_args rope;
    ds4_metal_kv_store_args store;
    ds4_metal_attention_view buffers[7]; /* q,qW,qOut,kv,kvW,kvOut,raw */
    uint64_t threads;
    uint32_t tasks, groups_y;
} ds4_metal_kv_launch;
void ds4_metal_schedule_kv_encode(void *encoder, const ds4_metal_hc_pipeline *,
    const ds4_metal_kv_launch *);

/* Sole facade declarations. The retained-token acquire/release ABI is shared
 * by existing Norm/F32/Index controllers and the thin remaining C encoders. */
void *ds4_native_metal_command_acquire(int *owned);
void *ds4_native_metal_new_command(void);
int ds4_native_metal_command_release(void *, int owned, int ok, const char *label, uintptr_t caller);
int ds4_native_metal_finish_command(void *, int owned, const char *label, uintptr_t caller);
void *ds4_native_metal_compute_encoder(void *, uintptr_t caller);
void ds4_native_metal_end_encoder(void *, void *);
void ds4_native_metal_close_encoder(void);
int ds4_native_metal_wait_pending(const char *);
int ds4_native_metal_begin_commands(void);
int ds4_native_metal_flush_encoder(void);
int ds4_native_metal_flush_commands(void);
int ds4_native_metal_end_commands(void);
int ds4_native_metal_synchronize(void);
int ds4_native_metal_commands_active(void);
void ds4_native_metal_parallel_reset(bool);
int ds4_native_metal_parallel_finish(void);
int ds4_native_metal_parallel_first(void *, void *);
int ds4_native_metal_parallel_second(void *);
void ds4_native_metal_defer_kv_next(void);
int ds4_native_metal_kv_flush(void);
int ds4_native_metal_kv_begin(void);
void ds4_native_metal_kv_end(void);

/* Whole public start/producer arguments borrow the actual Tensor API; no host
 * graph, model, session or worker layout enters this owner. */
typedef struct {
    ds4_gpu_tensor *gate, *up, *mid, *out;
    const void *map;
    uint64_t size, gate_offset, up_offset, down_offset;
    uint32_t model_dim, shared_dim, lane_offset, lane_count;
    const ds4_gpu_tensor *x, *selected;
    float clamp;
    bool split;
    uint32_t tp_rank, tp_world, experts, used, shift_q16;
} ds4_metal_parallel_request;
int ds4_native_metal_parallel_start(const ds4_metal_parallel_request *);
typedef struct {
    ds4_gpu_tensor *q_out;
    const ds4_gpu_tensor *q;
    const void *map;
    uint64_t size, q_weight_offset;
    uint32_t q_n;
    ds4_gpu_tensor *kv_out;
    const ds4_gpu_tensor *kv;
    uint64_t kv_weight_offset;
    uint32_t kv_n;
    ds4_gpu_tensor *raw;
    uint64_t raw_cap;
    uint32_t raw_row, n_rot, pos0, n_ctx_orig;
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps;
} ds4_metal_fused_kv_request;
int ds4_native_metal_fused_kv(const ds4_metal_fused_kv_request *);
_Static_assert(sizeof(ds4_metal_shared_split_args) == 28, "shared split packet");
_Static_assert(sizeof(ds4_metal_kv_store_args) == 12, "KV store packet");
#endif
