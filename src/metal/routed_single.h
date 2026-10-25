#ifndef DS4_METAL_ROUTED_SINGLE_H
#define DS4_METAL_ROUTED_SINGLE_H
#include <errno.h>
#include "routed_batch.h"

/* Three original shader packets, declared once for C and native callers. */
typedef struct {
    uint32_t expert_base;
    uint32_t expert_count;
    uint32_t accumulate;
    uint32_t pad0;
} ds4_gpu_moe_expert_group_args;

typedef struct {
    uint64_t expert_bytes;
    uint32_t group_size;
    uint32_t n_slots;
} ds4_gpu_q4_gather_slots6_args;

typedef struct {
    uint32_t active_mask;
    uint32_t accumulate;
} ds4_gpu_stream_expert_split_args;
_Static_assert(sizeof(ds4_gpu_moe_expert_group_args) == 16, "Metal expert group packet");
_Static_assert(sizeof(ds4_gpu_q4_gather_slots6_args) == 16, "Metal gather slot packet");
_Static_assert(sizeof(ds4_gpu_stream_expert_split_args) == 8, "Metal expert split packet");
enum {
    DS4_SINGLE_MASK_PAIR,
    DS4_SINGLE_MASK_SUM,
    DS4_SINGLE_IQ2_PACK2,
    DS4_SINGLE_IQ2_SUM,
    DS4_SINGLE_MX_FIXED_PAIR,
    DS4_SINGLE_MX_STATIC_PAIR,
    DS4_SINGLE_MX_NSG1_PAIR,
    DS4_SINGLE_MX_TG_PAIR,
    DS4_SINGLE_MX_FULL_SUM,
    DS4_SINGLE_MX_STATIC_SUM,
    DS4_SINGLE_MX_FIXED_SUM,
    DS4_SINGLE_MX_NSG1_SUM,
    DS4_SINGLE_MX_TG_SUM,
    DS4_SINGLE_SLOTS_IQ2_PAIR,
    DS4_SINGLE_SLOTS_Q2_SUM,
    DS4_SINGLE_Q4_GATHER,
    DS4_SINGLE_VALIDATE,
    DS4_SINGLE_PIPELINE_COUNT,
};
typedef struct {
    bool pending;
    int32_t mode;
    int32_t stage;
} ds4_metal_single_parallel;
typedef struct {
    ds4_metal_routed_facts batch;
    bool parallel_pending;
    int32_t parallel_mode;
    int32_t parallel_stage;
    const ds4_metal_hc_pipeline *cached[DS4_SINGLE_PIPELINE_COUNT];
} ds4_metal_single_facts;
typedef struct ds4_metal_single_resources ds4_metal_single_resources;
typedef struct {
    uint64_t hits, misses, wraps, evictions;
    bool summary;
} ds4_metal_single_stats;
enum {
    DS4_SINGLE_ARRAY_PAIR,
    DS4_SINGLE_ARRAY_SUM,
    DS4_SINGLE_GROUP_PAIR,
    DS4_SINGLE_GROUP_SUM,
    DS4_SINGLE_GROUP_ARRAY_PAIR,
    DS4_SINGLE_GROUP_ARRAY_SUM,
    DS4_SINGLE_GROUP24_MV,
    DS4_SINGLE_GROUP24_SUM,
    DS4_SINGLE_Q4_ADDRESS_PAIR,
    DS4_SINGLE_Q4_ADDRESS_SUM,
    DS4_SINGLE_MASKED_PAIR,
    DS4_SINGLE_MASKED_SUM,
    DS4_SINGLE_GATHER,
};
typedef struct {
    uint32_t kind;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_gpu_mul_mv_id_args mv;
    ds4_gpu_dsv4_moe_swiglu_weight_args activation;
    ds4_gpu_moe_expert_group_args group;
    ds4_gpu_q4_gather_slots6_args gather;
    ds4_gpu_stream_expert_split_args split;
    uint32_t part;
    uint32_t array_count;
    uint64_t groups_x, groups_y, groups_z;
    uint64_t threads_x, threads_y, shared_bytes;
    bool queue_residency;
    bool use_resources;
} ds4_metal_single_launch;
bool ds4_metal_single_scope_begin(void);
void ds4_metal_single_scope_end(bool armed);
void ds4_metal_single_get_facts(uint32_t total, ds4_metal_single_facts *out);
ds4_metal_single_resources *ds4_metal_single_open(const ds4_metal_routed_request *request,
                                                  const ds4_gpu_tensor *add_in);
void ds4_metal_single_close(ds4_metal_single_resources *resources);
/* The bundle is a real subclass of the common ARC bundle; borrowing its base
 * preserves one resource owner when common fixed encoder leaves are reused. */
ds4_metal_routed_resources *ds4_metal_single_base(ds4_metal_single_resources *resources);
void ds4_metal_single_note_token(uint32_t layer);
int ds4_metal_single_tables(ds4_metal_single_resources *resources, bool address, bool automatic);
/* mode0=base,1=exact,2=transient. Native supplies the checked byte ranges.
 * mask1/2/4 chooses gate/up/down; index<24 retains the group binding. */
bool ds4_metal_single_wrap(ds4_metal_single_resources *resources, uint32_t mask, uint64_t gate,
                           uint64_t up, uint64_t down, uint64_t gate_bytes, uint64_t down_bytes,
                           uint32_t mode, uint32_t index);
bool ds4_metal_single_wrap_slot(ds4_metal_single_resources *resources, uint32_t slot, uint64_t gate,
                                uint64_t up, uint64_t down, uint32_t mode);
bool ds4_metal_single_gather_storage(ds4_metal_single_resources *resources, uint64_t gate_bytes,
                                     uint64_t down_bytes);
void ds4_metal_single_cache_stats(ds4_metal_single_stats *out, bool query_summary);
bool ds4_metal_single_cache_note_size(ds4_metal_single_resources *resources);
bool ds4_metal_single_cache_available(ds4_metal_single_resources *resources);
bool ds4_metal_single_split_ready(void);
bool ds4_metal_single_addresses(ds4_metal_single_resources *resources);
bool ds4_metal_single_full_addresses(ds4_metal_single_resources *resources);
int ds4_metal_single_trace_replay(int32_t *ids, uint32_t count);
bool ds4_metal_single_trace_record(uint32_t layer, const int32_t *ids, uint32_t count,
                                   uint32_t total);
/* Return2 means the existing prefetch was consumed and shared GPU work must
 * be submitted while that actual pending load finishes;1 needs no flush. */
int ds4_metal_single_prefetch_take(ds4_metal_single_resources *resources, int32_t *ids);
bool ds4_metal_single_validate(ds4_metal_single_resources *resources, int32_t *ids, uint32_t *all,
                               uint32_t *miss, uint32_t *invalid);
bool ds4_metal_single_readback_event(void);
bool ds4_metal_single_peek(ds4_metal_single_resources *resources, uint32_t slot, uint32_t expert,
                           uint64_t gate, uint64_t up, uint64_t down);
bool ds4_metal_single_begin_load(ds4_metal_single_resources *resources, const int32_t *ids);
int ds4_metal_single_pending_gate(ds4_metal_single_resources *resources, const int32_t *ids,
                                  uint32_t missing);
bool ds4_metal_single_load_missing(ds4_metal_single_resources *resources, const int32_t *ids,
                                   const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                                   uint32_t missing);
bool ds4_metal_single_bind_loaded(ds4_metal_single_resources *resources, uint32_t missing);
bool ds4_metal_single_mark(ds4_metal_single_resources *resources, uint32_t active);
bool ds4_metal_single_mark_present(ds4_metal_single_resources *resources);
bool ds4_metal_single_publish_addresses(ds4_metal_single_resources *resources, const int32_t *ids);
void ds4_metal_single_prune(ds4_metal_single_resources *resources, const int32_t *ids);
bool ds4_metal_single_compact(ds4_metal_single_resources *resources);
bool ds4_metal_single_host_ids(ds4_metal_single_resources *resources, const int32_t *ids);
void ds4_metal_single_gate_submitted(void);
bool ds4_metal_single_wait_resident(void);
void ds4_metal_single_note_selected(double sync, double copy, double bind, uint32_t resident,
                                    uint32_t missing, bool cache);
void ds4_metal_single_note_split(uint32_t resident, uint32_t missing, double submit,
                                 double load_total, const double *details);
bool ds4_metal_single_parallel_pending(ds4_metal_single_resources *resources, void *command,
                                       ds4_metal_single_parallel *after);
bool ds4_metal_single_parallel_second(void *command);
bool ds4_metal_single_encode(ds4_metal_single_resources *resources, void *command,
                             const ds4_metal_single_launch *launch);
/* The same selected-route diagnostic transaction also serves a C batch caller.
 * Native model retirement closes it; bare legacy callers retain libc atexit. */
void ds4_metal_single_note_selected_hotness(uint32_t layer,
                                           const int32_t *ids,
                                           uint32_t count);
bool ds4_native_metal_selected_hotlist_record(uint32_t layer,
                                             const int32_t *ids,
                                             uint32_t count,
                                             uint32_t total);
void ds4_native_metal_selected_hotlist_close(void);
void ds4_native_metal_selected_trace_close(void);
typedef struct {
    int32_t ids[8];
    uint32_t count;
} ds4_metal_selected_override;
ds4_metal_selected_override *ds4_metal_single_override_state(void);
void **ds4_metal_single_diagnostics_slot(void);
void ds4_native_metal_selected_diagnostics_cleanup(void);
#endif
