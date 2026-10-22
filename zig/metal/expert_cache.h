#ifndef DS4_METAL_EXPERT_CACHE_H
#define DS4_METAL_EXPERT_CACHE_H
#include "../../ds4_gpu.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#define DS4_METAL_MAX_ROUTED_EXPERT_USED 8
enum {
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER = 80,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT = 384,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED = DS4_METAL_MAX_ROUTED_EXPERT_USED,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES =
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER *
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS = 256,
    DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS = 16,
    DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS = 16,
};
/* The actual C/ObjC record is shared ONCE. Native code only borrows or compares
 * these opaque identities; ARC leaves own every strong assignment/release. */
#ifdef __OBJC__
@protocol MTLBuffer;
typedef id<MTLBuffer> ds4_metal_expert_buffer_ref;
#define DS4_EXPERT_STRONG __strong
#else
typedef struct ds4_metal_expert_buffer *ds4_metal_expert_buffer_ref;
#define DS4_EXPERT_STRONG
#endif
typedef struct {
    uint64_t selected_calls;
    double selected_read_ms;
    double selected_sync_ms;
    double selected_copy_ms;
    double selected_bind_ms;
    uint64_t split_layers;
    uint64_t split_resident_experts;
    uint64_t split_missing_experts;
    double split_resident_ms;
    double split_missing_ms;
    double split_missing_load_ms;
    double split_missing_slot_ms;
    double split_missing_prune_ms;
    double split_missing_addr_ms;
    double split_missing_wait_ms;
    uint64_t load_calls;
    double load_prepare_ms;
    double load_pread_ms;
    double load_install_ms;
    uint64_t prepare_batch_reuse_calls;
    double prepare_batch_reuse_ms;
    uint64_t prepare_buffer_calls;
    double prepare_buffer_ms;
    uint64_t prepare_task_experts;
    double prepare_task_ms;
    uint64_t reuse_scan_calls;
    uint64_t reuse_scan_entries;
    double reuse_scan_ms;
    double reuse_clear_ms;
    uint64_t readahead_calls;
    uint64_t readahead_bytes;
    double readahead_ms;
    uint64_t cache_all_resident_layers;
    uint64_t cache_all_missing_layers;
    uint64_t cache_mixed_layers;
    uint64_t cache_resident_experts;
    uint64_t cache_missing_experts;
} ds4_gpu_stream_expert_timing_snapshot;

typedef struct {
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref gate_buffer;
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref up_buffer;
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref down_buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t gate_abs_offset;
    uint64_t up_abs_offset;
    uint64_t down_abs_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t logical_bytes;
    uint64_t last_used;
    uint64_t use_count;
    uintptr_t gate_inner;
    uintptr_t up_inner;
    uintptr_t down_inner;
    uint64_t inflight_seq;
    uint32_t slab_slot;
    uint8_t valid;
    uint8_t slab_backed;
} ds4_gpu_stream_expert_cache_entry;

typedef struct {
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref gate_buffer;
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref up_buffer;
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref down_buffer;
    uintptr_t gate_inner;
    uintptr_t up_inner;
    uintptr_t down_inner;
} ds4_gpu_stream_expert_reusable_buffers;

/* Original global cache storage, embedded once so all remaining load/table
 * libraries and native lifecycle policy refer to the SAME entries and counters.
 * C defines the storage to keep ARC aggregate initialization authoritative. */
typedef struct {
    uint64_t cache_bytes;
    uint64_t cache_expert_bytes;
    uint32_t cache_entry_count;
    uint32_t cache_budget_override;
    double pending_wait_ms;
    uint64_t pending_joins;
    uint64_t gate_probes;
    uint64_t gate_submissions;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t cache_wraps;
    uint64_t cache_clock;
    uint64_t cache_evict_advise_bytes;
    uint64_t cache_willneed_advise_bytes;
    uint64_t cache_pread_bytes;
    double cache_pread_ms;
    bool io_metrics_enabled;
    uint64_t io_pread_calls;
    uint64_t io_requested_bytes;
    uint64_t io_returned_bytes;
    uint64_t cache_buffer_allocs;
    uint64_t cache_buffer_reuses;
    uint64_t cache_decode_tokens;
    uint32_t cache_first_decode_layer;
    uint64_t cache_hotness_decay_token;
    uint64_t timing_selected_calls;
    double timing_selected_read_ms;
    double timing_selected_sync_ms;
    double timing_selected_copy_ms;
    double timing_selected_bind_ms;
    uint64_t timing_split_layers;
    uint64_t timing_split_resident_experts;
    uint64_t timing_split_missing_experts;
    double timing_split_resident_ms;
    double timing_split_missing_ms;
    double timing_split_missing_load_ms;
    double timing_split_missing_slot_ms;
    double timing_split_missing_prune_ms;
    double timing_split_missing_addr_ms;
    double timing_split_missing_wait_ms;
    uint64_t timing_load_calls;
    double timing_load_prepare_ms;
    double timing_load_pread_ms;
    double timing_load_install_ms;
    uint64_t timing_prepare_batch_reuse_calls;
    double timing_prepare_batch_reuse_ms;
    uint64_t timing_prepare_buffer_calls;
    double timing_prepare_buffer_ms;
    uint64_t timing_prepare_task_experts;
    double timing_prepare_task_ms;
    uint64_t timing_reuse_scan_calls;
    uint64_t timing_reuse_scan_entries;
    double timing_reuse_scan_ms;
    double timing_reuse_clear_ms;
    uint64_t timing_readahead_calls;
    uint64_t timing_readahead_bytes;
    double timing_readahead_ms;
    uint64_t timing_cache_all_resident_layers;
    uint64_t timing_cache_all_missing_layers;
    uint64_t timing_cache_mixed_layers;
    uint64_t timing_cache_resident_experts;
    uint64_t timing_cache_missing_experts;
    ds4_gpu_stream_expert_timing_snapshot timing_last_report;
    ds4_gpu_stream_expert_cache_entry entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER][DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    ds4_gpu_stream_expert_cache_entry full_entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint32_t cache_layer_count[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_hits[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_misses[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_evictions[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_pread_bytes[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    double cache_layer_pread_ms[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_last_hits[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_last_misses[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_last_evictions[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t cache_layer_last_pread_bytes[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    double cache_layer_last_pread_ms[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint32_t cache_route_hotness[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER][DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref cache_gate_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref cache_up_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref cache_down_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref cache_slabs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
    uint32_t cache_slab_start_slot[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
    uint32_t cache_slab_slot_count[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
    uint32_t cache_slab_slots_used[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
    uint32_t cache_slab_count;
    uint32_t cache_slab_total_slots;
    uint32_t cache_free_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES];
    uint32_t cache_free_slot_count;
    uint8_t cache_slab_slot_locked[DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES];
    uint32_t cache_mlock_budget_cap;
    uint8_t cache_mlock_relief_applied;
    uint64_t cache_mlock_bytes;
    uint64_t cache_mlock_fail_bytes;
    uint64_t cache_mlock_failures;
    double cache_mlock_ms;
    int cache_mlock_warned;
    uint64_t cache_slab_slot_bytes;
    uint64_t cache_cb_seq;
    uint64_t cache_done_seq;
    uint64_t cache_batch_seq;
    uint64_t cache_owned_seq;
    uint64_t cache_pending_max_seq;
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref compact_gate_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref compact_up_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref compact_down_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref compact_selected_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref selected_id_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    DS4_EXPERT_STRONG ds4_metal_expert_buffer_ref validate_status_buffer;
} ds4_metal_expert_cache_state;

#undef DS4_EXPERT_STRONG
typedef struct { void *data; uint64_t bytes; } ds4_metal_expert_buffer_memory;
/* Driver/ARC actions. Pending load and address-table construction are explicit
 * remaining whole libraries until their subsequent bounded owner. */
void ds4_metal_expert_reuse_reset(ds4_gpu_stream_expert_reusable_buffers *);
void ds4_metal_expert_entry_retire(ds4_gpu_stream_expert_cache_entry *, ds4_gpu_stream_expert_reusable_buffers *);
void ds4_metal_expert_entry_retain(ds4_gpu_stream_expert_cache_entry *, ds4_metal_expert_buffer_ref, ds4_metal_expert_buffer_ref, ds4_metal_expert_buffer_ref);
ds4_metal_expert_buffer_memory ds4_metal_expert_buffer_storage(ds4_metal_expert_buffer_ref);
void ds4_metal_expert_zero_address(uint32_t layer, uint32_t expert);
void ds4_metal_expert_full_layer_retire(uint32_t layer);
void ds4_metal_expert_zero_layer_address(uint32_t layer);
void ds4_metal_expert_retire_slab(ds4_metal_expert_cache_state *, uint32_t index);
bool ds4_metal_expert_install_address(uint32_t layer, uint32_t expert, ds4_metal_expert_buffer_ref, uintptr_t, ds4_metal_expert_buffer_ref, uintptr_t, ds4_metal_expert_buffer_ref, uintptr_t);
void ds4_metal_expert_pending_clear(void);
bool ds4_metal_expert_wait(const char *label);
bool ds4_metal_expert_on_service_thread(void);
double ds4_metal_expert_now_ms(void);

uint32_t ds4_native_expert_cache_configured_count(ds4_metal_expert_cache_state *state, bool streaming);
uint32_t ds4_native_expert_cache_current_count(ds4_metal_expert_cache_state *state, bool streaming);
uint32_t ds4_native_expert_cache_requested_budget(ds4_metal_expert_cache_state *state, bool streaming);
uint32_t ds4_native_expert_cache_configured_budget(ds4_metal_expert_cache_state *state, bool streaming);
int ds4_native_expert_split_ready(ds4_metal_expert_cache_state *state, bool streaming);
int ds4_native_expert_cache_note_expert_size(ds4_metal_expert_cache_state *state, bool streaming, 
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
uint32_t ds4_native_expert_cache_budget_for_expert_size(ds4_metal_expert_cache_state *state, bool streaming, 
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
uint32_t ds4_native_expert_cache_effective_cap(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected);
int ds4_native_expert_timing_summary_enabled(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_decay_route_hotness(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_reset_route_hotness(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_maybe_decay_route_hotness(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_route_hotness(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t layer,
        uint32_t expert,
        uint32_t amount);
void ds4_native_expert_cache_note_selected_hotness(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected);
void ds4_native_expert_cache_note_frequency_hotness(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t        layer,
        const uint32_t *frequency,
        uint32_t        n_total_expert);
void ds4_native_expert_cache_note_token(ds4_metal_expert_cache_state *state, bool streaming, uint32_t layer_index);
uint32_t ds4_native_expert_cache_decay_tokens(ds4_metal_expert_cache_state *state, bool streaming);
uint64_t ds4_native_expert_cache_next_cb_seq(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_batch_created(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_batch_committed(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_owned_created(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_pending_completed(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_cache_note_owned_completed(ds4_metal_expert_cache_state *state, bool streaming);
int ds4_native_expert_cache_entry_inflight(ds4_metal_expert_cache_state *state, bool streaming, 
        const ds4_gpu_stream_expert_cache_entry *e);
int ds4_native_expert_cache_mark_inflight(ds4_metal_expert_cache_state *state, bool streaming, 
        ds4_gpu_stream_expert_cache_entry *e);
int ds4_native_expert_cache_mark_entries_inflight(ds4_metal_expert_cache_state *state, bool streaming, 
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t n_entries,
        uint32_t active_mask);
int ds4_native_expert_slab_slot_range(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t slot,
        uint32_t *slab_index,
        uint64_t *slot_base);
int ds4_native_expert_slab_slot_for_buffer(ds4_metal_expert_cache_state *state, bool streaming, 
        ds4_metal_expert_buffer_ref buffer,
        uintptr_t    gate_inner,
        uint32_t     *slot_out);
void ds4_native_expert_slab_push_free_slot(ds4_metal_expert_cache_state *state, bool streaming, uint32_t slot);
int ds4_native_expert_slab_unlock_slot(ds4_metal_expert_cache_state *state, bool streaming, uint32_t slot);
void ds4_native_expert_unlock_explicit_buffer(ds4_metal_expert_cache_state *state, bool streaming, ds4_metal_expert_buffer_ref buffer);
int ds4_native_expert_evict_dontneed_enabled(ds4_metal_expert_cache_state *state, bool streaming);
void ds4_native_expert_evict_dontneed_range(ds4_metal_expert_cache_state *state, bool streaming, 
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len);
uint64_t ds4_native_expert_buffer_object_count(ds4_metal_expert_cache_state *state, bool streaming, 
        ds4_metal_expert_buffer_ref gate,
        ds4_metal_expert_buffer_ref up,
        ds4_metal_expert_buffer_ref down);
void ds4_native_expert_cache_clear_entry_internal(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t                                  layer,
        uint32_t                                  expert,
        int                                       count_eviction,
        int                                       recycle_slab_slot,
        ds4_gpu_stream_expert_reusable_buffers   *reuse);
int ds4_native_expert_cache_trim(ds4_metal_expert_cache_state *state, bool streaming, uint32_t experts);
void ds4_native_expert_cache_clear_all(ds4_metal_expert_cache_state *state, bool streaming, int reset_stats);
int ds4_native_expert_cache_is_protected(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t expert,
        const int32_t *protect_ids,
        uint32_t n_protect);
int ds4_native_expert_cache_entry_protected(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t layer,
        uint32_t expert,
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect);
int ds4_native_expert_cache_entry_reusable(ds4_metal_expert_cache_state *state, bool streaming, 
        const ds4_gpu_stream_expert_cache_entry *e,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
void ds4_native_expert_cache_prune_layer(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        const int32_t *protect_ids,
        uint32_t n_protect);
void ds4_native_expert_cache_prune_global(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect);
int ds4_native_expert_cache_take_reusable(ds4_metal_expert_cache_state *state, bool streaming, 
        int                                     force_reuse,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuse);
uint32_t ds4_native_expert_cache_take_reusable_batch(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t                                n_needed,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuses);
uint32_t ds4_native_expert_cache_release_mlock_margin(ds4_metal_expert_cache_state *state, bool streaming, 
        uint32_t       protect_layer,
        const int32_t *protect_ids,
        uint32_t       n_protect);
int ds4_native_expert_cache_entry_matches(ds4_metal_expert_cache_state *state, bool streaming, 
        const ds4_gpu_stream_expert_cache_entry *e,
        const void *model_map,
        uint64_t    model_size,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes);
int ds4_native_expert_cache_snapshot(ds4_metal_expert_cache_state *state, bool streaming, const ds4_gpu_stream_expert_table *table,
        uint8_t *resident);
int ds4_native_expert_cache_snapshot_selected(ds4_metal_expert_cache_state *state, bool streaming, const ds4_gpu_stream_expert_table *table,
        const int32_t *selected, uint32_t n_selected, uint8_t *resident);
ds4_gpu_stream_expert_cache_entry *ds4_native_expert_cache_peek(ds4_metal_expert_cache_state *state, bool streaming, 
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
        uint64_t    down_expert_bytes);
ds4_gpu_stream_expert_cache_entry *
ds4_native_expert_cache_install_loaded(ds4_metal_expert_cache_state *state, bool streaming, 
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        uint32_t       expert,
        uint64_t       gate_abs_offset,
        uint64_t       up_abs_offset,
        uint64_t       down_abs_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        ds4_metal_expert_buffer_ref  gate_buf,
        ds4_metal_expert_buffer_ref  up_buf,
        ds4_metal_expert_buffer_ref  down_buf,
        uintptr_t     gate_inner,
        uintptr_t     up_inner,
        uintptr_t     down_inner);
void ds4_native_expert_cache_clear_layer(ds4_metal_expert_cache_state *state, bool streaming, uint32_t layer);
/* One native process policy, shared by the cache and I/O objects. */
typedef struct {
    bool timing_checked, timing_enabled;
    uint32_t decay_interval;
} ds4_metal_expert_policy;
extern ds4_metal_expert_policy ds4_native_expert_policy;
#endif
