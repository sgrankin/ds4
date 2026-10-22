#ifndef DS4_METAL_EXPERT_TABLES_H
#define DS4_METAL_EXPERT_TABLES_H
#include "expert_cache.h"
/* These are the existing ARC owners, not alternate cache metadata. Returned
 * tables/residencies borrow their dictionary owner; create/wrap tokens are
 * explicitly retained until native publication or scoped release. */
#ifdef __OBJC__
@class DS4MetalQ4ExpertTable;
@protocol MTLArgumentEncoder;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
#define DS4_TABLE_BORROWED __unsafe_unretained
typedef DS4MetalQ4ExpertTable *ds4_metal_table_ref;
typedef id ds4_metal_residency_ref;
typedef id<MTLArgumentEncoder> ds4_metal_argument_ref;
typedef id<MTLCommandBuffer> ds4_metal_table_command_ref;
typedef id<MTLComputeCommandEncoder> ds4_metal_table_encoder_ref;
#else
#define DS4_TABLE_BORROWED
typedef struct ds4_metal_expert_table *ds4_metal_table_ref;
typedef struct ds4_metal_expert_residency *ds4_metal_residency_ref;
typedef struct ds4_metal_expert_argument *ds4_metal_argument_ref;
typedef struct ds4_metal_table_command *ds4_metal_table_command_ref;
typedef struct ds4_metal_table_encoder *ds4_metal_table_encoder_ref;
#endif
typedef struct {
    bool device, table_cache, layer_cache, streaming, glm, queue_residency;
    bool gpu_addresses, residency_sets, address_pipelines;
    ds4_metal_argument_ref DS4_TABLE_BORROWED encoders[3];
} ds4_metal_table_facts;
typedef struct {
    const void *model;
    uint64_t model_bytes, offset, expert_bytes;
    uint32_t experts, group;
    uint64_t encoded_bytes;
    bool address;
} ds4_metal_table_key;
typedef struct {
    ds4_metal_expert_buffer_ref DS4_TABLE_BORROWED argument, address;
    ds4_metal_residency_ref DS4_TABLE_BORROWED residency;
    uint32_t buffers;
    bool has_buffers;
} ds4_metal_table_regions;
typedef struct {
    ds4_metal_expert_buffer_ref DS4_TABLE_BORROWED buffer;
    uint64_t inner;
} ds4_metal_table_wrap;
enum {
    DS4_TABLE_CACHE_GATE,
    DS4_TABLE_CACHE_UP,
    DS4_TABLE_CACHE_DOWN,
    DS4_TABLE_COMPACT_GATE,
    DS4_TABLE_COMPACT_UP,
    DS4_TABLE_COMPACT_DOWN,
    DS4_TABLE_COMPACT_IDS,
    DS4_TABLE_HOST_IDS
};
void ds4_metal_tables_facts(ds4_metal_table_facts *);
ds4_metal_expert_buffer_ref ds4_metal_tables_ensure_buffer(ds4_metal_expert_cache_state *,
                                                           uint32_t kind, uint32_t layer,
                                                           uint64_t bytes, bool zero,
                                                           const char *label);
uint64_t ds4_metal_tables_address(ds4_metal_expert_buffer_ref, uintptr_t);
ds4_metal_table_wrap ds4_metal_tables_wrap(const void *, uint64_t, uint64_t, uint64_t, bool exact);
void ds4_metal_tables_release_buffer(ds4_metal_expert_buffer_ref);
uint64_t ds4_metal_tables_encoded_bytes(ds4_metal_argument_ref);
ds4_metal_table_ref ds4_metal_tables_lookup(const ds4_metal_table_key *);
ds4_metal_table_ref ds4_metal_tables_create(const ds4_metal_table_key *, ds4_metal_argument_ref,
                                            uint32_t capacity);
void ds4_metal_tables_drop(ds4_metal_table_ref);
void ds4_metal_tables_publish(const ds4_metal_table_key *, ds4_metal_table_ref);
ds4_metal_table_regions ds4_metal_tables_regions(ds4_metal_table_ref);
void *ds4_metal_tables_address_bytes(ds4_metal_table_ref);
void ds4_metal_tables_append(ds4_metal_table_ref, ds4_metal_expert_buffer_ref);
void ds4_metal_tables_argument_buffer(ds4_metal_argument_ref, ds4_metal_expert_buffer_ref,
                                      uint64_t offset, uint32_t index);
uint32_t ds4_metal_tables_buffers(ds4_metal_table_ref,
                                  ds4_metal_expert_buffer_ref DS4_TABLE_BORROWED *out);
ds4_metal_residency_ref ds4_metal_tables_residency_create(uint32_t capacity, bool layer);
void ds4_metal_tables_residency_add(ds4_metal_residency_ref, ds4_metal_expert_buffer_ref);
void ds4_metal_tables_residency_commit(ds4_metal_residency_ref);
void ds4_metal_tables_residency_drop(ds4_metal_residency_ref);
void ds4_metal_tables_tensor_residency(ds4_metal_table_ref, ds4_metal_residency_ref);
ds4_metal_residency_ref ds4_metal_tables_layer_lookup(ds4_metal_table_ref, ds4_metal_table_ref,
                                                      ds4_metal_table_ref);
void ds4_metal_tables_layer_publish(ds4_metal_table_ref, ds4_metal_table_ref, ds4_metal_table_ref,
                                    ds4_metal_residency_ref, bool queue);
bool ds4_metal_tables_use_model(ds4_metal_table_command_ref);
void ds4_metal_tables_use_residency(ds4_metal_table_command_ref, ds4_metal_residency_ref);
void ds4_metal_tables_use_buffer(ds4_metal_table_encoder_ref, ds4_metal_expert_buffer_ref);
void ds4_metal_tables_use_buffers(ds4_metal_table_encoder_ref,
                                  const ds4_metal_expert_buffer_ref DS4_TABLE_BORROWED *,
                                  uint32_t count);
void ds4_metal_tables_bind_buffer(ds4_metal_table_encoder_ref, ds4_metal_expert_buffer_ref,
                                  uintptr_t index);
#undef DS4_TABLE_BORROWED
/* Native facade. */
int ds4_native_table_stream_compact_addr_ensure_buffers(ds4_metal_expert_cache_state *state,
                                                        uint32_t layer);
int ds4_native_table_stream_compact_addr_prepare(ds4_metal_expert_cache_state *state,
                                                 uint32_t layer,
                                                 ds4_gpu_stream_expert_cache_entry *const *entries,
                                                 uint32_t n_entries);
int ds4_native_table_stream_selected_ids_prepare(ds4_metal_expert_cache_state *state,
                                                 uint32_t layer, const int32_t *selected_ids,
                                                 uint32_t n_selected);
int ds4_native_table_stream_expert_cache_ensure_addr_buffers(ds4_metal_expert_cache_state *state,
                                                             uint32_t layer);
void ds4_native_table_stream_expert_cache_zero_addr_slot(ds4_metal_expert_cache_state *state,
                                                         uint32_t layer, uint32_t expert);
int ds4_native_table_stream_expert_cache_set_addr_slot_raw(
    ds4_metal_expert_cache_state *state, uint32_t layer, uint32_t expert,
    ds4_metal_expert_buffer_ref gate_buffer, uintptr_t gate_inner,
    ds4_metal_expert_buffer_ref up_buffer, uintptr_t up_inner,
    ds4_metal_expert_buffer_ref down_buffer, uintptr_t down_inner);
int ds4_native_table_stream_expert_cache_addr_buffers(ds4_metal_expert_cache_state *state,
                                                      uint32_t layer);
void ds4_native_table_stream_full_expert_addr_clear_layer(ds4_metal_expert_cache_state *state,
                                                          uint32_t layer);
ds4_gpu_stream_expert_cache_entry *ds4_native_table_stream_full_expert_addr_table_prepare(
    ds4_metal_expert_cache_state *state, const void *model_map, uint64_t model_size, uint32_t layer,
    uint32_t n_total_expert, uint64_t gate_abs_offset, uint64_t up_abs_offset,
    uint64_t down_abs_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes);
uint32_t ds4_native_table_q4_expert_table_group_size(ds4_metal_expert_cache_state *state,
                                                     uint32_t n_total_expert);
bool ds4_native_table_q4_table_queue_residency_requested(ds4_metal_expert_cache_state *state);
bool ds4_native_table_q4_non_streaming_opt_in_enabled(ds4_metal_expert_cache_state *state);
bool ds4_native_table_q4_selected_paths_allowed(ds4_metal_expert_cache_state *state);
int ds4_native_table_pro_q4_expert_table_auto_available(ds4_metal_expert_cache_state *state);
bool ds4_native_table_q4_table_queue_residency_enabled(ds4_metal_expert_cache_state *state,
                                                       bool auto_queue_residency);
bool ds4_native_table_q4_table_model_residency_enabled(ds4_metal_expert_cache_state *state);
bool ds4_native_table_pro_q4_expert_indirect_shape_supported(ds4_metal_expert_cache_state *state,
                                                             uint32_t n_total_expert,
                                                             uint32_t n_expert,
                                                             uint64_t gate_tensor_bytes,
                                                             uint64_t down_tensor_bytes);
bool ds4_native_table_pro_q4_expert_table_auto_enabled(ds4_metal_expert_cache_state *state,
                                                       uint32_t n_total_expert, uint32_t n_expert,
                                                       uint64_t gate_tensor_bytes,
                                                       uint64_t down_tensor_bytes);
bool ds4_native_table_pro_q4_expert_address_auto_enabled(ds4_metal_expert_cache_state *state,
                                                         uint32_t n_total_expert, uint32_t n_expert,
                                                         uint64_t gate_tensor_bytes,
                                                         uint64_t down_tensor_bytes);
bool ds4_native_table_q4_table_bind_anchors_enabled(ds4_metal_expert_cache_state *state);
int ds4_native_table_bind_q4_expert_table_anchors(ds4_metal_expert_cache_state *state,
                                                  ds4_metal_table_encoder_ref enc,
                                                  ds4_metal_table_ref table, uintptr_t first_index,
                                                  uintptr_t max_count);
int ds4_native_table_use_model_residency_set(ds4_metal_expert_cache_state *state,
                                             ds4_metal_table_command_ref cb);
ds4_metal_residency_ref ds4_native_table_q4_expert_layer_residency_set(
    ds4_metal_expert_cache_state *state, ds4_metal_table_ref gate_table,
    ds4_metal_table_ref up_table, ds4_metal_table_ref down_table, bool auto_queue_residency);
ds4_metal_table_ref ds4_native_table_q4_expert_table(ds4_metal_expert_cache_state *state,
                                                     const void *model_map, uint64_t model_size,
                                                     uint64_t tensor_offset, uint64_t expert_bytes,
                                                     uint32_t n_total_expert,
                                                     ds4_metal_argument_ref encoder);
ds4_metal_table_ref
ds4_native_table_q4_expert_address_table(ds4_metal_expert_cache_state *state, const void *model_map,
                                         uint64_t model_size, uint64_t tensor_offset,
                                         uint64_t expert_bytes, uint32_t n_total_expert);
int ds4_native_table_preload_q4_expert_tables(ds4_metal_expert_cache_state *state,
                                              const void *model_map, uint64_t model_size,
                                              uint64_t gate_offset, uint64_t up_offset,
                                              uint64_t down_offset, uint64_t gate_expert_bytes,
                                              uint64_t down_expert_bytes, uint32_t n_total_expert);
void ds4_native_table_use_q4_expert_table_resources(ds4_metal_expert_cache_state *state,
                                                    ds4_metal_table_command_ref cb,
                                                    ds4_metal_table_encoder_ref enc,
                                                    ds4_metal_table_ref table,
                                                    bool queue_residency);
#endif
