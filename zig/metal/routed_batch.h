#ifndef DS4_METAL_ROUTED_BATCH_H
#define DS4_METAL_ROUTED_BATCH_H
#include "bridge_types.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    DS4_METAL_TENSOR_Q4_0 = 2,
    DS4_METAL_TENSOR_Q8_0 = 8,
    DS4_METAL_TENSOR_Q2_K = 10,
    DS4_METAL_TENSOR_Q4_K = 12,
    DS4_METAL_TENSOR_Q5_K = 13,
    DS4_METAL_TENSOR_Q6_K = 14,
    DS4_METAL_TENSOR_Q8_K = 15,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
    DS4_METAL_TENSOR_MXFP4 = 39,
};

/* Existing shader packets, moved from the C host once. All original C users
 * and this native controller see these exact declarations. */

typedef struct {
    int32_t nei0;
    int32_t nei1;
    uint64_t nbi1;
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t ne0;
    int32_t ne1;
    uint64_t nb1;
    int32_t nr0;
    /* Tensor-parallel expert ownership; see ds4_metal_args_mul_mv_id in
     * metal/moe.metal.  Zero (from struct literals) means no split. */
    int32_t tp_rank;
    int32_t tp_world;
    int32_t tp_addend;
    int32_t tp_expert_base;
} ds4_gpu_mul_mv_id_args;

typedef struct {
    int32_t ne02;
    int32_t ne10;
    int32_t ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t ne21;
    int32_t ne20;
    uint64_t nb21;
} ds4_gpu_mul_mm_id_map_args;

typedef struct {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne20;
    int32_t ne21;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
    int32_t tp_rank;
    int32_t tp_world;
    int32_t tp_expert_base;
} ds4_gpu_mul_mm_id_args;

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} ds4_gpu_dsv4_moe_swiglu_weight_args;

typedef struct {
    int32_t ne00;
    uint64_t nb01;
    int32_t ne10;
    uint64_t nb11;
    int32_t ne0;
    uint64_t nb1;
    int32_t i00;
    int32_t i10;
    float alpha;
    float limit;
} ds4_gpu_glu_args;

typedef struct {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ds4_gpu_bin_args;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} ds4_gpu_dsv4_moe_sum6_args;

_Static_assert(sizeof(ds4_gpu_mul_mv_id_args) == 136, "Metal routed MV packet");
_Static_assert(sizeof(ds4_gpu_mul_mm_id_map_args) == 48, "Metal routed map packet");
_Static_assert(sizeof(ds4_gpu_mul_mm_id_args) == 104, "Metal routed MM packet");
_Static_assert(sizeof(ds4_gpu_dsv4_moe_swiglu_weight_args) == 48, "Metal routed activation packet");
_Static_assert(sizeof(ds4_gpu_glu_args) == 64, "Metal routed GLU packet");
_Static_assert(sizeof(ds4_gpu_bin_args) == 216, "Metal routed binary packet");
_Static_assert(sizeof(ds4_gpu_dsv4_moe_sum6_args) == 24, "Metal routed sum packet");

/* One immutable invocation of the existing public batch action. */
typedef struct {
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *experts;
    const void *model;
    uint64_t model_bytes;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint32_t gate_type;
    uint32_t down_type;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    uint32_t input_width;
    uint32_t middle_width;
    uint32_t output_width;
    const ds4_gpu_tensor *selected;
    const ds4_gpu_tensor *weights;
    uint32_t total_experts;
    uint32_t used_experts;
    float clamp;
    const ds4_gpu_tensor *x;
    uint32_t layer;
    uint32_t rows;
    bool *mid_is_f16;
    bool force_resident;
} ds4_metal_routed_request;

typedef struct ds4_metal_routed_resources ds4_metal_routed_resources;
enum {
    DS4_ROUTED_QUALITY = 1,
    DS4_ROUTED_SSD = 2,
    DS4_ROUTED_M5 = 4,
    DS4_ROUTED_PRE_M5 = 8,
    DS4_ROUTED_GLM = 16,
    DS4_ROUTED_MPP = 32,
    DS4_ROUTED_QUEUE_RESIDENCY = 64,
    DS4_ROUTED_FULL_LAYER = 128
};
enum {
    DS4_ROUTED_IQ2,
    DS4_ROUTED_Q2,
    DS4_ROUTED_Q4,
    DS4_ROUTED_MX,
    DS4_ROUTED_IQ2_PAIR,
    DS4_ROUTED_Q4_PAIR,
    DS4_ROUTED_IQ2_SWI,
    DS4_ROUTED_Q4_SWI,
    DS4_ROUTED_MX_SWI,
    DS4_ROUTED_Q2_SUM,
    DS4_ROUTED_Q4_SUM,
    DS4_ROUTED_MX_SUM,
    DS4_ROUTED_MX_TP_SWI,
    DS4_ROUTED_MX_TP_SUM,
    DS4_ROUTED_ADDR_IQ2_SWI,
    DS4_ROUTED_ADDR_IQ2,
    DS4_ROUTED_ADDR_Q2_SUM,
    DS4_ROUTED_TABLE_Q4_SWI,
    DS4_ROUTED_TABLE_Q4_SUM,
    DS4_ROUTED_GROUP_Q4_SWI,
    DS4_ROUTED_GROUP_Q4_SUM,
    DS4_ROUTED_GROUP6_SWI,
    DS4_ROUTED_GROUP6_SUM,
    DS4_ROUTED_GROUP8_SWI,
    DS4_ROUTED_GROUP8_SUM,
    DS4_ROUTED_GROUP24,
    DS4_ROUTED_GROUP24_SUM,
    DS4_ROUTED_SLOTS_Q4_SWI,
    DS4_ROUTED_SLOTS_Q4_SUM,
    DS4_ROUTED_SLOTS_MX_SWI,
    DS4_ROUTED_SLOTS_MX_SUM,
    DS4_ROUTED_ADDR_Q4_SWI,
    DS4_ROUTED_ADDR_Q4_SUM,
    DS4_ROUTED_CLAMP,
    DS4_ROUTED_SILU,
    DS4_ROUTED_MUL,
    DS4_ROUTED_WEIGHT,
    DS4_ROUTED_ADD,
    DS4_ROUTED_SWIGLU,
    DS4_ROUTED_SUM6,
    DS4_ROUTED_SUM8,
    DS4_ROUTED_PIPELINE_COUNT
};
typedef struct {
    uint32_t flags;
    int32_t rank;
    int32_t world;
    uint32_t first_expert;
    uint32_t bound_experts;
    uint32_t cache_budget;
    uint32_t tests;
    uint64_t max_buffer_bytes;
    bool table_encoders;
    const ds4_metal_hc_pipeline *cached[DS4_ROUTED_PIPELINE_COUNT];
} ds4_metal_routed_facts;

enum {
    DS4_ROUTED_MV,
    DS4_ROUTED_PAIR,
    DS4_ROUTED_PAIR_SWI,
    DS4_ROUTED_DOWN_SUM,
    DS4_ROUTED_ADDR_PAIR,
    DS4_ROUTED_ADDR_MV,
    DS4_ROUTED_ADDR_SUM,
    DS4_ROUTED_TABLE_PAIR,
    DS4_ROUTED_TABLE_SUM,
    DS4_ROUTED_MAP,
    DS4_ROUTED_MM,
    DS4_ROUTED_MM_PAIR,
    DS4_ROUTED_PACK_OFFSETS,
    DS4_ROUTED_PACK_RHS,
    DS4_ROUTED_ACTIVATION,
    DS4_ROUTED_UNARY,
    DS4_ROUTED_BIN,
    DS4_ROUTED_FLAT,
    DS4_ROUTED_SUM
};
enum {
    DS4_ROUTED_X,
    DS4_ROUTED_GATE,
    DS4_ROUTED_UP,
    DS4_ROUTED_MID,
    DS4_ROUTED_OUT,
    DS4_ROUTED_DOWN,
    DS4_ROUTED_WEIGHTS,
    DS4_ROUTED_SELECTED,
    DS4_ROUTED_W_GATE,
    DS4_ROUTED_W_UP,
    DS4_ROUTED_W_DOWN,
    DS4_ROUTED_MAP_BUFFER,
    DS4_ROUTED_PACKED
};
typedef union {
    ds4_gpu_mul_mv_id_args mv;
    ds4_gpu_mul_mm_id_args mm;
    ds4_gpu_mul_mm_id_map_args map;
    ds4_gpu_dsv4_moe_swiglu_weight_args activation;
    ds4_metal_attention_unary_args unary;
    ds4_gpu_bin_args binary;
    ds4_gpu_glu_args glu;
    ds4_gpu_dsv4_moe_sum6_args sum;
    uint32_t count;
} ds4_metal_routed_args;
/* Fixed existing MoE encoder variants. Native owns packets and geometry;
 * ARC owns buffers, indirect-resource declarations and encoder API calls. */
typedef struct {
    uint32_t kind;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_metal_routed_args args;
    ds4_gpu_dsv4_moe_swiglu_weight_args activation;
    uint32_t source;
    uint32_t second;
    uint32_t destination;
    uint32_t weight;
    uint64_t source_extra;
    uint64_t second_extra;
    uint64_t destination_extra;
    uint64_t counts_bytes;
    uint64_t work_offset;
    uint64_t groups_x, groups_y, groups_z;
    uint64_t threads_x, threads_y;
    uint64_t shared_bytes;
    bool cached;
    bool queue_residency;
    bool use_resources;
} ds4_metal_routed_launch;
double ds4_metal_routed_now_ms(void);
void ds4_metal_routed_get_facts(uint32_t total, ds4_metal_routed_facts *out);
const ds4_metal_hc_pipeline *ds4_metal_routed_pipeline(uint32_t kind, const char *name,
                                                       int16_t nsg);
uint64_t ds4_metal_routed_max_threads(const ds4_metal_hc_pipeline *pipeline);
ds4_metal_routed_resources *ds4_metal_routed_open(const ds4_metal_routed_request *request);
void ds4_metal_routed_close(ds4_metal_routed_resources *resources);
bool ds4_metal_routed_wrap(ds4_metal_routed_resources *resources, uint64_t gate_bytes,
                           uint64_t down_bytes);
void ds4_metal_routed_debug_bind(ds4_metal_routed_resources *resources,
                                 const ds4_gpu_mul_mv_id_args *args, uint64_t gate_bytes);
bool ds4_metal_routed_scratch(ds4_metal_routed_resources *resources, uint32_t kind, uint64_t bytes);
uint32_t ds4_metal_routed_cache_admit(uint64_t gate_bytes, uint64_t down_bytes);
bool ds4_metal_routed_take_ready(const ds4_gpu_tensor *selected);
bool ds4_metal_routed_cache_prepare(ds4_metal_routed_resources *resources, uint32_t *unique,
                                    uint32_t *bad_index, uint32_t *resource_count);
void ds4_metal_routed_cache_clear(uint32_t layer);
bool ds4_metal_routed_mark(ds4_metal_routed_resources *resources);
int ds4_metal_routed_tables(ds4_metal_routed_resources *resources, bool automatic);
bool ds4_metal_routed_residency(ds4_metal_routed_resources *resources, void *command, bool model,
                                bool queue);
bool ds4_metal_routed_encode(ds4_metal_routed_resources *resources, void *command,
                             const ds4_metal_routed_launch *launch);
#endif
