#ifndef DS4_ZIG_METAL_BRIDGE_TYPES_H
#define DS4_ZIG_METAL_BRIDGE_TYPES_H

#include "../ds4_gpu.h"
#include <stddef.h>

/* The existing copy shader's argument packet. Keep the C encoder and Zig
 * controller on this same declaration when the native controller is enabled. */
typedef struct {
    int64_t nk0;
    int64_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int64_t ne0, ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;
} ds4_metal_copy_args;

enum {
    DS4_METAL_COPY_CONTIG_F32_F16 = 0,
    DS4_METAL_COPY_F32_F16 = 1,
};

/* Stack arguments are borrowed for encode only. Offsets are relative to each
 * tensor view; the bridge adds its underlying buffer offset exactly once. */
typedef struct {
    uint32_t pipeline;
    const void *args;
    uint64_t args_bytes;
    const ds4_gpu_tensor *src;
    ds4_gpu_tensor *dst;
    uint64_t src_offset, dst_offset;
    uint64_t groups, threads;
} ds4_metal_copy_launch;

_Static_assert(sizeof(ds4_metal_copy_args) == 136, "Metal copy shader arguments");
_Static_assert(offsetof(ds4_metal_copy_args, nb00) == 40, "Metal copy shader strides");

int ds4_zig_metal_env_bool(const char *name);
uint64_t ds4_zig_metal_copy_max_threads(uint32_t pipeline);

/* One retained token spans every chunk. Release always drops that retain;
 * only a successful sequence finishes an owned command buffer. A borrowed
 * caller batch stays active and is never committed here, even after failure. */
void *ds4_zig_metal_command_acquire(int *owned);
int ds4_zig_metal_copy_encode(void *command, const ds4_metal_copy_launch *launch);
int ds4_zig_metal_command_release(void *command, int owned, int ok, const char *label);

/* HC shader packets, shared with the existing C norm/matmul encoders. */
typedef struct {
    int32_t ne00, ne00_t;
    uint64_t nb1, nb2, nb3;
    float eps;
    int32_t nef1[3], nef2[3], nef3[3];
    uint64_t nbf1[3], nbf2[3], nbf3[3];
} ds4_metal_rms_args;

typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} ds4_metal_mm_args;

_Static_assert(sizeof(ds4_metal_rms_args) == 144, "Metal RMS shader arguments");
_Static_assert(sizeof(ds4_metal_mm_args) == 88, "Metal MM shader arguments");

/* PSOs are borrowed from engine-owned caches until GPU cleanup. Weight handles
 * retain the wrapped model buffer through encode and any owned command wait. */
typedef struct ds4_metal_hc_pipeline ds4_metal_hc_pipeline;
typedef struct ds4_metal_model_buffer ds4_metal_model_buffer;
enum {
    DS4_METAL_HC_M3 = 1,
    DS4_METAL_HC_TEST = 2,
    DS4_METAL_HC_SCALE_PRESENT = 4,
    DS4_METAL_HC_SCALE_STAGE = 0,
    DS4_METAL_HC_PROJECT_STAGE = 1,
};
typedef struct {
    uint32_t stage;
    const ds4_metal_hc_pipeline *pipeline;
    const void *args;
    uint64_t args_bytes;
    const ds4_gpu_tensor *x;
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *scales;
    const ds4_metal_model_buffer *weights;
    uint64_t weight_offset;
    uint64_t groups_x, groups_y, threads;
    uint32_t shared_bytes;
} ds4_metal_hc_launch;

int ds4_zig_metal_env_present(const char *name);
int ds4_zig_metal_error(const char *message);
int ds4_zig_metal_tensor_has_buffer(const ds4_gpu_tensor *tensor);
uint32_t ds4_zig_metal_hc_facts(void);
const ds4_metal_hc_pipeline *ds4_zig_metal_hc_mm_pipeline(int bc_in, int bc_out);
const ds4_metal_hc_pipeline *ds4_zig_metal_hc_scale_pipeline(void);
ds4_metal_model_buffer *ds4_metal_retain_model_range(const void *map, uint64_t size,
    uint64_t offset, uint64_t bytes, uint64_t *inner_offset);
void ds4_metal_release_model_buffer(ds4_metal_model_buffer *weights);
int ds4_zig_metal_hc_encode(void *command, const ds4_metal_hc_launch *launch);

/* Ordinary RMS dispatch family; no deferred KV task state is exposed. */
typedef struct {
    int32_t q_n, q_n4, kv_n, kv_n4;
    uint64_t q_row_stride, kv_row_stride;
    float eps;
    uint32_t tasks;
} ds4_metal_qkv_rms_args;
_Static_assert(sizeof(ds4_metal_qkv_rms_args) == 40, "Metal QKV RMS shader arguments");
enum {
    DS4_METAL_NORM_PLAIN = 0,
    DS4_METAL_NORM_WEIGHTED = 1,
    DS4_METAL_NORM_WEIGHTED_BF16 = 2,
    DS4_METAL_NORM_ADD = 3,
    DS4_METAL_NORM_QKV = 4,
};
typedef union {
    ds4_metal_rms_args rms;
    ds4_metal_qkv_rms_args qkv;
} ds4_metal_norm_args;
/* Only the five existing norm kernels. y/aux_out serve add or QKV; wrapped
 * weights remain retained by the caller through command release. */
typedef struct {
    uint32_t mode;
    ds4_metal_norm_args args;
    const ds4_gpu_tensor *x, *y;
    ds4_gpu_tensor *out, *aux_out;
    const ds4_metal_model_buffer *weights0, *weights1;
    uint64_t weight_offset0, weight_offset1;
    uint64_t rows, tasks, threads;
    uint32_t shared_bytes;
} ds4_metal_norm_launch;
int ds4_zig_metal_norm_encode(void *command, const ds4_metal_norm_launch *launch);


/* Only the three activation kernels. Packets are shared with the C fallback;
 * tensor references and stack descriptors are borrowed during encode. */
typedef struct { uint32_t width, rows, format; } ds4_metal_quant_args;
typedef struct { uint32_t width, rows; float eps; uint32_t masked; } ds4_metal_engram_args;
typedef struct { uint32_t width, pairs, odd; } ds4_metal_pool_args;
_Static_assert(sizeof(ds4_metal_quant_args) == 12, "Metal activation quantizer arguments");
_Static_assert(sizeof(ds4_metal_engram_args) == 16, "Metal Engram arguments");
_Static_assert(sizeof(ds4_metal_pool_args) == 12, "Metal pair-pool arguments");
typedef struct {
    uint32_t linear;
    ds4_metal_quant_args args;
    uint64_t count;
    ds4_gpu_tensor *x;
    uint64_t groups_x, groups_y, threads;
} ds4_metal_quant_launch;
typedef struct {
    ds4_metal_engram_args args;
    ds4_gpu_tensor *residual;
    const ds4_gpu_tensor *kv, *q_weight, *k_weight, *mask;
    uint64_t groups_x, groups_y, threads;
} ds4_metal_engram_launch;
typedef struct {
    ds4_metal_pool_args args;
    ds4_gpu_tensor *out;
    const ds4_gpu_tensor *kv, *scores;
    ds4_gpu_tensor *previous_kv, *previous_scores;
    uint64_t grid_x, grid_y, threads;
} ds4_metal_pool_launch;
uint64_t ds4_zig_metal_tensor_offset(const ds4_gpu_tensor *tensor);
int ds4_zig_metal_quant_encode(void *command, const ds4_metal_quant_launch *launch);
int ds4_zig_metal_engram_encode(void *command, const ds4_metal_engram_launch *launch);
int ds4_zig_metal_pool_encode(void *command, const ds4_metal_pool_launch *launch);


/* The actual shared plain-MV packets; C encoders and native policy use one
 * declaration, with the original padding/offsets required by the shaders. */
typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ds4_metal_mv_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_metal_mv_ext_args;
_Static_assert(sizeof(ds4_metal_mv_args) == 112, "Metal MV shader arguments");
_Static_assert(sizeof(ds4_metal_mv_ext_args) == 112, "Metal MV_EXT shader arguments");
_Static_assert(offsetof(ds4_metal_mv_args, nb10) == 64, "Metal MV activation strides");
_Static_assert(offsetof(ds4_metal_mv_ext_args, nb10) == 64, "Metal MV_EXT activation strides");
enum {
    DS4_METAL_F32_MV = 0, DS4_METAL_F32_MV_EXT = 1, DS4_METAL_F32_MM = 2,
    DS4_METAL_F32_M5 = 1, DS4_METAL_F32_QUALITY = 2, DS4_METAL_F32_VERIFY_EXACT = 4,
};
/* Fixed F32 bindings only: args0, model weights1, input2, output3. Stack packet
 * is borrowed during encode; retained weights survive any owned command wait. */
typedef struct {
    const ds4_metal_hc_pipeline *pipeline;
    const void *args;
    uint64_t args_bytes;
    uint32_t kind;
    const ds4_metal_model_buffer *weights;
    uint64_t weight_offset;
    const ds4_gpu_tensor *x;
    ds4_gpu_tensor *out;
    uint64_t groups_x, groups_y, threads_x, threads_y;
    uint32_t shared_bytes;
} ds4_metal_f32_launch;
uint32_t ds4_zig_metal_f32_facts(void);
uint64_t ds4_zig_metal_env_u64(const char *name, uint64_t fallback, uint64_t min, uint64_t max);
const ds4_metal_hc_pipeline *ds4_zig_metal_f32_pipeline(uint32_t kind,
    const char *name, int16_t nsg, int16_t nxpsg, int bc_out);
int ds4_zig_metal_f32_encode(void *command, const ds4_metal_f32_launch *launch);


#endif
