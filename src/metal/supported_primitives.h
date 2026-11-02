#ifndef DS4_METAL_SUPPORTED_PRIMITIVES_H
#define DS4_METAL_SUPPORTED_PRIMITIVES_H
#include "embedding.h"
enum {
  DS4_TAIL_CARRY,
  DS4_TAIL_PREROPE,
  DS4_TAIL_QUANTIZE,
  DS4_TAIL_BLOCKS,
  DS4_TAIL_FILTER
};
/* Original anonymous u32 packets: carry5, pre-RoPE/quantize3, candidates4.
 * These are immutable dispatch inputs; actual tensor/resource owners are
 * borrowed through the existing shared View record. */
typedef struct {
  uint32_t kind;
  const ds4_metal_hc_pipeline *pipeline;
  uint32_t args[5];
  ds4_metal_attention_view first, second, third;
  uint64_t x, y, threads;
  bool dispatch_threads;
} ds4_metal_supported_launch;
int ds4_metal_supported_encode(void *command,
                               const ds4_metal_supported_launch *);
/* Fixed encoder for the V4.1 kernels whose packets differ. Native controllers
 * own admission and the dispatch graph; resources and argument spans are
 * borrowed only during encode. */
typedef struct {
    uint32_t index;
    ds4_metal_attention_view view;
    const void *bytes;
    uint64_t length;
} ds4_metal_v41_binding;
typedef struct {
    const char *kernel;
    int16_t nsg;
    uint32_t count, shared_bytes;
    const ds4_metal_v41_binding *bindings;
    uint64_t gx, gy, gz, tx, ty;
    bool barrier;
} ds4_metal_v41_launch;
int ds4_metal_v41_encode(void *command, const ds4_metal_v41_launch *launch);
ds4_metal_attention_view ds4_metal_v41_completion(const char *name, uint64_t bytes);
int ds4_metal_v41_on(uint32_t path);
void ds4_metal_v41_note(const char *name);
uint32_t ds4_metal_v41_test_flags(void);
int ds4_metal_v41_dense_decode_allowed(void);
typedef struct { uint32_t rows, parity; } ds4_metal_v41_topk_state;
ds4_metal_v41_topk_state *ds4_metal_v41_topk_state_get(void);
#endif
