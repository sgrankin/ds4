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
#endif
