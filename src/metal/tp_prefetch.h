#ifndef DS4_METAL_TP_PREFETCH_H
#define DS4_METAL_TP_PREFETCH_H
#include "tp_control.h"
#define DS4_TP_PREFETCH_MAX 8u
#define DS4_TP_PREFETCH_BYTES_PER_US 450000ull /* ~450 GB/s streaming */
#define DS4_TP_PREFETCH_MARGIN_US 8.0          /* flush + wire + pickup */
/* Existing actual gate-kind plan and service-thread observations, once. */
typedef struct {
  uint64_t offset, bytes;
} ds4_tp_prefetch_range;
typedef struct {
  double exchange_ewma_us[2];
  ds4_tp_prefetch_range plan[2][DS4_TP_PREFETCH_MAX];
  uint32_t count[2];
  const void *map;
  uint64_t map_size;
  bool enabled;
} ds4_metal_tp_prefetch_policy;
ds4_metal_tp_prefetch_policy *ds4_metal_tp_prefetch_policy_state(void);
ds4_metal_attention_view ds4_metal_tp_prefetch_storage(void);
uint64_t ds4_metal_tp_prefetch_address(ds4_metal_attention_view view);
/* Fixed touch-table bindings. Native supplies addresses, resources and grid. */
typedef struct {
  const ds4_metal_hc_pipeline *pipeline;
  ds4_metal_attention_view ranges[DS4_TP_PREFETCH_MAX], scratch;
  uint64_t addresses[DS4_TP_PREFETCH_MAX];
  uint32_t lines[DS4_TP_PREFETCH_MAX], count;
  uint64_t groups, threads;
} ds4_metal_tp_prefetch_launch;
int ds4_metal_tp_prefetch_encode(void *command,
                                 const ds4_metal_tp_prefetch_launch *);
int ds4_native_metal_tp_prefetch_encode(uint32_t gate);
#endif
