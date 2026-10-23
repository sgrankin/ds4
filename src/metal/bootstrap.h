#ifndef DS4_METAL_BOOTSTRAP_H
#define DS4_METAL_BOOTSTRAP_H
#include "bridge_types.h"
#include "scheduling.h"
#include "expert_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The original process flags and device-name storage, shared once with the
 * remaining C selectors. ARC device/queue/library/PSO storage stays ObjC. */
typedef struct {
    int initialized;
    int runtime_available, family_supported, queue_supported;
    int neural_accelerators_hint, tensor_enabled, tensor_compile_supported;
    char device_name[128];
} ds4_metal_bootstrap_state;
typedef struct {
    ds4_metal_bootstrap_state *state;
    ds4_metal_scheduling *schedule;
    ds4_metal_expert_cache_state *cache;
    ds4_metal_expert_load_state *load;
    bool streaming;
} ds4_metal_bootstrap_context;
typedef struct { bool runtime, family, queue; } ds4_metal_bootstrap_features;
typedef struct {
    bool hc_stable, norm_unify, raw_f32, rope_exp2, math_safe;
} ds4_metal_shader_flags;
typedef struct { void *source, *options, *macros; } ds4_metal_shader_resources;
ds4_metal_bootstrap_context ds4_metal_bootstrap_driver(void);

/* Fixed ARC factories/retirement phases. The shader-source token is +1 and
 * holds source/options/macros between preparation and compilation. Pipeline
 * catalogues retain their exact function-constant and optional-resource order. */
void ds4_metal_bootstrap_reset_fast_cache(void);
bool ds4_metal_bootstrap_device(void);
void ds4_metal_bootstrap_diagnostics(void);
const char *ds4_metal_bootstrap_device_name(void);
ds4_metal_bootstrap_features ds4_metal_bootstrap_features_query(bool disabled);
int ds4_metal_bootstrap_tensor_probe(void);
bool ds4_metal_bootstrap_queue(void);
bool ds4_metal_bootstrap_bookkeeping(void);
bool ds4_metal_bootstrap_source(ds4_metal_shader_resources *);
bool ds4_metal_bootstrap_library(const ds4_metal_shader_resources *, const ds4_metal_shader_flags *);
void ds4_metal_bootstrap_source_release(ds4_metal_shader_resources *);
bool ds4_metal_bootstrap_base_pipelines(void);
bool ds4_metal_bootstrap_remaining_pipelines(void);
void ds4_metal_bootstrap_stop_keepalive(void);
void ds4_metal_bootstrap_commit_wait(void *);
void ds4_metal_bootstrap_memory_report(void);
void ds4_metal_bootstrap_retire_readback(void);
void ds4_metal_bootstrap_retire_addresses(void);
void ds4_metal_bootstrap_retire_pipelines(void);
void ds4_metal_bootstrap_retire_scratch(void);
void ds4_metal_bootstrap_retire_maps(void);
void ds4_metal_bootstrap_retire_bookkeeping(void);

int ds4_native_metal_bootstrap_init(const ds4_metal_bootstrap_context *);
bool ds4_native_metal_bootstrap_begin_cleanup(const ds4_metal_bootstrap_context *);
void ds4_native_metal_bootstrap_cleanup(const ds4_metal_bootstrap_context *);
#endif
