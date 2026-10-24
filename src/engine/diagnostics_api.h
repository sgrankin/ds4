#ifndef DS4_ENGINE_DIAGNOSTICS_API_H
#define DS4_ENGINE_DIAGNOSTICS_API_H

#include "ds4_gpu.h"

/* Optional diagnostic filtering/readback; model math and traversal stay native. */
const char *ds4_native_graph_debug_prefix(const char *name, uint32_t layer,
    uint32_t position);
void ds4_native_graph_debug_tensor(const char *name, const ds4_gpu_tensor *,
    uint64_t elements, uint32_t layer, uint32_t position);
void ds4_native_graph_debug_i32_tensor(const char *name, const ds4_gpu_tensor *,
    uint64_t elements, uint32_t layer, uint32_t position);

#endif
