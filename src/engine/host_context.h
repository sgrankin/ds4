#ifndef DS4_ENGINE_HOST_CONTEXT_H
#define DS4_ENGINE_HOST_CONTEXT_H
#include "ds4_streaming_policy.h"
#include "ds4_model_config.h"
/* Native Core owns host model configuration and its CPU work pool separately
 * from Metal. A thread entry must select the captured owner explicitly. */
typedef struct ds4_host_context ds4_host_context;
ds4_host_context *ds4_host_context_create(uint32_t requested_threads);
void ds4_host_context_destroy(ds4_host_context *context);
ds4_host_context *ds4_host_context_select(ds4_host_context *context);
ds4_host_context *ds4_host_context_selected(void);
ds4_streaming_policy *ds4_host_context_streaming(ds4_host_context *context);
const ds4_shape *ds4_host_context_shape(const ds4_host_context *context);
void **ds4_host_context_routes_slot(void);
typedef void (*ds4_host_rows_fn)(void *user, uint64_t first, uint64_t end);
/* One submitting caller per owner; different owners have independent pools. */
void ds4_host_context_parallel(ds4_host_context *context, uint64_t rows,
    ds4_host_rows_fn function, void *user, uint64_t min_parallel_rows);
#endif
