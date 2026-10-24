#ifndef DS4_ENGINE_SESSION_RESOURCES_API_H
#define DS4_ENGINE_SESSION_RESOURCES_API_H

#include "ds4_session_view.h"
#include "ds4_session_resource_pool.h"
#include "engine_admission_api.h"
#include "session_resource_pool_api.h"
#include "deepseek41/graph_api.h"
#include "qwen4/model_api.h"
#include "glm53/api.h"
#include "ds4_sampling.h"
#include "ds4_gpu_mgpu.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

/* Synchronous engine-policy/configuration inputs. Actual mutable model state
 * remains in SessionView, and pool owns the sole persistent engine accounting.
 * The private C shell allocates sizeof(Session) and captures these borrows;
 * it does not construct/free model resources or duplicate a Session layout. */
typedef struct {
    ds4_streaming_policy *policy;
    flint_engine *flint;
    const ds4_qwen4_config *qwen_config;
    const ds4_shape *shape;
    int session_count_hint;
} ds4_session_resource_inputs;

bool ds4_native_session_resources_init(ds4_session_view *,
    ds4_session_resource_pool *, const ds4_session_resource_inputs *, int context);
void ds4_native_session_resources_free(ds4_session_view *,
    ds4_session_resource_pool *);

#endif
