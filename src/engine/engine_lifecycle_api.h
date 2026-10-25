#ifndef DS4_ENGINE_LIFECYCLE_API_H
#define DS4_ENGINE_LIFECYCLE_API_H
#define _DARWIN_C_SOURCE 1
#include "ds4_engine_configuration.h"
#include "ds4_engine_core.h"
#include "ds4_session_resource_pool.h"
#include "ds4_streaming_policy.h"
#include "models/flint/engine.h"
#include <sys/mman.h>
#include <unistd.h>

enum {
    DS4_ENGINE_OPEN_OK = 0,
    DS4_ENGINE_OPEN_ERROR = 1,
    DS4_ENGINE_OPEN_LEGACY = 2,
};
int ds4_native_engine_lifecycle_open(ds4_engine_core *,
    ds4_engine_configuration *, ds4_session_resource_pool *, flint_engine **,
    ds4_streaming_policy *, const ds4_engine_options *, bool *native_owned);
void ds4_native_engine_lifecycle_close(ds4_engine_core *,
    ds4_session_resource_pool *, flint_engine **);
/* Normalize project rows; preserve additive magnitudes. Zero rows select no
 * layer. Returns the selected count, or -1 for an invalid mode/nonfinite bank. */
int ds4_steering_prepare_f32(float *, uint32_t layers, uint32_t width, int mode);
bool ds4_native_engine_overthink_set(ds4_engine_core *, ds4_model_family,
    const flint_tokenizer *, float penalty, bool think_only);
bool ds4_native_engine_overthink_configure(ds4_engine_core *, ds4_model_family,
    const flint_tokenizer *);

/* Mechanical process services. No model choice, option policy or allocation
 * plan lives behind these leaves. Compatibility publication only writes the
 * old global snapshots used by explicit legacy/reference paths. */
void ds4_core_acquire_instance_lock(int requested_threads);
void ds4_core_native_admission_acquire(void);
void ds4_core_native_admission_release(void);
void ds4_core_publish_configuration(const ds4_engine_configuration *);

#endif
