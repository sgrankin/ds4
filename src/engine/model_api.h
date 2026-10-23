#ifndef DS4_ENGINE_MODEL_API_H
#define DS4_ENGINE_MODEL_API_H

/* Declaration-only input for the shared gguf_c bindings. The loader and native
 * GGUF readers use the same actual project records; SDK declarations remain in
 * the compiler cache rather than becoming handwritten layout copies. */
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "ds4_gguf.h"
#include "ds4_model_config.h"
#include "ds4_model_weights.h"
#include "ds4_support_weights.h"
#include "vision_weights_api.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Preserve the engine's existing fatal diagnostics and warning verbosity. */
void ds4_engine_model_die(const char *message) __attribute__((noreturn));
void ds4_engine_model_die_errno(const char *operation, const char *path)
    __attribute__((noreturn));
void ds4_engine_model_prefetch_warning(int error_code);

/* Transitional C engine entry points. Successful open transfers all resources
 * into this actual record. Close releases them, clears it and restores fd=-1.
 * NULL close is allowed; metadata/tensor strings borrow the mapped prefix. */
void ds4_native_model_open(ds4_model *model, const char *path,
                           bool metal_mapping, bool prefetch_cpu);
void ds4_native_model_close(ds4_model *model);

#endif
