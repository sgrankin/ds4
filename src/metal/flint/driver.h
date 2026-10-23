#ifndef FLINT_METAL_DRIVER_H
#define FLINT_METAL_DRIVER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A second model uses the same Metal device and queue as the existing engine.
 * Buffers and commands are retained opaque Metal objects; no tensor runtime is
 * involved. All encoding for one engine is serialized by its scheduler. */
typedef struct flint_gpu flint_gpu;
typedef struct {
    void *handle;
    uint64_t offset;
    uint32_t index;
} flint_gpu_binding;

flint_gpu *flint_gpu_create(void);
void flint_gpu_free(flint_gpu *gpu);
const char *flint_gpu_device_name(flint_gpu *gpu);
bool flint_gpu_compile(flint_gpu *gpu, const char *source, size_t length,
                       char *error, size_t error_size);
void *flint_gpu_buffer(flint_gpu *gpu, size_t bytes);
void *flint_gpu_contents(void *buffer);
size_t flint_gpu_buffer_size(void *buffer);
void flint_gpu_buffer_free(void *buffer);
void *flint_gpu_command_begin(flint_gpu *gpu);
bool flint_gpu_command_end(flint_gpu *gpu, void *command, bool wait);
bool flint_gpu_dispatch(flint_gpu *gpu, void *command, const char *kernel,
                        const void *params, size_t params_bytes,
                        const flint_gpu_binding *bindings, size_t count,
                        const uint32_t groups[3], const uint32_t threads[3]);
const char *flint_gpu_error(flint_gpu *gpu);

#endif
