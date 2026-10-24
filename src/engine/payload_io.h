#ifndef DS4_ENGINE_PAYLOAD_IO_H
#define DS4_ENGINE_PAYLOAD_IO_H

#include "ds4_gpu.h"
#include <stdio.h>

/* Bounded file/device transfers shared by the native model codecs. Each
 * codec owns its span order, state validation and commit/rollback policy. */
int ds4_native_payload_write_bytes(FILE *, const void *, uint64_t,
    char *, size_t);
int ds4_native_payload_read_bytes(FILE *, void *, uint64_t, uint64_t *,
    char *, size_t);
int ds4_native_payload_write_u32(FILE *, uint32_t, char *, size_t);
int ds4_native_payload_read_u32(FILE *, uint32_t *, uint64_t *, char *, size_t);
int ds4_native_payload_write_tensor_span(FILE *, const ds4_gpu_tensor *,
    uint64_t offset, uint64_t bytes, uint8_t *scratch, size_t capacity,
    char *, size_t);
int ds4_native_payload_read_tensor_span(FILE *, ds4_gpu_tensor *,
    uint64_t offset, uint64_t bytes, uint8_t *scratch, size_t capacity,
    uint64_t *remaining, char *, size_t);
int ds4_native_payload_write_f16_as_f32(FILE *, const ds4_gpu_tensor *,
    uint64_t offset, uint64_t count, uint8_t *, size_t, char *, size_t);
int ds4_native_payload_read_f32_as_f16(FILE *, ds4_gpu_tensor *,
    uint64_t offset, uint64_t count, uint8_t *, size_t, uint64_t *,
    char *, size_t);

#endif
