/* Focused CPU fixture for the real payload leaves. Tensor storage is a tiny
 * in-memory substitute for the opaque device API; no model or GPU is opened. */
#include "engine/payload_io.h"
#include <arm_neon.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_tensor {
    void *data;
    uint64_t bytes;
    unsigned reads, writes, fail_read, fail_write;
};
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *destination, uint64_t bytes) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)tensor;
    if (++t->reads == t->fail_read) return 0;
    assert(offset <= t->bytes && bytes <= t->bytes - offset);
    memcpy(destination, (char *)t->data + offset, bytes);
    return 1;
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t offset,
                         const void *source, uint64_t bytes) {
    if (++t->writes == t->fail_write) return 0;
    assert(offset <= t->bytes && bytes <= t->bytes - offset);
    memcpy((char *)t->data + offset, source, bytes);
    return 1;
}

/* These are the frozen production C conversions on Apple Silicon. Read data
 * at runtime so constant-folding cannot hide NaN payload/rounding differences. */
static float reference_half(uint16_t bits) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(bits))), 0);
}
static uint16_t reference_float(float value) {
    return vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(value))), 0);
}
static int failed_write(void *context, const char *buffer, int bytes) {
    (void)context; (void)buffer; (void)bytes;
    errno = ENOSPC;
    return -1;
}
static void check_half_wire(void) {
    uint16_t *half = malloc(65536 * sizeof(*half));
    assert(half);
    for (unsigned i = 0; i < 65536; i++) half[i] = (uint16_t)i;
    ds4_gpu_tensor tensor = { .data = half, .bytes = 65536 * sizeof(*half) };
    char *wire = NULL, errors[128] = "";
    size_t wire_bytes = 0;
    /* This odd capacity gives several chunks and a non-full final chunk. */
    uint8_t scratch[6155];
    FILE *fp = open_memstream(&wire, &wire_bytes);
    assert(fp);
    assert(ds4_native_payload_write_f16_as_f32(fp, &tensor, 0, 65536,
        scratch, sizeof(scratch), errors, sizeof(errors)) == 0);
    assert(fclose(fp) == 0 && wire_bytes == 65536 * sizeof(float));
    for (unsigned i = 0; i < 65536; i++) {
        const float expected = reference_half(half[i]);
        if (memcmp(wire + i * sizeof(float), &expected, sizeof(expected)) != 0) {
            uint32_t actual, wanted;
            memcpy(&actual, wire + i * sizeof(float), sizeof(actual));
            memcpy(&wanted, &expected, sizeof(wanted));
            fprintf(stderr, "F16 wire %04x: actual=%08x expected=%08x\n", i, actual, wanted);
            abort();
        }
    }
    assert(tensor.reads > 1);
    free(wire);
    free(half);
}
static void check_float_wire(void) {
    /* Random binary32 encodings cover signs, exponents, finite/subnormal and
     * NaN payloads; every adjacent finite half pair adds midpoint/tie edges. */
    const size_t capacity = 1048576 + 3 * 0x7bff;
    float *values = malloc(capacity * sizeof(*values));
    uint16_t *half = malloc(capacity * sizeof(*half));
    assert(values && half);
    uint32_t random = 7319;
    size_t count = 0;
    for (; count < 1048576; count++) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        memcpy(values + count, &random, sizeof(random));
    }
    for (unsigned i = 0; i < 0x7bff; i++) {
        float midpoint = (reference_half(i) + reference_half(i + 1)) * 0.5f;
        uint32_t bits;
        memcpy(&bits, &midpoint, sizeof(bits));
        for (int step = -1; step <= 1; step++) {
            const uint32_t neighbor = bits + step;
            memcpy(values + count++, &neighbor, sizeof(neighbor));
        }
    }
    ds4_gpu_tensor tensor = { .data = half, .bytes = count * sizeof(*half) };
    FILE *fp = fmemopen(values, count * sizeof(*values), "r");
    assert(fp);
    uint64_t remaining = count * sizeof(*values);
    uint8_t scratch[6155];
    char errors[128] = "";
    assert(ds4_native_payload_read_f32_as_f16(fp, &tensor, 0, count,
        scratch, sizeof(scratch), &remaining, errors, sizeof(errors)) == 0);
    assert(remaining == 0 && fclose(fp) == 0 && tensor.writes > 1);
    for (size_t i = 0; i < count; i++) {
        const uint16_t expected = reference_float(values[i]);
        if (half[i] != expected) {
            uint32_t input;
            memcpy(&input, values + i, sizeof(input));
            fprintf(stderr, "F32 wire %08x: actual=%04x expected=%04x\n", input, half[i], expected);
            abort();
        }
    }
    printf("F16 wire: 65536 half encodings and %zu float encodings exact\n", count);
    free(values);
    free(half);
}
static void check_file_failures(void) {
    char input[] = {1, 2, 3}, buffer[8] = {0}, errors[128] = "";
    FILE *fp = fmemopen(input, sizeof(input), "r");
    assert(fp);
    assert(setvbuf(fp, NULL, _IONBF, 0) == 0);
    uint64_t remaining = 3;
    uint32_t value = 0x89abcdef;
    assert(ds4_native_payload_read_u32(fp, &value, &remaining, errors, sizeof(errors)) == 1);
    assert(value == 0x89abcdef && remaining == 3 && ftell(fp) == 0);
    assert(strcmp(errors, "truncated session payload") == 0);
    remaining = 4;
    assert(ds4_native_payload_read_u32(fp, &value, &remaining, errors, sizeof(errors)) == 1);
    assert(value == 0x89abcdef && remaining == 4 && ftell(fp) == 3);
    assert(strcmp(errors, "failed to read session payload") == 0);
    rewind(fp);
    assert(ds4_native_payload_skip_bytes(fp, 4, (uint8_t *)buffer, 2,
        &remaining, errors, sizeof(errors)) == 1);
    assert(remaining == 4 && ftell(fp) == 3);
    assert(strcmp(errors, "failed to skip session payload") == 0);
    rewind(fp);
    remaining = 2;
    assert(ds4_native_payload_skip_bytes(fp, 3, NULL, 0, &remaining, errors, sizeof(errors)) == 1);
    assert(strcmp(errors, "truncated session payload") == 0 && ftell(fp) == 0);
    assert(fclose(fp) == 0);

    fp = funopen(NULL, NULL, failed_write, NULL, NULL);
    assert(fp && setvbuf(fp, NULL, _IONBF, 0) == 0);
    assert(ds4_native_payload_write_bytes(fp, buffer, 1, errors, sizeof(errors)) == 1);
    assert(strcmp(errors, "failed to write session payload") == 0);
    fclose(fp);

    char *wire = NULL;
    size_t bytes = 0;
    fp = open_memstream(&wire, &bytes);
    assert(fp);
    assert(ds4_native_payload_write_u32(fp, 0x89abcdef, errors, sizeof(errors)) == 0);
    assert(fclose(fp) == 0 && bytes == 4 && memcmp(wire, "\xef\xcd\xab\x89", 4) == 0);
    free(wire);
}
static void check_tensor_failures(void) {
    char input[] = "0123456789", output[16] = {0}, errors[128] = "";
    uint8_t scratch[4];
    ds4_gpu_tensor tensor = { .data = output, .bytes = sizeof(output), .fail_write = 2 };
    FILE *fp = fmemopen(input, 10, "r");
    assert(fp);
    uint64_t remaining = 10;
    assert(ds4_native_payload_read_tensor_span(fp, &tensor, 1, 10, scratch,
        sizeof(scratch), &remaining, errors, sizeof(errors)) == 1);
    assert(remaining == 2 && ftell(fp) == 8 && memcmp(output + 1, input, 4) == 0);
    assert(output[5] == 0 && strcmp(errors, "failed to restore accelerator session tensor") == 0);
    rewind(fp);
    remaining = 10;
    assert(ds4_native_payload_read_tensor_span(fp, &tensor, 15, 10, scratch,
        sizeof(scratch), &remaining, errors, sizeof(errors)) == 1);
    assert(remaining == 10 && ftell(fp) == 0 && tensor.writes == 2);
    assert(strcmp(errors, "session tensor is smaller than the payload") == 0);
    assert(fclose(fp) == 0);

    tensor = (ds4_gpu_tensor){ .data = input, .bytes = 10, .fail_read = 2 };
    char *wire = NULL;
    size_t bytes = 0;
    fp = open_memstream(&wire, &bytes);
    assert(fp);
    assert(ds4_native_payload_write_tensor_span(fp, &tensor, 0, 10, scratch,
        sizeof(scratch), errors, sizeof(errors)) == 1);
    assert(fclose(fp) == 0 && bytes == 4 && memcmp(wire, input, 4) == 0);
    assert(strcmp(errors, "failed to read accelerator session tensor") == 0);
    free(wire);
}
static void check_file_copy(void) {
    const size_t bytes = 8 * 1024 * 1024 + 17;
    char *input = malloc(bytes), *wire = NULL;
    assert(input);
    for (size_t i = 0; i < bytes; i++) input[i] = (char)(i * 71 + 19);
    size_t output_bytes = 0;
    FILE *source = fmemopen(input, bytes, "r");
    FILE *destination = open_memstream(&wire, &output_bytes);
    char errors[128] = "";
    assert(source && destination);
    assert(ds4_native_payload_copy_file(source, destination, bytes, errors, sizeof(errors)) == 0);
    assert(fclose(destination) == 0 && output_bytes == bytes && memcmp(input, wire, bytes) == 0);
    free(wire);
    assert(fclose(source) == 0);

    source = fmemopen(input, 3, "r");
    wire = NULL;
    destination = open_memstream(&wire, &output_bytes);
    assert(source && destination);
    assert(ds4_native_payload_copy_file(source, destination, 4, errors, sizeof(errors)) == 1);
    assert(strcmp(errors, "failed to read staged session payload") == 0);
    assert(fclose(source) == 0 && fclose(destination) == 0 && output_bytes == 0);
    free(wire);
    free(input);
}

int main(void) {
    check_file_failures();
    check_tensor_failures();
    check_file_copy();
    check_half_wire();
    check_float_wire();
    puts("payload I/O CPU checks passed");
    return 0;
}
