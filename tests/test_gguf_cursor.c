/* Link this fixture with the selected production C or Zig cursor module.
 * All backing bytes are stack-owned; no model, filesystem or GPU is needed. */
#include "../ds4_gguf.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static ds4_cursor cursor(const void *bytes, size_t size) {
    return (ds4_cursor){.base = bytes, .size = size};
}

static void put32(uint8_t *out, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) out[i] = (uint8_t)(value >> (8 * i));
}

static void put64(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) out[i] = (uint8_t)(value >> (8 * i));
}

static void test_reads(void) {
    const uint8_t bytes[] = {
        0xaa, 0x78, 0x56, 0x34, 0x12,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0x9a, 0xbc, 0xde,
    };
    ds4_cursor c = cursor(bytes, sizeof(bytes));
    uint32_t word = 0;
    uint64_t wide = 0;
    assert(ds4_gguf_skip(&c, 1));
    assert(ds4_gguf_u32(&c, &word) && word == UINT32_C(0x12345678));
    assert(ds4_gguf_u64(&c, &wide) && wide == UINT64_C(0x1122334455667788));
    uint8_t out[2] = {0};
    assert(ds4_gguf_read(&c, out, sizeof(out)));
    assert(out[0] == 0x9a && out[1] == 0xbc && c.pos == 15);
    assert(!ds4_gguf_read(&c, out, 2));
    assert(out[0] == 0x9a && out[1] == 0xbc && c.pos == 15);
    assert(!strcmp(c.error, "truncated GGUF file at byte 15"));
    char first_error[sizeof(c.error)];
    memcpy(first_error, c.error, sizeof(first_error));
    assert(ds4_gguf_skip(&c, 1) && c.pos == sizeof(bytes));
    assert(ds4_gguf_read(&c, NULL, 0));
    assert(!ds4_gguf_skip(&c, UINT64_MAX));
    assert(c.pos == sizeof(bytes) && !memcmp(first_error, c.error, sizeof(first_error)));
    word = UINT32_C(0xfeedbeef);
    wide = UINT64_MAX;
    assert(!ds4_gguf_u32(&c, &word) && word == UINT32_C(0xfeedbeef));
    assert(!ds4_gguf_u64(&c, &wide) && wide == UINT64_MAX);
    /* Error reporting is sticky, but reads are not automatically poisoned. */
    c.pos = 1;
    assert(ds4_gguf_u32(&c, &word) && word == UINT32_C(0x12345678));
    assert(!memcmp(first_error, c.error, sizeof(first_error)));

    c = cursor(bytes, sizeof(bytes));
    c.pos = UINT64_MAX;
    assert(!ds4_gguf_skip(&c, 0) && c.pos == UINT64_MAX);
    assert(!strcmp(c.error, "truncated GGUF file at byte 18446744073709551615"));
}

static void test_strings(void) {
    uint8_t bytes[13];
    put64(bytes, 5);
    const uint8_t body[] = {'A', 0, 0xe4, 0xa7, 'Z'};
    memcpy(bytes + 8, body, sizeof(body));
    ds4_cursor c = cursor(bytes, sizeof(bytes));
    ds4_str value = {0};
    assert(ds4_gguf_string(&c, &value));
    assert(value.ptr == (const char *)bytes + 8 && value.len == sizeof(body));
    assert(!memcmp(value.ptr, body, sizeof(body)) && c.pos == sizeof(bytes));

    put64(bytes, 0);
    c = cursor(bytes, 8);
    assert(ds4_gguf_string(&c, &value));
    assert(value.ptr == (const char *)bytes + 8 && value.len == 0 && c.pos == 8);

    const ds4_str untouched = {.ptr = (const char *)bytes, .len = 99};
    value = untouched;
    c = cursor(bytes, 7);
    assert(!ds4_gguf_string(&c, &value) && c.pos == 0);
    assert(value.ptr == untouched.ptr && value.len == untouched.len);
    put64(bytes, 5);
    c = cursor(bytes, 11);
    assert(!ds4_gguf_string(&c, &value) && c.pos == 8);
    assert(value.ptr == untouched.ptr && value.len == untouched.len);
    assert(!strcmp(c.error, "truncated GGUF file at byte 8"));
    put64(bytes, UINT64_MAX);
    c = cursor(bytes, sizeof(bytes));
    assert(!ds4_gguf_string(&c, &value) && c.pos == 8);
    assert(value.ptr == untouched.ptr && value.len == untouched.len);
}

static void test_scalar_values(void) {
    const uint8_t bytes[8] = {0};
    const uint8_t widths[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    for (uint32_t type = 0; type < sizeof(widths); type++) {
        if (!widths[type]) continue;
        ds4_cursor c = cursor(bytes, sizeof(bytes));
        assert(ds4_gguf_skip_value(&c, type, 0) && c.pos == widths[type]);
        c = cursor(bytes, widths[type] - 1);
        assert(!ds4_gguf_skip_value(&c, type, 0) && c.pos == 0);
    }
    ds4_cursor c = cursor(bytes, sizeof(bytes));
    assert(!ds4_gguf_skip_value(&c, 99, 0) && c.pos == 0);
    assert(!strcmp(c.error, "unknown GGUF metadata type at byte 0"));
}

static void test_arrays(void) {
    uint8_t bytes[64] = {0};
    put32(bytes, GGUF_VALUE_UINT32);
    put64(bytes + 4, 3);
    ds4_cursor c = cursor(bytes, 24);
    assert(ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 24);
    c = cursor(bytes, 11);
    assert(!ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 4);
    c = cursor(bytes, 23);
    assert(!ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 12);
    put64(bytes + 4, UINT64_MAX / 4 + 1);
    c = cursor(bytes, sizeof(bytes));
    assert(!ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 12);
    assert(!strcmp(c.error, "metadata array is too large at byte 12"));

    /* Empty arrays do not decode or reject their element tag. */
    put32(bytes, 99);
    put64(bytes + 4, 0);
    c = cursor(bytes, 12);
    assert(ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 12);
    put64(bytes + 4, 1);
    c = cursor(bytes, 12);
    assert(!ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 12);
    assert(!strcmp(c.error, "unknown GGUF metadata type at byte 12"));

    /* Two nested arrays: one holds strings, the other fixed-width scalars. */
    put32(bytes, GGUF_VALUE_ARRAY); put64(bytes + 4, 2);
    put32(bytes + 12, GGUF_VALUE_STRING); put64(bytes + 16, 2);
    put64(bytes + 24, 1); bytes[32] = 'x'; put64(bytes + 33, 0);
    put32(bytes + 41, GGUF_VALUE_UINT16); put64(bytes + 45, 2);
    c = cursor(bytes, 57);
    assert(ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 57);
}

static void test_array_depth(void) {
    uint8_t bytes[10 * 12 + 1] = {0};
    for (unsigned levels = 9; levels <= 10; levels++) {
        for (unsigned i = 0; i < levels; i++) {
            put32(bytes + i * 12, i + 1 == levels ? GGUF_VALUE_UINT8 : GGUF_VALUE_ARRAY);
            put64(bytes + i * 12 + 4, 1);
        }
        ds4_cursor c = cursor(bytes, levels * 12 + 1);
        if (levels == 9) {
            assert(ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0));
            assert(c.pos == levels * 12 + 1);
        } else {
            assert(!ds4_gguf_skip_value(&c, GGUF_VALUE_ARRAY, 0) && c.pos == 108);
            assert(!strcmp(c.error, "metadata array nesting is too deep at byte 108"));
        }
    }
}

/* Write a key/type header; its returned position is the absolute value offset. */
static size_t metadata_key(uint8_t *bytes, size_t pos, const char *key, uint32_t type) {
    size_t len = strlen(key);
    put64(bytes + pos, len);
    memcpy(bytes + pos + 8, key, len);
    put32(bytes + pos + 8 + len, type);
    return pos + 12 + len;
}

static void test_metadata(void) {
    uint8_t bytes[512] = {0};
    const size_t start = 7; /* Offsets are relative to the file, not this table. */
    size_t pos = start, values[6], ends[6];
    values[0] = metadata_key(bytes, pos, "model", GGUF_VALUE_STRING);
    put64(bytes + values[0], 3);
    memcpy(bytes + values[0] + 8, "D\0S", 3);
    pos = ends[0] = values[0] + 11;
    values[1] = metadata_key(bytes, pos, "general.alignment", GGUF_VALUE_UINT32);
    put32(bytes + values[1], 64);
    pos = ends[1] = values[1] + 4;
    values[2] = metadata_key(bytes, pos, "empty", GGUF_VALUE_ARRAY);
    put32(bytes + values[2], 99); put64(bytes + values[2] + 4, 0);
    pos = ends[2] = values[2] + 12;
    values[3] = metadata_key(bytes, pos, "general.alignment", GGUF_VALUE_UINT32);
    put32(bytes + values[3], 0);
    pos = ends[3] = values[3] + 4;
    values[4] = metadata_key(bytes, pos, "general.alignment", GGUF_VALUE_UINT64);
    put64(bytes + values[4], 1024);
    pos = ends[4] = values[4] + 8;
    values[5] = metadata_key(bytes, pos, "general.alignment", GGUF_VALUE_UINT32);
    put32(bytes + values[5], 48); /* The original loader permits non-powers of two. */
    pos = ends[5] = values[5] + 4;

    ds4_gguf_kv entries[6] = {0};
    ds4_cursor c = cursor(bytes, pos);
    c.pos = start;
    uint64_t alignment = 999;
    assert(ds4_gguf_parse_metadata(&c, NULL, 0, &alignment));
    assert(alignment == 32 && c.pos == start);
    assert(ds4_gguf_parse_metadata(&c, entries, 4, &alignment));
    assert(alignment == 64 && c.pos == ends[3]); /* Zero does not override. */
    c.pos = start;
    assert(ds4_gguf_parse_metadata(&c, entries, 5, &alignment));
    assert(alignment == 64 && c.pos == ends[4]); /* Wrong type does not override. */
    c.pos = start;
    assert(ds4_gguf_parse_metadata(&c, entries, 6, &alignment));
    assert(alignment == 48 && c.pos == pos && !c.error[0]);
    assert(entries[0].key.ptr == (const char *)bytes + start + 8);
    assert(entries[0].key.len == 5 && !memcmp(entries[0].key.ptr, "model", 5));
    assert(entries[0].type == GGUF_VALUE_STRING && entries[2].type == GGUF_VALUE_ARRAY);
    for (size_t i = 0; i < 6; i++) assert(entries[i].value_pos == values[i]);
}

static void test_metadata_partial(void) {
    uint8_t bytes[96] = {0};
    size_t first = metadata_key(bytes, 0, "general.alignment", GGUF_VALUE_UINT32);
    put32(bytes + first, 64);
    size_t second = metadata_key(bytes, first + 4, "general.alignment", GGUF_VALUE_UINT32);
    put32(bytes + second, 128);
    ds4_cursor c = cursor(bytes, second + 3); /* One missing value byte. */
    ds4_gguf_kv entries[2] = {0};
    uint64_t alignment = 999;
    assert(!ds4_gguf_parse_metadata(&c, entries, 2, &alignment));
    assert(c.pos == second && alignment == 64);
    assert(entries[0].value_pos == first && entries[1].value_pos == second);
    assert(entries[1].key.len == 17 && entries[1].type == GGUF_VALUE_UINT32);
    char expected[96];
    snprintf(expected, sizeof(expected), "truncated GGUF file at byte %zu", second);
    assert(!strcmp(c.error, expected));
}


/* Typed getters borrow the real model directory, with original output timing. */
static void test_model_getters(void) {
    uint8_t bytes[32] = {0};
    ds4_kv entries[] = {
        {{"token", 5}, GGUF_VALUE_INT64, 0},
        {{"float", 5}, GGUF_VALUE_FLOAT64, 8},
        {{"flag", 4}, GGUF_VALUE_BOOL, 16},
        {{"array", 5}, GGUF_VALUE_ARRAY, 24},
    };
    ds4_model model = {.map = bytes, .size = sizeof(bytes), .n_kv = 4, .kv = entries};
    int token = 77;
    put64(bytes, (uint64_t)INT32_MAX + 1);
    assert(!ds4_gguf_get_token_id(&model, "token", &token) && token == 77);
    put64(bytes, 42);
    assert(ds4_gguf_get_token_id(&model, "token", &token) && token == 42);
    entries[0].type = GGUF_VALUE_INT32;
    put32(bytes, UINT32_MAX);
    assert(!ds4_gguf_get_token_id(&model, "token", &token) && token == 42);
    uint32_t untouched = 99;
    assert(!ds4_gguf_get_u32(&model, "missing", &untouched) && untouched == 99);
    const double value = 1.5;
    memcpy(bytes + 8, &value, sizeof(value));
    float converted = 0;
    assert(ds4_gguf_get_f32_compat(&model, "float", &converted) && converted == 1.5f);
    bytes[16] = 2;
    bool flag = false;
    assert(ds4_gguf_get_bool(&model, "flag", &flag) && flag);
    put32(bytes + 24, GGUF_VALUE_UINT8);
    model.size = 28; /* Item tag is readable, its length is truncated. */
    ds4_array_ref array = {91, 92, 93};
    assert(!ds4_gguf_get_array(&model, "array", &array));
    assert(array.type == GGUF_VALUE_UINT8 && array.len == 92 && array.data_pos == 93);
}

static size_t tensor_entry(uint8_t *bytes, uint64_t width, uint64_t rows,
                           uint32_t type, uint64_t offset) {
    put64(bytes, 1); bytes[8] = 'x';
    put32(bytes + 9, 2); put64(bytes + 13, width); put64(bytes + 21, rows);
    put32(bytes + 29, type); put64(bytes + 33, offset);
    return 41;
}

static void test_tensor_directory(void) {
    uint8_t bytes[512] = {0};
    ds4_tensor tensor = {0};
    ds4_model model = {.map = bytes, .size = sizeof(bytes), .n_tensors = 1,
                       .alignment = 32, .tensors = &tensor};
    ds4_cursor c = cursor(bytes, sizeof(bytes));
    const size_t end = tensor_entry(bytes, 33, 1, 8, 0); /* Two q8_0 blocks. */
    assert(ds4_gguf_parse_tensors(&model, &c) == NULL);
    assert(c.pos == end && tensor.name.ptr == (const char *)bytes + 8);
    assert(tensor.elements == 33 && tensor.bytes == 68 && tensor.abs_offset == 64);
    assert(model.tensor_data_pos == 64 && model.max_tensor_bytes == 68);
    uint64_t size = 99;
    assert(!ds4_gguf_tensor_nbytes(0, UINT64_MAX / 4 + 1, &size) && size == 99);
    /* Preserve the original defined unsigned block-rounding wrap. */
    assert(ds4_gguf_tensor_nbytes(8, UINT64_MAX, &size) && size == 0);

    memset(&tensor, 0, sizeof(tensor));
    tensor_entry(bytes, UINT64_MAX, 2, 0, 0);
    c = cursor(bytes, sizeof(bytes));
    assert(!strcmp(ds4_gguf_parse_tensors(&model, &c), "tensor element count overflow"));
    assert(c.pos == 29 && tensor.elements == UINT64_MAX && tensor.dim[1] == 2);
    tensor_entry(bytes, 33, 1, 8, UINT64_MAX);
    c = cursor(bytes, sizeof(bytes));
    assert(!strcmp(ds4_gguf_parse_tensors(&model, &c), "tensor offset overflow"));
}

int main(void) {
    test_reads();
    test_strings();
    test_scalar_values();
    test_arrays();
    test_array_depth();
    test_metadata();
    test_metadata_partial();
    test_model_getters();
    test_tensor_directory();
    puts("GGUF cursor reads, borrowed strings and metadata indexing: PASS");
    return 0;
}
