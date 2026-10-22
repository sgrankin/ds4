#ifndef DS4_GGUF_H
#define DS4_GGUF_H

#include <stdbool.h>
#include <stdint.h>

/* Private byte-reader boundary for GGUF metadata. The caller owns both the
 * cursor and its read-only backing bytes. Strings borrow that backing storage;
 * they can contain arbitrary bytes and need not be NUL-terminated. */
typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

/* Values stay in the backing bytes. Record absolute offsets so later model
 * validation can decode only the keys it needs, without copying value data. */
typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_gguf_kv;

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

/* Start with error[0] == 0. Bounds failures leave the destination and position
 * unchanged; the first error records its byte offset and survives later calls.
 * An existing error does not prevent a subsequent valid read. u32/u64 copy
 * native bytes, matching GGUF's little endian on the supported Metal host.
 */
bool ds4_gguf_read(ds4_cursor *cursor, void *dst, uint64_t bytes);
bool ds4_gguf_skip(ds4_cursor *cursor, uint64_t bytes);
bool ds4_gguf_u32(ds4_cursor *cursor, uint32_t *value);
bool ds4_gguf_u64(ds4_cursor *cursor, uint64_t *value);

/* A truncated string body still consumes its length prefix. On failure the
 * output string is unchanged. Value skipping likewise retains partial progress;
 * it checks nested-array depth and scalar-array byte-count overflow.
 */
bool ds4_gguf_string(ds4_cursor *cursor, ds4_str *value);
bool ds4_gguf_skip_value(ds4_cursor *cursor, uint32_t type, int depth);

/* Parse count metadata entries into caller-allocated records (NULL for zero
 * count). The caller checks the header count against file size before allocating
 * storage. Keys borrow cursor->base; values remain there at their recorded
 * offsets. This function neither allocates nor owns the backing bytes.
 *
 * Alignment starts at 32. Only a nonzero UINT32 general.alignment updates it;
 * duplicate entries keep file order, with the last such alignment taking effect.
 * A failure retains completed records, fields already read in the current record,
 * alignment updates and cursor progress, with the ordinary sticky cursor error.
 */
bool ds4_gguf_parse_metadata(ds4_cursor *cursor, ds4_gguf_kv *entries,
                              uint64_t count, uint64_t *alignment);

/* Actual loader records, shared rather than copied into Zig. The selected loader owns
 * mappings, descriptors, array allocation, file descriptors and disk-only tails.
 * Byte/metadata helpers borrow these fields; they never map, allocate or close. */
#define DS4_MAX_DIMS 8

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

typedef ds4_gguf_kv ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct ds4_model {
    int fd;
    const uint8_t *map;
    uint64_t size;       /* Addressable weights, excluding disk-only n-grams. */
    uint64_t file_size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;

    int ngram_fd;
    const ds4_tensor *ngram_tensor;
} ds4_model;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

/* One authoritative format table stays in C. Both implementations consult it;
 * unknown formats return NULL, and nbytes leaves output unchanged on failure. */
const gguf_type_info *ds4_gguf_tensor_type(uint32_t type);
bool ds4_gguf_tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);

/* First matching metadata key wins. Outputs stay unchanged on failed typed
 * reads, except array reads retain a successfully read item tag if length is
 * truncated. Strings and array offsets borrow the model mapping. */
ds4_kv *ds4_gguf_find_kv(const ds4_model *model, const char *key);
bool ds4_gguf_get_string(const ds4_model *model, const char *key, ds4_str *out);
bool ds4_gguf_get_u32(const ds4_model *model, const char *key, uint32_t *out);
bool ds4_gguf_get_token_id(const ds4_model *model, const char *key, int *out);
bool ds4_gguf_get_u64_compat(const ds4_model *model, const char *key, uint64_t *out);
bool ds4_gguf_get_f32_compat(const ds4_model *model, const char *key, float *out);
bool ds4_gguf_get_bool(const ds4_model *model, const char *key, bool *out);
bool ds4_gguf_get_array(const ds4_model *model, const char *key, ds4_array_ref *out);

/* Caller allocates n_tensors zeroed records after checking the file count;
 * alignment is nonzero. Retain partial fields/cursor progress on failure.
 * NULL means success; an error borrows cursor->error or static text. C keeps
 * its original fatal-error policy and unsupported-format warning formatter. */
const char *ds4_gguf_parse_tensors(ds4_model *model, ds4_cursor *cursor);
void ds4_gguf_warn_tensor_type(const ds4_tensor *tensor);

#endif
