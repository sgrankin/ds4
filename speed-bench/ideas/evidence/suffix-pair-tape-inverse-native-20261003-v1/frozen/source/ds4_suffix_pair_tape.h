#ifndef DS4_SUFFIX_PAIR_TAPE_H
#define DS4_SUFFIX_PAIR_TAPE_H

/* Internal, opt-in pre-pool2 row capture. Graph row views share this pointer;
 * only the owning graph frees it. These operations enqueue native copies and
 * do not synchronize. The caller synchronizes before immutable CPU capture. */
#include "ds4_gpu.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define DS41_SUFFIX_PAIR_OWNERS 3u
#define DS41_SUFFIX_PAIR_WIDTH 512u
#define DS41_SUFFIX_PAIR_MIN_CAP 64u
#define DS41_SUFFIX_PAIR_MAX_CAP 4096u
#define DS41_SUFFIX_PAIR_ROW_BYTES ((uint64_t)DS41_SUFFIX_PAIR_WIDTH * sizeof(float))

typedef struct {
    uint32_t capacity, first[DS41_SUFFIX_PAIR_OWNERS], end[DS41_SUFFIX_PAIR_OWNERS];
    bool started[DS41_SUFFIX_PAIR_OWNERS];
    ds4_gpu_tensor *kv[DS41_SUFFIX_PAIR_OWNERS], *score[DS41_SUFFIX_PAIR_OWNERS];
} ds41_suffix_pair_tape;

static inline bool ds41_suffix_pair_tape_capacity_valid(uint32_t capacity, uint32_t ctx) {
    return capacity >= DS41_SUFFIX_PAIR_MIN_CAP && capacity <= DS41_SUFFIX_PAIR_MAX_CAP && capacity <= ctx;
}

static inline void ds41_suffix_pair_tape_reset(ds41_suffix_pair_tape *t) {
    if (!t) return;
    memset(t->started, 0, sizeof(t->started));
    memset(t->first, 0, sizeof(t->first));
    memset(t->end, 0, sizeof(t->end));
}

static inline void ds41_suffix_pair_tape_free(ds41_suffix_pair_tape *t) {
    if (!t) return;
    for (uint32_t i = 0; i < DS41_SUFFIX_PAIR_OWNERS; i++) {
        ds4_gpu_tensor_free(t->kv[i]);
        ds4_gpu_tensor_free(t->score[i]);
    }
    free(t);
}

/* Exact app-owned tape metadata plus tensor payload bytes reported by the
 * backend. Opaque GPU tensor-wrapper/driver metadata is a separate quantity.
 * Saturate rather than wrap if a malformed backend byte query overflows. */
static inline uint64_t ds41_suffix_pair_tape_owned_bytes(const ds41_suffix_pair_tape *t) {
    if (!t) return 0;
    uint64_t bytes = sizeof(*t);
    for (uint32_t i = 0; i < DS41_SUFFIX_PAIR_OWNERS; i++) {
        const ds4_gpu_tensor *buffers[2] = {t->kv[i], t->score[i]};
        for (uint32_t kind = 0; kind < 2; kind++) {
            const uint64_t n = buffers[kind] ? ds4_gpu_tensor_bytes(buffers[kind]) : 0;
            if (n > UINT64_MAX - bytes) return UINT64_MAX;
            bytes += n;
        }
    }
    return bytes;
}

/* Capture before pool2 mutates pending-pair state. A gap, replay or truncation
 * replaces this owner's coverage with the latest contiguous range. A failed
 * copy may already have overwritten ring slots: invalidate coverage so those
 * slots can never be advertised as the preceding successful range. */
static inline bool ds41_suffix_pair_tape_record(ds41_suffix_pair_tape *t, uint32_t ctx,
        uint32_t owner, uint32_t start, uint32_t count,
        ds4_gpu_tensor *kv, ds4_gpu_tensor *score) {
    if (!t || owner >= DS41_SUFFIX_PAIR_OWNERS) return true;
    if (!ds41_suffix_pair_tape_capacity_valid(t->capacity, ctx) || !count ||
        start > ctx || count > ctx - start) return false;
    const uint32_t end = start + count;
    uint32_t first = !t->started[owner] || t->end[owner] != start ? start : t->first[owner];
    uint32_t p = count > t->capacity ? end - t->capacity : start;
    if (count > t->capacity && first < p) first = p;
    for (; p < end;) {
        const uint32_t slot = p % t->capacity;
        const uint32_t rows = end - p < t->capacity - slot ? end - p : t->capacity - slot;
        const uint64_t bytes = (uint64_t)rows * DS41_SUFFIX_PAIR_ROW_BYTES;
        const uint64_t destination = (uint64_t)slot * DS41_SUFFIX_PAIR_ROW_BYTES;
        const uint64_t source = (uint64_t)(p - start) * DS41_SUFFIX_PAIR_ROW_BYTES;
        if (!ds4_gpu_tensor_copy(t->kv[owner], destination, kv, source, bytes) ||
            !ds4_gpu_tensor_copy(t->score[owner], destination, score, source, bytes)) {
            t->started[owner] = false;
            t->first[owner] = t->end[owner] = 0;
            return false;
        }
        p += rows;
    }
    t->first[owner] = end - first > t->capacity ? end - t->capacity : first;
    t->end[owner] = end;
    t->started[owner] = true;
    return true;
}

/* Latest complete intersection for all three owners. Buffer extents are
 * checked without reading contents or synchronizing native commands. */
static inline bool ds41_suffix_pair_tape_coverage(const ds41_suffix_pair_tape *t,
        uint32_t ctx, uint32_t end, uint32_t *first_out, uint64_t *bytes_out) {
    if (!first_out || !bytes_out) return false;
    *first_out = 0; *bytes_out = 0;
    if (!t || !ds41_suffix_pair_tape_capacity_valid(t->capacity, ctx) || end > ctx) return false;
    uint32_t first = 0;
    const uint64_t ring_bytes = (uint64_t)t->capacity * DS41_SUFFIX_PAIR_ROW_BYTES;
    for (uint32_t i = 0; i < DS41_SUFFIX_PAIR_OWNERS; i++) {
        if (!t->started[i] || t->end[i] != end || t->first[i] >= end ||
            end - t->first[i] > t->capacity || !t->kv[i] || !t->score[i] ||
            ds4_gpu_tensor_bytes(t->kv[i]) < ring_bytes || ds4_gpu_tensor_bytes(t->score[i]) < ring_bytes)
            return false;
        if (t->first[i] > first) first = t->first[i];
    }
    const uint64_t bytes = (uint64_t)(end - first) * DS41_SUFFIX_PAIR_OWNERS * 2u * DS41_SUFFIX_PAIR_ROW_BYTES;
    if (bytes > SIZE_MAX) return false;
    *first_out = first; *bytes_out = bytes;
    return true;
}

/* Caller has synchronized. Destination is independently owned immutable CPU
 * storage, laid out owner-major, KV then score, chronological global rows.
 * Validate all content pointers before touching the destination. */
static inline bool ds41_suffix_pair_tape_copy(const ds41_suffix_pair_tape *t,
        uint32_t ctx, uint32_t first, uint32_t end, void *destination, uint64_t capacity) {
    uint32_t covered_first = 0; uint64_t covered_bytes = 0;
    if (!destination || !ds41_suffix_pair_tape_coverage(t, ctx, end, &covered_first, &covered_bytes) ||
        first < covered_first || first >= end) return false;
    const uint64_t bytes = (uint64_t)(end - first) * DS41_SUFFIX_PAIR_OWNERS * 2u * DS41_SUFFIX_PAIR_ROW_BYTES;
    if (bytes > capacity || bytes > SIZE_MAX) return false;
    const uint8_t *rings[DS41_SUFFIX_PAIR_OWNERS][2];
    for (uint32_t owner = 0; owner < DS41_SUFFIX_PAIR_OWNERS; owner++) {
        rings[owner][0] = ds4_gpu_tensor_contents(t->kv[owner]);
        rings[owner][1] = ds4_gpu_tensor_contents(t->score[owner]);
        if (!rings[owner][0] || !rings[owner][1]) return false;
    }
    uint8_t *out = destination;
    for (uint32_t owner = 0; owner < DS41_SUFFIX_PAIR_OWNERS; owner++) {
        for (uint32_t kind = 0; kind < 2; kind++) {
            for (uint32_t p = first; p < end;) {
                const uint32_t slot = p % t->capacity;
                const uint32_t count = end - p < t->capacity - slot ? end - p : t->capacity - slot;
                const uint64_t copied = (uint64_t)count * DS41_SUFFIX_PAIR_ROW_BYTES;
                memcpy(out, rings[owner][kind] + (uint64_t)slot * DS41_SUFFIX_PAIR_ROW_BYTES, (size_t)copied);
                out += copied; p += count;
            }
        }
    }
    return true;
}

#endif
