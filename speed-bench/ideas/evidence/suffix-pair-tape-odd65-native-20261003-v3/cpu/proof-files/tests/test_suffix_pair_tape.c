/* Production internal header, no GPU library/device/model. The reference is
 * an independently owned chronological vector of global token identities,
 * never a modulo-addressed ring. Every lane of all six tensors is checked. */
#include "ds4_suffix_pair_tape.h"
#include <inttypes.h>
#include <stdio.h>

static uint32_t trial;
static uint64_t allocations, frees, live_bytes, copy_calls, content_calls, checked_words;
static uint64_t fail_copy, fail_contents;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "trial=%u line=%d: %s\n", trial, __LINE__, #x); exit(1); } } while (0)

struct ds4_gpu_tensor {
    uint8_t *data;
    uint64_t bytes, reported;
    bool inaccessible;
};

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    CHECK(bytes && bytes <= SIZE_MAX);
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    CHECK(t);
    t->data = malloc((size_t)bytes);
    CHECK(t->data);
    t->bytes = t->reported = bytes;
    allocations++; live_bytes += bytes;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    live_bytes -= t->bytes; frees++;
    free(t->data); free(t);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->reported : 0; }

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *t) {
    content_calls++;
    return !t || t->inaccessible || content_calls == fail_contents ? NULL : t->data;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t destination,
        const ds4_gpu_tensor *src, uint64_t source, uint64_t bytes) {
    copy_calls++;
    if (!dst || !src || destination > dst->bytes || bytes > dst->bytes - destination ||
        source > src->bytes || bytes > src->bytes - source) return 0;
    if (copy_calls == fail_copy) {
        /* Failure can follow a partial destination mutation. */
        memset(dst->data + destination, 0xfa, bytes < 7 ? (size_t)bytes : 7);
        return 0;
    }
    memcpy(dst->data + destination, src->data + source, (size_t)bytes);
    return 1;
}

typedef struct { uint32_t position, epoch; } reference_row;
typedef struct { reference_row *rows; uint32_t count; } reference_owner;
typedef struct { uint32_t capacity; reference_owner owners[3]; } reference;

static uint32_t word(uint32_t position, uint32_t owner, uint32_t kind, uint32_t lane, uint32_t epoch) {
    uint32_t x = position * UINT32_C(0x9e3779b9) ^ owner * UINT32_C(0x7f4a7c15) ^
                 kind * UINT32_C(0xa341316c) ^ lane * UINT32_C(0x85ebca6b) ^ epoch * UINT32_C(0xc2b2ae35);
    x ^= x >> 16; x *= UINT32_C(0x7feb352d); x ^= x >> 15;
    return x;
}

static ds41_suffix_pair_tape *tape_new(uint32_t capacity, uint32_t padding) {
    ds41_suffix_pair_tape *t = calloc(1, sizeof(*t));
    CHECK(t); t->capacity = capacity;
    for (uint32_t owner = 0; owner < 3; owner++) {
        t->kv[owner] = ds4_gpu_tensor_alloc((uint64_t)capacity * DS41_SUFFIX_PAIR_ROW_BYTES + padding);
        t->score[owner] = ds4_gpu_tensor_alloc((uint64_t)capacity * DS41_SUFFIX_PAIR_ROW_BYTES + padding);
    }
    return t;
}

static reference reference_new(uint32_t capacity) {
    reference r = {.capacity = capacity};
    for (uint32_t owner = 0; owner < 3; owner++) {
        r.owners[owner].rows = malloc((size_t)capacity * sizeof(reference_row));
        CHECK(r.owners[owner].rows);
    }
    return r;
}

static void reference_reset(reference *r) {
    for (uint32_t owner = 0; owner < 3; owner++) r->owners[owner].count = 0;
}

static void reference_free(reference *r) {
    for (uint32_t owner = 0; owner < 3; owner++) free(r->owners[owner].rows);
}

static void reference_record(reference *r, uint32_t owner, uint32_t start, uint32_t count, uint32_t epoch) {
    reference_owner *o = &r->owners[owner];
    if (o->count && o->rows[o->count - 1].position + 1 != start) o->count = 0;
    const uint32_t total = o->count + count;
    reference_row *v = malloc((size_t)total * sizeof(*v));
    CHECK(v);
    memcpy(v, o->rows, (size_t)o->count * sizeof(*v));
    for (uint32_t i = 0; i < count; i++) v[o->count + i] = (reference_row){start + i, epoch};
    o->count = total < r->capacity ? total : r->capacity;
    memcpy(o->rows, v + total - o->count, (size_t)o->count * sizeof(*v));
    free(v);
}

static bool reference_coverage(const reference *r, uint32_t end, uint32_t *first) {
    *first = 0;
    for (uint32_t owner = 0; owner < 3; owner++) {
        const reference_owner *o = &r->owners[owner];
        if (!o->count || o->rows[o->count - 1].position + 1 != end) return false;
        if (o->rows[0].position > *first) *first = o->rows[0].position;
    }
    return *first < end;
}

static ds4_gpu_tensor *input_new(uint32_t owner, uint32_t kind, uint32_t start, uint32_t count, uint32_t epoch) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)count * DS41_SUFFIX_PAIR_ROW_BYTES);
    uint32_t *p = (uint32_t *)t->data;
    for (uint32_t row = 0; row < count; row++)
        for (uint32_t lane = 0; lane < DS41_SUFFIX_PAIR_WIDTH; lane++)
            *p++ = word(start + row, owner, kind, lane, epoch);
    return t;
}

static bool record(ds41_suffix_pair_tape *t, uint32_t ctx, reference *r,
        uint32_t owner, uint32_t start, uint32_t count, uint32_t epoch) {
    ds4_gpu_tensor *kv = input_new(owner, 0, start, count, epoch);
    ds4_gpu_tensor *score = input_new(owner, 1, start, count, epoch);
    const bool ok = ds41_suffix_pair_tape_record(t, ctx, owner, start, count, kv, score);
    if (ok) reference_record(r, owner, start, count, epoch);
    else r->owners[owner].count = 0;
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(score);
    return ok;
}

static void check_words(const reference *r, uint32_t first, uint32_t end, const void *out) {
    const uint32_t *p = out;
    for (uint32_t owner = 0; owner < 3; owner++) {
        const reference_owner *o = &r->owners[owner];
        for (uint32_t kind = 0; kind < 2; kind++) {
            for (uint32_t position = first; position < end; position++) {
                const reference_row row = o->rows[position - o->rows[0].position];
                CHECK(row.position == position);
                for (uint32_t lane = 0; lane < DS41_SUFFIX_PAIR_WIDTH; lane++) {
                    CHECK(*p++ == word(position, owner, kind, lane, row.epoch));
                    checked_words++;
                }
            }
        }
    }
}

static void *capture_check(ds41_suffix_pair_tape *t, uint32_t ctx, reference *r,
        uint32_t end, uint64_t *bytes_out) {
    uint32_t expected_first, first = UINT32_MAX;
    uint64_t bytes = UINT64_MAX;
    const bool expected = reference_coverage(r, end, &expected_first);
    CHECK(ds41_suffix_pair_tape_coverage(t, ctx, end, &first, &bytes) == expected);
    if (!expected) {
        CHECK(first == 0 && bytes == 0);
        uint8_t untouched[16]; memset(untouched, 0x91, sizeof(untouched));
        CHECK(!ds41_suffix_pair_tape_copy(t, ctx, 0, end, untouched, sizeof(untouched)));
        for (size_t i = 0; i < sizeof(untouched); i++) CHECK(untouched[i] == 0x91);
        *bytes_out = 0;
        return NULL;
    }
    CHECK(first == expected_first);
    CHECK(bytes == (uint64_t)(end - first) * 6 * DS41_SUFFIX_PAIR_ROW_BYTES);
    uint8_t *out = malloc((size_t)bytes + 16); CHECK(out);
    memset(out, 0xb3, (size_t)bytes + 16);
    CHECK(ds41_suffix_pair_tape_copy(t, ctx, first, end, out, bytes));
    check_words(r, first, end, out);
    for (size_t i = (size_t)bytes; i < bytes + 16; i++) CHECK(out[i] == 0xb3);
    *bytes_out = bytes;
    return out;
}

static void all_capacities(void) {
    for (uint32_t cap = 64; cap <= 4096; cap++) {
        trial++;
        ds41_suffix_pair_tape *t = tape_new(cap, cap & 15);
        reference r = reference_new(cap);
        const uint32_t start = 2 * cap - 1, end = start + 3, ctx = 32768;
        CHECK(ds41_suffix_pair_tape_capacity_valid(cap, ctx));
        CHECK(ds41_suffix_pair_tape_owned_bytes(t) == sizeof(*t) + 6 * ((uint64_t)cap * DS41_SUFFIX_PAIR_ROW_BYTES + (cap & 15)));
        for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(t, ctx, &r, owner, start, 3, 1));
        uint64_t bytes; void *out = capture_check(t, ctx, &r, end, &bytes); free(out);
        /* Subsets beginning/ending on either side of pair parity. */
        for (uint32_t first = start; first < end; first++) {
            const uint64_t n = (uint64_t)(end - first) * 6 * DS41_SUFFIX_PAIR_ROW_BYTES;
            out = malloc((size_t)n); CHECK(out);
            CHECK(ds41_suffix_pair_tape_copy(t, ctx, first, end, out, n));
            check_words(&r, first, end, out); free(out);
        }
        CHECK(!ds41_suffix_pair_tape_copy(t, ctx, start - 1, end, &bytes, sizeof(bytes)));
        ds41_suffix_pair_tape_free(t); reference_free(&r);
    }
}

static uint32_t rng = UINT32_C(0x457abc91);
static uint32_t random_u32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static void schedules(void) {
    for (uint32_t run = 0; run < 2200; run++) {
        trial++;
        const uint32_t cap = 64 + random_u32() % 129, ctx = 32768;
        ds41_suffix_pair_tape *t = tape_new(cap, 0);
        reference r = reference_new(cap);
        uint32_t cursor = cap - 3;
        for (uint32_t step = 0; step < 14; step++) {
            if (step == 6 && (run % 3 == 0)) { ds41_suffix_pair_tape_reset(t); reference_reset(&r); }
            const uint32_t mode = random_u32() % 5;
            uint32_t start = cursor;
            if (mode == 1) start += 1 + random_u32() % 7; /* gap */
            if (mode == 2 && start > 20) start -= 1 + random_u32() % 19; /* replay/truncate */
            const uint32_t count = 1 + random_u32() % 17;
            uint32_t order[3] = {0, 1, 2};
            for (uint32_t i = 2; i; i--) { uint32_t j = random_u32() % (i + 1), old = order[i]; order[i] = order[j]; order[j] = old; }
            for (uint32_t i = 0; i < 3; i++) {
                const uint32_t owner = order[i];
                const uint32_t owner_start = mode == 3 && owner == 1 ? start + 1 : start;
                const uint32_t owner_count = mode == 3 && owner == 1 && count > 1 ? count - 1 : count;
                CHECK(record(t, ctx, &r, owner, owner_start, owner_count, 100 + step));
                uint64_t bytes;
                void *out = capture_check(t, ctx, &r, start + count, &bytes); free(out);
            }
            cursor = start + count;
            uint64_t bytes;
            void *out = capture_check(t, ctx, &r, cursor, &bytes); free(out);
        }
        ds41_suffix_pair_tape_free(t); reference_free(&r);
    }
}

static void long_and_scalar(void) {
    const uint32_t capacities[] = {64, 127, 128, 4096};
    for (size_t i = 0; i < sizeof(capacities) / sizeof(*capacities); i++) {
        trial++;
        const uint32_t cap = capacities[i], ctx = 32768, start = cap - 1, count = 2 * cap + 3, end = start + count;
        ds41_suffix_pair_tape *batch = tape_new(cap, 0), *scalar = tape_new(cap, 0);
        reference rb = reference_new(cap), rs = reference_new(cap);
        for (uint32_t owner = 0; owner < 3; owner++) {
            CHECK(record(batch, ctx, &rb, owner, start, count, 73));
            for (uint32_t p = start; p < end; p++) CHECK(record(scalar, ctx, &rs, owner, p, 1, 73));
        }
        uint64_t nb, ns;
        void *ob = capture_check(batch, ctx, &rb, end, &nb);
        void *os = capture_check(scalar, ctx, &rs, end, &ns);
        CHECK(nb == ns && memcmp(ob, os, (size_t)nb) == 0);
        /* Immutable capture survives reset, overwrite, partial-copy failure,
         * and freeing original owner storage. */
        void *saved = malloc((size_t)nb); CHECK(saved); memcpy(saved, ob, (size_t)nb);
        ds41_suffix_pair_tape_reset(batch); reference_reset(&rb);
        for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(batch, ctx, &rb, owner, end, cap, 991));
        fail_copy = copy_calls + 2;
        CHECK(!record(batch, ctx, &rb, 1, end + cap, 2, 992)); fail_copy = 0;
        ds41_suffix_pair_tape_free(batch); reference_free(&rb);
        CHECK(memcmp(saved, ob, (size_t)nb) == 0);
        check_words(&rs, end - cap, end, ob);
        free(saved); free(ob); free(os);
        ds41_suffix_pair_tape_free(scalar); reference_free(&rs);
    }
}

static void failures_and_shared_view(void) {
    const uint32_t cap = 64, ctx = 32768;
    /* Four copy sites across a wrapped two-row record, with actual partial
     * destination mutation at each injected failure. */
    for (uint32_t fail = 1; fail <= 4; fail++) {
        trial++;
        ds41_suffix_pair_tape *t = tape_new(cap, 0);
        reference r = reference_new(cap);
        for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(t, ctx, &r, owner, 0, cap - 1, 1));
        const uint32_t old_first0 = t->first[0], old_end0 = t->end[0];
        fail_copy = copy_calls + fail;
        CHECK(!record(t, ctx, &r, 1, cap - 1, 2, 2)); fail_copy = 0;
        CHECK(!t->started[1] && t->first[1] == 0 && t->end[1] == 0);
        CHECK(t->first[0] == old_first0 && t->end[0] == old_end0 && t->started[0]);
        uint64_t n; void *out = capture_check(t, ctx, &r, cap - 1, &n); free(out);
        for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(t, ctx, &r, owner, cap + 1, 3, 3));
        out = capture_check(t, ctx, &r, cap + 4, &n); free(out);
        ds41_suffix_pair_tape_free(t); reference_free(&r);
    }
    trial++;
    ds41_suffix_pair_tape *t = tape_new(cap, 0);
    reference r = reference_new(cap);
    struct { uint32_t ctx; ds41_suffix_pair_tape *tape; } owner_graph = {ctx, t}, row_view = owner_graph;
    CHECK(row_view.tape == owner_graph.tape);
    for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(row_view.tape, row_view.ctx, &r, owner, 63, 5, 41));
    uint64_t bytes; void *out = capture_check(owner_graph.tape, ctx, &r, 68, &bytes);
    uint8_t *untouched = malloc((size_t)bytes); CHECK(untouched);
    for (uint32_t failure = 1; failure <= 6; failure++) {
        memset(untouched, 0x4e, (size_t)bytes);
        fail_contents = content_calls + failure;
        CHECK(!ds41_suffix_pair_tape_copy(t, ctx, 63, 68, untouched, bytes)); fail_contents = 0;
        for (uint64_t k = 0; k < bytes; k++) CHECK(untouched[k] == 0x4e);
    }
    memset(untouched, 0x4e, (size_t)bytes);
    CHECK(!ds41_suffix_pair_tape_copy(t, ctx, 63, 68, untouched, bytes - 1));
    for (uint64_t k = 0; k < bytes; k++) CHECK(untouched[k] == 0x4e);
    const uint64_t original_bytes = t->score[2]->reported;
    t->score[2]->reported = cap * DS41_SUFFIX_PAIR_ROW_BYTES - 1;
    uint32_t first; uint64_t n;
    CHECK(!ds41_suffix_pair_tape_coverage(t, ctx, 68, &first, &n));
    CHECK(!ds41_suffix_pair_tape_copy(t, ctx, 63, 68, untouched, bytes));
    t->score[2]->reported = original_bytes;
    const uint64_t reported = t->kv[0]->reported;
    t->kv[0]->reported = UINT64_MAX;
    CHECK(ds41_suffix_pair_tape_owned_bytes(t) == UINT64_MAX);
    t->kv[0]->reported = reported;
    ds41_suffix_pair_tape_reset(row_view.tape); reference_reset(&r);
    CHECK(!owner_graph.tape->started[0] && !owner_graph.tape->started[1] && !owner_graph.tape->started[2]);
    CHECK(memcmp(out, untouched, (size_t)bytes) != 0);
    /* One owning free; row views never free shared ownership. */
    ds41_suffix_pair_tape_free(owner_graph.tape); owner_graph.tape = row_view.tape = NULL;
    free(out); free(untouched); reference_free(&r);
}

static void guards_and_no_tape(void) {
    trial++;
    CHECK(!ds41_suffix_pair_tape_capacity_valid(0, 32768));
    CHECK(!ds41_suffix_pair_tape_capacity_valid(63, 32768));
    CHECK(!ds41_suffix_pair_tape_capacity_valid(4097, 32768));
    CHECK(!ds41_suffix_pair_tape_capacity_valid(64, 63));
    CHECK(ds41_suffix_pair_tape_capacity_valid(4096, 4096));
    const uint64_t old_copies = copy_calls, old_contents = content_calls, old_allocations = allocations;
    CHECK(ds41_suffix_pair_tape_record(NULL, 0, 0, UINT32_MAX, UINT32_MAX, NULL, NULL));
    ds41_suffix_pair_tape_reset(NULL); ds41_suffix_pair_tape_free(NULL);
    CHECK(ds41_suffix_pair_tape_owned_bytes(NULL) == 0);
    uint32_t first; uint64_t n; uint8_t out[8] = {0};
    CHECK(!ds41_suffix_pair_tape_coverage(NULL, 32768, 0, &first, &n));
    CHECK(!ds41_suffix_pair_tape_copy(NULL, 32768, 0, 1, out, sizeof(out)));
    CHECK(copy_calls == old_copies && content_calls == old_contents && allocations == old_allocations);
    ds41_suffix_pair_tape *t = tape_new(64, 0);
    CHECK(ds41_suffix_pair_tape_record(t, 0, 3, UINT32_MAX, UINT32_MAX, NULL, NULL));
    CHECK(!ds41_suffix_pair_tape_record(t, 32768, 0, 0, 0, NULL, NULL));
    CHECK(!ds41_suffix_pair_tape_record(t, 32768, 0, UINT32_MAX, 1, NULL, NULL));
    CHECK(!ds41_suffix_pair_tape_record(t, UINT32_MAX, 0, UINT32_MAX - 1, 3, NULL, NULL));
    CHECK(!ds41_suffix_pair_tape_record(t, 63, 0, 0, 1, NULL, NULL));
    CHECK(copy_calls == old_copies && content_calls == old_contents);
    reference r = reference_new(64);
    for (uint32_t owner = 0; owner < 3; owner++) CHECK(record(t, 32768, &r, owner, 32767, 1, 1));
    uint64_t bytes; void *captured = capture_check(t, 32768, &r, 32768, &bytes); free(captured);
    CHECK(!ds41_suffix_pair_tape_coverage(t, 32768, 32769, &first, &n));
    CHECK(!ds41_suffix_pair_tape_coverage(t, 32768, 32768, NULL, &n));
    CHECK(!ds41_suffix_pair_tape_coverage(t, 32768, 32768, &first, NULL));
    CHECK(!ds41_suffix_pair_tape_copy(t, 32768, 32767, 32768, NULL, bytes));
    t->first[0] = 0;
    CHECK(!ds41_suffix_pair_tape_coverage(t, 32768, 32768, &first, &n));
    ds41_suffix_pair_tape_free(t); reference_free(&r);
}

int main(void) {
    guards_and_no_tape();
    all_capacities();
    schedules();
    long_and_scalar();
    failures_and_shared_view();
    CHECK(live_bytes == 0 && allocations == frees);
    printf("suffix-pair-tape PASS trials=%u capacities=4033 schedules=2200 checked_words=%" PRIu64
           " copies=%" PRIu64 " allocations=%" PRIu64 " frees=%" PRIu64
           " live_bytes=%" PRIu64 " models=0 devices=0\n", trial, checked_words, copy_calls, allocations, frees, live_bytes);
    return 0;
}
