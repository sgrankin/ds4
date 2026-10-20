#define _POSIX_C_SOURCE 200809L
#include "../ds4_state_cache.h"
#include "../ds4_kvstore.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Real production cache and serializer hash, synthetic independently laid-out
 * native bytes. Supply a fresh persistent directory; no model/GPU is opened.
 * macOS example (dead_strip discards unused KV-store engine entry points):
 * clang -std=c11 -O2 -ffunction-sections -fdata-sections tests/test_state_cache.c
 *       ds4_state_cache.c ds4_kvstore.c -Wl,-dead_strip -lm -o test-state-cache
 */

typedef struct { uint8_t *p, *pre; uint64_t bytes, pre_bytes; uint32_t n; } fixture;

static uint64_t fixture_size(uint32_t n) {
    uint64_t z = 52u + ((uint64_t)n + 129280u) * 4u + 40u * (n < 128 ? n : 128) * 2048u;
    for (unsigned i = 0; i < 4; i++) {
        z += (uint64_t)(n / (i < 3 ? 2u : 1u)) * 2560u;
        if (i < 3 && (n & 1)) z += 4096u;
    }
    return z;
}

static void f32(uint8_t *p, float x) { memcpy(p, &x, 4); }

static fixture make_fixture(uint32_t n) {
    fixture f = {.bytes = fixture_size(n), .n = n};
    f.p = malloc((size_t)f.bytes); assert(f.p);
    const uint32_t h[13] = {0x34565344u, 2, 8192, 1, 128, 128, 8193, n,
                            40, 512, 128, 129280, 0x413431u};
    for (unsigned i = 0; i < 13; i++) ds4_kvstore_le_put32(f.p + i * 4u, h[i]);
    uint64_t off = 52;
    for (uint32_t i = 0; i < n; i++, off += 4)
        ds4_kvstore_le_put32(f.p + off, (i * 17u + 3u) % 129280u);
    for (uint32_t j = 0; j < 129280; j++, off += 4)
        f32(f.p + off, (float)(j % 131) * .01f + (float)n * .00001f);
    uint32_t raw = n < 128 ? n : 128;
    for (unsigned il = 0; il < 40; il++)
        for (uint32_t r = 0; r < raw; r++)
            for (unsigned j = 0; j < 512; j++, off += 4)
                f32(f.p + off, (float)(il + 1) + (float)r * .01f +
                               (float)j * .00001f + (float)n * .0001f);
    for (unsigned owner = 0; owner < 4; owner++) {
        uint32_t rows = n / (owner < 3 ? 2u : 1u);
        /* These cumulative rows depend on absolute row, never endpoint n. */
        for (uint32_t r = 0; r < rows; r++)
            for (unsigned j = 0; j < 512; j++, off += 4)
                f32(f.p + off, (float)(owner + 101) + (float)r * .001f + (float)j * .000001f);
        for (uint32_t r = 0; r < rows; r++)
            for (unsigned j = 0; j < 128; j++, off += 4)
                f32(f.p + off, (float)(owner + 201) + (float)r * .001f + (float)j * .000001f);
        if (owner < 3 && (n & 1))
            for (unsigned j = 0; j < 1024; j++, off += 4)
                f32(f.p + off, (float)(owner + 301) + (float)n * .001f + (float)j * .00001f);
    }
    assert(off == f.bytes);
    /* Original archive is stored opaquely; include its native DPR1 layout. */
    f.pre_bytes = 12u + 40u * 128u * 4u + 40u * 128u * 128u;
    for (unsigned i = 0; i < 4; i++) f.pre_bytes += (uint64_t)(n / (i < 3 ? 2u : 1u)) * 257u;
    f.pre = malloc((size_t)f.pre_bytes); assert(f.pre);
    for (uint64_t i = 0; i < f.pre_bytes; i++) f.pre[i] = (uint8_t)(i * 37u + i / 65536u);
    ds4_kvstore_le_put32(f.pre, 0x31525044u);
    ds4_kvstore_le_put32(f.pre + 4, 8192);
    ds4_kvstore_le_put32(f.pre + 8, n);
    return f;
}

static void equal_source(ds4_state_cache *c, const char *key, const fixture *f, bool approx) {
    char err[256] = {0};
    ds4_state_cache_ref *r = ds4_state_cache_get(c, key); assert(r);
    ds4_state_cache_blob b = {0};
    assert(ds4_state_cache_materialize(r, &b, err, sizeof(err)) == 0);
    assert(b.tokens == f->n && b.ctx_size == 8192 && b.approximate == approx);
    assert(b.state_bytes == f->bytes && b.prerope_bytes == f->pre_bytes);
    assert(!memcmp(b.state, f->p, (size_t)f->bytes));
    assert(!memcmp(b.prerope, f->pre, (size_t)f->pre_bytes));
    ds4_state_cache_release(r);
    /* Materialization is independently owned after releasing the cache pin. */
    assert(!memcmp(b.state, f->p, (size_t)f->bytes));
    ds4_state_cache_blob_free(&b);
}

int main(int argc, char **argv) {
    if (argc != 2 || mkdir(argv[1], 0700)) {
        fprintf(stderr, "usage: test-state-cache NEW_PERSISTENT_DIRECTORY\n"); return 2;
    }
    char dir[4096], err[256] = {0};
    snprintf(dir, sizeof(dir), "%s/tier", argv[1]);
    const uint64_t hot = 1024u * 1024u, disk = 96u * 1024u * 1024u;
    ds4_state_cache *c = ds4_state_cache_open(dir, hot, disk, err, sizeof(err)); assert(c);
    fixture a = make_fixture(2048), b = make_fixture(2051);
    char ka[41], kb[41], kd[41];

    /* Make an ordinary atomic rename fail independently of cache internals.
     * Its first immutable header chunk cannot replace this directory. */
    char faildir[4096], blocked[4096], header_key[41];
    snprintf(faildir, sizeof(faildir), "%s/publication-failure", argv[1]);
    ds4_state_cache *failcache = ds4_state_cache_open(faildir, hot, disk, err, sizeof(err)); assert(failcache);
    ds4_kvstore_sha1_bytes_hex(a.p, 52u, header_key);
    int path_n = snprintf(blocked, sizeof(blocked), "%s/ds4sc-%s.chunk", faildir, header_key);
    assert(path_n > 0 && (size_t)path_n < sizeof(blocked) && !mkdir(blocked, 0700));
    assert(ds4_state_cache_put(failcache, "native-model-profile", a.p, a.bytes, a.pre, a.pre_bytes,
                               false, ka, err, sizeof(err)) != 0);
    ds4_state_cache_stats failed;
    ds4_state_cache_get_stats(failcache, &failed);
    assert(!failed.endpoints && !failed.chunks && !failed.ssd_bytes && !failed.hot_bytes);
    assert(!rmdir(blocked));
    assert(ds4_state_cache_close(failcache, err, sizeof(err)) == 0);

    assert(ds4_state_cache_put(c, "native-model-profile", a.p, a.bytes, a.pre, a.pre_bytes,
                               false, ka, err, sizeof(err)) == 0);
    ds4_state_cache_stats first, two;
    ds4_state_cache_get_stats(c, &first);
    assert(ds4_state_cache_put(c, "native-model-profile", b.p, b.bytes, b.pre, b.pre_bytes,
                               false, kb, err, sizeof(err)) == 0);
    ds4_state_cache_get_stats(c, &two);
    assert(two.endpoints == 2 && two.hot_bytes <= hot && two.ssd_bytes <= disk);
    /* Sharing must save the 12.5 MiB full cumulative page, not just headers. */
    assert(two.ssd_bytes + 12u * 1024u * 1024u < a.bytes + a.pre_bytes + b.bytes + b.pre_bytes);
    equal_source(c, ka, &a, false); equal_source(c, kb, &b, false);
    ds4_state_cache_get_stats(c, &two); assert(two.disk_reads > 0);
    /* Later mutation of borrowed input storage cannot mutate cached sources. */
    uint8_t saved = a.p[52]; a.p[52] ^= 1;
    ds4_state_cache_ref *r = ds4_state_cache_get(c, ka); assert(r);
    ds4_state_cache_blob owned = {0};
    assert(ds4_state_cache_materialize(r, &owned, err, sizeof(err)) == 0);
    assert(owned.state[52] == saved);
    a.p[52] = saved; ds4_state_cache_release(r); ds4_state_cache_blob_free(&owned);

    int *ids = malloc((size_t)(b.n + 1u) * sizeof(*ids)); assert(ids);
    for (uint32_t i = 0; i <= b.n; i++) ids[i] = (int)((i * 17u + 3u) % 129280u);
    uint32_t matched = 0;
    r = ds4_state_cache_find_prefix(c, "native-model-profile", 8192, ids, b.n + 1u, &matched);
    assert(r && matched == b.n); ds4_state_cache_release(r);
    ids[0] ^= 1;
    assert(!ds4_state_cache_find_prefix(c, "native-model-profile", 8192, ids, b.n, NULL));
    ids[0] ^= 1;

    ds4_kvstore_le_put32(b.p, 0x44535344u);
    assert(ds4_state_cache_put(c, "approx-only", b.p, b.bytes, b.pre, b.pre_bytes,
                               true, kd, err, sizeof(err)) == 0);
    assert(!ds4_state_cache_find_prefix(c, "approx-only", 8192, ids, b.n, NULL));
    assert(ds4_state_cache_put(c, "approx-only", b.p, b.bytes, b.pre, b.pre_bytes,
                               false, kd, err, sizeof(err)) != 0);
    equal_source(c, kd, &b, true);
    assert(ds4_state_cache_close(c, err, sizeof(err)) == 0);

    c = ds4_state_cache_open(dir, hot, disk, err, sizeof(err)); assert(c);
    equal_source(c, ka, &a, false); equal_source(c, kd, &b, true);
    ds4_kvstore_le_put32(b.p, 0x34565344u);
    equal_source(c, kb, &b, false);
    char cursor[41] = {0}; unsigned seen = 0; const int *borrowed; uint32_t n;
    while ((r = ds4_state_cache_next_candidate(c, "native-model-profile", 8192,
                                              cursor, &borrowed, &n, NULL))) {
        assert(n == a.n || n == b.n);
        for (uint32_t i = 0; i < n; i++) assert(borrowed[i] == ids[i]);
        seen++; ds4_state_cache_release(r);
    }
    assert(seen == 2);
    cursor[0] = 0; bool approx = false;
    r = ds4_state_cache_next_candidate(c, "approx-only", 8192, cursor, &borrowed, &n, &approx);
    assert(r && approx && n == b.n);
    assert(ds4_state_cache_close(c, err, sizeof(err)) != 0); /* A live pin owns the source. */
    ds4_state_cache_release(r);
    uint32_t source_start;
    r = ds4_state_cache_find_suffix(c, "native-model-profile", 8192, ids + b.n - 64u, 64u, &source_start);
    assert(r && source_start == b.n - 64u); ds4_state_cache_release(r);
    assert(ds4_state_cache_close(c, err, sizeof(err)) == 0);

    /* Reopening with a smaller SSD budget evicts unpinned endpoints/pages. */
    c = ds4_state_cache_open(dir, hot, first.ssd_bytes + 65536u, err, sizeof(err)); assert(c);
    ds4_state_cache_get_stats(c, &two);
    assert(two.endpoint_evictions > 0 && two.endpoints < 3);
    assert(two.ssd_bytes <= first.ssd_bytes + 65536u);
    assert(ds4_state_cache_close(c, err, sizeof(err)) == 0);
    free(ids); free(a.p); free(a.pre); free(b.p); free(b.pre);
    puts("state-cache: dedup, RAM spill, SSD restart, immutable sources, discovery and budgets PASS");
    return 0;
}
