/* Real payload loader/source parser; no model, tensors or GPU are opened. */
#include "../ds4.c"
#include "../speed-bench/suffix_snapshot_compare.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "diagnostic test:%d %s\n", __LINE__, #x); abort(); } } while (0)

static void put_u32(uint8_t *p, uint32_t x) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(x >> (i * 8));
}

int main(void) {
#ifdef DS4_HAS_DEEPSEEK41_GPU
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_engine engine = {0};
    ds4_session s = {.engine = &engine, .frontier = {.ctx_size = 8}};
    s.ds41.graph.ctx = 8;
    const uint64_t len = 52u + ds41_payload_body_bytes(&s.ds41.graph, 1);
    ds4_session_snapshot canonical = {.ptr = calloc(1, (size_t)len), .len = len, .cap = len};
    ds4_session_snapshot diagnostic = {.ptr = calloc(1, (size_t)len), .len = len, .cap = len};
    CHECK(canonical.ptr && diagnostic.ptr);
    const uint32_t h[13] = {DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION,
        8, 1, 128, 128, 9, 1, 40, 512, 128, DS4_N_VOCAB, 0x413431u};
    for (unsigned i = 0; i < 13; i++) put_u32(canonical.ptr + i * 4, h[i]);
    put_u32(canonical.ptr + 52, 7); /* Valid checkpoint token. */
    memcpy(diagnostic.ptr, canonical.ptr, (size_t)len);
    put_u32(diagnostic.ptr, DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC);
    ds4_suffix_cache cache = {.state = canonical, .engine = &engine, .ctx_size = 8};
    ds41_suffix_payload view = {0};
    CHECK(ds41_suffix_payload_view(&s, &cache, &view));
    CHECK(view.end == 1 && ds41_suffix_u32(view.tokens) == 7);
    const ds4_model_config config = ds41_native_config();
    uint32_t native_end = 0;
    char native_error[160] = {0};
    CHECK(ds4_native_graph_suffix_validate_prefix41(&s.ds41.graph, &config,
        &cache, &native_end, native_error, sizeof(native_error)) == 0);
    CHECK(native_end == 1);
    cache.state = diagnostic;
    CHECK(!ds41_suffix_payload_view(&s, &cache, &view));
    CHECK(ds4_native_graph_suffix_validate_prefix41(&s.ds41.graph, &config,
        &cache, &native_end, native_error, sizeof(native_error)) != 0);
    CHECK(suffix_diagnostic_equal_canonical(&diagnostic, &canonical));
    diagnostic.ptr[52] ^= 1;
    CHECK(!suffix_diagnostic_equal_canonical(&diagnostic, &canonical));
    diagnostic.ptr[52] ^= 1;
    diagnostic.ptr[4] ^= 1;
    CHECK(!suffix_diagnostic_equal_canonical(&diagnostic, &canonical));
    diagnostic.ptr[4] ^= 1;
    put_u32(diagnostic.ptr, DS4_SESSION_PAYLOAD_MAGIC);
    CHECK(!suffix_diagnostic_equal_canonical(&diagnostic, &canonical));
    put_u32(diagnostic.ptr, DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC);

    /* This actual public loader must reject the tag before inspecting an
     * engine or invoking any backend. It also preserves prior provenance. */
    ds4_session engine_less = {.frontier = {
        .approximate_suffix = true, .checkpoint_valid = true}};
    char err[160] = {0};
    CHECK(ds4_session_load_snapshot(&engine_less, &diagnostic, err, sizeof(err)) != 0);
    CHECK(strstr(err, "unsupported session payload"));
    CHECK(engine_less.frontier.approximate_suffix && engine_less.frontier.checkpoint_valid);
    FILE *fp = fmemopen(diagnostic.ptr, (size_t)len, "rb");
    CHECK(fp);
    CHECK(ds4_session_load_payload(&engine_less, fp, len, err, sizeof(err)) != 0);
    CHECK(strstr(err, "unsupported session payload"));
    CHECK(fclose(fp) == 0);
    ds4_session_snapshot_free(&canonical);
    ds4_session_snapshot_free(&diagnostic);
    puts("PASS: canonical DS41 source parser accepts valid bytes; diagnostic tag rejected by real source parser and engine-less payload/snapshot loaders; comparison normalizes only magic; no model or GPU opened");
    return 0;
#else
    fputs("test requires compiled DS41 support, without opening it\n", stderr);
    return 2;
#endif
}
