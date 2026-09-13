/* Model-dependent regression for selected-expert small layer-major prefill. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void progress(void *ud, const char *event, int current, int total) {
    (void)current; (void)total;
    if (!strcmp(event, "prefill_display")) (*(unsigned *)ud)++;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL PROMPT [PREFIX [TAIL...]]\n", argv[0]); return 2; }
    const int initial_tokens = argc >= 4 ? atoi(argv[3]) : 512;
    if (initial_tokens < 1 || initial_tokens > 90000) return 2;
    int max_tail = 40;
    for (int i = 4; i < argc; i++) {
        char *end = NULL;
        long tail = strtol(argv[i], &end, 10);
        if (end == argv[i] || *end || tail < 2 || tail >= 768) return 2;
        if (tail > max_tail) max_tail = (int)tail;
    }
    if (initial_tokens + max_tail + 8 > 100000) return 2;
    int rc = 1;
    char err[256] = {0};
    FILE *fp = fopen(argv[2], "rb");
    char *text = calloc(131073, 1);
    ds4_engine *e = NULL; ds4_session *s = NULL; ds4_tokens tokens = {0};
    ds4_session_snapshot initial = {0}, expected = {0}, actual = {0};
    CHECK(fp && text && fread(text, 1, 131072, fp));
    fclose(fp); fp = NULL;
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 100000, .ssd_streaming = true, .power_percent = 100};
    const char *cache_env = getenv("DS4_TEST_EXPERT_CACHE_GB");
    if (cache_env && cache_env[0]) {
        char *end = NULL;
        unsigned long gb = strtoul(cache_env, &end, 10);
        CHECK(end != cache_env && !*end && gb >= 4 && gb <= 96);
        opt.ssd_streaming_cache_bytes = (uint64_t)gb << 30;
    }
    CHECK(ds4_engine_open(&e, &opt) == 0);
    ds4_tokenize_text(e, text, &tokens); CHECK(tokens.len > initial_tokens + max_tail + 8);
    CHECK(ds4_session_create(&s, e, 100000) == 0);
    ds4_tokens prefix = tokens; prefix.len = initial_tokens;
    CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(s, &initial, err, sizeof(err)) == 0);
    unsigned displays = 0;
    ds4_session_set_progress(s, progress, &displays);
    const int tails[] = {2, 3, 4, 5, 6, 7, 8, 9, 17, 40};
    for (size_t c = 0; c < (argc > 4 ? (size_t)(argc - 4) : sizeof(tails)/sizeof(tails[0])); c++) {
        const int tail = argc > 4 ? atoi(argv[4 + c]) : tails[c];
        prefix.len = initial_tokens + tail;
        CHECK(setenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS", "1", 1) == 0);
        CHECK(ds4_session_load_snapshot(s, &initial, err, sizeof(err)) == 0);
        displays = 0;
        double start = seconds();
        CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
        CHECK(displays == 0);
        const double scalar_seconds = seconds() - start;
        /* Also cover continuation after the tail, not just its output head. */
        for (int i = 0; i < 8; i++)
            CHECK(ds4_session_eval(s, tokens.v[prefix.len + i], err, sizeof(err)) == 0);
        CHECK(ds4_session_save_snapshot(s, &expected, err, sizeof(err)) == 0);
        CHECK(unsetenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") == 0);
        CHECK(ds4_session_load_snapshot(s, &initial, err, sizeof(err)) == 0);
        displays = 0;
        start = seconds();
        CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
        CHECK(displays > 0);
        const double batch_seconds = seconds() - start;
        for (int i = 0; i < 8; i++)
            CHECK(ds4_session_eval(s, tokens.v[prefix.len + i], err, sizeof(err)) == 0);
        CHECK(ds4_session_save_snapshot(s, &actual, err, sizeof(err)) == 0);
        CHECK(actual.len == expected.len && !memcmp(actual.ptr, expected.ptr, actual.len));
        printf("PASS: tail=%d plus 8 decoded tokens, full snapshot bit-identical; "
            "scalar %.3fs, batch %.3fs\n", tail, scalar_seconds, batch_seconds);
        fflush(stdout);
    }
    rc = 0;
done:
    if (fp) fclose(fp);
    unsetenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS");
    free(text); ds4_tokens_free(&tokens);
    ds4_session_snapshot_free(&initial); ds4_session_snapshot_free(&expected); ds4_session_snapshot_free(&actual);
    ds4_session_free(s); ds4_engine_close(e); return rc;
}
