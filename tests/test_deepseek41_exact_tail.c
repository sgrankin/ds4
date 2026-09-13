/* Model-dependent regression for scalar-equivalent layer-major tail prefill. */
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
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL PROMPT\n", argv[0]); return 2; }
    int rc = 1;
    char err[256] = {0};
    FILE *fp = fopen(argv[2], "rb");
    char *text = calloc(131073, 1);
    ds4_engine *e = NULL; ds4_session *s = NULL; ds4_tokens tokens = {0};
    ds4_session_snapshot initial = {0}, expected = {0}, actual = {0};
    CHECK(fp && text && fread(text, 1, 131072, fp));
    fclose(fp); fp = NULL;
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 8192, .ssd_streaming = true, .power_percent = 100};
    CHECK(ds4_engine_open(&e, &opt) == 0);
    ds4_tokenize_text(e, text, &tokens); CHECK(tokens.len > 5120);
    CHECK(ds4_session_create(&s, e, 8192) == 0);
    ds4_tokens prefix = tokens; prefix.len = 4096;
    CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(s, &initial, err, sizeof(err)) == 0);
    unsigned displays = 0;
    ds4_session_set_progress(s, progress, &displays);
    const int tails[] = {257, 513};
    for (size_t c = 0; c < sizeof(tails)/sizeof(tails[0]); c++) {
        prefix.len = 4096 + tails[c];
        CHECK(unsetenv("DS4_METAL_V41_EXACT_SHORT_PREFILL") == 0);
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
        CHECK(setenv("DS4_METAL_V41_EXACT_SHORT_PREFILL", "1", 1) == 0);
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
            "scalar %.3fs, batch %.3fs\n", tails[c], scalar_seconds, batch_seconds);
        fflush(stdout);
    }
    rc = 0;
done:
    if (fp) fclose(fp);
    unsetenv("DS4_METAL_V41_EXACT_SHORT_PREFILL");
    free(text); ds4_tokens_free(&tokens);
    ds4_session_snapshot_free(&initial); ds4_session_snapshot_free(&expected); ds4_session_snapshot_free(&actual);
    ds4_session_free(s); ds4_engine_close(e); return rc;
}
