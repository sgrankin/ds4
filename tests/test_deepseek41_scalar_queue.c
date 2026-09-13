/* Model-dependent regression: ordered scalar submission must preserve every
 * logit row and the complete serialized continuation state. Run explicitly. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL PROMPT [VISION]\n", argv[0]);
        return 2;
    }
    int rc = 1;
    char err[256] = {0};
    FILE *fp = fopen(argv[2], "rb");
    char *text = calloc(65537, 1);
    ds4_engine *engine = NULL;
    ds4_session *s = NULL;
    ds4_tokens tokens = {0};
    ds4_session_snapshot initial = {0}, expected = {0}, actual = {0};
    float *rows = NULL, *row = NULL;
    CHECK(fp && text);
    CHECK(fread(text, 1, 65536, fp) > 0);
    fclose(fp); fp = NULL;
    ds4_engine_options opt = {.model_path = argv[1],
        .vision_path = argc == 4 ? argv[3] : NULL,
        .backend = DS4_BACKEND_METAL, .context_size = 4096,
        .ssd_streaming = true, .power_percent = 100};
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &tokens);
    CHECK(tokens.len >= 544);
    CHECK(ds4_session_create(&s, engine, 4096) == 0);
    ds4_tokens prefix = tokens; prefix.len = 512;
    CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(s, &initial, err, sizeof(err)) == 0);
    int vocab = ds4_engine_vocab_size(engine);
    size_t bytes = (size_t)vocab * sizeof(float);
    rows = malloc(bytes * 32); row = malloc(bytes);
    CHECK(rows && row);
    CHECK(setenv("DS4_METAL_DISABLE_V41_SCALAR_QUEUE", "1", 1) == 0);
    for (int i = 0; i < 32; i++) {
        CHECK(ds4_session_eval(s, tokens.v[512 + i], err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(s, rows + (size_t)i * vocab, vocab) == vocab);
    }
    CHECK(ds4_session_save_snapshot(s, &expected, err, sizeof(err)) == 0);
    CHECK(ds4_session_load_snapshot(s, &initial, err, sizeof(err)) == 0);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_SCALAR_QUEUE") == 0);
    for (int i = 0; i < 32; i++) {
        CHECK(ds4_session_eval(s, tokens.v[512 + i], err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(s, row, vocab) == vocab);
        CHECK(memcmp(rows + (size_t)i * vocab, row, bytes) == 0);
    }
    CHECK(ds4_session_save_snapshot(s, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == expected.len);
    CHECK(memcmp(actual.ptr, expected.ptr, actual.len) == 0);
    puts("PASS: 32 full logit rows and serialized continuation state are bit-identical");
    rc = 0;
done:
    if (fp) fclose(fp);
    unsetenv("DS4_METAL_DISABLE_V41_SCALAR_QUEUE");
    free(text); free(rows); free(row);
    ds4_session_snapshot_free(&initial);
    ds4_session_snapshot_free(&expected);
    ds4_session_snapshot_free(&actual);
    ds4_tokens_free(&tokens);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return rc;
}
