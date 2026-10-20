/* Compare a full context-edit rebuild with exact saved-prefix restoration.
 * No stale suffix state is retained. A staged replay checks restoration itself;
 * cold replay is checked separately because changing batch boundaries can
 * change floating-point results even when both computations are causal. */
#include "ds4.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "context-frontier: line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

static double ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END)) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0 || n > 64 * 1024 * 1024 || fseek(fp, 0, SEEK_SET)) { fclose(fp); return NULL; }
    char *text = malloc((size_t)n + 1);
    if (!text) { fclose(fp); return NULL; }
    bool ok = fread(text, 1, (size_t)n, fp) == (size_t)n;
    if (fclose(fp)) ok = false;
    if (!ok) { free(text); return NULL; }
    text[n] = 0;
    return text;
}

static int number(const char *s) {
    char *end;
    long n = strtol(s, &end, 10);
    if (s == end || *end || n < 1 || n > 1048576) return -1;
    return (int)n;
}

static void append(ds4_tokens *out, const ds4_tokens *in, int start, int count) {
    for (int i = 0; i < count; i++) ds4_tokens_push(out, in->v[start + i]);
}

static uint64_t hash(const void *p, uint64_t n) {
    const uint8_t *b = p;
    uint64_t h = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < n; i++) h = (h ^ b[i]) * UINT64_C(1099511628211);
    return h;
}

static void drift(const float *a, const float *b, int n, int *mismatch, double *maximum, double *rmse) {
    *mismatch = 0; *maximum = 0; *rmse = 0;
    for (int i = 0; i < n; i++) {
        if (memcmp(a + i, b + i, sizeof(float))) (*mismatch)++;
        double d = fabs((double)a[i] - b[i]);
        if (d > *maximum) *maximum = d;
        *rmse += d * d;
    }
    *rmse = sqrt(*rmse / n);
}

int main(int argc, char **argv) {
    const char *model = NULL, *source = "speed-bench/promessi_sposi.txt", *tokens_path = NULL;
    int prefix_n = 2048, old_n = 256, new_n = 256, tail_n = 512;
    int ctx = 8192, chunk = 2048, repeat = 1, decode = 4;
    uint64_t cache_bytes = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("usage: context_frontier_bench --model MODEL [--prompt-file FILE | --tokens-file FILE] [--prefix N] [--old-span N] [--new-span N] [--tail N] [--ctx N] [--prefill-chunk N] [--repeat N] [--decode N] [--cache-gb N]");
            return 0;
        }
        if (i + 1 == argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        const char *opt = argv[i], *value = argv[++i];
        if (!strcmp(opt, "--model")) model = value;
        else if (!strcmp(opt, "--prompt-file")) source = value;
        else if (!strcmp(opt, "--tokens-file")) tokens_path = value;
        else {
            const int n = number(value);
            if (n < 0) { fprintf(stderr, "invalid number for %s\n", opt); return 2; }
            if (!strcmp(opt, "--prefix")) prefix_n = n;
            else if (!strcmp(opt, "--old-span")) old_n = n;
            else if (!strcmp(opt, "--new-span")) new_n = n;
            else if (!strcmp(opt, "--tail")) tail_n = n;
            else if (!strcmp(opt, "--ctx")) ctx = n;
            else if (!strcmp(opt, "--prefill-chunk")) chunk = n;
            else if (!strcmp(opt, "--repeat")) repeat = n;
            else if (!strcmp(opt, "--decode")) decode = n;
            else if (!strcmp(opt, "--cache-gb") && n <= 96) cache_bytes = (uint64_t)n << 30;
            else { fprintf(stderr, "unknown option %s\n", opt); return 2; }
        }
    }
    if (!model || (int64_t)prefix_n + old_n + tail_n >= ctx ||
        (int64_t)prefix_n + new_n + tail_n + decode >= ctx ||
        repeat > 32 || decode > 64 || chunk > 8192 || ctx > 131072) {
        fprintf(stderr, "model is required and both edited histories must fit in --ctx\n");
        return 2;
    }
    int rc = 1;
    char err[256] = {0};
    char *text = tokens_path ? NULL : read_text(source);
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens corpus = {0}, prefix = {0}, old = {0}, edited = {0};
    ds4_session_snapshot frontier = {0}, old_state = {0}, staged = {0}, observed = {0};
    float *reference = NULL, *cold = NULL, *row = NULL;
    int *continuation = NULL;
    CHECK(tokens_path || text);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = ctx, .prefill_chunk = (uint32_t)chunk,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = cache_bytes, .power_percent = 100};
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    if (tokens_path) {
        FILE *fp = fopen(tokens_path, "r");
        CHECK(fp);
        int token, got;
        while ((got = fscanf(fp, " %d", &token)) == 1) {
            if (token < 0 || token >= ds4_engine_vocab_size(engine) || corpus.len >= 1048576) {
                fclose(fp);
                snprintf(err, sizeof(err), "invalid token source");
                CHECK(false);
            }
            ds4_tokens_push(&corpus, token);
        }
        bool valid = got == EOF && !ferror(fp);
        if (fclose(fp)) valid = false;
        CHECK(valid);
        source = tokens_path;
    } else ds4_tokenize_text(engine, text, &corpus);
    free(text); text = NULL;
    CHECK((int64_t)corpus.len >= (int64_t)prefix_n + old_n + tail_n + new_n);
    append(&prefix, &corpus, 0, prefix_n);
    append(&old, &corpus, 0, prefix_n + old_n + tail_n);
    append(&edited, &corpus, 0, prefix_n);
    append(&edited, &corpus, old.len, new_n);
    append(&edited, &corpus, prefix_n + old_n, tail_n);
    CHECK(edited.v[prefix_n] != old.v[prefix_n]);
    CHECK(ds4_session_create(&session, engine, ctx) == 0);
    const int vocab = ds4_engine_vocab_size(engine);
    const size_t logits_bytes = (size_t)vocab * sizeof(float);
    reference = malloc(logits_bytes * (decode + 1));
    cold = malloc(logits_bytes);
    row = malloc(logits_bytes);
    continuation = malloc((size_t)decode * sizeof(int));
    CHECK(reference && cold && row && continuation);
    CHECK(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    double t = ms();
    CHECK(ds4_session_save_snapshot(session, &frontier, err, sizeof(err)) == 0);
    const double save_ms = ms() - t;
    CHECK(ds4_session_sync(session, &old, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(session, &old_state, err, sizeof(err)) == 0);

    /* Reference: rebuild the edited context with exactly the saved boundary.
     * This must match the restored-prefix path bit for bit, including decode. */
    ds4_session_invalidate(session);
    CHECK(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(session, &edited, err, sizeof(err)) == 0);
    CHECK(ds4_session_copy_logits(session, reference, vocab) == vocab);
    CHECK(ds4_session_save_snapshot(session, &staged, err, sizeof(err)) == 0);
    for (int i = 0; i < decode; i++) {
        continuation[i] = ds4_session_argmax(session);
        CHECK(ds4_session_eval(session, continuation[i], err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(session, reference + (size_t)(i + 1) * vocab, vocab) == vocab);
    }
    fprintf(stderr, "FRONTIER source=%s prefix=%d old_span=%d new_span=%d tail=%d ctx=%d chunk=%d snapshot_bytes=%llu save_ms=%.3f edited_token_hash=%016llx\n",
        source, prefix_n, old_n, new_n, tail_n, ctx, chunk,
        (unsigned long long)frontier.len, save_ms,
        (unsigned long long)hash(edited.v, (uint64_t)edited.len * sizeof(int)));
    puts("trial,mode,prefix,new_tokens,snapshot_bytes,save_ms,restore_ms,sync_ms,total_ms,staged_logit_mismatch,cold_logit_mismatch,cold_max_abs,cold_rmse,staged_state_equal,decode_rows_equal,logit_hash,state_hash");
    static const int order[] = {0, 1, 1, 0};
    bool have_cold = false;
    for (int trial = 0; trial < repeat * 4; trial++) {
        const int candidate = order[trial % 4];
        CHECK(ds4_session_load_snapshot(session, &old_state, err, sizeof(err)) == 0);
        const double t0 = ms();
        if (candidate) CHECK(ds4_session_load_snapshot(session, &frontier, err, sizeof(err)) == 0);
        const double t1 = ms();
        CHECK(ds4_session_sync(session, &edited, err, sizeof(err)) == 0);
        const double t2 = ms();
        CHECK(ds4_session_copy_logits(session, row, vocab) == vocab);
        if (!candidate && !have_cold) { memcpy(cold, row, logits_bytes); have_cold = true; }
        int staged_mismatch, cold_mismatch;
        double max_abs, rmse, discard1, discard2;
        drift(reference, row, vocab, &staged_mismatch, &discard1, &discard2);
        drift(cold, row, vocab, &cold_mismatch, &max_abs, &rmse);
        CHECK(ds4_session_save_snapshot(session, &observed, err, sizeof(err)) == 0);
        const bool state_equal = staged.len == observed.len && !memcmp(staged.ptr, observed.ptr, staged.len);
        const uint64_t logit_hash = hash(row, logits_bytes), state_hash = hash(observed.ptr, observed.len);
        bool decode_equal = true;
        for (int i = 0; i < decode; i++) {
            CHECK(ds4_session_eval(session, continuation[i], err, sizeof(err)) == 0);
            CHECK(ds4_session_copy_logits(session, row, vocab) == vocab);
            if (memcmp(reference + (size_t)(i + 1) * vocab, row, logits_bytes)) decode_equal = false;
        }
        printf("%d,%s,%d,%d,%llu,%.3f,%.3f,%.3f,%.3f,%d,%d,%.9g,%.9g,%d,%d,%016llx,%016llx\n",
            trial, candidate ? "restore" : "rebuild", prefix_n, edited.len - prefix_n,
            (unsigned long long)frontier.len, save_ms, t1 - t0, t2 - t1, t2 - t0,
            staged_mismatch, cold_mismatch, max_abs, rmse, state_equal, decode_equal,
            (unsigned long long)logit_hash, (unsigned long long)state_hash);
        fflush(stdout);
        if (candidate && (staged_mismatch || !state_equal || !decode_equal)) {
            snprintf(err, sizeof(err), "restored frontier differs from partition-matched replay");
            CHECK(false);
        }
    }
    rc = 0;
done:
    free(text); free(reference); free(cold); free(row); free(continuation);
    ds4_session_snapshot_free(&frontier); ds4_session_snapshot_free(&old_state);
    ds4_session_snapshot_free(&staged); ds4_session_snapshot_free(&observed);
    ds4_tokens_free(&corpus); ds4_tokens_free(&prefix); ds4_tokens_free(&old); ds4_tokens_free(&edited);
    ds4_session_free(session); ds4_engine_close(engine);
    return rc;
}
