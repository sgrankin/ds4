/* Semantic oracle for genuine same-position stale-suffix reuse.
 * Prefix/edit/suffix partitions match between exact and approximate paths.
 * Greedy decoding uses untouched logits, independent of teacher forcing. */
#include "ds4.h"
#include "suffix_snapshot_compare.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "suffix-fact:%d: %s: %s\n", __LINE__, #x, err); \
    goto done; } } while (0)

typedef struct {
    double gold_logprob, stale_logprob, edit_ms;
    char *greedy;
    ds4_tokens generated;
    bool stopped, gold_answer, stale_answer;
} result;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void phase(const char *name, double started) {
    fprintf(stderr, "PHASE %s complete_ms=%.3f\n", name, now_ms() - started);
}

static bool cancel_after_first_check(void *ud) {
    int *calls = ud; return ++*calls > 1;
}

static char *read_text(const char *dir, const char *name) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path))
        return NULL;
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > 1024 * 1024 || fseek(f, 0, SEEK_SET)) {
        fclose(f); return NULL;
    }
    char *s = malloc((size_t)n + 1);
    bool ok = s && fread(s, 1, (size_t)n, f) == (size_t)n;
    if (fclose(f)) ok = false;
    if (!ok) { free(s); return NULL; }
    s[n] = 0; return s;
}

static int positive(const char *s) {
    char *end; long n = strtol(s, &end, 10);
    return end == s || *end || n < 1 || n > 131072 ? -1 : (int)n;
}

static void append(ds4_tokens *dst, const ds4_tokens *src, int offset, int n) {
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[offset + i]);
}

static bool pad(ds4_tokens *dst, const ds4_tokens *filler, int wanted) {
    if (!filler->len || wanted < dst->len) return false;
    int n = wanted - dst->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, filler->v[i % filler->len]);
    return dst->len == wanted;
}

static double log_z(const float *z, int n) {
    double mx = -INFINITY, sum = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(z[i])) return NAN;
        if (z[i] > mx) mx = z[i];
    }
    for (int i = 0; i < n; i++) sum += exp((double)z[i] - mx);
    return mx + log(sum);
}

static int argmax(const float *z, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    return best;
}

static void distance(const float *p, const float *q, int n,
                     double *kl, double *tv) {
    double lp = log_z(p, n), lq = log_z(q, n); *kl = *tv = 0;
    for (int i = 0; i < n; i++) {
        double a = exp((double)p[i] - lp), b = exp((double)q[i] - lq);
        *kl += a * (((double)p[i] - lp) - ((double)q[i] - lq));
        *tv += fabs(a - b) * .5;
    }
    if (*kl < 0 && *kl > -1e-10) *kl = 0;
}

static bool answer_equal(const char *text, const char *answer) {
    while (isspace((unsigned char)*text)) text++;
    size_t n = strlen(text);
    while (n && isspace((unsigned char)text[n - 1])) n--;
    return n == strlen(answer) && !memcmp(text, answer, n);
}

static bool query_is_fresh(int endpoint, int total, int query_tokens) {
    return query_tokens > 0 && query_tokens <= total && endpoint >= 0 &&
           endpoint <= total - query_tokens;
}

static void json_text(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 32 || *p >= 127) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

static void json_tokens(FILE *f, const ds4_tokens *t) {
    fputc('[', f);
    for (int i = 0; i < t->len; i++) fprintf(f, "%s%d", i ? "," : "", t->v[i]);
    fputc(']', f);
}

static FILE *output_file(const char *dir, const char *name, const char *mode) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path))
        return NULL;
    return fopen(path, mode);
}

static int close_file(FILE **f) {
    FILE *owned = *f; *f = NULL; return fclose(owned);
}

static int write_rendered(FILE *f, ds4_engine *e, const ds4_tokens *t) {
    for (int i = 0; i < t->len; i++) {
        size_t n; char *piece = ds4_token_text(e, t->v[i], &n);
        bool ok = piece && fwrite(piece, 1, n, f) == n; free(piece);
        if (!ok) return 1;
    }
    return ferror(f) != 0;
}

typedef struct {
    const uint8_t *window[40], *compressed[4], *index[4];
} cache_view;

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
        (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* Inspect the durable DS41 diagnostic format independently of the graft's
 * tensor copy offsets. This is a test-only parser of the existing 13-u32
 * header, not a canonical cache import API. */
static bool inspect_cache(const ds4_session_snapshot *snap, int pos, int vocab,
                          cache_view *v) {
    if (!snap->ptr || snap->len < 52 || pos < 1 ||
        (suffix_snapshot_magic(snap) != DS4_SESSION_PAYLOAD_MAGIC &&
         suffix_snapshot_magic(snap) != DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC) ||
        read_u32(snap->ptr + 28) != (uint32_t)pos ||
        read_u32(snap->ptr + 32) != 40 || read_u32(snap->ptr + 36) != 512 ||
        read_u32(snap->ptr + 40) != 128 ||
        read_u32(snap->ptr + 44) != (uint32_t)vocab ||
        read_u32(snap->ptr + 48) != UINT32_C(0x413431)) return false;
    uint64_t off = 52u + ((uint64_t)pos + (uint32_t)vocab) * 4u;
    uint32_t raw = pos < 128 ? (uint32_t)pos : 128u;
    for (int il = 0; il < 40; il++) {
        uint64_t n = (uint64_t)raw * 512u * 4u;
        if (off > snap->len || n > snap->len - off) return false;
        v->window[il] = snap->ptr + off; off += n;
    }
    for (int owner = 0; owner < 4; owner++) {
        uint32_t rows = (uint32_t)pos / (owner < 3 ? 2u : 1u);
        uint64_t n = (uint64_t)rows * 512u * 4u;
        if (off > snap->len || n > snap->len - off) return false;
        v->compressed[owner] = snap->ptr + off; off += n;
        n = (uint64_t)rows * 128u * 4u;
        if (off > snap->len || n > snap->len - off) return false;
        v->index[owner] = snap->ptr + off; off += n;
        if (owner < 3 && (pos & 1)) {
            n = 2u * 512u * 4u;
            if (off > snap->len || n > snap->len - off) return false;
            off += n;
        }
    }
    return off == snap->len;
}

static bool fresh_prefix_equal(const ds4_session_snapshot *prefix,
                               const ds4_session_snapshot *full,
                               int prefix_len, int full_len, int vocab,
                               int *raw_rows, int *compressed_rows) {
    cache_view a = {0}, b = {0};
    if (!inspect_cache(prefix, prefix_len, vocab, &a) ||
        !inspect_cache(full, full_len, vocab, &b)) return false;
    int first = full_len > 128 ? full_len - 128 : 0;
    *raw_rows = first < prefix_len ? prefix_len - first : 0;
    *compressed_rows = 0;
    for (int il = 0; il < 40; il++) {
        for (int p = first; p < prefix_len; p++) {
            uint64_t offset = (uint64_t)(p % 128) * 512u * 4u;
            if (memcmp(a.window[il] + offset, b.window[il] + offset, 512u * 4u))
                return false;
        }
    }
    for (int owner = 0; owner < 4; owner++) {
        int rows = prefix_len / (owner < 3 ? 2 : 1);
        if (memcmp(a.compressed[owner], b.compressed[owner], (size_t)rows * 512u * 4u) ||
            memcmp(a.index[owner], b.index[owner], (size_t)rows * 128u * 4u)) return false;
        *compressed_rows += rows;
    }
    return true;
}

static bool mock_cache(ds4_session_snapshot *s, uint32_t pos, uint32_t vocab) {
    uint64_t n = 52u + ((uint64_t)pos + vocab) * 4u +
        40u * (uint64_t)(pos < 128u ? pos : 128u) * 512u * 4u;
    for (uint32_t owner = 0; owner < 4; owner++) {
        n += (uint64_t)(pos / (owner < 3 ? 2u : 1u)) * 640u * 4u;
        if (owner < 3 && (pos & 1u)) n += 2u * 512u * 4u;
    }
    s->ptr = calloc(1, (size_t)n); if (!s->ptr) return false; s->len = s->cap = n;
    uint32_t h[13] = {DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION};
    h[7] = pos; h[8] = 40; h[9] = 512;
    h[10] = 128; h[11] = vocab; h[12] = UINT32_C(0x413431);
    for (int i = 0; i < 13; i++)
        for (int j = 0; j < 4; j++) s->ptr[i * 4 + j] = (uint8_t)(h[i] >> (j * 8));
    return true;
}

static bool ownership_parser_test(void) {
    ds4_session_snapshot a = {0}, b = {0}; cache_view view = {0};
    int raw = 0, compressed = 0; bool ok = false;
    if (!mock_cache(&a, 384, 3) || !mock_cache(&b, 448, 3) ||
        !inspect_cache(&b, 448, 3, &view)) goto done;
    b.ptr[0] ^= 1;
    if (inspect_cache(&b, 448, 3, &view)) goto done;
    b.ptr[0] ^= 1;
    if (!fresh_prefix_equal(&a, &b, 384, 448, 3, &raw, &compressed) ||
        raw != 64 || compressed != 960) goto done;
    /* Differences in grafted C rows are allowed; differences in any retained
     * prefix raw, compressed or index value must be detected. */
    ((uint8_t *)view.window[0])[0] ^= 1;
    if (!fresh_prefix_equal(&a, &b, 384, 448, 3, &raw, &compressed)) goto done;
    uint8_t *prefix_raw = (uint8_t *)view.window[0] + 64u * 512u * 4u;
    *prefix_raw ^= 1;
    if (fresh_prefix_equal(&a, &b, 384, 448, 3, &raw, &compressed)) goto done;
    *prefix_raw ^= 1; ((uint8_t *)view.compressed[2])[0] ^= 1;
    if (fresh_prefix_equal(&a, &b, 384, 448, 3, &raw, &compressed)) goto done;
    ((uint8_t *)view.compressed[2])[0] ^= 1; ((uint8_t *)view.index[3])[0] ^= 1;
    if (fresh_prefix_equal(&a, &b, 384, 448, 3, &raw, &compressed)) goto done;
    ((uint8_t *)view.index[3])[0] ^= 1; b.len--;
    if (inspect_cache(&b, 448, 3, &view)) goto done;
    b.len++; if (inspect_cache(&b, 449, 3, &view)) goto done;
    ok = true;
done:
    ds4_session_snapshot_free(&a); ds4_session_snapshot_free(&b); return ok;
}

static int prepare(ds4_session *s, const ds4_session_snapshot *a,
                   const ds4_session_snapshot *full, const ds4_tokens *ab,
                   const ds4_tokens *end, const ds4_tokens *prompt,
                   const ds4_suffix_cache *cache, uint32_t source_start,
                   bool approximate, ds4_suffix_relocation_stats *stats,
                   char *err, size_t errlen) {
    if (full && !approximate) return ds4_session_load_snapshot(s, full, err, errlen);
    if (ds4_session_load_snapshot(s, a, err, errlen) ||
        ds4_session_sync(s, ab, err, errlen)) return 1;
    if (approximate)
        return ds4_session_reuse_suffix_relocated(s, cache, prompt, source_start,
                                                   stats, err, errlen);
    return ds4_session_sync(s, end, err, errlen) ||
           ds4_session_sync(s, prompt, err, errlen);
}

static int greedy(ds4_session *s, ds4_engine *e, int limit, int vocab,
                  float *row, result *r, char *err, size_t errlen) {
    size_t size = 1; r->greedy = malloc(size);
    if (!r->greedy) return 1; r->greedy[0] = 0;
    for (int i = 0; i < limit; i++) {
        if (ds4_session_copy_logits(s, row, vocab) != vocab ||
            !isfinite(log_z(row, vocab))) return 1;
        int tok = argmax(row, vocab); ds4_tokens_push(&r->generated, tok);
        if (ds4_token_is_stop_for_think_mode(e, tok, DS4_THINK_NONE)) {
            r->stopped = true; break;
        }
        size_t n; char *piece = ds4_token_text(e, tok, &n);
        if (!piece || n > 16 * 1024 || size + n > 256 * 1024) {
            free(piece); return 1;
        }
        char *next = realloc(r->greedy, size + n);
        if (!next) { free(piece); return 1; }
        r->greedy = next; memcpy(next + size - 1, piece, n);
        size += n; next[size - 1] = 0; free(piece);
        if (ds4_session_eval(s, tok, err, errlen)) return 1;
    }
    return 0;
}

static int self_test(void) {
    const float p[3] = {logf(.6f), logf(.3f), logf(.1f)};
    const float q[3] = {logf(.3f), logf(.6f), logf(.1f)};
    const float shifted[3] = {p[0] + 4, p[1] + 4, p[2] + 4};
    double kl, tv; distance(p, q, 3, &kl, &tv);
    if (fabs(kl - .3 * log(2.0)) > 1e-7 || fabs(tv - .3) > 1e-7 ||
        fabs((log_z(q, 3) - q[0]) - (log_z(p, 3) - p[0]) - log(2.0)) > 1e-7)
        return 1;
    distance(p, shifted, 3, &kl, &tv);
    if (fabs(kl) > 1e-10 || tv > 1e-7 ||
        !answer_equal(" \n3141\n", "3141") || answer_equal("2718", "3141") ||
        answer_equal("3141.", "3141")) return 1;
    ds4_tokens a = {0}, filler = {0};
    ds4_tokens_push(&a, 7); ds4_tokens_push(&filler, 8); ds4_tokens_push(&filler, 9);
    bool ok = !pad(&a, &filler, 0) && pad(&a, &filler, 4) &&
        a.len == 4 && a.v[0] == 7 && a.v[1] == 8 && a.v[2] == 9 && a.v[3] == 8;
    ds4_tokens_free(&a); ds4_tokens_free(&filler);
    if (!ok || !ownership_parser_test() ||
        !query_is_fresh(384, 512, 128) || !query_is_fresh(383, 512, 128) ||
        query_is_fresh(385, 512, 128) || query_is_fresh(0, 512, 513) ||
        query_is_fresh(384, 512, 0) || query_is_fresh(-1, 512, 128)) return 1;
    puts("suffix-fact metrics, exact answer, padding, prefix ownership and fresh-query bounds: PASS; no model opened");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) return self_test();
    const char *model = NULL, *fixtures = NULL, *case_name = "code", *out = NULL;
    const char *gold_override = NULL, *stale_override = NULL;
    bool require_fresh_query = false, calibrate_original = false;
    int prefix_n = 1024, edit_n = 256, source_edit_n = 0, unchanged_n = 512, tail_n = 1;
    int limit = 16, ctx = 4096, chunk = 256, cache_gb = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("suffix_fact_oracle --model MODEL --fixtures DIR --out EXISTING_DIR "
                 "[--case code|secret|prior-secret] [--prefix N] [--edit N] [--source-edit N] [--unchanged N] "
                 "[--fresh-tail N] [--greedy N] [--ctx N] [--chunk N] [--cache-gb N] "
                 "[--gold-answer TEXT] [--stale-answer TEXT] [--require-fresh-query] [--calibrate-original]");
            return 0;
        }
        if (!strcmp(argv[i], "--require-fresh-query")) { require_fresh_query = true; continue; }
        if (!strcmp(argv[i], "--calibrate-original")) { calibrate_original = true; continue; }
        if (i + 1 == argc) return 2;
        const char *o = argv[i], *v = argv[++i];
        if (!strcmp(o, "--model")) model = v;
        else if (!strcmp(o, "--fixtures")) fixtures = v;
        else if (!strcmp(o, "--out")) out = v;
        else if (!strcmp(o, "--case")) case_name = v;
        else if (!strcmp(o, "--gold-answer")) gold_override = v;
        else if (!strcmp(o, "--stale-answer")) stale_override = v;
        else {
            int n = positive(v); if (n < 0) return 2;
            if (!strcmp(o, "--prefix")) prefix_n = n;
            else if (!strcmp(o, "--edit")) edit_n = n;
            else if (!strcmp(o, "--source-edit")) source_edit_n = n;
            else if (!strcmp(o, "--unchanged")) unchanged_n = n;
            else if (!strcmp(o, "--fresh-tail")) tail_n = n;
            else if (!strcmp(o, "--greedy")) limit = n;
            else if (!strcmp(o, "--ctx")) ctx = n;
            else if (!strcmp(o, "--chunk")) chunk = n;
            else if (!strcmp(o, "--cache-gb")) cache_gb = n;
            else return 2;
        }
    }
    if (!source_edit_n) source_edit_n = edit_n;
    bool prior_secret = !strcmp(case_name, "prior-secret");
    bool secret = !strcmp(case_name, "secret") || prior_secret;
    if (prior_secret) { require_fresh_query = true; calibrate_original = true; }
    int start = prefix_n + edit_n, total = start + unchanged_n, end_n = total - tail_n;
    int source_start = prefix_n + source_edit_n, source_total = source_start + unchanged_n;
    int source_end_n = source_total - tail_n;
    if (!model || !fixtures || !out || (!secret && strcmp(case_name, "code")) ||
        (start & 1) || source_start < start || (source_start - start) % 2 ||
        tail_n >= unchanged_n || limit > 128 || chunk > 8192 ||
        cache_gb > 96 || source_total + limit + 16 >= ctx) return 2;
    const char *gold_text = gold_override ? gold_override : prior_secret ? "UNKNOWN" : secret ? "REDACTED" : "3141";
    const char *stale_text = stale_override ? stale_override : secret ? "6829" : "2718";
    if (!*gold_text || !*stale_text || strlen(gold_text) > 128 || strlen(stale_text) > 128 ||
        !strcmp(gold_text, stale_text)) return 2;
    const char *names[5] = {"prefix.txt", "old-register.txt",
        secret ? "new-register-secret.txt" : "new-register-code.txt", "notes.txt",
        secret ? "question-secret.txt" : "question-code.txt"};
    int rc = 1, vocab = 0; char err[256] = {0}, *texts[5] = {0};
    ds4_engine *e = NULL; ds4_session *s = NULL;
    ds4_tokens parts[5] = {0}, a = {0}, old_b = {0}, new_b = {0}, c = {0};
    ds4_tokens old = {0}, prompt = {0}, old_ab = {0}, new_ab = {0};
    ds4_tokens old_end = {0}, new_end = {0}, answers[2] = {0};
    ds4_session_snapshot a_state = {0}, old_ab_state = {0}, old_final = {0};
    ds4_session_snapshot observed = {0}, fresh_final = {0}, new_ab_state = {0}; ds4_suffix_cache cache = {0};
    ds4_suffix_relocation_stats stats = {0}; result results[2] = {0}, original_result = {0};
    float *reference = NULL, *row = NULL, *endpoint = NULL; FILE *raw = NULL, *csv = NULL, *json = NULL, *render = NULL;
    double setup_ms = 0, identity_ms = 0, sum_kl = 0, sum_tv = 0, sum_dnll = 0;
    int flips = 0, equal_rows = 0, fresh_prefix_raw_rows = 0, fresh_prefix_compressed_rows = 0;
    for (int i = 0; i < 5; i++) { texts[i] = read_text(fixtures, names[i]); REQUIRE(texts[i]); }
    if (prior_secret) {
        /* The removed digits may appear only in the original register. */
        REQUIRE(!strstr(texts[0], stale_text) && !strstr(texts[2], stale_text) &&
                !strstr(texts[3], stale_text) && !strstr(texts[4], stale_text));
        REQUIRE(strstr(texts[1], stale_text));
    }
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = (uint32_t)ctx, .prefill_chunk = (uint32_t)chunk,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = (uint64_t)cache_gb << 30,
        .power_percent = 100};
    REQUIRE(ds4_engine_open(&e, &opt) == 0); REQUIRE(ds4_engine_is_deepseek41(e));
    REQUIRE(ds4_session_create(&s, e, ctx) == 0); vocab = ds4_engine_vocab_size(e);
    size_t bytes = (size_t)vocab * sizeof(float);
    for (int i = 0; i < 5; i++) { ds4_tokenize_text(e, texts[i], &parts[i]); REQUIRE(parts[i].len); }
    ds4_chat_begin(e, &a); ds4_tokens_push(&a, ds4_token_user(e)); append(&a, &parts[0], 0, parts[0].len);
    REQUIRE(pad(&a, &parts[3], prefix_n));
    REQUIRE(pad(&old_b, &parts[3], source_edit_n - parts[1].len)); append(&old_b, &parts[1], 0, parts[1].len);
    REQUIRE(pad(&new_b, &parts[3], edit_n - parts[2].len)); append(&new_b, &parts[2], 0, parts[2].len);
    ds4_chat_append_assistant_prefix(e, &parts[4], DS4_THINK_NONE);
    if (require_fresh_query) REQUIRE(query_is_fresh(end_n, total, parts[4].len) &&
                                     query_is_fresh(source_end_n, source_total, parts[4].len));
    REQUIRE(pad(&c, &parts[3], unchanged_n - parts[4].len)); append(&c, &parts[4], 0, parts[4].len);
    append(&old, &a, 0, a.len); append(&old, &old_b, 0, old_b.len); append(&old, &c, 0, c.len);
    append(&prompt, &a, 0, a.len); append(&prompt, &new_b, 0, new_b.len); append(&prompt, &c, 0, c.len);
    REQUIRE(old.len == source_total && prompt.len == total);
    append(&old_ab, &old, 0, source_start); append(&new_ab, &prompt, 0, start);
    append(&old_end, &old, 0, source_end_n); append(&new_end, &prompt, 0, end_n);
    ds4_tokenize_text(e, gold_text, &answers[0]); ds4_tokenize_text(e, stale_text, &answers[1]);
    for (int i = 0; i < 2; i++) { REQUIRE(answers[i].len && answers[i].len < 16); ds4_tokens_push(&answers[i], ds4_token_eos(e)); }
    reference = malloc(bytes * (size_t)answers[0].len); row = malloc(bytes); endpoint = malloc(bytes);
    REQUIRE(reference && row && endpoint);
    csv = output_file(out, "rows.csv", "wx"); raw = output_file(out, "logits.bin", "wbx");
    json = output_file(out, "summary.json", "wx"); REQUIRE(csv && raw && json);
    render = output_file(out, "old-prompt.txt", "wbx"); REQUIRE(render); REQUIRE(write_rendered(render, e, &old) == 0); REQUIRE(close_file(&render) == 0);
    render = output_file(out, "new-prompt.txt", "wbx"); REQUIRE(render); REQUIRE(write_rendered(render, e, &prompt) == 0); REQUIRE(close_file(&render) == 0);
    render = output_file(out, "tokens.json", "wx"); REQUIRE(render);
    fputs("{\"old\":", render); json_tokens(render, &old); fputs(",\"edited\":", render); json_tokens(render, &prompt);
    fputs(",\"gold\":", render); json_tokens(render, &answers[0]); fputs(",\"stale\":", render); json_tokens(render, &answers[1]); fputs("}\n", render);
    REQUIRE(close_file(&render) == 0);
    uint32_t header[4] = {UINT32_C(0x53554632), (uint32_t)vocab, (uint32_t)answers[0].len, (uint32_t)total};
    REQUIRE(fwrite(header, sizeof(header), 1, raw) == 1);
    REQUIRE(fwrite(answers[0].v, sizeof(int), (size_t)answers[0].len, raw) == (size_t)answers[0].len);
    double t = now_ms(), setup_started = t; REQUIRE(ds4_session_sync(s, &a, err, sizeof(err)) == 0); phase("old_A", t);
    REQUIRE(ds4_session_save_snapshot(s, &a_state, err, sizeof(err)) == 0);
    t = now_ms(); REQUIRE(ds4_session_sync(s, &old_ab, err, sizeof(err)) == 0); phase("old_B", t);
    REQUIRE(ds4_session_save_snapshot(s, &old_ab_state, err, sizeof(err)) == 0);
    t = now_ms(); REQUIRE(ds4_session_sync(s, &old_end, err, sizeof(err)) == 0); phase("old_C", t);
    REQUIRE(ds4_session_capture_suffix_cache(s, &cache, err, sizeof(err)) == 0);
    t = now_ms(); REQUIRE(ds4_session_sync(s, &old, err, sizeof(err)) == 0); phase("old_fresh_tail", t);
    REQUIRE(ds4_session_save_snapshot(s, &old_final, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(s, reference, vocab) == vocab); setup_ms = now_ms() - setup_started;
    REQUIRE(ds4_session_load_snapshot(s, &old_ab_state, err, sizeof(err)) == 0);
    t = now_ms(); REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &old, (uint32_t)source_start, &stats, err, sizeof(err)) == 0);
    identity_ms = now_ms() - t; REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab);
    REQUIRE(stats.raw_keys_repaired == 0 && stats.compressed_keys_repaired == 0 && stats.index_keys_repaired == 0);
    REQUIRE(!memcmp(reference, row, bytes));
    REQUIRE(ds4_session_suffix_diagnostic_snapshot(s, &observed, err, sizeof(err)) == 0);
    REQUIRE(suffix_diagnostic_equal_canonical(&observed, &old_final));
    REQUIRE(suffix_snapshot_magic(&observed) == DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC);
    REQUIRE(ds4_session_load_snapshot(s, &observed, err, sizeof(err)) != 0);
    REQUIRE(strstr(err, "unsupported session payload"));
    REQUIRE(ds4_session_is_approximate(s));
    REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab && !memcmp(reference, row, bytes));
    REQUIRE(ds4_session_suffix_diagnostic_snapshot(s, &observed, err, sizeof(err)) == 0);
    REQUIRE(suffix_diagnostic_equal_canonical(&observed, &old_final));
    fprintf(stderr, "DIAGNOSTIC_IMPORT distinct_magic=1 import_rejected=1 graph_and_logits_unchanged=1\n");
    err[0] = 0;
    REQUIRE(ds4_session_is_approximate(s) && ds4_session_payload_bytes(s) == 0);
    REQUIRE(ds4_session_save_snapshot(s, &observed, err, sizeof(err)) != 0); err[0] = 0;
    phase("identity_graph_and_logits", t);
    if (calibrate_original) {
        REQUIRE(ds4_session_load_snapshot(s, &old_final, err, sizeof(err)) == 0);
        REQUIRE(!ds4_session_is_approximate(s));
        REQUIRE(ds4_session_copy_logits(s, endpoint, vocab) == vocab);
        for (int branch = 0; branch < 2; branch++) {
            REQUIRE(ds4_session_load_snapshot(s, &old_final, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab && !memcmp(endpoint, row, bytes));
            for (int j = 0; j < answers[branch].len; j++) {
                REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab);
                double z = log_z(row, vocab); REQUIRE(isfinite(z));
                double *lp = branch ? &original_result.stale_logprob : &original_result.gold_logprob;
                *lp += row[answers[branch].v[j]] - z;
                if (j + 1 < answers[branch].len) REQUIRE(ds4_session_eval(s, answers[branch].v[j], err, sizeof(err)) == 0);
            }
        }
        REQUIRE(ds4_session_load_snapshot(s, &old_final, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab && !memcmp(endpoint, row, bytes));
        REQUIRE(greedy(s, e, limit, vocab, row, &original_result, err, sizeof(err)) == 0);
        original_result.gold_answer = answer_equal(original_result.greedy, gold_text);
        original_result.stale_answer = answer_equal(original_result.greedy, stale_text);
        fprintf(stderr, "ORIGINAL_CALIBRATION greedy=%s expected=%s known=%d gold_lp=%.9g stale_lp=%.9g endpoint_bytes_equal=1\n",
                original_result.greedy, stale_text, original_result.stale_answer,
                original_result.gold_logprob, original_result.stale_logprob);
    }
    /* Invalid candidates and cancellation before the first write preserve
     * the live exact prefix; cancellation after grafting invalidates it. */
    REQUIRE(ds4_session_load_snapshot(s, &a_state, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_sync(s, &new_ab, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_save_snapshot(s, &new_ab_state, err, sizeof(err)) == 0);
    /* All bounds/tokens in this exact source are otherwise valid. Changing
     * only its magic to DSSD must reject it before any graph mutation. */
    uint8_t canonical_magic[4]; memcpy(canonical_magic, cache.state.ptr, 4);
    for (unsigned i = 0; i < 4; i++)
        cache.state.ptr[i] = (uint8_t)(DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC >> (i * 8u));
    REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt,
                (uint32_t)source_start, &stats, err, sizeof(err)) != 0);
    memcpy(cache.state.ptr, canonical_magic, 4);
    REQUIRE(ds4_session_save_snapshot(s, &observed, err, sizeof(err)) == 0);
    REQUIRE(observed.len == new_ab_state.len &&
            !memcmp(observed.ptr, new_ab_state.ptr, (size_t)observed.len));
    fprintf(stderr, "DIAGNOSTIC_SOURCE tag_only_rejected=1 canonical_prefix_unchanged=1\n");
    err[0] = 0;
    REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start + 1u, &stats, err, sizeof(err)) != 0);
    if (start >= 128) REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)start - 128u, &stats, err, sizeof(err)) != 0);
    prompt.len = end_n; REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) != 0); prompt.len = total;
    int original = prompt.v[start]; prompt.v[start] = (original + 1) % vocab;
    REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) != 0); prompt.v[start] = original;
    cache.state.ptr[0] ^= 1; REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) != 0); cache.state.ptr[0] ^= 1;
    int cancel_calls = 1; ds4_session_set_cancel(s, cancel_after_first_check, &cancel_calls);
    REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) != 0);
    ds4_session_set_cancel(s, NULL, NULL);
    REQUIRE(ds4_session_save_snapshot(s, &observed, err, sizeof(err)) == 0);
    REQUIRE(observed.len == new_ab_state.len &&
            !memcmp(observed.ptr, new_ab_state.ptr, (size_t)observed.len));
    cancel_calls = 0; ds4_session_set_cancel(s, cancel_after_first_check, &cancel_calls);
    REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) != 0);
    ds4_session_set_cancel(s, NULL, NULL); REQUIRE(cancel_calls > 1);
    REQUIRE(!ds4_session_is_approximate(s) && ds4_session_pos(s) == 0 && ds4_session_payload_bytes(s) == 0);
    fprintf(stderr, "TRANSACTIONS invalid_candidates_unchanged=1 precancel_unchanged=1 postgraft_cancel_invalidated=1\n"); err[0] = 0;
    fprintf(csv, "step,target,fresh_nll,reuse_nll,delta_nll,kl,tv,argmax_flip,byte_equal\n");
    for (int mode = 0; mode < 2; mode++) {
        REQUIRE(ds4_session_load_snapshot(s, &a_state, err, sizeof(err)) == 0);
        REQUIRE(!ds4_session_is_approximate(s)); t = now_ms();
        REQUIRE(ds4_session_sync(s, &new_ab, err, sizeof(err)) == 0); phase(mode ? "reuse_B_prime" : "fresh_B_prime", t);
        double suffix_started = now_ms();
        if (mode) REQUIRE(ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)) == 0);
        else { REQUIRE(ds4_session_sync(s, &new_end, err, sizeof(err)) == 0); REQUIRE(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0); }
        phase(mode ? "relocated_C_and_fresh_tail" : "replayed_C_and_fresh_tail", suffix_started);
        results[mode].edit_ms = now_ms() - t;
        REQUIRE(ds4_session_copy_logits(s, endpoint, vocab) == vocab);
        if (!mode) REQUIRE(ds4_session_save_snapshot(s, &fresh_final, err, sizeof(err)) == 0);
        else {
            REQUIRE(ds4_session_suffix_diagnostic_snapshot(s, &observed, err, sizeof(err)) == 0);
            REQUIRE(fresh_prefix_equal(&new_ab_state, &observed, start, total, vocab,
                                        &fresh_prefix_raw_rows, &fresh_prefix_compressed_rows));
            fprintf(stderr, "PREFIX_OWNERSHIP raw_rows_per_layer=%d compressed_rows=%d index_rows=%d bytes_equal=1\n",
                    fresh_prefix_raw_rows, fresh_prefix_compressed_rows, fresh_prefix_compressed_rows);
        }
        /* Gold sequence: raw full vocabulary retained at every target, including EOS. */
        for (int j = 0; j < answers[0].len; j++) {
            REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab);
            double z = log_z(row, vocab); REQUIRE(isfinite(z));
            results[mode].gold_logprob += row[answers[0].v[j]] - z;
            if (!mode) memcpy(reference + (size_t)j * vocab, row, bytes);
            else {
                const float *p = reference + (size_t)j * vocab;
                double kl, tv; distance(p, row, vocab, &kl, &tv); REQUIRE(isfinite(kl) && isfinite(tv));
                double pnll = log_z(p, vocab) - p[answers[0].v[j]], qnll = z - row[answers[0].v[j]];
                int flip = argmax(p, vocab) != argmax(row, vocab), equal = !memcmp(p, row, bytes);
                sum_kl += kl; sum_tv += tv; sum_dnll += qnll - pnll; flips += flip; equal_rows += equal;
                fprintf(csv, "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d\n", j, answers[0].v[j], pnll, qnll, qnll - pnll, kl, tv, flip, equal);
            }
            REQUIRE(fwrite(row, bytes, 1, raw) == 1);
            if (j + 1 < answers[0].len) REQUIRE(ds4_session_eval(s, answers[0].v[j], err, sizeof(err)) == 0);
        }
        /* Alternate old answer and free greedy answer each start at the same edited endpoint.
         * Approximate endpoints are rebuilt from exact A rather than published as canonical KV. */
        REQUIRE(prepare(s, &a_state, &fresh_final, &new_ab, &new_end, &prompt, &cache, (uint32_t)source_start, mode != 0, &stats, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab && !memcmp(endpoint, row, bytes));
        for (int j = 0; j < answers[1].len; j++) {
            REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab);
            double z = log_z(row, vocab); REQUIRE(isfinite(z));
            results[mode].stale_logprob += row[answers[1].v[j]] - z;
            if (j + 1 < answers[1].len) REQUIRE(ds4_session_eval(s, answers[1].v[j], err, sizeof(err)) == 0);
        }
        REQUIRE(prepare(s, &a_state, &fresh_final, &new_ab, &new_end, &prompt, &cache, (uint32_t)source_start, mode != 0, &stats, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab && !memcmp(endpoint, row, bytes));
        REQUIRE(greedy(s, e, limit, vocab, row, &results[mode], err, sizeof(err)) == 0);
        results[mode].gold_answer = answer_equal(results[mode].greedy, gold_text);
        results[mode].stale_answer = answer_equal(results[mode].greedy, stale_text);
        REQUIRE(ds4_session_is_approximate(s) == (mode != 0));
        fprintf(stderr, "FACT case=%s mode=%s C=%d tail=%d greedy=%s correct=%d stale=%d gold_lp=%.6f stale_lp=%.6f edit_ms=%.3f\n",
                case_name, mode ? "reuse" : "fresh", unchanged_n, tail_n, results[mode].greedy,
                results[mode].gold_answer, results[mode].stale_answer, results[mode].gold_logprob,
                results[mode].stale_logprob, results[mode].edit_ms);
    }
    bool semantic_controls_valid = results[0].gold_answer &&
        (!calibrate_original || original_result.stale_answer) &&
        (!prior_secret || (results[0].stopped && original_result.stopped));
    fputs("{\n  \"case\":", json); json_text(json, case_name);
    fputs(",\n  \"gold_answer_text\":", json); json_text(json, gold_text);
    fputs(",\"stale_answer_text\":", json); json_text(json, stale_text);
    fputs(",\n  \"diagnostic_distinct_magic\":true,\"diagnostic_import_rejected\":true,\"diagnostic_source_rejected\":true,\"diagnostic_rejections_preserve_graph\":true", json);
    fprintf(json, ",\n  \"query_tokens_including_assistant_prefix\":%d,\"query_start\":%d,\"query_fully_fresh_required\":%s,\"query_fully_fresh\":%s,\n  \"branch_endpoint_logit_bytes_equal\":true,\"semantic_controls_valid\":%s,\"original_calibration\":",
            parts[4].len, total - parts[4].len, require_fresh_query ? "true" : "false",
            query_is_fresh(end_n, total, parts[4].len) ? "true" : "false", semantic_controls_valid ? "true" : "false");
    if (calibrate_original) {
        fprintf(json, "{\"gold_logprob\":%.12g,\"stale_logprob\":%.12g,\"correct_answer\":%s,\"stopped\":%s,\"endpoint_logit_bytes_equal\":true,\"expected_answer\":",
                original_result.gold_logprob, original_result.stale_logprob,
                original_result.stale_answer ? "true" : "false", original_result.stopped ? "true" : "false");
        json_text(json, stale_text); fputs(",\"greedy\":", json);
        json_text(json, original_result.greedy); fputs(",\"greedy_tokens\":", json); json_tokens(json, &original_result.generated); fputc('}', json);
    } else fputs("null", json);
    fprintf(json, ",\n  \"fresh_prefix_bytes_equal\":true,\"fresh_prefix_raw_rows_per_layer\":%d,\"fresh_prefix_compressed_rows\":%d,\"fresh_prefix_index_rows\":%d",
            fresh_prefix_raw_rows, fresh_prefix_compressed_rows, fresh_prefix_compressed_rows);
    fprintf(json, ",\n  \"prefix_tokens\":%d,\"edit_tokens\":%d,\"source_edit_tokens\":%d,\"unchanged_tokens\":%d,\"fresh_tail_tokens\":%d,\n  \"source_start\":%u,\"destination_start\":%u,\"source_end\":%u,\"destination_end\":%u,\n  \"raw_keys_repaired\":%u,\"compressed_keys_repaired\":%u,\"index_keys_repaired\":%u,\n  \"reused_tokens\":%u,\"copied_bytes\":%llu,\"pending_pairs\":%u,\n  \"identity_logit_bytes_equal\":true,\"identity_graph_bytes_equal\":true,\"canonical_export_blocked\":true,\n  \"invalid_candidates_unchanged\":true,\"precancel_unchanged\":true,\"postgraft_cancel_invalidated\":true,\n  \"setup_ms\":%.6f,\"identity_ms\":%.6f,\"rows\":%d,\"gold_tokens_including_eos\":%d,\"stale_tokens_including_eos\":%d,\n  \"mean_kl\":%.12g,\"mean_tv\":%.12g,\"sum_delta_nll\":%.12g,\"argmax_flips\":%d,\"byte_equal_rows\":%d,\n  \"greedy_encoding\":\"JSON byte escapes for non-ASCII\",\n  \"results\":[\n",
            prefix_n, edit_n, source_edit_n, unchanged_n, tail_n, stats.source_start, stats.destination_start, stats.source_end, stats.destination_end,
            stats.raw_keys_repaired, stats.compressed_keys_repaired, stats.index_keys_repaired,
            stats.reuse.reused_tokens, (unsigned long long)stats.reuse.copied_bytes,
            stats.reuse.pending_pairs, setup_ms, identity_ms, answers[0].len, answers[0].len, answers[1].len,
            sum_kl / answers[0].len, sum_tv / answers[0].len, sum_dnll, flips, equal_rows);
    for (int mode = 0; mode < 2; mode++) {
        result *r = &results[mode];
        fprintf(json, "    {\"mode\":\"%s\",\"edit_ms\":%.6f,\"gold_logprob\":%.12g,\"stale_logprob\":%.12g,\"gold_minus_stale_logprob\":%.12g,\"greedy\":",
                mode ? "reuse" : "fresh", r->edit_ms, r->gold_logprob, r->stale_logprob, r->gold_logprob - r->stale_logprob);
        json_text(json, r->greedy); fprintf(json, ",\"correct_answer\":%s,\"stale_answer\":%s,\"stopped\":%s,\"greedy_tokens\":",
                r->gold_answer ? "true" : "false", r->stale_answer ? "true" : "false", r->stopped ? "true" : "false");
        json_tokens(json, &r->generated); fprintf(json, "}%s\n", mode ? "" : ",");
    }
    fputs("  ]\n}\n", json);
    fprintf(stderr, "FACT_ORACLE %s identity_bytes=1 case=%s C=%d tail=%d baseline_correct=%d original_known=%d reuse_correct=%d reuse_stale=%d branch_endpoint_bytes_equal=1 mean_kl=%.9g delta_nll=%.9g\n",
            prior_secret && !semantic_controls_valid ? "CONTROLS_INVALID" : "PASS", case_name, unchanged_n, tail_n,
            results[0].gold_answer, calibrate_original ? original_result.stale_answer : -1,
            results[1].gold_answer, results[1].stale_answer, sum_kl / answers[0].len, sum_dnll);
    rc = prior_secret && !semantic_controls_valid ? 3 : 0;
done:
    if (render && fclose(render)) rc = 1;
    if (raw && fclose(raw)) rc = 1; if (csv && fclose(csv)) rc = 1; if (json && fclose(json)) rc = 1;
    for (int i = 0; i < 5; i++) { free(texts[i]); ds4_tokens_free(&parts[i]); }
    for (int i = 0; i < 2; i++) { free(results[i].greedy); ds4_tokens_free(&results[i].generated); ds4_tokens_free(&answers[i]); }
    ds4_tokens *seq[] = {&a, &old_b, &new_b, &c, &old, &prompt, &old_ab, &new_ab, &old_end, &new_end};
    for (size_t i = 0; i < sizeof(seq) / sizeof(*seq); i++) ds4_tokens_free(seq[i]);
    ds4_session_snapshot *states[] = {&a_state, &old_ab_state, &old_final, &observed, &fresh_final, &new_ab_state};
    for (size_t i = 0; i < sizeof(states) / sizeof(*states); i++) ds4_session_snapshot_free(states[i]);
    free(original_result.greedy); ds4_tokens_free(&original_result.generated);
    ds4_suffix_cache_free(&cache); free(reference); free(row); free(endpoint);
    if (s) ds4_session_set_cancel(s, NULL, NULL);
    ds4_session_free(s); ds4_engine_close(e);
    return rc;
}
