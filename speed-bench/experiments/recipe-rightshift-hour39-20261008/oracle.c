/* Experimental right-shift screen, adapted from Item38/Item14.
 * Teacher metrics compare identical continuation tokens, without sampling.
 * Free greedy code starts separately from each prompt endpoint. */
#include "ds4.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "rightshift:%d: %s: %s\n", __LINE__, #x, err); \
    goto done; } } while (0)

typedef struct {
    double teacher_logprob, edit_ms;
    char *greedy;
    ds4_tokens generated;
    bool stopped;
} result;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void phase(const char *name, double started) {
    fprintf(stderr, "PHASE %s complete_ms=%.3f\n", name, now_ms() - started);
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
    if (!r->greedy) return 1;
    r->greedy[0] = 0;
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

int main(int argc, char **argv) {
    const char *model = NULL, *fixtures = NULL, *out = NULL, *case_name = "rightshift";
    int prefix_n = 256, source_edit_n = 256, edit_n = 1024;
    int unchanged_n = 1536, tail_n = 64, teacher_limit = 512, greedy_limit = 512;
    int ctx = 8192, chunk = 512, cache_gb = 64;
    bool dump_raw = false, system_edit = false, pure_insertion = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("oracle --model MODEL --fixtures DIR --out EXISTING_DIR [--case NAME] "
                 "[--prefix N] [--source-edit N] [--edit N] [--unchanged N] "
                 "[--fresh-tail N] [--teacher N] [--greedy N] [--ctx N] "
                 "[--chunk N] [--cache-gb N] [--raw] [--system-edit] [--pure-insertion]");
            return 0;
        }
        if (!strcmp(argv[i], "--pure-insertion")) { pure_insertion = true; continue; }
        if (!strcmp(argv[i], "--raw")) { dump_raw = true; continue; }
        if (!strcmp(argv[i], "--system-edit")) { system_edit = true; continue; }
        if (++i >= argc) return 2;
        const char *key = argv[i - 1], *value = argv[i];
        if (!strcmp(key, "--model")) model = value;
        else if (!strcmp(key, "--fixtures")) fixtures = value;
        else if (!strcmp(key, "--out")) out = value;
        else if (!strcmp(key, "--case")) case_name = value;
        else {
            int n = positive(value); if (n < 0) return 2;
            if (!strcmp(key, "--prefix")) prefix_n = n;
            else if (!strcmp(key, "--source-edit")) source_edit_n = n;
            else if (!strcmp(key, "--edit")) edit_n = n;
            else if (!strcmp(key, "--unchanged")) unchanged_n = n;
            else if (!strcmp(key, "--fresh-tail")) tail_n = n;
            else if (!strcmp(key, "--teacher")) teacher_limit = n;
            else if (!strcmp(key, "--greedy")) greedy_limit = n;
            else if (!strcmp(key, "--ctx")) ctx = n;
            else if (!strcmp(key, "--chunk")) chunk = n;
            else if (!strcmp(key, "--cache-gb")) cache_gb = n;
            else return 2;
        }
    }
    int start = prefix_n + edit_n, source_start = prefix_n + source_edit_n;
    int total = start + unchanged_n, source_total = source_start + unchanged_n;
    int end_n = total - tail_n, source_end_n = source_total - tail_n;
    if (!model || !fixtures || !out || (start & 1) || (source_start & 1) ||
        tail_n >= unchanged_n || teacher_limit > 512 ||
        (pure_insertion && edit_n <= source_edit_n) ||
        greedy_limit > 512 || chunk > 8192 || cache_gb > 96 ||
        source_total + teacher_limit + greedy_limit >= ctx ||
        total + teacher_limit + greedy_limit >= ctx) return 2;

    int rc = 1, vocab = 0, latest_user = -1; char err[256] = {0};
    const char *names[] = {"prefix.txt", "old-edit.txt", "new-edit.txt", "notes.txt", "question.txt", "target.txt", "filler.txt"};
    char *texts[7] = {0}; ds4_tokens parts[7] = {0};
    ds4_tokens a = {0}, old_b = {0}, new_b = {0}, c = {0}, query = {0};
    ds4_tokens old = {0}, prompt = {0}, old_ab = {0}, new_ab = {0}, old_end = {0}, new_end = {0};
    ds4_session_snapshot a_state = {0}, fresh_final = {0}; ds4_suffix_cache cache = {0};
    ds4_suffix_relocation_stats stats = {0}; result results[2] = {0};
    ds4_engine *e = NULL; ds4_session *s = NULL;
    float *reference = NULL, *row = NULL; FILE *csv = NULL, *raw = NULL, *json = NULL, *render = NULL;
    double setup_ms = 0, capture_ms = 0, sum_kl = 0, sum_tv = 0, sum_dnll = 0;
    int flips = 0, equal_rows = 0;
    for (int i = 0; i < 7; i++) { texts[i] = read_text(fixtures, names[i]); REQUIRE(texts[i]); }
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = (uint32_t)ctx, .prefill_chunk = (uint32_t)chunk,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = (uint64_t)cache_gb << 30,
        .power_percent = 100};
    REQUIRE(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    REQUIRE(ds4_session_create(&s, e, ctx) == 0);
    REQUIRE(ds4_session_enable_suffix_prerope(s, err, sizeof(err)) == 0);
    vocab = ds4_engine_vocab_size(e); size_t bytes = (size_t)vocab * sizeof(float);
    for (int i = 0; i < 7; i++) { ds4_tokenize_text(e, texts[i], &parts[i]); REQUIRE(parts[i].len); }
    if (parts[5].len > teacher_limit) parts[5].len = teacher_limit;

    ds4_chat_begin(e, &a);
    ds4_chat_append_message(e, &a, system_edit ? "system" : "user", texts[0]);
    REQUIRE(pad(&a, &parts[6], prefix_n));
    append(&old_b, &parts[1], 0, parts[1].len); REQUIRE(pad(&old_b, &parts[6], source_edit_n));
    if (pure_insertion) append(&new_b, &old_b, 0, old_b.len);
    append(&new_b, &parts[2], 0, parts[2].len); REQUIRE(pad(&new_b, &parts[6], edit_n));
    if (system_edit) ds4_tokens_push(&c, ds4_token_user(e));
    /* Keep contract/config first; varied review notes fill remaining older context. */
    ds4_chat_append_message(e, &query, "assistant", "Ready for the current task.");
    ds4_chat_append_message(e, &query, "user", texts[4]);
    ds4_chat_append_assistant_prefix(e, &query, DS4_THINK_NONE);
    REQUIRE(query.len < tail_n);
    append(&c, &parts[3], 0, parts[3].len);
    REQUIRE(pad(&c, &parts[6], unchanged_n - query.len)); append(&c, &query, 0, query.len);
    append(&old, &a, 0, a.len); append(&old, &old_b, 0, old_b.len); append(&old, &c, 0, c.len);
    append(&prompt, &a, 0, a.len); append(&prompt, &new_b, 0, new_b.len); append(&prompt, &c, 0, c.len);
    REQUIRE(old.len == source_total && prompt.len == total);
    for (int i = prompt.len - 1; i >= 0; i--) if (prompt.v[i] == ds4_token_user(e)) { latest_user = i; break; }
    REQUIRE(latest_user >= end_n && query_is_fresh(end_n, total, query.len));
    append(&old_ab, &old, 0, source_start); append(&new_ab, &prompt, 0, start);
    append(&old_end, &old, 0, source_end_n); append(&new_end, &prompt, 0, end_n);
    reference = malloc(bytes * (size_t)parts[5].len); row = malloc(bytes); REQUIRE(reference && row);
    csv = output_file(out, "rows.csv", "wx"); json = output_file(out, "summary.json", "wx"); REQUIRE(csv && json);
    if (dump_raw) {
        raw = output_file(out, "logits.bin", "wbx"); REQUIRE(raw);
        uint32_t header[4] = {UINT32_C(0x53554632), (uint32_t)vocab, (uint32_t)parts[5].len, (uint32_t)total};
        REQUIRE(fwrite(header, sizeof(header), 1, raw) == 1);
        REQUIRE(fwrite(parts[5].v, sizeof(int), (size_t)parts[5].len, raw) == (size_t)parts[5].len);
    }
    render = output_file(out, "old-prompt.txt", "wbx"); REQUIRE(render); REQUIRE(!write_rendered(render, e, &old)); REQUIRE(!close_file(&render));
    render = output_file(out, "new-prompt.txt", "wbx"); REQUIRE(render); REQUIRE(!write_rendered(render, e, &prompt)); REQUIRE(!close_file(&render));
    render = output_file(out, "tokens.json", "wx"); REQUIRE(render);
    fputs("{\"old\":", render); json_tokens(render, &old); fputs(",\"edited\":", render); json_tokens(render, &prompt);
    fputs(",\"teacher\":", render); json_tokens(render, &parts[5]); fputs("}\n", render); REQUIRE(!close_file(&render));
    double t = now_ms();
    REQUIRE(!ds4_session_sync(s, &a, err, sizeof(err)));
    REQUIRE(!ds4_session_save_snapshot(s, &a_state, err, sizeof(err)));
    REQUIRE(!ds4_session_sync(s, &old_ab, err, sizeof(err)));
    REQUIRE(!ds4_session_sync(s, &old_end, err, sizeof(err)));
    double capture_started = now_ms(); REQUIRE(!ds4_session_capture_suffix_cache(s, &cache, err, sizeof(err)));
    REQUIRE(cache.prerope && cache.prerope_bytes); capture_ms = now_ms() - capture_started; setup_ms = now_ms() - t;
    fprintf(csv, "step,target,fresh_nll,reuse_nll,delta_nll,kl,tv,argmax_flip,byte_equal\n");
    for (int mode = 0; mode < 2; mode++) {
        REQUIRE(!ds4_session_load_snapshot(s, &a_state, err, sizeof(err))); t = now_ms();
        REQUIRE(!ds4_session_sync(s, &new_ab, err, sizeof(err)));
        if (mode) REQUIRE(!ds4_session_reuse_suffix_relocated(s, &cache, &prompt, (uint32_t)source_start, &stats, err, sizeof(err)));
        else { REQUIRE(!ds4_session_sync(s, &new_end, err, sizeof(err))); REQUIRE(!ds4_session_sync(s, &prompt, err, sizeof(err))); }
        results[mode].edit_ms = now_ms() - t; phase(mode ? "reuse_edit" : "fresh_edit", t);
        if (!mode) REQUIRE(!ds4_session_save_snapshot(s, &fresh_final, err, sizeof(err)));
        for (int j = 0; j < parts[5].len; j++) {
            REQUIRE(ds4_session_copy_logits(s, row, vocab) == vocab);
            double z = log_z(row, vocab); REQUIRE(isfinite(z));
            results[mode].teacher_logprob += row[parts[5].v[j]] - z;
            if (!mode) memcpy(reference + (size_t)j * vocab, row, bytes);
            else {
                const float *p = reference + (size_t)j * vocab; double kl, tv;
                distance(p, row, vocab, &kl, &tv); REQUIRE(isfinite(kl) && isfinite(tv));
                double pnll = log_z(p, vocab) - p[parts[5].v[j]], qnll = z - row[parts[5].v[j]];
                int flip = argmax(p, vocab) != argmax(row, vocab), equal = !memcmp(p, row, bytes);
                sum_kl += kl; sum_tv += tv; sum_dnll += qnll - pnll; flips += flip; equal_rows += equal;
                fprintf(csv, "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d\n", j, parts[5].v[j], pnll, qnll, qnll - pnll, kl, tv, flip, equal);
            }
            if (raw) REQUIRE(fwrite(row, bytes, 1, raw) == 1);
            if (j + 1 < parts[5].len) REQUIRE(!ds4_session_eval(s, parts[5].v[j], err, sizeof(err)));
        }
        REQUIRE(!prepare(s, &a_state, &fresh_final, &new_ab, &new_end, &prompt, &cache, (uint32_t)source_start, mode != 0, &stats, err, sizeof(err)));
        REQUIRE(!greedy(s, e, greedy_limit, vocab, row, &results[mode], err, sizeof(err)));
        render = output_file(out, mode ? "reuse-code.py" : "fresh-code.py", "wbx"); REQUIRE(render);
        REQUIRE(fwrite(results[mode].greedy, 1, strlen(results[mode].greedy), render) == strlen(results[mode].greedy)); REQUIRE(!close_file(&render));
        fprintf(stderr, "RIGHTSHIFT case=%s mode=%s teacher=%d greedy=%d stopped=%d edit_ms=%.3f\n", case_name, mode ? "reuse" : "fresh", parts[5].len, results[mode].generated.len, results[mode].stopped, results[mode].edit_ms);
    }
    fputs("{\"case\":", json); json_text(json, case_name);
    fprintf(json, ",\"source_prompt_tokens\":%d,\"destination_prompt_tokens\":%d,\"shift_tokens\":%d,\"pure_insertion\":%s",
        source_total, total, start - source_start, pure_insertion ? "true" : "false");
    fprintf(json, ",\"system_edit\":%s,\"teacher_tokens\":%d,\"rows\":%d,\"teacher_cap\":%d,\"greedy_cap\":%d,\"raw_dump\":%s,\"prerope_originals\":true,\"prerope_bytes\":%llu,\"source_capture_ms\":%.6f,\"setup_ms\":%.6f,\"latest_user_start\":%d,\"query_fully_fresh\":true,\"source_start\":%u,\"source_end\":%u,\"destination_start\":%u,\"destination_end\":%u,\"reused_tokens\":%u,\"fresh_tail_tokens\":%u,\"copied_bytes\":%llu,\"sum_delta_nll\":%.12g,\"mean_delta_nll\":%.12g,\"perplexity_ratio\":%.12g,\"mean_kl\":%.12g,\"mean_tv\":%.12g,\"argmax_flips\":%d,\"byte_equal_rows\":%d,\"results\":[",
        system_edit ? "true" : "false", parts[5].len, parts[5].len, teacher_limit, greedy_limit, dump_raw ? "true" : "false", (unsigned long long)cache.prerope_bytes, capture_ms, setup_ms, latest_user, stats.source_start, stats.source_end, stats.destination_start, stats.destination_end, stats.reuse.reused_tokens, stats.reuse.fresh_tail_tokens, (unsigned long long)stats.reuse.copied_bytes, sum_dnll, sum_dnll / parts[5].len, exp(sum_dnll / parts[5].len), sum_kl / parts[5].len, sum_tv / parts[5].len, flips, equal_rows);
    for (int mode = 0; mode < 2; mode++) {
        fprintf(json, "%s{\"mode\":\"%s\",\"edit_ms\":%.6f,\"teacher_logprob\":%.12g,\"stopped\":%s,\"at_cap\":%s,\"greedy_tokens\":%d,\"greedy\":", mode ? "," : "", mode ? "reuse" : "fresh", results[mode].edit_ms, results[mode].teacher_logprob, results[mode].stopped ? "true" : "false", results[mode].generated.len >= greedy_limit && !results[mode].stopped ? "true" : "false", results[mode].generated.len);
        json_text(json, results[mode].greedy); fputc('}', json);
    }
    fputs("]}\n", json); rc = 0;
done:
    if (render && fclose(render)) rc = 1;
    if (csv && fclose(csv)) rc = 1;
    if (raw && fclose(raw)) rc = 1;
    if (json && fclose(json)) rc = 1;
    for (int i = 0; i < 7; i++) { free(texts[i]); ds4_tokens_free(&parts[i]); }
    for (int i = 0; i < 2; i++) { free(results[i].greedy); ds4_tokens_free(&results[i].generated); }
    ds4_tokens *vectors[] = {&a, &old_b, &new_b, &c, &query, &old, &prompt, &old_ab, &new_ab, &old_end, &new_end};
    for (size_t i = 0; i < sizeof(vectors) / sizeof(*vectors); i++) ds4_tokens_free(vectors[i]);
    ds4_session_snapshot_free(&a_state); ds4_session_snapshot_free(&fresh_final);
    ds4_suffix_cache_free(&cache); free(reference); free(row);
    ds4_session_free(s); ds4_engine_close(e); return rc;
}
