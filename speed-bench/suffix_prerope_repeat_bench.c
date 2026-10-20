/* Capture once, then relocate the SAME immutable suffix through 3/8 edits.
 * Source-only fixture adapted from suffix_fact_oracle.c's padding, scoring and
 * native-prefix preparation. Run only through the existing model pressure guard.
 * Native and reuse arms have the same archive setting. Prefix rebuild is timed.
 * This measures stale conditioning/rotation error; it is not an archive-tool test.
 */
#include "ds4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "suffix-repeat:%d: %s: %s\n", \
    __LINE__, #x, err); goto done; } } while (0)
#define PREFIX 256
#define OLD_EDIT 1792
#define SOURCE_START (PREFIX + OLD_EDIT)
#define REUSED 2049
#define SOURCE_END (SOURCE_START + REUSED)
#define FRESH 128
#define TEACHER 4

static const int shifts[8] = {2, 66, 128, 256, 514, 768, 1024, 1280};
static double ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static uint64_t hash_bytes(const void *data, uint64_t n) {
    const unsigned char *p = data; uint64_t h = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static void append(ds4_tokens *to, const ds4_tokens *from, int n) {
    for (int i = 0; i < n; i++) ds4_tokens_push(to, from->v[i]);
}
static bool pad(ds4_tokens *to, const ds4_tokens *filler, int n) {
    if (!filler->len || n < to->len) return false;
    int first = to->len;
    for (int i = first; i < n; i++) ds4_tokens_push(to, filler->v[(i - first) % filler->len]);
    return to->len == n;
}
static double log_z(const float *p, int n) {
    double mx = -INFINITY, sum = 0;
    for (int i = 0; i < n; i++) {
        /* Compile this caller without finite-math assumptions. */
        if (!isfinite(p[i])) return NAN;
        if (p[i] > mx) mx = p[i];
    }
    for (int i = 0; i < n; i++) sum += exp((double)p[i] - mx);
    return mx + log(sum);
}
static int argmax(const float *p, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    return best;
}
static double kl(const float *p, const float *q, int n, double pz, double qz) {
    double v = 0;
    for (int i = 0; i < n; i++)
        v += exp((double)p[i] - pz) * (((double)p[i] - pz) - ((double)q[i] - qz));
    return v < 0 && v > -1e-10 ? 0 : v;
}
static FILE *output(const char *dir, const char *name, const char *mode) {
    char p[4096];
    if (snprintf(p, sizeof(p), "%s/%s", dir, name) >= (int)sizeof(p)) return NULL;
    return fopen(p, mode);
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL;
    int moves = 3, prerope = 1, ctx = 8192, chunk = 512, cache_gb = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("suffix_prerope_repeat_bench --model MODEL --out EXISTING_DIR "
                 "[--moves 3|8] [--prerope 0|1] [--ctx 8192] [--chunk 512] [--cache-gb 64]");
            return 0;
        }
        if (++i >= argc) return 2;
        const char *key = argv[i - 1], *value = argv[i];
        if (!strcmp(key, "--model")) model = value;
        else if (!strcmp(key, "--out")) out = value;
        else {
            char *end; long n = strtol(value, &end, 10);
            if (end == value || *end || n < 0 || n > 131072) return 2;
            if (!strcmp(key, "--moves")) moves = (int)n;
            else if (!strcmp(key, "--prerope")) prerope = (int)n;
            else if (!strcmp(key, "--ctx")) ctx = (int)n;
            else if (!strcmp(key, "--chunk")) chunk = (int)n;
            else if (!strcmp(key, "--cache-gb")) cache_gb = (int)n;
            else return 2;
        }
    }
    if (!model || !out || (moves != 3 && moves != 8) || prerope > 1 ||
        ctx <= SOURCE_END + FRESH + TEACHER || chunk < 1 || chunk > 8192 ||
        cache_gb < 1 || cache_gb > 96) return 2;
    int rc = 1, vocab = 0; char err[256] = {0};
    ds4_engine *e = NULL; ds4_session *s = NULL; ds4_suffix_cache source = {0};
    ds4_tokens parts[6] = {0}, a = {0}, b = {0}, c = {0}, fresh = {0};
    ds4_tokens original = {0}, prefix = {0}, prompt = {0};
    FILE *csv = NULL, *timing = NULL, *raw = NULL, *tokens = NULL;
    float *reference = NULL, *row = NULL;
    const char *texts[6] = {
        "Keep the current record. Later questions ask for its current access code. "
        "Reply only with CODE= followed by four digits. Use the latest record in the conversation.",
        "Harmless old audit note: the desk lamp, shelf boxes and weather sensor were checked. "
        "This prose contains no access code and can be discarded.\n",
        "Current register: the access code is 2718.\n",
        "Current register: the access code is 3141.\n",
        "What is the current access code? Reply only CODE= followed by the four digits, "
        "without spaces or explanation.",
        "CODE=3141"
    };
    csv = output(out, "rows.csv", "wx"); timing = output(out, "moves.csv", "wx");
    raw = output(out, "logits.bin", "wbx"); tokens = output(out, "tokens.json", "wx");
    CHECK(csv && timing && raw && tokens);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = (uint32_t)ctx, .prefill_chunk = (uint32_t)chunk,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = (uint64_t)cache_gb << 30,
        .power_percent = 100};
    CHECK(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    CHECK(ds4_session_create(&s, e, ctx) == 0);
    if (prerope) CHECK(ds4_session_enable_suffix_prerope(s, err, sizeof(err)) == 0);
    vocab = ds4_engine_vocab_size(e); CHECK(vocab > 0);
    for (int i = 0; i < 6; i++) { ds4_tokenize_text(e, texts[i], &parts[i]); CHECK(parts[i].len); }
    CHECK(parts[5].len == TEACHER);
    ds4_chat_begin(e, &a); ds4_tokens_push(&a, ds4_token_user(e));
    append(&a, &parts[0], parts[0].len); CHECK(pad(&a, &parts[1], PREFIX));
    CHECK(pad(&b, &parts[1], OLD_EDIT - parts[2].len)); append(&b, &parts[2], parts[2].len);
    CHECK(pad(&c, &parts[1], REUSED));
    ds4_chat_append_assistant_prefix(e, &parts[4], DS4_THINK_NONE);
    CHECK(pad(&fresh, &parts[1], FRESH - parts[4].len)); append(&fresh, &parts[4], parts[4].len);
    append(&original, &a, a.len); append(&original, &b, b.len); append(&original, &c, c.len);
    CHECK(original.len == SOURCE_END);
    double started = ms(); CHECK(ds4_session_sync(s, &original, err, sizeof(err)) == 0);
    CHECK(ds4_session_capture_suffix_cache(s, &source, err, sizeof(err)) == 0);
    CHECK(!source.approximate && source.state.len && (!prerope || source.prerope_bytes));
    uint64_t state_bytes = source.state.len, pre_bytes = source.prerope_bytes;
    uint64_t state_hash = hash_bytes(source.state.ptr, state_bytes);
    uint64_t pre_hash = hash_bytes(source.prerope, pre_bytes);
    fprintf(stderr, "SOURCE capture_count=1 start=%d end=%d reused=%d setup_ms=%.3f "
        "state_bytes=%llu prerope_bytes=%llu live_prerope_bytes=%llu "
        "state_fnv=%016llx prerope_fnv=%016llx\n", SOURCE_START, SOURCE_END, REUSED,
        ms() - started, (unsigned long long)state_bytes, (unsigned long long)pre_bytes,
        (unsigned long long)ds4_session_suffix_prerope_bytes(s),
        (unsigned long long)state_hash, (unsigned long long)pre_hash);
    size_t bytes = (size_t)vocab * sizeof(float);
    reference = malloc(bytes * TEACHER); row = malloc(bytes); CHECK(reference && row);
    /* Raw order: move0 native four rows, move0 reuse four rows, move1 ... */
    uint32_t h[4] = {UINT32_C(0x31525053), (uint32_t)vocab, (uint32_t)moves, TEACHER};
    CHECK(fwrite(h, sizeof(h), 1, raw) == 1);
    CHECK(fwrite(parts[5].v, sizeof(int), TEACHER, raw) == TEACHER);
    fprintf(csv, "move,shift,step,target,native_nll,reuse_nll,delta_nll,kl,argmax_flip,byte_equal\n");
    fprintf(timing, "move,shift,arm,destination_start,destination_end,prefix_ms,suffix_and_fresh_ms,total_edit_ms,teacher_ms,reused,fresh,pending_pairs,raw_repaired,compressed_repaired,index_repaired,state_fnv,prerope_fnv\n");
    fprintf(tokens, "{\"source_start\":%d,\"source_end\":%d,\"source\":[", SOURCE_START, SOURCE_END);
    for (int i = 0; i < original.len; i++) fprintf(tokens, "%s%d", i ? "," : "", original.v[i]);
    fprintf(tokens, "],\"fresh\":[");
    for (int i = 0; i < fresh.len; i++) fprintf(tokens, "%s%d", i ? "," : "", fresh.v[i]);
    fprintf(tokens, "],\"teacher\":[");
    for (int i = 0; i < TEACHER; i++) fprintf(tokens, "%s%d", i ? "," : "", parts[5].v[i]);
    fprintf(tokens, "],\"edited_prefixes\":[");
    for (int move = 0; move < moves; move++) {
        int shift = shifts[move], destination = SOURCE_START - shift;
        prefix.len = prompt.len = b.len = 0;
        CHECK(pad(&b, &parts[1], OLD_EDIT - shift - parts[3].len)); append(&b, &parts[3], parts[3].len);
        append(&prefix, &a, a.len); append(&prefix, &b, b.len); CHECK(prefix.len == destination);
        append(&prompt, &prefix, prefix.len); append(&prompt, &c, c.len); append(&prompt, &fresh, fresh.len);
        CHECK(!(destination & 1) && prompt.len == SOURCE_END - shift + FRESH);
        CHECK(!memcmp(prompt.v + destination, original.v + SOURCE_START, REUSED * sizeof(int)));
        fprintf(tokens, "%s[", move ? "," : "");
        for (int i = 0; i < prefix.len; i++) fprintf(tokens, "%s%d", i ? "," : "", prefix.v[i]);
        fprintf(tokens, "]");
        for (int reuse = 0; reuse < 2; reuse++) {
            ds4_suffix_relocation_stats stats = {0};
            double begin = ms(); ds4_session_invalidate(s);
            CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
            double prefix_ms = ms() - begin, tail_begin = ms();
            if (reuse) CHECK(ds4_session_reuse_suffix_relocated(s, &source, &prompt,
                SOURCE_START, &stats, err, sizeof(err)) == 0);
            else CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
            double suffix_ms = ms() - tail_begin, edit_ms = ms() - begin, teacher_begin = ms();
            CHECK(ds4_session_pos(s) == prompt.len && ds4_session_is_approximate(s) == (reuse != 0));
            if (reuse) CHECK(stats.source_start == SOURCE_START && stats.source_end == SOURCE_END &&
                stats.destination_start == (uint32_t)destination &&
                stats.destination_end == (uint32_t)(SOURCE_END - shift) &&
                stats.reuse.reused_tokens == REUSED && stats.reuse.fresh_tail_tokens == FRESH &&
                stats.reuse.pending_pairs == 3);
            for (int step = 0; step < TEACHER; step++) {
                CHECK(ds4_session_copy_logits(s, row, vocab) == vocab);
                double qz = log_z(row, vocab); CHECK(isfinite(qz));
                if (!reuse) memcpy(reference + (size_t)step * vocab, row, bytes);
                else {
                    const float *p = reference + (size_t)step * vocab;
                    double pz = log_z(p, vocab), k = kl(p, row, vocab, pz, qz);
                    int target = parts[5].v[step]; CHECK(isfinite(k));
                    double pnll = pz - p[target], qnll = qz - row[target];
                    fprintf(csv, "%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%d,%d\n",
                        move + 1, shift, step, target, pnll, qnll, qnll - pnll, k,
                        argmax(p, vocab) != argmax(row, vocab), !memcmp(p, row, bytes));
                }
                CHECK(fwrite(row, bytes, 1, raw) == 1);
                CHECK(ds4_session_eval(s, parts[5].v[step], err, sizeof(err)) == 0);
            }
            double teacher_ms = ms() - teacher_begin;
            CHECK(source.state.len == state_bytes && source.prerope_bytes == pre_bytes &&
                hash_bytes(source.state.ptr, state_bytes) == state_hash &&
                hash_bytes(source.prerope, pre_bytes) == pre_hash);
            fprintf(timing, "%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%u,%u,%u,%u,%u,%u,%016llx,%016llx\n",
                move + 1, shift, reuse ? "reuse" : "native", destination, SOURCE_END - shift,
                prefix_ms, suffix_ms, edit_ms, teacher_ms, stats.reuse.reused_tokens,
                stats.reuse.fresh_tail_tokens, stats.reuse.pending_pairs, stats.raw_keys_repaired,
                stats.compressed_keys_repaired, stats.index_keys_repaired,
                (unsigned long long)state_hash, (unsigned long long)pre_hash);
            fprintf(stderr, "MOVE %d arm=%s shift=%d source=%d:%d destination=%d:%d "
                "edit_ms=%.3f teacher_ms=%.3f immutable_state=1 immutable_prerope=1\n",
                move + 1, reuse ? "reuse" : "native", shift, SOURCE_START, SOURCE_END,
                destination, SOURCE_END - shift, edit_ms, teacher_ms);
            CHECK(fflush(csv) == 0 && fflush(timing) == 0 && fflush(raw) == 0);
        }
    }
    fprintf(tokens, "]}\n");
    CHECK(!ferror(tokens)); rc = 0;
done:
    if (csv && fclose(csv)) rc = 1;
    if (timing && fclose(timing)) rc = 1;
    if (raw && fclose(raw)) rc = 1;
    if (tokens && fclose(tokens)) rc = 1;
    free(reference); free(row); ds4_suffix_cache_free(&source);
    ds4_tokens_free(&a); ds4_tokens_free(&b); ds4_tokens_free(&c); ds4_tokens_free(&fresh);
    ds4_tokens_free(&original); ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    for (int i = 0; i < 6; i++) ds4_tokens_free(&parts[i]);
    ds4_session_free(s); ds4_engine_close(e);
    return rc;
}
