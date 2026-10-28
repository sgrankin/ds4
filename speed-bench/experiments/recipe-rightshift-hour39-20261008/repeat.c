/* Item39: capture once, then move the same originals right/left/right.
 * Item51 adds source/destination cache layouts and an optional zero shift.
 * Adapted from suffix_prerope_repeat_bench.c; strict hash checks retained.
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
#define OLD_EDIT 768
#define SOURCE_START (PREFIX + OLD_EDIT)
#define REUSED 513
#define SOURCE_END (SOURCE_START + REUSED)
#define FRESH 128
#define TEACHER 4

static const int shifts[4] = {-66, 128, -514, 0};
static double ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static uint64_t hash_bytes(const void *data, uint64_t n) {
    const unsigned char *p = data; uint64_t h = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static uint32_t payload_word(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int capture_private(ds4_session *s, ds4_suffix_cache *cache, int fp4, int index_fp4, int move,
                           char *err, size_t errlen) {
    int rc = 0;
    CHECK(ds4_session_capture_suffix_cache(s, cache, err, errlen) == 0);
    CHECK(cache->approximate && cache->state.len && cache->prerope_bytes);
    CHECK(ds4_session_suffix_cache_bytes(s) == cache->state.len + cache->prerope_bytes);
    CHECK(payload_word(cache->state.ptr) == UINT32_C(0x44535344));
    CHECK(payload_word(cache->state.ptr + 4) == 2u + 2u * (fp4 != 0) + (index_fp4 != 0));
    fprintf(stderr, "DSSD_CAPTURE move=%d version=%u state_bytes=%llu prerope_bytes=%llu reported_equal=1\n",
        move, payload_word(cache->state.ptr + 4), (unsigned long long)cache->state.len,
        (unsigned long long)cache->prerope_bytes);
    rc = 1;
done:
    return rc;
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
    int moves = 3, prerope = 1, ctx = 8192, chunk = 512, cache_gb = 64, chain = 0;
    const char *f16_env = getenv("DS4_V41_COMPRESSED_KV_F16");
    int source_f16 = f16_env && !strcmp(f16_env, "1"), destination_f16 = source_f16;
    const char *fp4_env = getenv("DS4_V41_COMPRESSED_KV_FP4");
    int source_fp4 = fp4_env && !strcmp(fp4_env, "1"), destination_fp4 = source_fp4;
    const char *index_fp4_env = getenv("DS4_V41_INDEX_FP4");
    int source_index_fp4 = index_fp4_env && !strcmp(index_fp4_env, "1"), destination_index_fp4 = source_index_fp4;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("suffix_prerope_repeat_bench --model MODEL --out EXISTING_DIR "
                 "[--moves 3|4] [--prerope 0|1] [--ctx 8192] [--chunk 512] [--cache-gb 64] "
                 "[--source-f16 0|1] [--destination-f16 0|1] "
                 "[--source-fp4 0|1] [--destination-fp4 0|1] "
                 "[--source-index-fp4 0|1] [--destination-index-fp4 0|1] [--chain 0|1]\n"
                 "Layout defaults inherit DS4_V41_COMPRESSED_KV_F16/FP4 and DS4_V41_INDEX_FP4. "
                 "--moves 1 --prerope 0 runs only legacy left128.");
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
            else if (!strcmp(key, "--source-f16")) source_f16 = (int)n;
            else if (!strcmp(key, "--destination-f16")) destination_f16 = (int)n;
            else if (!strcmp(key, "--source-fp4")) source_fp4 = (int)n;
            else if (!strcmp(key, "--destination-fp4")) destination_fp4 = (int)n;
            else if (!strcmp(key, "--source-index-fp4")) source_index_fp4 = (int)n;
            else if (!strcmp(key, "--destination-index-fp4")) destination_index_fp4 = (int)n;
            else if (!strcmp(key, "--chain")) chain = (int)n;
            else return 2;
        }
    }
    if (!model || !out || (moves != 3 && moves != 4 && !(moves == 1 && !prerope)) ||
        prerope > 1 || source_f16 > 1 || destination_f16 > 1 || source_fp4 > 1 || destination_fp4 > 1 ||
        source_index_fp4 > 1 || destination_index_fp4 > 1 || chain > 1 || (chain && !prerope) ||
        ctx <= SOURCE_END + FRESH + TEACHER || chunk < 1 || chunk > 8192 ||
        cache_gb < 1 || cache_gb > 96) return 2;
    int rc = 1, vocab = 0; char err[256] = {0};
    ds4_engine *e = NULL; ds4_session *s = NULL;
    ds4_suffix_cache source = {0}, recaptured = {0};
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
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", source_f16 ? "1" : "0", 1) == 0);
    CHECK(setenv("DS4_V41_COMPRESSED_KV_FP4", source_fp4 ? "1" : "0", 1) == 0);
    CHECK(setenv("DS4_V41_INDEX_FP4", source_index_fp4 ? "1" : "0", 1) == 0);
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
    CHECK(payload_word(source.state.ptr + 4) == 2u + 2u * (source_fp4 != 0) + (source_index_fp4 != 0));
    CHECK(ds4_session_suffix_cache_bytes(s) == source.state.len + source.prerope_bytes);
    uint64_t state_bytes = source.state.len, pre_bytes = source.prerope_bytes;
    uint64_t state_hash = hash_bytes(source.state.ptr, state_bytes);
    uint64_t pre_hash = hash_bytes(source.prerope, pre_bytes);
    int source_start = SOURCE_START, source_end = SOURCE_END;
    fprintf(stderr, "SOURCE capture_count=1 start=%d end=%d reused=%d setup_ms=%.3f "
        "state_bytes=%llu prerope_bytes=%llu live_prerope_bytes=%llu "
        "state_fnv=%016llx prerope_fnv=%016llx\n", SOURCE_START, SOURCE_END, REUSED,
        ms() - started, (unsigned long long)state_bytes, (unsigned long long)pre_bytes,
        (unsigned long long)ds4_session_suffix_prerope_bytes(s),
        (unsigned long long)state_hash, (unsigned long long)pre_hash);
    /* Keep one mutable session at a time; source bytes outlive its graph. */
    ds4_session_free(s); s = NULL;
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", destination_f16 ? "1" : "0", 1) == 0);
    CHECK(setenv("DS4_V41_COMPRESSED_KV_FP4", destination_fp4 ? "1" : "0", 1) == 0);
    CHECK(setenv("DS4_V41_INDEX_FP4", destination_index_fp4 ? "1" : "0", 1) == 0);
    CHECK(ds4_session_create(&s, e, ctx) == 0);
    if (prerope) CHECK(ds4_session_enable_suffix_prerope(s, err, sizeof(err)) == 0);
    CHECK(ds4_session_load_suffix_prefix(s, &source, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == SOURCE_END && !ds4_session_is_approximate(s));
    CHECK(ds4_session_capture_suffix_cache(s, &recaptured, err, sizeof(err)) == 0);
    const int same_layout = payload_word(recaptured.state.ptr + 4) == payload_word(source.state.ptr + 4);
    CHECK(!recaptured.approximate &&
        recaptured.prerope_bytes == pre_bytes &&
        !memcmp(recaptured.state.ptr + 52, source.state.ptr + 52, ((size_t)SOURCE_END + vocab) * 4) &&
        (!same_layout || (recaptured.state.len == state_bytes && !memcmp(recaptured.state.ptr, source.state.ptr, (size_t)state_bytes))) &&
        (!pre_bytes || !memcmp(recaptured.prerope, source.prerope, (size_t)pre_bytes)));
    ds4_suffix_cache_free(&recaptured);
    fprintf(stderr, "PREFIX_RESTORE source_f16=%d destination_f16=%d source_fp4=%d destination_fp4=%d "
        "source_index_fp4=%d destination_index_fp4=%d same_layout=%d token_logits_equal=1 originals_equal=1\n",
        source_f16, destination_f16, source_fp4, destination_fp4, source_index_fp4, destination_index_fp4, same_layout);
    size_t bytes = (size_t)vocab * sizeof(float);
    reference = malloc(bytes * TEACHER); row = malloc(bytes); CHECK(reference && row);
    /* Raw order: move0 native four rows, move0 reuse four rows, move1 ... */
    uint32_t h[4] = {UINT32_C(0x31525053), (uint32_t)vocab, (uint32_t)moves, TEACHER};
    CHECK(fwrite(h, sizeof(h), 1, raw) == 1);
    CHECK(fwrite(parts[5].v, sizeof(int), TEACHER, raw) == TEACHER);
    fprintf(csv, "move,shift,step,target,native_nll,reuse_nll,delta_nll,kl,argmax_flip,byte_equal\n");
    fprintf(timing, "move,shift,arm,destination_start,destination_end,prefix_ms,suffix_and_fresh_ms,total_edit_ms,teacher_ms,reused,fresh,pending_pairs,raw_repaired,compressed_repaired,index_repaired,state_fnv,prerope_fnv,remaining_tail_sync_ms\n");
    fprintf(tokens, "{\"source_f16\":%d,\"destination_f16\":%d,\"source_fp4\":%d,\"destination_fp4\":%d,"
        "\"source_index_fp4\":%d,\"destination_index_fp4\":%d,"
        "\"chain\":%d,\"source_start\":%d,\"source_end\":%d,\"source\":[",
        source_f16, destination_f16, source_fp4, destination_fp4, source_index_fp4, destination_index_fp4, chain, SOURCE_START, SOURCE_END);
    for (int i = 0; i < original.len; i++) fprintf(tokens, "%s%d", i ? "," : "", original.v[i]);
    fprintf(tokens, "],\"fresh\":[");
    for (int i = 0; i < fresh.len; i++) fprintf(tokens, "%s%d", i ? "," : "", fresh.v[i]);
    fprintf(tokens, "],\"teacher\":[");
    for (int i = 0; i < TEACHER; i++) fprintf(tokens, "%s%d", i ? "," : "", parts[5].v[i]);
    fprintf(tokens, "],\"edited_prefixes\":[");
    for (int move = 0; move < moves; move++) {
        int shift = moves == 1 ? 128 : shifts[move], destination = SOURCE_START - shift;
        const int reused = source_end - source_start;
        const int imported_end = destination + reused;
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
            double remaining_tail_ms = 0;
            double begin = ms(); ds4_session_invalidate(s);
            CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
            double prefix_ms = ms() - begin, tail_begin = ms();
            if (reuse && move == 0 && prerope) {
                /* Concrete guard: unsupported legacy RIGHT must not mutate KV. */
                ds4_suffix_cache legacy = source;
                legacy.prerope = NULL; legacy.prerope_bytes = 0;
                ds4_session_snapshot before = {0}, after = {0};
                CHECK(ds4_session_save_snapshot(s, &before, err, sizeof(err)) == 0);
                CHECK(ds4_session_reuse_suffix_relocated(s, &legacy, &prompt,
                    source_start, &stats, err, sizeof(err)) != 0);
                CHECK(ds4_session_save_snapshot(s, &after, err, sizeof(err)) == 0);
                CHECK(before.len == after.len && !memcmp(before.ptr, after.ptr, before.len));
                ds4_session_snapshot_free(&before); ds4_session_snapshot_free(&after);
                fprintf(stderr, "LEGACY_RIGHT rejected_before_mutation=1\n");
                err[0] = 0;
            }
            if (reuse) {
                const int full_length = prompt.len;
                if (chain) prompt.len = imported_end + 2;
                CHECK(ds4_session_reuse_suffix_relocated(s, &source, &prompt,
                    source_start, &stats, err, sizeof(err)) == 0);
                if (chain) {
                    CHECK(capture_private(s, &recaptured, destination_fp4, destination_index_fp4, move, err, sizeof(err)));
                    CHECK(payload_word(recaptured.state.ptr + 28) == (uint32_t)(imported_end + 2));
                    prompt.len = full_length;
                    const double remaining_begin = ms();
                    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
                    remaining_tail_ms = ms() - remaining_begin;
                }
            } else CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
            double suffix_ms = ms() - tail_begin, edit_ms = ms() - begin;
            CHECK(ds4_session_pos(s) == prompt.len && ds4_session_is_approximate(s) == (reuse != 0));
            if (reuse) CHECK(stats.source_start == (uint32_t)source_start && stats.source_end == (uint32_t)source_end &&
                stats.destination_start == (uint32_t)destination &&
                stats.destination_end == (uint32_t)imported_end &&
                stats.reuse.reused_tokens == (uint32_t)reused && stats.reuse.fresh_tail_tokens == (chain ? 2u : FRESH) &&
                stats.reuse.pending_pairs == 3);
            if (reuse && source_start == destination) CHECK(!stats.raw_keys_repaired &&
                !stats.compressed_keys_repaired && !stats.index_keys_repaired);
            if (reuse && prerope) {
                if (!chain) {
                    CHECK(capture_private(s, &recaptured, destination_fp4, destination_index_fp4, move, err, sizeof(err)));
                    ds4_suffix_cache_free(&recaptured);
                }
                fprintf(stderr, "RECAPTURE move=%d approximate=1 originals=1\n", move + 1);
            }
            double teacher_begin = ms();
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
            fprintf(timing, "%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%u,%u,%u,%u,%u,%u,%016llx,%016llx,%.6f\n",
                move + 1, shift, reuse ? "reuse" : "native", destination, imported_end,
                prefix_ms, suffix_ms, edit_ms, teacher_ms, stats.reuse.reused_tokens,
                stats.reuse.fresh_tail_tokens, stats.reuse.pending_pairs, stats.raw_keys_repaired,
                stats.compressed_keys_repaired, stats.index_keys_repaired,
                (unsigned long long)state_hash, (unsigned long long)pre_hash, remaining_tail_ms);
            fprintf(stderr, "MOVE %d arm=%s shift=%d actual_shift=%d source=%d:%d destination=%d:%d "
                "edit_ms=%.3f teacher_ms=%.3f immutable_state=1 immutable_prerope=1\n",
                move + 1, reuse ? "reuse" : "native", shift, source_start - destination, source_start, source_end,
                destination, imported_end, edit_ms, teacher_ms);
            CHECK(fflush(csv) == 0 && fflush(timing) == 0 && fflush(raw) == 0);
            if (chain && reuse) {
                ds4_suffix_cache_free(&source);
                source = recaptured;
                memset(&recaptured, 0, sizeof(recaptured));
                source_start = destination;
                source_end = imported_end + 2;
                state_bytes = source.state.len; pre_bytes = source.prerope_bytes;
                state_hash = hash_bytes(source.state.ptr, state_bytes);
                pre_hash = hash_bytes(source.prerope, pre_bytes);
                fprintf(stderr, "CHAIN_SOURCE move=%d start=%d end=%d reused_next=%d version=%u\n",
                    move + 1, source_start, source_end, source_end - source_start, payload_word(source.state.ptr + 4));
            }
        }
    }
    fprintf(tokens, "]}\n");
    CHECK(!ferror(tokens)); rc = 0;
done:
    if (csv && fclose(csv)) rc = 1;
    if (timing && fclose(timing)) rc = 1;
    if (raw && fclose(raw)) rc = 1;
    if (tokens && fclose(tokens)) rc = 1;
    free(reference); free(row); ds4_suffix_cache_free(&source); ds4_suffix_cache_free(&recaptured);
    ds4_tokens_free(&a); ds4_tokens_free(&b); ds4_tokens_free(&c); ds4_tokens_free(&fresh);
    ds4_tokens_free(&original); ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    for (int i = 0; i < 6; i++) ds4_tokens_free(&parts[i]);
    ds4_session_free(s); ds4_engine_close(e);
    return rc;
}
