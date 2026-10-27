/* Experimental SWB1 +128-row SWA restore. Adapted from decoder_replay_bench.c;
 * run through the existing pressure guard. Teacher logits and greedy answers
 * use independent prefills of identical token vectors. No server/cache layer. */
#include "ds4.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CASES 3
#define CONTEXT 16384
#define MAX_NEW 48
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "decoder-replay:%d: %s: %s\n", \
    __LINE__, #x, err); goto done; } } while (0)

typedef struct {
    const char *name, *record, *expected, *old_record;
} fixture;

static const fixture fixtures[CASES] = {
    {"ordinary", "\n\nCurrent project register, revision 4: the Boreal access code is EMBER-7319.\n\n",
        "CODE=EMBER-7319", NULL},
    {"favorable", "\n\nCurrent project register, revision 4: the Boreal access code is CEDAR-4826.\n\n",
        "CODE=CEDAR-4826", NULL},
    {"changed", "\n\nCorrection, revision 5: the Boreal access code is now LARCH-9062. "
        "The previous code is obsolete. This is the latest record.\n\n",
        "CODE=LARCH-9062", "\n\nOld project register, revision 2: the Boreal access code was MAPLE-1044.\n\n"},
};

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static FILE *output(const char *dir, const char *name, const char *mode) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return NULL;
    return fopen(path, mode);
}

static int number(const char *value) {
    char *end; long n = strtol(value, &end, 10);
    return end == value || *end || n < 0 || n > 1048576 ? -1 : (int)n;
}

static void append(ds4_tokens *dst, const ds4_tokens *src) {
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

static bool pad_to(ds4_tokens *dst, const ds4_tokens *filler, int wanted) {
    if (!filler->len || wanted < dst->len) return false;
    int start = dst->len;
    while (dst->len < wanted) ds4_tokens_push(dst, filler->v[(dst->len - start) % filler->len]);
    return true;
}

static bool make_prompt(ds4_engine *e, int which, int wanted, ds4_tokens *prompt,
                        ds4_tokens *teacher, int *record_start, int *query_start,
                        bool thinking) {
    const fixture *f = &fixtures[which];
    ds4_tokens filler = {0}, tail = {0};
    const char *notes =
        "Historical operations note: the east shelf was inspected on Monday. "
        "The blue notebooks are filed beside the window. The workshop lamp was repaired. "
        "The weekly inventory lists paper, cables, labels and spare hinges.\n"
        "Meeting summary: the team reviewed weather observations and ordinary maintenance. "
        "No credentials were changed in this meeting. These notes do not replace a project register.\n"
        "Archive entry: a delivery arrived at the south entrance; the boxes were moved to storage. "
        "The reference manual explains labels and routine inspection schedules.\n";
    ds4_tokenize_text(e, notes, &filler);
    ds4_chat_begin(e, prompt);
    ds4_chat_append_message(e, prompt, "system",
        "Answer using the latest project register in this conversation. A later correction overrides "
        "an old record. Return only CODE= followed by the complete access code, with no explanation.");
    ds4_chat_append_message(e, prompt, "user",
        "Here are archived Boreal project notes and its access-code register. Retain the current record.\n");
    ds4_chat_append_message(e, &tail, "assistant", "I will consult the latest register when asked.");
    ds4_tokens_push(&tail, ds4_token_eos(e));
    int query_offset = tail.len;
    ds4_chat_append_message(e, &tail, "user",
        "What is the current Boreal access code? Return only CODE= followed by the complete code.");
    ds4_chat_append_assistant_prefix(e, &tail, thinking ? DS4_THINK_HIGH : DS4_THINK_NONE);
    bool ok = filler.len && wanted > tail.len + prompt->len + 1024;
    if (ok && f->old_record) {
        ok = pad_to(prompt, &filler, wanted / 8);
        if (ok) ds4_tokenize_text(e, f->old_record, prompt);
    }
    int at = which == 0 ? wanted / 2 : which == 1 ? wanted - 256 - 48 : wanted - 512;
    if (ok) ok = pad_to(prompt, &filler, at);
    *record_start = prompt->len;
    if (ok) ds4_tokenize_text(e, f->record, prompt);
    if (ok) ok = pad_to(prompt, &filler, wanted - tail.len);
    *query_start = wanted - tail.len + query_offset;
    if (ok) append(prompt, &tail);
    ds4_tokenize_text(e, f->expected, teacher);
    ds4_tokens_push(teacher, ds4_token_eos(e));
    ds4_tokens_free(&filler); ds4_tokens_free(&tail);
    return ok && prompt->len == wanted && teacher->len > 1 && teacher->len <= 24;
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

static void json_text(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 32) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

static void json_tokens(FILE *f, const ds4_tokens *t) {
    fputc('[', f);
    for (int i = 0; i < t->len; i++) fprintf(f, "%s%d", i ? "," : "", t->v[i]);
    fputc(']', f);
}

static bool answer_equal(const char *text, const char *expected) {
    while (isspace((unsigned char)*text)) text++;
    size_t n = strlen(text);
    while (n && isspace((unsigned char)text[n - 1])) n--;
    return n == strlen(expected) && !memcmp(text, expected, n);
}


static const char *restore_mode = "recover";
static int verification = 0;
static char full_path[4096], blocks_path[4096], frontier_path[4096];
static int probe_vocab, restore_calls;

static uint32_t u32_at(const uint8_t *p) {
    uint32_t value; memcpy(&value, p, sizeof(value)); return value;
}

static bool save_file(const char *path, const ds4_session_snapshot *snapshot) {
    FILE *f = fopen(path, "wbx");
    if (!f) return false;
    bool ok = fwrite(snapshot->ptr, 1, (size_t)snapshot->len, f) == snapshot->len;
    return fclose(f) == 0 && ok;
}

static bool read_file(const char *path, ds4_session_snapshot *snapshot) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseeko(f, 0, SEEK_END)) { fclose(f); return false; }
    off_t size = ftello(f);
    if (size <= 52 || fseeko(f, 0, SEEK_SET)) { fclose(f); return false; }
    snapshot->ptr = malloc((size_t)size);
    snapshot->len = snapshot->cap = (uint64_t)size;
    bool ok = snapshot->ptr && fread(snapshot->ptr, 1, (size_t)size, f) == (size_t)size;
    return fclose(f) == 0 && ok;
}

/* Private fixed wire format: SWB1/version1, unchanged remaining DSV4 header,
 * exact token IDs, then the existing F32 main/index spans. No logits/raw rings.
 * Only snapshots created by this same engine/run are admitted by this probe. */
static bool make_blocks(const ds4_session_snapshot *full, ds4_session_snapshot *blocks) {
    if (full->len < 52 || u32_at(full->ptr) != UINT32_C(0x34565344) || u32_at(full->ptr + 4) != 2) return false;
    uint32_t end = u32_at(full->ptr + 28), vocab = u32_at(full->ptr + 44);
    if (end < 256 || end % 128) return false;
    uint64_t first = 52 + (uint64_t)end * 4;
    uint64_t omit = (uint64_t)vocab * 4 + UINT64_C(40) * 128 * 512 * 4;
    if (first + omit > full->len) return false;
    blocks->len = blocks->cap = full->len - omit;
    blocks->ptr = malloc((size_t)blocks->len);
    if (!blocks->ptr) return false;
    memcpy(blocks->ptr, full->ptr, (size_t)first);
    memcpy(blocks->ptr + first, full->ptr + first + omit, (size_t)(full->len - first - omit));
    uint32_t magic = UINT32_C(0x31425753), version = 1;
    memcpy(blocks->ptr, &magic, 4); memcpy(blocks->ptr + 4, &version, 4);
    return true;
}

static int prepare(ds4_session *session, const ds4_tokens *prompt, char *err, size_t err_size) {
    if (!strcmp(restore_mode, "fresh")) return ds4_session_sync(session, prompt, err, err_size);
    ds4_session_snapshot snapshot = {0};
    int rc = 1;
    if (!read_file(!strcmp(restore_mode, "recover") ? blocks_path : full_path, &snapshot)) {
        snprintf(err, err_size, "cannot read immutable checkpoint input"); goto done;
    }
    rc = !strcmp(restore_mode, "recover")
        ? ds4_session_load_blocks_snapshot_replay128(session, &snapshot, verification != 0, err, err_size)
        : ds4_session_load_snapshot(session, &snapshot, err, err_size);
    if (!rc && !strcmp(restore_mode, "recover") && restore_calls++ == 0) {
        float *frontier = malloc((size_t)probe_vocab * sizeof(float));
        FILE *f = NULL;
        if (!frontier || ds4_session_copy_logits(session, frontier, probe_vocab) != probe_vocab ||
            !(f = fopen(frontier_path, "wbx"))) {
            free(frontier); rc = 1; goto done;
        }
        bool ok = fwrite(frontier, sizeof(float), (size_t)probe_vocab, f) == (size_t)probe_vocab;
        if (fclose(f) || !ok) rc = 1;
        free(frontier);
    }
    if (!rc) rc = ds4_session_sync(session, prompt, err, err_size);
done:
    ds4_session_snapshot_free(&snapshot);
    return rc;
}

/* Baseline-generated non-stop IDs, whitespace separated; append the model EOS. */
static bool load_teacher(ds4_engine *engine, const char *directory, const char *name,
                         int room, ds4_tokens *teacher, char *err, size_t err_size) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.txt", directory, name);
    if (n < 0 || n >= (int)sizeof(path)) {
        snprintf(err, err_size, "teacher path is too long"); return false;
    }
    FILE *file = fopen(path, "r");
    if (!file) { snprintf(err, err_size, "cannot open teacher trace %s", path); return false; }
    teacher->len = 0;
    int token, scanned, vocab = ds4_engine_vocab_size(engine);
    bool ok = true;
    while ((scanned = fscanf(file, "%d", &token)) == 1) {
        if (token < 0 || token >= vocab || ds4_token_is_stop(engine, token) || teacher->len + 1 >= room) {
            ok = false; break;
        }
        ds4_tokens_push(teacher, token);
    }
    ok = ok && scanned == EOF && !ferror(file) && teacher->len > 0;
    if (fclose(file)) ok = false;
    if (!ok) { snprintf(err, err_size, "invalid teacher trace %s: require nonempty non-stop token IDs fitting context", path); return false; }
    ds4_tokens_push(teacher, ds4_token_eos(engine));
    return true;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL, *selected = "all", *teacher_dir = NULL;
    int rows = 0, wanted = 8448, chunk = 8192, ctx = CONTEXT, compressed_f16 = 0;
    int thinking = 1, max_new = 256, cache_gb = 48;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("swa_restore_probe --model MODEL --out EXISTING_DIR --mode exact|recover|fresh "
                 "[--tokens 8448] [--chunk 8192] [--ctx 16384] [--compressed-kv-f16 0|1] "
                 "[--case all|ordinary|favorable|changed] [--thinking 0|1] [--max-new 256] [--cache-gb 48] [--verify 0|1] [--teacher-dir DIR]");
            return 0;
        }
        if (++i >= argc) return 2;
        const char *key = argv[i - 1], *value = argv[i];
        if (!strcmp(key, "--model")) model = value;
        else if (!strcmp(key, "--out")) out = value;
        else if (!strcmp(key, "--case")) selected = value;
        else if (!strcmp(key, "--teacher-dir")) teacher_dir = value;
        else if (!strcmp(key, "--mode")) restore_mode = value;
        else if (!strcmp(key, "--verify")) verification = number(value);
        else if (!strcmp(key, "--tokens")) wanted = number(value);
        else if (!strcmp(key, "--chunk")) chunk = number(value);
        else if (!strcmp(key, "--ctx")) ctx = number(value);
        else if (!strcmp(key, "--compressed-kv-f16")) compressed_f16 = number(value);
        else if (!strcmp(key, "--thinking")) thinking = number(value);
        else if (!strcmp(key, "--max-new")) max_new = number(value);
        else if (!strcmp(key, "--cache-gb")) cache_gb = number(value);
        else return 2;
    }
    bool chosen[CASES] = {0}; int count = 0;
    for (int i = 0; i < CASES; i++) {
        chosen[i] = !strcmp(selected, "all") || !strcmp(selected, fixtures[i].name);
        count += chosen[i];
    }
    rows = !strcmp(restore_mode, "recover") ? 128 : 0;
    if (!model || !out || !count ||
        (strcmp(restore_mode, "recover") && strcmp(restore_mode, "exact") && strcmp(restore_mode, "fresh")) ||
        (verification != 0 && verification != 1) || (wanted - 256) % 128 ||
        wanted < 2048 || max_new < 1 || wanted + max_new >= ctx || chunk < 1 || chunk > 8192 ||
        (thinking != 0 && thinking != 1) || cache_gb < 1 || cache_gb > 128 ||
        (compressed_f16 != 0 && compressed_f16 != 1)) return 2;
    int rc = 1; char err[256] = {0}, setting[16];
    ds4_engine *e = NULL; ds4_session *s = NULL;
    ds4_tokens prompt = {0}, teacher = {0}, generated = {0};
    FILE *raw = NULL, *csv = NULL, *timings = NULL, *tokens = NULL, *answers = NULL, *formats = NULL;
    ds4_session_snapshot full_snapshot = {0}, blocks_snapshot = {0};
    float *logits = NULL; char *answer = NULL;
    snprintf(setting, sizeof(setting), "%d", 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", setting, 1) == 0);
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", compressed_f16 ? "1" : "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    raw = output(out, "logits.bin", "wbx"); csv = output(out, "rows.csv", "wx");
    timings = output(out, "timings.csv", "wx"); tokens = output(out, "tokens.json", "wx");
    answers = output(out, "answers.jsonl", "wx");
    formats = output(out, "formats.csv", "wx");
    CHECK(raw && csv && timings && tokens && answers && formats);
    fprintf(formats, "case,prefix_tokens,full_bytes,blocks_bytes,source_sync_ms,source_save_ms,mode,verify\n");
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = (uint32_t)ctx, .prefill_chunk = (uint32_t)chunk, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = (uint64_t)cache_gb << 30,
        .power_percent = 100};
    double open_started = now_ms();
    CHECK(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    double open_ms = now_ms() - open_started;
    CHECK(ds4_session_create(&s, e, ctx) == 0);
    int vocab = ds4_engine_vocab_size(e); CHECK(vocab > 0); probe_vocab = vocab;
    logits = malloc((size_t)vocab * sizeof(float)); CHECK(logits);
    fprintf(csv, "case,step,target,nll,argmax\n");
    fprintf(timings, "case,rows,prompt_tokens,teacher_tokens,sync_ms,teacher_ms,greedy_sync_ms,greedy_ms,generated_tokens,stopped,exact,approximate,payload_bytes,suffix_cache_bytes\n");
    ds4_context_memory memory = ds4_context_memory_estimate_with_prefill(DS4_BACKEND_METAL, ctx, chunk);
    fprintf(tokens, "{\"rows\":%d,\"vocab\":%d,\"context\":%d,\"chunk\":%d,\"compressed_kv_f16\":%d,"
        "\"planned_raw_bytes\":%llu,\"planned_compressed_bytes\":%llu,\"planned_total_bytes\":%llu,\"ssd_cache_gib\":%d,"
        "\"mode\":\"%s\",\"verify\":%d,\"teacher_trace\":%s,\"mtp\":false,\"thinking\":%s,\"max_new\":%d,\"native_routing\":true,\"logits_dtype\":\"float32_le\",\"engine_open_ms\":%.6f,\"cases\":[",
        rows, vocab, ctx, chunk, compressed_f16, (unsigned long long)memory.raw_bytes,
        (unsigned long long)memory.compressed_bytes, (unsigned long long)memory.total_bytes, cache_gb,
        restore_mode, verification, teacher_dir ? "true" : "false", thinking ? "true" : "false", max_new, open_ms);
    int row_start = 0, emitted = 0;
    for (int which = 0; which < CASES; which++) {
        if (!chosen[which]) continue;
        restore_calls = 0;
        CHECK(snprintf(frontier_path, sizeof(frontier_path), "%s/%s-recovered-frontier.bin", out, fixtures[which].name) < (int)sizeof(frontier_path));
        int record_start = 0, query_start = 0;
        CHECK(make_prompt(e, which, wanted, &prompt, &teacher, &record_start, &query_start, thinking));
        if (teacher_dir) CHECK(load_teacher(e, teacher_dir, fixtures[which].name, ctx - prompt.len, &teacher, err, sizeof(err)));
        fprintf(tokens, "%s{\"name\":", emitted++ ? "," : ""); json_text(tokens, fixtures[which].name);
        fprintf(tokens, ",\"logit_row_start\":%d,\"record_start\":%d,\"query_start\":%d,\"expected\":",
            row_start, record_start, query_start); json_text(tokens, fixtures[which].expected);
        fprintf(tokens, ",\"prompt\":"); json_tokens(tokens, &prompt);
        fprintf(tokens, ",\"teacher\":"); json_tokens(tokens, &teacher); fprintf(tokens, "}");
        fprintf(stderr, "CASE %s rows=%d prompt=%d teacher=%d record_start=%d query_start=%d\n",
            fixtures[which].name, rows, prompt.len, teacher.len, record_start, query_start);

        /* Build the same exact immutable prefix inputs in every arm. Setup is
         * reported separately and excluded from load/recovery timings. */
        int endpoint = wanted - 256;
        ds4_tokens prefix = {.v = prompt.v, .len = endpoint, .cap = endpoint};
        ds4_session_invalidate(s);
        double setup_started = now_ms();
        CHECK(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
        double source_sync_ms = now_ms() - setup_started;
        setup_started = now_ms();
        CHECK(ds4_session_save_snapshot(s, &full_snapshot, err, sizeof(err)) == 0);
        CHECK(make_blocks(&full_snapshot, &blocks_snapshot));
        CHECK(snprintf(full_path, sizeof(full_path), "%s/%s-full.bin", out, fixtures[which].name) < (int)sizeof(full_path));
        CHECK(snprintf(blocks_path, sizeof(blocks_path), "%s/%s-blocks.bin", out, fixtures[which].name) < (int)sizeof(blocks_path));
        CHECK(save_file(full_path, &full_snapshot) && save_file(blocks_path, &blocks_snapshot));
        fprintf(formats, "%s,%d,%llu,%llu,%.6f,%.6f,%s,%d\n", fixtures[which].name, endpoint,
            (unsigned long long)full_snapshot.len, (unsigned long long)blocks_snapshot.len,
            source_sync_ms, now_ms() - setup_started, restore_mode, verification);
        CHECK(fflush(formats) == 0);
        ds4_session_snapshot_free(&full_snapshot);
        ds4_session_snapshot_free(&blocks_snapshot);

        ds4_session_invalidate(s);
        double start = now_ms();
        CHECK(prepare(s, &prompt, err, sizeof(err)) == 0);
        double sync_ms = now_ms() - start;
        CHECK(ds4_session_pos(s) == prompt.len);
        bool approximate = ds4_session_is_approximate(s);
        uint64_t payload_bytes = ds4_session_payload_bytes(s);
        uint64_t suffix_cache_bytes = ds4_session_suffix_cache_bytes(s);
        CHECK(approximate == (!strcmp(restore_mode, "recover")));
        CHECK(!strcmp(restore_mode, "recover") ? payload_bytes == 0 && suffix_cache_bytes == 0 : payload_bytes > 0);
        CHECK(!ds4_session_is_decoder_replay(s));
        start = now_ms();
        for (int step = 0; step < teacher.len; step++) {
            CHECK(ds4_session_copy_logits(s, logits, vocab) == vocab);
            double z = log_z(logits, vocab); CHECK(isfinite(z));
            int target = teacher.v[step]; CHECK(target >= 0 && target < vocab);
            fprintf(csv, "%s,%d,%d,%.12g,%d\n", fixtures[which].name, step,
                target, z - logits[target], argmax(logits, vocab));
            CHECK(fwrite(logits, sizeof(float), (size_t)vocab, raw) == (size_t)vocab);
            if (step + 1 < teacher.len)
                CHECK(ds4_session_eval(s, target, err, sizeof(err)) == 0);
        }
        double teacher_ms = now_ms() - start;
        row_start += teacher.len;
        /* A fresh prefill prevents teacher tokens from conditioning the answer. */
        ds4_session_invalidate(s);
        start = now_ms();
        CHECK(prepare(s, &prompt, err, sizeof(err)) == 0);
        double greedy_sync_ms = now_ms() - start;
        size_t answer_len = 0; answer = malloc(1); CHECK(answer); answer[0] = '\0';
        bool stopped = false, in_thinking = thinking;
        start = now_ms();
        for (int step = 0; step < max_new; step++) {
            int token = ds4_session_argmax(s); CHECK(token >= 0 && token < vocab);
            if (ds4_token_is_stop(e, token)) { stopped = true; break; }
            ds4_tokens_push(&generated, token);
            if (in_thinking) {
                if (ds4_token_is_thinking_control(e, token)) in_thinking = false;
                CHECK(ds4_session_eval(s, token, err, sizeof(err)) == 0);
                continue;
            }
            size_t n = 0; char *piece = ds4_token_text(e, token, &n); CHECK(piece);
            char *grown = realloc(answer, answer_len + n + 1);
            if (!grown) { free(piece); CHECK(grown); }
            answer = grown; memcpy(answer + answer_len, piece, n); answer_len += n;
            answer[answer_len] = '\0'; free(piece);
            CHECK(ds4_session_eval(s, token, err, sizeof(err)) == 0);
        }
        double greedy_ms = now_ms() - start;
        bool exact = answer_equal(answer, fixtures[which].expected);
        fprintf(timings, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%llu,%llu\n",
            fixtures[which].name, rows, prompt.len, teacher.len, sync_ms, teacher_ms,
            greedy_sync_ms, greedy_ms, generated.len, stopped, exact, approximate,
            (unsigned long long)payload_bytes, (unsigned long long)suffix_cache_bytes);
        fprintf(answers, "{\"case\":"); json_text(answers, fixtures[which].name);
        fprintf(answers, ",\"expected\":"); json_text(answers, fixtures[which].expected);
        fprintf(answers, ",\"text\":"); json_text(answers, answer);
        fprintf(answers, ",\"tokens\":"); json_tokens(answers, &generated);
        fprintf(answers, ",\"stopped\":%s,\"exact\":%s}\n", stopped ? "true" : "false", exact ? "true" : "false");
        fprintf(stderr, "RESULT %s rows=%d sync_ms=%.3f teacher_ms=%.3f greedy_sync_ms=%.3f "
            "greedy_ms=%.3f generated=%d stopped=%d exact=%d answer=%s\n",
            fixtures[which].name, rows, sync_ms, teacher_ms, greedy_sync_ms,
            greedy_ms, generated.len, stopped, exact, answer);
        CHECK(fflush(raw) == 0 && fflush(csv) == 0 && fflush(timings) == 0 && fflush(answers) == 0);
        free(answer); answer = NULL;
        ds4_tokens_free(&prompt); ds4_tokens_free(&teacher); ds4_tokens_free(&generated);
    }
    fprintf(tokens, "]}\n"); CHECK(!ferror(tokens)); rc = 0;
done:
    if (raw && fclose(raw)) rc = 1;
    if (csv && fclose(csv)) rc = 1;
    if (timings && fclose(timings)) rc = 1;
    if (tokens && fclose(tokens)) rc = 1;
    if (answers && fclose(answers)) rc = 1;
    if (formats && fclose(formats)) rc = 1;
    ds4_session_snapshot_free(&full_snapshot); ds4_session_snapshot_free(&blocks_snapshot);
    free(logits); free(answer);
    ds4_tokens_free(&prompt); ds4_tokens_free(&teacher); ds4_tokens_free(&generated);
    ds4_session_free(s); ds4_engine_close(e);
    return rc;
}
