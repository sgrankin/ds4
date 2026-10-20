/* Paired DS4.1 decoder replay experiment. One process per replay setting;
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
    return end == value || *end || n < 0 || n > 131072 ? -1 : (int)n;
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
                        ds4_tokens *teacher, int *record_start, int *query_start) {
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
    ds4_chat_append_assistant_prefix(e, &tail, DS4_THINK_NONE);
    bool ok = filler.len && wanted > tail.len + prompt->len + 1024;
    if (ok && f->old_record) {
        ok = pad_to(prompt, &filler, wanted / 8);
        if (ok) ds4_tokenize_text(e, f->old_record, prompt);
    }
    int at = which == 0 ? wanted / 2 : which == 1 ? wanted - tail.len - 48 : wanted - 512;
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

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL, *selected = "all";
    int rows = 0, wanted = 8192, chunk = 8192;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("decoder_replay_bench --model MODEL --out EXISTING_DIR --rows 0|128|256 "
                 "[--tokens 8192] [--chunk 8192] [--case all|ordinary|favorable|changed]");
            return 0;
        }
        if (++i >= argc) return 2;
        const char *key = argv[i - 1], *value = argv[i];
        if (!strcmp(key, "--model")) model = value;
        else if (!strcmp(key, "--out")) out = value;
        else if (!strcmp(key, "--case")) selected = value;
        else if (!strcmp(key, "--rows")) rows = number(value);
        else if (!strcmp(key, "--tokens")) wanted = number(value);
        else if (!strcmp(key, "--chunk")) chunk = number(value);
        else return 2;
    }
    bool chosen[CASES] = {0}; int count = 0;
    for (int i = 0; i < CASES; i++) {
        chosen[i] = !strcmp(selected, "all") || !strcmp(selected, fixtures[i].name);
        count += chosen[i];
    }
    if (!model || !out || !count || (rows != 0 && rows != 128 && rows != 256) ||
        wanted < 2048 || wanted + MAX_NEW >= CONTEXT || chunk < 1 || chunk > 8192) return 2;
    int rc = 1; char err[256] = {0}, setting[16];
    ds4_engine *e = NULL; ds4_session *s = NULL;
    ds4_tokens prompt = {0}, teacher = {0}, generated = {0};
    FILE *raw = NULL, *csv = NULL, *timings = NULL, *tokens = NULL, *answers = NULL;
    float *logits = NULL; char *answer = NULL;
    snprintf(setting, sizeof(setting), "%d", rows);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", setting, 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    raw = output(out, "logits.bin", "wbx"); csv = output(out, "rows.csv", "wx");
    timings = output(out, "timings.csv", "wx"); tokens = output(out, "tokens.json", "wx");
    answers = output(out, "answers.jsonl", "wx");
    CHECK(raw && csv && timings && tokens && answers);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = CONTEXT, .prefill_chunk = (uint32_t)chunk, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
        .power_percent = 100};
    double open_started = now_ms();
    CHECK(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    double open_ms = now_ms() - open_started;
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    int vocab = ds4_engine_vocab_size(e); CHECK(vocab > 0);
    logits = malloc((size_t)vocab * sizeof(float)); CHECK(logits);
    fprintf(csv, "case,step,target,nll,argmax\n");
    fprintf(timings, "case,rows,prompt_tokens,teacher_tokens,sync_ms,teacher_ms,greedy_sync_ms,greedy_ms,generated_tokens,stopped,exact,approximate,payload_bytes,suffix_cache_bytes\n");
    fprintf(tokens, "{\"rows\":%d,\"vocab\":%d,\"context\":%d,\"chunk\":%d,\"ssd_cache_gib\":64,"
        "\"mtp\":false,\"native_routing\":true,\"logits_dtype\":\"float32_le\",\"engine_open_ms\":%.6f,\"cases\":[",
        rows, vocab, CONTEXT, chunk, open_ms);
    int row_start = 0, emitted = 0;
    for (int which = 0; which < CASES; which++) {
        if (!chosen[which]) continue;
        int record_start = 0, query_start = 0;
        CHECK(make_prompt(e, which, wanted, &prompt, &teacher, &record_start, &query_start));
        fprintf(tokens, "%s{\"name\":", emitted++ ? "," : ""); json_text(tokens, fixtures[which].name);
        fprintf(tokens, ",\"logit_row_start\":%d,\"record_start\":%d,\"query_start\":%d,\"expected\":",
            row_start, record_start, query_start); json_text(tokens, fixtures[which].expected);
        fprintf(tokens, ",\"prompt\":"); json_tokens(tokens, &prompt);
        fprintf(tokens, ",\"teacher\":"); json_tokens(tokens, &teacher); fprintf(tokens, "}");
        fprintf(stderr, "CASE %s rows=%d prompt=%d teacher=%d record_start=%d query_start=%d\n",
            fixtures[which].name, rows, prompt.len, teacher.len, record_start, query_start);
        ds4_session_invalidate(s);
        double start = now_ms();
        CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        double sync_ms = now_ms() - start;
        CHECK(ds4_session_pos(s) == prompt.len);
        bool approximate = ds4_session_is_approximate(s);
        uint64_t payload_bytes = ds4_session_payload_bytes(s);
        uint64_t suffix_cache_bytes = ds4_session_suffix_cache_bytes(s);
        if (wanted == 8192 && chunk == 8192) {
            CHECK(approximate == (rows != 0));
            CHECK(rows ? payload_bytes == 0 && suffix_cache_bytes == 0 : payload_bytes > 0);
        }
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
        CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        double greedy_sync_ms = now_ms() - start;
        size_t answer_len = 0; answer = malloc(1); CHECK(answer); answer[0] = '\0';
        bool stopped = false;
        start = now_ms();
        for (int step = 0; step < MAX_NEW; step++) {
            int token = ds4_session_argmax(s); CHECK(token >= 0 && token < vocab);
            if (ds4_token_is_stop(e, token)) { stopped = true; break; }
            ds4_tokens_push(&generated, token);
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
    free(logits); free(answer);
    ds4_tokens_free(&prompt); ds4_tokens_free(&teacher); ds4_tokens_free(&generated);
    ds4_session_free(s); ds4_engine_close(e);
    return rc;
}
