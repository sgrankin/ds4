/* Canonical F32 -> compressed-KV F16 -> F32 restore check.
 * One engine, fresh sessions, in-memory snapshots only. Restored logits are
 * never used as quality evidence: each arm evaluates one common new token.
 * Snapshot the odd 8193-token frontier to retain pending compression state. */
#include "ds4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTEXT 16384
#define PREFIX 8192
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "kv-pack-roundtrip stage=%s line=%d: %s; %s\n", \
            result.stage, __LINE__, #x, err); goto done; \
} } while (0)

typedef struct {
    const char *stage;
    int prefix_tokens, snapshot_teacher_token, teacher_token, vocab;
    uint64_t payload_a, payload_b, payload_c;
    bool snapshot_ab, snapshot_bc, logits_f16, logits_restored;
    bool native_finite, f16_finite, restored_finite, pass;
} roundtrip_result;

static const char *boolean(bool value) { return value ? "true" : "false"; }

static void summary(FILE *fp, const roundtrip_result *r) {
    fprintf(fp,
        "{\n  \"pass\":%s,\n  \"stage\":\"%s\",\n"
        "  \"prefix_tokens\":%d,\n  \"snapshot_prefix_teacher_token\":%d,\n"
        "  \"common_fresh_teacher_token\":%d,\n"
        "  \"vocab\":%d,\n  \"canonical_payload_a_bytes\":%llu,\n"
        "  \"canonical_payload_b_bytes\":%llu,\n"
        "  \"canonical_payload_c_bytes\":%llu,\n"
        "  \"native_to_f16_snapshot_byte_equal\":%s,\n"
        "  \"f16_to_native_snapshot_byte_equal\":%s,\n"
        "  \"f16_fresh_logits_byte_equal\":%s,\n"
        "  \"restored_native_fresh_logits_byte_equal\":%s,\n"
        "  \"all_fresh_logits_finite\":%s,\n"
        "  \"scope\":\"Fresh one-token cache reads after canonical cross-layout restore; "
        "loaded logits are not a computed-quality check. No latency claim.\"\n}\n",
        boolean(r->pass), r->stage, r->prefix_tokens, r->snapshot_teacher_token,
        r->teacher_token, r->vocab,
        (unsigned long long)r->payload_a, (unsigned long long)r->payload_b,
        (unsigned long long)r->payload_c, boolean(r->snapshot_ab),
        boolean(r->snapshot_bc), boolean(r->logits_f16), boolean(r->logits_restored),
        boolean(r->native_finite && r->f16_finite && r->restored_finite));
}

static bool finite_logits(const float *logits, int vocab) {
    for (int i = 0; i < vocab; i++) if (!isfinite(logits[i])) return false;
    return true;
}

static bool snapshot_equal(const ds4_session_snapshot *a,
                           const ds4_session_snapshot *b) {
    return a->len == b->len && a->len > 0 &&
        !memcmp(a->ptr, b->ptr, (size_t)a->len);
}

static bool make_prompt(ds4_engine *e, ds4_tokens *prompt, ds4_tokens *teacher) {
    ds4_tokens filler = {0}, tail = {0};
    ds4_tokenize_text(e,
        "Archive inspection: the shelves were cleaned on Monday. "
        "The notebooks are stored beside the workshop window. "
        "The cabinet label is WILLOW-48. The lamps and hinges were checked.\n",
        &filler);
    ds4_chat_begin(e, prompt);
    ds4_chat_append_message(e, prompt, "system",
        "Use the archive notes as context. Keep replies short.");
    ds4_chat_append_message(e, prompt, "user", "Archived inspection notes follow.\n");
    ds4_tokenize_text(e, "\nQuestion: what is the cabinet label?\n", &tail);
    ds4_chat_append_assistant_prefix(e, &tail, DS4_THINK_NONE);
    bool ok = filler.len > 0 && prompt->len + tail.len < PREFIX;
    if (ok) {
        const int start = prompt->len;
        while (prompt->len < PREFIX - tail.len)
            ds4_tokens_push(prompt, filler.v[(prompt->len - start) % filler.len]);
        for (int i = 0; i < tail.len; i++) ds4_tokens_push(prompt, tail.v[i]);
        ds4_tokenize_text(e, "WILLOW-48", teacher);
        ok = prompt->len == PREFIX && teacher->len > 1;
    }
    ds4_tokens_free(&filler);
    ds4_tokens_free(&tail);
    return ok;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("kv_pack_roundtrip_bench --model MODEL --out EXISTING_DIR");
            return 0;
        }
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else return 2;
    }
    if (!model || !out) return 2;
    char path[4096];
    const int path_len = snprintf(path, sizeof(path), "%s/summary.json", out);
    if (path_len < 0 || path_len >= (int)sizeof(path)) return 2;
    FILE *report = fopen(path, "wx");
    if (!report) { perror(path); return 2; }

    int rc = 1;
    char err[256] = {0};
    roundtrip_result result = {.stage = "settings", .snapshot_teacher_token = -1,
        .teacher_token = -1};
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_session_snapshot a = {0}, b = {0}, c = {0};
    ds4_tokens prompt = {0}, teacher = {0};
    float *native = NULL, *current = NULL;
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", "0", 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", "0", 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_SELECTED", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = CONTEXT, .prefill_chunk = PREFIX, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
        .power_percent = 100};
    result.stage = "engine_open";
    CHECK(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    result.vocab = ds4_engine_vocab_size(e);
    CHECK(result.vocab > 0);
    native = malloc((size_t)result.vocab * sizeof(float));
    current = malloc((size_t)result.vocab * sizeof(float));
    CHECK(native && current);
    CHECK(make_prompt(e, &prompt, &teacher));
    result.prefix_tokens = prompt.len;
    result.snapshot_teacher_token = teacher.v[0];
    result.teacher_token = teacher.v[1];
    CHECK(result.snapshot_teacher_token >= 0 && result.snapshot_teacher_token < result.vocab);
    CHECK(result.teacher_token >= 0 && result.teacher_token < result.vocab);

    result.stage = "native_sync_warm_save_a";
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == PREFIX && !ds4_session_is_approximate(s));
    CHECK(ds4_session_eval(s, result.snapshot_teacher_token, err, sizeof(err)) == 0);
    result.prefix_tokens = ds4_session_pos(s);
    CHECK(result.prefix_tokens == PREFIX + 1);
    CHECK(ds4_session_save_snapshot(s, &a, err, sizeof(err)) == 0);
    result.payload_a = a.len;
    result.stage = "native_fresh_eval";
    CHECK(ds4_session_eval(s, result.teacher_token, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix_tokens + 1);
    CHECK(ds4_session_copy_logits(s, native, result.vocab) == result.vocab);
    result.native_finite = finite_logits(native, result.vocab);
    CHECK(result.native_finite);
    ds4_session_free(s); s = NULL;

    result.stage = "f16_load_a_save_b";
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", "1", 1) == 0);
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    CHECK(ds4_session_load_snapshot(s, &a, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix_tokens && !ds4_session_is_approximate(s));
    CHECK(ds4_session_save_snapshot(s, &b, err, sizeof(err)) == 0);
    result.payload_b = b.len;
    result.snapshot_ab = snapshot_equal(&a, &b);
    CHECK(result.snapshot_ab);
    ds4_session_snapshot_free(&a); /* Keep at most two canonical snapshots. */
    result.stage = "f16_fresh_eval";
    CHECK(ds4_session_eval(s, result.teacher_token, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix_tokens + 1);
    CHECK(ds4_session_copy_logits(s, current, result.vocab) == result.vocab);
    result.f16_finite = finite_logits(current, result.vocab);
    result.logits_f16 = !memcmp(native, current, (size_t)result.vocab * sizeof(float));
    CHECK(result.f16_finite && result.logits_f16);
    ds4_session_free(s); s = NULL;

    result.stage = "native_load_b_save_c";
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", "0", 1) == 0);
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    CHECK(ds4_session_load_snapshot(s, &b, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix_tokens && !ds4_session_is_approximate(s));
    CHECK(ds4_session_save_snapshot(s, &c, err, sizeof(err)) == 0);
    result.payload_c = c.len;
    result.snapshot_bc = snapshot_equal(&b, &c);
    CHECK(result.snapshot_bc);
    result.stage = "restored_native_fresh_eval";
    CHECK(ds4_session_eval(s, result.teacher_token, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix_tokens + 1);
    CHECK(ds4_session_copy_logits(s, current, result.vocab) == result.vocab);
    result.restored_finite = finite_logits(current, result.vocab);
    result.logits_restored = !memcmp(native, current, (size_t)result.vocab * sizeof(float));
    CHECK(result.restored_finite && result.logits_restored);
    result.stage = "complete";
    result.pass = true;
    rc = 0;
done:
    summary(report, &result);
    if (fclose(report)) rc = 1;
    summary(stdout, &result);
    if (fflush(stdout)) rc = 1;
    ds4_session_free(s);
    ds4_engine_close(e);
    ds4_session_snapshot_free(&a);
    ds4_session_snapshot_free(&b);
    ds4_session_snapshot_free(&c);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&teacher);
    free(native);
    free(current);
    return rc;
}
