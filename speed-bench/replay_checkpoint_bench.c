/* Private replay checkpoint -> SSD close/reopen -> F16 receiving session.
 * Loaded logits are not quality evidence: compare one freshly evaluated token. */
#include "ds4.h"
#include "ds4_state_cache.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTEXT 16384
#define PREFIX 8192
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "replay-checkpoint stage=%s line=%d: %s; %s\n", \
            result.stage, __LINE__, #x, err); goto done; \
} } while (0)

typedef struct {
    const char *stage;
    int prefix, snapshot_token, fresh_token, vocab;
    uint64_t payload, disk_reads;
    bool source_replay, restored_replay, restored_approximate, canonical_blocked;
    bool suffix_blocked, profile_miss, suffix_miss, disk_equal, snapshot_equal;
    bool native_finite, restored_finite, logits_equal, pass;
} checkpoint_result;

static const char *boolean(bool x) { return x ? "true" : "false"; }
static void summary(FILE *fp, const checkpoint_result *r) {
    fprintf(fp,
        "{\n  \"pass\":%s,\n  \"stage\":\"%s\",\n"
        "  \"prefix_tokens\":%d,\n  \"snapshot_teacher_token\":%d,\n"
        "  \"common_fresh_teacher_token\":%d,\n  \"vocab\":%d,\n"
        "  \"private_snapshot_bytes\":%llu,\n  \"cache_disk_reads\":%llu,\n"
        "  \"source_decoder_replay\":%s,\n  \"restored_decoder_replay\":%s,\n"
        "  \"restored_approximate\":%s,\n  \"canonical_export_blocked\":%s,\n"
        "  \"suffix_capture_blocked\":%s,\n  \"different_profile_miss\":%s,\n"
        "  \"suffix_lookup_miss\":%s,\n  \"disk_materialization_byte_equal\":%s,\n"
        "  \"private_resave_byte_equal\":%s,\n  \"fresh_logits_byte_equal\":%s,\n"
        "  \"all_fresh_logits_finite\":%s,\n"
        "  \"scope\":\"K128/selected1 odd-frontier private SSD checkpoint, F32 to F16; "
        "one fresh cache-reading token, not loaded-logit or latency evidence.\"\n}\n",
        boolean(r->pass), r->stage, r->prefix, r->snapshot_token, r->fresh_token,
        r->vocab, (unsigned long long)r->payload, (unsigned long long)r->disk_reads,
        boolean(r->source_replay), boolean(r->restored_replay),
        boolean(r->restored_approximate), boolean(r->canonical_blocked),
        boolean(r->suffix_blocked), boolean(r->profile_miss), boolean(r->suffix_miss),
        boolean(r->disk_equal), boolean(r->snapshot_equal), boolean(r->logits_equal),
        boolean(r->native_finite && r->restored_finite));
}
static bool finite_logits(const float *p, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(p[i])) return false;
    return true;
}
static bool make_prompt(ds4_engine *e, ds4_tokens *prompt, ds4_tokens *teacher) {
    ds4_tokens filler = {0}, tail = {0};
    ds4_tokenize_text(e, "Archive inspection: shelves and hinges were checked. "
        "The cabinet label is WILLOW-50. The notebooks remain beside the window.\n", &filler);
    ds4_chat_begin(e, prompt);
    ds4_chat_append_message(e, prompt, "system", "Use the archive notes. Keep replies short.");
    ds4_chat_append_message(e, prompt, "user", "Archived inspection notes follow.\n");
    ds4_tokenize_text(e, "\nQuestion: what is the cabinet label?\n", &tail);
    ds4_chat_append_assistant_prefix(e, &tail, DS4_THINK_NONE);
    bool ok = filler.len > 0 && prompt->len + tail.len < PREFIX;
    if (ok) {
        const int first = prompt->len;
        while (prompt->len < PREFIX - tail.len)
            ds4_tokens_push(prompt, filler.v[(prompt->len - first) % filler.len]);
        for (int i = 0; i < tail.len; i++) ds4_tokens_push(prompt, tail.v[i]);
        ds4_tokenize_text(e, "WILLOW-50", teacher);
        ok = prompt->len == PREFIX && teacher->len > 1;
    }
    ds4_tokens_free(&filler); ds4_tokens_free(&tail);
    return ok;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            puts("replay_checkpoint_bench --model MODEL --out EXISTING_DIR"); return 0;
        }
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else return 2;
    }
    if (!model || !out) return 2;
    char path[4096], cache_dir[4096], err[256] = {0}, key[41];
    int n = snprintf(path, sizeof(path), "%s/summary.json", out);
    if (n < 0 || n >= (int)sizeof(path)) return 2;
    n = snprintf(cache_dir, sizeof(cache_dir), "%s/cache", out);
    if (n < 0 || n >= (int)sizeof(cache_dir)) return 2;
    FILE *report = fopen(path, "wx");
    if (!report) { perror(path); return 2; }
    const char *profile = "replay-checkpoint-1017-K128-selected1-native-routing";
    int rc = 1;
    checkpoint_result result = {.stage = "settings", .snapshot_token = -1, .fresh_token = -1};
    ds4_engine *e = NULL; ds4_session *s = NULL;
    ds4_state_cache *cache = NULL; ds4_state_cache_ref *ref = NULL;
    ds4_state_cache_blob blob = {0}; ds4_state_cache_stats stats = {0};
    ds4_session_snapshot a = {0}, b = {0}; ds4_suffix_cache suffix = {0};
    ds4_tokens prompt = {0}, teacher = {0};
    float *reference = NULL, *current = NULL;
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", "0", 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", "128", 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_SELECTED", "1", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = CONTEXT, .prefill_chunk = PREFIX, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
        .power_percent = 100};
    result.stage = "source_sync";
    CHECK(ds4_engine_open(&e, &opt) == 0 && ds4_engine_is_deepseek41(e));
    result.vocab = ds4_engine_vocab_size(e); CHECK(result.vocab > 0);
    reference = malloc((size_t)result.vocab * sizeof(float));
    current = malloc((size_t)result.vocab * sizeof(float)); CHECK(reference && current);
    CHECK(make_prompt(e, &prompt, &teacher));
    result.snapshot_token = teacher.v[0]; result.fresh_token = teacher.v[1];
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    result.source_replay = ds4_session_is_decoder_replay(s); CHECK(result.source_replay);
    CHECK(ds4_session_eval(s, result.snapshot_token, err, sizeof(err)) == 0);
    ds4_tokens_push(&prompt, result.snapshot_token);
    result.prefix = ds4_session_pos(s); CHECK(result.prefix == PREFIX + 1);
    result.stage = "save_private_and_fresh_reference";
    CHECK(ds4_session_save_replay_snapshot(s, &a, err, sizeof(err)) == 0);
    result.payload = a.len; CHECK(a.len > DS4_SESSION_REPLAY_SNAPSHOT_HEADER_BYTES);
    CHECK(ds4_session_eval(s, result.fresh_token, err, sizeof(err)) == 0);
    CHECK(ds4_session_copy_logits(s, reference, result.vocab) == result.vocab);
    result.native_finite = finite_logits(reference, result.vocab); CHECK(result.native_finite);
    ds4_session_free(s); s = NULL;
    result.stage = "ssd_put_close_reopen";
    cache = ds4_state_cache_open(cache_dir, 0, UINT64_C(512) << 20, err, sizeof(err)); CHECK(cache);
    CHECK(ds4_state_cache_put_kind(cache, profile, a.ptr, a.len, NULL, 0,
        DS4_STATE_CACHE_REPLAY, key, err, sizeof(err)) == 0);
    CHECK(ds4_state_cache_close(cache, err, sizeof(err)) == 0); cache = NULL;
    cache = ds4_state_cache_open(cache_dir, 0, UINT64_C(512) << 20, err, sizeof(err)); CHECK(cache);
    ref = ds4_state_cache_find_prefix(cache, "different-K256-profile", CONTEXT,
        prompt.v, (uint32_t)prompt.len, NULL);
    result.profile_miss = ref == NULL; CHECK(result.profile_miss);
    ref = ds4_state_cache_find_suffix(cache, profile, CONTEXT,
        prompt.v + prompt.len - 64, 64, NULL);
    result.suffix_miss = ref == NULL; CHECK(result.suffix_miss);
    uint32_t matched = 0;
    ref = ds4_state_cache_find_prefix(cache, profile, CONTEXT,
        prompt.v, (uint32_t)prompt.len, &matched);
    CHECK(ref && matched == (uint32_t)result.prefix);
    CHECK(ds4_state_cache_materialize(ref, &blob, err, sizeof(err)) == 0);
    ds4_state_cache_release(ref); ref = NULL;
    CHECK(blob.kind == DS4_STATE_CACHE_REPLAY && blob.approximate && !blob.prerope_bytes);
    result.disk_equal = blob.state_bytes == a.len && !memcmp(blob.state, a.ptr, (size_t)a.len);
    CHECK(result.disk_equal); ds4_session_snapshot_free(&a);
    ds4_state_cache_get_stats(cache, &stats); result.disk_reads = stats.disk_reads;
    CHECK(result.disk_reads > 0 && stats.hot_bytes == 0);
    CHECK(ds4_state_cache_close(cache, err, sizeof(err)) == 0); cache = NULL;
    result.stage = "fresh_f16_private_restore";
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", "1", 1) == 0);
    CHECK(ds4_session_create(&s, e, CONTEXT) == 0);
    ds4_session_snapshot view = {.ptr = blob.state, .len = blob.state_bytes, .cap = blob.state_bytes};
    CHECK(ds4_session_load_replay_snapshot(s, &view, err, sizeof(err)) == 0);
    result.restored_replay = ds4_session_is_decoder_replay(s);
    result.restored_approximate = ds4_session_is_approximate(s);
    CHECK(result.restored_replay && result.restored_approximate && ds4_session_pos(s) == result.prefix);
    result.canonical_blocked = ds4_session_payload_bytes(s) == 0; CHECK(result.canonical_blocked);
    result.suffix_blocked = ds4_session_capture_suffix_cache(s, &suffix, err, sizeof(err)) != 0;
    CHECK(result.suffix_blocked); err[0] = 0;
    CHECK(ds4_session_save_replay_snapshot(s, &b, err, sizeof(err)) == 0);
    result.snapshot_equal = b.len == blob.state_bytes && !memcmp(b.ptr, blob.state, (size_t)b.len);
    CHECK(result.snapshot_equal);
    result.stage = "restored_fresh_eval";
    CHECK(ds4_session_eval(s, result.fresh_token, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(s) == result.prefix + 1);
    CHECK(ds4_session_copy_logits(s, current, result.vocab) == result.vocab);
    result.restored_finite = finite_logits(current, result.vocab);
    result.logits_equal = !memcmp(reference, current, (size_t)result.vocab * sizeof(float));
    CHECK(result.restored_finite && result.logits_equal);
    result.stage = "complete"; result.pass = true; rc = 0;
done:
    summary(report, &result); if (fclose(report)) rc = 1;
    summary(stdout, &result); if (fflush(stdout)) rc = 1;
    ds4_state_cache_release(ref);
    if (cache && ds4_state_cache_close(cache, err, sizeof(err))) rc = 1;
    ds4_state_cache_blob_free(&blob); ds4_suffix_cache_free(&suffix);
    ds4_session_snapshot_free(&a); ds4_session_snapshot_free(&b);
    ds4_session_free(s); ds4_engine_close(e);
    ds4_tokens_free(&prompt); ds4_tokens_free(&teacher); free(reference); free(current);
    return rc;
}
