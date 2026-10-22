/* Capture-only DSpark target verification: no trained drafter is invoked.
 * Include the real engine as existing graph/DSpark fixtures do. */
#include "../../../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "dspark-target stage=%s line=%d: %s; %s\n", \
            stage, __LINE__, #x, err); goto done; \
} } while (0)

static const char *bool_text(bool value) { return value ? "true" : "false"; }

static bool capture_positions_equal(const ds41_dspark *draft, uint32_t end) {
    uint32_t expected[DS41_DRAFT_WINDOW];
    for (unsigned slot = 0; slot < DS41_DRAFT_WINDOW; slot++) expected[slot] = UINT32_MAX;
    const uint32_t first = end > DS41_DRAFT_WINDOW ? end - DS41_DRAFT_WINDOW : 0;
    for (uint32_t pos = first; pos < end; pos++) expected[pos % DS41_DRAFT_WINDOW] = pos;
    for (unsigned target = 0; target < 3; target++)
        if (memcmp(draft->positions[target], expected, sizeof(expected))) return false;
    return true;
}

static bool compare_logits(FILE *report, FILE *raw, uint32_t prefix,
        uint32_t keep, unsigned step, const float *verified, const float *ordinary,
        uint32_t vocab, uint64_t *offset) {
    double maximum = 0;
    for (uint32_t token = 0; token < vocab; token++) {
        if (!isfinite(verified[token]) || !isfinite(ordinary[token])) return false;
        const double difference = fabs((double)verified[token] - ordinary[token]);
        if (difference > maximum) maximum = difference;
    }
    const size_t bytes = (size_t)vocab * sizeof(float);
    if (fwrite(verified, 1, bytes, raw) != bytes ||
        fwrite(ordinary, 1, bytes, raw) != bytes) return false;
    fprintf(report, "{\"kind\":\"logits\",\"prefix\":%u,\"keep\":%u,\"step\":%u,"
        "\"freshly_computed\":%s,\"finite\":true,\"byte_equal\":%s,\"max_abs\":%.9g,"
        "\"verified_fnv\":\"%016llx\",\"ordinary_fnv\":\"%016llx\",\"raw_offset\":%llu}\n",
        prefix, keep, step, bool_text(step != 0),
        bool_text(!memcmp(verified, ordinary, bytes)), maximum,
        (unsigned long long)hash_bytes(verified, bytes),
        (unsigned long long)hash_bytes(ordinary, bytes), (unsigned long long)*offset);
    *offset += 2u * bytes;
    return fflush(report) == 0 && fflush(raw) == 0;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL, *f16 = "0";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--f16") && i + 1 < argc) f16 = argv[++i];
        else return 2;
    }
    if (!model || !out || (strcmp(f16, "0") && strcmp(f16, "1"))) return 2;
    char path[4096], err[256] = {0};
    const char *stage = "output";
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *verified = NULL, *ordinary = NULL;
    ds4_tokens prompt = {0};
    ds4_session_snapshot a = {0}, b = {0};
    FILE *report = NULL, *raw = NULL;
    int length = snprintf(path, sizeof(path), "%s/results.jsonl", out);
    if (length < 0 || length >= (int)sizeof(path)) return 2;
    report = fopen(path, "wx");
    if (!report) return 2;
    length = snprintf(path, sizeof(path), "%s/logits.bin", out);
    CHECK(length >= 0 && length < (int)sizeof(path));
    raw = fopen(path, "wbx");
    CHECK(raw);
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", f16, 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    ds4_engine_options options = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 256, .prefill_chunk = 256, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(32) << 30,
        .power_percent = 100};
    /* Only allocation/capture/verify/commit read this metadata. Every trained
     * draft tensor remains NULL; seed/stage/forward/Markov are never called. */
    const ds4_dspark_weights capture_weights = {.v41 = true, .n_stages = 3,
        .block_size = DS41_DRAFT_ROWS, .markov_rank = 256,
        .target_layer_count = 3, .target_layers = {37, 38, 39}};
    stage = "engine_and_capture_owner";
    CHECK(ds4_engine_open(&engine, &options) == 0 && ds4_engine_is_deepseek41(engine));
    CHECK(ds4_session_create(&verified, engine, 256) == 0);
    CHECK(ds4_session_create(&ordinary, engine, 256) == 0);
    CHECK(!verified->ds41.graph.dspark && !ordinary->ds41.graph.dspark);
    verified->ds41.graph.dspark = ds41_dspark_alloc(&capture_weights);
    CHECK(verified->ds41.graph.dspark);
    const uint32_t vocab = DS4_N_VOCAB;
    /* Six little-endian u32 words, then paired full-F32 vectors in JSONL order:
     * capture/verified first, ordinary target second. step0 is the committed
     * frontier; steps1..4 follow an actual new common target token. */
    const uint32_t header[6] = {UINT32_C(0x31565344), vocab, 2, 3, 5, 2};
    CHECK(fwrite(header, sizeof(header), 1, raw) == 1);
    uint64_t offset = sizeof(header);
    for (unsigned i = 0; i < 129; i++) ds4_tokens_push(&prompt, 300 + (int)(i % 23));
    const uint32_t prefixes[] = {127, 129}, keeps[] = {0, 3, 6};
    const int target[6] = {700, 701, 702, 703, 704, 705};
    for (unsigned pi = 0; pi < 2; pi++) for (unsigned ki = 0; ki < 3; ki++) {
        const uint32_t prefix = prefixes[pi], keep = keeps[ki];
        stage = "prefix_capture";
        ds4_session_invalidate(verified);
        ds4_session_invalidate(ordinary);
        prompt.len = (int)prefix;
        CHECK(ds4_session_sync(verified, &prompt, err, sizeof(err)) == 0);
        CHECK(ds4_session_sync(ordinary, &prompt, err, sizeof(err)) == 0);
        CHECK(capture_positions_equal(verified->ds41.graph.dspark, prefix));
        stage = "verify_six_commit_prefix";
        CHECK(ds41_session_verify(verified, target, 6));
        CHECK(verified->ds41.graph.dspark->verify_count == 6);
        CHECK(ds41_session_commit(verified, target, 6, keep));
        for (unsigned i = 0; i < keep; i++)
            CHECK(ds4_session_eval(ordinary, target[i], err, sizeof(err)) == 0);
        const uint32_t accepted = prefix + keep;
        CHECK(verified->frontier.checkpoint_valid && ordinary->frontier.checkpoint_valid &&
            verified->frontier.checkpoint.len == (int)accepted && ordinary->frontier.checkpoint.len == (int)accepted &&
            verified->ds41.graph.pos == accepted && ordinary->ds41.graph.pos == accepted &&
            !memcmp(verified->frontier.checkpoint.v, ordinary->frontier.checkpoint.v, accepted * sizeof(int)));
        CHECK(capture_positions_equal(verified->ds41.graph.dspark, accepted));
        CHECK(!memcmp(&verified->ds41.graph.history, &ordinary->ds41.graph.history,
            sizeof(verified->ds41.graph.history)));
        stage = "canonical_frontier";
        CHECK(ds4_session_save_snapshot(verified, &a, err, sizeof(err)) == 0);
        CHECK(ds4_session_save_snapshot(ordinary, &b, err, sizeof(err)) == 0);
        CHECK(a.len && a.len == b.len && a.len == ds4_session_payload_bytes(verified));
        const size_t cache_offset = DS4_SESSION_PAYLOAD_U32_FIELDS * 4u +
            ((size_t)accepted + vocab) * sizeof(float);
        CHECK(cache_offset < a.len);
        fprintf(report, "{\"kind\":\"frontier\",\"prefix\":%u,\"keep\":%u,\"position\":%u,"
            "\"payload_bytes\":%llu,\"payload_equal\":%s,\"cache_equal\":%s,"
            "\"capture_positions_equal\":true,\"history_equal\":true}\n",
            prefix, keep, accepted, (unsigned long long)a.len,
            bool_text(!memcmp(a.ptr, b.ptr, (size_t)a.len)),
            bool_text(!memcmp(a.ptr + cache_offset, b.ptr + cache_offset, (size_t)a.len - cache_offset)));
        ds4_session_snapshot_free(&a); ds4_session_snapshot_free(&b);
        CHECK(compare_logits(report, raw, prefix, keep, 0, verified->frontier.logits, ordinary->frontier.logits, vocab, &offset));
        stage = "fresh_target_continuation";
        for (unsigned step = 1; step <= 4; step++) {
            const int token = 1200 + (int)step;
            CHECK(ds4_session_eval(verified, token, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(ordinary, token, err, sizeof(err)) == 0);
            CHECK(verified->frontier.checkpoint_valid && ordinary->frontier.checkpoint_valid &&
                verified->ds41.graph.pos == accepted + step && ordinary->ds41.graph.pos == accepted + step);
            CHECK(compare_logits(report, raw, prefix, keep, step, verified->frontier.logits, ordinary->frontier.logits, vocab, &offset));
        }
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    fprintf(report, "{\"kind\":\"result\",\"pass\":%s,\"stage\":\"%s\"}\n", bool_text(rc == 0), stage);
    if (raw && fclose(raw)) rc = 1;
    if (report && fclose(report)) rc = 1;
    ds4_session_snapshot_free(&a); ds4_session_snapshot_free(&b);
    ds4_tokens_free(&prompt);
    ds4_session_free(ordinary); ds4_session_free(verified);
    ds4_engine_close(engine);
    return rc;
}
