/* Public-API canonical payload exchange. One graph, six short frontiers.
 * A freshly evaluated teacher token forces real cache reads after each load;
 * restored/cached logits alone are never treated as continuation evidence. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "payload-roundtrip stage=%s line=%d: %s; %s\n", \
            stage, __LINE__, #x, err); goto done; \
} } while (0)

static bool equal(const ds4_session_snapshot *a, const ds4_session_snapshot *b) {
    return a->len && a->len == b->len && !memcmp(a->ptr, b->ptr, (size_t)a->len);
}
static const char *boolean(bool value) { return value ? "true" : "false"; }
static double max_error(const float *a, const float *b, int count) {
    double maximum = 0;
    for (int i = 0; i < count; i++) {
        double difference = fabs((double)a[i] - b[i]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}
static bool finite_logits(const float *values, int count) {
    for (int i = 0; i < count; i++) if (!isfinite(values[i])) return false;
    return true;
}
static bool write_bytes(const char *directory, const char *name,
                        const void *bytes, size_t count) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (n < 0 || n >= (int)sizeof(path)) return false;
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool ok = fwrite(bytes, 1, count, fp) == count;
    return fclose(fp) == 0 && ok;
}
static bool read_bytes(const char *directory, const char *name,
                       ds4_session_snapshot *bytes) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (n < 0 || n >= (int)sizeof(path)) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    bool ok = fseek(fp, 0, SEEK_END) == 0;
    long length = ok ? ftell(fp) : -1;
    if (length <= 0 || fseek(fp, 0, SEEK_SET) != 0) ok = false;
    if (ok) {
        bytes->ptr = malloc((size_t)length);
        bytes->len = bytes->cap = (uint64_t)length;
        ok = bytes->ptr && fread(bytes->ptr, 1, (size_t)length, fp) == (size_t)length;
    }
    return fclose(fp) == 0 && ok;
}

int main(int argc, char **argv) {
    const char *model = NULL, *out = NULL, *source = NULL, *f16 = "0", *source_f16 = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--source") && i + 1 < argc) source = argv[++i];
        else if (!strcmp(argv[i], "--f16") && i + 1 < argc) f16 = argv[++i];
        else if (!strcmp(argv[i], "--source-f16") && i + 1 < argc) source_f16 = argv[++i];
        else return 2;
    }
    if (!source_f16) source_f16 = f16;
    if (!model || !out || (strcmp(f16, "0") && strcmp(f16, "1")) ||
        (strcmp(source_f16, "0") && strcmp(source_f16, "1"))) return 2;
    const bool cross_storage = source && strcmp(f16, source_f16);
    char path[4096], name[80], err[256] = {0};
    int n = snprintf(path, sizeof(path), "%s/results.jsonl", out);
    if (n < 0 || n >= (int)sizeof(path)) return 2;
    FILE *report = fopen(path, "wx");
    if (!report) { perror(path); return 2; }
    const char *stage = "settings";
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot own = {0}, peer = {0}, roundtrip = {0}, scratch = {0};
    ds4_tokens tokens = {0};
    float *reference = NULL, *actual = NULL;
    FILE *truncated = NULL;
    CHECK(setenv("DS4_V41_COMPRESSED_KV_F16", f16, 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_ROWS", "0", 1) == 0);
    CHECK(setenv("DS4_V41_DECODER_REPLAY_SELECTED", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP", "0", 1) == 0);
    ds4_engine_options options = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 256, .prefill_chunk = 256, .mtp_draft_tokens = 1,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(32) << 30,
        .power_percent = 100};
    stage = "engine";
    CHECK(ds4_engine_open(&engine, &options) == 0 && ds4_engine_is_deepseek41(engine));
    int vocab = ds4_engine_vocab_size(engine);
    CHECK(vocab > 987);
    const size_t logits_bytes = (size_t)vocab * sizeof(float);
    reference = malloc(logits_bytes); actual = malloc(logits_bytes);
    CHECK(reference && actual);
    CHECK(ds4_session_create(&session, engine, 256) == 0);
    for (int i = 0; i < 129; i++) ds4_tokens_push(&tokens, 100 + i);
    const int frontiers[] = {1, 2, 3, 127, 128, 129};
    for (size_t i = 0; i < sizeof(frontiers) / sizeof(*frontiers); i++) {
        tokens.len = frontiers[i];
        stage = "independent_frontier";
        CHECK(ds4_session_sync(session, &tokens, err, sizeof(err)) == 0);
        CHECK(ds4_session_pos(session) == tokens.len && ds4_session_checkpoint_valid(session));
        CHECK(ds4_session_save_snapshot(session, &own, err, sizeof(err)) == 0);
        CHECK(own.len == ds4_session_payload_bytes(session));
        CHECK(own.len > DS4_SESSION_PAYLOAD_U32_FIELDS * 4u && !memcmp(own.ptr, "DSV4", 4));
        CHECK(own.ptr[4] == DS4_SESSION_PAYLOAD_VERSION && !own.ptr[5] && !own.ptr[6] && !own.ptr[7]);
        snprintf(name, sizeof(name), "frontier-%03d.dsv4", tokens.len);
        CHECK(write_bytes(out, name, own.ptr, (size_t)own.len));
        stage = "fresh_reference";
        CHECK(ds4_session_eval(session, 987, err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(session, reference, vocab) == vocab);
        CHECK(finite_logits(reference, vocab));
        snprintf(name, sizeof(name), "frontier-%03d.fresh.f32", tokens.len);
        CHECK(write_bytes(out, name, reference, logits_bytes));
        bool source_wire_equal = true, source_fresh_equal = true;
        if (source) {
            stage = "peer_canonical";
            snprintf(name, sizeof(name), "frontier-%03d.dsv4", tokens.len);
            CHECK(read_bytes(source, name, &peer));
            source_wire_equal = equal(&own, &peer);
            CHECK(cross_storage || source_wire_equal);
            snprintf(name, sizeof(name), "frontier-%03d.fresh.f32", tokens.len);
            CHECK(read_bytes(source, name, &scratch));
            CHECK(scratch.len == logits_bytes && finite_logits((float *)scratch.ptr, vocab));
            source_fresh_equal = !memcmp(reference, scratch.ptr, logits_bytes);
            CHECK(cross_storage || source_fresh_equal);
        }
        const ds4_session_snapshot *saved = source ? &peer : &own;
        stage = "restore_and_resave";
        CHECK(ds4_session_load_snapshot(session, saved, err, sizeof(err)) == 0);
        CHECK(ds4_session_pos(session) == tokens.len && ds4_session_checkpoint_valid(session));
        CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) == 0);
        const bool resave_equal = equal(saved, &roundtrip);
        CHECK(cross_storage || resave_equal);
        stage = "fresh_after_restore";
        CHECK(ds4_session_eval(session, 987, err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(session, actual, vocab) == vocab);
        CHECK(finite_logits(actual, vocab));
        const bool fresh_equal = !memcmp(reference, actual, logits_bytes);
        CHECK(cross_storage || fresh_equal);
        const float *source_reference = source ? (const float *)scratch.ptr : reference;
        const bool source_restore_equal = !memcmp(source_reference, actual, logits_bytes);
        const double own_drift = max_error(reference, actual, vocab);
        const double source_drift = max_error(source_reference, actual, vocab);
        CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) == 0);
        fprintf(report, "{\"kind\":\"frontier\",\"position\":%d,\"f16\":%s,\"peer_import\":%s,\"cross_storage\":%s,\"payload_bytes\":%llu,"
            "\"own_source_wire_equal\":%s,\"own_source_fresh_equal\":%s,\"source_resave_wire_equal\":%s,"
            "\"restored_own_fresh_equal\":%s,\"restored_source_fresh_equal\":%s,\"own_fresh_max_abs\":%.9g,\"source_fresh_max_abs\":%.9g,\"fresh_logits_finite\":true}\n",
            tokens.len, boolean(!strcmp(f16, "1")), boolean(source != NULL), boolean(cross_storage),
            (unsigned long long)own.len, boolean(source_wire_equal), boolean(source_fresh_equal),
            boolean(resave_equal), boolean(fresh_equal), boolean(source_restore_equal), own_drift, source_drift);
        fflush(report);
        ds4_session_snapshot_free(&peer); ds4_session_snapshot_free(&roundtrip);
        ds4_session_snapshot_free(&scratch);
        if (i + 1 < sizeof(frontiers) / sizeof(*frontiers)) ds4_session_snapshot_free(&own);
    }
    /* A bad header or first token must not mutate the existing frontier. */
    stage = "invalid_header";
    own.ptr[0] ^= 1;
    CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) != 0);
    own.ptr[0] ^= 1;
    CHECK(ds4_session_pos(session) == 129 && ds4_session_checkpoint_valid(session));
    CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) == 0 && equal(&own, &roundtrip));
    ds4_session_snapshot_free(&roundtrip);
    stage = "invalid_token";
    const size_t token_start = DS4_SESSION_PAYLOAD_U32_FIELDS * 4u;
    uint8_t token_bytes[4];
    memcpy(token_bytes, own.ptr + token_start, 4);
    for (unsigned i = 0; i < 4; i++) own.ptr[token_start + i] = (uint8_t)((uint32_t)vocab >> (8 * i));
    CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) != 0);
    memcpy(own.ptr + token_start, token_bytes, 4);
    CHECK(ds4_session_pos(session) == 129 && ds4_session_checkpoint_valid(session));
    CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) == 0 && equal(&own, &roundtrip));
    ds4_session_snapshot_free(&roundtrip);
    /* Declare the full logical payload but physically stop just after logits.
     * Header/token validation succeeds; the later cache read fails after the
     * restore has started mutating host/device logits. */
    stage = "postmutation_short_read";
    n = snprintf(path, sizeof(path), "%s/truncated.dsv4", out);
    CHECK(n >= 0 && n < (int)sizeof(path));
    truncated = fopen(path, "w+b");
    CHECK(truncated);
    const size_t cut = token_start + 129u * 4u + logits_bytes + 4u;
    CHECK(cut < own.len && fwrite(own.ptr, 1, cut, truncated) == cut);
    rewind(truncated);
    CHECK(ds4_session_load_payload(session, truncated, own.len, err, sizeof(err)) != 0);
    CHECK(!ds4_session_checkpoint_valid(session) && ds4_session_payload_bytes(session) == 0);
    CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) != 0);
    CHECK(fclose(truncated) == 0); truncated = NULL;
    stage = "recover_after_failed_restore";
    CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) == 0);
    CHECK(ds4_session_eval(session, 987, err, sizeof(err)) == 0);
    CHECK(ds4_session_copy_logits(session, actual, vocab) == vocab);
    CHECK(finite_logits(actual, vocab) && !memcmp(reference, actual, logits_bytes));
    fprintf(report, "{\"kind\":\"failure_boundaries\",\"invalid_header_preserved\":true,\"invalid_token_preserved\":true,\"postmutation_failure_invalidated\":true,\"recovery_fresh_logits_equal\":true}\n");
    rc = 0;
done:
    fprintf(report, "{\"kind\":\"result\",\"pass\":%s,\"stage\":\"%s\"}\n", rc == 0 ? "true" : "false", stage);
    if (fclose(report) != 0) rc = 1;
    if (truncated) fclose(truncated);
    ds4_session_snapshot_free(&own); ds4_session_snapshot_free(&peer);
    ds4_session_snapshot_free(&roundtrip); ds4_session_snapshot_free(&scratch);
    ds4_tokens_free(&tokens); free(reference); free(actual);
    ds4_session_free(session); ds4_engine_close(engine);
    return rc;
}
