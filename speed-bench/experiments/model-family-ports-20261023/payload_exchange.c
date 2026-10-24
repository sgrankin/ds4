/* Adapted from ../zig-engine-payload-20261022/payload_roundtrip.c.
 * Public-API Qwen/GLM canonical payload exchange: one graph, five short frontiers.
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
    const char *model = NULL, *out = NULL, *source = NULL, *family = NULL;
    bool check_early_invalidation = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--source") && i + 1 < argc) source = argv[++i];
        else if (!strcmp(argv[i], "--family") && i + 1 < argc) family = argv[++i];
        else if (!strcmp(argv[i], "--check-early-invalidation")) check_early_invalidation = true;
        else return 2;
    }
    if (!model || !out || !family || (strcmp(family, "qwen") && strcmp(family, "glm"))) {
        fprintf(stderr, "usage: %s --family qwen|glm --model MODEL --out DIRECTORY [--source PEER_DIRECTORY] [--check-early-invalidation]\n", argv[0]);
        return 2;
    }
    const bool glm = !strcmp(family, "glm");
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
    if (glm) {
        CHECK(setenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP", "1", 1) == 0);
    } else {
        CHECK(setenv("DS4_METAL_NO_RESIDENCY", "1", 1) == 0);
        CHECK(setenv("DS4_METAL_NO_MODEL_WARMUP", "1", 1) == 0);
    }
    ds4_engine_options options = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 256, .prefill_chunk = glm ? 0 : 128, .mtp_draft_tokens = 1,
        .ssd_streaming = glm, .ssd_streaming_cache_bytes = glm ? UINT64_C(24) << 30 : 0,
        .power_percent = 100};
    stage = "engine";
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(glm ? ds4_engine_is_glm53(engine) : ds4_engine_is_qwen4(engine));
    fprintf(report, "{\"kind\":\"settings\",\"family\":\"%s\",\"context\":256,\"expert_cache_gib\":%u,\"peer_import\":%s}\n",
            family, glm ? 24u : 0u, boolean(source != NULL));
    int vocab = ds4_engine_vocab_size(engine);
    CHECK(vocab > 987);
    const size_t logits_bytes = (size_t)vocab * sizeof(float);
    reference = malloc(logits_bytes); actual = malloc(logits_bytes);
    CHECK(reference && actual);
    CHECK(ds4_session_create(&session, engine, 256) == 0);
    for (int i = 0; i < 129; i++) ds4_tokens_push(&tokens, 100 + i);
    /* Three/four/five cover GLM's incomplete/full/new indexer pool and
     * Qwen's short convolution histories. 129 spans a two-chunk extension. */
    const int frontiers[] = {1, 3, 4, 5, 129};
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
            CHECK(source_wire_equal);
            snprintf(name, sizeof(name), "frontier-%03d.fresh.f32", tokens.len);
            CHECK(read_bytes(source, name, &scratch));
            CHECK(scratch.len == logits_bytes && finite_logits((float *)scratch.ptr, vocab));
            source_fresh_equal = !memcmp(reference, scratch.ptr, logits_bytes);
            CHECK(source_fresh_equal);
        }
        const ds4_session_snapshot *saved = source ? &peer : &own;
        stage = "restore_and_resave";
        CHECK(ds4_session_load_snapshot(session, saved, err, sizeof(err)) == 0);
        CHECK(ds4_session_pos(session) == tokens.len && ds4_session_checkpoint_valid(session));
        CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) == 0);
        const bool resave_equal = equal(saved, &roundtrip);
        CHECK(resave_equal);
        stage = "fresh_after_restore";
        CHECK(ds4_session_eval(session, 987, err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(session, actual, vocab) == vocab);
        CHECK(finite_logits(actual, vocab));
        const bool fresh_equal = !memcmp(reference, actual, logits_bytes);
        CHECK(fresh_equal);
        const float *source_reference = source ? (const float *)scratch.ptr : reference;
        const bool source_restore_equal = !memcmp(source_reference, actual, logits_bytes);
        const double own_drift = max_error(reference, actual, vocab);
        const double source_drift = max_error(source_reference, actual, vocab);
        CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) == 0);
        fprintf(report, "{\"kind\":\"frontier\",\"position\":%d,\"family\":\"%s\",\"peer_import\":%s,\"payload_bytes\":%llu,"
            "\"own_source_wire_equal\":%s,\"own_source_fresh_equal\":%s,\"source_resave_wire_equal\":%s,"
            "\"restored_own_fresh_equal\":%s,\"restored_source_fresh_equal\":%s,\"own_fresh_max_abs\":%.9g,\"source_fresh_max_abs\":%.9g,\"fresh_logits_finite\":true}\n",
            tokens.len, family, boolean(source != NULL),
            (unsigned long long)own.len, boolean(source_wire_equal), boolean(source_fresh_equal),
            boolean(resave_equal), boolean(fresh_equal), boolean(source_restore_equal), own_drift, source_drift);
        fflush(report);
        ds4_session_snapshot_free(&peer); ds4_session_snapshot_free(&roundtrip);
        ds4_session_snapshot_free(&scratch);
        if (i + 1 < sizeof(frontiers) / sizeof(*frontiers)) ds4_session_snapshot_free(&own);
    }
    /* A bad header must not mutate the existing frontier. */
    stage = "invalid_header";
    own.ptr[0] ^= 1;
    CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) != 0);
    own.ptr[0] ^= 1;
    CHECK(ds4_session_pos(session) == 129 && ds4_session_checkpoint_valid(session));
    CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) == 0 && equal(&own, &roundtrip));
    ds4_session_snapshot_free(&roundtrip);
    /* A candidate-only early cut catches GLM's inherited live-logit mutation
     * before row-count validation. Keep the later device-span cut for every
     * arm, including frozen peers that predate the early-invalidation fix. */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS];
    memcpy(header, own.ptr, sizeof(header)); /* DSV4 and this Apple host are LE. */
    const size_t token_start = sizeof(header);
    const size_t logits_start = token_start + 129u * 4u;
    const size_t logits_end = logits_start + logits_bytes;
    const size_t row_counts = glm ? 2u * (size_t)header[8] * sizeof(uint32_t) : sizeof(uint32_t);
    const size_t cuts[] = {logits_end, logits_end + row_counts + 4u};
    const bool check_early = glm && check_early_invalidation;
    for (int cut_index = check_early ? 0 : 1; cut_index < 2; cut_index++) {
        stage = cut_index == 0 ? "postlogits_short_read" : "postmutation_short_read";
        n = snprintf(path, sizeof(path), "%s/truncated%s.dsv4", out,
                     cut_index == 0 ? "-logits" : "");
        CHECK(n >= 0 && n < (int)sizeof(path));
        truncated = fopen(path, "w+b");
        CHECK(truncated);
        const size_t cut = cuts[cut_index];
        CHECK(cut < own.len && fwrite(own.ptr, 1, cut, truncated) == cut);
        if (cut_index == 0) {
            float changed_logit;
            memcpy(&changed_logit, own.ptr + logits_start, sizeof(changed_logit));
            changed_logit += 1.0f;
            CHECK(fseek(truncated, (long)logits_start, SEEK_SET) == 0);
            CHECK(fwrite(&changed_logit, 1, sizeof(changed_logit), truncated) == sizeof(changed_logit));
        }
        rewind(truncated);
        CHECK(ds4_session_load_payload(session, truncated, own.len, err, sizeof(err)) != 0);
        CHECK(!ds4_session_checkpoint_valid(session) && ds4_session_payload_bytes(session) == 0);
        CHECK(ds4_session_argmax(session) == -1);
        CHECK(ds4_session_save_snapshot(session, &roundtrip, err, sizeof(err)) != 0);
        CHECK(fclose(truncated) == 0); truncated = NULL;
        stage = "recover_after_failed_restore";
        CHECK(ds4_session_load_snapshot(session, &own, err, sizeof(err)) == 0);
        CHECK(ds4_session_eval(session, 987, err, sizeof(err)) == 0);
        CHECK(ds4_session_copy_logits(session, actual, vocab) == vocab);
        CHECK(finite_logits(actual, vocab) && !memcmp(reference, actual, logits_bytes));
    }
    fprintf(report, "{\"kind\":\"failure_boundaries\",\"invalid_header_preserved\":true,\"early_logit_failure_invalidated\":%s,\"postmutation_failure_invalidated\":true,\"recovery_fresh_logits_equal\":true}\n",
            check_early ? "true" : "null");
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
