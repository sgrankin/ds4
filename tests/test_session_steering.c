/* Public-API hardware qualification of live steering ownership. Run one model
 * and its matching direction bank; this fixture never downloads weights. */
#include "../src/ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error_text[512];

static void require(int condition, const char *what) {
    if (!condition) {
        fprintf(stderr, "session steering: %s: %s\n", what, error_text);
        exit(1);
    }
}

static void copy_logits(ds4_session *session, float *out, int vocabulary) {
    require(ds4_session_checkpoint_valid(session), "valid checkpoint");
    require(ds4_session_copy_logits(session, out, vocabulary) == vocabulary,
            "copy all vocabulary logits");
    for (int i = 0; i < vocabulary; i++) require(isfinite(out[i]), "finite logits");
}

static double gap(const float *a, const float *b, int count) {
    double maximum = 0;
    for (int i = 0; i < count; i++) {
        const double delta = fabs((double)a[i] - b[i]);
        if (delta > maximum) maximum = delta;
    }
    return maximum;
}

static void set_scale(ds4_session *session, float value) {
    require(ds4_session_set_directional_steering_ffn(session, value) == 0,
            "set live FFN scale");
    require(ds4_session_directional_steering_ffn(session) == value,
            "get actual session scale");
}

static void sync_pair(ds4_session **sessions, const ds4_tokens *prompt) {
    for (int i = 0; i < 2; i++) {
        set_scale(sessions[i], 0);
        require(ds4_session_sync(sessions[i], prompt, error_text, sizeof(error_text)) == 0,
                "synchronize zero-scale prompt");
        require(ds4_session_pos(sessions[i]) == prompt->len, "prompt position");
    }
}

static void eval(ds4_session *session, int token) {
    const int before = ds4_session_pos(session);
    require(ds4_session_eval(session, token, error_text, sizeof(error_text)) == 0,
            "evaluate fixed token");
    require(ds4_session_pos(session) == before + 1, "one committed input");
}

static void check_batch(ds4_session **sessions, const ds4_tokens *prompt,
                        const int *tokens, float **reference, float *actual,
                        int vocabulary, int mixed) {
    sync_pair(sessions, prompt);
    set_scale(sessions[0], 1);
    set_scale(sessions[1], mixed ? 0 : 1);
    for (int step = 0; step < 2; step++) {
        eval(sessions[0], tokens[step]);
        eval(sessions[1], tokens[step]);
    }
    for (int i = 0; i < 2; i++) copy_logits(sessions[i], reference[i], vocabulary);
    sync_pair(sessions, prompt);
    set_scale(sessions[0], 1);
    set_scale(sessions[1], mixed ? 0 : 1);
    for (int step = 0; step < 2; step++) {
        ds4_decode_item items[2] = {{sessions[0], tokens[step]}, {sessions[1], tokens[step]}};
        require(ds4_sessions_eval_batch(items, 2, error_text, sizeof(error_text)) == 0,
                "homogeneous/mixed steering batch");
        for (int i = 0; i < 2; i++) {
            require(ds4_session_pos(sessions[i]) == prompt->len + step + 1,
                    "batch committed frontier");
            require(ds4_session_directional_steering_ffn(sessions[i]) == (i == 1 && mixed ? 0 : 1),
                    "batch retained independent scales");
            const ds4_tokens *history = ds4_session_tokens(sessions[i]);
            require(history && history->len == prompt->len + step + 1 &&
                    history->v[history->len - 1] == tokens[step],
                    "batch committed the requested input");
        }
    }
    for (int i = 0; i < 2; i++) {
        copy_logits(sessions[i], actual, vocabulary);
        /* Batched arithmetic may differ from scalar replay. Record its whole-
         * vocabulary gap; exact identity is required only by the scalar zero
         * and setter-state checks, not imposed on a different graph shape. */
        printf("batch %s member%d replay_max_delta=%.9g\n",
                mixed ? "mixed" : "homogeneous", i, gap(reference[i], actual, vocabulary));
    }
}

static void check_speculative(ds4_session **sessions, const ds4_tokens *prompt,
                              float *a, float *b, int vocabulary, int limit) {
    sync_pair(sessions, prompt);
    set_scale(sessions[0], 1);
    set_scale(sessions[1], 1);
    for (int cycle = 0; cycle < 3; cycle++) {
        const int token = ds4_session_argmax(sessions[0]);
        require(token >= 0, "speculative first token");
        int accepted[8] = {0};
        const int before = ds4_session_pos(sessions[0]);
        const int count = ds4_session_eval_speculative_argmax(sessions[0], token,
                limit, -1, accepted, 8, error_text, sizeof(error_text));
        require(count >= 1 && count <= limit && accepted[0] == token,
                "speculative committed inputs");
        require(ds4_session_pos(sessions[0]) == before + count,
                "speculative committed frontier");
        const ds4_tokens *committed = ds4_session_tokens(sessions[0]);
        require(committed && committed->len == before + count, "speculative token history length");
        for (int i = 0; i < count; i++) {
            require(accepted[i] >= 0 && accepted[i] < vocabulary, "accepted token range");
            require(committed->v[before + i] == accepted[i], "speculative published token history");
            eval(sessions[1], accepted[i]);
        }
        require(ds4_session_pos(sessions[0]) == ds4_session_pos(sessions[1]),
                "speculative replay position");
        require(ds4_session_directional_steering_ffn(sessions[0]) == 1 &&
                ds4_session_directional_steering_ffn(sessions[1]) == 1,
                "speculation retained scales");
        copy_logits(sessions[0], a, vocabulary);
        copy_logits(sessions[1], b, vocabulary);
        printf("speculative cycle%d committed=%d replay_max_delta=%.9g\n",
                cycle, count, gap(a, b, vocabulary));
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL DIRECTIONS [--mode project|add] [--prefill-chunk N] [--ssd] [--cache-gb N] [--batch] [--mtp|--packet]\n", argv[0]);
        return 2;
    }
    int batch = 0, mtp = 0;
    ds4_engine_options options = {
        .model_path = argv[1], .directional_steering_file = argv[2],
        .directional_steering_attn = 0, .directional_steering_ffn = 0,
        .backend = DS4_BACKEND_METAL, .context_size = 256, .prefill_chunk = 64,
        .power_percent = 100, .placement_session_count_hint = 2,
        .share_session_prefill_workspace = true,
    };
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--ssd")) options.ssd_streaming = true;
        else if (!strcmp(argv[i], "--batch")) batch = 1;
        else if (!strcmp(argv[i], "--mtp")) options.glm_mtp = mtp = 1;
        else if (!strcmp(argv[i], "--packet")) mtp = 1;
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *mode = argv[++i];
            if (!strcmp(mode, "project")) options.directional_steering_mode = DS4_STEERING_PROJECT;
            else if (!strcmp(mode, "add")) options.directional_steering_mode = DS4_STEERING_ADD;
            else { fprintf(stderr, "unknown mode: %s\n", mode); return 2; }
        }
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc)
            options.ssd_streaming_cache_bytes = strtoull(argv[++i], NULL, 10) << 30;
        else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc)
            options.prefill_chunk = (uint32_t)strtoul(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    ds4_engine *engine = NULL;
    require(ds4_engine_open(&engine, &options) == 0, "open model with zero-preloaded bank");
    const int vocabulary = ds4_engine_vocab_size(engine);
    require(vocabulary > 0, "vocabulary");
    const size_t bytes = (size_t)vocabulary * sizeof(float);
    float *a = malloc(bytes), *b = malloc(bytes), *saved = malloc(bytes);
    require(a && b && saved, "host logit buffers");
    ds4_session *sessions[2] = {NULL, NULL};
    for (int i = 0; i < 2; i++) {
        require(ds4_session_create(&sessions[i], engine, 256) == 0, "create session");
        require(ds4_session_directional_steering_ffn(sessions[i]) == 0, "initial zero scale");
    }
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to one hundred, writing every number.", DS4_THINK_NONE, &prompt);
    require(prompt.len > 0 && prompt.len < 224, "bounded prompt");
    sync_pair(sessions, &prompt);
    copy_logits(sessions[0], a, vocabulary);
    copy_logits(sessions[1], b, vocabulary);
    require(!memcmp(a, b, bytes), "zero prefill exact FP32 identity");
    int tokens[3];
    tokens[0] = ds4_session_argmax(sessions[1]);
    require(tokens[0] >= 0, "fixed decode token");
    eval(sessions[0], tokens[0]);
    eval(sessions[1], tokens[0]);
    copy_logits(sessions[0], a, vocabulary);
    copy_logits(sessions[1], b, vocabulary);
    require(!memcmp(a, b, bytes), "zero decode exact FP32 identity");
    puts("zero prefill/decode: all FP32 logits identical");

    const int position = ds4_session_pos(sessions[0]);
    set_scale(sessions[0], 1);
    require(ds4_session_directional_steering_ffn(sessions[1]) == 0, "independent session scale");
    copy_logits(sessions[0], saved, vocabulary);
    require(!memcmp(a, saved, bytes) && ds4_session_pos(sessions[0]) == position,
            "scale change leaves existing frontier/logits intact");
    const float invalid[] = {NAN, INFINITY, -INFINITY, 101, -101};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        require(ds4_session_set_directional_steering_ffn(sessions[0], invalid[i]) != 0,
                "reject invalid scale");
        require(ds4_session_directional_steering_ffn(sessions[0]) == 1 &&
                ds4_session_directional_steering_ffn(sessions[1]) == 0 &&
                ds4_session_pos(sessions[0]) == position,
                "invalid setter preserves ownership/frontier");
        copy_logits(sessions[0], a, vocabulary);
        require(!memcmp(a, saved, bytes), "invalid setter preserves logits");
    }
    /* A fresh session inherits the engine's most recently published default,
     * while the already-existing zero-scale session stayed independent. Reuse
     * its slot so this test never has more than two GPU sessions alive. */
    ds4_session_free(sessions[1]);
    sessions[1] = NULL;
    require(ds4_session_create(&sessions[1], engine, 256) == 0, "create session after live change");
    require(ds4_session_directional_steering_ffn(sessions[1]) == 1,
            "new session inherits published default");
    set_scale(sessions[1], 0);
    require(ds4_session_sync(sessions[1], &prompt, error_text, sizeof(error_text)) == 0,
            "replay independent zero-scale control");
    eval(sessions[1], tokens[0]);
    copy_logits(sessions[1], b, vocabulary);
    require(!memcmp(b, saved, bytes), "independent zero-scale replay identity");
    require(ds4_session_directional_steering_ffn(sessions[0]) == 1,
            "new-session defaults do not mutate existing scale");
    tokens[1] = ds4_session_argmax(sessions[1]);
    require(tokens[1] >= 0, "next fixed decode token");
    eval(sessions[0], tokens[1]);
    eval(sessions[1], tokens[1]);
    copy_logits(sessions[0], a, vocabulary);
    copy_logits(sessions[1], b, vocabulary);
    require(memcmp(a, b, bytes) != 0, "nonzero steering changes actual decode");
    printf("live nonzero decode: max_delta=%.9g\n", gap(a, b, vocabulary));
    const int steered_position = ds4_session_pos(sessions[0]);
    set_scale(sessions[0], 0);
    copy_logits(sessions[0], saved, vocabulary);
    require(!memcmp(a, saved, bytes) && ds4_session_pos(sessions[0]) == steered_position,
            "restoring zero retains the existing steered state");
    const ds4_tokens *retained = ds4_session_tokens(sessions[0]);
    require(retained && retained->len == prompt.len + 2 &&
            ds4_tokens_starts_with(retained, &prompt) &&
            retained->v[prompt.len] == tokens[0] && retained->v[prompt.len + 1] == tokens[1],
            "scale restoration retains exact committed token history");
    tokens[2] = ds4_session_argmax(sessions[1]);
    require(tokens[2] >= 0, "post-restore fixed token");
    eval(sessions[0], tokens[2]);
    eval(sessions[1], tokens[2]);
    copy_logits(sessions[0], a, vocabulary);
    copy_logits(sessions[1], b, vocabulary);
    printf("restored-zero continued decode: historical_state_max_delta=%.9g\n", gap(a, b, vocabulary));
    puts("live ownership, invalid scales and future-only publication: OK");

    if (batch) {
        float *reference[2] = {a, b};
        check_batch(sessions, &prompt, tokens, reference, saved, vocabulary, 0);
        check_batch(sessions, &prompt, tokens, reference, saved, vocabulary, 1);
    }
    if (mtp) {
        int limit = ds4_engine_decode_packet_size(engine);
        if (limit < 1) limit = 1;
        if (limit > 8) limit = 8;
        check_speculative(sessions, &prompt, a, b, vocabulary, limit);
    }
    ds4_tokens_free(&prompt);
    ds4_session_free(sessions[1]);
    ds4_session_free(sessions[0]);
    ds4_engine_close(engine);
    free(saved); free(b); free(a);
    puts("session steering: PASS");
    return 0;
}
