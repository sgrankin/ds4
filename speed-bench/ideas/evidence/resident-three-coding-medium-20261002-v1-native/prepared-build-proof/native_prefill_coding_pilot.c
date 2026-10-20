#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void json_string(FILE *fp, const char *text) {
    fputc('"', fp);
    for (; *text; text++) {
        unsigned char c = (unsigned char)*text;
        if (c == '"' || c == '\\') { fputc('\\', fp); fputc(c, fp); }
        else if (c < 32) fprintf(fp, "\\u%04x", c);
        else fputc(c, fp);
    }
    fputc('"', fp);
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) || ftell(fp) < 0) { fclose(fp); return NULL; }
    long size = ftell(fp);
    if (size > 65536 || fseek(fp, 0, SEEK_SET)) { fclose(fp); return NULL; }
    char *text = malloc((size_t)size + 1);
    if (!text) { fclose(fp); return NULL; }
    if (fread(text, 1, (size_t)size, fp) != (size_t)size) { free(text); fclose(fp); return NULL; }
    text[size] = '\0'; fclose(fp); return text;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fputs("usage: native_prefill_coding_pilot MODEL FRESH_OUTPUT_DIR PROMPT_DIR [CACHE_GIB=64] [CAP=512]\n", stderr);
        return 2;
    }
    char *end = NULL;
    double cache = argc > 4 ? strtod(argv[4], &end) : 64.0;
    if ((argc > 4 && (end == argv[4] || *end)) || !isfinite(cache) || cache <= 0 || cache > 256) return 2;
    long cap_long = argc > 5 ? strtol(argv[5], &end, 10) : 512;
    if ((argc > 5 && (end == argv[5] || *end)) || cap_long < 1 || cap_long > 2048) return 2;
    int cap = (int)cap_long;
    if (strlen(argv[2]) > 3800 || strlen(argv[3]) > 3800) return 2;
    char path[4096];
    snprintf(path, sizeof(path), "%s/tasks.txt", argv[3]);
    FILE *tasks = fopen(path, "rb");
    if (!tasks) { perror(path); return 2; }
    /* Validate every prompt before opening the model. */
    char ids[16][128], *prompts[16] = {0};
    int count = 0;
    while (count < 16 && fgets(ids[count], sizeof(ids[count]), tasks)) {
        ids[count][strcspn(ids[count], "\r\n")] = '\0';
        if (!ids[count][0] || strspn(ids[count], "abcdefghijklmnopqrstuvwxyz0123456789_") != strlen(ids[count])) return 2;
        for (int prior = 0; prior < count; prior++) if (!strcmp(ids[prior], ids[count])) return 2;
        snprintf(path, sizeof(path), "%s/%s.txt", argv[3], ids[count]);
        prompts[count] = read_text(path);
        if (!prompts[count] || !prompts[count][0]) { fprintf(stderr, "invalid prompt %s\n", path); return 2; }
        count++;
    }
    if (!count || !feof(tasks)) return 2;
    fclose(tasks);
    if (mkdir(argv[2], 0755)) { perror("fresh output directory"); return 2; }
    snprintf(path, sizeof(path), "%s/generation.jsonl", argv[2]);
    FILE *results = fopen(path, "wx");
    if (!results) { perror(path); return 2; }
    char *fixed[3] = {0};
    const char *fixed_names[3] = {"warm-user.txt", "warm-assistant.txt", "preamble.txt"};
    for (int fi = 0; fi < 3; fi++) {
        snprintf(path, sizeof(path), "%s/%s", argv[3], fixed_names[fi]);
        fixed[fi] = read_text(path);
        if (!fixed[fi] || !fixed[fi][0]) { fprintf(stderr, "invalid fixed assembly file %s\n", path); return 2; }
    }
    /* These two live prefill knobs are restored only after each native warm. */
    const char *tile_env = getenv("DS4_METAL_V41_SELECTED_TILE");
    const char *limit_env = getenv("DS4_METAL_V41_SELECTED_LIMIT");
    char *append_tile = tile_env ? strdup(tile_env) : NULL;
    char *append_limit = limit_env ? strdup(limit_env) : NULL;
    if ((tile_env && !append_tile) || (limit_env && !append_limit)) return 1;
    ds4_engine_options opt = {0};
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_METAL;
    opt.n_threads = 2; opt.context_size = 4096; opt.prefill_chunk = 512;
    opt.power_percent = 100; opt.ssd_streaming = true;
    opt.ssd_streaming_cache_bytes = (uint64_t)(cache * 1073741824.0 + 0.5);
    opt.mtp_draft_tokens = 1;
    fprintf(stderr, "coding pilot tasks=%d cap=%d cache=%.6g GiB ctx4096/chunk512/no-thinking/greedy output=%s\n", count, cap, cache, argv[2]);
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    if (ds4_engine_open(&engine, &opt) || ds4_session_create(&session, engine, opt.context_size)) return 1;
    if (strcmp(ds4_engine_model_name(engine), "DeepSeek V4.1 Flash")) {
        fprintf(stderr, "unexpected model: %s\n", ds4_engine_model_name(engine)); return 1;
    }
    if (ds4_session_prefill_cap(session) < 64) { fputs("INVALID prefill capacity below selected tile64\n", stderr); return 2; }
    ds4_session_gpu_warmup(session);
    char error[512] = {0};
    const int eos = ds4_token_eos(engine);
    /* Render and validate the entire append fixture before any task is run. */
    ds4_tokens warm = {0}, rendered[16] = {{0}};
    ds4_chat_begin(engine, &warm);
    ds4_chat_append_think_prefix(engine, &warm, DS4_THINK_NONE);
    ds4_chat_append_message(engine, &warm, "user", fixed[0]);
    ds4_chat_append_message(engine, &warm, "assistant", fixed[1]);
    if (warm.len <= 0) return 2;
    for (int ti = 0; ti < count; ti++) {
        size_t a = strlen(fixed[2]), b = strlen(prompts[ti]);
        if (a + b > 65536) return 2;
        char *content = malloc(a + b + 1);
        if (!content) return 1;
        memcpy(content, fixed[2], a); memcpy(content + a, prompts[ti], b + 1);
        ds4_tokens_copy(&rendered[ti], &warm);
        ds4_chat_append_message(engine, &rendered[ti], "user", content);
        ds4_chat_append_assistant_prefix(engine, &rendered[ti], DS4_THINK_NONE);
        free(content);
        int append = rendered[ti].len - warm.len;
        if (append < 256 || append >= 1024 || rendered[ti].len + cap > opt.context_size ||
            memcmp(rendered[ti].v, warm.v, (size_t)warm.len * sizeof(int))) {
            fprintf(stderr, "INVALID medium append %s warm=%d append=%d context=%d cap=%d\n",
                    ids[ti], warm.len, append, rendered[ti].len, cap);
            return 2;
        }
    }
    for (int ti = 0; ti < count; ti++) {
        if (unsetenv("DS4_METAL_V41_SELECTED_TILE") || unsetenv("DS4_METAL_V41_SELECTED_LIMIT")) return 1;
        ds4_session_invalidate(session);
        fprintf(stderr, "prefill_policy warm task=%s pos=%d tile=%s limit=%s\n", ids[ti], ds4_session_pos(session),
                getenv("DS4_METAL_V41_SELECTED_TILE") ? getenv("DS4_METAL_V41_SELECTED_TILE") : "default",
                getenv("DS4_METAL_V41_SELECTED_LIMIT") ? getenv("DS4_METAL_V41_SELECTED_LIMIT") : "default");
        double warm_start = seconds();
        if (ds4_session_sync(session, &warm, error, sizeof(error))) {
            fprintf(stderr, "warm %s: %s\n", ids[ti], error); return 1;
        }
        double warm_seconds = seconds() - warm_start;
        int warm_position = ds4_session_pos(session);
        if (warm_position != warm.len || ds4_session_common_prefix(session, &rendered[ti]) != warm.len) {
            fprintf(stderr, "INVALID existing warm context %s pos=%d len=%d\n", ids[ti], warm_position, warm.len);
            return 2;
        }
        int warm_vocab = ds4_engine_vocab_size(engine);
        float *warm_logits = malloc((size_t)warm_vocab * sizeof(float));
        if (!warm_logits || ds4_session_copy_logits(session, warm_logits, warm_vocab) != warm_vocab) return 1;
        snprintf(path, sizeof(path), "%s/%s.warm-logits.bin", argv[2], ids[ti]);
        FILE *warm_record = fopen(path, "wx");
        if (!warm_record || fwrite(warm_logits, sizeof(float), (size_t)warm_vocab, warm_record) != (size_t)warm_vocab) return 1;
        fclose(warm_record); free(warm_logits);
        if ((append_tile && setenv("DS4_METAL_V41_SELECTED_TILE", append_tile, 1)) ||
            (append_limit && setenv("DS4_METAL_V41_SELECTED_LIMIT", append_limit, 1))) return 1;
        fprintf(stderr, "prefill_policy append task=%s pos=%d count=%d tile=%s limit=%s\n", ids[ti], ds4_session_pos(session), rendered[ti].len - warm.len,
                getenv("DS4_METAL_V41_SELECTED_TILE") ? getenv("DS4_METAL_V41_SELECTED_TILE") : "default",
                getenv("DS4_METAL_V41_SELECTED_LIMIT") ? getenv("DS4_METAL_V41_SELECTED_LIMIT") : "default");
        double prefill_start = seconds();
        if (ds4_session_sync(session, &rendered[ti], error, sizeof(error))) {
            fprintf(stderr, "sync %s: %s\n", ids[ti], error); return 1;
        }
        double prefill_seconds = seconds() - prefill_start;
        if (ds4_session_pos(session) != rendered[ti].len) return 2;
        snprintf(path, sizeof(path), "%s/%s.prompt.json", argv[2], ids[ti]);
        FILE *prompt_record = fopen(path, "wx");
        if (!prompt_record) return 1;
        fprintf(prompt_record, "{\"warm_tokens\":%d,\"append_tokens\":%d,\"warm_position\":%d,\"prefix_exact\":true,\"ids\":[",
                warm.len, rendered[ti].len - warm.len, warm_position);
        for (int pi = 0; pi < rendered[ti].len; pi++) fprintf(prompt_record, "%s%d", pi ? "," : "", rendered[ti].v[pi]);
        fputs("]}\n", prompt_record); fclose(prompt_record);
        int vocab = ds4_engine_vocab_size(engine);
        float *logits = malloc((size_t)vocab * sizeof(float));
        if (!logits || ds4_session_copy_logits(session, logits, vocab) != vocab) return 1;
        snprintf(path, sizeof(path), "%s/%s.initial-logits.bin", argv[2], ids[ti]);
        FILE *logit_record = fopen(path, "wx");
        if (!logit_record || fwrite(logits, sizeof(float), (size_t)vocab, logit_record) != (size_t)vocab) return 1;
        fclose(logit_record); free(logits);
        snprintf(path, sizeof(path), "%s/%s.txt", argv[2], ids[ti]);
        FILE *raw = fopen(path, "wx");
        snprintf(path, sizeof(path), "%s/%s.tokens.jsonl", argv[2], ids[ti]);
        FILE *trace = fopen(path, "wx");
        if (!raw || !trace) return 1;
        uint64_t rng = UINT64_C(20261002);
        int generated = 0, decisions = 0, stop_id = -1;
        bool stopped = false;
        double decode_start = seconds();
        while (generated < cap) {
            int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
            if (token < 0 || token >= ds4_engine_vocab_size(engine)) { fputs("invalid sampled token\n", stderr); return 1; }
            bool stop = ds4_token_is_stop_for_think_mode(engine, token, DS4_THINK_NONE);
            size_t bytes = 0;
            char *piece = ds4_token_text(engine, token, &bytes);
            if (!piece) return 1;
            fprintf(trace, "{\"step\":%d,\"token_id\":%d,\"stop\":%s,\"eos\":%s,\"piece_hex\":\"", decisions++, token, stop ? "true" : "false", token == eos ? "true" : "false");
            for (size_t bi = 0; bi < bytes; bi++) fprintf(trace, "%02x", (unsigned char)piece[bi]);
            fputs("\"}\n", trace);
            if (stop) { stopped = true; stop_id = token; free(piece); break; }
            if (fwrite(piece, 1, bytes, raw) != bytes) return 1;
            free(piece); generated++;
            if (generated < cap && ds4_session_eval(session, token, error, sizeof(error))) {
                fprintf(stderr, "eval %s: %s\n", ids[ti], error); return 1;
            }
        }
        double decode_seconds = seconds() - decode_start;
        fclose(raw); fclose(trace);
        fputs("{\"task\":", results); json_string(results, ids[ti]);
        fprintf(results, ",\"generated_tokens\":%d,\"sampling_decisions\":%d,\"stopped\":%s,\"eos\":%s,\"stop_token_id\":%d,\"cap_hit\":%s,\"max_tokens\":%d,\"cache_gib\":%.9g,\"context\":4096,\"prefill_chunk\":512,\"think_mode\":\"none\",\"temperature\":0,\"seed\":20261002,\"warm_tokens\":%d,\"append_tokens\":%d,\"warm_position\":%d,\"prefix_exact\":true,\"vocab\":%d,\"actual_prefill_cap\":%d,\"warm_seconds\":%.6f,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,\"raw_trace\":\"%s.txt\",\"token_trace\":\"%s.tokens.jsonl\"}\n", generated, decisions, stopped ? "true" : "false", stop_id == eos ? "true" : "false", stop_id, !stopped && generated >= cap ? "true" : "false", cap, cache, warm.len, rendered[ti].len - warm.len, warm_position, vocab, ds4_session_prefill_cap(session), warm_seconds, prefill_seconds, decode_seconds, ids[ti], ids[ti]);
        fflush(results);
        fprintf(stderr, "coding %s tokens=%d stopped=%d eos=%d cap=%d\n", ids[ti], generated, stopped, stop_id == eos, !stopped && generated >= cap);
    }
    fclose(results);
    ds4_session_free(session); ds4_engine_close(engine);
    for (int i = 0; i < count; i++) { free(prompts[i]); ds4_tokens_free(&rendered[i]); }
    ds4_tokens_free(&warm);
    for (int i = 0; i < 3; i++) free(fixed[i]);
    free(append_tile); free(append_limit);
    return 0;
}
