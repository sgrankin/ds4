/* Real-model qualification of the direct agent compaction path. This is a
 * fixed synthetic conversation, not a whole-task speed benchmark. */
#define DS4_AGENT_TEST_NO_MAIN
#include "../../../ds4_agent.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "compaction probe line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

static void append_review(ds4_engine *engine, ds4_tokens *out, int target,
                          const char *topic) {
    static const char *files[] = {"router.c", "cache.c", "session.c", "api.c"};
    agent_buf text = {0}; ds4_tokens message = {0};
    for (int i = 0; i < 128; i++) {
        char line[256];
        snprintf(line, sizeof(line),
            "%s review %03d: inspected %s checkpoint %d. Ownership is clear, state is bounded, and the next review will check the saved request boundary.\n",
            topic, i, files[i % 4], (i * 17) % 97);
        agent_buf_puts(&text, line);
        ds4_tokens_free(&message);
        ds4_chat_append_message(engine, &message, "assistant", text.ptr);
        if (message.len >= target) break;
    }
    agent_tokens_append_range(out, &message, 0, message.len);
    ds4_tokens_free(&message); free(text.ptr);
}

static bool save_text(const char *dir, const char *name, const char *text) {
    char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool ok = fwrite(text, 1, strlen(text), fp) == strlen(text);
    return fclose(fp) == 0 && ok;
}

static bool save_tokens(agent_worker *w, const char *dir, const char *name) {
    size_t len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript, &len);
    bool ok = text && save_text(dir, name, text);
    free(text); return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL OUTPUT_DIRECTORY\n", argv[0]); return 2; }
    int rc = 1, before = 0, compacted = 0, source_end = 0, retained_start = 0;
    bool mutex_ready = false, stopped = false, reused = false;
    double compact_seconds = 0, answer_seconds = 0;
    char err[256] = {0}, trace_path[PATH_MAX], cache_path[PATH_MAX];
    agent_config cfg = {.non_interactive = true,
        .gen = {.ctx_size = 8192, .think_mode = DS4_THINK_NONE,
                .system = "Qualification conversation: keep the release code and concrete task state. A compaction summary should be brief. Do not call tools when asked for a release code."},
        .engine = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
                   .context_size = 8192, .prefill_chunk = 512,
                   .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
                   .power_percent = 100}};
    agent_worker w = {.cfg = &cfg, .wake_fd = {-1, -1}, .initialized = true,
                      .user_activity = true};
    ds4_tokens sys = {0}, question = {0}; agent_buf answer = {0};
    CHECK(agent_mkdir_p(argv[2]));
    snprintf(trace_path, sizeof(trace_path), "%s/compaction.trace", argv[2]);
    snprintf(cache_path, sizeof(cache_path), "%s/state-cache", argv[2]);
    CHECK(setenv("DS4_STATE_CACHE_DIR", cache_path, 1) == 0);
    CHECK(setenv("DS4_STATE_CACHE_HOT_MB", "0", 1) == 0);
    CHECK(setenv("DS4_STATE_CACHE_SSD_MB", "4096", 1) == 0);
    CHECK(pthread_mutex_init(&w.mu, NULL) == 0); mutex_ready = true;
    w.trace = fopen(trace_path, "wb"); CHECK(w.trace);
    CHECK(ds4_engine_open(&w.engine, &cfg.engine) == 0);
    CHECK(ds4_engine_is_deepseek41(w.engine));
    CHECK(ds4_session_create(&w.session, w.engine, cfg.gen.ctx_size) == 0);
    CHECK(agent_context_prerope_requested());
    CHECK(ds4_session_enable_suffix_prerope(w.session, err, sizeof(err)) == 0);
    w.state_cache = ds4_state_cache_runtime_open(w.engine, &cfg.engine,
        cfg.gen.ctx_size, w.state_cache_profile, err, sizeof(err));
    CHECK(w.state_cache);
    agent_worker_build_system_tokens(&w, &sys);
    ds4_tokens_copy(&w.transcript, &sys);
    ds4_chat_append_message(w.engine, &w.transcript, "user",
        "We are reviewing a persistent local agent. Earlier planning and repeated review notes can be summarized. Preserve only concrete decisions, requirements, and unresolved work.");
    append_review(w.engine, &w.transcript, 768, "Earlier planning");
    retained_start = w.transcript.len;
    ds4_chat_append_message(w.engine, &w.transcript, "user",
        "Current task: prepare the Ember release. The authoritative release code is EMBER-7319. It is a literal identifier, not a number to calculate. Preserve that code and this complete turn. No code changes or external actions are required in this qualification.");
    append_review(w.engine, &w.transcript, 640, "Recent checkpoint");
    source_end = w.transcript.len;
    ds4_chat_append_message(w.engine, &w.transcript, "user",
        "Keep the current task state while shortening earlier planning. The saved review files are reloadable, and their repetitive status messages do not need to remain verbatim. Retain the earlier complete user turn containing the authoritative release identifier, together with enough recent review context to understand the task.\n"
        "This is a local inference qualification only. Do not call tools, run commands, create files, contact people, or invent additional release requirements. We will ask about the release identifier after compaction. For now the pending action is simply to preserve the exact identifier and the fact that there are no required code changes.\n"
        "A short durable summary should say what the user requested, which decisions are settled, and what remains pending. Routine ownership and checkpoint review notes can be summarized collectively. If a file would be needed later, its path can be retained for checking; no file content must be copied into this conversation. Preserve this entire latest message freshly when rebuilding the context.");
    before = w.transcript.len;
    CHECK(before - source_end >= 128);
    CHECK(agent_compact_tail_start(&w, before, sys.len) == retained_start);
    CHECK(before + agent_compact_reserve_tokens(&w) < cfg.gen.ctx_size);
    ds4_tokens endpoint = w.transcript; endpoint.len = source_end;
    CHECK(agent_worker_sync_tokens(&w, &endpoint, false, err, sizeof(err)) == 0);
    CHECK(agent_worker_capture_context_frontier(&w));
    CHECK(agent_worker_sync_tokens(&w, &w.transcript, false, err, sizeof(err)) == 0);
    CHECK(save_tokens(&w, argv[2], "before-context.txt"));
    double t0 = now_sec();
    CHECK(agent_worker_compact_transcript(&w, "qualification of cached retained turns", NULL, err, sizeof(err)));
    compact_seconds = now_sec() - t0;
    compacted = w.transcript.len;
    reused = ds4_session_is_approximate(w.session);
    CHECK(agent_tokens_equal(ds4_session_tokens(w.session), &w.transcript));
    CHECK(save_tokens(&w, argv[2], "after-context.txt"));
    CHECK(save_text(argv[2], "compaction-output.txt", w.out ? w.out : ""));
    ds4_tokens_copy(&question, &w.transcript);
    ds4_chat_append_message(w.engine, &question, "user",
        "What is the authoritative Ember release code? Reply with only that exact identifier. Do not call tools.");
    ds4_chat_append_assistant_prefix(w.engine, &question, DS4_THINK_NONE);
    t0 = now_sec();
    CHECK(ds4_session_sync(w.session, &question, err, sizeof(err)) == 0);
    for (int i = 0; i < 64; i++) {
        int token = ds4_session_argmax(w.session);
        if (ds4_token_is_stop_for_think_mode(w.engine, token, DS4_THINK_NONE)) { stopped = true; break; }
        CHECK(ds4_session_eval(w.session, token, err, sizeof(err)) == 0);
        size_t len = 0; char *piece = ds4_token_text(w.engine, token, &len);
        CHECK(piece); agent_buf_append(&answer, piece, len); free(piece);
    }
    answer_seconds = now_sec() - t0;
    CHECK(save_text(argv[2], "answer.txt", answer.ptr ? answer.ptr : ""));
    char result[1024];
    snprintf(result, sizeof(result),
        "{\"system_tokens\":%d,\"before_tokens\":%d,\"source_end\":%d,\"retained_start\":%d,\"after_tokens\":%d,\"suffix_reused\":%s,\"compact_seconds\":%.9f,\"answer_seconds\":%.9f,\"answer_stopped\":%s,\"fact_correct\":%s}\n",
        sys.len, before, source_end, retained_start, compacted, reused ? "true" : "false",
        compact_seconds, answer_seconds, stopped ? "true" : "false",
        answer.ptr && strstr(answer.ptr, "EMBER-7319") ? "true" : "false");
    CHECK(save_text(argv[2], "result.json", result));
    fputs(result, stdout); rc = 0;
done:
    if (w.trace) fclose(w.trace);
    agent_context_frontiers_clear(&w);
    if (w.state_cache) ds4_state_cache_close(w.state_cache, NULL, 0);
    ds4_tokens_free(&sys); ds4_tokens_free(&question); ds4_tokens_free(&w.transcript);
    free(answer.ptr); free(w.out);
    ds4_session_free(w.session); ds4_engine_close(w.engine);
    if (mutex_ready) pthread_mutex_destroy(&w.mu);
    return rc;
}
