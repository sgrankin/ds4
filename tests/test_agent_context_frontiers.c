/* Exact recent-frontier selection, eviction, state identity, and fallback.
 * Native guarded regression passed after recovery from a host restart. */
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "context-frontiers line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

static void append_range(ds4_tokens *out, const ds4_tokens *src, int start, int count) {
    for (int i = 0; i < count; i++) ds4_tokens_push(out, src->v[start + i]);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    int rc = 1;
    char err[256] = {0}, *text = NULL;
    size_t text_len = 0;
    agent_config cfg = {.non_interactive = true,
        .gen = {.ctx_size = 8192, .think_mode = DS4_THINK_HIGH,
                .live_context_path = "/tmp/ds4-context-frontier-test.txt"},
        .engine = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
                   .context_size = 8192, .prefill_chunk = 2048,
                   .ssd_streaming = true,
                   .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
                   .power_percent = 100}};
    agent_worker w = {.cfg = &cfg, .wake_fd = {-1, -1}};
    ds4_tokens source = {0}, first = {0}, prefix = {0}, old = {0}, edited = {0}, wrong = {0};
    ds4_session_snapshot frontier = {0}, expected = {0}, actual = {0};
    CHECK(ds4_engine_open(&w.engine, &cfg.engine) == 0);
    CHECK(ds4_session_create(&w.session, w.engine, cfg.gen.ctx_size) == 0);
    CHECK(agent_read_file_bytes("speed-bench/promessi_sposi.txt", &text, &text_len,
                                err, sizeof(err)) == 0);
    ds4_tokenize_text(w.engine, text, &source);
    CHECK(source.len > 3328);
    append_range(&first, &source, 0, 2048);
    append_range(&prefix, &source, 0, 2304);
    append_range(&old, &source, 0, 2816);
    append_range(&edited, &source, 0, 2304);
    int new_start = 2816;
    while (new_start + 512 <= source.len && source.v[new_start] == old.v[2304]) new_start++;
    CHECK(new_start + 512 <= source.len);
    append_range(&edited, &source, new_start, 512);
    CHECK(edited.v[2304] != old.v[2304]);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_FRONTIERS") == 0);
    CHECK(!agent_worker_restore_context_frontier(&w, &edited));
    CHECK(setenv("DS4_AGENT_CONTEXT_FRONTIERS", "2", 1) == 0);
    CHECK(setenv("DS4_AGENT_CONTEXT_FRONTIER_MB", "64", 1) == 0);
    ds4_tokens_copy(&w.transcript, &first);
    CHECK(ds4_session_sync(w.session, &w.transcript, err, sizeof(err)) == 0);
    CHECK(agent_worker_capture_context_frontier(&w));
    ds4_tokens_copy(&w.transcript, &prefix);
    CHECK(ds4_session_sync(w.session, &w.transcript, err, sizeof(err)) == 0);
    CHECK(agent_worker_capture_context_frontier(&w));
    CHECK(ds4_session_save_snapshot(w.session, &frontier, err, sizeof(err)) == 0);
    ds4_tokens_copy(&w.transcript, &old);
    CHECK(ds4_session_sync(w.session, &w.transcript, err, sizeof(err)) == 0);
    CHECK(agent_worker_capture_context_frontier(&w));
    CHECK(w.context_frontier_count == 2 && w.context_frontiers[0].prefix.len == 2304);
    CHECK(w.context_frontier_bytes <= UINT64_C(64) * 1024 * 1024);

    ds4_session_invalidate(w.session);
    CHECK(ds4_session_sync(w.session, &prefix, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(w.session, &edited, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(w.session, &expected, err, sizeof(err)) == 0);
    double t0 = now_sec();
    CHECK(agent_worker_restore_context_frontier(&w, &edited));
    double restore_ms = (now_sec() - t0) * 1000;
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == frontier.len && !memcmp(actual.ptr, frontier.ptr, frontier.len));
    t0 = now_sec();
    CHECK(ds4_session_sync(w.session, &edited, err, sizeof(err)) == 0);
    double suffix_ms = (now_sec() - t0) * 1000;
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == expected.len && !memcmp(actual.ptr, expected.ptr, expected.len));

    ds4_tokens_copy(&wrong, &edited); wrong.v[0] = wrong.v[0] == 1 ? 2 : 1;
    CHECK(!agent_worker_restore_context_frontier(&w, &wrong));
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == expected.len && !memcmp(actual.ptr, expected.ptr, expected.len));
    w.context_frontiers[1].ctx++;
    CHECK(!agent_worker_restore_context_frontier(&w, &wrong));
    CHECK(w.context_frontier_count == 1);
    w.context_frontiers[0].cache.state.ptr[0] ^= 1;
    CHECK(!agent_worker_restore_context_frontier(&w, &edited));
    CHECK(w.context_frontier_count == 0);
    CHECK(ds4_session_sync(w.session, &edited, err, sizeof(err)) == 0);
    CHECK(agent_tokens_equal(ds4_session_tokens(w.session), &edited));
    ds4_tokens_copy(&w.transcript, &edited);
    CHECK(agent_worker_capture_context_frontier(&w));
    w.image_count = 1;
    CHECK(!agent_worker_capture_context_frontier(&w));
    CHECK(w.context_frontier_count == 0 && w.context_frontier_bytes == 0);
    w.image_count = 0;
    CHECK(setenv("DS4_AGENT_CONTEXT_FRONTIER_MB", "1", 1) == 0);
    CHECK(!agent_worker_capture_context_frontier(&w));
    CHECK(w.context_frontier_count == 0 && w.context_frontier_bytes == 0);
    printf("PASS: longest frontier=2304, retained<=64MiB, restore_ms=%.3f suffix_ms=%.3f; full-state identity, eviction, wrong-prefix/context, corrupt fallback, image exclusion, 1MiB capture cap\n",
           restore_ms, suffix_ms);
    rc = 0;
done:
    unsetenv("DS4_AGENT_CONTEXT_FRONTIERS"); unsetenv("DS4_AGENT_CONTEXT_FRONTIER_MB");
    agent_context_frontiers_clear(&w);
    free(text);
    ds4_tokens_free(&source); ds4_tokens_free(&first); ds4_tokens_free(&prefix);
    ds4_tokens_free(&old); ds4_tokens_free(&edited); ds4_tokens_free(&wrong);
    ds4_tokens_free(&w.transcript);
    ds4_session_snapshot_free(&frontier); ds4_session_snapshot_free(&expected); ds4_session_snapshot_free(&actual);
    ds4_session_free(w.session); ds4_engine_close(w.engine);
    return rc;
}
