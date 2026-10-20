/* Explicit model-dependent regression for the compaction cache opt-in. */
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "compact-prefix line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    int rc = 1;
    char err[256] = {0}, dir[] = "/tmp/ds4-compact-prefix-XXXXXX", sha[41];
    agent_config cfg = {.non_interactive = true,
        .gen = {.ctx_size = 4096, .think_mode = DS4_THINK_HIGH},
        .engine = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
                   .context_size = 4096, .prefill_chunk = 2048,
                   .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
                   .power_percent = 100}};
    agent_worker w = {.cfg = &cfg, .wake_fd = {-1, -1}};
    ds4_tokens sys = {0}, changed = {0};
    ds4_session_snapshot expected = {0}, actual = {0}, old = {0};
    CHECK(mkdtemp(dir));
    w.sysprompt_path = ds4_kvstore_path_join(dir, "sysprompt.kv");
    CHECK(ds4_engine_open(&w.engine, &cfg.engine) == 0);
    CHECK(ds4_session_create(&w.session, w.engine, cfg.gen.ctx_size) == 0);
    agent_worker_build_system_tokens(&w, &sys);
    CHECK(ds4_session_sync(w.session, &sys, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(w.session, &expected, err, sizeof(err)) == 0);
    CHECK(agent_kv_save_path(&w, w.sysprompt_path, &sys, "agent-system", sha, NULL, 0, err, sizeof(err)));
    ds4_tokens_copy(&w.transcript, &sys);
    ds4_chat_append_message(w.engine, &w.transcript, "user", "This conversation will be compacted. Keep the task state.");
    CHECK(ds4_session_sync(w.session, &w.transcript, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(w.session, &old, err, sizeof(err)) == 0);

    /* Disabled: an attempted restoration leaves the full conversation live. */
    CHECK(unsetenv("DS4_AGENT_COMPACT_PREFIX_CACHE") == 0);
    CHECK(!agent_worker_restore_compact_prefix(&w, &sys));
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == old.len && !memcmp(actual.ptr, old.ptr, old.len));

    CHECK(setenv("DS4_AGENT_COMPACT_PREFIX_CACHE", "1", 1) == 0);
    double t0 = now_sec();
    CHECK(agent_worker_restore_compact_prefix(&w, &sys));
    double restore_ms = (now_sec() - t0) * 1000;
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == expected.len && !memcmp(actual.ptr, expected.ptr, expected.len));
    CHECK(agent_tokens_equal(ds4_session_tokens(w.session), &sys));

    /* A different bootstrap prompt must not load a shape-compatible old KV. */
    CHECK(ds4_session_load_snapshot(w.session, &old, err, sizeof(err)) == 0);
    ds4_tokens_copy(&changed, &sys);
    changed.v[changed.len - 1] = changed.v[changed.len - 1] == 1 ? 2 : 1;
    CHECK(!agent_worker_restore_compact_prefix(&w, &changed));
    CHECK(ds4_session_save_snapshot(w.session, &actual, err, sizeof(err)) == 0);
    CHECK(actual.len == old.len && !memcmp(actual.ptr, old.ptr, old.len));

    /* Missing or corrupt disk state falls back without consuming stale KV. */
    FILE *fp = fopen(w.sysprompt_path, "wb");
    CHECK(fp);
    CHECK(fputs("corrupt", fp) >= 0);
    CHECK(fclose(fp) == 0);
    CHECK(!agent_worker_restore_compact_prefix(&w, &sys));
    CHECK(ds4_session_sync(w.session, &changed, err, sizeof(err)) == 0);
    CHECK(agent_tokens_equal(ds4_session_tokens(w.session), &changed));
    CHECK(unlink(w.sysprompt_path) == 0);
    CHECK(!agent_worker_restore_compact_prefix(&w, &sys));
    printf("PASS: prefix=%d bytes=%llu restore_ms=%.3f; exact restored state, disabled/mismatched/corrupt/missing cache fallback\n",
        sys.len, (unsigned long long)expected.len, restore_ms);
    rc = 0;
done:
    unsetenv("DS4_AGENT_COMPACT_PREFIX_CACHE");
    if (w.sysprompt_path) unlink(w.sysprompt_path);
    rmdir(dir);
    free(w.sysprompt_path);
    ds4_tokens_free(&sys); ds4_tokens_free(&changed); ds4_tokens_free(&w.transcript);
    ds4_session_snapshot_free(&expected); ds4_session_snapshot_free(&actual); ds4_session_snapshot_free(&old);
    ds4_session_free(w.session); ds4_engine_close(w.engine);
    return rc;
}
