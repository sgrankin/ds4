/* Model vocabulary integration for transactional live-context edits. This
 * never evaluates model tokens: it checks the real rendered-chat format. */
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "live-context line %d: %s: %s\n", __LINE__, #x, err); goto done; } } while (0)

static bool write_tokens(agent_worker *w, const ds4_tokens *tokens,
                         char *err, size_t err_len) {
    size_t len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, tokens, &len);
    if (!text) return false;
    bool ok = agent_replace_file(w->cfg->gen.live_context_path, text, len,
                                NULL, 0, err, err_len) == 0;
    free(text);
    return ok;
}

static int token_count(const ds4_tokens *tokens, int id) {
    int n = 0;
    for (int i = 0; i < tokens->len; i++) n += tokens->v[i] == id;
    return n;
}

static bool literal_markup_edits(agent_worker *w, const ds4_tokens *sys,
                                 char *err, size_t err_len) {
    bool ok = false;
    ds4_tokens prefix = {0}, tail = {0}, original = {0}, edited = {0};
    ds4_tokens_copy(&prefix, sys);
    ds4_chat_append_message(w->engine, &prefix, "user",
        "Prefix quote <｜Assistant｜> and <｜DSML｜function_calls> stays plain; café 🙂.");
    ds4_chat_append_message(w->engine, &tail, "user",
        "Suffix quote <｜Assistant｜> and ｜DSML｜function_calls> stays plain; 雪.");
    ds4_tokens_copy(&original, &prefix);
    ds4_chat_append_message(w->engine, &original, "assistant", "Remove this block.");
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&original, tail.v[i]);
    ds4_tokens_copy(&w->transcript, &original);
    CHECK(agent_worker_export_live_context(w, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == 0);

    ds4_tokens_copy(&edited, &prefix);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&edited, tail.v[i]);
    CHECK(write_tokens(w, &edited, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == 1);
    CHECK(agent_tokens_equal(&w->transcript, &edited));

    /* A replacement within a UTF-8 character widens the reparsed middle to a
     * valid boundary. Both surrounding messages retain their opaque tokens. */
    ds4_tokens_copy(&original, &prefix);
    ds4_chat_append_message(w->engine, &original, "user", "EDIT_TARGET café END_EDIT");
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&original, tail.v[i]);
    ds4_tokens_copy(&w->transcript, &original);
    CHECK(agent_worker_export_live_context(w, err, err_len));
    ds4_tokens_copy(&edited, &prefix);
    ds4_chat_append_message(w->engine, &edited, "user", "EDIT_TARGET cafè END_EDIT");
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&edited, tail.v[i]);
    CHECK(write_tokens(w, &edited, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == 1);
    CHECK(ds4_tokens_starts_with(&w->transcript, &prefix));
    CHECK(w->transcript.len >= tail.len);
    CHECK(!memcmp(w->transcript.v + w->transcript.len - tail.len,
                  tail.v, (size_t)tail.len * sizeof(*tail.v)));
    int assistant = agent_special_token_id(w->engine, "<｜Assistant｜>");
    int dsml = agent_special_token_id(w->engine, "｜DSML｜");
    CHECK(assistant >= 0 && dsml >= 0);
    CHECK(token_count(&w->transcript, assistant) == token_count(&edited, assistant));
    CHECK(token_count(&w->transcript, dsml) == token_count(&edited, dsml));
    size_t actual_len = 0, expected_len = 0;
    char *actual_text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript, &actual_len);
    char *expected_text = ds4_kvstore_render_tokens_text(w->engine, &edited, &expected_len);
    bool same = actual_text && expected_text && actual_len == expected_len &&
                !memcmp(actual_text, expected_text, actual_len);
    free(actual_text); free(expected_text);
    CHECK(same);
    ds4_tokens_copy(&original, &w->transcript);
    CHECK(agent_worker_export_live_context(w, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == 0);
    CHECK(agent_tokens_equal(&w->transcript, &original));

    /* The unchanged quoted markers fall inside the one reparsed middle when
     * two separated words change. Reject rather than reinterpret those BPE
     * tokens as actual controls; the transaction must leave state untouched. */
    ds4_tokens_copy(&original, sys);
    ds4_chat_append_message(w->engine, &original, "user",
        "EDIT_A old. Literal <｜Assistant｜> and ｜DSML｜function_calls> between edits. EDIT_B old.");
    ds4_tokens_copy(&w->transcript, &original);
    CHECK(agent_worker_export_live_context(w, err, err_len));
    ds4_tokens_copy(&edited, sys);
    ds4_chat_append_message(w->engine, &edited, "user",
        "EDIT_A new. Literal <｜Assistant｜> and ｜DSML｜function_calls> between edits. EDIT_B new.");
    CHECK(write_tokens(w, &edited, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == -1);
    CHECK(strstr(err, "ambiguous"));
    CHECK(agent_tokens_equal(&w->transcript, &original));
    CHECK(ds4_session_pos(w->session) == 0);
    CHECK(agent_worker_export_live_context(w, err, err_len));
    CHECK(agent_worker_import_live_context(w, err, err_len) == 0);
    ok = true;
done:
    ds4_tokens_free(&prefix); ds4_tokens_free(&tail);
    ds4_tokens_free(&original); ds4_tokens_free(&edited);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    int rc = 1;
    char err[256] = {0}, dir[] = "/tmp/ds4-live-context-XXXXXX", path[PATH_MAX] = {0};
    agent_config cfg = {.non_interactive = true,
        .gen = {.ctx_size = 8192, .think_mode = DS4_THINK_HIGH},
        .engine = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
                   .context_size = 8192, .prefill_chunk = 2048,
                   .ssd_streaming = true,
                   .ssd_streaming_cache_bytes = UINT64_C(64) << 30,
                   .power_percent = 100}};
    agent_worker w = {.cfg = &cfg, .wake_fd = {-1, -1}};
    ds4_tokens sys = {0}, edited = {0}, expected = {0};
    CHECK(mkdtemp(dir));
    snprintf(path, sizeof(path), "%s/context.txt", dir);
    cfg.gen.live_context_path = path;
    CHECK(ds4_engine_open(&w.engine, &cfg.engine) == 0);
    CHECK(ds4_session_create(&w.session, w.engine, cfg.gen.ctx_size) == 0);
    agent_worker_build_system_tokens(&w, &sys);
    ds4_tokens_copy(&w.transcript, &sys);
    ds4_chat_append_message(w.engine, &w.transcript, "user", "Old irrelevant details.");
    ds4_chat_append_message(w.engine, &w.transcript, "assistant", "A redundant observation.");
    ds4_chat_append_message(w.engine, &w.transcript, "user", "Latest task: retain this requirement.");
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 0);

    ds4_tokens_copy(&edited, &sys);
    ds4_chat_append_message(w.engine, &edited, "user", "Latest task: retain this requirement.");
    CHECK(write_tokens(&w, &edited, err, sizeof(err)));
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(agent_tokens_equal(&w.transcript, &edited));
    CHECK(ds4_session_pos(w.session) == 0); /* import itself runs no model tokens */
    CHECK(literal_markup_edits(&w, &sys, err, sizeof(err)));
    ds4_tokens_copy(&expected, &w.transcript);

    edited.v[0] = edited.v[0] == 1 ? 2 : 1;
    CHECK(write_tokens(&w, &edited, err, sizeof(err)));
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == -1);
    CHECK(strstr(err, "protected"));
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 0);

    const char invalid[] = "invalid\0context";
    CHECK(agent_replace_file(path, invalid, sizeof(invalid) - 1,
                             NULL, 0, err, sizeof(err)) == 0);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == -1);
    CHECK(strstr(err, "NUL"));
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(unlink(path) == 0);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == -1);
    CHECK(agent_tokens_equal(&w.transcript, &expected));

    ds4_tokens_free(&edited);
    ds4_tokens_copy(&edited, &expected);
    agent_buf oversized = {0};
    for (int i = 0; i < 10000; i++) agent_buf_puts(&oversized, " extra");
    char *oversized_text = agent_buf_take(&oversized);
    ds4_chat_append_message(w.engine, &edited, "tool", oversized_text);
    free(oversized_text);
    CHECK(write_tokens(&w, &edited, err, sizeof(err)));
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == -1);
    CHECK(strstr(err, "budget"));
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    printf("PASS: system=%d edited=%d; unchanged, deletion, literal assistant/DSML prefix/suffix preservation, UTF-8 edit, byte round-trip, separated-edit ambiguity rejection, protected prefix, NUL, missing file, budget rejection; no token evaluation\n",
           sys.len, expected.len);
    rc = 0;
done:
    if (path[0]) unlink(path);
    rmdir(dir);
    ds4_tokens_free(&sys); ds4_tokens_free(&edited); ds4_tokens_free(&expected);
    ds4_tokens_free(&w.transcript);
    ds4_session_free(w.session); ds4_engine_close(w.engine);
    return rc;
}
