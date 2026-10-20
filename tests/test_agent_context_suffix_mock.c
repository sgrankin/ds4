/* Model-free transaction tests. Only the session/tokenizer seams below are
 * mocked; selection, file import, budgets, rollback and persistence are the
 * actual agent implementation. No model is opened and no GPU API is called. */
#include <inttypes.h>
#define ds4_session_is_approximate mock_approximate
#define ds4_session_payload_bytes mock_payload_bytes
#define ds4_session_enable_suffix_prerope mock_enable_prerope
#define ds4_session_suffix_prerope_bytes mock_prerope_bytes
#define ds4_session_suffix_cache_bytes mock_cache_bytes
#define ds4_session_save_snapshot mock_save_snapshot
#define ds4_session_load_snapshot mock_load_snapshot
#define ds4_session_load_suffix_prefix mock_load_suffix_prefix
#define ds4_session_capture_suffix_cache mock_capture
#define ds4_session_reuse_suffix_relocated mock_graft
#define ds4_session_stage_payload mock_stage_payload
#define ds4_session_invalidate mock_invalidate
#define ds4_session_tokens mock_tokens
#define ds4_session_pos mock_pos
#define ds4_session_ctx mock_ctx
#define ds4_session_common_prefix mock_common
#define ds4_session_sync mock_sync
#define ds4_session_argmax mock_argmax
#define ds4_session_eval mock_eval
#define ds4_session_set_cancel mock_set_cancel
#define ds4_session_set_progress mock_set_progress
#define ds4_session_set_display_progress mock_set_display
#define ds4_session_is_distributed mock_distributed
#define ds4_engine_is_deepseek41 mock_ds41
#define ds4_engine_is_qwen4 mock_qwen
#define ds4_engine_is_glm_dsa mock_glm
#define ds4_engine_has_vision mock_vision
#define ds4_engine_model_id mock_model_id
#define ds4_engine_routed_quant_bits mock_quant
#define ds4_kvstore_render_tokens_text mock_render
#define ds4_token_text mock_piece
#define ds4_token_user mock_user
#define ds4_token_assistant mock_assistant_id
#define ds4_token_is_stop_for_think_mode mock_stop
#define ds4_tokenize_rendered_chat mock_tokenize
#define ds4_chat_begin mock_chat_begin
#define ds4_chat_append_think_prefix mock_think
#define ds4_chat_append_message mock_message
#define ds4_chat_append_assistant_prefix mock_assistant_prefix
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "suffix mock line %d: %s\n", __LINE__, #x); abort(); } } while (0)

struct ds4_session {
    ds4_tokens tokens;
    bool valid, approximate, prerope_enabled;
    uint32_t graph_tag;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
};
typedef struct { uint32_t magic, count, graph_tag; } mock_header;
static ds4_engine *const fake_engine = (ds4_engine *)(uintptr_t)1;
static agent_worker *active_worker;
static int sync_calls, load_calls, snapshot_calls, capture_calls, graft_calls, stage_calls, eval_calls;
static int fail_load_call, cancel_load_call, graft_mode;
static bool reject_capture, cancel_sync;
static int mock_user_id = 'U';

bool mock_approximate(const ds4_session *s) { return s && s->approximate; }
const ds4_tokens *mock_tokens(ds4_session *s) { return &s->tokens; }
int mock_pos(ds4_session *s) { return s->valid ? s->tokens.len : 0; }
int mock_ctx(ds4_session *s) { (void)s; return 4096; }
bool mock_distributed(ds4_session *s) { (void)s; return false; }
bool mock_ds41(ds4_engine *e) { return e == fake_engine; }
bool mock_qwen(ds4_engine *e) { (void)e; return false; }
bool mock_glm(ds4_engine *e) { (void)e; return false; }
bool mock_vision(ds4_engine *e) { (void)e; return false; }
int mock_model_id(ds4_engine *e) { (void)e; return 1; }
int mock_quant(ds4_engine *e) { (void)e; return 2; }
int mock_user(ds4_engine *e) { (void)e; return mock_user_id; }
int mock_assistant_id(ds4_engine *e) { (void)e; return 1000; }
int mock_argmax(ds4_session *s) { (void)s; return eval_calls ? 1003 : 'q'; }
int mock_eval(ds4_session *s, int token, char *err, size_t err_len) {
    (void)err; (void)err_len; eval_calls++; ds4_tokens_push(&s->tokens, token); return 0;
}
bool mock_stop(ds4_engine *e, int token, ds4_think_mode mode) {
    (void)e; (void)mode; return token == 1003;
}
void mock_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) { s->cancel = fn; s->cancel_ud = ud; }
void mock_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) { (void)s; (void)fn; (void)ud; }
void mock_set_display(ds4_session *s, ds4_session_progress_fn fn, void *ud) { (void)s; (void)fn; (void)ud; }
void mock_invalidate(ds4_session *s) { s->valid = false; s->approximate = false; s->tokens.len = 0; }
int mock_common(ds4_session *s, const ds4_tokens *t) {
    int i = 0;
    while (s->valid && i < s->tokens.len && i < t->len && s->tokens.v[i] == t->v[i]) i++;
    return i;
}
int mock_sync(ds4_session *s, const ds4_tokens *t, char *err, size_t err_len) {
    sync_calls++;
    if (cancel_sync) active_worker->interrupt = true;
    if (s->cancel && s->cancel(s->cancel_ud)) {
        mock_invalidate(s); snprintf(err, err_len, "interrupted"); return DS4_SESSION_SYNC_INTERRUPTED;
    }
    bool keep = s->valid && s->approximate && mock_common(s, t) == s->tokens.len && t->len >= s->tokens.len;
    ds4_tokens_copy(&s->tokens, t); s->valid = true; s->approximate = keep;
    s->graph_tag = UINT32_C(0x12345678);
    for (int i = 0; i < t->len; i++) s->graph_tag = s->graph_tag * 33u + (uint32_t)t->v[i];
    return 0;
}
uint64_t mock_payload_bytes(ds4_session *s) {
    return s->valid && !s->approximate ? sizeof(mock_header) + (uint64_t)s->tokens.len * sizeof(int) : 0;
}
int mock_enable_prerope(ds4_session *s, char *err, size_t err_len) {
    (void)err; (void)err_len; s->prerope_enabled = true; return 0;
}
uint64_t mock_prerope_bytes(const ds4_session *s) {
    return s && s->prerope_enabled ? 8u : 0u;
}
uint64_t mock_cache_bytes(ds4_session *s) {
    return s->valid ? sizeof(mock_header) + (uint64_t)s->tokens.len * sizeof(int) +
                     mock_prerope_bytes(s) : 0;
}
int mock_save_snapshot(ds4_session *s, ds4_session_snapshot *out, char *err, size_t err_len) {
    snapshot_calls++;
    uint64_t bytes = mock_payload_bytes(s);
    if (!bytes) { snprintf(err, err_len, "not exact"); return 1; }
    if (out->cap < bytes + 1) { out->ptr = realloc(out->ptr, (size_t)bytes + 1); CHECK(out->ptr); out->cap = bytes + 1; }
    mock_header h = {UINT32_C(0x4d534e50), (uint32_t)s->tokens.len, s->graph_tag};
    memcpy(out->ptr, &h, sizeof(h)); memcpy(out->ptr + sizeof(h), s->tokens.v, (size_t)s->tokens.len * sizeof(int));
    out->len = bytes; out->ptr[out->len] = 0;
    return 0;
}
int mock_load_snapshot(ds4_session *s, const ds4_session_snapshot *in, char *err, size_t err_len) {
    load_calls++;
    mock_header h = {0};
    if (in->len >= sizeof(h)) memcpy(&h, in->ptr, sizeof(h));
    if (load_calls == fail_load_call || h.magic != UINT32_C(0x4d534e50) ||
        in->len != sizeof(h) + (uint64_t)h.count * sizeof(int)) {
        mock_invalidate(s); snprintf(err, err_len, "bad mock snapshot"); return 1;
    }
    ds4_tokens t = {(int *)(in->ptr + sizeof(h)), (int)h.count, (int)h.count};
    ds4_tokens_copy(&s->tokens, &t); s->valid = true; s->approximate = false; s->graph_tag = h.graph_tag;
    if (load_calls == cancel_load_call) active_worker->interrupt = true;
    return 0;
}
int mock_load_suffix_prefix(ds4_session *s, const ds4_suffix_cache *cache,
                              char *err, size_t err_len) {
    if (cache->approximate) { snprintf(err, err_len, "approximate prefix"); return 1; }
    return mock_load_snapshot(s, &cache->state, err, err_len);
}
int mock_capture(ds4_session *s, ds4_suffix_cache *cache, char *err, size_t err_len) {
    capture_calls++;
    if (reject_capture) { snprintf(err, err_len, "unsupported state"); return 1; }
    cache->engine = fake_engine; cache->ctx_size = mock_ctx(s);
    return mock_save_snapshot(s, &cache->state, err, err_len);
}
int mock_graft(ds4_session *s, const ds4_suffix_cache *cache, const ds4_tokens *t,
                 uint32_t source_start, ds4_suffix_relocation_stats *stats,
                 char *err, size_t err_len) {
    graft_calls++;
    mock_header h; memcpy(&h, cache->state.ptr, sizeof(h));
    CHECK(s->valid && !s->approximate && cache->engine == fake_engine);
    CHECK(!(source_start & 1u) && !(s->tokens.len & 1));
    uint32_t start = (uint32_t)s->tokens.len, end = start + h.count - source_start;
    if (start > source_start) CHECK(cache->prerope && cache->prerope_bytes);
    const int *old = (const int *)(cache->state.ptr + sizeof(h));
    for (uint32_t i = start; i < end; i++) CHECK(t->v[i] == old[source_start + i - start]);
    CHECK(t->len - (int)end >= 64);
    if (graft_mode) {
        mock_invalidate(s);
        if (graft_mode == 2 || graft_mode == 3) active_worker->interrupt = true;
        if (graft_mode != 3) { snprintf(err, err_len, "mock graft failure"); return 1; }
    }
    ds4_tokens_copy(&s->tokens, t); s->valid = true; s->approximate = true; s->graph_tag ^= UINT32_C(0x80000000);
    *stats = (ds4_suffix_relocation_stats){.source_start = source_start, .destination_start = start,
        .source_end = h.count, .destination_end = end,
        .reuse = {.reused_tokens = end - start, .fresh_tail_tokens = (uint32_t)t->len - end, .copied_bytes = 1234}};
    return 0;
}
int mock_stage_payload(ds4_session *s, ds4_session_payload_file *out, char *err, size_t err_len) {
    (void)s; (void)out; stage_calls++; snprintf(err, err_len, "unexpected canonical stage"); return 1;
}

static const char *control_text(int token) {
    switch (token) {
        case 1000: return "<｜Assistant｜>";
        case 1001: return "｜DSML｜";
        case 1002: return "<think>";
        case 'U': return "<｜User｜>";
        case 1004: return "U"; /* Ordinary text U is distinct from the toy user role. */
        default: return NULL;
    }
}
char *mock_piece(ds4_engine *e, int token, size_t *len) {
    (void)e; const char *special = control_text(token);
    char *p;
    if (special) { *len = strlen(special); p = strdup(special); }
    else { CHECK(token > 0 && token < 256); *len = 1; p = malloc(2); p[0] = (char)token; p[1] = 0; }
    return p;
}
char *mock_render(ds4_engine *e, const ds4_tokens *t, size_t *len) {
    agent_buf b = {0};
    for (int i = 0; i < t->len; i++) { size_t n; char *p = mock_piece(e, t->v[i], &n); agent_buf_append(&b, p, n); free(p); }
    *len = b.len; return agent_buf_take(&b);
}
void mock_tokenize(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;
    /* The fixture has one protected S token; trusted built-in tools are not
     * part of this toy tokenizer. User/file text takes the real splice path. */
    if (!strncmp(text, "You are a coding agent", 22)) return;
    while (*text) {
        int id = 0;
        if (!strncmp(text, control_text('U'), strlen(control_text('U')))) {
            id = 'U'; text += strlen(control_text('U'));
        }
        for (int t = 1000; !id && t <= 1002; t++) {
            const char *m = control_text(t);
            if (!strncmp(text, m, strlen(m))) { id = t; text += strlen(m); break; }
        }
        if (!id) { id = (unsigned char)*text++; if (id == 'U') id = 1004; }
        ds4_tokens_push(out, id);
    }
}
void mock_chat_begin(ds4_engine *e, ds4_tokens *t) { (void)e; ds4_tokens_push(t, 'S'); }
void mock_think(ds4_engine *e, ds4_tokens *t, ds4_think_mode mode) { (void)e; (void)t; (void)mode; }
void mock_message(ds4_engine *e, ds4_tokens *t, const char *role, const char *content) {
    (void)e;
    if (!strcmp(role, "user") && strstr(content, "[ds4-agent compacted earlier conversation.")) {
        ds4_tokens_push(t, 'U'); ds4_tokens_push(t, 'd'); ds4_tokens_push(t, 'd');
    }
}
void mock_assistant_prefix(ds4_engine *e, ds4_tokens *t, ds4_think_mode mode) {
    (void)e; (void)t; (void)mode;
}

static void reset_counters(void) {
    sync_calls = load_calls = snapshot_calls = capture_calls = graft_calls = stage_calls = eval_calls = 0;
    fail_load_call = cancel_load_call = graft_mode = 0; reject_capture = cancel_sync = false;
}
static void prepare_with_user(agent_worker *w, agent_config *cfg, ds4_session *s,
                              int user_position) {
    reset_counters();
    memset(w, 0, sizeof(*w)); memset(cfg, 0, sizeof(*cfg)); memset(s, 0, sizeof(*s));
    cfg->non_interactive = true; cfg->gen.ctx_size = 4096; cfg->gen.live_context_path = "mock-context";
    *w = (agent_worker){.engine = fake_engine, .cfg = cfg, .session = s, .wake_fd = {-1,-1},
        .user_activity = true, .initialized = true, .session_dirty = true};
    CHECK(pthread_mutex_init(&w->mu, NULL) == 0); active_worker = w;
    for (int i = 0; i < 640; i++) ds4_tokens_push(&w->transcript, i == 0 ? 'S' : i < 256 ? 'a' : i < 384 ? 'b' : 'c');
    if (user_position >= 0) {
        CHECK(user_position > 0 && user_position < w->transcript.len);
        w->transcript.v[user_position] = 'U';
    }
    char err[160] = {0};
    for (int n = 256; n <= 512; n += 256) {
        ds4_tokens t = w->transcript; t.len = n;
        CHECK(mock_sync(s, &t, err, sizeof(err)) == 0);
        CHECK(agent_worker_capture_context_frontier(w));
    }
    CHECK(mock_sync(s, &w->transcript, err, sizeof(err)) == 0);
    reset_counters();
}
static void prepare(agent_worker *w, agent_config *cfg, ds4_session *s) {
    prepare_with_user(w, cfg, s, 576);
}
static void cleanup(agent_worker *w) {
    agent_live_context_clear(w);
    agent_context_frontiers_clear(w); ds4_tokens_free(&w->transcript); ds4_tokens_free(&w->session->tokens);
    free(w->session_title); free(w->out); CHECK(pthread_mutex_destroy(&w->mu) == 0);
}
static ds4_tokens replacement(agent_worker *w, int deletion) {
    ds4_tokens t = {0};
    for (int i = 0; i < 256; i++) ds4_tokens_push(&t, w->transcript.v[i]);
    for (int i = 0; i < 128 - deletion; i++) ds4_tokens_push(&t, 'd');
    for (int i = 384; i < w->transcript.len; i++) ds4_tokens_push(&t, w->transcript.v[i]);
    return t;
}
static void add_originals(agent_worker *w) {
    agent_context_frontier *f = &w->context_frontiers[1];
    f->cache.prerope = malloc(8); CHECK(f->cache.prerope);
    memset(f->cache.prerope, 0x5a, 8);
    f->cache.prerope_bytes = 8; w->context_frontier_bytes += 8;
}
static void test_selection(void) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    agent_context_suffix_plan p;
    ds4_tokens t = replacement(&w, 0);
    CHECK(agent_context_suffix_select(&w, &t, 64, &p));
    CHECK(p.source_start == 384 && p.destination_start == 384 && p.reused_tokens == 128 && p.fresh_tokens == 128);
    ds4_tokens_free(&t); t = replacement(&w, 64);
    CHECK(agent_context_suffix_select(&w, &t, 128, &p));
    CHECK(p.source_start == 384 && p.destination_start == 320 && p.destination_end == 448 && p.fresh_tokens == 128);
    CHECK(!agent_context_suffix_select(&w, &t, 129, &p));
    ds4_tokens_free(&t); t = replacement(&w, 1); CHECK(!agent_context_suffix_select(&w, &t, 64, &p));
    ds4_tokens_free(&t); t = replacement(&w, -2); CHECK(!agent_context_suffix_select(&w, &t, 64, &p));
    ds4_tokens_free(&t); t = replacement(&w, 0); t.v[384] = 'd';
    CHECK(agent_context_suffix_select(&w, &t, 64, &p)); CHECK(p.source_start == 386 && p.destination_start == 386);
    w.context_frontiers[1].ctx++; CHECK(!agent_context_suffix_select(&w, &t, 64, &p));
    ds4_tokens_free(&t); cleanup(&w);
}

/* Only the native serializer layout is supplied here; selection uses the
 * actual disk cache and agent. Tensor values and the archive are opaque to
 * this selector regression. No session load, model, or GPU work is needed. */
static void store_disk_selection_source(agent_worker *w, unsigned n) {
    uint64_t bytes = 52u + ((uint64_t)n + 129280u) * 4u +
        40u * (n < 128u ? n : 128u) * 512u * 4u;
    for (unsigned owner = 0; owner < 4; owner++) {
        bytes += (uint64_t)(n / (owner < 3 ? 2u : 1u)) * 640u * 4u;
        if (owner < 3 && (n & 1u)) bytes += 4096u;
    }
    uint8_t *state = calloc(1, (size_t)bytes); CHECK(state);
    const uint32_t header[13] = {0x34565344u, 2, 4096, 1, 128, 128, 4097,
                                n, 40, 512, 128, 129280, 0x413431u};
    for (unsigned i = 0; i < 13; i++) ds4_kvstore_le_put32(state + i * 4u, header[i]);
    for (unsigned i = 0; i < n; i++)
        ds4_kvstore_le_put32(state + 52u + i * 4u, (uint32_t)w->transcript.v[i]);
    char key[41], err[160] = {0}; const uint8_t tape = 1;
    CHECK(ds4_state_cache_put(w->state_cache, w->state_cache_profile,
        state, bytes, &tape, 1, false, key, err, sizeof(err)) == 0);
    free(state);
}

static void test_disk_prefix_is_not_graft(const char *directory) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    /* Unique prefix IDs prevent repeated text from creating another shifted
     * anchor: this source is genuinely the unchanged exact prefix only. */
    for (int i = 0; i < 256; i++) w.transcript.v[i] = 2000 + i;
    char path[PATH_MAX], err[160] = {0};
    snprintf(path, sizeof(path), "%s/selection-cache", directory);
    w.state_cache = ds4_state_cache_open(path, 0, UINT64_C(64) << 20, err, sizeof(err));
    CHECK(w.state_cache); strcpy(w.state_cache_profile, "suffix-selection-regression");
    store_disk_selection_source(&w, 256);
    ds4_tokens edited = replacement(&w, 64);
    agent_context_suffix_plan plan = {.source_frontier = -1};
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", "1", 1) == 0);
    CHECK(!agent_context_suffix_select_disk(&w, &edited, 128, &plan));
    CHECK(!plan.disk_ref); /* Native prefix remains available for exact restore. */
    uint32_t matched = 0;
    ds4_state_cache_ref *prefix = ds4_state_cache_find_prefix(w.state_cache,
        w.state_cache_profile, 4096, edited.v, (uint32_t)edited.len, &matched);
    CHECK(prefix && matched == 256); ds4_state_cache_release(prefix);
    store_disk_selection_source(&w, 512);
    CHECK(agent_context_suffix_select_disk(&w, &edited, 128, &plan));
    CHECK(plan.source_start == 384 && plan.destination_start == 320 && plan.reused_tokens == 128);
    ds4_state_cache_release(plan.disk_ref);
    for (int i = 512; i < w.transcript.len; i++)
        if (i != 576) w.transcript.v[i] = 'e';
    ds4_tokens expanded = replacement(&w, -66);
    plan = (agent_context_suffix_plan){.source_frontier = -1};
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT", "1", 1) == 0);
    CHECK(agent_context_suffix_select_disk(&w, &expanded, 128, &plan));
    CHECK(plan.source_start == 384 && plan.destination_start == 450 &&
          plan.destination_end == 578 && plan.reused_tokens == 128);
    ds4_state_cache_release(plan.disk_ref); ds4_tokens_free(&expanded);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") == 0);
    plan = (agent_context_suffix_plan){.source_frontier = -1};
    ds4_tokens tail_only = {.v = w.transcript.v + 384, .len = 256, .cap = 256};
    CHECK(!agent_context_suffix_select_disk(&w, &tail_only, 128, &plan));
    CHECK(ds4_state_cache_close(w.state_cache, err, sizeof(err)) == 0);
    w.state_cache = NULL; ds4_tokens_free(&edited); cleanup(&w);
}

static void test_original_source_after_prefix_edit(void) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    agent_context_frontier *f = &w.context_frontiers[1];
    /* The source is the immutable original endpoint, not an inverse-rotated
     * copy of the already edited live graph. Tape contents are opaque here. */
    f->cache.prerope = calloc(1, 8); CHECK(f->cache.prerope);
    f->cache.prerope_bytes = 8; w.context_frontier_bytes += 8;
    ds4_tokens first = replacement(&w, 64), second = {0};
    for (int i = 0; i < 256; i++) ds4_tokens_push(&second, w.transcript.v[i]);
    for (int i = 0; i < 32; i++) ds4_tokens_push(&second, 'd');
    for (int i = 384; i < w.transcript.len; i++) ds4_tokens_push(&second, w.transcript.v[i]);
    ds4_tokens_copy(&w.transcript, &first); ds4_tokens_copy(&s.tokens, &first); s.approximate = true;
    CHECK(!ds4_tokens_starts_with(&w.transcript, &f->prefix));
    agent_context_suffix_plan plan;
    CHECK(!agent_context_suffix_select(&w, &second, 128, &plan)); /* Legacy gate. */
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", "1", 1) == 0);
    CHECK(agent_context_suffix_select(&w, &second, 128, &plan));
    CHECK(plan.source_frontier == 1 && plan.source_start == 384 && plan.source_end == 512);
    CHECK(plan.destination_start == 288 && plan.destination_end == 416);
    CHECK(plan.reused_tokens == 128 && plan.fresh_tokens == 128);
    CHECK(f->prefix.len == 512 && f->prefix.v[256] == 'b' && f->prefix.v[384] == 'c');
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") == 0);
    ds4_tokens_free(&first); ds4_tokens_free(&second); cleanup(&w);
}

static void test_right_original_source(void) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    /* A distinct fresh tail makes the original 512 endpoint unambiguous. */
    for (int i = 512; i < w.transcript.len; i++)
        if (i != 576) w.transcript.v[i] = 'e';
    char err[160] = {0}; CHECK(mock_sync(&s, &w.transcript, err, sizeof(err)) == 0);
    add_originals(&w); reset_counters();
    agent_context_frontier *source = &w.context_frontiers[1];
    ds4_session_snapshot original = {0};
    original.ptr = malloc(source->cache.state.len); CHECK(original.ptr);
    original.len = source->cache.state.len;
    memcpy(original.ptr, source->cache.state.ptr, original.len);
    ds4_tokens edited = replacement(&w, -66), sys = {.v=w.transcript.v, .len=1, .cap=1};
    agent_context_suffix_plan plan;
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", "1", 1) == 0);
    CHECK(!agent_context_suffix_select(&w, &edited, 128, &plan)); /* Default left only. */
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT", "1", 1) == 0);
    CHECK(agent_context_suffix_select(&w, &edited, 128, &plan));
    CHECK(plan.source_start == 384 && plan.source_end == 512);
    CHECK(plan.destination_start == 450 && plan.destination_end == 578);
    CHECK(plan.fresh_tokens == 128 && plan.destination_end <= 642); /* Newest user fully fresh. */
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0.90", 1) == 0);
    CHECK(!agent_context_suffix_resident_conflict());
    CHECK(agent_worker_try_context_suffix(&w, &edited, &sys, err, sizeof(err)) == 1);
    CHECK(graft_calls == 1 && s.approximate && agent_tokens_equal(&s.tokens, &edited));
    CHECK(source->cache.state.len == original.len &&
          !memcmp(source->cache.state.ptr, original.ptr, original.len));
    for (int i = 0; i < 8; i++) CHECK(((unsigned char *)source->cache.prerope)[i] == 0x5a);
    /* Re-import from the same endpoint after the live graph has changed. */
    ds4_tokens_copy(&w.transcript, &edited);
    ds4_tokens second = {0};
    for (int i = 0; i < 256; i++) ds4_tokens_push(&second, edited.v[i]);
    for (int i = 0; i < 192; i++) ds4_tokens_push(&second, 'f');
    for (int i = 450; i < edited.len; i++) ds4_tokens_push(&second, edited.v[i]);
    CHECK(agent_worker_try_context_suffix(&w, &second, &sys, err, sizeof(err)) == 1);
    CHECK(graft_calls == 2 && s.approximate && agent_tokens_equal(&s.tokens, &second));
    CHECK(source->cache.state.len == original.len &&
          !memcmp(source->cache.state.ptr, original.ptr, original.len));
    for (int i = 0; i < 8; i++) CHECK(((unsigned char *)source->cache.prerope)[i] == 0x5a);
    ds4_tokens odd = replacement(&w, -1);
    /* A repeated-token tail can still offer an even aligned subspan despite
     * an odd total edit length. Admission depends on the selected span. */
    CHECK(agent_context_suffix_select(&w, &odd, 128, &plan));
    CHECK(!(plan.source_start & 1u) && !(plan.destination_start & 1u));
    CHECK(plan.fresh_tokens >= 128);
    ds4_tokens_free(&odd); ds4_tokens_free(&second); ds4_tokens_free(&edited);
    ds4_session_snapshot_free(&original); cleanup(&w);
    CHECK(unsetenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") == 0);
}

static void test_builtin_compaction_reuses_tail(void) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    /* An older complete turn precedes the newest user. The bounded retained
     * tail includes an immutable endpoint and leaves that latest user fresh. */
    w.transcript.v[128] = 'U';
    for (int i = 0; i < w.context_frontier_count; i++) {
        agent_context_frontier *f = &w.context_frontiers[i];
        f->prefix.v[128] = 'U';
        ((int *)(f->cache.state.ptr + sizeof(mock_header)))[128] = 'U';
    }
    add_originals(&w);
    char err[160] = {0}; CHECK(mock_sync(&s, &w.transcript, err, sizeof(err)) == 0);
    cfg.gen.live_context_path = NULL; /* Automatic compaction needs no mirror file. */
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", "1", 1) == 0);
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0.90", 1) == 0);
    reset_counters();
    CHECK(agent_worker_compact_transcript(&w, "mock compaction", NULL, err, sizeof(err)));
    CHECK(eval_calls == 1 && graft_calls == 1 && s.approximate);
    CHECK(agent_tokens_equal(&w.transcript, &s.tokens));
    CHECK(w.transcript.len < 640 && w.transcript.v[w.transcript.len - 64] == 'U');
    cleanup(&w);
    CHECK(unsetenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") == 0);
}

static void test_transactions(void) {
    for (int mode = 0; mode < 12; mode++) {
        agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
        ds4_tokens t = replacement(&w, mode == 9 ? 64 : 0), sys = {.v = w.transcript.v, .len = 1, .cap = 1};
        char err[160] = {0}; ds4_session_snapshot before = {0}, after = {0};
        CHECK(mock_save_snapshot(&s, &before, err, sizeof(err)) == 0); reset_counters();
        if (mode == 1) w.interrupt = true;
        if (mode == 2) cancel_load_call = 1;
        if (mode == 3) cancel_sync = true;
        if (mode == 4) graft_mode = 1;
        if (mode == 5) graft_mode = 2;
        if (mode == 6) graft_mode = 3;
        if (mode == 7) reject_capture = true;
        if (mode == 8) w.context_frontiers[0].cache.state.ptr[0] ^= 1; /* Own source before this entry is dropped/memmoved. */
        if (mode == 10) fail_load_call = 1;
        if (mode == 11) { graft_mode = 1; fail_load_call = 3; }
        int rc = agent_worker_try_context_suffix(&w, &t, &sys, err, sizeof(err));
        if (mode == 0 || mode == 8 || mode == 9) {
            CHECK(rc == 1 && s.approximate && agent_tokens_equal(&s.tokens, &t));
            CHECK(agent_tokens_equal(&w.transcript, &(ds4_tokens){(int *)(before.ptr + sizeof(mock_header)), 640, 640}));
            if (mode == 8) CHECK(w.context_frontier_count == 1);
        } else if (mode == 11) {
            CHECK(rc == -1 && strstr(err, "rollback failed") && !s.valid && !s.approximate);
            CHECK(w.transcript.len == 640); /* No token edit committed after failed recovery. */
        } else {
            bool fallback = mode == 4 || mode == 7 || mode == 10;
            if (rc != (fallback ? 0 : -1))
                fprintf(stderr, "transaction mode=%d rc=%d err=%s loads=%d captures=%d grafts=%d\n",
                        mode, rc, err, load_calls, capture_calls, graft_calls);
            CHECK(rc == (fallback ? 0 : -1));
            if (rc < 0) CHECK(!strcmp(err, "interrupted"));
            CHECK(mock_save_snapshot(&s, &after, err, sizeof(err)) == 0);
            CHECK(after.len == before.len && !memcmp(after.ptr, before.ptr, before.len));
            CHECK(!s.approximate && s.tokens.len == 640 && w.transcript.len == 640);
        }
        ds4_session_snapshot_free(&before); ds4_session_snapshot_free(&after); ds4_tokens_free(&t); cleanup(&w);
    }
}
static void test_latest_user_fresh(void) {
    /* Endpoints are exclusive: marker512 may be wholly fresh after a512
     * frontier; marker511 must force exact fallback before graph mutation.
     * Test both same-position replacement and a64-token left shift. */
    const int positions[] = {-1, 384, 448, 511, 512, 513, 576, 639};
    for (int shifted = 0; shifted < 2; shifted++) {
        for (size_t i = 0; i < sizeof(positions)/sizeof(positions[0]); i++) {
            agent_worker w; agent_config cfg; ds4_session s;
            prepare_with_user(&w, &cfg, &s, positions[i]);
            ds4_tokens edited = replacement(&w, shifted ? 64 : 0);
            ds4_tokens sys = {.v=w.transcript.v, .len=1, .cap=1};
            agent_context_suffix_plan plan; char err[160] = {0};
            const bool allowed = positions[i] >= 512;
            CHECK(agent_context_suffix_select(&w, &edited, 64, &plan) == allowed);
            if (allowed) {
                CHECK(plan.destination_end <= (uint32_t)(positions[i] - (shifted ? 64 : 0)));
                CHECK(plan.fresh_tokens >= (uint32_t)(edited.len - positions[i] + (shifted ? 64 : 0)));
            } else {
                uint32_t graph = s.graph_tag;
                CHECK(agent_worker_try_context_suffix(&w, &edited, &sys, err, sizeof(err)) == 0);
                CHECK(load_calls == 0 && snapshot_calls == 0 && capture_calls == 0 && graft_calls == 0);
                CHECK(s.valid && !s.approximate && s.graph_tag == graph && s.tokens.len == 640);
                CHECK(agent_tokens_equal(&s.tokens, &w.transcript));
            }
            ds4_tokens_free(&edited); cleanup(&w);
        }
    }
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    ds4_tokens edited = replacement(&w, 0); agent_context_suffix_plan plan;
    edited.v[400] = 'U'; /* An earlier role must not replace the latest one. */
    CHECK(agent_context_suffix_select(&w, &edited, 64, &plan));
    CHECK(plan.destination_end == 512 && plan.destination_end <= 576);
    mock_user_id = -1;
    CHECK(!agent_context_suffix_select(&w, &edited, 64, &plan));
    mock_user_id = 'U';
    ds4_tokens_free(&edited); cleanup(&w);
}
static void test_gates_and_budget(void) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    ds4_tokens t = replacement(&w, 0), sys = {.v = w.transcript.v, .len = 1, .cap = 1}; char err[160] = {0};
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", "0", 1) == 0);
    CHECK(agent_worker_try_context_suffix(&w, &t, &sys, err, sizeof(err)) == 0 && load_calls == 0);
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", "1", 1) == 0);
    CHECK(setenv("DS4_AGENT_CONTEXT_FRONTIER_MB", "1", 1) == 0);
    CHECK(agent_worker_try_context_suffix(&w, &t, &sys, err, sizeof(err)) == 0 && load_calls == 0 && snapshot_calls == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_FRONTIER_MB") == 0);
    for (int k = 0; k < 3; k++) {
        CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", k == 0 ? "0.01" : k == 1 ? "nan" : "garbage", 1) == 0);
        CHECK(agent_context_suffix_resident_conflict());
        CHECK(agent_worker_try_context_suffix(&w, &t, &sys, err, sizeof(err)) == 0 && load_calls == 0);
    }
    CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0", 1) == 0); CHECK(!agent_context_suffix_resident_conflict());
    CHECK(unsetenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP") == 0);
    s.approximate = true; CHECK(agent_worker_try_context_suffix(&w, &t, &sys, err, sizeof(err)) == 0 && load_calls == 0);
    s.approximate = false;
    ds4_tokens current = w.context_frontiers[1].prefix; CHECK(mock_sync(&s, &current, err, sizeof(err)) == 0);
    s.approximate = true; reset_counters();
    CHECK(!agent_worker_capture_context_frontier(&w)); CHECK(snapshot_calls == 0); /* Duplicate token prefix must still refuse export. */
    ds4_tokens_free(&t); cleanup(&w);
}
static void test_import_and_persistence(const char *directory) {
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    char context[PATH_MAX], err[256] = {0}, sha[41]; snprintf(context, sizeof(context), "%s/context.txt", directory);
    cfg.gen.live_context_path = context; w.cache_dir = (char *)directory;
    w.session_title = strdup("Model-free suffix persistence"); w.session_created_at = 123456;
    ds4_tokens t = replacement(&w, 0); size_t n; char *text = mock_render(w.engine, &t, &n);
    CHECK(agent_replace_file(context, text, n, NULL, 0, err, sizeof(err)) == 0); free(text);
    uint32_t original_tag = s.graph_tag; graft_mode = 2;
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == -1 && !strcmp(err, "interrupted"));
    CHECK(w.transcript.v[256] == 'b' && s.tokens.v[256] == 'b' && s.graph_tag == original_tag && !s.approximate);
    w.interrupt = false; reset_counters();
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 2);
    CHECK(s.approximate && agent_tokens_equal(&w.transcript, &t));
    int prior_syncs = sync_calls, saved_tokens = 0;
    CHECK(agent_worker_save_session_now(&w, sha, &saved_tokens, err, sizeof(err)));
    CHECK(sync_calls == prior_syncs && stage_calls == 0 && saved_tokens == 640 && s.approximate);
    char *path = agent_kv_path_for_sha(directory, sha); FILE *fp = fopen(path, "rb"); CHECK(fp);
    ds4_kvstore_entry hdr = {0}; uint32_t bytes = 0;
    CHECK(ds4_kvstore_read_header(fp, &hdr, &bytes)); CHECK(hdr.payload_bytes == 0 && hdr.tokens == 640); fclose(fp);
    ds4_tokens loaded = {0};
    CHECK(agent_kv_load_path(&w, path, sha, NULL, 0, &loaded, NULL, err, sizeof(err)));
    CHECK(!s.approximate && sync_calls == prior_syncs + 1 && agent_tokens_equal(&loaded, &t));
    char *saved = NULL; size_t saved_len;
    CHECK(agent_read_file_bytes(path, &saved, &saved_len, err, sizeof(err)) == 0);
    const char *literal = control_text(1000);
    for (const unsigned char *p = (const unsigned char *)literal; *p; p++) ds4_tokens_push(&w.transcript, *p);
    s.approximate = true; w.session_dirty = true;
    CHECK(!agent_worker_save_session_now(&w, sha, &saved_tokens, err, sizeof(err)));
    CHECK(strstr(err, "literal control markup") && w.session_dirty && stage_calls == 0);
    char *same = NULL; size_t same_len;
    CHECK(agent_read_file_bytes(path, &same, &same_len, err, sizeof(err)) == 0);
    CHECK(same_len == saved_len && !memcmp(same, saved, saved_len)); free(saved); free(same);
    ds4_tokens control = {0}; ds4_tokens_push(&control, 1000); text = mock_render(w.engine, &control, &n);
    CHECK(agent_text_only_control_safe(&w, &control, text, n, err, sizeof(err))); free(text); ds4_tokens_free(&control);

    /* An edit entirely after an approximate live endpoint must still reset
     * approximation, rather than accepting it as an append-only prefix. */
    ds4_tokens_copy(&w.transcript, &t); ds4_tokens short_live = t; short_live.len = 600;
    CHECK(mock_sync(&s, &short_live, err, sizeof(err)) == 0); s.approximate = true;
    t.v[620] = 'z'; text = mock_render(w.engine, &t, &n);
    CHECK(agent_replace_file(context, text, n, NULL, 0, err, sizeof(err)) == 0); free(text);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(!s.approximate && agent_tokens_equal(&w.transcript, &t) && mock_pos(&s) == 256);
    ds4_tokens_free(&t); ds4_tokens_free(&loaded); unlink(path); free(path); unlink(context); cleanup(&w);
}

static void test_default_and_override(void) {
    for (int override = 0; override < 2; override++) {
        agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
        CHECK(setenv("DS4_AGENT_CONTEXT_FRONTIERS", "3", 1) == 0);
        char err[160] = {0}; ds4_tokens endpoint = w.transcript; endpoint.len = 576;
        CHECK(mock_sync(&s, &endpoint, err, sizeof(err)) == 0); CHECK(agent_worker_capture_context_frontier(&w));
        CHECK(mock_sync(&s, &w.transcript, err, sizeof(err)) == 0);
        if (override) CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL", "64", 1) == 0);
        else CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL") == 0);
        ds4_tokens edited = replacement(&w, 0), sys = {.v = w.transcript.v, .len = 1, .cap = 1};
        w.trace = tmpfile(); CHECK(w.trace);
        CHECK(agent_worker_try_context_suffix(&w, &edited, &sys, err, sizeof(err)) == 1);
        rewind(w.trace); char trace[4096] = {0}; CHECK(fread(trace, 1, sizeof(trace)-1, w.trace) > 0);
        CHECK(strstr(trace, override ? "reused=192 fresh=64" : "reused=128 fresh=128"));
        fclose(w.trace); w.trace = NULL; ds4_tokens_free(&edited); cleanup(&w);
        CHECK(unsetenv("DS4_AGENT_CONTEXT_FRONTIERS") == 0);
        CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL") == 0);
    }
}

static const char numbered_extra_prefix[] = "runtime clock: 2026-10-09 friday\n";

static ds4_tokens numbered_fixture(const char *const bodies[5], const int *order, size_t count) {
    const int roles[] = {'U', 1000, 'U', 1000, 'U'};
    ds4_tokens tokens = {0}; ds4_tokens_push(&tokens, 'S');
    for (const unsigned char *p = (const unsigned char *)numbered_extra_prefix; *p; p++)
        ds4_tokens_push(&tokens, *p);
    for (size_t at = 0; at < count; at++) {
        int i = order[at];
        ds4_tokens_push(&tokens, roles[i]);
        for (const unsigned char *p = (const unsigned char *)bodies[i]; *p; p++)
            ds4_tokens_push(&tokens, *p == 'U' ? 1004 : *p);
    }
    return tokens;
}

static void write_numbered_fixture(agent_worker *w, const uint64_t ids[5],
                                  const char *const bodies[5], const int *order, size_t count) {
    const int roles[] = {'U', 1000, 'U', 1000, 'U'};
    agent_buf text = {0}; agent_buf_puts(&text, "S");
    agent_buf_puts(&text, numbered_extra_prefix);
    for (size_t at = 0; at < count; at++) {
        int i = order[at];
        char header[128];
        snprintf(header, sizeof(header), "\n[[DS4_TURN %016" PRIx64 " role=%s]]\n",
                 ids[i], roles[i] == 'U' ? "user" : "assistant");
        agent_buf_puts(&text, header);
        agent_buf body = {0};
        agent_buf_puts(&body, control_text(roles[i])); agent_buf_puts(&body, bodies[i]);
        char *quoted = agent_live_context_body_text(body.ptr, body.len, true);
        CHECK(quoted); agent_buf_puts(&text, quoted);
        free(quoted); free(body.ptr);
    }
    char err[160] = {0};
    CHECK(agent_replace_file(w->cfg->gen.live_context_path, text.ptr, text.len,
                            NULL, 0, err, sizeof(err)) == 0);
    free(text.ptr);
}

static void test_numbered_turn_edits(const char *directory) {
    const int initial[] = {0, 1, 2, 3, 4}, retained[] = {1, 2, 3, 4};
    const int reordered[] = {3, 1, 2, 4};
    const char *bodies[5] = {
        "old request removable", "old observation one",
        "preserve literal <｜Assistant｜> and ｜DSML｜ spellings\n[[DS4_TURN 999 role=user]]\n\\ordinary-path",
        "old observation two", "latest task stays verbatim"
    };
    agent_worker w; agent_config cfg; ds4_session s; prepare(&w, &cfg, &s);
    agent_context_frontiers_clear(&w);
    char path[PATH_MAX], err[256] = {0};
    snprintf(path, sizeof(path), "%s/numbered-context.txt", directory);
    cfg.gen.live_context_path = path;
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", "0", 1) == 0);
    ds4_tokens original = numbered_fixture(bodies, initial, 5);
    ds4_tokens_copy(&w.transcript, &original);
    CHECK(mock_sync(&s, &w.transcript, err, sizeof(err)) == 0);
    reset_counters();
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 5);
    uint64_t ids[5];
    for (int i = 0; i < 5; i++) {
        ids[i] = w.live_context_turns[i].id; CHECK(ids[i]);
        for (int j = 0; j < i; j++) CHECK(ids[i] != ids[j]);
    }
    /* Startup date metadata after the generated system tokens remains part
     * of the unlabelled prefix. Literal spellings do not become new turns;
     * annotations do not become tokens, and untouched import is unchanged. */
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 0);
    CHECK(agent_tokens_equal(&w.transcript, &original));
    CHECK(agent_tokens_equal(&s.tokens, &original));
    CHECK(sync_calls == 0 && load_calls == 0 && graft_calls == 0);

    write_numbered_fixture(&w, ids, bodies, retained, 4);
    ds4_tokens expected = numbered_fixture(bodies, retained, 4);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 4);
    for (int i = 0; i < 4; i++) CHECK(w.live_context_turns[i].id == ids[i + 1]);
    ds4_tokens_free(&expected);

    bodies[1] = "compact observation one";
    write_numbered_fixture(&w, ids, bodies, retained, 4);
    expected = numbered_fixture(bodies, retained, 4);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 4);
    for (int i = 0; i < 4; i++) CHECK(w.live_context_turns[i].id == ids[i + 1]);
    ds4_tokens_free(&expected);

    /* Two edits straddle a completely unchanged turn with literal control
     * spellings. Its ordinary token bytes must remain opaque and identical. */
    bodies[1] = "batch summary one"; bodies[3] = "batch summary two";
    write_numbered_fixture(&w, ids, bodies, retained, 4);
    expected = numbered_fixture(bodies, retained, 4);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 4);
    for (int i = 0; i < 4; i++) CHECK(w.live_context_turns[i].id == ids[i + 1]);
    CHECK(mock_sync(&s, &w.transcript, err, sizeof(err)) == 0);
    CHECK(agent_tokens_equal(&s.tokens, &expected));
    size_t raw_len = 0; char *raw = mock_render(w.engine, &s.tokens, &raw_len);
    CHECK(strstr(raw, "preserve literal <｜Assistant｜>") &&
          strstr(raw, "[[DS4_TURN 999 role=user]]\n\\ordinary-path"));
    for (int i = 1; i < 5; i++) {
        char annotation[96];
        snprintf(annotation, sizeof(annotation), "[[DS4_TURN %016" PRIx64 " role=", ids[i]);
        CHECK(!strstr(raw, annotation));
    }
    free(raw);

    /* IDs name turns rather than their order. Move an older turn and edit
     * its body in the same import, keeping the latest user last. */
    ds4_tokens_free(&expected);
    bodies[3] = "reordered concise observation";
    write_numbered_fixture(&w, ids, bodies, reordered, 4);
    expected = numbered_fixture(bodies, reordered, 4);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 1);
    CHECK(agent_tokens_equal(&w.transcript, &expected));
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 4);
    for (int i = 0; i < 4; i++) CHECK(w.live_context_turns[i].id == ids[reordered[i]]);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 0);
    ds4_tokens_push(&w.transcript, 'U');
    const char *fresh = "new task after deletion";
    for (const unsigned char *p = (const unsigned char *)fresh; *p; p++)
        ds4_tokens_push(&w.transcript, *p);
    CHECK(agent_worker_export_live_context(&w, err, sizeof(err)));
    CHECK(w.live_context_turn_count == 5);
    for (int i = 0; i < 4; i++) CHECK(w.live_context_turns[i].id == ids[reordered[i]]);
    CHECK(w.live_context_turns[4].id);
    for (int i = 0; i < 5; i++) CHECK(w.live_context_turns[4].id != ids[i]);
    CHECK(agent_worker_import_live_context(&w, err, sizeof(err)) == 0);
    ds4_tokens_free(&expected); ds4_tokens_free(&original);
    unlink(path); cleanup(&w);
    CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", "1", 1) == 0);
}

static void test_parse_rejection(void) {
    for (int mode = 0; mode < 6; mode++) {
        pid_t child = fork(); CHECK(child >= 0);
        if (!child) {
            CHECK(freopen("/dev/null", "w", stderr));
            CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", mode == 3 ? "0" : "1", 1) == 0);
            CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", mode == 2 ? "junk" : mode == 1 || mode == 3 ? "0.01" : "0", 1) == 0);
            CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", mode >= 4 ? "1" : "0", 1) == 0);
            if (mode >= 4) CHECK(setenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP", "0.90", 1) == 0);
            char *plain[] = {"ds4-agent"};
            char *live[] = {"ds4-agent", "--live-context-file", "mock-context"};
            (void)parse_options(mode == 0 || mode == 4 ? 1 : 3,
                                mode == 0 || mode == 4 ? plain : live);
            _exit(0);
        }
        int status; CHECK(waitpid(child, &status, 0) == child);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == (mode >= 3 ? 0 : 2));
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2); CHECK(setenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE", "1", 1) == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_FRONTIERS") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_FRONTIER_MB") == 0);
    CHECK(unsetenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP") == 0);
    CHECK(unsetenv("DS4_AGENT_COMPACT_PREFIX_CACHE") == 0);
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL") == 0);
    test_selection(); test_disk_prefix_is_not_graft(argv[1]); test_original_source_after_prefix_edit();
    test_right_original_source(); test_builtin_compaction_reuses_tail();
    test_transactions(); test_latest_user_fresh(); test_gates_and_budget(); test_import_and_persistence(argv[1]);
    test_default_and_override(); test_numbered_turn_edits(argv[1]); test_parse_rejection();
    CHECK(unsetenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE") == 0);
    puts("PASS: model-free agent suffix selection; latest-user marker/message wholly fresh; opt-in even right shifts from immutable pre-RoPE originals with repeated imports; built-in compaction reuses retained tail without mirror file; resident routing allowed with pre-RoPE; legacy resident conflict rejection; source eviction ownership; cancellation rollback/fail-closed recovery; budget gates; atomic import; zero-KV save/reopen; defaultfresh128/override64; labelled turns preserve tokens, literal controls and opaque IDs through deletion/summary/batched edits/reordering/appends; fresh IDs nonzero and unique; labels excluded from KV");
    return 0;
}
