#define DS4_AGENT_DEFINE_PROMPTS 1
#include "ds4_agent_prompts.h"
#include "ds4_agent_summary_prompts.h"
#undef DS4_AGENT_DEFINE_PROMPTS
#include "ds4.h"
#include "ds4_agent_presentation.h"
#include "ds4_agent_stream.h"
#include "ds4_agent_tool_actions.h"
#include "ds4_agent_edit.h"
#include "ds4_agent_sessions.h"
#include "ds4_agent_history.h"
#include "ds4_agent_checkpoint.h"
#include "ds4_agent_executor.h"
#include "ds4_agent_context_runtime.h"
#include "ds4_agent_result.h"
#include "ds4_agent_communications.h"
#include "ds4_agent_turn.h"
#include "ds4_agent_summary.h"
#include "ds4_agent_lifecycle.h"
#include "ds4_agent_terminal.h"
#define write_all ds4_agent_ui_write
#define agent_input_buf_append ds4_agent_ui_input_append
#define agent_input_buf_take ds4_agent_ui_input_take
#define agent_input_buf_free ds4_agent_ui_input_free
#define stdout_is_tty ds4_agent_ui_tty
#define agent_format_user_prompt_echo ds4_agent_ui_echo_text
#define agent_echo_user_prompt ds4_agent_ui_echo
#define agent_format_ctx_size ds4_agent_ui_context_size
#define agent_progress_bar ds4_agent_ui_progress
#define build_prompt_text ds4_agent_ui_prompt
#define build_status_text ds4_agent_ui_status
#define build_footer_text ds4_agent_ui_footer
#define agent_prompt_queue_push ds4_agent_ui_queue_push
#define agent_prompt_queue_pop ds4_agent_ui_queue_pop
#define agent_prompt_queue_push_front ds4_agent_ui_queue_front
#define agent_prompt_queue_take_all ds4_agent_ui_queue_take
#define agent_prompt_queue_take_all_echo ds4_agent_ui_queue_echo
#define agent_prompt_queue_peek ds4_agent_ui_queue_peek
#define agent_prompt_queue_free ds4_agent_ui_queue_free
#define editor_get_terminal_size ds4_agent_ui_size
#define editor_read_stdin ds4_agent_ui_read
#define editor_take_queued_byte ds4_agent_ui_take_byte
#define editor_take_bare_escape ds4_agent_ui_take_escape
#define editor_replace_input ds4_agent_ui_replace
#define editor_note_output ds4_agent_ui_note
#define editor_start ds4_agent_ui_start
#define editor_stop ds4_agent_ui_stop
#define editor_hide ds4_agent_ui_hide
#define editor_show ds4_agent_ui_show
#define editor_set_prompt_status ds4_agent_ui_prompt_status
#define editor_flush_prompt_status ds4_agent_ui_flush
#define editor_write_scroll_output_preserve_prompt ds4_agent_ui_scroll
#define editor_write_async ds4_agent_ui_async
#define editor_cancel_input_with_hint ds4_agent_ui_cancel
#define editor_restore_terminal_layout ds4_agent_ui_restore
#define agent_format_welcome_banner ds4_agent_ui_welcome
#define editor_write_welcome_banner ds4_agent_ui_write_welcome
#define agent_kv_load_path ds4_agent_worker_load
#define agent_worker_reset_to_sysprompt ds4_agent_worker_reset
#define agent_worker_has_user_session ds4_agent_worker_has_user
#define agent_worker_needs_save ds4_agent_worker_needs_save
#define agent_worker_save_session_now ds4_agent_worker_save_now
#define agent_worker_save_session ds4_agent_worker_save
#define agent_worker_show_history ds4_agent_worker_history
#define agent_worker_switch_session ds4_agent_worker_switch
#define agent_numeric_think_prefix ds4_agent_worker_numeric_prefix
#define worker_apply_requested_think ds4_agent_worker_apply_thinking
#define agent_worker_wait_distributed_route ds4_agent_worker_wait_route
#define worker_run_deferred_save ds4_agent_worker_deferred_save
#define worker_run_deferred_compact ds4_agent_worker_deferred_compact
#define worker_main ds4_agent_worker_thread
#define agent_worker_init ds4_agent_worker_init
#define agent_worker_free ds4_agent_worker_free
#define agent_worker_build_system_tokens ds4_agent_runtime_build_system
#define agent_worker_compact_transcript ds4_agent_turn_compact_private
#define agent_worker_compact ds4_agent_result_compact
#define worker_run_turn ds4_agent_turn_run
#define worker_run_raw_prompt ds4_agent_turn_raw
#define agent_worker_sync_tokens ds4_agent_turn_sync
#define agent_build_tools_prompt ds4_agent_turn_tools_prompt
#define agent_build_dsml_tools_prompt ds4_agent_turn_dsml_prompt
#define agent_build_glm_tools_prompt ds4_agent_turn_glm_prompt
#define agent_build_qwen_tools_prompt ds4_agent_turn_qwen_prompt
#define agent_dsml41_tools_prompt ds4_agent_turn_dsml41_prompt
#define agent_hints_note ds4_agent_turn_hints_note
#define agent_hints_compacted ds4_agent_turn_hints_compacted
#define agent_worker_append_hints_context ds4_agent_turn_hints_context
#define agent_append_system_prompt ds4_agent_turn_append_system
#define agent_worker_maybe_append_datetime_context ds4_agent_turn_datetime
#define agent_worker_maybe_append_system_prompt_reminder ds4_agent_turn_system_reminder
#define agent_publish_system_status ds4_agent_turn_system_status
#define agent_trace_text ds4_agent_turn_trace_text
#define agent_trace_token ds4_agent_turn_trace_token
#define agent_trace_tokens ds4_agent_turn_trace_tokens

#define agent_wake_locked ds4_agent_comms_wake_locked
#define agent_publish ds4_agent_comms_publish
#define agent_set_status ds4_agent_comms_status
#define agent_set_error ds4_agent_comms_error
#define worker_progress_cb ds4_agent_comms_progress
#define worker_should_interrupt ds4_agent_comms_interrupted
#define worker_clear_interrupt ds4_agent_comms_clear_interrupt
#define worker_cancel_session_cb ds4_agent_comms_cancel
#define agent_web_confirm ds4_agent_comms_web_confirm
#define worker_take_web_approval_request ds4_agent_comms_take_web
#define worker_answer_web_approval ds4_agent_comms_answer_web
#define worker_request_queued_user_drain ds4_agent_comms_request_drain
#define worker_take_queued_user_drain_request ds4_agent_comms_take_drain
#define worker_answer_queued_user_drain ds4_agent_comms_answer_drain
#define worker_request_save ds4_agent_comms_request_save
#define worker_request_compact ds4_agent_comms_request_compact
#define worker_request_power ds4_agent_comms_request_power
#define worker_take_save_requested ds4_agent_comms_take_save
#define worker_take_compact_requested ds4_agent_comms_take_compact
#define worker_apply_pending_power ds4_agent_comms_apply_power
#define worker_check_raw_mode_restore ds4_agent_comms_take_raw_restore
#define drain_wake_fd ds4_agent_comms_drain_fd
#define worker_submit ds4_agent_comms_submit
#define worker_interrupt ds4_agent_comms_interrupt
#define worker_stop ds4_agent_comms_stop
#define worker_consume ds4_agent_comms_consume
#define worker_get_status ds4_agent_comms_get_status
#define worker_is_idle ds4_agent_comms_idle
#define worker_is_initialized ds4_agent_comms_initialized
#define agent_next_prefill_label ds4_agent_comms_next_label
#include "ds4_agent_context.h"
#include "ds4_agent_bash.h"
#include "ds4_agent_compaction.h"
#include "ds4_tool_text.h"
#include "ds4_agent_tools.h"
#include "ds4_agent_file.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_kvstore.h"
#include "ds4_state_cache_runtime.h"
#include "ds4_suffix_match.h"
#include "ds4_prompt_prefix.h"
#include "ds4_tp.h"
#include "ds4_web.h"
#include "linenoise.h"

#define agent_open_regular_file ds4_agent_file_open_regular
#define agent_read_file_bytes ds4_agent_file_read_bytes
#define agent_replace_file ds4_agent_file_replace
#define agent_find_unique ds4_agent_file_find_unique
#define agent_edit_find_old_span ds4_agent_file_find_old_span

#define agent_tool_call_free ds4_agent_tool_call_free
#define agent_tool_calls_free ds4_agent_tool_calls_free
#define agent_tool_call_add_arg ds4_agent_tool_call_add_arg
#define agent_tool_calls_push ds4_agent_tool_calls_push
#define agent_tool_arg_value ds4_agent_tool_arg_value
#define agent_dsml_parser_free ds4_agent_tool_parser_free
#define agent_dsml_parser_reset ds4_agent_tool_parser_reset
#define agent_dsml_start ds4_agent_tool_parser_start
#define agent_dsml_feed ds4_agent_tool_parser_feed
#define agent_dsml_finish ds4_agent_tool_parser_finish
#define agent_dsml_set_error ds4_agent_tool_parser_error
#define agent_tool_start ds4_agent_tool_start
#define agent_dsml_tag_name ds4_agent_tool_tag_name
#define agent_dsml_param_close ds4_agent_tool_parameter_close
#define agent_tool_value_close_tail ds4_agent_tool_value_close_tail

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <copyfile.h>
#elif defined(__linux__)
#include <sys/xattr.h>
#endif

extern char **environ;

/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

#include "ds4_agent_frontend.h"
#define parse_power_percent ds4_agent_frontend_power
#define parse_steering_level ds4_agent_frontend_steering
#define agent_parse_hints ds4_agent_frontend_hints
#define agent_slash_command_with_args ds4_agent_frontend_with_args
#define agent_slash_command_known ds4_agent_frontend_known
#define agent_apply_model_sampling_defaults ds4_agent_frontend_defaults
#define run_agent_non_interactive ds4_agent_frontend_headless
#define run_agent ds4_agent_frontend_interactive
/* One real OS signal action; storage and the write-only handler belong to
 * the sole native facade. Restore the exact caller mask/handler on return. */
bool ds4_agent_frontend_signal(bool install, void (*handler)(int),
                               struct sigaction *previous) {
    if (!install) return sigaction(SIGINT, previous, NULL) == 0;
    struct sigaction action = {0};
    sigemptyset(&action.sa_mask);
    action.sa_handler = handler;
    return sigaction(SIGINT, &action, previous) == 0;
}

static int set_nonblock(int fd, bool on, int *old_flags);
static bool agent_parse_bool_default(const char *s, bool def);
static int agent_parse_int_default(const char *s, int def, int min, int max);
static bool agent_context_suffix_requested(void) {
    return agent_parse_bool_default(getenv("DS4_AGENT_CONTEXT_SUFFIX_REUSE"), false);
}

static bool agent_context_prerope_requested(void) {
    return agent_context_suffix_requested() &&
           agent_parse_bool_default(getenv("DS4_AGENT_CONTEXT_SUFFIX_PREROPE"), false);
}

static bool agent_context_suffix_right_requested(void) {
    return agent_context_prerope_requested() &&
           agent_parse_bool_default(getenv("DS4_AGENT_CONTEXT_SUFFIX_RIGHT"), false);
}

static bool agent_context_suffix_resident_conflict(void) {
    if (agent_context_prerope_requested()) return false;
    const char *s = getenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP");
    if (!s || !s[0]) return false;
    char *end = NULL;
    double gap = strtod(s, &end);
    return !end || *end || !isfinite(gap) || gap != 0;
}

/* ============================================================================
 * Configuration, Worker State, And Streaming Types
 * ============================================================================
 *
 * The agent is intentionally a single process: the UI thread owns terminal
 * input/output, while the worker thread owns the live DS4 session and KV state.
 * These types define the shared state and the small streaming state machines
 * used to render sampled assistant text and DSML tool calls as they arrive.
 */

/* Actual Worker/config/status/frontier records live in ds4_agent_worker.h. */

static void agent_trace(agent_worker *w, const char *fmt, ...);
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr);
static void agent_web_log(void *privdata, const char *message);
static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
static bool agent_text_only_control_safe(agent_worker *w, const ds4_tokens *tokens,
                                         const char *text, size_t text_len,
                                         char *err, size_t err_len);
static int agent_read_default_lines(agent_worker *w);
static int agent_compact_reserve_tokens(agent_worker *w);

static agent_tool_syntax agent_tool_syntax_for_engine(ds4_engine *engine) {
    if (ds4_engine_is_qwen4(engine)) return AGENT_TOOL_SYNTAX_QWEN;
    return ds4_engine_is_glm_dsa(engine) ? AGENT_TOOL_SYNTAX_GLM
         : ds4_engine_is_deepseek41(engine) ? AGENT_TOOL_SYNTAX_DSML41
                                           : AGENT_TOOL_SYNTAX_DSML;
}


static bool agent_tool_syntax_assistant_turn_uses_eos(agent_tool_syntax syntax) {
    return syntax != AGENT_TOOL_SYNTAX_GLM;
}

/* GLM and Qwen both open a call with <tool_call> and render as chat messages */
static bool agent_syntax_is_xml_tool_call(agent_tool_syntax syntax) {
    return syntax == AGENT_TOOL_SYNTAX_GLM || syntax == AGENT_TOOL_SYNTAX_QWEN;
}

static __attribute__((unused)) void agent_worker_append_assistant_turn_end(agent_worker *w) {
    if (agent_tool_syntax_assistant_turn_uses_eos(
            agent_tool_syntax_for_engine(w->engine)))
        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
}

static int agent_worker_effective_ctx_size(const agent_worker *w) {
    int ctx = 0;
    if (w && w->session) ctx = ds4_session_ctx(w->session);
    if (ctx <= 0 && w && w->cfg) ctx = w->cfg->gen.ctx_size;
    return ctx;
}

/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */


static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4-agent: malloc");
        exit(1);
    }
    return p;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        perror("ds4-agent: realloc");
        exit(1);
    }
    return p;
}

static void agent_worker_images_clear(agent_worker *w) { ds4_agent_runtime_images_clear(w); }
static void agent_worker_images_append(agent_worker *w, ds4_vision_span *images, size_t count) {
    ds4_agent_runtime_images_append(w, images, count);
}
static bool agent_worker_images_fit_tokens(const agent_worker *w, const ds4_tokens *tokens) {
    return ds4_agent_runtime_images_fit(w, tokens);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static agent_config parse_options(int argc, char **argv) {
    return ds4_agent_frontend_options(argc, argv, default_backend());
}


static ds4_think_mode effective_think_mode(const agent_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

/* ============================================================================
 * System Prompt Rendering And Worker Output Queues
 * ============================================================================
 */


/* Original literal agent_tools_prompt_intro lives in ds4_agent_prompts.h. */


/* Original literal agent_tools_prompt_edit_exact lives in ds4_agent_prompts.h. */

/* Original literal agent_tools_prompt_edit_upto lives in ds4_agent_prompts.h. */

/* Original literal agent_tools_prompt_after_edit lives in ds4_agent_prompts.h. */

/* Original literal agent_vision_tool_schema lives in ds4_agent_prompts.h. */

/* the GLM schema list, each line wrapped as {"type": "function", "function": ...} */

/* The full instructions are sent once per live context, not on each enable. */


static void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}


/* Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe. */

/* Renderer emits bytes through the selected whole Worker publication. */
static void agent_presentation_emit(void *userdata, const char *text, size_t len) {
    agent_publish(userdata, text, len);
}

static void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish(w, stack, (size_t)n);
        return;
    }

    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish(w, heap, (size_t)n);
    free(heap);
}

/* ============================================================================
 * Trace Logging
 * ============================================================================
 */

static void agent_trace(agent_worker *w, const char *fmt, ...) {
    if (!w || !w->trace) return;
    char stack[1024]; va_list ap;
    va_start(ap, fmt); int n = vsnprintf(stack, sizeof(stack), fmt, ap); va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(stack)) { ds4_agent_turn_trace_message(w, stack); return; }
    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt); vsnprintf(heap, (size_t)n + 1, fmt, ap); va_end(ap);
    ds4_agent_turn_trace_message(w, heap); free(heap);
}


/* ============================================================================
 * DSML Tool-Call Parser
 * ============================================================================
 *
 * The model streams raw text tokens.  This parser recognizes completed DSML
 * tool stanzas and keeps a copy of the raw stanza for diagnostics.  It is
 * deliberately strict after the opening marker: typo recovery belongs to the
 * streaming detector so the actual tool parser stays small and predictable.
 */

static __attribute__((unused)) bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static __attribute__((unused)) bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}


/* ============================================================================
 * Assistant Markdown Rendering
 * ============================================================================
 *
 * This renderer handles only the cheap markdown cues that make terminal output
 * readable: **bold**, *italic*, inline code, hint asides, and fenced code blocks.
 * It is a streaming parser with a bounded buffer for ambiguous inline spans.
 */

/* Transitional C callers invoke whole renderer actions; no Worker fields cross. */
static __attribute__((unused)) void renderer_flush_utf8(agent_token_renderer *r) {
    ds4_agent_presentation_flush_utf8(r);
}
static __attribute__((unused)) void renderer_markdown_feed(agent_token_renderer *r, char ch) {
    ds4_agent_presentation_markdown_byte(r, (unsigned char)ch);
}
static __attribute__((unused)) void renderer_markdown_finish(agent_token_renderer *r) {
    ds4_agent_presentation_markdown_finish(r);
}
static __attribute__((unused)) void agent_tail_capture_append(agent_tail_capture *t, const char *s, size_t n) {
    ds4_agent_presentation_capture_append(t, s, n);
}
static char * agent_tail_capture_take(agent_tail_capture *t, size_t *n) {
    return ds4_agent_presentation_capture_take(t, n);
}
#ifdef DS4_AGENT_TEST
static const agent_syntax * agent_syntax_for_path(const char *path) {
    return ds4_agent_presentation_syntax_path(path);
}
#endif
static int renderer_terminal_cols(void) {
    return ds4_agent_presentation_columns();
}
#ifdef DS4_AGENT_TEST
static void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    ds4_agent_presentation_write(r, s, n);
}
static void renderer_write_char(agent_token_renderer *r, char ch) {
    ds4_agent_presentation_char(r, (unsigned char)ch);
}
static void renderer_restore_text_attrs(agent_token_renderer *r) {
    ds4_agent_presentation_restore(r);
}
static void renderer_reset_color(agent_token_renderer *r) {
    ds4_agent_presentation_reset(r);
}
static void renderer_hint_end(agent_token_renderer *r) {
    ds4_agent_presentation_hint_end(r);
}
#endif
static __attribute__((unused)) void renderer_process(agent_token_renderer *r, const char *s, size_t n, bool final) {
    ds4_agent_presentation_process(r, s, n, final);
}
static void renderer_finish(agent_token_renderer *r) {
    ds4_agent_presentation_finish(r);
}
#ifdef DS4_AGENT_TEST
static void renderer_color(agent_token_renderer *r, const char *color) {
    ds4_agent_presentation_color(r, color);
}
static void renderer_plain(agent_token_renderer *r, const char *s, size_t n) {
    ds4_agent_presentation_plain(r, s, n);
}
static void renderer_code_stream_begin(agent_token_renderer *r, const agent_syntax *syntax) {
    ds4_agent_presentation_code_begin(r, syntax);
}
static void renderer_code_stream_set_prefix(agent_token_renderer *r, const char *prefix, const char *color) {
    ds4_agent_presentation_code_prefix(r, prefix, color);
}
static void renderer_code_stream_set_upto_marker(agent_token_renderer *r, bool enabled) {
    ds4_agent_presentation_code_upto(r, enabled);
}
static void renderer_code_end(agent_token_renderer *r) {
    ds4_agent_presentation_code_end(r);
}
static void renderer_code_byte(agent_token_renderer *r, char ch) {
    ds4_agent_presentation_code_byte(r, (unsigned char)ch);
}
#endif

/* ============================================================================
 * Streaming Tool Visualization
 * ============================================================================
 *
 * Tool calls are parsed for execution later, but they are also visualized while
 * the model is still sampling.  This state machine suppresses raw DSML and
 * prints compact, tool-specific progress such as "$ command" or
 * "Reading file 1:500...".
 */

/* Whole existing Worker actions; the native projection owns no Worker fields. */
void ds4_agent_stream_trace(void *worker, const char *message) {
    agent_trace(worker, "%s", message);
}
int ds4_agent_stream_read_lines(void *worker) {
    return agent_read_default_lines(worker);
}
bool ds4_agent_stream_preflight(void *worker, const agent_tool_call *call,
        char *error, size_t error_size) {
    return agent_preflight_edit_old(worker, call, error, error_size);
}
static void agent_stream_text(agent_stream_renderer *stream,
        const char *text, size_t len, bool finish) {
    ds4_agent_stream_text(stream, text, len, finish);
}

/* ============================================================================
 * Worker Progress And Generic Buffers
 * ============================================================================
 */

static bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}


typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    size_t limit;
    bool truncated;
} agent_buf;


static void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = b->limit ? b->limit : SIZE_MAX - 1;
    if (n > max - b->len) {
        n = max > b->len ? max - b->len : 0;
        while (n && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

static char *agent_buf_take(agent_buf *b) {
    if (b->truncated) {
        b->truncated = false;
        b->limit = 0;
        agent_buf_puts(b, "\n[Output truncated at the tool byte limit. Narrow the request.]\n");
    }
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}


static bool agent_tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}

static __attribute__((unused)) bool agent_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static __attribute__((unused)) char *agent_default_cache_dir(void) {
    const char *override = getenv("DS4_AGENT_CACHE_DIR");
    if (override && override[0]) return xstrdup(override);
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    agent_buf b = {0};
    agent_buf_puts(&b, home);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') agent_buf_puts(&b, "/");
    agent_buf_puts(&b, ".ds4/kvcache");
    return agent_buf_take(&b);
}

static char *agent_kv_path_for_sha(const char *dir, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(dir, name);
}

static void agent_session_identity_sha(const char *title, uint64_t created_at, char sha_out[41]) {
    ds4_agent_sessions_identity_sha(title, created_at, sha_out);
}


static void agent_worker_clear_session_identity(agent_worker *w) {
    ds4_agent_sessions_identity_clear(&w->identity);
}


static void agent_kv_session_meta_free(agent_kv_session_meta *meta) {
    ds4_agent_sessions_meta_free(meta);
}


/* ============================================================================
 * Agent KV Store And Session Persistence
 * ============================================================================
 */

static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);

/* Agent sessions deliberately use a different policy from ds4-server:
 *
 * - sysprompt.kv is a fixed bootstrap checkpoint for the current tool/system
 *   prompt.  Because its name is fixed, the current rendered text is compared
 *   with the text stored in the file before loading.  A mismatch simply rebuilds
 *   and overwrites the file.
 * - conversation sessions are explicit saves only.  Their stable file name is
 *   SHA1(title || created_at_le64).kv, where title is the first user prompt and
 *   created_at is preserved across future saves.  The title is stored in an
 *   agent-only trailer after the KV payload.
 *
 * The DS4 payload stores the exact token sequence and graph state.  The rendered
 * text is retained for listing, history rendering, and stripped-session rebuilds. */

/* Cohesive saved-file codec; live payload restore/sync remains below. */
static bool agent_kv_read_text(FILE *fp, uint32_t bytes, char **out,
        char *err, size_t errlen) {
    return ds4_agent_sessions_read_text(fp, bytes, out, err, errlen);
}
static bool agent_kv_write_title_trailer(FILE *fp, const char *title,
        char *err, size_t errlen) {
    return ds4_agent_sessions_write_title(fp, title, err, errlen);
}
static bool agent_kv_read_title_trailer(FILE *fp, const ds4_kvstore_entry *header,
        char **out, char *err, size_t errlen) {
    return ds4_agent_sessions_read_title(fp, header, out, err, errlen);
}
static void agent_kv_identity_sha(const ds4_kvstore_entry *header,
        const char *text, uint32_t bytes, const char *title, char out[41]) {
    ds4_agent_sessions_file_sha(header, text, bytes, title, out);
}

/* Save the current live KV under the rendered transcript identity.  The caller
 * decides the policy: fixed sysprompt path or SHA-named session path. */
static __attribute__((unused)) bool agent_worker_text_only_session(agent_worker *w, const ds4_tokens *tokens) {
    return tokens->len >= agent_worker_effective_ctx_size(w) ||
           ds4_session_is_approximate(w->session);
}

static bool agent_kv_save_path(agent_worker *w, const char *path,
        const ds4_tokens *tokens, const char *reason, char sha_out[41],
        const char *session_title, uint64_t session_created_at,
        char *err, size_t errlen) {
    const bool text_only = session_title != NULL &&
        agent_worker_text_only_session(w, tokens);
    return ds4_agent_checkpoint_save(w->engine, w->session, path, tokens,
        reason, sha_out, session_title, session_created_at, text_only,
        err, errlen);
}

static __attribute__((unused)) void agent_publishf_system_status(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish_system_status(w, stack);
        return;
    }

    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish_system_status(w, heap);
    free(heap);
}


static void agent_web_log(void *privdata, const char *message) {
    agent_worker *w = privdata;
    if (!w || !message || !message[0]) return;
    agent_trace(w, "web: %s", message);
}

static bool agent_web_cancel(void *privdata) {
    return worker_should_interrupt(privdata);
}

/* Synchronize the live DS4 session to a transcript.  This is the agent's main
 * cache-saving operation: if the requested transcript extends the live session,
 * only the suffix is prefetched; otherwise the DS4 session rebuilds from the
 * longest common prefix it can retain. */
static bool agent_state_cache_allowed(agent_worker *w) { return ds4_agent_runtime_cache_allowed(w); }
static __attribute__((unused)) bool agent_state_cache_materialize(agent_worker *w, ds4_state_cache_ref *ref,
        ds4_suffix_cache *out, ds4_state_cache_kind *kind, char *err, size_t len) {
    return ds4_agent_runtime_cache_materialize(w, ref, out, kind, err, len);
}
static __attribute__((unused)) bool agent_state_cache_put_source(agent_worker *w, const ds4_suffix_cache *source, char key[41]) {
    return ds4_agent_runtime_cache_put(w, source, key);
}
static void agent_state_cache_store_endpoint(agent_worker *w) { ds4_agent_runtime_cache_endpoint(w); }
static bool agent_state_cache_restore_prefix(agent_worker *w, const ds4_tokens *tokens) { return ds4_agent_runtime_cache_restore(w, tokens); }

int ds4_agent_runtime_sync(agent_worker *w, const ds4_tokens *tokens, bool progress, char *err, size_t len) {
    return agent_worker_sync_tokens(w, tokens, progress, err, len);
}

/* Experimental exact-prefix restoration for compaction. Reuse the existing
 * validated bootstrap payload instead of replaying the system/tool prefix.
 * The suffix still runs normally; no stale states survive from the discarded
 * conversation. Cold replay can choose different floating-point partitions,
 * so this stays opt-in until task-level validation covers that difference. */
static bool agent_worker_restore_compact_prefix(agent_worker *w, const ds4_tokens *sys) { return ds4_agent_runtime_compact_prefix(w, sys); }
static __attribute__((unused)) void agent_context_frontier_drop(agent_worker *w, int index) { ds4_agent_runtime_frontier_drop(w, index); }
static void agent_context_frontiers_clear(agent_worker *w) { ds4_agent_runtime_frontiers_clear(w); }
static __attribute__((unused)) int agent_context_frontier_slots(agent_worker *w, uint64_t *budget) { return ds4_agent_runtime_frontier_slots(w, budget); }
static bool agent_worker_capture_context_frontier(agent_worker *w) { return ds4_agent_runtime_capture(w); }
static bool agent_worker_restore_context_frontier(agent_worker *w, const ds4_tokens *tokens) { return ds4_agent_runtime_restore(w, tokens); }


static void agent_live_context_clear(agent_worker *w) { ds4_agent_runtime_context_clear(w); }
static __attribute__((unused)) char *agent_live_context_body_text(const char *text, size_t len, bool quote) {
    return ds4_agent_context_body_text(text, len, quote);
}
static char *agent_live_context_render(agent_worker *w, size_t *length) { return ds4_agent_runtime_render(w, length); }


/* Mirroring is text-only. With suffix reuse disabled, context-file edits
 * restore an exact frontier and the next sync evaluates the changed suffix. */
static bool agent_worker_export_live_context(agent_worker *w, char *err, size_t len) {
    return ds4_agent_runtime_export(w, err, len);
}


static bool agent_text_only_control_safe(agent_worker *w,
        const ds4_tokens *tokens, const char *text, size_t len,
        char *err, size_t errlen) {
    return ds4_agent_context_text_safe(w->engine, tokens, text, len, err, errlen);
}


static int agent_worker_try_context_suffix(agent_worker *w, const ds4_tokens *edited,
        const ds4_tokens *sys, char *err, size_t len) {
    return ds4_agent_runtime_try_suffix(w, edited, sys, err, len);
}


static int agent_worker_import_live_context(agent_worker *w, char *err, size_t len) {
    return ds4_agent_runtime_import(w, err, len);
}

/* ============================================================================
 * Session Listing, History Rendering, And Completion
 * ============================================================================
 */

static char *agent_session_title_from_prompt(const char *prompt, size_t budget) {
    return ds4_agent_sessions_title_prompt(prompt, budget);
}
static char *agent_session_title_from_text(const char *text, size_t bytes, size_t budget) {
    return ds4_agent_sessions_title_text(text, bytes, budget);
}


#define AGENT_HISTORY_DEFAULT_TURNS DS4_AGENT_HISTORY_DEFAULT_TURNS
#define AGENT_HISTORY_MAX_TURNS DS4_AGENT_HISTORY_MAX_TURNS
#define AGENT_HISTORY_ASSISTANT_MAX_LINES DS4_AGENT_HISTORY_ASSISTANT_MAX_LINES
#define AGENT_HISTORY_ASSISTANT_MAX_BYTES DS4_AGENT_HISTORY_ASSISTANT_MAX_BYTES

/* Whole borrowed-text presentation. Idle checks and live token rendering
 * remain in the existing Worker action below. */
static void agent_history_render_text(agent_worker *w, const char *text,
        size_t len, int user_turns) {
    ds4_agent_history_render(w->engine, agent_tool_syntax_for_engine(w->engine),
        w, agent_presentation_emit, isatty(STDOUT_FILENO) != 0,
        text, len, user_turns);
}

/* Whole catalog actions take actual directory/model inputs, never Worker. */
static void agent_worker_list_sessions(agent_worker *w) {
    ds4_agent_sessions_list(w->cache_dir, (uint8_t)ds4_engine_model_id(w->engine),
        renderer_terminal_cols());
}
static bool agent_worker_find_session(agent_worker *w, const char *prefix,
        char sha[41], char **path, char *err, size_t errlen) {
    return ds4_agent_sessions_find(w->cache_dir,
        (uint8_t)ds4_engine_model_id(w->engine), prefix, sha, path, err, errlen);
}
static bool agent_worker_delete_session(agent_worker *w, const char *prefix,
        char sha[41], char *err, size_t errlen) {
    return ds4_agent_sessions_delete(w->cache_dir,
        (uint8_t)ds4_engine_model_id(w->engine), prefix, sha, err, errlen);
}
static bool agent_worker_strip_session(agent_worker *w, const char *prefix,
        char sha[41], uint32_t *tokens, char *err, size_t errlen) {
    return ds4_agent_sessions_strip(w->cache_dir,
        (uint8_t)ds4_engine_model_id(w->engine), w->engine,
        prefix, sha, tokens, err, errlen);
}

/* ============================================================================
 * Tool Argument Parsing And File Tool Helpers
 * ============================================================================
 */

static __attribute__((unused)) int agent_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}

static int agent_parse_int_default(const char *s, int def, int min, int max) {
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return def;
    if (v < min) v = min;
    if (v > max) v = max;
    return (int)v;
}

static bool agent_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}

#define AGENT_FILE_MAX_BYTES DS4_AGENT_FILE_MAX_BYTES
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_COMPACT_SOFT_PERCENT DS4_AGENT_COMPACT_SOFT_PERCENT
#define AGENT_COMPACT_MIN_FREE_TOKENS DS4_AGENT_COMPACT_MIN_FREE_TOKENS
#define AGENT_COMPACT_TAIL_DIVISOR DS4_AGENT_COMPACT_TAIL_DIVISOR
#define AGENT_COMPACT_TAIL_CAP_TOKENS DS4_AGENT_COMPACT_TAIL_CAP_TOKENS
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS DS4_AGENT_COMPACT_SUMMARY_MAX_TOKENS

static __attribute__((unused)) int agent_tool_result_reserve_tokens(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    int reserve = AGENT_TOOL_RESULT_RESERVE_TOKENS;
    if (ctx > 0) {
        int proportional = ctx / 8;
        if (proportional < 16) proportional = 16;
        if (reserve > proportional) reserve = proportional;
    }
    int compact = agent_compact_reserve_tokens(w) + 128;
    return reserve > compact ? reserve : compact;
}

static int agent_read_default_lines(agent_worker *w) {
    return ds4_agent_read_default_lines(agent_worker_effective_ctx_size(w));
}


/* Read file text for the model.  Normal mode shows plain line numbers.  Raw
 * mode is reserved for cases where line decoration would corrupt the payload
 * being inspected. */
static __attribute__((unused)) char *agent_read_range_from(agent_worker *w, const char *path,
        int start, int count, bool whole, bool bare, bool set_more,
        off_t offset, bool mid_line) {
    return ds4_agent_read_range(&w->tools.pager, agent_worker_effective_ctx_size(w),
        path, start, count, whole, bare, set_more, offset, mid_line);
}
static __attribute__((unused)) char *agent_read_range(agent_worker *w, const char *path,
        int start, int count, bool whole, bool bare, bool set_more) {
    return agent_read_range_from(w, path, start, count, whole, bare, set_more,
        0, false);
}
static char *agent_tool_read(agent_worker *w, const agent_tool_call *call) {
    return ds4_agent_tool_read(&w->tools.pager, agent_worker_effective_ctx_size(w), call);
}
static char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    return ds4_agent_tool_more(&w->tools.pager, agent_worker_effective_ctx_size(w), call);
}


static char *agent_tool_write(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    return ds4_agent_edit_write(call);
}

static char *agent_tool_list(const agent_tool_call *call) {
    return ds4_agent_tool_list(call);
}


/* ============================================================================
 * Edit And Search Tools
 * ============================================================================
 */

static bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
        agent_dsml_parser *parser, const char *text, size_t len) {
    return ds4_agent_edit_force_upto(forcer, parser, text, len);
}

#ifdef DS4_AGENT_TEST
static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static void test_agent_edit_upto_tail_newline_is_not_part_of_anchor(void) {
    const char *data =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "clean:\n"
        "\trm -f bc\n";
    const char *old =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "[upto]\n"
        "clean:\n";

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};
    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old, false,
                                               &match, &match_len, &anchored,
                                               err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "not found") != NULL);
    err[0] = '\0';
    AGENT_TEST_ASSERT(agent_edit_find_old_span(data, strlen(data), old, true,
                                              &match, &match_len, &anchored,
                                              err, sizeof(err)));
    AGENT_TEST_ASSERT(anchored);
    AGENT_TEST_ASSERT(match == data);
    AGENT_TEST_ASSERT(match_len == strlen(data) - strlen("\trm -f bc\n"));
}

static void test_agent_edit_upto_requires_tail_after_newline_strip(void) {
    const char *data = "head\nbody\ntail\n";
    const char *old = "head\n[upto]\n";
    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};

    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old, true,
                                               &match, &match_len, &anchored,
                                               err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "must include a unique tail anchor") != NULL);
}

static char *agent_test_stream_capture(agent_tool_syntax syntax,
                                       const char **chunks,
                                       size_t chunk_count,
                                       agent_dsml_parser *p,
                                       bool *dsml_in_think) {
    agent_tail_capture tail = {
        .cap = 16384,
    };
    agent_token_renderer renderer = {
        .format_thinking = true,
        .format_markdown = false,
        .last_output_newline = true,
        .capture = &tail,
    };
    *p = (agent_dsml_parser){
        .syntax = syntax,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = p,
        .syntax = syntax,
    };

    for (size_t i = 0; i < chunk_count; i++)
        agent_stream_text(&stream, chunks[i], strlen(chunks[i]), false);
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    if (dsml_in_think) *dsml_in_think = stream.dsml_in_think;
    return agent_tail_capture_take(&tail, NULL);
}

static void test_agent_qwen_tool_parser_chunked_multi_arg(void) {
    const char *a = "<tool_call>\n<function=bash>\n<parameter=command>\nprintf hi\n";
    const char *b = "</param";
    const char *c = "eter>\n<parameter=refresh_sec>\n1\n</parameter>\n</function>\n</tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_QWEN,
        .state = AGENT_DSML_SEARCH,
    };
    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 2);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"), "printf hi"));
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].is_string);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"), "1"));
    AGENT_TEST_ASSERT(!p.calls.v[0].args[1].is_string);
    agent_dsml_parser_free(&p);
}

static void test_agent_qwen_tool_parser_two_calls_and_error(void) {
    const char *text =
        "<tool_call>\n<function=list>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_QWEN,
        .state = AGENT_DSML_SEARCH,
    };
    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 2);
    AGENT_TEST_ASSERT(!strcmp(p.calls.v[1].name, "read"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[1], "path"), "/tmp/x"));
    agent_dsml_parser_free(&p);

    const char *bad = "<tool_call>\n<function=list>\n<parameter=path>\n.\n</parameter>\n</tool_call>";
    agent_dsml_parser q = {
        .syntax = AGENT_TOOL_SYNTAX_QWEN,
        .state = AGENT_DSML_SEARCH,
    };
    agent_dsml_feed(&q, bad, strlen(bad));
    agent_dsml_finish(&q);
    AGENT_TEST_ASSERT(q.state == AGENT_DSML_ERROR);
    agent_dsml_parser_free(&q);
}

static void test_agent_qwen_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "intro ",
        "<to",
        "ol_call>\n<function=bash>\n<parameter=command>\nprintf hi\n</param",
        "eter>\n<parameter=refresh_sec>\n1\n</parameter>\n</function>\n</tool_call>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_QWEN, chunks,
                                          sizeof(chunks)/sizeof(chunks[0]), &p, NULL);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"), "printf hi"));
    AGENT_TEST_ASSERT(strstr(out, "intro ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "<parameter=") == NULL);
    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_qwen_argument_markers_bytewise(void) {
    const char *raw = "<tool_call><function=write><parameter=content>\n"
        "literal </tool_call> </think> <think>\n"
        "</parameter></function></tool_call>";
    agent_dsml_parser p = {.syntax = AGENT_TOOL_SYNTAX_QWEN, .state = AGENT_DSML_SEARCH};
    for (size_t i = 0; raw[i]; i++) {
        agent_dsml_feed(&p, raw + i, 1);
        AGENT_TEST_ASSERT(p.state != AGENT_DSML_ERROR);
        if (p.state == AGENT_DSML_ERROR) break;
    }
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    if (p.calls.len == 1)
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"),
            "literal </tool_call> </think> <think>"));
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_single_arg(void) {
    const char *text =
        "prose before <tool_call>list"
        "<arg_key>path</arg_key><arg_value>.</arg_value>"
        "</tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].name &&
                      !strcmp(p.calls.v[0].args[0].name, "path"));
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].value &&
                      !strcmp(p.calls.v[0].args[0].value, "."));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_chunked_multi_arg(void) {
    const char *a = "<tool_call>bash<arg_key>command</arg_key>";
    const char *b = "<arg_value>printf hi</arg_value>";
    const char *c =
        "<arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 2);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_streams_param_state(void) {
    const char *a =
        "<tool_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    const char *b = "</arg";
    const char *c =
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.current.name && !strcmp(p.current.name, "bash"));
    AGENT_TEST_ASSERT(p.param_name && !strcmp(p.param_name, "command"));
    AGENT_TEST_ASSERT(!p.param_close_prefix);

    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.param_close_prefix);

    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_multiple_adjacent_calls(void) {
    const char *text =
        "<tool_call>list<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>\n"
        "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 2);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "."));
    AGENT_TEST_ASSERT(p.calls.v[1].name && !strcmp(p.calls.v[1].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[1], "command"), "pwd"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "intro ",
        "<to",
        "ol_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg",
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(strstr(out, "intro ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "printf hi") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "<arg_key>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_tool_argument_literal_markup(void) {
    const char *glm[] = {
        "<tool_call>write<arg_key>content</arg_key><arg_value><p>&amp; &lt;</p> </tool_call> </think> ",
        "&lt;/arg_value> &amp;lt;/arg_value></arg_value></tool_call>",
    };
    const char *deepseek[] = {
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write\"><｜DSML｜parameter name=\"content\" string=\"true\"><p>&amp; &lt;</p> </tool_call> </think> ",
        "&lt;/｜DSML｜parameter> &amp;lt;/｜DSML｜parameter></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
    };
    const char *qwen[] = {
        "<tool_call>\n<function=write>\n<parameter=content>\n<p>&amp; &lt;</p> ",
        "&lt;/parameter> &amp;lt;/parameter>\n</parameter>\n</function>\n</tool_call>",
    };
    const char *expected[] = {
        "<p>&amp; &lt;</p> </tool_call> </think> </｜DSML｜parameter> &lt;/｜DSML｜parameter>",
        "<p>&amp; &lt;</p> </tool_call> </think> </arg_value> &lt;/arg_value>",
        "<p>&amp; &lt;</p> </parameter> &lt;/parameter>",
    };
    for (int is_glm = 0; is_glm <= 2; is_glm++) {
        agent_dsml_parser p;
        char *out = agent_test_stream_capture(
            is_glm == 2 ? AGENT_TOOL_SYNTAX_QWEN : is_glm ? AGENT_TOOL_SYNTAX_GLM : AGENT_TOOL_SYNTAX_DSML,
            is_glm == 2 ? qwen : is_glm ? glm : deepseek, 2, &p, NULL);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
        AGENT_TEST_ASSERT(p.calls.len == 1);
        if (p.calls.len)
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"), expected[is_glm]));
        free(out);
        agent_dsml_parser_free(&p);
    }
}

static void test_agent_glm_stream_ignores_tool_inside_think(void) {
    const char *chunks[] = {
        "<think>plan <tool",
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg_value></tool_call>",
        "</think>done",
    };
    agent_dsml_parser p;
    bool dsml_in_think = false;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          &dsml_in_think);

    AGENT_TEST_ASSERT(dsml_in_think);
    AGENT_TEST_ASSERT(p.calls.len == 0);
    AGENT_TEST_ASSERT(strstr(out, "[tool call ignored:") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_greedy_sampling_boundaries(void) {
    agent_tail_capture tail = {
        .cap = 4096,
    };
    agent_token_renderer renderer = {
        .format_thinking = true,
        .format_markdown = false,
        .last_output_newline = true,
        .capture = &tail,
    };
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &p,
        .syntax = AGENT_TOOL_SYNTAX_GLM,
    };

    agent_stream_text(&stream, "<tool", strlen("<tool"), false);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *body =
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    agent_stream_text(&stream, body, strlen(body), false);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));

    agent_stream_text(&stream, "</arg", strlen("</arg"), false);
    AGENT_TEST_ASSERT(p.param_close_prefix);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *close = "_value></tool_call>";
    agent_stream_text(&stream, close, strlen(close), false);
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    char *out = agent_tail_capture_take(&tail, NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_dsml_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "<｜D",
        "SML｜tool_calls><｜DSML｜invoke name=\"read\">"
        "<｜DSML｜parameter name=\"path\" string=\"true\">ds4_agent.c</｜DSML｜parameter>"
        "</｜DSML｜invoke></｜DSML｜tool_calls>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_DSML,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "read"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"),
                              "ds4_agent.c"));
    AGENT_TEST_ASSERT(strstr(out, "Reading ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "ds4_agent.c") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "DSML") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_rejects_missing_value(void) {
    const char *text = "<tool_call>list<arg_key>path</arg_key></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(strstr(p.error, "expected <arg_value>") != NULL);

    agent_dsml_parser_free(&p);
}

static void test_agent_edit_upto_prompt_is_opt_in(void) {
    char *dsml_default = agent_build_dsml_tools_prompt(false, false);
    char *dsml_upto = agent_build_dsml_tools_prompt(true, false);
    char *glm_default = agent_build_glm_tools_prompt(false, false);
    char *glm_upto = agent_build_glm_tools_prompt(true, false);

    AGENT_TEST_ASSERT(strstr(dsml_default, "[upto]") == NULL);
    AGENT_TEST_ASSERT(strstr(dsml_upto, "[upto]") != NULL);
    AGENT_TEST_ASSERT(strstr(glm_default, "[upto]") == NULL);
    AGENT_TEST_ASSERT(strstr(glm_upto, "[upto]") != NULL);

    free(dsml_default);
    free(dsml_upto);
    free(glm_default);
    free(glm_upto);
}

static void test_agent_glm_tools_prompt_is_native(void) {
    char *prompt = agent_build_glm_tools_prompt(false, true);

    AGENT_TEST_ASSERT(strstr(prompt, "<tools>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<tool_call>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<arg_key>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"name\":\"bash\"") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"command\"") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"type\":\"function\"") == NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"function\":") == NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "put path first") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "Emit at most one") == NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<｜DSML｜") == NULL);

    free(prompt);
}

static void test_agent_glm_template_policy(void) {
    AGENT_TEST_ASSERT(!agent_tool_syntax_assistant_turn_uses_eos(
        AGENT_TOOL_SYNTAX_GLM));
    AGENT_TEST_ASSERT(agent_tool_syntax_assistant_turn_uses_eos(
        AGENT_TOOL_SYNTAX_DSML));
    AGENT_TEST_ASSERT(!strcmp(ds4_glm_reasoning_effort_text(DS4_THINK_HIGH),
                              "Reasoning Effort: High"));
    AGENT_TEST_ASSERT(!strcmp(ds4_glm_reasoning_effort_text(DS4_THINK_MAX),
                              "Reasoning Effort: Max"));
    AGENT_TEST_ASSERT(ds4_glm_reasoning_effort_text(DS4_THINK_NONE) == NULL);
}

static void test_agent_read_default_lines_follow_context(void) {
    agent_config cfg = {0};
    agent_worker w = {
        .cfg = &cfg,
    };

    cfg.gen.ctx_size = 4096;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_SMALL);

    cfg.gen.ctx_size = AGENT_READ_SMALL_CONTEXT_MAX + 1;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_MEDIUM);

    cfg.gen.ctx_size = AGENT_READ_MEDIUM_CONTEXT_MAX + 1;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_LARGE);
}

static void test_agent_cache_rejects_impossible_lengths(void) {
    FILE *fp = tmpfile();
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;

    AGENT_TEST_ASSERT(fputc('x', fp) != EOF);
    rewind(fp);
    char *text = NULL;
    char err[128] = {0};
    AGENT_TEST_ASSERT(!agent_kv_read_text(fp, UINT32_MAX, &text,
                                         err, sizeof(err)));
    AGENT_TEST_ASSERT(text == NULL);
    AGENT_TEST_ASSERT(strstr(err, "truncated cached text") != NULL);
    fclose(fp);

    fp = tmpfile();
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, UINT32_MAX);
    AGENT_TEST_ASSERT(fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb));
    rewind(fp);
    ds4_kvstore_entry hdr = {0};
    char *title = NULL;
    err[0] = '\0';
    AGENT_TEST_ASSERT(!agent_kv_read_title_trailer(fp, &hdr, &title,
                                                   err, sizeof(err)));
    AGENT_TEST_ASSERT(title == NULL);
    AGENT_TEST_ASSERT(strstr(err, "truncated agent session title trailer") != NULL);
    AGENT_TEST_ASSERT(ftello(fp) == 0);
    fclose(fp);
}

/* Exercise the selected owner's same payload/TP decision, not the old body. */
static bool agent_kv_payload_requires_rebuild(agent_worker *w, uint64_t bytes) {
    return ds4_agent_checkpoint_requires_rebuild(bytes,
        w && w->cfg && ds4_tp_enabled(&w->cfg->engine.tp));
}

static void test_agent_tp_cache_payload_rebuild_policy(void) {
    agent_config cfg = {0};
    agent_worker w = {
        .cfg = &cfg,
    };

    AGENT_TEST_ASSERT(agent_kv_payload_requires_rebuild(&w, 0));
    AGENT_TEST_ASSERT(!agent_kv_payload_requires_rebuild(&w, 1));
    cfg.engine.tp.role = DS4_TP_LEADER;
    AGENT_TEST_ASSERT(agent_kv_payload_requires_rebuild(&w, 1));
}

static void test_agent_steering_command(void) {
    float scale = 0.0f;
    AGENT_TEST_ASSERT(parse_steering_level("-100", &scale) && scale == -100.0f);
    AGENT_TEST_ASSERT(parse_steering_level("0", &scale) && scale == 0.0f);
    AGENT_TEST_ASSERT(parse_steering_level("1.25", &scale) && scale == 1.25f);
    AGENT_TEST_ASSERT(parse_steering_level("100", &scale) && scale == 100.0f);
    AGENT_TEST_ASSERT(!parse_steering_level("", &scale));
    AGENT_TEST_ASSERT(!parse_steering_level("101", &scale));
    AGENT_TEST_ASSERT(!parse_steering_level("nan", &scale));
    AGENT_TEST_ASSERT(!parse_steering_level("1x", &scale));
    AGENT_TEST_ASSERT(agent_slash_command_known("/steer"));
    AGENT_TEST_ASSERT(agent_slash_command_known("/steer 1"));
    AGENT_TEST_ASSERT(!agent_slash_command_known("/steering"));
}

static void test_agent_terminal_wrap_output_is_deferred(void);
static void test_agent_idle_context_rejection_precedes_user_append(void);

static void test_agent_hints_state(void) {
    bool enabled = false;
    AGENT_TEST_ASSERT(agent_parse_hints("on", &enabled) && enabled);
    AGENT_TEST_ASSERT(agent_parse_hints("off", &enabled) && !enabled);
    AGENT_TEST_ASSERT(!agent_parse_hints("", &enabled));
    AGENT_TEST_ASSERT(!agent_parse_hints("toggle", &enabled));
    AGENT_TEST_ASSERT(!agent_parse_hints("on off", &enabled));
    AGENT_TEST_ASSERT(agent_slash_command_known("/hints on"));
    AGENT_TEST_ASSERT(!agent_slash_command_known("/hintson"));

    agent_hints h = {0};
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = true;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = false;
    const char *note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = true;
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && note != agent_hints_prompt);
    AGENT_TEST_ASSERT(note && strstr(note, "enabled"));

    agent_hints_compacted(&h);
    AGENT_TEST_ASSERT(h.enabled && !h.instructed);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);
    h.enabled = false;
    agent_hints_compacted(&h);
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    AGENT_TEST_ASSERT(!h.instructed);
    agent_hints_compacted(&h);
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));

    /* A restored session must override old on notes without restoring a setting. */
    h = (agent_hints){.applied = AGENT_HINTS_UNKNOWN};
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(!h.enabled && !h.instructed);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    h.enabled = true;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);

    /* Multiple changes before a safe point coalesce to the latest request. */
    h = (agent_hints){0};
    h.enabled = true;
    h.enabled = false;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    agent_hints_compacted(&h);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);

    char *dsml = agent_build_dsml_tools_prompt(false, false);
    char *glm = agent_build_glm_tools_prompt(false, false);
    AGENT_TEST_ASSERT(strstr(dsml, "programming hints") == NULL);
    AGENT_TEST_ASSERT(strstr(glm, "programming hints") == NULL);
    free(dsml);
    free(glm);
}

static void ds4_agent_unit_tests_run(void) {
    test_agent_hints_state();
    test_agent_edit_upto_tail_newline_is_not_part_of_anchor();
    test_agent_edit_upto_requires_tail_after_newline_strip();
    test_agent_cache_rejects_impossible_lengths();
    test_agent_tp_cache_payload_rebuild_policy();
    test_agent_steering_command();
    test_agent_read_default_lines_follow_context();
    test_agent_glm_template_policy();
    test_agent_edit_upto_prompt_is_opt_in();
    test_agent_glm_tools_prompt_is_native();
    test_agent_glm_tool_parser_single_arg();
    test_agent_glm_tool_parser_chunked_multi_arg();
    test_agent_glm_tool_parser_streams_param_state();
    test_agent_glm_tool_parser_multiple_adjacent_calls();
    test_agent_glm_stream_tool_call_chunked();
    test_agent_tool_argument_literal_markup();
    test_agent_qwen_tool_parser_chunked_multi_arg();
    test_agent_qwen_tool_parser_two_calls_and_error();
    test_agent_qwen_stream_tool_call_chunked();
    test_agent_qwen_argument_markers_bytewise();
    test_agent_glm_stream_ignores_tool_inside_think();
    test_agent_glm_stream_greedy_sampling_boundaries();
    test_agent_dsml_stream_tool_call_chunked();
    test_agent_glm_tool_parser_rejects_missing_value();
    test_agent_terminal_wrap_output_is_deferred();
    test_agent_idle_context_rejection_precedes_user_append();
}
#endif

/* Worker/config ownership remains here; each action is otherwise native. */
static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
        char *err, size_t err_len) {
    return ds4_agent_edit_preflight(w && w->cfg && w->cfg->edit_upto,
        call, err, err_len);
}
static char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    return ds4_agent_edit_apply(w && w->cfg && w->cfg->edit_upto, call);
}


static char *agent_tool_search(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    return ds4_agent_tool_search(call);
}


/* ============================================================================
 * Browser Web Tools
 * ============================================================================
 *
 * The browser subsystem lives in src/web/: it owns visible Chrome and CDP. The
 * agent side only asks for permission, dispatches tools, and caps visit_page
 * output using the same "head plus temp file" shape as bash.
 */


/* ============================================================================
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report the total captured output size.
 */

#define AGENT_BASH_HEAD_BYTES DS4_AGENT_BASH_HEAD_BYTES
#define AGENT_BASH_HEAD_LINES DS4_AGENT_BASH_HEAD_LINES
#define AGENT_BASH_TAIL_BYTES DS4_AGENT_BASH_TAIL_BYTES
#define AGENT_BASH_PROGRESS_TAIL_LINES DS4_AGENT_BASH_PROGRESS_TAIL_LINES
#define AGENT_BASH_FINAL_TAIL_LINES DS4_AGENT_BASH_FINAL_TAIL_LINES

static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}


/* Header adds the actual observation record and opaque process owner. C keeps
 * only list linkage; no native private job/mutex/fd layout is exposed. */
static void agent_bash_terminal_changed(void *userdata) {
    agent_worker_note_terminal_mode_may_have_changed(userdata);
}
static ds4_agent_bash_snapshot agent_bash_snapshot(agent_bash_job *job) {
    ds4_agent_bash_snapshot snapshot;
    ds4_agent_bash_get_snapshot(job->process, false, &snapshot);
    return snapshot;
}
static bool agent_bash_is_running(agent_bash_job *job) {
    return ds4_agent_bash_running(job->process);
}
static void agent_bash_signal(agent_bash_job *job, int number) {
    ds4_agent_bash_signal(job->process, number);
}
static __attribute__((unused)) void agent_bash_job_free(agent_bash_job *job) {
    if (!job) return;
    ds4_agent_bash_free(job->process);
    free(job);
}
static void agent_bash_jobs_free(agent_worker *w) {
    ds4_agent_executor_jobs_free(&w->tools);
}
static __attribute__((unused)) agent_bash_job *agent_bash_find_job(agent_worker *w,
        int id, pid_t pid) {
    return ds4_agent_executor_find(&w->tools, id, pid);
}
static void agent_bash_remove_job(agent_worker *w, agent_bash_job *job) {
    ds4_agent_executor_remove(&w->tools, job);
}
static agent_bash_job *agent_bash_start(agent_worker *w, const char *command,
        int timeout, char *err, size_t errlen) {
    return ds4_agent_executor_start(&w->tools, command, timeout,
        agent_bash_terminal_changed, w, err, errlen);
}


static char *agent_bash_observation(agent_bash_job *job, bool observed,
                                    bool *done) {
    return ds4_agent_bash_observation(job->process, observed, done);
}


/* Shell bytes are data, not instructions to our terminal. Keep only SGR color
 * sequences; OSC, cursor motion, erasure and other controls stay in the log. */
static char *agent_terminal_safe_text(const char *text, size_t len) {
    return ds4_agent_executor_terminal_safe(text, len);
}


/* ============================================================================
 * Tool Dispatch
 * ============================================================================
 */

static void agent_tool_observation_init(agent_tool_observation *obs) {
    ds4_agent_observation_init(obs);
}
static void agent_tool_observation_puts(agent_tool_observation *obs,
        const char *text) {
    ds4_agent_observation_puts(obs, text);
}
static __attribute__((unused)) void agent_tool_observation_add_image(
        agent_tool_observation *obs, ds4_vision_embedding *embedding) {
    ds4_agent_observation_add_image(obs, embedding);
}
static void agent_tool_observation_free(agent_tool_observation *obs) {
    ds4_agent_observation_free(obs);
}


/* Execute one parsed DSML tool call and return the text that will be appended as
 * the tool-role result.  UI visualization already happened while streaming; this
 * function is only about side effects and the model-visible observation. */
static char *agent_execute_tool_call(agent_worker *w, const agent_tool_call *call) {
    return ds4_agent_executor_call(&w->tools, agent_worker_effective_ctx_size(w),
        w && w->cfg && w->cfg->edit_upto,
        w && w->cfg && w->cfg->non_interactive, isatty(STDOUT_FILENO) != 0,
        w, agent_presentation_emit, worker_cancel_session_cb,
        agent_bash_terminal_changed, call);
}


/* Dispatch and publication remain whole C actions, not native getters. */
char *ds4_agent_tool_execute_one(void *worker, const agent_tool_call *call) {
    return agent_execute_tool_call(worker, call);
}
void ds4_agent_tool_publish(void *worker, const char *text, size_t len) {
    agent_publish(worker, text, len);
}
static agent_tool_observation agent_execute_tool_observation(
        agent_worker *w, const agent_tool_calls *calls) {
    agent_tool_observation obs;
    ds4_agent_observation_execute(&obs, w, w->engine, calls);
    return obs;
}
static __attribute__((unused)) bool agent_tool_observation_build(agent_worker *w,
        const agent_tool_observation *obs, ds4_tokens *tokens,
        ds4_vision_span **spans, char *err, size_t errlen) {
    return ds4_agent_observation_build(w->engine, &w->transcript, obs,
        tokens, spans, err, errlen);
}
static __attribute__((unused)) int agent_tool_observation_fits(agent_worker *w,
        const agent_tool_observation *obs, int reserve, int *tokens_out,
        char *err, size_t errlen) {
    return ds4_agent_observation_fits(w->engine, &w->transcript,
        agent_worker_effective_ctx_size(w), obs, reserve, tokens_out, err, errlen);
}

/* Preserve running process knowledge when compaction rebuilds the context. */
static char *agent_bash_jobs_compaction_observation(agent_worker *w) {
    return ds4_agent_executor_compaction_observation(&w->tools);
}


/* ============================================================================
 * Context Compaction
 * ============================================================================
 *
 * Compaction asks the model for durable task state, then rebuilds the live
 * transcript as: system prompt + summary + recent verbatim tail.  This keeps
 * the active KV usable while avoiding unbounded transcript growth.
 */

/* Decide when to compact before an ordinary turn or before appending a large
 * tool result.  The fixed free-token threshold is capped proportionally for
 * smaller contexts so tests with tiny contexts still compact rather than fail. */
static bool agent_worker_should_compact(agent_worker *w) {
    return ds4_agent_compaction_should_compact(w->engine, agent_worker_effective_ctx_size(w), w->transcript.len);
}

static int agent_special_token_id(ds4_engine *engine, const char *rendered) {
    ds4_tokens t = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    ds4_tokens_free(&t);
    return id;
}

static __attribute__((unused)) int agent_compact_tail_boundary(const ds4_tokens *tokens, int bottom,
                                       int sys_len, int tail_budget, int user_id) {
    return ds4_agent_compaction_tail_boundary(tokens, bottom, sys_len, tail_budget, user_id);
}

/* Prefer a complete recent user turn, with a bounded fragment as fallback. */

static void agent_tokens_append_range(ds4_tokens *dst, const ds4_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) ds4_tokens_push(dst, src->v[i]);
}

/* Build the private prompt used to ask the model for durable state.  The prompt
 * explicitly forbids tool calls because the result is consumed internally, not
 * delivered as an assistant turn. */

static __attribute__((unused)) int agent_compact_summary_budget(int ctx) {
    return ds4_agent_compaction_summary_budget(ctx);
}

static int agent_compact_reserve_tokens(agent_worker *w) {
    return ds4_agent_compaction_reserve(w->engine, agent_worker_effective_ctx_size(w));
}

/* A summary prefix or retained tail must never split an image block. */
static __attribute__((unused)) int agent_compact_image_boundary(const ds4_vision_span *images, size_t count,
                                        bool glm, int pos) {
    return ds4_agent_compaction_image_boundary(images, count, glm, pos);
}

/* Text-only stages. C owns all model/session/UI/image-embedding actions. */
static __attribute__((unused)) void agent_compaction_plan_free(ds4_agent_compaction_plan *plan) {
    ds4_agent_compaction_free(plan);
}

static __attribute__((unused)) bool agent_worker_compaction_prepare_plan(agent_worker *w,
        const ds4_tokens *sys, const char *reason, int ctx,
        ds4_agent_compaction_plan *plan, char *err, size_t err_len) {
    return ds4_agent_compaction_prepare(plan, w->engine, &w->transcript, sys,
        w->images, w->image_count, ctx, reason, err, err_len);
}

static __attribute__((unused)) bool agent_worker_compaction_rebuild_plan(agent_worker *w,
        const ds4_tokens *sys, const char *summary, const int *open_assistant,
        int ctx, ds4_agent_compaction_plan *plan, char *err, size_t err_len) {
    return ds4_agent_compaction_rebuild(plan, w->engine, &w->transcript, sys,
        w->images, w->image_count, ctx, agent_context_prerope_requested(),
        summary, open_assistant, err, err_len);
}


static __attribute__((unused)) bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len) {
    if (!agent_worker_should_compact(w)) return true;
    return agent_worker_compact(w, reason, err, err_len);
}


/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* DSML structure is a machine-readable grammar, so once the model has clearly
 * started a tool stanza we decode grammar bytes greedily.  Parameter values are
 * different: they can be shell commands, code, file contents, or edit bodies,
 * and should keep the configured sampling behavior.  The only exception is a
 * parameter closing tag once it is clearly DSML syntax, not ordinary text such
 * as HTML/XML/code containing "</".
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy. */
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    return ds4_agent_stream_greedy(sr);
}


static bool agent_stream_compaction_needs_lookahead(const agent_stream_renderer *sr) {
    return ds4_agent_stream_lookahead(sr);
}

/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model native DSML tool iteration without a client/server protocol. */

#ifdef DS4_AGENT_TEST
static void test_agent_idle_context_rejection_precedes_user_append(void) {
    agent_config cfg = {.gen = {.live_context_path = "/dev/null/ds4-agent-idle-context",
                                .ctx_size = 4096, .think_mode = DS4_THINK_NONE}};
    int original[] = {11, 22, 33};
    agent_worker w = {.cfg = &cfg, .wake_pending = true,
                      .transcript = {.v = original, .len = 3, .cap = 3}};
    AGENT_TEST_ASSERT(pthread_mutex_init(&w.mu, NULL) == 0);
    /* No engine/session is provided: rejection must precede any token/model
     * work as well as appending the incoming user or replacing the mirror. */
    AGENT_TEST_ASSERT(worker_run_turn(&w, "incoming user") == 1);
    AGENT_TEST_ASSERT(w.status.state == AGENT_WORKER_ERROR);
    AGENT_TEST_ASSERT(strstr(w.status.error, "live context edit rejected:") != NULL);
    AGENT_TEST_ASSERT(w.transcript.v == original && w.transcript.len == 3);
    AGENT_TEST_ASSERT(original[0] == 11 && original[1] == 22 && original[2] == 33);
    AGENT_TEST_ASSERT(!w.user_activity && !w.session_dirty && w.out == NULL);
    AGENT_TEST_ASSERT(pthread_mutex_destroy(&w.mu) == 0);
}
#endif

/* ============================================================================
 * Worker/UI Synchronization Helpers
 * ============================================================================
 */

static int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}

/* Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode. */

/* Actual agent_editor lives once in ds4_agent_terminal.h. */


#ifdef DS4_AGENT_TEST
static void test_agent_terminal_wrap_output_is_deferred(void) {
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) return;
    int saved_stdout = dup(STDOUT_FILENO);
    AGENT_TEST_ASSERT(saved_stdout >= 0);
    if (saved_stdout < 0) {
        fclose(sink);
        return;
    }
    AGENT_TEST_ASSERT(dup2(fileno(sink), STDOUT_FILENO) >= 0);

    agent_editor ed = {
        .active = true,
        .scroll_region = true,
        .output_bottom = 22,
        .prompt_row = 23,
        .output_cursor_saved = true,
    };
    ed.edit.cols = 80;
    char line[80];
    memset(line, 'x', sizeof(line));

    AGENT_TEST_ASSERT(!editor_write_scroll_output_preserve_prompt(
        &ed, line, sizeof(line), false));
    AGENT_TEST_ASSERT(ed.deferred_output.len == sizeof(line));
    AGENT_TEST_ASSERT(ed.output_col == 0);
    AGENT_TEST_ASSERT(!ed.output_line_open);

    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(
        &ed, "y", 1, false));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0);
    AGENT_TEST_ASSERT(ed.output_col == 1);
    AGENT_TEST_ASSERT(ed.output_line_open);

    ed.output_col = 0;
    ed.output_line_open = false;
    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(
        &ed, line, sizeof(line), true));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0);
    AGENT_TEST_ASSERT(ed.output_col == 0);
    AGENT_TEST_ASSERT(!ed.output_line_open);

    AGENT_TEST_ASSERT(!editor_write_scroll_output_preserve_prompt(&ed, "\xe4", 1, false));
    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(&ed, NULL, 0, true));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0 && ed.output_utf8_len == 0);
    AGENT_TEST_ASSERT(dup2(saved_stdout, STDOUT_FILENO) >= 0);
    close(saved_stdout);
    fclose(sink);
}
#endif
