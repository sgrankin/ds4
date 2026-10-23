#ifndef DS4_SERVER_GENERATION_H
#define DS4_SERVER_GENERATION_H

#include "ds4_server_persist.h"
#include "ds4_server_reuse.h"
#include "ds4_server_decode.h"

/* Actual existing progress/diagnostic packets moved once. The worker owns
 * their lifetime and borrows Server/Slot/Job; Engine/Session remain opaque. */
#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER 8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int rewind_to;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

typedef struct {
    server *srv;
    server_slot *slot;
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
    /* SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;
    bool stream;
    bool enable_cors;
    bool headers_sent;
    bool stream_failed;
    double last_keepalive;
    job *request_job;
} server_prefill_progress;

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

void ds4_server_generation_run(server *runtime, server_slot *slot, job *job,
    const volatile sig_atomic_t *stop);
void ds4_server_generation_progress(server_prefill_progress *progress,
    const char *event, int current, int total, const volatile sig_atomic_t *stop);

/* These existing logical actions are also called by the unchanged focused
 * server fixtures. They select the same native bodies used by Generation. */
void ds4_server_generation_thinking_feed(thinking_state *state, const char *text, size_t length);
thinking_state ds4_server_generation_thinking_prompt(const request *req);
bool ds4_server_generation_reasoning_due(int limit, int count, bool forced,
    bool inside, int output_left, int context_left, bool safe);
bool ds4_server_generation_complete_thinking(server_model_syntax syntax,
    const char *text, size_t length, size_t *cursor);
bool ds4_server_generation_remember_thinking(const request *req,
    const thinking_state *thinking, const char *finish);
bool ds4_server_generation_canonical_needed(const server *runtime, const tool_calls *calls);
int ds4_server_generation_multimodal_frontier(int live, int common,
    int prompt_length, bool image_matches);
int ds4_server_generation_decode_budget(int requested, int generated, int room);
void ds4_server_generation_assign_ids(server *runtime, tool_calls *calls, api_style api);
char *ds4_server_generation_assistant_suffix(const request *req, const char *content,
    const char *reasoning, const tool_calls *calls, bool visible);
char *ds4_server_generation_thinking_visible(const request *req, const char *content);
char *ds4_server_generation_qwen_tool_visible(const request *req,
    const char *finish, bool inside_thinking, const char *content, const tool_calls *calls);
char *ds4_server_generation_invalid_suffix(const request *req,
    const thinking_state *thinking, const char *detail);
bool ds4_server_generation_find_stop(const stop_list *stops, const char *text,
    size_t from, size_t *position, size_t *length);
size_t ds4_server_generation_stream_safe(const stop_list *stops, size_t length);
bool ds4_server_generation_structured_stream(const request *req);

/* Complete existing synchronized trace I/O actions. No runtime field getters,
 * model policy callbacks or borrowed trace strings escape these calls. */
uint64_t ds4_server_generation_trace_begin(server *runtime, const job *job,
    int cached, int prompt_tokens, const trace_cache_diag *diagnostic,
    const char *cache_source, int disk_cached, const char *disk_path);
void ds4_server_generation_trace_piece(server *runtime, uint64_t id,
    const char *piece, size_t length);
void ds4_server_generation_trace_event(server *runtime, uint64_t id,
    const char *complete_message);
void ds4_server_generation_trace_finish(server *runtime, uint64_t id,
    const request *req, const char *finish, int completion, bool saw_start,
    bool saw_end, const char *content, const char *reasoning,
    const tool_calls *calls, double elapsed);

#endif
