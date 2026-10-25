#ifndef DS4_SERVER_OUTPUT_H
#define DS4_SERVER_OUTPUT_H
#include "ds4_server_input.h"
#include "ds4_server_http.h"
#include <stdio.h>

/* Actual C buffer/Responses stream/item records, shared once. Strings and
 * buffers remain libc-owned and NUL-terminated; len excludes the terminator.
 * Request/Engine/Session state is not mirrored here. The worker owns when a
 * stream becomes active and retains its socket/stop flag for each action. */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

typedef enum {
    RESP_STREAM_THINKING,
    RESP_STREAM_TEXT,
    RESP_STREAM_SUPPRESS,
} responses_stream_mode;

typedef struct {
    responses_stream_mode mode;
    const server_response_sink *sink;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool reasoning_item_opened;
    bool reasoning_item_closed;
    bool reasoning_summary_started;
    bool reasoning_closed_naturally;
    bool message_item_opened;
    bool message_text_part_open;
    bool message_item_closed;
    bool reasoning_emitted_any;
    bool message_emitted_any;
    buf reasoning_text;
    buf message_text;
    char response_id[40];
    char reasoning_id[40];
    char message_id[40];
    int reasoning_index;   /* output_index of the reasoning item (0 if present) */
    int message_index;     /* output_index of the assistant message item */
    int next_output_index; /* monotonic counter for upcoming output items */
    int sequence;          /* monotonic per-event sequence_number Codex consumes */
} responses_stream;

/* Item identity is stable across added/done/completed. */
typedef struct {
    char fc_id[40];
    char call_id[64];
    bool is_custom;
    int output_index;
} responses_tool_item;

/* Complete output actions; sockets and request/state records are borrowed.
 * IDs remain stable across added/done/terminal events; sequence_number advances
 * on every framed event even when its send fails. */
void ds4_server_output_random_id(char *dst, size_t capacity, const char *prefix);
void ds4_server_output_init(const request *req, responses_stream *stream);
void ds4_server_output_free(responses_stream *stream);
void ds4_server_output_append_item(buf *out, const tool_call *call,
    const responses_tool_item *item, const char *status, bool with_args,
    const tool_schema_orders *orders);
void ds4_server_output_append_usage(buf *out, const request *req,
    int input_tokens, int output_tokens);
bool ds4_server_output_created(int fd, const request *req, responses_stream *stream,
    long created_at, const volatile sig_atomic_t *stop);
bool ds4_server_output_update(int fd, const request *req, responses_stream *stream,
    const char *raw, size_t length, bool final, const volatile sig_atomic_t *stop);
bool ds4_server_output_completed(int fd, request *req, responses_stream *stream,
    const tool_calls *calls, const responses_tool_item *items, const char *finish,
    int input_tokens, int output_tokens, long created_at,
    const volatile sig_atomic_t *stop);
bool ds4_server_output_finish(int fd, request *req, responses_stream *stream,
    const char *raw, size_t length, const char *recovered_content,
    const tool_calls *calls, const char *finish, int input_tokens,
    int output_tokens, long created_at, const volatile sig_atomic_t *stop);
bool ds4_server_output_final(int fd, bool cors, request *req,
    const char *text, const char *reasoning, const tool_calls *calls,
    const char *finish, int input_tokens, int output_tokens,
    const volatile sig_atomic_t *stop);

/* Hosted rounds remain inside one Response. Closed round items are appended
 * to req->responses_output_prefix; reset retains response identity/sequence. */
void ds4_server_output_round_reset(const request *req, responses_stream *stream);
bool ds4_server_output_round_close(int fd, request *req, responses_stream *stream,
    const char *raw, size_t length, const char *content, const char *reasoning,
    bool recovered, const char *finish, const volatile sig_atomic_t *stop);
bool ds4_server_output_hosted_call(int fd, request *req, responses_stream *stream,
    const tool_call *call, const tool_calls *group, responses_tool_item *item, const char *status,
    const char *model, const char *input_json, const char *result_json,
    const volatile sig_atomic_t *stop);
bool ds4_server_output_program_call(int fd, request *req, responses_stream *stream,
    const tool_call *call, const tool_calls *group, responses_tool_item *program_item,
    responses_tool_item *output_item, const char *status, const char *bindings_json,
    const char *result_json, const char *audit_json, const volatile sig_atomic_t *stop);
bool ds4_server_output_external_call(int fd, request *req, responses_stream *stream,
    const tool_call *call, responses_tool_item *item, const char *finish,
    const volatile sig_atomic_t *stop);
void ds4_server_output_append_decision_item(buf *out, const tool_call *call,
    const tool_calls *group, const responses_tool_item *item, const char *status, const char *model,
    const char *input_json, const char *result_json);

/* Native compatibility entries, not callbacks. Canonical arguments use the
 * existing tolerant object codec; the returned string is libc-owned. Durable
 * publication retains Conversations version checks, store files and locking.
 * Native output must publish before it reports successful completion. */
char *ds4_server_output_object_json(const char *arguments);
bool ds4_server_output_publish(request *req, const char *id, const char *json,
    char *error, size_t error_capacity);
#endif
