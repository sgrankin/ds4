#ifndef DS4_SERVER_PROTOCOL_H
#define DS4_SERVER_PROTOCOL_H

#include "ds4_server_reply.h"
#include "ds4_server_tool_memory.h"

/* Actual stream records moved from ds4_server.c once. Strings in tool.ids
 * remain individually libc-owned; byte offsets refer to the caller's sampled
 * text. The worker owns Request/Server/Session, sockets and cancellation. */
typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for protocol-specific DSML stream projections.  The model
 * still samples DSML; these states only translate already-sampled bytes into
 * OpenAI / Anthropic wire events while final parsing remains authoritative. */
typedef struct {
    dsml_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool guard_second_reasoning;
    bool sent_reasoning;
    bool sent_content;
    openai_tool_stream tool;
} openai_stream;

typedef struct {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} dsml_syntax;

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_TOOL,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
    ANTH_BLOCK_TOOL,
} anthropic_block_type;

typedef struct {
    dsml_tool_stream_state state;
    const dsml_syntax *syn;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} anthropic_tool_stream;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool guard_second_reasoning;
    bool sent_thinking;
    bool sent_text;
    anthropic_tool_stream tool;
} anthropic_stream;

/* Shared byte helpers also serve the remaining C Anthropic projector. Its
 * generation tracker retains C policy over the same actual syntax record. */
size_t ds4_server_protocol_utf8_limit(const char *raw, size_t start, size_t limit, bool final);
size_t ds4_server_protocol_text_limit(const char *raw, size_t start, size_t length, bool has_tools, bool final);
size_t ds4_server_protocol_parameter_limit(const char *raw, size_t start, size_t length, const char *closing, bool string);
void ds4_server_protocol_random_tool_id(char *out, size_t capacity, api_style api);

/* Complete OpenAI actions. Tool memory and its mutex are borrowed together;
 * both are NULL when exact replay is disabled or no Server was supplied. */
void ds4_server_openai_init(const request *req, openai_stream *stream);
void ds4_server_openai_free(openai_stream *stream);
void ds4_server_openai_apply_ids(tool_calls *calls, const openai_stream *stream);
void ds4_server_openai_calls(buf *out, const tool_calls *calls, const char *prefix, bool deltas);
void ds4_server_openai_usage(buf *out, const request *req, int input, int output);
bool ds4_server_openai_chunk(int fd, const request *req, const char *id, const char *text, const char *finish,
    const volatile sig_atomic_t *stop);
bool ds4_server_openai_done(int fd, const request *req, const char *id, int input, int output,
    const volatile sig_atomic_t *stop);
bool ds4_server_openai_chat_finish(int fd, const request *req, const char *id, const char *content,
    const char *reasoning, const tool_calls *calls, const char *finish, int input,
    int output, const volatile sig_atomic_t *stop);
bool ds4_server_openai_update(int fd, tool_memory *memory, pthread_mutex_t *mutex, const request *req,
    const char *id, openai_stream *stream, const char *raw, size_t length, bool final,
    const volatile sig_atomic_t *stop);
bool ds4_server_openai_finish(int fd, tool_memory *memory, pthread_mutex_t *mutex, const request *req,
    const char *id, openai_stream *stream, const char *raw, size_t length,
    const tool_calls *calls, const char *finish, int input, int output,
    const volatile sig_atomic_t *stop);
bool ds4_server_openai_final(int fd, bool cors, const request *req, const char *id, const char *content,
    const char *reasoning, const tool_calls *calls, const char *finish, int input,
    int output, const volatile sig_atomic_t *stop);
/* Complete Anthropic projection actions over the same real records/library.
 * Live blocks preserve their own index/signature/stop lifecycle; they never
 * rewrite the model-visible sampled text or exact replay key. */
void ds4_server_anthropic_content(buf *out, const char *content,
    const char *reasoning, const tool_calls *calls, const char *prefix,
    const tool_schema_orders *orders);
void ds4_server_anthropic_usage(buf *out, const request *req, int input, int output);
void ds4_server_anthropic_free(anthropic_stream *stream);
void ds4_server_anthropic_apply_ids(tool_calls *calls, const anthropic_stream *stream);
bool ds4_server_anthropic_final(int fd, bool cors, const request *req,
    const char *id, const char *content, const char *reasoning,
    const tool_calls *calls, const char *finish, int input, int output,
    const volatile sig_atomic_t *stop);
bool ds4_server_anthropic_start(int fd, const request *req, const char *id,
    int input, anthropic_stream *stream, const volatile sig_atomic_t *stop);
bool ds4_server_anthropic_update(int fd, tool_memory *memory,
    pthread_mutex_t *mutex, const request *req, const char *id,
    anthropic_stream *stream, const char *raw, size_t length, bool final,
    const volatile sig_atomic_t *stop);
bool ds4_server_anthropic_finish(int fd, tool_memory *memory,
    pthread_mutex_t *mutex, const request *req, const char *id,
    anthropic_stream *stream, const char *raw, size_t length,
    const tool_calls *calls, const char *finish, int output,
    const volatile sig_atomic_t *stop);
#endif
