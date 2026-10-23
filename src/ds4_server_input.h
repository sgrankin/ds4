#ifndef DS4_SERVER_INPUT_H
#define DS4_SERVER_INPUT_H

#include "ds4.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Actual server wire/logical request records, defined once for C materialization
 * and native decoding. Each vector and string is libc-owned; image embeddings
 * and prompt IDs retain their public DS4 destructors. responses_store_dir is
 * the sole borrowed string, immutable for the server lifetime. No engine or
 * session layout crosses this boundary. */
#define SERVER_IMAGE_MARKER_BYTES 64

typedef struct {
    char marker[SERVER_IMAGE_MARKER_BYTES];
    uint8_t *encoded;
    size_t encoded_len;
} server_image_input;

typedef struct {
    server_image_input *v;
    size_t len;
    size_t cap;
} server_image_inputs;

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

typedef enum {
    SERVER_MODEL_SYNTAX_DEEPSEEK,
    SERVER_MODEL_SYNTAX_GLM,
    SERVER_MODEL_SYNTAX_DEEPSEEK41,
    SERVER_MODEL_SYNTAX_QWEN,
} server_model_syntax;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_tool_text;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *namespace;
    /* Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    char **prop;
    char **prop_schema;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    size_t begin, end;
    char *id, *content;
} tool_result_span;

typedef struct {
    char *role;
    char *content;
    /* Original Responses declarations, kept privately for durable replay. */
    char *discovered_tools;
    server_image_inputs images;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
    tool_result_span *tool_results;
    int tool_results_len;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

/* Optional borrowed transport for Responses semantic events. A connection owns
 * this sink until its job completes. load_parent returns libc-owned response
 * and logical replay strings (0 found, 1 missing); durable fallback is separate.
 * HTTP leaves it NULL. Model and canonical input records do not depend on it. */
typedef struct {
    void *context;
    bool (*emit)(void *context, const char *json, size_t length);
    int (*load_parent)(void *context, const char *id, char **response, char **replay);
} server_response_sink;

typedef struct {
    req_kind kind;
    api_style api;
    const server_response_sink *response_sink;
    server_model_syntax model_syntax;
    ds4_tokens prompt;
    ds4_vision_span *images;
    char (*image_markers)[SERVER_IMAGE_MARKER_BYTES];
    size_t image_count;
    char *model;
    bool model_from_request;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    char *prompt_cache_key; /* Client affinity hint; never enables approximation. */
    const char *responses_store_dir; /* Borrowed immutable server configuration. */
    bool responses_store;
    char *previous_response_id;
    char *conversation_id;
    char *conversation_input; /* New public items; history is loaded separately. */
    unsigned long long conversation_version;
    int responses_store_error_status;
    char *responses_replay; /* Private logical history, without top-level instructions. */
    char responses_published_id[40];
    tool_schema_orders tool_orders;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    /* Explicit client sampling wins even in thinking mode; the fixed
     * DeepSeek-style thinking defaults apply only to omitted knobs. */
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    bool top_k_set;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    bool ignore_eos;
    int cache_read_tokens;
    int cache_write_tokens;
    ds4_think_mode think_mode;
    int reasoning_budget; /* Responses opt-in; generated thinking tokens only. */
    int generated_thinking_tokens;
    bool has_tools;
    bool prompt_preserves_reasoning;
    /* For /v1/responses: emit reasoning_summary_* events / fields only when the
     * client opted in via reasoning.summary. Other APIs leave this false; the
     * field is ignored on those code paths. */
    bool reasoning_summary_emit;
    /* Responses continuation contract:
     *
     * A live Responses tool loop is not a normal "new prompt with a long
     * prefix" request.  The protocol gives tool outputs a call_id that binds
     * them to a prior assistant tool call.  If that call_id is still known in
     * memory, the live KV is the authoritative prefix, including any hidden
     * thinking that the client did not replay.  These fields carry the parsed
     * evidence needed by generate_job() to append only the new suffix.
     *
     * A tool-output-only request has no stateless prefix to match.  If the live
     * call_id binding is gone by the time the worker executes it, DS4 must ask
     * for a full replay rather than cold-prefilling a prompt that starts with a
     * naked tool result.  Similarly, if live state is gone, a reasoning-mode
     * tool replay must contain the prior reasoning item (or an equivalent
     * opaque reasoning state from a future implementation). */
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    tool_replay_stats tool_replay;
} request;

/* Decode stops before durable-history/live-tool resolution and prompt rendering.
 * The existing C materializer consumes this same logical input; it does not
 * parse the endpoint again. Nested content/schema codecs remain the transitional
 * C parser library, preserving existing acceptance and partial-failure policy. */
typedef enum {
    DS4_SERVER_INPUT_CHAT,
    DS4_SERVER_INPUT_ANTHROPIC,
    DS4_SERVER_INPUT_RESPONSES,
    DS4_SERVER_INPUT_COMPLETION,
} ds4_server_input_endpoint;

typedef struct {
    int max_tokens;
    int ctx_size;
    server_model_syntax syntax;
    const char *responses_store_dir;
    bool incremental_tool_schemas;
} ds4_server_input_defaults;

typedef struct {
    request req;
    chat_msgs messages;
    char *tool_schemas;
    char *loaded_tool_schemas;
    char *explicit_tools;
    char *instructions;
    char *completion_prompt;
    bool got_input;
    bool tool_choice_none;
    bool got_thinking;
    bool thinking_enabled;
    ds4_think_mode reasoning_effort;
} ds4_server_input;

/* out is initialized before parsing and cleared on failure. Success transfers
 * every owned field to out; deinit also releases fields filled by materialize.
 * No final missing-input/storage/live-state check is done here: their original
 * order relative to durable parent loading stays in the C materializer. */
bool ds4_server_input_decode(ds4_server_input_endpoint endpoint, const char *body,
    const ds4_server_input_defaults *defaults, ds4_server_input *out,
    char *err, size_t errlen);
void ds4_server_input_deinit(ds4_server_input *input);
/* Complete logical owners are reusable by C runtime finalizers too. */
void ds4_server_input_messages_free(chat_msgs *messages);
void ds4_server_input_request_free(request *request);

/* Existing raw-value skip ceiling, shared by the C reference and native owner. */
#define DS4_SERVER_JSON_MAX_NESTING 256

/* Transitional real JSON/value-codec library. Calls are direct functions,
 * never callback getters into a server. Cursor/output behavior is the original
 * C parser, including lone surrogates, nonfinite numbers and duplicate keys. */
void ds4_server_json_ws(const char **cursor);
bool ds4_server_json_lit(const char **cursor, const char *literal);
bool ds4_server_json_string(const char **cursor, char **out);
bool ds4_server_json_number(const char **cursor, double *out);
bool ds4_server_json_int(const char **cursor, int *out);
bool ds4_server_json_bool(const char **cursor, bool *out);
bool ds4_server_json_skip(const char **cursor);
bool ds4_server_json_raw(const char **cursor, char **out);
/* Existing string-aware raw-value minifier; returned text is libc-owned. */
char *ds4_server_json_minify(const char *json);
bool ds4_server_json_tools(const char **cursor, char **schemas, tool_schema_orders *orders);
bool ds4_server_json_messages(const char **cursor, chat_msgs *messages, bool trusted_store);
bool ds4_server_json_anthropic_messages(const char **cursor, chat_msgs *messages);
bool ds4_server_json_anthropic_system(const char **cursor, char **out);
/* loaded_schemas is an optional owned accumulated string, including partial
 * codec output on error. Both it and orders may be NULL. A thin C adapter wraps
 * its existing buf local; no buf ABI is copied. */
bool ds4_server_json_responses_input(const char **cursor, chat_msgs *messages,
    char **loaded_schemas, tool_schema_orders *orders, bool incremental_schemas);
bool ds4_server_json_responses_content(const char **cursor, char **out,
    server_image_inputs *images);
int ds4_server_input_restore_discovered_tools(chat_msgs *messages,
    tool_schema_orders *orders, bool tool_choice_none,
    const char *explicit_tools);
bool ds4_server_json_prompt(const char **cursor, char **out);

/* Native declaration/message codec extensions; the records above are unchanged. */
void ds4_server_input_schema_order_add(tool_schema_orders *orders,
    const char *json, const char *namespace, const char *wire_name,
    bool responses_tool_search);
bool ds4_server_json_content(const char **cursor, char **out);
bool ds4_server_json_function_call(const char **cursor, tool_call *call);
bool ds4_server_json_tool_calls(const char **cursor, tool_calls *calls);
bool ds4_server_json_image_url(const char **cursor, char **out);
/* Image byte admission, limits and random marker creation retain one C owner. */
bool ds4_server_image_data_uri(server_image_inputs *images, const char *uri,
    char marker[SERVER_IMAGE_MARKER_BYTES]);
bool ds4_server_image_base64(server_image_inputs *images,
    const char *media_type, const char *data,
    char marker[SERVER_IMAGE_MARKER_BYTES]);

#endif
