#ifndef DS4_SERVER_RENDER_H
#define DS4_SERVER_RENDER_H

#include "ds4_server_output.h"

/* Actual argument records extracted from ds4_server.c once. Strings/vectors
 * retain libc ownership; successful parsing appends to an initialized vector.
 * These are also used by existing tool coercion and production test code. */
typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

bool ds4_server_render_arguments_parse(const char *json, json_args *args);
void ds4_server_render_arguments_free(json_args *args);
int ds4_server_render_arguments_find(json_args *args, const char *key);
void ds4_server_render_tools(buf *out, const char *schemas, bool v41);
void ds4_server_render_glm_body(buf *out, const char *text, const char *end);
void ds4_server_render_glm_result(buf *out, const char *text);
void ds4_server_render_tool_result(buf *out, const char *text);
void ds4_server_render_object(buf *out, const char *json);
void ds4_server_render_dsml_calls(buf *out, const tool_calls *calls, bool v41);
void ds4_server_render_glm_calls(buf *out, const tool_calls *calls,
    const tool_schema_orders *orders);
void ds4_server_render_flint_calls(buf *out, const tool_calls *calls,
    const tool_schema_orders *orders);
void ds4_server_render_qwen_calls(buf *out, const tool_calls *calls,
    bool has_content, const tool_schema_orders *orders);

/* Inputs are borrowed for this call. Returned text is libc-owned and terminated.
 * start/skip are existing nonnegative logical-message positions. No Server,
 * Engine, Session, cache or durable-storage state enters this boundary. */
char *ds4_server_render_deepseek41(const chat_msgs *messages, int start,
    const char *schemas, ds4_think_mode mode, bool live_tail,
    const stop_list *initial_order);
char *ds4_server_render_flint(const chat_msgs *messages, int start,
    const char *schemas);
char *ds4_server_render_qwen(const chat_msgs *messages, int start,
    const char *schemas, const tool_schema_orders *orders,
    ds4_think_mode mode, bool live_tail);
char *ds4_server_render_glm(const chat_msgs *messages, int start,
    const char *schemas, const tool_schema_orders *orders,
    ds4_think_mode mode, bool live_tail);
void ds4_server_render_replay_message(buf *out, const chat_msg *message);
char *ds4_server_render_replay_history(const chat_msgs *messages, int skip);

#endif
