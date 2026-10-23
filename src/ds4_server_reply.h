#ifndef DS4_SERVER_REPLY_H
#define DS4_SERVER_REPLY_H

#include "ds4_server_render.h"

/* Actual protocol spellings moved from ds4_server.c once; live output and
 * native completed-reply decoding consume the same constants. */
#define DS41_TOOL_CALLS_START "<｜DSML｜ calls>"
#define DS41_TOOL_CALLS_END "</｜DSML｜ calls>"
#define DS41_INVOKE_START "<｜DSML｜ invoke"
#define DS41_INVOKE_END "</｜DSML｜ invoke>"
#define DS41_PARAM_START "<｜DSML｜ parameter"
#define DS41_PARAM_END "</｜DSML｜ parameter>"

#define DS4_DSML "｜DSML｜"
#define DS4_DSML_SHORT "DSML｜"
#define DS4_TOOL_CALLS_START "<" DS4_DSML "tool_calls>"
#define DS4_TOOL_CALLS_END "</" DS4_DSML "tool_calls>"
#define DS4_INVOKE_START "<" DS4_DSML "invoke"
#define DS4_INVOKE_END "</" DS4_DSML "invoke>"
#define DS4_PARAM_START "<" DS4_DSML "parameter"
#define DS4_PARAM_END "</" DS4_DSML "parameter>"
#define DS4_TOOL_CALLS_START_SHORT "<" DS4_DSML_SHORT "tool_calls>"
#define DS4_TOOL_CALLS_END_SHORT "</" DS4_DSML_SHORT "tool_calls>"
#define DS4_INVOKE_START_SHORT "<" DS4_DSML_SHORT "invoke"
#define DS4_INVOKE_END_SHORT "</" DS4_DSML_SHORT "invoke>"
#define DS4_PARAM_START_SHORT "<" DS4_DSML_SHORT "parameter"
#define DS4_PARAM_END_SHORT "</" DS4_DSML_SHORT "parameter>"

/* Searches borrow the supplied NUL-terminated text; attribute returns a
 * libc-owned string or NULL. Buffer append/repair preserves its sole owner. */
const char *ds4_server_reply_tool_start(const char *text);
const char *ds4_server_reply_tool_end(const char *text);
size_t ds4_server_reply_trim_separator(const char *text, size_t start, size_t limit);
const char *ds4_server_reply_last_substring(const char *text, const char *needle);
char *ds4_server_reply_attribute(const char *tag, const char *name);

/* Output strings and call vectors retain the existing libc ownership rules,
 * including partial call vectors on parse failure. The response action clears
 * failed calls and supplies the original raw-text fallback/error policy. */
bool ds4_server_reply_parse(server_model_syntax syntax, const char *text,
    bool require_thinking_closed, char **content, char **reasoning,
    tool_calls *calls);
bool ds4_server_reply_repair(const char *text, size_t length, buf *out);
bool ds4_server_reply_for_response(server_model_syntax syntax, const char *text,
    bool has_tools, bool saw_tool_start, bool require_thinking_closed,
    const char **finish, char *error, size_t error_capacity,
    char **content, char **reasoning, tool_calls *calls, bool *recovered,
    const tool_schema_orders *orders);

#endif
