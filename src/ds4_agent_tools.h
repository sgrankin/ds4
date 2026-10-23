#ifndef DS4_AGENT_TOOLS_H
#define DS4_AGENT_TOOLS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "ds4_tool_text.h"

/* Private streaming-tool boundary. Strings, argument/call vectors and raw text
 * are libc-owned. Moving a current call into calls transfers all its fields and
 * zeroes the source; parser reset/free releases every owner and retains syntax.
 * UI/worker code borrows these records and must not retain raw pointers across
 * feed, which may grow raw/argument buffers. No engine or renderer state. */
typedef struct {
    char *name;
    char *value;
    bool is_string;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

typedef enum {
    AGENT_TOOL_SYNTAX_DSML,
    AGENT_TOOL_SYNTAX_GLM,
    AGENT_TOOL_SYNTAX_DSML41,
    AGENT_TOOL_SYNTAX_QWEN,
} agent_tool_syntax;

typedef enum {
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
    agent_tool_syntax syntax;
    agent_dsml_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    agent_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    bool param_close_prefix;
    bool glm_after_call;
    agent_tool_calls calls;
    char error[160];
} agent_dsml_parser;

void ds4_agent_tool_call_free(agent_tool_call *call);
void ds4_agent_tool_calls_free(agent_tool_calls *calls);
void ds4_agent_tool_call_add_arg(agent_tool_call *call, const char *name,
    const char *value, size_t value_len, bool is_string, const char *end_tag);
void ds4_agent_tool_calls_push(agent_tool_calls *calls, agent_tool_call *call);
const char *ds4_agent_tool_arg_value(const agent_tool_call *call, const char *name);
void ds4_agent_tool_parser_free(agent_dsml_parser *parser);
void ds4_agent_tool_parser_reset(agent_dsml_parser *parser);
void ds4_agent_tool_parser_start(agent_dsml_parser *parser);
void ds4_agent_tool_parser_feed(agent_dsml_parser *parser, const char *bytes, size_t len);
void ds4_agent_tool_parser_finish(agent_dsml_parser *parser);
void ds4_agent_tool_parser_error(agent_dsml_parser *parser, const char *message);
const char *ds4_agent_tool_start(agent_tool_syntax syntax);
const char *ds4_agent_tool_tag_name(agent_tool_syntax syntax, const char *name);
const char *ds4_agent_tool_parameter_close(agent_tool_syntax syntax);
/* Online renderer/sampler probe: distinguish a possible partial close from a
 * complete one without hiding literal tool/reasoning tags inside argument data. */
bool ds4_agent_tool_value_close_tail(agent_tool_syntax syntax,
    const char *tail, size_t len, bool *complete);

#endif
