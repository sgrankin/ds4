#ifndef DS4_AGENT_STREAM_H
#define DS4_AGENT_STREAM_H
#include "ds4_agent_presentation.h"
#include "ds4_agent_tools.h"

/* Actual streaming projection state moved once. Renderer/parser are borrowed
 * for one assistant round; their owners remain responsible for buffers. The
 * sampled transcript is unchanged: this owner only projects terminal output. */
typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

typedef struct {
    char tail[32];
    size_t len;
} agent_dsml_marker_detector;

typedef struct {
    agent_token_renderer *renderer;
    agent_dsml_parser *parser;
    agent_tool_syntax syntax;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    agent_dsml_marker_detector plain_dsml;
    agent_dsml_marker_detector think_dsml;
    bool dsml_in_think;
    bool dsml_in_think_reported;
    bool post_think_gap;
    bool tool_preflight_error;
    char tool_preflight_error_msg[256];
} agent_stream_renderer;

/* Whole Worker actions retained in C until tool/worker ownership migrates.
 * The native owner neither dereferences Worker nor imports its layout. */
void ds4_agent_stream_trace(void *worker, const char *message);
int ds4_agent_stream_read_lines(void *worker);
bool ds4_agent_stream_preflight(void *worker, const agent_tool_call *call,
    char *error, size_t error_size);

void ds4_agent_stream_text(agent_stream_renderer *, const char *, size_t, bool);
bool ds4_agent_stream_greedy(const agent_stream_renderer *);
bool ds4_agent_stream_lookahead(const agent_stream_renderer *);
#endif
