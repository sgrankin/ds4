#ifndef DS4_AGENT_PRESENTATION_H
#define DS4_AGENT_PRESENTATION_H
#include "ds4.h"
#include <stdbool.h>
#include <stddef.h>
/* One output action; the renderer never dereferences Worker/UI state. */
typedef void (*ds4_agent_presentation_emit_fn)(void *, const char *, size_t);
/* Actual renderer/capture state moved once. Buffers are libc-owned; syntax
 * handles and code-prefix strings are borrowed for the rendering lifetime. */
typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    ds4_engine *engine;
    void *worker;
    ds4_agent_presentation_emit_fn emit;
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_hint;
    bool md_line_started;
    char md_hint_prefix[sizeof("> **Hint:**") - 1];
    size_t md_hint_prefix_len;
    char md_inline[4096];
    size_t md_inline_len;
    bool md_escape;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    bool md_code_highlight_upto;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

void ds4_agent_presentation_capture_append(agent_tail_capture *, const char *, size_t);
char *ds4_agent_presentation_capture_take(agent_tail_capture *, size_t *);
const agent_syntax *ds4_agent_presentation_syntax_path(const char *);
int ds4_agent_presentation_columns(void);
void ds4_agent_presentation_write(agent_token_renderer *, const char *, size_t);
void ds4_agent_presentation_char(agent_token_renderer *, unsigned char);
void ds4_agent_presentation_restore(agent_token_renderer *);
void ds4_agent_presentation_reset(agent_token_renderer *);
void ds4_agent_presentation_hint_end(agent_token_renderer *);
void ds4_agent_presentation_process(agent_token_renderer *, const char *, size_t, bool);
void ds4_agent_presentation_finish(agent_token_renderer *);
void ds4_agent_presentation_color(agent_token_renderer *, const char *);
void ds4_agent_presentation_plain(agent_token_renderer *, const char *, size_t);
void ds4_agent_presentation_code_begin(agent_token_renderer *, const agent_syntax *);
void ds4_agent_presentation_code_prefix(agent_token_renderer *, const char *, const char *);
void ds4_agent_presentation_code_upto(agent_token_renderer *, bool);
void ds4_agent_presentation_code_end(agent_token_renderer *);
void ds4_agent_presentation_code_byte(agent_token_renderer *, unsigned char);
/* Existing production-linked C fixture invokes these presentation actions. */
void ds4_agent_presentation_flush_utf8(agent_token_renderer *);
void ds4_agent_presentation_markdown_byte(agent_token_renderer *, unsigned char);
void ds4_agent_presentation_markdown_finish(agent_token_renderer *);
#endif
