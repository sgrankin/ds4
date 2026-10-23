#ifndef DS4_AGENT_TERMINAL_H
#define DS4_AGENT_TERMINAL_H
#include "ds4_agent_lifecycle.h"
#include "linenoise.h"
/* Actual UI storage moved once. linenoiseState remains its real vendor record;
 * model/Worker state is not mirrored. Queued strings/input bytes use libc. */
#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    bool prompt_dirty, status_dirty;
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool output_pending_wrap;
    char output_utf8[4];
    size_t output_utf8_len;
    int output_escape;
    bool output_zwj, output_regional;
    int output_glyph_width;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    agent_input_buf deferred_output;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

typedef enum {
    CPR_INVALID,
    CPR_PARTIAL,
    CPR_COMPLETE,
} cpr_state;

void ds4_agent_ui_write(int,const char*,size_t);
void ds4_agent_ui_input_append(agent_input_buf*,const char*,size_t);
char *ds4_agent_ui_input_take(agent_input_buf*);
void ds4_agent_ui_input_free(agent_input_buf*);
bool ds4_agent_ui_tty(void);
char *ds4_agent_ui_echo_text(const char*);
void ds4_agent_ui_echo(const char*);
void ds4_agent_ui_context_size(int,char*,size_t);
void ds4_agent_ui_progress(int,int,double,char*,size_t,bool);
void ds4_agent_ui_prompt(const agent_status*,char*,size_t);
void ds4_agent_ui_status(const agent_status*,char*,size_t);
void ds4_agent_ui_footer(const agent_status*,const agent_prompt_queue*,int,char*,size_t);
void ds4_agent_ui_queue_push(agent_prompt_queue*,const char*);
char *ds4_agent_ui_queue_pop(agent_prompt_queue*);
void ds4_agent_ui_queue_front(agent_prompt_queue*,char*);
char *ds4_agent_ui_queue_take(agent_prompt_queue*);
char *ds4_agent_ui_queue_echo(agent_prompt_queue*);
const char *ds4_agent_ui_queue_peek(const agent_prompt_queue*);
void ds4_agent_ui_queue_free(agent_prompt_queue*);
bool ds4_agent_ui_size(int*,int*);
void ds4_agent_ui_read(agent_editor*);
bool ds4_agent_ui_take_byte(agent_editor*,unsigned char);
bool ds4_agent_ui_take_escape(agent_editor*);
void ds4_agent_ui_replace(agent_editor*,const char*);
void ds4_agent_ui_note(agent_editor*,const char*,size_t);
int ds4_agent_ui_start(agent_editor*,const char*,const char*,const char*);
void ds4_agent_ui_stop(agent_editor*);
void ds4_agent_ui_hide(agent_editor*);
void ds4_agent_ui_show(agent_editor*);
void ds4_agent_ui_prompt_status(agent_editor*,const char*,const char*);
void ds4_agent_ui_flush(agent_editor*,bool);
bool ds4_agent_ui_scroll(agent_editor*,const char*,size_t,bool);
void ds4_agent_ui_async(agent_editor*,const char*,size_t,const char*,const char*,bool);
void ds4_agent_ui_cancel(agent_editor*,const char*,const char*);
void ds4_agent_ui_restore(agent_editor*);
void ds4_agent_ui_welcome(const agent_config*,char*,size_t);
void ds4_agent_ui_write_welcome(agent_editor*,const agent_config*,const char*,const char*);
#endif
