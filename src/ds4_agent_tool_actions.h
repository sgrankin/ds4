#ifndef DS4_AGENT_TOOL_ACTIONS_H
#define DS4_AGENT_TOOL_ACTIONS_H
#include "ds4.h"
#include "ds4_agent_tools.h"
#include "ds4_agent_file.h"
#include <limits.h>
#include <sys/types.h>

#define AGENT_TOOL_MAX_BYTES (128 * 1024)
#define AGENT_READ_DEFAULT_LINES_SMALL 120
#define AGENT_READ_DEFAULT_LINES_MEDIUM 240
#define AGENT_READ_DEFAULT_LINES_LARGE 500
#define AGENT_READ_SMALL_CONTEXT_MAX 8192
#define AGENT_READ_MEDIUM_CONTEXT_MAX 16384

/* The actual pager state, embedded once in Worker. Byte position plus the
 * within-line flag permits exact continuation of long lines and CR/LF files. */
typedef struct {
    char more_path[PATH_MAX];
    int more_next_line;
    off_t more_byte_offset;
    bool more_mid_line;
    bool more_bare;
    bool more_valid;
} ds4_agent_pager;

/* Actual tool observation owners. Parts/images are libc-compatible vectors;
 * each text buffer and embedding is owned until freed or explicitly moved. */
typedef struct {
    char *text;
    size_t len;
    size_t cap;
} agent_tool_text_part;

typedef struct {
    agent_tool_text_part *parts;
    size_t part_count;
    size_t part_cap;
    ds4_vision_embedding *images;
    size_t image_count;
    size_t image_cap;
} agent_tool_observation;

int ds4_agent_read_default_lines(int context_size);
char *ds4_agent_read_range(ds4_agent_pager *, int context_size, const char *,
    int start, int count, bool whole, bool bare, bool set_more,
    off_t offset, bool mid_line);
char *ds4_agent_tool_read(ds4_agent_pager *, int context_size, const agent_tool_call *);
char *ds4_agent_tool_more(ds4_agent_pager *, int context_size, const agent_tool_call *);
char *ds4_agent_tool_list(const agent_tool_call *);
char *ds4_agent_tool_search(const agent_tool_call *);
void ds4_agent_observation_init(agent_tool_observation *);
void ds4_agent_observation_puts(agent_tool_observation *, const char *);
void ds4_agent_observation_add_image(agent_tool_observation *, ds4_vision_embedding *);
void ds4_agent_observation_free(agent_tool_observation *);
void ds4_agent_observation_execute(agent_tool_observation *, void *worker,
    ds4_engine *, const agent_tool_calls *);
bool ds4_agent_observation_build(ds4_engine *, const ds4_tokens *transcript,
    const agent_tool_observation *, ds4_tokens *, ds4_vision_span **,
    char *error, size_t error_size);
int ds4_agent_observation_fits(ds4_engine *, const ds4_tokens *transcript,
    int context_size, const agent_tool_observation *, int reserve, int *tokens_out,
    char *error, size_t error_size);

/* Whole existing actions: tool dispatch (including remaining write/edit/bash/
 * web work) and UI publication remain C until Worker orchestration migrates. */
char *ds4_agent_tool_execute_one(void *worker, const agent_tool_call *);
void ds4_agent_tool_publish(void *worker, const char *, size_t);
#endif
