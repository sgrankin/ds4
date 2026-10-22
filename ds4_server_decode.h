#ifndef DS4_SERVER_DECODE_H
#define DS4_SERVER_DECODE_H

#include "ds4_server_protocol.h"

/* The actual sampled-byte recognizer records, extracted once from the server.
 * This is a forgiving sampling-policy recognizer, not the tool validator. */
typedef enum {
    DSML_DECODE_OUTSIDE,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string;
    bool json_escaped;
    server_model_syntax model_syntax;
} dsml_decode_tracker;

void ds4_server_decode_init(dsml_decode_tracker *tracker);
void ds4_server_decode_update(dsml_decode_tracker *tracker,
    const char *sampled, size_t length);
bool ds4_server_decode_is_tool(dsml_decode_state state);
bool ds4_server_decode_is_payload(dsml_decode_state state);
void ds4_server_decode_observe(const dsml_decode_tracker *tracker,
    const char *scan, bool *saw_start, bool *saw_end, bool *orphan_end);
bool ds4_server_decode_boundary_safe(const buf *sampled,
    const dsml_decode_tracker *tracker);

#endif
