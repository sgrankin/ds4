#ifndef DS4_SERVER_MATERIALIZE_H
#define DS4_SERVER_MATERIALIZE_H

#include "ds4_server_render.h"
#include "ds4_server_tool_memory.h"

/* Consumes the actual decoded Input fields. Success transfers one complete
 * Request to out; failure clears out and frees every consumed field. Input
 * deinit remains the caller's responsibility for fields not consumed here.
 * Engine is opaque and Server is its one actual shared runtime record. */
bool ds4_server_materialize(ds4_engine *engine, server *runtime,
    ds4_server_input_endpoint endpoint, ds4_server_input *input, int context,
    bool incremental_schemas, request *out, char *error, size_t capacity);

/* These whole logical actions also serve the existing C generation/routes and
 * focused server tests; there is no second validator or stored-record parser. */
bool ds4_server_materialize_validate(server *runtime, const chat_msgs *messages,
    api_style api, ds4_think_mode mode, bool *requires_live,
    bool *requires_reasoning, char *error, size_t capacity);
void ds4_server_materialize_live(server *runtime, request *req,
    const chat_msgs *messages);
char *ds4_server_response_string_field(const char *object, const char *wanted);
int ds4_server_response_record_load(const char *directory, const char *id,
    char **response, char **replay, char *error, size_t capacity);

/* Remaining complete resource/library actions. Native code owns all request
 * policy and DS4.1 rendering; C retains other-family templates, image inference
 * and disk KV trailer scanning. None returns individual Server field getters. */
char *ds4_server_materialize_render_other(server_model_syntax syntax,
    const chat_msgs *messages, int start, const char *schemas,
    const tool_schema_orders *orders, ds4_think_mode mode, bool live_tail);
bool ds4_server_materialize_tokenize(ds4_engine *engine, server *runtime,
    request *req, const chat_msgs *messages, char *error, size_t capacity);
void ds4_server_materialize_restore_tools(server *runtime,
    const chat_msgs *messages);

#endif
