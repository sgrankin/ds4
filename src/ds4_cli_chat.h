#ifndef DS4_CLI_CHAT_H
#define DS4_CLI_CHAT_H

#include "ds4.h"
#include <stdlib.h>

/* Private CLI state shared with the handwritten chat helpers. The caller owns
 * this record; its transcript/image vectors use libc storage. The session is
 * owned here, while the engine remains borrowed from the CLI. */
typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    ds4_vision_span *images;
    size_t image_count;
    size_t image_cap;
    int ctx_size;
    int think_prefix_pos;
    int think_prefix_tokens;
    bool initial_system_text;
} ds4_cli_chat;

void ds4_cli_chat_trim_images(ds4_cli_chat *chat, size_t count);
ds4_vision_span *ds4_cli_chat_add_image(ds4_cli_chat *chat);

/* Replace the prefix immediately after BOS, before system/user text. Reject a
 * change that fills the context before mutating transcript, image positions or
 * the live session. An identical prefix does not invalidate the session. */
bool ds4_cli_chat_apply_think_prefix(ds4_engine *engine, ds4_cli_chat *chat,
                                      ds4_think_mode mode);

/* create_session retains the old session if creation fails. set_ctx deliberately
 * releases it first; failure then leaves session=NULL and ctx_size=0. */
int ds4_cli_chat_create_session(ds4_engine *engine, ds4_cli_chat *chat, int ctx_size);
int ds4_cli_chat_set_ctx(ds4_engine *engine, ds4_cli_chat *chat, int ctx_size);
void ds4_cli_chat_free(ds4_cli_chat *chat);

#endif
