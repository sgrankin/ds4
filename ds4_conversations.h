#ifndef DS4_CONVERSATIONS_H
#define DS4_CONVERSATIONS_H

/* These records outlive model slots and response files. Manual item edits
 * deliberately discard private replay; their visible items remain available
 * for a new rendered context. */
#include "ds4_server_input.h"

typedef struct {
    char *object, *items, *replay;
    unsigned long long version;
    bool deleted;
} ds4_conversation;

/* These are complete storage/route actions, independent of Server/Session.
 * Owned record strings and returned JSON use libc; statuses match HTTP. */
char *ds4_conversation_json_field(const char *object, const char *wanted);
void ds4_conversation_free(ds4_conversation *conversation);
int ds4_conversation_load(const char *root, const char *id,
    ds4_conversation *conversation, char *error, size_t error_capacity);
bool ds4_conversation_input_items(const char *raw, char **normalized,
    char *error, size_t error_capacity);
int ds4_conversation_publish(const char *root, const char *id,
    unsigned long long expected_version, const char *input, const char *output,
    const char *replay, char *error, size_t error_capacity);
int ds4_conversation_route(const char *root, const char *method,
    const char *path, const char *body, char **public_json,
    char *error, size_t error_capacity);

#endif
