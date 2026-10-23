#ifndef DS4_AGENT_SESSIONS_H
#define DS4_AGENT_SESSIONS_H
#include "ds4.h"
#include "ds4_kvstore.h"
#include "linenoise.h"

/* Actual saved-session identity, embedded once in Worker. title and
 * legacy_path are libc-owned; filenames remain stable as transcripts evolve. */
typedef struct {
    char sha[41];
    char *title;
    uint64_t created_at;
    char *legacy_path;
} ds4_agent_session_identity;

/* Existing load-result metadata, moved once. The optional title is owned;
 * no live Engine/Session/Worker record is duplicated at this boundary. */
typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

void ds4_agent_sessions_identity_clear(ds4_agent_session_identity *);
void ds4_agent_sessions_identity_sha(const char *title, uint64_t created_at, char out[41]);
void ds4_agent_sessions_meta_free(agent_kv_session_meta *);
bool ds4_agent_sessions_read_text(FILE *, uint32_t bytes, char **out, char *error, size_t cap);
bool ds4_agent_sessions_write_title(FILE *, const char *title, char *error, size_t cap);
bool ds4_agent_sessions_read_title(FILE *, const ds4_kvstore_entry *, char **out, char *error, size_t cap);
void ds4_agent_sessions_file_sha(const ds4_kvstore_entry *, const char *text, uint32_t bytes, const char *title, char out[41]);
char *ds4_agent_sessions_title_prompt(const char *, size_t max_bytes);
char *ds4_agent_sessions_title_text(const char *, size_t text_bytes, size_t max_bytes);

/* Catalog actions borrow cache_dir and Engine. find publishes libc-owned path
 * and SHA only on success; delete/strip publish optional outputs only after
 * commit. Completion only appends to the actual vendor completion object. */
void ds4_agent_sessions_list(const char *cache_dir, uint8_t model_id, int columns);
void ds4_agent_sessions_complete(const char *cache_dir, uint8_t model_id, const char *input, linenoiseCompletions *);
bool ds4_agent_sessions_find(const char *cache_dir, uint8_t model_id, const char *prefix, char out[41], char **path, char *error, size_t cap);
bool ds4_agent_sessions_delete(const char *cache_dir, uint8_t model_id, const char *prefix, char out[41], char *error, size_t cap);
bool ds4_agent_sessions_strip(const char *cache_dir, uint8_t model_id, ds4_engine *, const char *prefix, char out[41], uint32_t *tokens, char *error, size_t cap);
#endif
