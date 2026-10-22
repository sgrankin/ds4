#ifndef DS4_SERVER_TOOL_MEMORY_H
#define DS4_SERVER_TOOL_MEMORY_H

#include "ds4_server_runtime.h"

#define DS4_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define DS4_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u

/* Exact sampled DSML is immutable and shared by all IDs in one tool block.
 * Entries own their ID; the last entry releases the block. These are the actual
 * runtime records, not a second cache or a copy of the Server layout.
 *
 * The caller retains Server enable/disable policy. Actions borrow its existing
 * tool mutex: attachment and trailer writing hold it throughout; loading reads
 * outside the lock and inserts each complete record under the lock. Returned
 * message text is libc-owned. free requires a quiescent owner. */
void ds4_server_tool_memory_free(tool_memory *memory);
bool ds4_server_tool_memory_contains(tool_memory *memory,
    pthread_mutex_t *mutex, const char *id);
void ds4_server_tool_memory_put(tool_memory *memory, pthread_mutex_t *mutex,
    const char *id, const char *dsml, tool_memory_source source);
void ds4_server_tool_memory_remember(tool_memory *memory,
    pthread_mutex_t *mutex, const tool_calls *calls);
void ds4_server_tool_memory_attach(tool_memory *memory, pthread_mutex_t *mutex,
    chat_msgs *messages, tool_replay_stats *stats);

/* KTM1 is the existing optional KV trailer. Only blocks present verbatim in
 * saved text are written. Empty maps write no header. A truncated load keeps
 * and counts earlier complete, wanted records; it does not roll them back.
 * FILE streams and wanted IDs are borrowed; a NULL wanted list means all IDs.
 * Allocation failure retains the server's fatal out-of-memory policy. */
bool ds4_server_tool_memory_size(tool_memory *memory, pthread_mutex_t *mutex,
    const char *text, uint64_t *bytes);
bool ds4_server_tool_memory_write(tool_memory *memory, pthread_mutex_t *mutex,
    FILE *file, const char *text, uint64_t *written_bytes);
int ds4_server_tool_memory_load(tool_memory *memory, pthread_mutex_t *mutex,
    FILE *file, const stop_list *wanted);

#endif
