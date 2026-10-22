#ifndef DS4_AGENT_FILE_H
#define DS4_AGENT_FILE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define DS4_AGENT_FILE_MAX_BYTES (16 * 1024 * 1024)

/* Private agent file boundary: no worker/model records. Only regular files
 * open; callers own returned FILE and libc byte buffers. Successful reads
 * include a trailing NUL, with length authoritative for embedded NULs. Failed
 * reads leave both outputs untouched. */
FILE *ds4_agent_file_open_regular(const char *path);
int ds4_agent_file_read_bytes(const char *path, char **data, size_t *len,
                              char *err, size_t errlen);

/* Commit complete files, preserving current symlink/metadata/version policy.
 * expected != NULL verifies exact old bytes before publication. Returns0/-1;
 * failed writes clean their temporary file and preserve the old destination. */
int ds4_agent_file_replace(const char *path, const char *data, size_t len,
                            const char *expected, size_t expected_len,
                            char *err, size_t errlen);

/* Search bounded binary file bytes; matches borrow data. Overlapping matches
 * make an anchor nonunique. Output pointers change only on successful search.
 * find_old_span also sets anchored=false before an ordinary (non-upto) search,
 * including when that ordinary search subsequently fails. */
bool ds4_agent_file_find_unique(const char *data, size_t len,
                                const char *needle, size_t needle_len,
                                const char **match, const char *label,
                                char *err, size_t errlen);
bool ds4_agent_file_find_old_span(const char *data, size_t len,
                                  const char *old, bool allow_upto,
                                  const char **match, size_t *match_len,
                                  bool *anchored, char *err, size_t errlen);

#endif
