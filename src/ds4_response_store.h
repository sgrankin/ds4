#ifndef DS4_RESPONSE_STORE_H
#define DS4_RESPONSE_STORE_H

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Raw JSON files, independent of model/cache lifetime. The caller creates the
 * parent directory and owns JSON parsing. Read buffers belong to the caller.
 * Writes publish by same-directory rename; readers see an old or new whole
 * file. Return 0 on success, 1 on missing read/delete, -1 on other errors.
 */
#define DS4_RESPONSE_STORE_MAX_BYTES ((size_t)64 * 1024 * 1024)
#define DS4_RESPONSE_STORE_ID_MAX 40

bool ds4_response_store_valid_id(const char *id);
int ds4_response_store_error(char *err, size_t errlen, const char *operation, int code);
int ds4_response_store_path(char path[PATH_MAX], const char *dir,
                            const char *id, bool temporary, char *err, size_t errlen);
int ds4_response_store_init(const char *dir, char *err, size_t errlen);
int ds4_response_store_write(const char *dir, const char *id,
                             const char *json, size_t bytes, char *err, size_t errlen);
int ds4_response_store_read(const char *dir, const char *id,
                            char **json, size_t *bytes, char *err, size_t errlen);
int ds4_response_store_delete(const char *dir, const char *id, char *err, size_t errlen);

#endif
