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

#ifdef DS4_NATIVE_RESPONSE_STORE
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
#else

static inline bool ds4_response_store_valid_id(const char *id) {
    if (!id || strncmp(id, "resp_", 5) || !id[5]) return false;
    size_t n = 5;
    for (; id[n]; n++) {
        unsigned char c = (unsigned char)id[n];
        if (n >= DS4_RESPONSE_STORE_ID_MAX ||
            !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return true;
}

static inline int ds4_response_store_error(char *err, size_t errlen,
                                           const char *operation, int code) {
    if (err && errlen) snprintf(err, errlen, "%s: %s", operation, strerror(code));
    return -1;
}

static inline int ds4_response_store_path(char path[PATH_MAX], const char *dir,
                                          const char *id, bool temporary,
                                          char *err, size_t errlen) {
    if (!dir || !*dir || !ds4_response_store_valid_id(id))
        return ds4_response_store_error(err, errlen, "response store path", EINVAL);
    int n = snprintf(path, PATH_MAX, temporary ? "%s/.%s.XXXXXX" : "%s/%s.json", dir, id);
    if (n < 0 || n >= PATH_MAX)
        return ds4_response_store_error(err, errlen, "response store path", ENAMETOOLONG);
    return 0;
}

static inline int ds4_response_store_init(const char *dir, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if (!dir || !*dir) return ds4_response_store_error(err, errlen, "response store directory", EINVAL);
    if (mkdir(dir, 0700) && errno != EEXIST)
        return ds4_response_store_error(err, errlen, "mkdir response store", errno);
    struct stat st;
    if (stat(dir, &st)) return ds4_response_store_error(err, errlen, "stat response store", errno);
    if (!S_ISDIR(st.st_mode)) return ds4_response_store_error(err, errlen, "response store directory", ENOTDIR);
    if (chmod(dir, 0700)) return ds4_response_store_error(err, errlen, "chmod response store", errno);
    return 0;
}

static inline int ds4_response_store_write(const char *dir, const char *id,
                                           const char *json, size_t bytes,
                                           char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if ((!json && bytes) || bytes > DS4_RESPONSE_STORE_MAX_BYTES)
        return ds4_response_store_error(err, errlen, "response store payload", EINVAL);
    char path[PATH_MAX], temporary[PATH_MAX];
    if (ds4_response_store_path(path, dir, id, false, err, errlen) ||
        ds4_response_store_path(temporary, dir, id, true, err, errlen)) return -1;
    int fd = mkstemp(temporary);
    if (fd < 0) return ds4_response_store_error(err, errlen, "create response temporary", errno);
    int failure = 0;
    if (fchmod(fd, 0600)) failure = errno;
    size_t done = 0;
    while (!failure && done < bytes) {
        ssize_t n = write(fd, json + done, bytes - done);
        if (n < 0) { if (errno == EINTR) continue; failure = errno; }
        else if (!n) failure = EIO;
        else done += (size_t)n;
    }
    while (!failure && fsync(fd)) {
        if (errno != EINTR) { failure = errno; break; }
    }
    if (close(fd) && !failure) failure = errno;
    if (!failure && rename(temporary, path)) failure = errno;
    if (failure) {
        unlink(temporary);
        return ds4_response_store_error(err, errlen, "write response store", failure);
    }
    return 0;
}

static inline int ds4_response_store_read(const char *dir, const char *id,
                                          char **json, size_t *bytes,
                                          char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if (json) *json = NULL;
    if (bytes) *bytes = 0;
    if (!json || !bytes) return ds4_response_store_error(err, errlen, "response read outputs", EINVAL);
    char path[PATH_MAX];
    if (ds4_response_store_path(path, dir, id, false, err, errlen)) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return errno == ENOENT ? 1 : ds4_response_store_error(err, errlen, "open response store", errno);
    struct stat st;
    int failure = 0;
    if (fstat(fd, &st)) failure = errno;
    else if (!S_ISREG(st.st_mode) || st.st_size < 0 ||
             (unsigned long long)st.st_size > DS4_RESPONSE_STORE_MAX_BYTES) failure = EFBIG;
    char *buffer = NULL;
    size_t size = failure ? 0 : (size_t)st.st_size;
    if (!failure && !(buffer = malloc(size + 1))) failure = ENOMEM;
    size_t done = 0;
    while (!failure && done < size) {
        ssize_t n = read(fd, buffer + done, size - done);
        if (n < 0) { if (errno == EINTR) continue; failure = errno; }
        else if (!n) failure = EIO;
        else done += (size_t)n;
    }
    if (close(fd) && !failure) failure = errno;
    if (failure) {
        free(buffer);
        return ds4_response_store_error(err, errlen, "read response store", failure);
    }
    buffer[size] = 0;
    *json = buffer;
    *bytes = size;
    return 0;
}

static inline int ds4_response_store_delete(const char *dir, const char *id,
                                            char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    char path[PATH_MAX];
    if (ds4_response_store_path(path, dir, id, false, err, errlen)) return -1;
    if (unlink(path)) return errno == ENOENT ? 1 : ds4_response_store_error(err, errlen, "delete response store", errno);
    return 0;
}

#endif /* DS4_NATIVE_RESPONSE_STORE */

#endif
