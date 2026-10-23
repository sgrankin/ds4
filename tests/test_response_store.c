#define _POSIX_C_SOURCE 200809L
#include "../src/ds4_response_store.h"

#include <assert.h>

/* Standalone, no engine/model linkage. Pass a persistent test directory whose
 * parent exists. Existing response files are replaced only under fixed IDs.
 */
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test-response-store DIRECTORY\n"); return 2; }
    char err[256], path[PATH_MAX], *json = NULL;
    size_t bytes = 0;
    const char *id = "resp_roundtrip_1";
    const char first[] = "{\"response\":{\"id\":\"resp_roundtrip_1\"},\"replay\":[\"old\"]}";
    const char second[] = "{\"response\":{\"id\":\"resp_roundtrip_1\"},\"replay\":[\"new\",\"\\u03bb\"]}";
    assert(ds4_response_store_init(argv[1], err, sizeof(err)) == 0);
    struct stat st;
    assert(stat(argv[1], &st) == 0 && (st.st_mode & 0777) == 0700);
    assert(ds4_response_store_delete(argv[1], id, err, sizeof(err)) >= 0);
    assert(ds4_response_store_read(argv[1], id, &json, &bytes, err, sizeof(err)) == 1);
    assert(!json && bytes == 0);
    assert(ds4_response_store_write(argv[1], id, first, sizeof(first) - 1, err, sizeof(err)) == 0);
    assert(ds4_response_store_read(argv[1], id, &json, &bytes, err, sizeof(err)) == 0);
    assert(bytes == sizeof(first) - 1 && !memcmp(json, first, sizeof(first)));
    free(json);
    assert(ds4_response_store_path(path, argv[1], id, false, err, sizeof(err)) == 0);
    assert(stat(path, &st) == 0 && (st.st_mode & 0777) == 0600);
    int old = open(path, O_RDONLY);
    assert(old >= 0);
    assert(ds4_response_store_write(argv[1], id, second, sizeof(second) - 1, err, sizeof(err)) == 0);
    assert(ds4_response_store_read(argv[1], id, &json, &bytes, err, sizeof(err)) == 0);
    assert(bytes == sizeof(second) - 1 && !memcmp(json, second, sizeof(second)));
    free(json);
    /* An already-open reader retains the complete old inode across rename. */
    char old_bytes[sizeof(first)];
    assert(read(old, old_bytes, sizeof(old_bytes)) == (ssize_t)(sizeof(first) - 1));
    assert(!memcmp(old_bytes, first, sizeof(first) - 1));
    assert(close(old) == 0);
    assert(!ds4_response_store_valid_id("resp_../escape"));
    assert(!ds4_response_store_valid_id("resp_"));
    assert(ds4_response_store_write(argv[1], "resp_../escape", first, sizeof(first) - 1, err, sizeof(err)) == -1);
    assert(ds4_response_store_read(argv[1], "resp_../escape", &json, &bytes, err, sizeof(err)) == -1);
    assert(!json && bytes == 0);
    assert(ds4_response_store_delete(argv[1], id, err, sizeof(err)) == 0);
    assert(ds4_response_store_delete(argv[1], id, err, sizeof(err)) == 1);
    assert(ds4_response_store_read(argv[1], id, &json, &bytes, err, sizeof(err)) == 1);
    puts("Response store atomic roundtrip/missing/delete/unsafe ID: PASS");
    return 0;
}
