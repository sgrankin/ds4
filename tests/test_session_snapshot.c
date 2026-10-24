/* CPU qualification of the common envelope over public model payload APIs. */
#include "engine/session_snapshot_api.h"
#include <assert.h>

/* A public-API test double, not the engine's private Session layout. Model
 * codecs are qualified separately; this fixture exercises only FILE lifetime
 * and snapshot/staged publication with a nonzero final payload byte. */
struct ds4_session {
    uint8_t payload[64];
    size_t bytes;
    bool valid, distributed, fail_save;
};
uint64_t ds4_session_payload_bytes(ds4_session *session) {
    return session->valid ? session->bytes : 0;
}
bool ds4_session_checkpoint_valid(const ds4_session *session) {
    return session && session->valid;
}
bool ds4_session_is_distributed(ds4_session *session) {
    return session && session->distributed;
}
int ds4_session_save_payload(ds4_session *session, FILE *stream, char *error, size_t capacity) {
    if (session->fail_save) {
        snprintf(error, capacity, "fixture payload save failure");
        return 1;
    }
    return fwrite(session->payload, 1, session->bytes, stream) != session->bytes;
}
int ds4_session_load_payload(ds4_session *session, FILE *stream, uint64_t bytes, char *error, size_t capacity) {
    (void)error; (void)capacity;
    assert(bytes <= sizeof(session->payload));
    if (fread(session->payload, 1, bytes, stream) != bytes) return 1;
    session->bytes = bytes;
    session->valid = true;
    return 0;
}
static void check_snapshot_envelope(void) {
    ds4_session source = { .valid = true }, destination = {0};
    for (unsigned i = 0; i < sizeof(source.payload); i++) source.payload[i] = (uint8_t)(0x31 + i);
    ds4_session_snapshot snapshot = {0};
    char errors[128] = "";
    /* Reuse the resize sequence from the existing real session fixture. */
    const unsigned lengths[] = {1, 3, 2, 3};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        source.bytes = lengths[i] * sizeof(float);
        assert(ds4_native_session_save_snapshot(&source, &snapshot, errors, sizeof(errors)) == 0);
        assert(snapshot.len == source.bytes && snapshot.cap >= source.bytes + 1);
        assert(memcmp(snapshot.ptr, source.payload, source.bytes) == 0);
        assert(ds4_native_session_load_snapshot(&destination, &snapshot, errors, sizeof(errors)) == 0);
        assert(destination.bytes == source.bytes && memcmp(destination.payload, source.payload, source.bytes) == 0);
    }
    source.fail_save = true;
    const uint64_t saved_length = snapshot.len;
    assert(ds4_native_session_save_snapshot(&source, &snapshot, errors, sizeof(errors)) == 1);
    assert(snapshot.len == saved_length && strcmp(errors, "fixture payload save failure") == 0);
    source.fail_save = false;
    source.distributed = true;
    assert(ds4_native_session_save_snapshot(&source, &snapshot, errors, sizeof(errors)) == 1);
    assert(strcmp(errors, "distributed session snapshots are not supported yet") == 0);
    source.distributed = false;
    ds4_session_snapshot_free(&snapshot);
    assert(snapshot.ptr == NULL && snapshot.len == 0 && snapshot.cap == 0);
    ds4_session_snapshot_free(NULL);

    ds4_session_payload_file staged = {0};
    assert(ds4_native_session_stage_payload(&source, &staged, errors, sizeof(errors)) == 0);
    assert(staged.path && staged.bytes == source.bytes);
    char *path = strdup(staged.path), *wire = NULL;
    size_t bytes = 0;
    assert(path);
    FILE *stream = open_memstream(&wire, &bytes);
    assert(stream && ds4_session_write_staged_payload(&staged, stream, errors, sizeof(errors)) == 0);
    assert(fclose(stream) == 0 && bytes == source.bytes && memcmp(wire, source.payload, bytes) == 0);
    free(wire);
    ds4_session_payload_file_free(&staged);
    assert(staged.path == NULL && staged.bytes == 0 && access(path, F_OK) != 0 && errno == ENOENT);
    free(path);
    source.valid = false;
    staged.bytes = 7319;
    assert(ds4_native_session_stage_payload(&source, &staged, errors, sizeof(errors)) == 1);
    assert(staged.path == NULL && staged.bytes == 0);
    assert(strcmp(errors, "session has no valid checkpoint to stage") == 0);
    ds4_session_payload_file_free(NULL);
}
int main(void) {
    check_snapshot_envelope();
    puts("session snapshot/staged envelope CPU checks passed");
    return 0;
}
