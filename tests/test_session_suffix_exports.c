/* Exercise the real opaque-session export guards without a model or GPU.
 * Including the core gives this test access to the private provenance bit. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "suffix export line %d: %s\n", __LINE__, #x); abort(); } } while (0)

int main(int argc, char **argv) {
    CHECK(argc == 2);
    ds4_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    s->checkpoint_valid = true;
    s->approximate_suffix = true;
    CHECK(ds4_session_is_approximate(s));
    CHECK(ds4_session_payload_bytes(s) == 0);
    CHECK(ds4_session_layer_payload_bytes(s, 0, 0) == 0);
    CHECK(ds4_session_layer_payload_bytes(s, UINT32_MAX, UINT32_MAX) == 0);

    FILE *fp = fopen(argv[1], "w+b");
    CHECK(fp);
    CHECK(fwrite("SENTINEL", 1, 8, fp) == 8);
    char err[160] = {0};
    CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) != 0);
    CHECK(strstr(err, "private approximate suffix"));
    CHECK(ftell(fp) == 8);
    memset(err, 0, sizeof(err));
    CHECK(ds4_session_save_layer_payload(s, fp, 0, 0, err, sizeof(err)) != 0);
    CHECK(strstr(err, "private approximate suffix"));
    CHECK(ftell(fp) == 8);
    CHECK(fflush(fp) == 0 && fseek(fp, 0, SEEK_SET) == 0);
    char bytes[9] = {0};
    CHECK(fread(bytes, 1, sizeof(bytes), fp) == 8);
    CHECK(memcmp(bytes, "SENTINEL", 8) == 0);
    CHECK(fclose(fp) == 0);

    uint8_t storage[16], before[16];
    memset(storage, 0xa5, sizeof(storage));
    memcpy(before, storage, sizeof(before));
    ds4_session_snapshot snapshot = {.ptr = storage, .len = 7, .cap = sizeof(storage)};
    CHECK(ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)) != 0);
    CHECK(snapshot.ptr == storage && snapshot.len == 7 && snapshot.cap == sizeof(storage));
    CHECK(memcmp(storage, before, sizeof(storage)) == 0);

    s->approximate_suffix = false;
    s->checkpoint_valid = false;
    CHECK(!ds4_session_is_approximate(s));
    CHECK(ds4_session_layer_payload_bytes(s, 0, 0) == 0);
    CHECK(ds4_session_save_layer_payload(s, NULL, 0, 0, err, sizeof(err)) != 0);
    CHECK(strstr(err, "invalid session layer payload"));
    free(s);
    puts("PASS real suffix export guards; no model or GPU opened");
    return 0;
}
