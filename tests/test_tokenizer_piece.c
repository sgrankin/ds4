/* Link this fixture to the selected production C or Zig map/BPE implementation.
 * Keys are caller-owned static/stack bytes; no model or GPU is involved. */
#include "../src/ds4_tokenizer.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static ds4_token_map *new_map(void) {
    ds4_token_map *map = ds4_token_map_create(16);
    assert(map);
    return map;
}

static void put(ds4_token_map *map, const char *key, int value) {
    assert(ds4_token_map_put(map, key, strlen(key), value));
}

static void expect(const ds4_tokens *tokens, const int *ids, size_t count) {
    assert(tokens->len == (int)count);
    if (count) assert(!memcmp(tokens->v, ids, count * sizeof(*ids)));
}

static void test_map(void) {
    ds4_token_map *map = new_map();
    const char key[] = {'a', 0, 'b'};
    const char duplicate[] = {'a', 0, 'b'};
    int value = 999;
    assert(!ds4_token_map_get(map, "missing", 7, &value) && value == 999);
    assert(ds4_token_map_put(map, key, sizeof(key), 17));
    assert(ds4_token_map_put(map, duplicate, sizeof(duplicate), 23));
    assert(ds4_token_map_get(map, key, sizeof(key), &value) && value == 23);
    assert(ds4_token_map_put(map, NULL, 0, -7));
    assert(ds4_token_map_get(map, NULL, 0, &value) && value == -7);
    ds4_token_map_free(map); /* Must not free either borrowed stack key. */
    ds4_token_map_free(NULL);
}

static void test_merges(void) {
    ds4_token_map *tokens = new_map(), *merges = new_map();
    put(tokens, "a", 1); put(tokens, "b", 2); put(tokens, "c", 3);
    put(tokens, "ab", 4); put(tokens, "abc", 5); put(tokens, "bc", 6);
    put(merges, "a b", 100); put(merges, "b c", 1);
    put(merges, "a b", 0); /* Duplicate rank replaces 100, so left pair wins. */
    put(merges, "ab c", 2);
    ds4_tokens out = {0};
    ds4_tokens_push(&out, 77);
    assert(ds4_bpe_emit_piece(tokens, merges, NULL, 0, &out));
    expect(&out, (int[]){77}, 1);
    assert(ds4_bpe_emit_piece(tokens, merges, "abc", 3, &out));
    expect(&out, (int[]){77, 5}, 2); /* Append, not replace. */
    ds4_tokens_free(&out);
    ds4_token_map_free(merges); ds4_token_map_free(tokens);
}

static void test_leftmost_tie(void) {
    ds4_token_map *tokens = new_map(), *merges = new_map();
    put(tokens, "a", 1); put(tokens, "aa", 2); put(tokens, "aaa", 3);
    put(merges, "a a", 0); put(merges, "aa a", 1);
    ds4_tokens out = {0};
    assert(ds4_bpe_emit_piece(tokens, merges, "aaa", 3, &out));
    expect(&out, (int[]){3}, 1); /* Rightmost-first would emit {1, 2}. */
    out.len = 0;
    put(merges, "a a", INT_MAX);
    assert(ds4_bpe_emit_piece(tokens, merges, "aa", 2, &out));
    expect(&out, (int[]){1, 1}, 2); /* The sentinel rank is never merged. */
    ds4_tokens_free(&out);
    ds4_token_map_free(merges); ds4_token_map_free(tokens);
}

static void test_utf8_and_fallback(void) {
    ds4_token_map *tokens = new_map(), *merges = new_map();
    /* Raw UTF-8 C3 A9 becomes the GPT-2 code points C3 and A9: C3 83 C2 A9. */
    put(tokens, "\xc3\x83", 20); put(tokens, "\xc2\xa9", 21);
    put(tokens, "\xc3\x83\xc2\xa9", 22);
    put(merges, "\xc3\x83 \xc2\xa9", 0);
    ds4_tokens out = {0};
    assert(ds4_bpe_emit_piece(tokens, merges, "\xc3\xa9", 2, &out));
    expect(&out, (int[]){22}, 1);
    out.len = 0;
    /* Raw zero maps to U+0100 (C4 80). A missing whole symbol falls back to
     * those encoded bytes, and an absent byte is silently skipped. */
    put(tokens, "\xc4", 70);
    assert(ds4_bpe_emit_piece(tokens, merges, "\0", 1, &out));
    expect(&out, (int[]){70}, 1);
    out.len = 0;
    put(tokens, "\x80", 71);
    assert(ds4_bpe_emit_piece(tokens, merges, "\0", 1, &out));
    expect(&out, (int[]){70, 71}, 2);
    ds4_tokens_free(&out);
    ds4_token_map_free(merges); ds4_token_map_free(tokens);
}

int main(void) {
    test_map();
    test_merges();
    test_leftmost_tie();
    test_utf8_and_fallback();
    puts("borrowed token maps and BPE pieces: PASS");
    return 0;
}
