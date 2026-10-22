/* Native production lifecycle API; no copied vocabulary body or mock maps. */
#include "ds4_vocab.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t string(unsigned char *data, size_t pos, const char *bytes, size_t len) {
    const uint64_t count = len;
    memcpy(data + pos, &count, sizeof(count));
    memcpy(data + pos + sizeof(count), bytes, len);
    return pos + sizeof(count) + len;
}
static void decoded(ds4_vocab *vocab, int id, const char *expected, size_t size) {
    size_t length = SIZE_MAX;
    char *bytes = ds4_vocab_decode_token(vocab, id, &length);
    assert(bytes && length == size && !memcmp(bytes, expected, size) && bytes[size] == 0);
    free(bytes);
}
int main(void) {
    unsigned char tokens[512], merges[64];
    const char *pieces[] = {"<sop>", "<|endoftext|>", "<|user|>", "<|assistant|>",
        "<think>", "</think>", "A", ("\xc4\x80" "A"), "\xef\xbd\x9c", "\xc0\xa1", "A"};
    size_t token_bytes = 0, merge_bytes = 0;
    for (unsigned i = 0; i < sizeof(pieces) / sizeof(pieces[0]); ++i)
        token_bytes = string(tokens, token_bytes, pieces[i], strlen(pieces[i]));
    merge_bytes = string(merges, merge_bytes, "A A", 3);
    merge_bytes = string(merges, merge_bytes, "A A", 3);
    const ds4_vocab_source source = {
        .tokens = {.base = tokens, .size = token_bytes},
        .merges = {.base = merges, .size = merge_bytes},
        .token_count = sizeof(pieces) / sizeof(pieces[0]), .merge_count = 2,
        .family = DS4_CHAT_GLM4, .glm_bos_id = 6, .glm_eos_id = 0,
    };
    ds4_vocab vocab;
    char error[256];
    assert(ds4_vocab_build(&vocab, &source, error));
    assert(!error[0] && vocab.n_vocab == (int)source.token_count);
    assert(vocab.bos_id == 6 && vocab.eos_id == 0 && vocab.observation_id == -1);
    assert(source.tokens.pos == 0 && source.merges.pos == 0);
    for (int i = 0; i < vocab.n_vocab; ++i)
        assert(vocab.token[i].ptr >= (const char *)tokens &&
               vocab.token[i].ptr + vocab.token[i].len <= (const char *)tokens + token_bytes);
    int id = -1;
    assert(ds4_token_map_get(vocab.token_to_id, "A", 1, &id) && id == 10);
    assert(ds4_token_map_get(vocab.merge_rank, "A A", 3, &id) && id == 1);
    decoded(&vocab, 6, "A", 1);
    decoded(&vocab, 7, "\0A", 2); /* Encoded GPT-2 byte0, not an empty C string. */
    decoded(&vocab, 8, "\xef\xbd\x9c", 3); /* Literal fullwidth control marker. */
    decoded(&vocab, 9, "!", 1); /* Preserve C's permissive overlong decoding. */
    decoded(&vocab, -1, "", 0);
    decoded(&vocab, vocab.n_vocab, "", 0);
    ds4_vocab_release(&vocab);
    const ds4_vocab empty = {0};
    assert(!memcmp(&vocab, &empty, sizeof(vocab)));

    ds4_vocab_source fallback = source;
    fallback.glm_bos_id = fallback.glm_eos_id = -1;
    assert(ds4_vocab_build(&vocab, &fallback, error));
    assert(vocab.bos_id == 0 && vocab.eos_id == 1);
    ds4_vocab_release(&vocab);

    ds4_vocab_source truncated = source;
    truncated.merges.size--; /* Fail after both maps and all token spans exist. */
    assert(!ds4_vocab_build(&vocab, &truncated, error));
    assert(strstr(error, "truncated GGUF file at byte ") == error);
    assert(!memcmp(&vocab, &empty, sizeof(vocab)));
    assert(ds4_vocab_build(&vocab, &source, error));
    ds4_vocab_release(&vocab);
    puts("PASS: native vocabulary borrowed strings, duplicate maps, byte codec and partial-failure release");
}
