/* Link the selected production pretokenizer AND maps/BPE. Synthetic ranks make
 * the important split boundaries observable without a model or copied lexer. */
#include "../ds4_pretokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void put(ds4_token_map *map, const char *key, int value) {
    assert(ds4_token_map_put(map, key, strlen(key), value));
}

int main(void) {
    ds4_token_map *tokens = ds4_token_map_create(64);
    ds4_token_map *merges = ds4_token_map_create(64);
    assert(tokens && merges);
    put(tokens, "1", 1); put(tokens, "2", 2); put(tokens, "3", 3); put(tokens, "4", 4);
    put(tokens, "12", 5); put(tokens, "123", 6); put(tokens, "1234", 7);
    put(merges, "1 2", 0); put(merges, "12 3", 1);
    put(merges, "123 4", 2); /* Must NEVER bridge the three-digit split. */
    put(tokens, ">", 10); put(tokens, ";", 11);
    put(tokens, "\xc4\x8a", 12); /* GPT-2 newline U+010A. */
    put(tokens, ">;", 19); put(tokens, ">;\xc4\x8a", 20);
    put(merges, "> ;", 3); put(merges, ">; \xc4\x8a", 4);
    put(tokens, "\xc4\xa0", 13); /* GPT-2 space U+0120. */
    put(tokens, "i", 14); put(tokens, "n", 15); put(tokens, "t", 16);
    put(tokens, "\xc4\xa0\xc4\xa0", 29);
    put(tokens, "\xc4\xa0\xc4\xa0\xc4\xa0", 30);
    put(tokens, "\xc4\xa0\xc4\xa0\xc4\xa0\xc4\xa0", 31);
    put(merges, "\xc4\xa0 \xc4\xa0", 5);
    put(merges, "\xc4\xa0\xc4\xa0 \xc4\xa0", 6);
    put(merges, "\xc4\xa0\xc4\xa0\xc4\xa0 \xc4\xa0", 7);
    put(tokens, "\xc4\xa0" "i", 38);
    put(tokens, "\xc4\xa0" "in", 39);
    put(tokens, "\xc4\xa0" "int", 40);
    put(merges, "\xc4\xa0 i", 8);
    put(merges, "\xc4\xa0" "i n", 9);
    put(merges, "\xc4\xa0" "in t", 10);

    const char input[] = "1234>;\n    int";
    const int grouped[] = {99, 6, 4, 20, 30, 40};
    const int individual[] = {99, 1, 2, 3, 4, 20, 30, 40};
    const ds4_pretokenizer_family families[] = {
        DS4_PRETOKENIZER_JOYAI, DS4_PRETOKENIZER_GLM4, DS4_PRETOKENIZER_QWEN35
    };
    for (size_t i = 0; i < sizeof(families) / sizeof(*families); i++) {
        ds4_tokens out = {0};
        ds4_tokens_push(&out, 99);
        assert(ds4_pretokenize(families[i], tokens, merges, NULL, 0, &out));
        assert(out.len == 1 && out.v[0] == 99);
        assert(ds4_pretokenize(families[i], tokens, merges, input, sizeof(input)-1, &out));
        const int *expected = i == 2 ? individual : grouped;
        size_t count = i == 2 ? sizeof(individual)/sizeof(*individual) : sizeof(grouped)/sizeof(*grouped);
        assert(out.len == (int)count && !memcmp(out.v, expected, count * sizeof(*expected)));
        ds4_tokens_free(&out);
    }
    /* C3 followed by ASCII '1' is malformed UTF-8. The original permissive
     * decoder still consumes both bytes as U+00F1, a letter in all three
     * policies. Its BPE piece therefore retains the following raw digit;
     * validating/falling back to one byte would split it and lose this merge. */
    put(tokens, "\xc3\x83", 50); /* GPT-2 encoding of raw byte C3. */
    put(tokens, "\xc3\x83" "1", 51);
    put(merges, "\xc3\x83 1", 20);
    for (size_t i = 0; i < sizeof(families) / sizeof(*families); i++) {
        ds4_tokens out = {0};
        assert(ds4_pretokenize(families[i], tokens, merges, "\xc3" "1", 2, &out));
        assert(out.len == 1 && out.v[0] == 51);
        ds4_tokens_free(&out);
    }
    ds4_token_map_free(merges);
    ds4_token_map_free(tokens);
    puts("JoyAI/GLM4/Qwen35 numeric, newline and leading-space boundaries: PASS");
    return 0;
}
