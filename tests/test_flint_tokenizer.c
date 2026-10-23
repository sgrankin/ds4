/* CPU-only production tokenizer fixture. Emits IDs and decoded byte pieces for
 * comparison with the checkpoint's Hugging Face tokenizers implementation. */
#include "../src/flint_tokenizer.h"
#include <assert.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 3);
    flint_tokenizer *tokenizer = flint_tokenizer_load(argv[1]);
    assert(tokenizer && flint_tokenizer_vocab_size(tokenizer) == 150144);
    assert(flint_tokenizer_special_id(tokenizer, "<|assistant|>") == 150004);
    assert(flint_tokenizer_is_stop(tokenizer, 150001));
    assert(!flint_tokenizer_is_stop(tokenizer, 150004));
    FILE *input = fopen(argv[2], "rb");
    assert(input && !fseek(input, 0, SEEK_END));
    long length = ftell(input);
    assert(length >= 0 && !fseek(input, 0, SEEK_SET));
    char *text = malloc((size_t)length + 1);
    assert(text && fread(text, 1, (size_t)length, input) == (size_t)length);
    fclose(input);
    ds4_tokens tokens = {0};
    assert(flint_tokenizer_encode(tokenizer, text, (size_t)length, &tokens));
    putchar('[');
    for (int i = 0; i < tokens.len; i++) printf("%s%d", i ? "," : "", tokens.v[i]);
    puts("]");
    for (int i = 0; i < tokens.len; i++) {
        size_t bytes = 0;
        const char *piece = flint_tokenizer_piece(tokenizer, tokens.v[i], &bytes);
        assert(piece);
        for (size_t j = 0; j < bytes; j++) printf("%02x", (unsigned char)piece[j]);
        putchar('\n');
    }
    ds4_tokens_free(&tokens);
    flint_tokenizer_free(tokenizer);
    free(text);
    return 0;
}
