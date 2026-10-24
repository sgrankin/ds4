/* CPU-only production tokenizer fixture. Emits IDs and decoded byte pieces for
 * comparison with the checkpoint's Hugging Face tokenizers implementation. */
#include "../src/flint_tokenizer.h"
#include "../src/ds4_session_sampling.h"
#include <assert.h>
#include <stdlib.h>

/* Flint ignores EOS using raw logits, retains thinking controls, and
 * stops on developer/user/observation boundaries owned by its tokenizer. */
static void check_selection_policy(const flint_tokenizer *tokenizer) {
    const int count = flint_tokenizer_vocab_size(tokenizer);
    const int start = flint_tokenizer_special_id(tokenizer, "<think>");
    const int end = flint_tokenizer_special_id(tokenizer, "</think>");
    assert(start >= 0 && end >= 0 && !flint_tokenizer_is_stop(tokenizer, 0));
    float *raw = malloc((size_t)count * sizeof(float));
    float *adjusted = malloc((size_t)count * sizeof(float));
    assert(raw && adjusted);
    for (int i = 0; i < count; i++) raw[i] = -10;
    raw[0] = 5; raw[1] = 4;
    const char *stops[] = {"<|end_of_text|>","<|user|>","<|observation|>","<|developer|>"};
    for (size_t i = 0; i < sizeof(stops)/sizeof(stops[0]); i++) {
        const int token = flint_tokenizer_special_id(tokenizer, stops[i]);
        assert(token >= 0);
        raw[token] = 20;
    }
    ds4_vocab vocab = {.eos_id=flint_tokenizer_special_id(tokenizer,"<|end_of_text|>"),
        .endoftext_id=-1, .think_start_id=start, .think_end_id=end};
    ds4_session_frontier frontier = {.logits=raw};
    const ds4_overthink_policy policy = {.penalty=8, .think_only=false, .ids={0}, .count=1};
    const ds4_session_sampling_view view = {.frontier=&frontier, .policy=&policy,
        .vocab=&vocab, .count=(uint32_t)count, .family=DS4_MODEL_FAMILY_DEEPSEEK41, .flint=tokenizer};
    for (size_t i = 0; i < sizeof(stops)/sizeof(stops[0]); i++) {
        const int token = flint_tokenizer_special_id(tokenizer, stops[i]);
        assert(ds4_native_token_is_stop(&vocab, view.family, tokenizer, token));
        assert(ds4_native_token_is_stop_for_think_mode(&vocab, view.family, tokenizer, token, DS4_THINK_NONE));
    }
    assert(ds4_native_session_argmax_ignoring_eos(&view, DS4_THINK_NONE) == 0);
    assert(!frontier.sampling_logits); /* Ignore-EOS keeps the raw path. */
    assert(ds4_native_session_copy_sampling_logits(&view, adjusted) == count);
    assert(adjusted[0] == -3 && raw[0] == 5);
    raw[start] = 8;
    assert(ds4_native_session_argmax_ignoring_eos(&view, DS4_THINK_NONE) == start);
    raw[start] = -10; raw[end] = 8;
    assert(ds4_native_session_argmax_ignoring_eos(&view, DS4_THINK_NONE) == end);
    assert(ds4_native_token_is_thinking_control(&vocab, start));
    assert(!ds4_native_token_is_stop_for_think_mode(&vocab, view.family, tokenizer, start, DS4_THINK_NONE));
    free(frontier.sampling_logits);
    free(adjusted); free(raw);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    flint_tokenizer *tokenizer = flint_tokenizer_load(argv[1]);
    assert(tokenizer && flint_tokenizer_vocab_size(tokenizer) == 150144);
    check_selection_policy(tokenizer);
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
