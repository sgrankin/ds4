#ifndef FLINT_TOKENIZER_H
#define FLINT_TOKENIZER_H

#include "ds4_tokenizer.h"
/* Translate-C supplies these system declarations to the Zig implementation. */
#include <unicode/uregex.h>
#include <unicode/utext.h>

/* Native tokenizer.json loader. The immutable vocabulary may be shared between
 * sessions. Encoding creates its own regex match state and appends to out. */
typedef struct flint_tokenizer flint_tokenizer;
flint_tokenizer *flint_tokenizer_load(const char *tokenizer_json_path);
void flint_tokenizer_free(flint_tokenizer *tokenizer);
bool flint_tokenizer_encode(const flint_tokenizer *tokenizer,
                            const char *text, size_t len, ds4_tokens *out);

/* Borrowed raw bytes, including partial UTF-8 sequences. Invalid IDs return
 * NULL with len=0. Added control tokens retain their literal spellings. */
const char *flint_tokenizer_piece(const flint_tokenizer *tokenizer,
                                 int id, size_t *len);
int flint_tokenizer_vocab_size(const flint_tokenizer *tokenizer);
int flint_tokenizer_special_id(const flint_tokenizer *tokenizer,
                               const char *spelling);
bool flint_tokenizer_is_stop(const flint_tokenizer *tokenizer, int id);
/* libc-owned canonical prompt text for CLI callers. */
char *flint_render_message(const char *role, const char *content);
char *flint_render_chat(const char *system, const char *user);

#endif
