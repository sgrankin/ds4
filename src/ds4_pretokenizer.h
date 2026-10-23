#ifndef DS4_PRETOKENIZER_H
#define DS4_PRETOKENIZER_H

#include "ds4_tokenizer.h"

/* Generated Unicode DATA remains the single source for the Qwen35 classes.
 * Zig imports these declarations in its compiler cache, not a copied table. */
#include "ds4_qwen4_unicode.inc"

typedef enum {
    DS4_PRETOKENIZER_JOYAI = 0,
    DS4_PRETOKENIZER_GLM4 = 1,
    DS4_PRETOKENIZER_QWEN35 = 2,
} ds4_pretokenizer_family;

/* Split raw bytes using the selected model's ordered pre-tokenization rules,
 * then append each piece through the selected production BPE implementation.
 * Maps borrow immutable vocabulary keys and are not modified here. Input may
 * be NULL when len==0; output retains its existing tokens.
 *
 * false reports a BPE allocation failure (earlier pieces may already have
 * appended tokens), or an unknown family. The engine wrapper keeps allocation
 * admission and its fatal OOM policy. This API does not validate/normalize UTF-8:
 * it deliberately retains the original permissive stepping rules.
 */
bool ds4_pretokenize(ds4_pretokenizer_family family,
                    const ds4_token_map *tokens, const ds4_token_map *merges,
                    const char *raw, uint64_t len, ds4_tokens *out);

#endif
