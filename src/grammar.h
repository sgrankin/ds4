#ifndef DS4_GRAMMAR_H
#define DS4_GRAMMAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_grammar_compiler ds4_grammar_compiler;
typedef struct ds4_grammar_compiled ds4_grammar_compiled;
typedef struct ds4_grammar_matcher ds4_grammar_matcher;

typedef struct {
    const uint8_t *bytes;
    size_t len;
} ds4_grammar_piece;

typedef enum {
    DS4_GRAMMAR_STRUCTURAL = 0,
    DS4_GRAMMAR_JSON_SCHEMA = 1,
    DS4_GRAMMAR_REGEX = 2,
} ds4_grammar_kind;

/* Vocabulary pieces are decoded bytes, not GPT-2 alphabet strings. They may
 * contain NUL. Construction copies them; the caller retains its input arrays.
 * One compiler belongs to one model vocabulary. Its compiled cache is bounded. */
ds4_grammar_compiler *ds4_grammar_compiler_new(
    const ds4_grammar_piece *pieces, size_t vocab_size,
    const int32_t *stops, size_t stop_count, char *error, size_t error_size);
/* Token-aware compilers reserve these non-stop control IDs for TokenFormat
 * nodes. Literal marker spellings assembled from ordinary IDs remain data. */
ds4_grammar_compiler *ds4_grammar_compiler_new_token_aware(
    const ds4_grammar_piece *pieces, size_t vocab_size,
    const int32_t *stops, size_t stop_count,
    const int32_t *controls, size_t control_count, char *error, size_t error_size);
void ds4_grammar_compiler_free(ds4_grammar_compiler *compiler);

/* Compiled grammars are immutable and can outlive the compiler. strict_json
 * selects XGrammar's JSON-schema strict mode; structural tags carry their own
 * schema policy. Unsupported schema promises must be checked by the adapter. */
ds4_grammar_compiled *ds4_grammar_compile(
    ds4_grammar_compiler *compiler, ds4_grammar_kind kind,
    const char *source, size_t source_len, bool strict_json,
    char *error, size_t error_size);
void ds4_grammar_compiled_free(ds4_grammar_compiled *compiled);
size_t ds4_grammar_compiled_vocab_size(const ds4_grammar_compiled *compiled);

/* Each generation owns a matcher. It shares the immutable compiled grammar,
 * which may be released after matcher construction. All masks have exactly
 * ceil(vocab_size/32) uint32 words. Bit one means the token is allowed. */
ds4_grammar_matcher *ds4_grammar_matcher_new(
    const ds4_grammar_compiled *compiled, char *error, size_t error_size);
void ds4_grammar_matcher_free(ds4_grammar_matcher *matcher);
/* accept: 1 accepted, 0 rejected, -1 native error. fill: 1 constrained mask,
 * 0 all tokens allowed, -1 native error. No C++ exception crosses this ABI. */
int ds4_grammar_accept(ds4_grammar_matcher *matcher, int32_t token,
                       char *error, size_t error_size);
int ds4_grammar_fill(ds4_grammar_matcher *matcher, uint32_t *words,
                    size_t word_count, char *error, size_t error_size);
bool ds4_grammar_rollback(ds4_grammar_matcher *matcher, size_t tokens,
                          char *error, size_t error_size);
bool ds4_grammar_completed(const ds4_grammar_matcher *matcher);
bool ds4_grammar_terminated(const ds4_grammar_matcher *matcher);

#ifdef __cplusplus
}
#endif
#endif
