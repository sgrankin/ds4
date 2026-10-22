#ifndef DS4_TOKENIZER_H
#define DS4_TOKENIZER_H

#include "ds4.h"

/* Private tokenizer boundary, not the vocabulary/model layout. Map keys borrow
 * immutable caller bytes (normally GGUF metadata) until map_free. Map storage is
 * owned by the implementation; do not free a handle or its keys directly.
 * Construction/put require one owner; concurrent read-only lookups are allowed
 * after vocabulary loading has finished. */
typedef struct ds4_token_map ds4_token_map;

/* Reserve for the loader's maximum number of entries. NULL/false reports a
 * reservation/allocation failure; the C loader retains its fatal error policy.
 * put overwrites an existing value without copying its key. Keys are byte
 * strings, not necessarily NUL-terminated; NULL is allowed for zero length. */
ds4_token_map *ds4_token_map_create(uint64_t max_entries);
void ds4_token_map_free(ds4_token_map *map);
bool ds4_token_map_put(ds4_token_map *map, const char *key, uint64_t len, int value);
/* On a miss, leave *value unchanged. */
bool ds4_token_map_get(const ds4_token_map *map, const char *key, uint64_t len,
                         int *value);

/* Apply GPT-2 byte-level BPE to one already pre-tokenized raw piece. Append to
 * the caller's token vector; do not replace its existing contents. The maps and
 * their borrowed keys remain immutable throughout the call. raw may be NULL
 * for an empty piece and is otherwise a byte span, not a C string.
 *
 * Returns false on temporary-allocation failure. The engine wrapper performs
 * its allocation-guard admission before this call and retains fatal OOM handling.
 * Token-vector growth still uses ds4_tokens_push and its existing C guard.
 * C and Zig map representations are private: select maps and BPE together.
 */
bool ds4_bpe_emit_piece(const ds4_token_map *tokens, const ds4_token_map *merges,
                         const char *raw, uint64_t len, ds4_tokens *out);

#endif
