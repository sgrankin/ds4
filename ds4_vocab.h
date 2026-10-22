#ifndef DS4_VOCAB_H
#define DS4_VOCAB_H

#include "ds4_gguf.h"
#include "ds4_pretokenizer.h"

/* Private vocabulary record, unchanged from the engine. Token strings and map
 * keys borrow the model mapping; vocab_free releases only the token array and
 * two map handles, before that mapping is closed. Chat functions borrow all
 * fields and append to the caller-owned C token vector. No model/engine graph
 * layout or mutable global family selection crosses this boundary. */
typedef struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int system_id;
    int user_id;
    int assistant_id;
    int observation_id;
    int sop_id;
    int think_start_id;
    int think_end_id;
    int tool_call_start_id;
    int tool_call_end_id;
    int tool_response_start_id;
    int tool_response_end_id;
    int arg_key_start_id;
    int arg_key_end_id;
    int arg_value_start_id;
    int arg_value_end_id;
    int dsml_id;
    int im_start_id;
    int im_end_id;
    int endoftext_id;
    ds4_token_map *token_to_id;
    ds4_token_map *merge_rank;
} ds4_vocab;

typedef enum {
    DS4_CHAT_DEEPSEEK4 = 0,
    DS4_CHAT_DEEPSEEK41 = 1,
    DS4_CHAT_GLM4 = 2,
    DS4_CHAT_QWEN35 = 3,
} ds4_chat_family;

/* Preserve the engine allocation-admission policy before nonempty BPE spans.
 * The engine implements this narrow helper; token-vector growth keeps its
 * existing C admission and fatal OOM behavior. */
void ds4_vocab_admit_text(uint64_t bytes);

void ds4_vocab_chat_begin(const ds4_vocab *vocab, ds4_chat_family family,
                         ds4_tokens *out);
bool ds4_vocab_chat_think_prefix(const ds4_vocab *vocab, ds4_chat_family family,
                                ds4_think_mode mode, ds4_tokens *out);
/* NULL is success; failure is a borrowed static diagnostic. Public engine
 * wrappers keep their existing fatal error policy. All bool failures below
 * mean temporary BPE allocation failure; earlier output remains appended. */
const char *ds4_vocab_encode_chat(const ds4_vocab *vocab, ds4_chat_family family,
                                 const char *system, const char *prompt,
                                 ds4_think_mode mode, ds4_tokens *out);
bool ds4_vocab_tokenize_rendered(const ds4_vocab *vocab, ds4_chat_family family,
                                 const char *text, ds4_tokens *out);
bool ds4_vocab_chat_append(const ds4_vocab *vocab, ds4_chat_family family,
                          const char *role, const char *content, ds4_tokens *out);
bool ds4_vocab_assistant_prefix(const ds4_vocab *vocab, ds4_chat_family family,
                               ds4_think_mode mode, ds4_tokens *out);
/* Used by both ordinary and existing C multimodal tool-message assembly. The
 * closing wrapper is a C string; literal payload control spellings stay data. */
bool ds4_vocab_tokenize_payload(const ds4_vocab *vocab, ds4_chat_family family,
                               const char *content, const char *closing,
                               ds4_tokens *out);

/* C validates/locates these two GGUF string arrays. Cursors and their backing
 * mapping remain caller-owned; construction reads independent cursor copies.
 * Negative GLM ids request the original spelling-based fallback. */
typedef struct {
    ds4_cursor tokens, merges;
    uint64_t token_count, merge_count;
    ds4_chat_family family;
    int glm_bos_id, glm_eos_id;
} ds4_vocab_source;

/* out starts empty. On success it owns a libc token array and two map handles,
 * whose keys/strings borrow source mapping bytes until release. On failure it
 * is zeroed and error holds the original fatal diagnostic (without "ds4: ").
 * C retains metadata lookup, allocation admission and fatal error policy. */
bool ds4_vocab_build(ds4_vocab *out, const ds4_vocab_source *source, char error[256]);
void ds4_vocab_release(ds4_vocab *vocab);
/* Result is libc-owned, including an allocated empty string for invalid ids.
 * NULL reports allocation failure; len receives decoded byte length on success.
 * The returned bytes may contain NUL; len is authoritative. */
char *ds4_vocab_decode_token(const ds4_vocab *vocab, int token, size_t *len);
void ds4_vocab_admit_allocation(const char *kind, uint64_t bytes);

#endif
