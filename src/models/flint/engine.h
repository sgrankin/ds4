#ifndef DS4_FLINT_ENGINE_H
#define DS4_FLINT_ENGINE_H
#include "../../ds4.h"
#include "../../flint_tokenizer.h"
typedef struct flint_engine flint_engine;
typedef struct flint_session flint_session;
bool flint_engine_detect(const char *directory);
flint_engine *flint_engine_open(const char *directory, uint32_t chunk,
                               const char *profile,
                               char *error, size_t error_size);
void flint_engine_close(flint_engine *engine);
const flint_tokenizer *flint_engine_tokenizer(flint_engine *engine);
uint32_t flint_engine_prefill_chunk(flint_engine *engine);
uint64_t flint_engine_weight_bytes(flint_engine *engine);
/* Planned per-session KV, decoder/prefill scratch and prefill-base snapshot. */
ds4_context_memory flint_engine_context_memory(flint_engine *engine,
                                               uint32_t context);
flint_session *flint_session_create(flint_engine *engine, uint32_t context);
void flint_session_free(flint_session *session);
float *flint_session_logits(flint_session *session);
uint32_t flint_session_position(flint_session *session);
uint32_t flint_session_prefill_position(flint_session *session);
/* sync/eval: 0 success, 1 error, DS4_SESSION_SYNC_INTERRUPTED cancellation. */
int flint_session_sync(flint_session *session, const ds4_tokens *tokens,
                       ds4_session_progress_fn progress, void *progress_ud,
                       ds4_session_cancel_fn cancel, void *cancel_ud,
                       char *error, size_t error_size);
int flint_session_eval(flint_session *session, int token,
                       char *error, size_t error_size);
/* Return committed input tokens: first_token, then predictions which were fed
 * back through the model. The final uncommitted prediction remains in logits.
 * Greedy uses a resident packet; filtered sampling may return a scalar step. */
int flint_session_packet(flint_session *session, int first_token,
                         uint32_t max_tokens, float temperature, int top_k,
                         float top_p, float min_p, uint64_t *rng, int *accepted,
                         char *error, size_t error_size);
void flint_session_reset(flint_session *session);
#endif
