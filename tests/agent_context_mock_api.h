/* Existing fixture's FIVE service seams, not copied context algorithms. Use
 * only as the generated agent_context_c input for suffix-mock's native object. */
#define ds4_kvstore_render_tokens_text mock_render
#define ds4_token_text mock_piece
#define ds4_tokenize_rendered_chat mock_tokenize
#define ds4_token_user mock_user
#define ds4_token_assistant mock_assistant_id
#include "src/agent_context_api.h"
#undef ds4_kvstore_render_tokens_text
#undef ds4_token_text
#undef ds4_tokenize_rendered_chat
#undef ds4_token_user
#undef ds4_token_assistant

static inline char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
        const ds4_tokens *tokens, size_t *len) {
    return mock_render(engine, tokens, len);
}
static inline char *ds4_token_text(ds4_engine *engine, int id, size_t *len) {
    return mock_piece(engine, id, len);
}
static inline void ds4_tokenize_rendered_chat(ds4_engine *engine,
        const char *text, ds4_tokens *tokens) {
    mock_tokenize(engine, text, tokens);
}
static inline int ds4_token_user(ds4_engine *engine) { return mock_user(engine); }
static inline int ds4_token_assistant(ds4_engine *engine) {
    return mock_assistant_id(engine);
}
