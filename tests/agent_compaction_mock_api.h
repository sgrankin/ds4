/* Same six existing engine services used by the pure planner. Definitions live
 * in the original suffix fixture; no Context/compaction algorithms are copied. */
#define ds4_tokenize_rendered_chat mock_tokenize
#define ds4_chat_append_message mock_message
#define ds4_chat_append_assistant_prefix mock_assistant_prefix
#define ds4_token_user mock_user
#define ds4_token_assistant mock_assistant_id
#define ds4_engine_is_glm_dsa mock_glm
#include "src/agent_compaction_api.h"
#undef ds4_tokenize_rendered_chat
#undef ds4_chat_append_message
#undef ds4_chat_append_assistant_prefix
#undef ds4_token_user
#undef ds4_token_assistant
#undef ds4_engine_is_glm_dsa

static inline void ds4_tokenize_rendered_chat(ds4_engine *engine,
        const char *text, ds4_tokens *tokens) { mock_tokenize(engine, text, tokens); }
static inline void ds4_chat_append_message(ds4_engine *engine, ds4_tokens *tokens,
        const char *role, const char *text) { mock_message(engine, tokens, role, text); }
static inline void ds4_chat_append_assistant_prefix(ds4_engine *engine,
        ds4_tokens *tokens, ds4_think_mode mode) { mock_assistant_prefix(engine, tokens, mode); }
static inline int ds4_token_user(ds4_engine *engine) { return mock_user(engine); }
static inline int ds4_token_assistant(ds4_engine *engine) { return mock_assistant_id(engine); }
static inline bool ds4_engine_is_glm_dsa(ds4_engine *engine) { return mock_glm(engine); }
