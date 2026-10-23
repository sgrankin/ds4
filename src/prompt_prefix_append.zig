//! Replay actual parsed prefix turns through the selected public chat API.
//! The binding namespace is supplied by each frontend, so its real opaque
//! Engine and prefix records keep their generated type identity.
pub fn append(comptime c: type, engine: ?*c.ds4_engine, tokens: *c.ds4_tokens, prefix: ?*const c.ds4_prompt_prefix) void {
    const history = prefix orelse return;
    for (0..history.count) |index| {
        const turn = history.turns[index];
        const assistant = turn.role == c.DS4_PROMPT_PREFIX_ASSISTANT;
        c.ds4_chat_append_message(engine, tokens, if (assistant) "assistant" else "user", turn.content);
        if (assistant and !c.ds4_engine_is_glm_dsa(engine)) c.ds4_tokens_push(tokens, c.ds4_token_eos(engine));
    }
}
