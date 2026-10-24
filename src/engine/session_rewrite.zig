//! Canonical token rewrites may extend the current frontier, but replacing
//! sampled tokens requires a real backend rebuild. This CPU-safe policy borrows
//! the actual token frontier and uses the opaque public session API for sync.
const c = @import("gguf_c");

pub fn requiresRebuild(live: c_int, canonical: c_int, common: c_int) bool {
    return live < 0 or canonical < 0 or common < 0 or common > live or
        common > canonical or common < live;
}
fn fail(errors: []u8, text: [:0]const u8) c.ds4_session_rewrite_result {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", text.ptr);
    return c.DS4_SESSION_REWRITE_ERROR;
}

pub fn rewrite(session: ?*c.ds4_session, frontier: ?*const c.ds4_session_frontier, prompt: ?*const c.ds4_tokens, common: c_int, errors: []u8) c.ds4_session_rewrite_result {
    const state = frontier orelse return fail(errors, "missing session or prompt");
    const input = prompt orelse return fail(errors, "missing session or prompt");
    if (session == null) return fail(errors, "missing session or prompt");
    if (input.len <= 0) return fail(errors, "empty prompt");
    if (input.len >= state.ctx_size) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "prompt length %d exceeds context %d (one token of generation room is required)", input.len, state.ctx_size);
        return c.DS4_SESSION_REWRITE_ERROR;
    }
    if (!state.checkpoint_valid) return fail(errors, "session has no valid checkpoint");
    if (common < 0 or common > state.checkpoint.len or common > input.len) return fail(errors, "invalid rewrite prefix");
    for (0..@intCast(common)) |index| {
        if (state.checkpoint.v[index] != input.v[index]) return fail(errors, "rewrite prefix does not match live checkpoint");
    }
    if (common == state.checkpoint.len) {
        return if (c.ds4_session_sync(session, input, if (errors.len == 0) null else errors.ptr, errors.len) == 0)
            c.DS4_SESSION_REWRITE_OK
        else
            c.DS4_SESSION_REWRITE_ERROR;
    }
    if (requiresRebuild(state.checkpoint.len, input.len, common)) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "rewrite needs rebuild: common=%d live=%d canonical=%d", common, state.checkpoint.len, input.len);
        return c.DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }
    return fail(errors, "unexpected canonical rewrite state");
}
