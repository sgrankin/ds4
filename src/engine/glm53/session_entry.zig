//! Session C ABI: borrow the actual state and common frontier for one call.
const c = @import("gguf_c");
const session = @import("session.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    const text: []u8 = if (capacity == 0) &.{} else pointer[0..capacity];
    if (text.len != 0) text[0] = 0;
    return text;
}
export fn ds4_native_glm_session_dense_reset(state: *c.ds4_glm_session_state, pending: *c.ds4_model_speculation_state) void {
    session.resetDense(state, pending);
}
export fn ds4_native_glm_session_dense_cap(state: *c.ds4_glm_session_state, frontier: *const c.ds4_session_frontier) void {
    session.capDense(state, frontier);
}
export fn ds4_native_glm_session_dense_note(state: *c.ds4_glm_session_state, frontier: *const c.ds4_session_frontier, position: u32, count: u32) void {
    session.noteDense(state, frontier, position, count);
}
export fn ds4_native_glm_session_sync(runtime: *c.ds4_glm_runtime, prompt: *const c.ds4_tokens, text: [*c]u8, capacity: usize) c_int {
    return session.sync(runtime, prompt, errors(text, capacity));
}
export fn ds4_native_glm_session_eval(runtime: *c.ds4_glm_runtime, token: c_int, text: [*c]u8, capacity: usize) c_int {
    return session.eval(runtime, token, errors(text, capacity));
}
export fn ds4_native_glm_session_replay(runtime: *c.ds4_glm_runtime, text: [*c]u8, capacity: usize) c_int {
    return session.replay(runtime, errors(text, capacity));
}
export fn ds4_native_glm_session_rewind(runtime: *c.ds4_glm_runtime, position: u32) bool {
    return session.rewind(runtime, position);
}
export fn ds4_native_glm_session_speculate(runtime: *c.ds4_glm_runtime, first: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, exact: bool, accepted: [*c]c_int, accepted_capacity: c_int, text: [*c]u8, capacity: usize) c_int {
    const output: []c_int = if (accepted_capacity <= 0) &.{} else accepted[0..@intCast(accepted_capacity)];
    return session.speculate(runtime, first, eos, temperature, top_k, top_p, min_p, rng, exact, output, errors(text, capacity));
}
