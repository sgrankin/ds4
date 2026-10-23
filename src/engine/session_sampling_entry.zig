//! Sole C facade for the CPU-capable Session sampling/frontier policy owner.
const c = @import("gguf_c");
const owner = @import("session_sampling.zig");

export fn ds4_native_session_argmax(view: *const c.ds4_session_sampling_view) c_int {
    return owner.Selection.init(view).argmax();
}
export fn ds4_native_session_argmax_excluding(view: *const c.ds4_session_sampling_view, excluded: c_int) c_int {
    return owner.Selection.init(view).argmaxExcluding(excluded);
}
export fn ds4_native_session_argmax_ignoring_eos(view: *const c.ds4_session_sampling_view, mode: c.ds4_think_mode) c_int {
    return owner.Selection.init(view).argmaxIgnoringEos(mode);
}
export fn ds4_native_session_sample(view: *const c.ds4_session_sampling_view, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: *u64) c_int {
    return owner.Selection.init(view).sample(temperature, top_k, top_p, min_p, rng);
}
export fn ds4_native_session_top_logprobs(view: *const c.ds4_session_sampling_view, output: [*c]c.ds4_token_score, requested: c_int) c_int {
    const selection = owner.Selection.init(view);
    const count: usize = @min(@as(u32, @intCast(requested)), selection.count);
    return selection.topLogprobs(output[0..count]);
}
export fn ds4_native_session_token_logprob(view: *const c.ds4_session_sampling_view, token: c_int, output: *c.ds4_token_score) c_int {
    return owner.Selection.init(view).tokenLogprob(token, output);
}
export fn ds4_native_session_copy_logits(view: *const c.ds4_session_sampling_view, output: [*c]f32) c_int {
    return owner.Selection.init(view).copyRaw(output);
}
export fn ds4_native_session_copy_sampling_logits(view: *const c.ds4_session_sampling_view, output: [*c]f32) c_int {
    return owner.Selection.init(view).copySampling(output);
}
export fn ds4_native_session_set_logits(view: *const c.ds4_session_sampling_view, input: [*c]const f32) c_int {
    return owner.Selection.init(view).setRaw(input);
}
export fn ds4_native_frontier_common_prefix(state: *c.ds4_session_frontier, prompt: *const c.ds4_tokens) c_int {
    return owner.commonPrefix(state, prompt);
}
