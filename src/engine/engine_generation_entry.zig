//! Native stateless generation entry; all Engine and Session handles stay opaque.
const c = @import("generation_c");
const owner = @import("engine_generation.zig");
export fn ds4_native_engine_generate_argmax(engine: *c.ds4_engine, core: *const c.ds4_engine_core, qwen: ?*const c.ds4_qwen4_config, prompt: ?*const c.ds4_tokens, count: c_int, context: c_int, emit: c.ds4_token_emit_fn, done: c.ds4_generation_done_fn, user: ?*anyopaque, progress: c.ds4_session_progress_fn, progress_user: ?*anyopaque) c_int {
    return owner.generate(engine, core, qwen, prompt, count, context, .{ .emit = emit, .done = done, .user = user, .progress = progress, .progress_user = progress_user });
}
