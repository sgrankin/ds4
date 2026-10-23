//! Sole C facade for the private native transaction owner.
const c = @import("agent_sessions_c");
const owner = @import("checkpoint.zig");
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (pointer == null or length == 0) &.{} else pointer[0..length];
}
fn native(load: *c.ds4_agent_checkpoint_load) *owner.Load {
    return @ptrCast(@alignCast(load));
}
export fn ds4_agent_checkpoint_prepare_load(engine: ?*c.ds4_engine, session: ?*c.ds4_session, path: [*c]const u8, expected_sha: [*c]const u8, expected_text: [*c]const u8, expected_length: usize, tensor_parallel: bool, rebuilt: *c.ds4_tokens, output: *?*c.ds4_agent_checkpoint_load, error_text: [*c]u8, capacity: usize) c_int {
    const load = owner.Load.prepare(engine, session, path, expected_sha, expected_text, expected_length, tensor_parallel, rebuilt, errors(error_text, capacity)) orelse return -1;
    output.* = @ptrCast(load);
    return if (load.rebuild) 1 else 0;
}
export fn ds4_agent_checkpoint_finish_load(load: *c.ds4_agent_checkpoint_load, synchronized: bool, loaded: [*c]c.ds4_tokens, metadata: [*c]c.agent_kv_session_meta, error_text: [*c]u8, capacity: usize) bool {
    return native(load).finish(synchronized, loaded, metadata, errors(error_text, capacity));
}
export fn ds4_agent_checkpoint_free_load(load: ?*c.ds4_agent_checkpoint_load) void {
    if (load) |value| native(value).deinit();
}
export fn ds4_agent_checkpoint_save(engine: ?*c.ds4_engine, session: ?*c.ds4_session, path: [*c]const u8, tokens: *const c.ds4_tokens, reason: [*c]const u8, sha: [*c]u8, title: [*c]const u8, created_at: u64, text_only: bool, error_text: [*c]u8, capacity: usize) bool {
    return owner.save(engine, session, path, tokens, reason, sha, title, created_at, text_only, errors(error_text, capacity));
}

export fn ds4_agent_checkpoint_requires_rebuild(payload_bytes: u64, tensor_parallel: bool) bool {
    return owner.requiresRebuild(payload_bytes, tensor_parallel);
}
