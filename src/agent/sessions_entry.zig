//! Sole C facade for reusable saved-session identity/file/catalog actions.
const c = @import("agent_sessions_c");
const Sessions = @import("sessions.zig");
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (pointer == null or length == 0) &.{} else pointer[0..length];
}
export fn ds4_agent_sessions_identity_clear(identity: *c.ds4_agent_session_identity) void {
    Sessions.clearIdentity(identity);
}
export fn ds4_agent_sessions_identity_sha(title: [*c]const u8, created_at: u64, output: [*c]u8) void {
    Sessions.identitySha(title, created_at, output);
}
export fn ds4_agent_sessions_meta_free(meta: *c.agent_kv_session_meta) void {
    Sessions.freeMeta(meta);
}
export fn ds4_agent_sessions_read_text(stream: *c.FILE, count: u32, output: *[*c]u8, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.readText(stream, count, output, errors(error_text, capacity));
}
export fn ds4_agent_sessions_write_title(stream: *c.FILE, title: [*c]const u8, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.writeTitle(stream, title, errors(error_text, capacity));
}
export fn ds4_agent_sessions_read_title(stream: *c.FILE, header: *const c.ds4_kvstore_entry, output: *[*c]u8, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.readTitle(stream, header, output, errors(error_text, capacity));
}
export fn ds4_agent_sessions_file_sha(header: *const c.ds4_kvstore_entry, rendered: [*c]const u8, count: u32, title: [*c]const u8, output: [*c]u8) void {
    Sessions.fileSha(header, rendered, count, title, output);
}
export fn ds4_agent_sessions_title_prompt(prompt: [*c]const u8, budget: usize) [*c]u8 {
    return Sessions.titlePrompt(prompt, budget);
}
export fn ds4_agent_sessions_title_text(rendered: [*c]const u8, length: usize, budget: usize) [*c]u8 {
    return Sessions.titleText(rendered, length, budget);
}
export fn ds4_agent_sessions_list(directory: [*c]const u8, model_id: u8, columns: c_int) void {
    Sessions.list(directory, model_id, columns);
}
export fn ds4_agent_sessions_complete(directory: [*c]const u8, model_id: u8, input: [*c]const u8, output: *c.linenoiseCompletions) void {
    Sessions.complete(directory, model_id, input, output);
}
export fn ds4_agent_sessions_find(directory: [*c]const u8, model_id: u8, prefix: [*c]const u8, sha: [*c]u8, path: *[*c]u8, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.find(directory, model_id, prefix, sha, path, errors(error_text, capacity));
}
export fn ds4_agent_sessions_delete(directory: [*c]const u8, model_id: u8, prefix: [*c]const u8, sha: [*c]u8, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.delete(directory, model_id, prefix, sha, errors(error_text, capacity));
}
export fn ds4_agent_sessions_strip(directory: [*c]const u8, model_id: u8, engine: ?*c.ds4_engine, prefix: [*c]const u8, sha: [*c]u8, tokens: [*c]u32, error_text: [*c]u8, capacity: usize) bool {
    return Sessions.strip(directory, model_id, engine, prefix, sha, tokens, errors(error_text, capacity));
}
