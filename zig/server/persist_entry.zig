//! Sole C facade for complete exact/private/suffix persistence transactions.
const c = @import("server_persist_c");
const owner = @import("persist.zig");

export fn ds4_server_persist_store(runtime: ?*c.server, slot: ?*c.server_slot, tokens: [*c]const c.ds4_tokens, length: c_int, reason: [*c]const u8, visible: [*c]const u8, extension: u8, key: [*c]const u8) bool {
    return owner.store(runtime, slot, tokens, length, reason, visible, extension, key);
}
export fn ds4_server_persist_store_current(runtime: ?*c.server, slot: ?*c.server_slot, reason: [*c]const u8) void {
    owner.storeCurrent(runtime, slot, reason);
}
export fn ds4_server_persist_continued_target(runtime: ?*c.server, slot: ?*c.server_slot, live_tokens: c_int) c_int {
    return owner.continuedTarget(runtime, slot, live_tokens);
}
export fn ds4_server_persist_note_store(slot: ?*c.server_slot, tokens: c_int) void {
    owner.noteStore(slot, tokens);
}
export fn ds4_server_persist_suppress(runtime: ?*c.server, slot: ?*c.server_slot, tokens: c_int) c_int {
    return owner.suppress(runtime, slot, tokens);
}
export fn ds4_server_persist_restore_suppressed(slot: ?*c.server_slot, old: c_int, suppressed: c_int) void {
    owner.restoreSuppressed(slot, old, suppressed);
}
export fn ds4_server_persist_discard(runtime: ?*c.server, slot: ?*c.server_slot, path: [*c]const u8) void {
    owner.discard(runtime, slot, path);
}
export fn ds4_server_persist_maybe_store(runtime: ?*c.server, slot: ?*c.server_slot) void {
    owner.maybeStore(runtime, slot);
}
export fn ds4_server_persist_load(runtime: ?*c.server, slot: ?*c.server_slot, prompt: [*c]const u8, effective: *c.ds4_tokens, path: ?*[*c]u8, extension: ?*u8, responses: bool) c_int {
    return owner.load(runtime, slot, prompt, effective, path, extension, responses);
}
export fn ds4_server_persist_restore_tools(runtime: ?*c.server, messages: ?*const c.chat_msgs) void {
    owner.restoreTools(runtime, messages);
}
export fn ds4_server_persist_load_state(runtime: *c.server, slot: *c.server_slot, prompt: *const c.ds4_tokens) c_int {
    return owner.loadState(runtime, slot, prompt);
}
export fn ds4_server_persist_endpoint(runtime: *c.server, slot: *c.server_slot) void {
    owner.endpoint(runtime, slot);
}
export fn ds4_server_persist_suffix(runtime: *c.server, slot: *c.server_slot, prompt: *const c.ds4_tokens, padded: *c.ds4_tokens, stats: *c.ds4_suffix_relocation_stats, stop: *const volatile c.sig_atomic_t) c_int {
    return owner.suffix(runtime, slot, prompt, padded, stats, stop);
}
