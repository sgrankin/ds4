//! Sole C facade for complete live-state/reuse actions over actual records.
const c = @import("server_reuse_c");
const owner = @import("reuse.zig");

export fn ds4_server_reuse_visible_key(req: ?*const c.request, value: [*c]const u8, images: *c.visible_image_key) [*c]u8 {
    return owner.visibleKey(req, value, images);
}
export fn ds4_server_reuse_images_match(incoming: *const c.visible_image_key, old: *const c.visible_image_key, boundary: usize) bool {
    return owner.imagesMatch(incoming, old, boundary);
}
export fn ds4_server_reuse_profile(runtime: ?*const c.server, req: ?*const c.request, profile: [*c]u8) void {
    owner.profile(runtime, req, profile[0..41]);
}
export fn ds4_server_reuse_tool_clear(state: ?*c.live_tool_state) void {
    owner.clearTool(state);
}
export fn ds4_server_reuse_tool_free(state: ?*c.live_tool_state) void {
    owner.freeTool(state);
}
export fn ds4_server_reuse_visible_clear(state: ?*c.visible_live_state) void {
    owner.clearVisible(state);
}
export fn ds4_server_reuse_visible_free(state: ?*c.visible_live_state) void {
    owner.freeVisible(state);
}
export fn ds4_server_reuse_clear(runtime: ?*c.server, slot: ?*c.server_slot, api: c.api_style) void {
    owner.clear(runtime, slot, api);
}
export fn ds4_server_reuse_remember(runtime: ?*c.server, slot: ?*c.server_slot, api: c.api_style, visible: [*c]const u8, calls: ?*const c.tool_calls, req: ?*const c.request) void {
    owner.remember(runtime, slot, api, visible, calls, req);
}
export fn ds4_server_reuse_matches_locked(slot: *c.server_slot, api: c.api_style, ids: *const c.stop_list, live: c_int) bool {
    return owner.matchesLocked(slot, api, ids, live);
}
export fn ds4_server_reuse_matches(runtime: ?*c.server, slot: ?*c.server_slot, api: c.api_style, ids: ?*const c.stop_list, live: c_int) bool {
    return owner.matches(runtime, slot, api, ids, live);
}
export fn ds4_server_reuse_canonical(runtime: ?*const c.server, req: ?*const c.request) bool {
    return owner.canonical(runtime, req);
}
export fn ds4_server_reuse_probe_locked(runtime: ?*c.server, slot: ?*c.server_slot, req: ?*const c.request) c.slot_reuse {
    return owner.probeLocked(runtime, slot, req);
}
export fn ds4_server_reuse_probe(runtime: *c.server, slot: ?*c.server_slot, req: ?*const c.request) c.slot_reuse {
    return owner.probe(runtime, slot, req);
}
export fn ds4_server_reuse_suffix(runtime: *c.server, slot: *c.server_slot, req: *const c.request, suffix: [*c]const u8, out: *c.ds4_tokens) bool {
    return owner.suffix(runtime, slot, req, suffix, out);
}
export fn ds4_server_reuse_unavailable(req: ?*const c.request, materialized: bool) bool {
    return owner.unavailable(req, materialized);
}
export fn ds4_server_reuse_refresh(runtime: ?*c.server, slot: ?*c.server_slot) void {
    owner.refresh(runtime, slot);
}
export fn ds4_server_reuse_rewind_target(can_rewind: bool, old_pos: c_int, prompt_len: c_int, common: c_int) c_int {
    return owner.rewindTarget(can_rewind, old_pos, prompt_len, common);
}
