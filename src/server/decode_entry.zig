//! Sole sampled-byte tracker C facade; no generation/session layout here.
const c = @import("server_decode_c");
const owner = @import("decode.zig");

export fn ds4_server_decode_init(tracker: *c.dsml_decode_tracker) void {
    owner.init(tracker);
}
export fn ds4_server_decode_update(tracker: ?*c.dsml_decode_tracker, sampled: [*c]const u8, length: usize) void {
    if (sampled == null) return;
    owner.update(tracker orelse return, sampled[0..length]);
}
export fn ds4_server_decode_is_tool(state: c.dsml_decode_state) bool {
    return owner.isTool(state);
}
export fn ds4_server_decode_is_payload(state: c.dsml_decode_state) bool {
    return owner.isPayload(state);
}
export fn ds4_server_decode_observe(tracker: *const c.dsml_decode_tracker, scan: [*c]const u8, saw_start: *bool, saw_end: *bool, orphan: ?*bool) void {
    owner.observe(tracker, scan, saw_start, saw_end, orphan);
}
export fn ds4_server_decode_boundary_safe(sampled: *const c.buf, tracker: *const c.dsml_decode_tracker) bool {
    return owner.boundarySafe(sampled, tracker);
}
