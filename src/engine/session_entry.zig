//! Hot native model dispatch. Keep it out of the CPU-safe frontier/control
//! object used by include-ds4.c fixtures.
const c = @import("gguf_c");
const session = @import("session.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (capacity == 0) &.{} else pointer[0..capacity];
}
export fn ds4_native_session_sync(view: *c.ds4_session_view, prompt: ?*const c.ds4_tokens, message: [*c]u8, capacity: usize) c_int {
    return session.sync(view, prompt, errors(message, capacity));
}
export fn ds4_native_session_multimodal_sync(view: *c.ds4_session_view, prompt: ?*const c.ds4_tokens, images: [*c]const c.ds4_vision_span, count: usize, message: [*c]u8, capacity: usize) c_int {
    return session.syncMultimodal(view, prompt, images, count, errors(message, capacity));
}
export fn ds4_native_session_eval(view: *c.ds4_session_view, token: c_int, message: [*c]u8, capacity: usize) c_int {
    return session.eval(view, token, errors(message, capacity));
}
export fn ds4_native_session_invalidate(view: *c.ds4_session_view) void {
    session.invalidate(view);
}
export fn ds4_native_session_payload_bytes(view: *c.ds4_session_view) u64 {
    return session.payloadBytes(view);
}
export fn ds4_native_session_payload_save(view: *c.ds4_session_view, stream: ?*c.FILE, message: [*c]u8, capacity: usize) c_int {
    return session.savePayload(view, stream, errors(message, capacity));
}
export fn ds4_native_session_payload_load(view: *c.ds4_session_view, stream: ?*c.FILE, bytes: u64, message: [*c]u8, capacity: usize) c_int {
    return session.loadPayload(view, stream, bytes, errors(message, capacity));
}
