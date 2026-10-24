//! Hot native model dispatch. Keep it out of the CPU-safe frontier/control
//! object used by include-ds4.c fixtures.
const c = @import("gguf_c");
const session = @import("session.zig");
const steering = @import("session_steering.zig");
const continuation = @import("session_continuation.zig");

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

export fn ds4_native_session_speculate(view: *c.ds4_session_view, first: c_int, maximum: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, ignore_eos: bool, accepted: [*c]c_int, count: c_int, text: [*c]u8, capacity: usize) c_int {
    return continuation.speculate(view, first, maximum, eos, temperature, top_k, top_p, min_p, rng, ignore_eos, accepted, count, errors(text, capacity));
}
export fn ds4_native_session_rewind(view: *c.ds4_session_view, position: c_int) void {
    continuation.rewind(view, position);
}

export fn ds4_native_session_steering_scale(view: *const c.ds4_session_view) f32 {
    return steering.scale(view);
}
export fn ds4_native_session_steering_set(view: *c.ds4_session_view, value: f32) c_int {
    return steering.set(view, value);
}
