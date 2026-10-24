//! CPU-safe common controls/frontiers. Keep this object separate from model
//! dispatch so include-ds4.c fixtures do not extract a second GPU/C engine.
const c = @import("gguf_c");
const control = @import("session_control.zig");
const rewrite = @import("session_rewrite.zig");
const Frontier = @import("session_frontier.zig").Frontier;

fn images(pointer: [*c]const c.ds4_vision_span, count: usize) []const c.ds4_vision_span {
    return if (count == 0) &.{} else pointer[0..count];
}

export fn ds4_native_frontier_vision_prefix(state: *const c.ds4_session_frontier, source: [*c]const c.ds4_vision_span, count: usize) bool {
    if (count != 0 and source == null) return false;
    return (Frontier{ .state = @constCast(state) }).visionPrefixMatches(images(source, count));
}
export fn ds4_native_frontier_vision_state(state: *const c.ds4_session_frontier, source: [*c]const c.ds4_vision_span, count: usize) bool {
    if (count != 0 and source == null) return false;
    return (Frontier{ .state = @constCast(state) }).visionStateMatches(images(source, count));
}
export fn ds4_native_frontier_vision_fingerprints(state: *const c.ds4_session_frontier, source: [*c]const c.ds4_vision_span, count: usize) bool {
    if (count != 0 and source == null) return false;
    return (Frontier{ .state = @constCast(state) }).visionFingerprintPrefixMatches(images(source, count));
}
export fn ds4_native_frontier_rebase_vision(state: *const c.ds4_session_frontier, source: [*c]c.ds4_vision_span, count: usize) bool {
    if (count != 0 and source == null) return false;
    return (Frontier{ .state = @constCast(state) }).rebaseVision(if (count == 0) &.{} else source[0..count]);
}
export fn ds4_native_frontier_vision_overlap(state: *const c.ds4_session_frontier, start: u32, count: u32) bool {
    return (Frontier{ .state = @constCast(state) }).visionRangeOverlaps(start, count);
}
export fn ds4_native_frontier_remember_images(state: *c.ds4_session_frontier) bool {
    return (Frontier{ .state = state }).rememberImages();
}

export fn ds4_native_control_progress(record: ?*c.ds4_session_control, callback: c.ds4_session_progress_fn, user: ?*anyopaque) void {
    const owner = record orelse return;
    owner.progress = callback;
    owner.progress_ud = user;
}
export fn ds4_native_control_display(record: ?*c.ds4_session_control, callback: c.ds4_session_progress_fn, user: ?*anyopaque) void {
    const owner = record orelse return;
    owner.display_progress = callback;
    owner.display_progress_ud = user;
}
export fn ds4_native_control_cancel(record: ?*c.ds4_session_control, callback: c.ds4_session_cancel_fn, user: ?*anyopaque) void {
    const owner = record orelse return;
    owner.cancel = callback;
    owner.cancel_ud = user;
}
export fn ds4_native_control_cancelled(record: ?*const c.ds4_session_control) bool {
    return if (record) |owner| control.cancelled(owner) else false;
}
export fn ds4_native_control_report(record: ?*const c.ds4_session_control, event: [*c]const u8, current: c_int, total: c_int) void {
    if (record) |owner| control.report(owner, event, current, total);
}

export fn ds4_session_rewrite_requires_rebuild(live: c_int, canonical: c_int, common: c_int) bool {
    return rewrite.requiresRebuild(live, canonical, common);
}
export fn ds4_native_session_rewrite(session: ?*c.ds4_session, frontier: ?*const c.ds4_session_frontier, prompt: ?*const c.ds4_tokens, common: c_int, text: [*c]u8, capacity: usize) c.ds4_session_rewrite_result {
    return rewrite.rewrite(session, frontier, prompt, common, if (capacity == 0) &.{} else text[0..capacity]);
}
