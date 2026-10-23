//! Sole transitional C facade; presentation.zig contains the reusable owner.
const std = @import("std");
const c = @import("agent_presentation_c");
const owner = @import("presentation.zig");
fn renderer(record: *c.agent_token_renderer) owner.Renderer {
    return .{ .record = record };
}
fn bytes(pointer: [*c]const u8, count: usize) []const u8 {
    return if (count == 0) &.{} else pointer[0..count];
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
export fn ds4_agent_presentation_capture_append(capture: ?*c.agent_tail_capture, pointer: [*c]const u8, count: usize) void {
    owner.captureAppend(capture orelse return, bytes(pointer, count));
}
export fn ds4_agent_presentation_capture_take(capture: *c.agent_tail_capture, length: [*c]usize) [*:0]u8 {
    return owner.captureTake(capture, if (length == null) null else @ptrCast(length));
}
export fn ds4_agent_presentation_syntax_path(path: [*c]const u8) *const c.agent_syntax {
    return @ptrCast(owner.syntaxForPath(text(path)));
}
export fn ds4_agent_presentation_columns() c_int {
    return owner.columns();
}
export fn ds4_agent_presentation_write(record: *c.agent_token_renderer, pointer: [*c]const u8, count: usize) void {
    renderer(record).write(bytes(pointer, count));
}
export fn ds4_agent_presentation_char(record: *c.agent_token_renderer, byte: u8) void {
    renderer(record).character(byte);
}
export fn ds4_agent_presentation_restore(record: *c.agent_token_renderer) void {
    renderer(record).restore();
}
export fn ds4_agent_presentation_reset(record: *c.agent_token_renderer) void {
    renderer(record).reset();
}
export fn ds4_agent_presentation_hint_end(record: *c.agent_token_renderer) void {
    renderer(record).hintEnd();
}
export fn ds4_agent_presentation_process(record: *c.agent_token_renderer, pointer: [*c]const u8, count: usize, final: bool) void {
    renderer(record).process(bytes(pointer, count), final);
}
export fn ds4_agent_presentation_finish(record: *c.agent_token_renderer) void {
    renderer(record).finish();
}
export fn ds4_agent_presentation_color(record: *c.agent_token_renderer, sequence: [*c]const u8) void {
    renderer(record).color(text(sequence));
}
export fn ds4_agent_presentation_plain(record: *c.agent_token_renderer, pointer: [*c]const u8, count: usize) void {
    renderer(record).plain(bytes(pointer, count));
}
export fn ds4_agent_presentation_code_begin(record: *c.agent_token_renderer, syntax: ?*const c.agent_syntax) void {
    renderer(record).codeBegin(if (syntax) |value| @ptrCast(@alignCast(value)) else null);
}
export fn ds4_agent_presentation_code_prefix(record: *c.agent_token_renderer, prefix: [*c]const u8, color: [*c]const u8) void {
    renderer(record).setCodePrefix(prefix, color);
}
export fn ds4_agent_presentation_code_upto(record: *c.agent_token_renderer, enabled: bool) void {
    renderer(record).setCodeUpto(enabled);
}
export fn ds4_agent_presentation_code_end(record: *c.agent_token_renderer) void {
    renderer(record).codeEnd();
}
export fn ds4_agent_presentation_code_byte(record: *c.agent_token_renderer, byte: u8) void {
    renderer(record).codeByte(byte);
}

// Existing C fixture exercises these exact owner actions directly.
export fn ds4_agent_presentation_flush_utf8(record: *c.agent_token_renderer) void {
    renderer(record).flushUtf8();
}
export fn ds4_agent_presentation_markdown_byte(record: *c.agent_token_renderer, byte: u8) void {
    renderer(record).markdownByte(byte);
}
export fn ds4_agent_presentation_markdown_finish(record: *c.agent_token_renderer) void {
    renderer(record).finishMarkdown();
}
