//! Complete edit actions: exact anchors, atomic file publication and the nearby
//! post-edit shape shown to the model. Worker/config policy stays at the caller.
const std = @import("std");
const c = @import("agent_tool_actions_c");
const Buffer = @import("tool_actions.zig").Buffer;
const allocator = std.heap.c_allocator;
pub const Call = c.agent_tool_call;
pub const Parser = c.agent_dsml_parser;
pub const Forcer = c.agent_edit_upto_forcer;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn bytes(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
fn fatal(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal("ds4-agent: malloc")).ptr;
}
fn argument(call: *const Call, name: [*:0]const u8) [*c]const u8 {
    return c.ds4_agent_tool_arg_value(call, name);
}
fn toolError(message: []const u8) [*c]u8 {
    var result: Buffer = .{};
    result.append("Tool error: ");
    result.append(message);
    result.append("\n");
    return result.take();
}

const Line = struct {
    start: usize,
    content_end: usize,
    end: usize,
};
// content_end excludes CR/LF so line comparisons and output keep the original
// newline spelling separate from the content itself.
fn splitLines(data: []const u8) std.ArrayList(Line) {
    var lines: std.ArrayList(Line) = .empty;
    var position: usize = 0;
    while (position < data.len) {
        const start = position;
        while (position < data.len and data[position] != '\n' and data[position] != '\r') position += 1;
        const content_end = position;
        if (position < data.len) {
            position += if (data[position] == '\r' and position + 1 < data.len and data[position + 1] == '\n') @as(usize, 2) else 1;
        }
        lines.append(allocator, .{ .start = start, .content_end = content_end, .end = position }) catch fatal("ds4-agent: realloc");
    }
    return lines;
}
fn lineForOffset(lines: []const Line, offset: usize) c_int {
    for (lines, 0..) |line, index| if (offset < line.end) return @intCast(index + 1);
    return if (lines.len == 0) 1 else @intCast(lines.len);
}
const LineEffect = struct { start: c_int = 0, end: c_int = 0, delta: c_int = 0 };
fn lineEffect(old: []const u8, new: []const u8, offset: usize, removed: usize) LineEffect {
    var old_lines = splitLines(old);
    defer old_lines.deinit(allocator);
    var new_lines = splitLines(new);
    defer new_lines.deinit(allocator);
    if (old_lines.items.len == 0) return .{};
    var last = if (removed != 0) offset + removed - 1 else offset;
    if (last >= old.len) last = old.len - 1;
    return .{
        .start = lineForOffset(old_lines.items, offset),
        .end = lineForOffset(old_lines.items, last),
        .delta = @as(c_int, @intCast(new_lines.items.len)) - @as(c_int, @intCast(old_lines.items.len)),
    };
}
fn appendLine(output: *Buffer, data: []const u8, line: Line, number: c_int) void {
    var prefix: [64]u8 = undefined;
    _ = c.snprintf(&prefix, prefix.len, "%d ", number);
    output.append(text(&prefix));
    output.append(data[line.start..line.content_end]);
    output.append("\n");
}
// Successful edits return the nearby post-edit file shape. This spends cheap
// prefill tokens to save expensive model retries: the model immediately sees
// shifted line numbers, braces, semicolons and accidental duplication.
fn appendContext(output: *Buffer, path: [*c]const u8, data: []const u8, start: c_int, end: c_int) void {
    var lines = splitLines(data);
    defer lines.deinit(allocator);
    const count: c_int = @intCast(lines.items.len);
    if (count == 0) return;
    const anchor_start: c_int = @min(count, @max(1, start));
    const anchor_end: c_int = @min(count, @max(anchor_start, end));
    const context_start: c_int = @max(1, anchor_start - 5);
    const context_end: c_int = @min(count, anchor_end + 8);
    var header: [c.PATH_MAX + 160]u8 = undefined;
    _ = c.snprintf(&header, header.len, "Current file around edit: %s lines %d-%d of %d\n", path, context_start, context_end, count);
    output.append(text(&header));
    if (anchor_end - anchor_start + 1 <= 36) {
        var number = context_start;
        while (number <= context_end) : (number += 1) appendLine(output, data, lines.items[@intCast(number - 1)], number);
    } else {
        const head_end: c_int = anchor_start + 17;
        const tail_start: c_int = anchor_end - 17;
        var number = context_start;
        while (number <= head_end) : (number += 1) appendLine(output, data, lines.items[@intCast(number - 1)], number);
        _ = c.snprintf(&header, header.len, "... %d edited lines omitted ...\n", tail_start - head_end - 1);
        output.append(text(&header));
        number = tail_start;
        while (number <= context_end) : (number += 1) appendLine(output, data, lines.items[@intCast(number - 1)], number);
    }
}
fn editResult(path: [*c]const u8, effect: LineEffect, data: []const u8, kind: [*:0]const u8) [*c]u8 {
    var output: Buffer = .{ .limit = c.AGENT_TOOL_MAX_BYTES };
    var message: [c.PATH_MAX + 180]u8 = undefined;
    _ = c.snprintf(&message, message.len, "Edited %s using %s\n", path, kind);
    output.append(text(&message));
    if (effect.start > 0 and effect.end >= effect.start) {
        _ = c.snprintf(&message, message.len, "Touched old lines %d-%d; current post-edit context follows.\n", effect.start, effect.end);
        output.append(text(&message));
        if (effect.delta != 0) {
            _ = c.snprintf(&message, message.len, "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n", effect.end, effect.delta, effect.end + 1, effect.end + 1 + effect.delta);
            output.append(text(&message));
        }
        appendContext(&output, path, data, effect.start, @max(effect.start, effect.end + effect.delta));
    }
    return output.take();
}

pub fn write(call: *const Call) [*c]u8 {
    const path = argument(call, "path");
    const content = argument(call, "content");
    if (path == null or path[0] == 0) return duplicate("Tool error: write requires path\n");
    if (content == null) return duplicate("Tool error: write requires content\n");
    const length = text(content).len;
    var errors: [256]u8 = undefined;
    if (c.ds4_agent_file_replace(path, content, length, null, 0, &errors, errors.len) != 0) return toolError(text(&errors));
    var message: [c.PATH_MAX + 160]u8 = undefined;
    _ = c.snprintf(&message, message.len, "Wrote %zu bytes to %s\n", length, path);
    return duplicate(text(&message));
}

pub fn preflight(allow_upto: bool, call: *const Call, errors: []u8) bool {
    const path = argument(call, "path");
    if (path == null or path[0] == 0) return true; // Cannot preflight until path is known.
    const old = argument(call, "old");
    if (old == null or old[0] == 0) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "edit requires non-empty old text");
        return false;
    }
    var data: [*c]u8 = null;
    var length: usize = 0;
    if (c.ds4_agent_file_read_bytes(path, &data, &length, errors.ptr, errors.len) != 0) return false;
    defer c.free(data);
    var match: [*c]const u8 = null;
    var match_length: usize = 0;
    var anchored = false;
    return c.ds4_agent_file_find_old_span(data, length, old, allow_upto, &match, &match_length, &anchored, errors.ptr, errors.len);
}
fn splice(path: [*c]const u8, data: []const u8, offset: usize, removed: usize, insertion: []const u8, kind: [*:0]const u8) [*c]u8 {
    const length = offset + insertion.len + (data.len - offset - removed);
    const memory = allocator.allocSentinel(u8, length, 0) catch fatal("ds4-agent: malloc");
    defer allocator.free(memory);
    @memcpy(memory[0..offset], data[0..offset]);
    @memcpy(memory[offset..][0..insertion.len], insertion);
    @memcpy(memory[offset + insertion.len .. length], data[offset + removed ..]);
    var errors: [256]u8 = undefined;
    if (c.ds4_agent_file_replace(path, memory.ptr, length, data.ptr, data.len, &errors, errors.len) != 0) return toolError(text(&errors));
    return editResult(path, lineEffect(data, memory, offset, removed), memory, kind);
}
// Old/new editing is intentionally conservative: exact old text must be unique.
// For large replacements, one [upto] marker permits unique head + unique later
// tail anchors; the whole intervening span is then replaced atomically.
pub fn apply(allow_upto: bool, call: *const Call) [*c]u8 {
    const path = argument(call, "path");
    if (path == null or path[0] == 0) return duplicate("Tool error: edit requires path\n");
    const old = argument(call, "old");
    const new = argument(call, "new");
    if (old == null or old[0] == 0) return duplicate("Tool error: edit requires non-empty old text\n");
    if (new == null) return duplicate("Tool error: edit requires new text\n");
    var errors: [256]u8 = undefined;
    var data: [*c]u8 = null;
    var length: usize = 0;
    if (c.ds4_agent_file_read_bytes(path, &data, &length, &errors, errors.len) != 0) return toolError(text(&errors));
    defer c.free(data);
    var match: [*c]const u8 = null;
    var match_length: usize = 0;
    var anchored = false;
    if (!c.ds4_agent_file_find_old_span(data, length, old, allow_upto, &match, &match_length, &anchored, &errors, errors.len)) return toolError(text(&errors));
    return splice(path, bytes(data, length), @intFromPtr(match) - @intFromPtr(data), match_length, text(new), if (anchored) "anchored old/new replacement" else "old/new replacement");
}

fn mayClose(next: []const u8) bool {
    const trimmed = std.mem.trimStart(u8, next, " \t\r\n");
    return trimmed.len != 0 and trimmed[0] == '<';
}
fn onlySpace(next: []const u8) bool {
    for (next) |byte| if (std.mem.findScalar(u8, " \t\r\n", byte) == null) return false;
    return true;
}
fn maturePrefix(old: []const u8) bool {
    if (old.len < c.AGENT_EDIT_UPTO_MIN_PREFIX_BYTES) return false;
    var nonempty_lines: c_int = 0;
    var line_has_text = false;
    for (old) |byte| {
        if (byte == '\n') {
            if (line_has_text) nonempty_lines += 1;
            line_has_text = false;
        } else if (c.isspace(byte) == 0) line_has_text = true;
    }
    return nonempty_lines >= c.AGENT_EDIT_UPTO_MIN_PREFIX_LINES;
}
fn readyForUpto(pointer: [*c]const u8, old: []const u8) bool {
    if (old.len == 0 or std.mem.find(u8, text(pointer), "[upto]") != null or !maturePrefix(old)) return false;
    const end = std.mem.trimEnd(u8, old, " \t\r");
    return end.len != 0 and end[end.len - 1] == '\n';
}
// Inspect the next sampled token BEFORE eval. Once a large exact old prefix is
// a unique file anchor, the caller evaluates a complete "[upto]" line instead
// of allowing the model to retype the rest of the old block.
pub fn forceUpto(maybe_forcer: ?*Forcer, maybe_parser: ?*Parser, next: []const u8) bool {
    const forcer = maybe_forcer orelse return false;
    const parser = maybe_parser orelse return false;
    const in_edit_old = parser.state == c.AGENT_DSML_PARAM_VALUE and
        std.mem.eql(u8, text(parser.current.name), "edit") and
        std.mem.eql(u8, text(parser.param_name), "old");
    if (!in_edit_old) {
        forcer.* = .{ .active = false, .done = false };
        return false;
    }
    if (!forcer.active) forcer.* = .{ .active = true, .done = false };
    if (forcer.done or mayClose(next) or onlySpace(next)) return false;
    const path = argument(&parser.current, "path");
    if (path == null or path[0] == 0 or parser.param_value_start > parser.raw_len) return false;
    const length = parser.raw_len - parser.param_value_start;
    if (length == 0) return false;
    const old = parser.raw + parser.param_value_start;
    if (!readyForUpto(old, bytes(old, length))) return false;
    var errors: [256]u8 = undefined;
    var data: [*c]u8 = null;
    var data_length: usize = 0;
    if (c.ds4_agent_file_read_bytes(path, &data, &data_length, &errors, errors.len) != 0) return false;
    defer c.free(data);
    var match: [*c]const u8 = null;
    if (!c.ds4_agent_file_find_unique(data, data_length, old, length, &match, "old prefix", &errors, errors.len)) return false;
    forcer.done = true;
    return true;
}
