//! Projection of sampled V4.1 tokens. Only marked special-token occurrences
//! carry control meaning; identical bytes produced by ordinary tokens are data.
const std = @import("std");

pub const Range = extern struct { begin: usize, end: usize };
pub const Call = struct {
    name: []const u8,
    arguments: []u8,
    complete: bool,
};
pub const Snapshot = struct {
    reasoning: []const u8 = "",
    content: []const u8 = "",
    reasoning_closed: bool = false,
    tool_start: ?usize = null,
    tool_end: ?usize = null,
    in_payload: bool = false,
    invalid: bool = false,
    calls: std.ArrayList(Call) = .empty,

    pub fn deinit(self: *Snapshot, allocator: std.mem.Allocator) void {
        for (self.calls.items) |call| allocator.free(call.arguments);
        self.calls.deinit(allocator);
    }
};

const marker = "｜DSML｜";
pub const Tag = struct { begin: usize, end: usize, body: []const u8, closing: bool };
pub const View = struct {
    raw: []const u8,
    ranges: []const Range,

    pub fn visibleEnd(self: View, final: bool) usize {
        var end = safeEnd(self.raw, final);
        for (self.ranges) |range| {
            if (range.begin >= self.raw.len or range.end <= self.raw.len) continue;
            var begin = range.begin;
            if (std.mem.startsWith(u8, marker, self.raw[begin..])) {
                if (begin >= 2 and std.mem.eql(u8, self.raw[begin - 2 .. begin], "</")) begin -= 2 else if (begin >= 1 and self.raw[begin - 1] == '<') begin -= 1;
            }
            end = @min(end, begin);
        }
        return end;
    }

    pub fn special(self: View, spelling: []const u8, start: usize) ?Range {
        for (self.ranges) |range| {
            if (range.begin >= start and range.end <= self.raw.len and
                std.mem.eql(u8, self.raw[range.begin..range.end], spelling)) return range;
        }
        return null;
    }
    pub fn nextTag(self: View, start: usize) ?Tag {
        for (self.ranges) |range| {
            if (range.begin < start or range.end > self.raw.len or
                !std.mem.eql(u8, self.raw[range.begin..range.end], marker)) continue;
            const closing = range.begin >= 2 and std.mem.eql(u8, self.raw[range.begin - 2 .. range.begin], "</");
            const begin = if (closing) range.begin - 2 else if (range.begin >= 1 and self.raw[range.begin - 1] == '<') range.begin - 1 else continue;
            const end = std.mem.findScalarPos(u8, self.raw, range.end, '>') orelse return .{ .begin = begin, .end = self.raw.len, .body = "", .closing = closing };
            return .{ .begin = begin, .end = end + 1, .body = self.raw[range.end..end], .closing = closing };
        }
        return null;
    }
    pub fn tag(self: View, name: []const u8, closing: bool, start: usize) ?Tag {
        var position = start;
        while (self.nextTag(position)) |candidate| {
            if (candidate.closing == closing and std.mem.eql(u8, candidate.body, name)) return candidate;
            position = candidate.end;
            if (position >= self.raw.len) break;
        }
        return null;
    }
};

fn attr(tag: []const u8, key: []const u8) ?[]const u8 {
    var position: usize = 0;
    while (std.mem.findPos(u8, tag, position, key)) |found| {
        const start = found + key.len;
        if ((found == 0 or tag[found - 1] == ' ') and std.mem.startsWith(u8, tag[start..], "=\"")) {
            const end = std.mem.findScalarPos(u8, tag, start + 2, '"') orelse return null;
            return tag[start + 2 .. end];
        }
        position = start;
    }
    return null;
}
fn whitespace(raw: []const u8, start: usize) usize {
    var at = start;
    while (at < raw.len and std.ascii.isWhitespace(raw[at])) at += 1;
    return at;
}
pub fn safeEnd(raw: []const u8, final: bool) usize {
    if (final or raw.len == 0) return raw.len;
    var end = raw.len;
    // '<' and '</' can precede the next DSML special token. Hold only those
    // possible prefixes; an ordinary marker spelling is released as data.
    if (raw[end - 1] == '<') end -= 1 else if (end >= 2 and std.mem.endsWith(u8, raw, "</")) end -= 2;
    var begin = end;
    while (begin > 0 and (raw[begin - 1] & 0xc0) == 0x80) begin -= 1;
    if (begin > 0) {
        const lead = raw[begin - 1];
        const needed: usize = if (lead >= 0xf0) 4 else if (lead >= 0xe0) 3 else if (lead >= 0xc0) 2 else 1;
        if (end - (begin - 1) < needed) end = begin - 1;
    }
    return end;
}
pub fn textEnd(raw: []const u8, limit: usize) usize {
    var end = limit;
    while (end > 0 and std.ascii.isWhitespace(raw[end - 1])) end -= 1;
    return end;
}
pub fn quoted(out: *std.ArrayList(u8), allocator: std.mem.Allocator, raw: []const u8, close: bool) !void {
    try out.append(allocator, '"');
    const hex = "0123456789abcdef";
    for (raw) |byte| switch (byte) {
        '"', '\\' => try out.appendSlice(allocator, &.{ '\\', byte }),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        0...8, 11, 12, 14...31 => try out.appendSlice(allocator, &.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15] }),
        else => try out.append(allocator, byte),
    };
    if (close) try out.append(allocator, '"');
}
pub fn jsonBytes(out: *std.ArrayList(u8), allocator: std.mem.Allocator, raw: []const u8) !void {
    var string = false;
    var escaped = false;
    for (raw) |byte| {
        if (!string and std.ascii.isWhitespace(byte)) continue;
        try out.append(allocator, byte);
        if (escaped) escaped = false else if (string and byte == '\\') escaped = true else if (byte == '"') string = !string;
    }
}

pub fn project(allocator: std.mem.Allocator, view: View, thinking: bool, final: bool) !Snapshot {
    return projectImpl(allocator, view, thinking, final, true);
}
/// Generation needs only control state. Avoid rebuilding every accumulated
/// argument byte merely to select the next token's sampling policy.
pub fn projectState(allocator: std.mem.Allocator, view: View, thinking: bool, final: bool) !Snapshot {
    return projectImpl(allocator, view, thinking, final, false);
}
fn projectImpl(allocator: std.mem.Allocator, view: View, thinking: bool, final: bool, comptime build_arguments: bool) !Snapshot {
    var out: Snapshot = .{};
    errdefer out.deinit(allocator);
    const begin = if (view.special("<think>", 0)) |range| (if (range.begin == 0) range.end else 0) else 0;
    const open = view.tag(" calls", false, begin);
    const close = view.special("</think>", begin);
    const text_end = if (open) |tag| textEnd(view.raw, tag.begin) else if (final) view.visibleEnd(final) else textEnd(view.raw, view.visibleEnd(final));
    if (close != null and close.?.begin < text_end) {
        out.reasoning = view.raw[begin..close.?.begin];
        out.content = view.raw[close.?.end..text_end];
        out.reasoning_closed = true;
    } else if (thinking) out.reasoning = view.raw[begin..text_end] else out.content = view.raw[begin..text_end];
    const opening = open orelse {
        // An incomplete actual DSML tag is control syntax, not visible text.
        if (view.nextTag(begin)) |pending| if (pending.body.len == 0) {
            const end = textEnd(view.raw, pending.begin);
            if (thinking and !out.reasoning_closed) out.reasoning = view.raw[begin..@max(begin, end)] else {
                const content_begin = if (close) |range| range.end else begin;
                if (end >= content_begin) out.content = view.raw[content_begin..end];
            }
        };
        return out;
    };
    out.tool_start = opening.begin;
    var cursor = opening.end;
    while (true) {
        cursor = whitespace(view.raw, cursor);
        const invoke = view.nextTag(cursor) orelse return out;
        if (invoke.begin != cursor or invoke.body.len == 0) return out;
        if (invoke.closing and std.mem.eql(u8, invoke.body, " calls")) {
            out.tool_end = invoke.end;
            return out;
        }
        if (invoke.closing or !std.mem.startsWith(u8, invoke.body, " invoke ")) {
            out.invalid = true;
            return out;
        }
        const name = attr(invoke.body, "name") orelse {
            out.invalid = true;
            return out;
        };
        cursor = invoke.end;
        var arguments: std.ArrayList(u8) = .empty;
        defer arguments.deinit(allocator);
        if (build_arguments) try arguments.append(allocator, '{');
        var complete = false;
        var members: usize = 0;
        while (true) {
            cursor = whitespace(view.raw, cursor);
            const parameter = view.nextTag(cursor) orelse break;
            if (parameter.begin != cursor or parameter.body.len == 0) break;
            if (parameter.closing and std.mem.eql(u8, parameter.body, " invoke")) {
                cursor = parameter.end;
                if (build_arguments) try arguments.append(allocator, '}');
                complete = true;
                break;
            }
            if (parameter.closing or !std.mem.startsWith(u8, parameter.body, " parameter ")) {
                out.invalid = true;
                break;
            }
            const key = attr(parameter.body, "name") orelse {
                out.invalid = true;
                break;
            };
            const kind = attr(parameter.body, "string") orelse "true";
            const ending = view.tag(" parameter", true, parameter.end);
            if (!std.mem.eql(u8, kind, "true") and !std.mem.eql(u8, kind, "false")) out.invalid = true;
            if (build_arguments) {
                if (members != 0) try arguments.appendSlice(allocator, ",");
                try quoted(&arguments, allocator, key, true);
                try arguments.appendSlice(allocator, ":");
                const pending = view.nextTag(parameter.end);
                const value_end = if (ending) |tag| tag.begin else if (pending != null and pending.?.body.len == 0) pending.?.begin else view.visibleEnd(final);
                if (std.mem.eql(u8, kind, "true")) try quoted(&arguments, allocator, view.raw[parameter.end..value_end], ending != null) else if (std.mem.eql(u8, kind, "false")) try jsonBytes(&arguments, allocator, view.raw[parameter.end..value_end]);
            }
            if (ending == null) {
                out.in_payload = true;
                break;
            }
            cursor = ending.?.end;
            members += 1;
        }
        if (build_arguments) try out.calls.append(allocator, .{ .name = name, .arguments = try arguments.toOwnedSlice(allocator), .complete = complete });
        if (!complete or out.invalid) return out;
    }
}

test "ordinary spellings are data and actual DSML controls delimit raw strings" {
    const allocator = std.testing.allocator;
    const raw = "literal </think> <｜DSML｜ calls>";
    var literal = try project(allocator, .{ .raw = raw, .ranges = &.{} }, false, true);
    defer literal.deinit(allocator);
    try std.testing.expectEqualStrings(raw, literal.content);
    try std.testing.expect(literal.tool_start == null);

    const generated = "</think><｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"text\" string=\"true\">raw </think> </｜DSML｜ parameter> &lt; &amp;</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    var ranges: std.ArrayList(Range) = .empty;
    defer ranges.deinit(allocator);
    try ranges.append(allocator, .{ .begin = 0, .end = 8 });
    var position: usize = 8;
    var count: usize = 0;
    while (std.mem.findPos(u8, generated, position, marker)) |found| {
        // The first apparent parameter close occurs in literal payload.
        if (count != 3) try ranges.append(allocator, .{ .begin = found, .end = found + marker.len });
        count += 1;
        position = found + marker.len;
    }
    var actual = try project(allocator, .{ .raw = generated, .ranges = ranges.items }, true, true);
    defer actual.deinit(allocator);
    try std.testing.expect(actual.tool_end != null);
    try std.testing.expectEqual(@as(usize, 1), actual.calls.items.len);
    try std.testing.expectEqualStrings("{\"text\":\"raw </think> </｜DSML｜ parameter> &lt; &amp;\"}", actual.calls.items[0].arguments);
    var previous: std.ArrayList(u8) = .empty;
    defer previous.deinit(allocator);
    var prefix: usize = 0;
    while (prefix < generated.len) {
        prefix += 1;
        for (ranges.items) |range| if (prefix > range.begin and prefix < range.end) {
            prefix = range.end;
        };
        var partial = try project(allocator, .{ .raw = generated[0..prefix], .ranges = ranges.items }, true, false);
        defer partial.deinit(allocator);
        var phase = try projectState(allocator, .{ .raw = generated[0..prefix], .ranges = ranges.items }, true, false);
        defer phase.deinit(allocator);
        try std.testing.expectEqual(partial.tool_start, phase.tool_start);
        try std.testing.expectEqual(partial.tool_end, phase.tool_end);
        try std.testing.expectEqual(partial.in_payload, phase.in_payload);
        try std.testing.expectEqual(partial.invalid, phase.invalid);
        try std.testing.expect(phase.calls.items.len == 0);
        if (partial.calls.items.len == 0) continue;
        const arguments = partial.calls.items[0].arguments;
        try std.testing.expect(std.mem.startsWith(u8, arguments, previous.items));
        previous.clearRetainingCapacity();
        try previous.appendSlice(allocator, arguments);
    }
    try std.testing.expectEqualStrings(actual.calls.items[0].arguments, previous.items);
}
