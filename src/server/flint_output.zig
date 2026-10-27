//! Flint's sampled XML projection, sharing token provenance and argument JSON
//! with the V4.1 projector. Ambiguous value types wait for the closing token.
const std = @import("std");
const Shared = @import("ds41_output.zig");
const Schema = @import("flint_schema.zig");
pub const Snapshot = Shared.Snapshot;
pub const View = Shared.View;

const Representation = enum { string, json, wait };
fn representation(allocator: std.mem.Allocator, orders: anytype, name: []const u8, key: []const u8, raw: []const u8, complete: bool) Representation {
    if (@TypeOf(orders) != @TypeOf(null)) {
        for (0..@as(usize, @intCast(orders.*.len))) |index| {
            const order = orders.*.v[index];
            if (order.name == null or !std.mem.eql(u8, std.mem.span(order.name), name)) continue;
            for (0..@as(usize, @intCast(order.len))) |member| {
                if (!std.mem.eql(u8, std.mem.span(order.prop[member]), key)) continue;
                const property = std.json.parseFromSlice(std.json.Value, allocator, std.mem.span(order.prop_schema[member]), .{}) catch break;
                defer property.deinit();
                switch (Schema.direct(property.value)) {
                    .string => return .string,
                    .json => return .json,
                    .ambiguous => {},
                }
                if (!complete) return .wait;
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                const a = arena.allocator();
                const root = if (order.parameters_json == null or !Schema.references(property.value)) property.value else std.json.parseFromSliceLeaky(std.json.Value, a, std.mem.span(order.parameters_json), .{}) catch property.value;
                return if (Schema.stringValue(a, root, property.value, raw) catch true) .string else .json;
            }
        }
    }
    if (!complete) return .wait;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return .string;
    parsed.deinit();
    return .json;
}
/// Legacy text-only reply fixtures share the same completed value selection.
pub fn isString(allocator: std.mem.Allocator, orders: anytype, name: []const u8, key: []const u8, value: []const u8) bool {
    return representation(allocator, orders, name, key, value, true) == .string;
}
fn space(raw: []const u8, start: usize) usize {
    var index = start;
    while (index < raw.len and std.ascii.isWhitespace(raw[index])) index += 1;
    return index;
}
pub fn project(allocator: std.mem.Allocator, view: View, orders: anytype, final: bool) !Snapshot {
    return projectImpl(allocator, view, orders, final, true);
}
pub fn projectState(allocator: std.mem.Allocator, view: View, orders: anytype, final: bool) !Snapshot {
    return projectImpl(allocator, view, orders, final, false);
}
fn projectImpl(allocator: std.mem.Allocator, view: View, orders: anytype, final: bool, comptime build_arguments: bool) !Snapshot {
    var out: Snapshot = .{};
    errdefer out.deinit(allocator);
    const thinking = view.special("<think>", 0);
    const begin = if (thinking != null and thinking.?.begin == 0) thinking.?.end else 0;
    const close = view.special("</think>", begin);
    const open = view.special("<tool_call>", begin);
    const text_end = if (open) |range| Shared.textEnd(view.raw, range.begin) else if (final) view.visibleEnd(final) else Shared.textEnd(view.raw, view.visibleEnd(final));
    if (close != null and close.?.begin < text_end) {
        out.reasoning = view.raw[begin..close.?.begin];
        out.content = view.raw[close.?.end..text_end];
        out.reasoning_closed = true;
    } else if (thinking != null) out.reasoning = view.raw[begin..text_end] else out.content = view.raw[begin..text_end];
    var opening = open orelse return out;
    out.tool_start = opening.begin;
    while (true) {
        const key_start = view.special("<arg_key>", opening.end);
        const ending = view.special("</tool_call>", opening.end);
        const name_end = if (key_start != null and (ending == null or key_start.?.begin < ending.?.begin)) key_start.?.begin else if (ending) |range| range.begin else return out;
        const name = std.mem.trim(u8, view.raw[opening.end..name_end], " \r\n\t");
        if (name.len == 0) {
            out.invalid = true;
            return out;
        }
        var arguments: std.ArrayList(u8) = .empty;
        defer arguments.deinit(allocator);
        if (build_arguments) try arguments.append(allocator, '{');
        var cursor = name_end;
        var members: usize = 0;
        var complete = false;
        while (true) {
            cursor = space(view.raw, cursor);
            if (ending != null and cursor == ending.?.begin) {
                cursor = ending.?.end;
                if (build_arguments) try arguments.append(allocator, '}');
                complete = true;
                break;
            }
            const key_open = view.special("<arg_key>", cursor) orelse break;
            if (key_open.begin != cursor) {
                out.invalid = true;
                break;
            }
            const key_close = view.special("</arg_key>", key_open.end) orelse break;
            const key = view.raw[key_open.end..key_close.begin];
            const value_open = view.special("<arg_value>", key_close.end) orelse break;
            if (space(view.raw, key_close.end) != value_open.begin) {
                out.invalid = true;
                break;
            }
            const value_close = view.special("</arg_value>", value_open.end);
            if (build_arguments) {
                const value_end = if (value_close) |range| range.begin else view.visibleEnd(final);
                const value = view.raw[value_open.end..value_end];
                const kind = representation(allocator, orders, name, key, value, value_close != null);
                // Unknown values can turn out to be either raw strings or JSON;
                // emitting a guessed opening quote would make that irreversible.
                if (kind == .wait) {
                    out.in_payload = true;
                    break;
                }
                if (members != 0) try arguments.appendSlice(allocator, ",");
                try Shared.quoted(&arguments, allocator, key, true);
                try arguments.appendSlice(allocator, ":");
                if (kind == .string) try Shared.quoted(&arguments, allocator, value, value_close != null) else try Shared.jsonBytes(&arguments, allocator, value);
            }
            if (value_close == null) {
                out.in_payload = true;
                break;
            }
            cursor = value_close.?.end;
            members += 1;
        }
        if (build_arguments) try out.calls.append(allocator, .{ .name = name, .arguments = try arguments.toOwnedSlice(allocator), .complete = complete });
        if (!complete or out.invalid) return out;
        out.tool_end = cursor;
        cursor = space(view.raw, cursor);
        const next = view.special("<tool_call>", cursor) orelse return out;
        if (next.begin != cursor) return out;
        opening = next;
        out.tool_end = null;
    }
}

test "Flint markers are token controls and unknown raw values become stable JSON" {
    const a = std.testing.allocator;
    const raw = "<tool_call>echo<arg_key>text</arg_key><arg_value>literal </arg_value> data</arg_value></tool_call>";
    const markers = [_][]const u8{ "<tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</tool_call>" };
    var ranges: std.ArrayList(Shared.Range) = .empty;
    defer ranges.deinit(a);
    var cursor: usize = 0;
    for (markers, 0..) |marker, index| {
        const start = if (index == 4) std.mem.findLast(u8, raw, marker).? else std.mem.findPos(u8, raw, cursor, marker).?;
        try ranges.append(a, .{ .begin = start, .end = start + marker.len });
        cursor = start + marker.len;
    }
    var literal = try project(a, .{ .raw = raw, .ranges = &.{} }, null, true);
    defer literal.deinit(a);
    try std.testing.expectEqualStrings(raw, literal.content);
    var actual = try project(a, .{ .raw = raw, .ranges = ranges.items }, null, true);
    defer actual.deinit(a);
    try std.testing.expect(actual.tool_end != null and actual.calls.items[0].complete);
    try std.testing.expectEqualStrings("{\"text\":\"literal </arg_value> data\"}", actual.calls.items[0].arguments);
    var prefix: usize = 0;
    while (prefix < raw.len) {
        prefix += 1;
        var partial = try project(a, .{ .raw = raw[0..prefix], .ranges = ranges.items }, null, false);
        defer partial.deinit(a);
        var phase = try projectState(a, .{ .raw = raw[0..prefix], .ranges = ranges.items }, null, false);
        defer phase.deinit(a);
        try std.testing.expectEqual(partial.tool_start, phase.tool_start);
        try std.testing.expectEqual(partial.tool_end, phase.tool_end);
        try std.testing.expectEqual(partial.in_payload, phase.in_payload);
        try std.testing.expectEqual(partial.invalid, phase.invalid);
        try std.testing.expect(phase.calls.items.len == 0);
    }
}
