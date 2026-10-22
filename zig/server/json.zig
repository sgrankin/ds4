//! Primitive server JSON cursor, preserving the existing selective grammar.
//! Endpoint/value owners decide which fields to consume. This layer preserves
//! byte strings, cursor positions, allocation ownership and strtod acceptance.
const std = @import("std");
const c = @import("server_json_c");
const allocator = std.heap.c_allocator;
const Cursor = *[*c]const u8;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}

pub export fn ds4_server_json_ws(cursor: Cursor) void {
    while (cursor.*[0] != 0 and c.isspace(cursor.*[0]) != 0) cursor.* += 1;
}
pub export fn ds4_server_json_lit(cursor: Cursor, literal: [*c]const u8) bool {
    // Literal matching does not skip whitespace or require a token boundary.
    // Stop at the first mismatch, including an input NUL, without scanning the
    // entire remaining request for every small literal.
    const expected = std.mem.span(literal);
    for (expected, 0..) |value, position| {
        if (cursor.*[position] != value) return false;
    }
    cursor.* += expected.len;
    return true;
}

fn hex(value: u8) ?u4 {
    return switch (value) {
        '0'...'9' => @intCast(value - '0'),
        'a'...'f' => @intCast(value - 'a' + 10),
        'A'...'F' => @intCast(value - 'A' + 10),
        else => null,
    };
}
fn unicode16(cursor: Cursor) ?u32 {
    if (cursor.*[0] != '\\' or cursor.*[1] != 'u') return null;
    var codepoint: u32 = 0;
    for (0..4) |position| {
        const digit = hex(cursor.*[position + 2]) orelse return null;
        codepoint = (codepoint << 4) | @as(u32, digit);
    }
    cursor.* += 6;
    return codepoint;
}
fn appendCodepoint(out: *std.ArrayList(u8), value: u32) void {
    const codepoint = if (value > 0x10ffff or (value >= 0xd800 and value <= 0xdfff)) 0xfffd else value;
    var encoded: [4]u8 = undefined;
    const count = std.unicode.utf8Encode(@intCast(codepoint), &encoded) catch unreachable;
    out.appendSlice(allocator, encoded[0..count]) catch fatal();
}

pub export fn ds4_server_json_string(cursor: Cursor, output: *[*c]u8) bool {
    // Always define output, including failures after a caller freed an earlier
    // duplicate field. Leaving that pointer stale would cause a second free.
    output.* = null;
    ds4_server_json_ws(cursor);
    if (cursor.*[0] != '"') return false;
    cursor.* += 1;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    while (cursor.*[0] != 0 and cursor.*[0] != '"') {
        const value = cursor.*[0];
        cursor.* += 1;
        if (value != '\\') {
            // C accepts raw control/non-UTF8 bytes here; decoding escapes does
            // not add validation of the rest of the caller's byte string.
            out.append(allocator, value) catch fatal();
            continue;
        }
        const escape = cursor.*[0];
        cursor.* += 1;
        const decoded: ?u8 = switch (escape) {
            '"', '\\', '/' => escape,
            'b' => 8,
            'f' => 12,
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            else => null,
        };
        if (decoded) |byte| {
            out.append(allocator, byte) catch fatal();
            continue;
        }
        if (escape != 'u') return false;
        cursor.* -= 2;
        var codepoint = unicode16(cursor) orelse return false;
        if (codepoint >= 0xd800 and codepoint <= 0xdbff) {
            const low_start = cursor.*;
            const low = unicode16(cursor);
            if (low != null and low.? >= 0xdc00 and low.? <= 0xdfff) {
                codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low.? - 0xdc00);
            } else {
                // A non-low escape belongs to the next iteration. A lone high
                // surrogate becomes replacement text without consuming it.
                cursor.* = low_start;
                codepoint = 0xfffd;
            }
        }
        appendCodepoint(&out, codepoint);
    }
    if (cursor.*[0] != '"') return false;
    cursor.* += 1;
    // Embedded decoded NULs remain in the owned allocation, exactly as in C.
    // Its ordinary C-string consumers still stop at the first NUL.
    output.* = (out.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    return true;
}

pub export fn ds4_server_json_number(cursor: Cursor, output: *f64) bool {
    @setFloatMode(.strict);
    ds4_server_json_ws(cursor);
    var end: [*c]u8 = null;
    const value = c.strtod(cursor.*, &end);
    if (end == cursor.*) return false;
    cursor.* = end;
    output.* = value;
    return true;
}
pub export fn ds4_server_json_int(cursor: Cursor, output: *c_int) bool {
    @setFloatMode(.strict);
    var value: f64 = undefined;
    if (!ds4_server_json_number(cursor, &value)) return false;
    // strtod accepts NaN/Infinity. The comparison deliberately maps NaN and
    // negatives to zero before the otherwise undefined integer conversion.
    if (!(value >= 0)) value = 0;
    const maximum: f64 = @floatFromInt(c.INT_MAX);
    if (value > maximum) value = maximum;
    output.* = @intFromFloat(value);
    return true;
}
pub export fn ds4_server_json_bool(cursor: Cursor, output: *bool) bool {
    ds4_server_json_ws(cursor);
    if (ds4_server_json_lit(cursor, "true")) {
        output.* = true;
        return true;
    }
    if (ds4_server_json_lit(cursor, "false")) {
        output.* = false;
        return true;
    }
    return false;
}

fn skipArray(cursor: Cursor, depth: usize) bool {
    if (depth >= c.DS4_SERVER_JSON_MAX_NESTING) return false;
    ds4_server_json_ws(cursor);
    if (cursor.*[0] != '[') return false;
    cursor.* += 1;
    ds4_server_json_ws(cursor);
    if (cursor.*[0] == ']') {
        cursor.* += 1;
        return true;
    }
    while (true) {
        if (!skipValue(cursor, depth + 1)) return false;
        ds4_server_json_ws(cursor);
        if (cursor.*[0] == ']') {
            cursor.* += 1;
            return true;
        }
        if (cursor.*[0] != ',') return false;
        cursor.* += 1;
    }
}
fn skipObject(cursor: Cursor, depth: usize) bool {
    if (depth >= c.DS4_SERVER_JSON_MAX_NESTING) return false;
    ds4_server_json_ws(cursor);
    if (cursor.*[0] != '{') return false;
    cursor.* += 1;
    ds4_server_json_ws(cursor);
    if (cursor.*[0] == '}') {
        cursor.* += 1;
        return true;
    }
    while (true) {
        var key: [*c]u8 = null;
        if (!ds4_server_json_string(cursor, &key)) return false;
        c.free(key);
        ds4_server_json_ws(cursor);
        if (cursor.*[0] != ':') return false;
        cursor.* += 1;
        if (!skipValue(cursor, depth + 1)) return false;
        ds4_server_json_ws(cursor);
        if (cursor.*[0] == '}') {
            cursor.* += 1;
            return true;
        }
        if (cursor.*[0] != ',') return false;
        cursor.* += 1;
    }
}
fn skipValue(cursor: Cursor, depth: usize) bool {
    ds4_server_json_ws(cursor);
    switch (cursor.*[0]) {
        '"' => {
            var value: [*c]u8 = null;
            const result = ds4_server_json_string(cursor, &value);
            c.free(value);
            return result;
        },
        '{' => return skipObject(cursor, depth),
        '[' => return skipArray(cursor, depth),
        else => {},
    }
    if (ds4_server_json_lit(cursor, "true") or ds4_server_json_lit(cursor, "false") or ds4_server_json_lit(cursor, "null")) return true;
    var value: f64 = undefined;
    return ds4_server_json_number(cursor, &value);
}
pub export fn ds4_server_json_skip(cursor: Cursor) bool {
    return skipValue(cursor, 0);
}
pub export fn ds4_server_json_raw(cursor: Cursor, output: *[*c]u8) bool {
    ds4_server_json_ws(cursor);
    const begin = cursor.*;
    if (!ds4_server_json_skip(cursor)) return false;
    const count = @intFromPtr(cursor.*) - @intFromPtr(begin);
    // Unlike json_string, failure intentionally leaves the previous output
    // pointer untouched. Only a successfully captured raw value transfers one.
    output.* = (allocator.dupeSentinel(u8, begin[0..count], 0) catch fatal()).ptr;
    return true;
}

pub export fn ds4_server_json_minify(raw: [*c]const u8) [*c]u8 {
    var cursor: [*c]const u8 = if (raw == null) "null" else raw;
    ds4_server_json_ws(&cursor);
    const begin = cursor;
    if (!ds4_server_json_skip(&cursor)) {
        return (allocator.dupeSentinel(u8, if (raw == null) "null" else std.mem.span(raw), 0) catch fatal()).ptr;
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    var in_string = false;
    var escaped = false;
    for (begin[0 .. @intFromPtr(cursor) - @intFromPtr(begin)]) |value| {
        if (in_string) {
            out.append(allocator, value) catch fatal();
            if (escaped) escaped = false else if (value == '\\') escaped = true else if (value == '"') in_string = false;
        } else if (value == '"') {
            in_string = true;
            out.append(allocator, value) catch fatal();
        } else if (c.isspace(value) == 0) out.append(allocator, value) catch fatal();
    }
    return (out.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
}
