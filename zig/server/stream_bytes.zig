//! Byte visibility and owned wire text shared by the protocol projections.
//! These operate on already-sampled text; they never alter its model transcript.
const std = @import("std");
pub const c = @import("server_protocol_c");
pub const allocator = std.heap.c_allocator;

pub fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
pub fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
pub fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
pub fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    return if (count == 0) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
pub fn at(raw: []const u8, position: usize, literal: []const u8) bool {
    return position <= raw.len and std.mem.startsWith(u8, raw[position..], literal);
}
pub fn partial(raw: []const u8, position: usize, literal: []const u8) bool {
    return position <= raw.len and raw.len - position < literal.len and
        std.mem.startsWith(u8, literal, raw[position..]);
}
pub fn whitespace(raw: []const u8, position: usize) usize {
    var next = position;
    while (next < raw.len and c.isspace(raw[next]) != 0) next += 1;
    return next;
}
pub fn guardSecondReasoning(request: *const c.request) bool {
    return c.ds4_think_mode_enabled(request.think_mode) and request.has_tools and
        request.model_syntax != c.SERVER_MODEL_SYNTAX_QWEN;
}

pub const Text = struct {
    list: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    pub fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal();
    }
    pub fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    pub fn number(self: *Text, value: anytype) void {
        self.list.print(allocator, "{d}", .{value}) catch fatal();
    }
    pub fn fragment(self: *Text, value: []const u8) void {
        const hex = "0123456789abcdef";
        for (value) |character| switch (character) {
            '"', '\\' => self.append(&.{ '\\', character }),
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            0...8, 11, 12, 14...31 => self.append(&.{ '\\', 'u', '0', '0', hex[character >> 4], hex[character & 15] }),
            else => self.byte(character),
        };
    }
    pub fn quoted(self: *Text, value: []const u8) void {
        // json_escape_n first makes a terminated copy. Preserve its embedded
        // NUL rule, including when a sampled raw buffer contains later bytes.
        const visible = value[0..(std.mem.findScalar(u8, value, 0) orelse value.len)];
        self.byte('"');
        self.fragment(visible);
        self.byte('"');
    }
    pub fn terminated(self: *Text) [*:0]const u8 {
        self.list.ensureUnusedCapacity(allocator, 1) catch fatal();
        self.list.items.ptr[self.list.items.len] = 0;
        return @ptrCast(self.list.items.ptr);
    }
    pub fn appendTo(self: *const Text, buffer: *c.buf) void {
        // Adopt the caller's one actual allocation only for this append.
        var list: std.ArrayList(u8) = .empty;
        if (buffer.ptr != null) list.items = buffer.ptr[0..buffer.len];
        list.capacity = buffer.cap;
        list.appendSlice(allocator, self.list.items) catch fatal();
        list.ensureUnusedCapacity(allocator, 1) catch fatal();
        list.items.ptr[list.items.len] = 0;
        buffer.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
    }
    pub fn objectString(self: *Text, arguments: [*c]const u8) void {
        var object = std.mem.zeroes(c.buf);
        defer c.free(object.ptr);
        c.ds4_server_render_object(&object, arguments);
        self.quoted(if (object.ptr == null) "{}" else text(object.ptr));
    }
};

fn expectedLength(character: u8) usize {
    return switch (character) {
        0xc2...0xdf => 2,
        0xe0...0xef => 3,
        0xf0...0xf4 => 4,
        else => 1,
    };
}
/// Tokenizers can split UTF-8 across tokens. Hold only its incomplete tail so
/// clients cannot replace bytes with U+FFFD and break the next cache prefix.
pub fn utf8Limit(raw: [*c]const u8, start: usize, limit: usize, final: bool) usize {
    if (final or raw == null or limit <= start) return limit;
    var position = limit;
    var continuation: usize = 0;
    while (position > start and continuation < 4 and raw[position - 1] & 0xc0 == 0x80) {
        position -= 1;
        continuation += 1;
    }
    if (position == limit) return if (expectedLength(raw[limit - 1]) > 1) limit - 1 else limit;
    if (position == start and raw[position] & 0xc0 == 0x80) return start;
    const lead = position - 1;
    return if (limit - lead < expectedLength(raw[lead])) lead else limit;
}
pub fn textLimit(raw: [*c]const u8, start: usize, length: usize, has_tools: bool, final: bool) usize {
    if (length <= start) return length;
    var limit = length;
    if (has_tools) {
        const tool = c.ds4_server_reply_tool_start(raw + start);
        if (tool != null) {
            limit = c.ds4_server_reply_trim_separator(raw, start, @intFromPtr(tool) - @intFromPtr(raw));
            return utf8Limit(raw, start, limit, true);
        }
        if (!final) {
            // Whitespace immediately before hidden DSML is syntax too. Wait
            // for a following ordinary byte before exposing it to the client.
            while (limit > start and c.isspace(raw[limit - 1]) != 0) limit -= 1;
            const scan = if (length - start > 80) length - 80 else start;
            var position = length;
            while (position > scan) : (position -= 1) {
                if (raw[position - 1] == '<') {
                    limit = @min(limit, position - 1);
                    break;
                }
            }
            limit = c.ds4_server_reply_trim_separator(raw, start, limit);
        }
    }
    return utf8Limit(raw, start, limit, final);
}
pub fn parameterLimit(raw: [*c]const u8, start: usize, length: usize, closing: [*c]const u8, string: bool) usize {
    var limit = length;
    const ending = text(closing);
    const scan = if (length > start + ending.len) length - ending.len else start;
    var position = length;
    while (position > scan) : (position -= 1) {
        if (raw[position - 1] != '<') continue;
        const marker = position - 1;
        const tail = length - marker;
        if (tail < ending.len and std.mem.eql(u8, raw[marker..length], ending[0..tail])) limit = marker;
        break;
    }
    if (string) {
        position = limit;
        while (position > start) : (position -= 1) {
            if (raw[position - 1] != '&') continue;
            if (c.ds4_tool_text_partial_escape(raw + position - 1, limit - position + 1, closing)) limit = position - 1;
            break;
        }
    }
    return utf8Limit(raw, start, limit, false);
}

var fallback_counter: u64 = 0;
fn randomBytes(out: []u8) bool {
    const fd = c.open("/dev/urandom", c.O_RDONLY);
    if (fd < 0) return false;
    defer _ = c.close(fd);
    var done: usize = 0;
    while (done < out.len) {
        const count = c.read(fd, out.ptr + done, out.len - done);
        if (count < 0 and c.__error().* == c.EINTR) continue;
        if (count <= 0) return false;
        done += @intCast(count);
    }
    return true;
}
/// All selected C/native tool-ID callers share this one counter. Responses
/// item IDs keep their distinct existing 12-byte action and fallback counter.
pub fn randomToolId(out: []u8, api: c.api_style) void {
    if (out.len == 0) return;
    const prefix: []const u8 = if (api == c.API_ANTHROPIC) "toolu_" else "call_";
    const copied = @min(prefix.len, out.len - 1);
    @memcpy(out[0..copied], prefix[0..copied]);
    out[copied] = 0;
    if (prefix.len >= out.len) return;
    var random: [16]u8 = undefined;
    if (!randomBytes(&random)) {
        const time = @as(u64, @bitCast(@as(i64, c.time(null))));
        const a = time << 32 ^ @as(u64, @intCast(c.getpid()));
        fallback_counter +%= 1;
        const b = fallback_counter ^ @as(u64, @intFromPtr(out.ptr));
        @memcpy(random[0..8], std.mem.asBytes(&a));
        @memcpy(random[8..16], std.mem.asBytes(&b));
    }
    const hex = "0123456789abcdef";
    var position = prefix.len;
    for (random) |value| {
        if (position + 2 >= out.len) break;
        out[position] = hex[value >> 4];
        out[position + 1] = hex[value & 15];
        position += 2;
    }
    out[position] = 0;
}
