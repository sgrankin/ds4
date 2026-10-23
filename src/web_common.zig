//! Shared owned bytes and the existing narrow CDP string projection grammar.
const std = @import("std");
pub const c = @import("web_c");
pub const allocator = std.heap.c_allocator;
pub fn fatal(operation: [*:0]const u8) noreturn {
    c.perror(operation);
    c.exit(1);
}
pub fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
pub fn visible(bytes: []const u8) []const u8 {
    return bytes[0 .. std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len];
}
pub fn duplicate(bytes: []const u8) [:0]u8 {
    return allocator.dupeSentinel(u8, bytes, 0) catch fatal("ds4_web: malloc");
}
pub const Buffer = struct {
    list: std.ArrayList(u8) = .empty,
    pub fn deinit(self: *Buffer) void {
        self.list.deinit(allocator);
    }
    pub fn add(self: *Buffer, data: []const u8) void {
        if (data.len == 0) return;
        const required = self.list.items.len + data.len + 1;
        if (required > self.list.capacity) {
            var capacity: usize = if (self.list.capacity == 0) 4096 else self.list.capacity * 2;
            while (capacity < required) capacity *= 2;
            self.list.ensureTotalCapacityPrecise(allocator, capacity) catch fatal("ds4_web: realloc");
        }
        self.list.appendSliceAssumeCapacity(data);
        self.list.items.ptr[self.list.items.len] = 0;
    }
    pub fn byte(self: *Buffer, value: u8) void {
        self.add(&.{value});
    }
    pub fn take(self: *Buffer) [:0]u8 {
        if (self.list.capacity == 0) return duplicate("");
        const result = self.list.items.ptr[0..self.list.items.len :0];
        self.list = .empty;
        return result;
    }
    pub fn bytes(self: *const Buffer) []const u8 {
        return self.list.items;
    }
};
pub const Errors = struct {
    bytes: []u8,
    pub fn set(self: Errors, comptime pattern: [*:0]const u8, arguments: anytype) void {
        if (self.bytes.len != 0) _ = @call(.auto, c.snprintf, .{ self.bytes.ptr, self.bytes.len, pattern } ++ arguments);
    }
    pub fn empty(self: Errors) bool {
        return self.bytes.len == 0 or self.bytes[0] == 0;
    }
    pub fn interrupted(self: Errors) bool {
        return self.bytes.len != 0 and std.mem.eql(u8, text(self.bytes.ptr), "interrupted");
    }
};
pub fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) / 1000000000.0;
}
pub fn cancelled(web: ?*c.ds4_web) bool {
    const owner = web orelse return false;
    return if (owner.cancel) |callback| callback(owner.cancel_privdata) else false;
}
pub fn cancel(web: ?*c.ds4_web, errors: Errors) bool {
    if (!cancelled(web)) return false;
    errors.set("interrupted", .{});
    return true;
}
pub fn sleep(web: ?*c.ds4_web, milliseconds: c_int) bool {
    var left = milliseconds;
    while (left > 0) {
        if (cancelled(web)) return false;
        const step = @min(left, 50);
        _ = c.usleep(@as(c.useconds_t, @intCast(step)) * 1000);
        left -= step;
    }
    return !cancelled(web);
}
pub fn log(web: *c.ds4_web, message: [*c]const u8) void {
    if (web.log) |callback| callback(web.log_privdata, message);
}
pub fn random(bytes: []u8) void {
    const fd = c.open("/dev/urandom", c.O_RDONLY);
    if (fd >= 0) {
        var position: usize = 0;
        while (position < bytes.len) {
            const size = c.read(fd, bytes.ptr + position, bytes.len - position);
            if (size < 0 and c.__error().* == c.EINTR) continue;
            if (size <= 0) break;
            position += @intCast(size);
        }
        _ = c.close(fd);
        if (position == bytes.len) return;
    }
    var value = @as(u64, @bitCast(@as(i64, c.time(null)))) ^ (@as(u64, @intCast(c.getpid())) << 32);
    for (bytes) |*byte| {
        value = value *% 6364136223846793005 +% 1;
        byte.* = @truncate(value >> 32);
    }
}
pub fn urlEncode(bytes: []const u8) [:0]u8 {
    var buffer = Buffer{};
    defer buffer.deinit();
    const hex = "0123456789ABCDEF";
    for (bytes) |byte| {
        if (c.isalnum(byte) != 0 or std.mem.indexOfScalar(u8, "-_.~", byte) != null) buffer.byte(byte) else buffer.add(&.{ '%', hex[byte >> 4], hex[byte & 15] });
    }
    return buffer.take();
}
pub fn quote(bytes: []const u8) [:0]u8 {
    var buffer = Buffer{};
    defer buffer.deinit();
    buffer.byte('"');
    const hex = "0123456789abcdef";
    for (bytes) |byte| switch (byte) {
        '\\' => buffer.add("\\\\"),
        '"' => buffer.add("\\\""),
        '\n' => buffer.add("\\n"),
        '\r' => buffer.add("\\r"),
        '\t' => buffer.add("\\t"),
        else => if (byte < 0x20) buffer.add(&.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15] }) else buffer.byte(byte),
    };
    buffer.byte('"');
    return buffer.take();
}
fn hex4(bytes: []const u8) ?u16 {
    if (bytes.len < 4) return null;
    var value: u16 = 0;
    for (bytes[0..4]) |byte| {
        const digit: u16 = switch (byte) {
            '0'...'9' => byte - '0',
            'a'...'f' => byte - 'a' + 10,
            'A'...'F' => byte - 'A' + 10,
            else => return null,
        };
        value = (value << 4) | digit;
    }
    return value;
}
fn unicode(buffer: *Buffer, code: u32) void {
    // Keep legacy lone-surrogate bytes: a strict full JSON decoder would reject
    // inputs the narrow original CDP projection accepted.
    if (code <= 0x7f) {
        buffer.byte(@intCast(code));
    } else if (code <= 0x7ff) {
        buffer.add(&.{ @intCast(0xc0 | (code >> 6)), @intCast(0x80 | (code & 63)) });
    } else if (code <= 0xffff) {
        buffer.add(&.{ @intCast(0xe0 | (code >> 12)), @intCast(0x80 | ((code >> 6) & 63)), @intCast(0x80 | (code & 63)) });
    } else {
        buffer.add(&.{ @intCast(0xf0 | (code >> 18)), @intCast(0x80 | ((code >> 12) & 63)), @intCast(0x80 | ((code >> 6) & 63)), @intCast(0x80 | (code & 63)) });
    }
}
fn parseString(source: []const u8) ?[:0]u8 {
    if (source.len == 0 or source[0] != '"') return null;
    var out = Buffer{};
    defer out.deinit();
    var position: usize = 1;
    while (position < source.len and source[position] != '"') {
        const byte = source[position];
        position += 1;
        if (byte != '\\') {
            out.byte(byte);
            continue;
        }
        if (position == source.len) return null;
        const escaped = source[position];
        position += 1;
        switch (escaped) {
            '"', '\\', '/' => out.byte(escaped),
            'b' => out.byte(8),
            'f' => out.byte(12),
            'n' => out.byte(10),
            'r' => out.byte(13),
            't' => out.byte(9),
            'u' => {
                const high = hex4(source[position..]) orelse return null;
                position += 4;
                if (high >= 0xd800 and high <= 0xdbff and std.mem.startsWith(u8, source[position..], "\\u")) {
                    const low = hex4(source[position + 2 ..]);
                    if (low != null and low.? >= 0xdc00 and low.? <= 0xdfff) {
                        unicode(&out, 0x10000 + ((@as(u32, high) - 0xd800) << 10) + low.? - 0xdc00);
                        position += 6;
                        continue;
                    }
                }
                unicode(&out, high);
            },
            else => out.byte(escaped),
        }
    }
    if (position == source.len) return null;
    return out.take();
}
pub fn string(json: []const u8, key: []const u8) ?[:0]u8 {
    var pattern: [128]u8 = undefined;
    _ = c.snprintf(&pattern, pattern.len, "\"%.*s\"", @as(c_int, @intCast(key.len)), key.ptr);
    const needle = text(&pattern);
    var position: usize = 0;
    while (std.mem.findPos(u8, json, position, needle)) |found| {
        position = found + needle.len;
        while (position < json.len and std.mem.indexOfScalar(u8, " \t\r\n", json[position]) != null) position += 1;
        const colon = position;
        position = @min(position + 1, json.len);
        if (colon == json.len or json[colon] != ':') continue;
        while (position < json.len and std.mem.indexOfScalar(u8, " \t\r\n", json[position]) != null) position += 1;
        if (position < json.len and json[position] == '"') return parseString(json[position..]);
    }
    return null;
}
