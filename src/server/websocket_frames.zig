//! RFC6455 server transport. Responses and connection/job ownership live in
//! the caller. One reader owns fragmentation state; callers serialize writes
//! so an event's header and payload cannot interleave with another event.
const std = @import("std");
const c = @import("server_http_c");
const Allocator = std.mem.Allocator;
pub const max_message_bytes = 64 * 1024 * 1024;

pub const Error = error{ Closed, Stopped, Expired, Io, Protocol, InvalidUtf8, MessageTooLarge } || Allocator.Error;
pub const Opcode = enum(u4) { continuation = 0, text = 1, binary = 2, close = 8, ping = 9, pong = 10 };
pub const Event = union(enum) {
    text: []u8,
    binary: []u8,
    ping: []u8,
    pong: []u8,
    close: []u8,

    pub fn payload(self: Event) []u8 {
        return switch (self) {
            inline else => |bytes| bytes,
        };
    }
    pub fn deinit(self: Event, allocator: Allocator) void {
        allocator.free(self.payload());
    }
};

pub fn milliseconds() i64 {
    var value: c.struct_timeval = undefined;
    _ = c.gettimeofday(&value, null);
    return @as(i64, @intCast(value.tv_sec)) * 1000 + @divTrunc(value.tv_usec, 1000);
}

pub fn headerValue(header: []const u8, name: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, header, "\r\n");
    _ = lines.next(); // HTTP request line.
    while (lines.next()) |line| {
        const colon = std.mem.find(u8, line, ":") orelse continue;
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, line[0..colon], " \t"), name))
            return std.mem.trim(u8, line[colon + 1 ..], " \t");
    }
    return null;
}

pub fn hasToken(value: []const u8, token: []const u8) bool {
    var parts = std.mem.splitScalar(u8, value, ',');
    while (parts.next()) |part| {
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, part, " \t"), token)) return true;
    }
    return false;
}

pub fn acceptKey(key: []const u8) error{BadHandshake}![28]u8 {
    // The handshake nonce is 16 random bytes, encoded with standard Base64.
    if (key.len != 24) return error.BadHandshake;
    const size = std.base64.standard.Decoder.calcSizeForSlice(key) catch return error.BadHandshake;
    if (size != 16) return error.BadHandshake;
    var nonce: [16]u8 = undefined;
    std.base64.standard.Decoder.decode(&nonce, key) catch return error.BadHandshake;
    var hash = std.crypto.hash.Sha1.init(.{});
    hash.update(key);
    hash.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    var digest: [20]u8 = undefined;
    hash.final(&digest);
    var result: [28]u8 = undefined;
    _ = std.base64.standard.Encoder.encode(&result, &digest);
    return result;
}

pub const Header = struct { bytes: [10]u8, len: usize };
pub fn frameHeader(opcode: Opcode, length: usize) Header {
    var result: Header = .{ .bytes = undefined, .len = 2 };
    result.bytes[0] = 0x80 | @as(u8, @backingInt(opcode));
    if (length < 126) {
        result.bytes[1] = @intCast(length);
    } else if (length <= 65535) {
        result.bytes[1] = 126;
        std.mem.writeInt(u16, result.bytes[2..4], @intCast(length), .big);
        result.len = 4;
    } else {
        result.bytes[1] = 127;
        std.mem.writeInt(u64, result.bytes[2..10], length, .big);
        result.len = 10;
    }
    return result;
}

pub const Connection = struct {
    descriptor: c_int,
    stop_requested: *const volatile c.sig_atomic_t,
    /// Optional absolute wall-clock deadline, including partial-frame waits.
    expires_at_ms: ?i64 = null,

    pub fn accept(self: Connection, key: []const u8) bool {
        const accepted = acceptKey(key) catch return false;
        var bytes: [256]u8 = undefined;
        const response = std.mem.print(&bytes, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {s}\r\n\r\n", .{accepted}) catch unreachable;
        return self.sendAll(response);
    }

    pub fn send(self: Connection, opcode: Opcode, bytes: []const u8) bool {
        if (@backingInt(opcode) >= 8 and bytes.len > 125) return false;
        const head = frameHeader(opcode, bytes.len);
        return self.sendAll(head.bytes[0..head.len]) and self.sendAll(bytes);
    }
    pub fn sendText(self: Connection, bytes: []const u8) bool {
        return self.send(.text, bytes);
    }
    pub fn sendPong(self: Connection, bytes: []const u8) bool {
        return self.send(.pong, bytes);
    }
    pub fn sendClose(self: Connection, code: u16, reason: []const u8) bool {
        if (reason.len > 123) return false;
        var payload: [125]u8 = undefined;
        std.mem.writeInt(u16, payload[0..2], code, .big);
        @memcpy(payload[2 .. reason.len + 2], reason);
        return self.send(.close, payload[0 .. reason.len + 2]);
    }
    fn sendAll(self: Connection, bytes: []const u8) bool {
        return c.ds4_server_http_send_all(self.descriptor, bytes.ptr, bytes.len, self.stop_requested);
    }
};

fn closeCode(code: u16) bool {
    return (code >= 3000 and code < 5000) or
        (code >= 1000 and code <= 1014 and code != 1004 and code != 1005 and code != 1006);
}

pub const Reader = struct {
    allocator: Allocator,
    initial: []const u8 = &.{},
    initial_pos: usize = 0,
    fragment: std.ArrayList(u8) = .empty,
    fragment_opcode: ?Opcode = null,

    pub fn init(allocator: Allocator) Reader {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *Reader) void {
        self.fragment.deinit(self.allocator);
    }
    /// Bytes beyond the HTTP header may already have been received. Borrow
    /// them until consumed; this call must not replace unread bytes.
    pub fn feed(self: *Reader, bytes: []const u8) void {
        std.debug.assert(self.initial_pos == self.initial.len);
        self.initial = bytes;
        self.initial_pos = 0;
    }

    fn exact(self: *Reader, connection: Connection, bytes: []u8) Error!void {
        if (connection.stop_requested.* != 0) return error.Stopped;
        if (connection.expires_at_ms) |limit| if (milliseconds() >= limit) return error.Expired;
        var position: usize = 0;
        const buffered = @min(bytes.len, self.initial.len - self.initial_pos);
        @memcpy(bytes[0..buffered], self.initial[self.initial_pos .. self.initial_pos + buffered]);
        self.initial_pos += buffered;
        position += buffered;
        while (position < bytes.len) {
            if (connection.stop_requested.* != 0) return error.Stopped;
            if (connection.expires_at_ms) |limit| if (milliseconds() >= limit) return error.Expired;
            var descriptor = std.mem.zeroes(c.struct_pollfd);
            descriptor.fd = connection.descriptor;
            descriptor.events = c.POLLIN;
            const ready = c.poll(&descriptor, 1, 50);
            if (ready < 0) {
                if (c.__error().* == c.EINTR) continue;
                return error.Io;
            }
            if (ready == 0) continue;
            if ((descriptor.revents & (c.POLLERR | c.POLLNVAL)) != 0) return error.Io;
            // POLLHUP can accompany the final readable bytes. recv decides
            // whether the peer has actually reached EOF.
            const count = c.recv(connection.descriptor, bytes[position..].ptr, bytes.len - position, 0);
            if (count < 0) {
                const failure = c.__error().*;
                if (failure == c.EINTR or failure == c.EAGAIN or failure == c.EWOULDBLOCK) continue;
                return error.Io;
            }
            if (count == 0) return error.Closed;
            position += @intCast(count);
        }
    }

    /// Returns a whole data message or one control frame. Ping/pong can be
    /// delivered between fragments without losing the partially read message.
    /// The caller frees the returned event and handles control-frame replies.
    pub fn read(self: *Reader, connection: Connection) Error!Event {
        while (true) {
            var base: [2]u8 = undefined;
            try self.exact(connection, &base);
            if ((base[0] & 0x70) != 0 or (base[1] & 0x80) == 0) return error.Protocol;
            const final = (base[0] & 0x80) != 0;
            const opcode: Opcode = switch (base[0] & 15) {
                0 => .continuation,
                1 => .text,
                2 => .binary,
                8 => .close,
                9 => .ping,
                10 => .pong,
                else => return error.Protocol,
            };
            const control = @backingInt(opcode) >= 8;
            const small = base[1] & 127;
            if (control and (!final or small > 125)) return error.Protocol;
            var length: u64 = small;
            if (small == 126) {
                var extra: [2]u8 = undefined;
                try self.exact(connection, &extra);
                length = std.mem.readInt(u16, &extra, .big);
                if (length < 126) return error.Protocol;
            } else if (small == 127) {
                var extra: [8]u8 = undefined;
                try self.exact(connection, &extra);
                length = std.mem.readInt(u64, &extra, .big);
                if (length <= 65535 or (extra[0] & 128) != 0) return error.Protocol;
            }
            if (length > max_message_bytes or (!control and self.fragment.items.len > max_message_bytes - length)) return error.MessageTooLarge;
            if (!control) {
                if (opcode == .continuation and self.fragment_opcode == null) return error.Protocol;
                if (opcode != .continuation and self.fragment_opcode != null) return error.Protocol;
            }
            var mask: [4]u8 = undefined;
            try self.exact(connection, &mask);
            const payload = try self.allocator.alloc(u8, @intCast(length));
            var payload_owned = true;
            errdefer if (payload_owned) self.allocator.free(payload);
            try self.exact(connection, payload);
            for (payload, 0..) |*byte, position| byte.* ^= mask[position & 3];
            if (control) {
                switch (opcode) {
                    .ping => return .{ .ping = payload },
                    .pong => return .{ .pong = payload },
                    .close => {
                        if (payload.len == 1) return error.Protocol;
                        if (payload.len >= 2) {
                            if (!closeCode(std.mem.readInt(u16, payload[0..2], .big))) return error.Protocol;
                            if (!std.unicode.utf8ValidateSlice(payload[2..])) return error.InvalidUtf8;
                        }
                        return .{ .close = payload };
                    },
                    else => unreachable,
                }
            }
            if (opcode != .continuation and final) {
                if (opcode == .text and !std.unicode.utf8ValidateSlice(payload)) return error.InvalidUtf8;
                return if (opcode == .text) .{ .text = payload } else .{ .binary = payload };
            }
            // Both append success and failure release this frame allocation.
            // The fragment list owns its independent copy until completion.
            if (opcode != .continuation) self.fragment_opcode = opcode;
            try self.fragment.appendSlice(self.allocator, payload);
            self.allocator.free(payload);
            payload_owned = false;
            if (!final) continue;
            const kind = self.fragment_opcode.?;
            self.fragment_opcode = null;
            const message = try self.fragment.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(message);
            if (kind == .text and !std.unicode.utf8ValidateSlice(message)) return error.InvalidUtf8;
            return if (kind == .text) .{ .text = message } else .{ .binary = message };
        }
    }
};
