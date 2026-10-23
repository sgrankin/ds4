//! Local HTTP/WebSocket/CDP transport. Descriptors remain scoped to a request
//! or Connection; cancellation is checked between bounded poll slices.
const std = @import("std");
const common = @import("web_common.zig");
const c = common.c;
const allocator = common.allocator;
const Buffer = common.Buffer;
const Errors = common.Errors;
const timeout = c.DS4_WEB_CONNECT_TIMEOUT_MS;
fn tcp(host: [*:0]const u8, port: c_int, errors: Errors) c_int {
    var service: [32]u8 = undefined;
    _ = c.snprintf(&service, service.len, "%d", port);
    var hints = std.mem.zeroes(c.struct_addrinfo);
    hints.ai_socktype = c.SOCK_STREAM;
    hints.ai_family = c.AF_UNSPEC;
    var addresses: [*c]c.struct_addrinfo = null;
    const status = c.getaddrinfo(host, &service, &hints, &addresses);
    if (status != 0) {
        errors.set("getaddrinfo %s: %s", .{ host, c.gai_strerror(status) });
        return -1;
    }
    var descriptor: c_int = -1;
    var cursor = addresses;
    while (cursor != null) : (cursor = cursor.*.ai_next) {
        const address: *c.struct_addrinfo = @ptrCast(cursor);
        descriptor = c.socket(address.ai_family, address.ai_socktype, address.ai_protocol);
        if (descriptor < 0) continue;
        const flags = c.fcntl(descriptor, c.F_GETFL, @as(c_int, 0));
        if (flags >= 0) _ = c.fcntl(descriptor, c.F_SETFL, flags | c.O_NONBLOCK);
        const connected = c.connect(descriptor, address.ai_addr, address.ai_addrlen);
        if (connected == 0) {
            if (flags >= 0) _ = c.fcntl(descriptor, c.F_SETFL, flags);
            break;
        }
        if (c.__error().* == c.EINPROGRESS) {
            var poll = c.struct_pollfd{ .fd = descriptor, .events = c.POLLOUT, .revents = 0 };
            if (c.poll(&poll, 1, timeout) > 0) {
                var socket_error: c_int = 0;
                var length: c.socklen_t = @sizeOf(c_int);
                _ = c.getsockopt(descriptor, c.SOL_SOCKET, c.SO_ERROR, &socket_error, &length);
                if (socket_error == 0) {
                    if (flags >= 0) _ = c.fcntl(descriptor, c.F_SETFL, flags);
                    break;
                }
                c.__error().* = socket_error;
            }
        }
        _ = c.close(descriptor);
        descriptor = -1;
    }
    c.freeaddrinfo(addresses);
    if (descriptor < 0) errors.set("connect %s:%d failed: %s", .{ host, port, c.strerror(c.__error().*) });
    return descriptor;
}
fn writeAll(fd: c_int, bytes: []const u8) bool {
    var position: usize = 0;
    while (position < bytes.len) {
        const size = if (@hasDecl(c, "MSG_NOSIGNAL")) c.send(fd, bytes.ptr + position, bytes.len - position, c.MSG_NOSIGNAL) else c.write(fd, bytes.ptr + position, bytes.len - position);
        if (size < 0 and c.__error().* == c.EINTR) continue;
        if (size <= 0) return false;
        position += @intCast(size);
    }
    return true;
}
fn readSome(fd: c_int, bytes: []u8, milliseconds: c_int) isize {
    var poll = c.struct_pollfd{ .fd = fd, .events = c.POLLIN, .revents = 0 };
    const status = c.poll(&poll, 1, milliseconds);
    if (status <= 0) return if (status == 0) 0 else -1;
    while (true) {
        const size = c.read(fd, bytes.ptr, bytes.len);
        if (size < 0 and c.__error().* == c.EINTR) continue;
        return size;
    }
}
pub fn http(method: [*:0]const u8, port: c_int, path: [*:0]const u8, errors: Errors) ?[:0]u8 {
    const fd = tcp("127.0.0.1", port, errors);
    if (fd < 0) return null;
    defer _ = c.close(fd);
    var line: [512]u8 = undefined;
    _ = c.snprintf(&line, line.len, "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n", method, path, port);
    if (!writeAll(fd, common.text(&line))) {
        errors.set("write HTTP request failed: %s", .{c.strerror(c.__error().*)});
        return null;
    }
    var response = Buffer{};
    defer response.deinit();
    var chunk: [4096]u8 = undefined;
    while (true) {
        const size = readSome(fd, &chunk, timeout);
        if (size < 0) {
            errors.set("read HTTP response failed: %s", .{c.strerror(c.__error().*)});
            return null;
        }
        if (size == 0) break;
        response.add(chunk[0..@intCast(size)]);
    }
    if (response.bytes().len == 0) {
        errors.set("empty HTTP response", .{});
        return null;
    }
    const bytes = common.visible(response.bytes());
    const body = std.mem.find(u8, bytes, "\r\n\r\n") orelse {
        errors.set("malformed HTTP response", .{});
        return null;
    };
    return common.duplicate(bytes[body + 4 ..]);
}
pub const Connection = struct {
    fd: c_int = -1,
    next_id: c_int = 0,
    web: ?*c.ds4_web,
    pub fn close(self: *Connection) void {
        if (self.fd >= 0) {
            _ = c.close(self.fd);
            self.fd = -1;
        }
    }
    pub fn connect(self: *Connection, url: []const u8, errors: Errors) bool {
        if (!std.mem.startsWith(u8, url, "ws://")) {
            const value = common.duplicate(url);
            defer allocator.free(value);
            errors.set("unsupported websocket URL: %s", .{value.ptr});
            return false;
        }
        const slash = std.mem.findScalarPos(u8, url, 5, '/') orelse {
            errors.set("malformed websocket URL", .{});
            return false;
        };
        var host: [256]u8 = undefined;
        const length = @min(slash - 5, host.len - 1);
        @memcpy(host[0..length], url[5..][0..length]);
        host[length] = 0;
        var port: c_int = 80;
        if (std.mem.findScalarLast(u8, host[0..length], ':')) |colon| {
            host[colon] = 0;
            port = c.atoi(@as([*c]const u8, &host) + colon + 1);
        }
        const fd = tcp(@ptrCast(&host), port, errors);
        if (fd < 0) return false;
        var keep = false;
        defer {
            if (!keep) _ = c.close(fd);
        }
        var random: [16]u8 = undefined;
        common.random(&random);
        var key: [24]u8 = undefined;
        _ = std.base64.standard.Encoder.encode(&key, &random);
        var key_z: [25]u8 = undefined;
        @memcpy(key_z[0..24], &key);
        key_z[24] = 0;
        const path = common.duplicate(url[slash..]);
        defer allocator.free(path);
        var request: [512]u8 = undefined;
        _ = c.snprintf(&request, request.len, "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n", path.ptr, @as([*c]const u8, &host), port, @as([*c]const u8, &key_z));
        if (!writeAll(fd, common.text(&request))) {
            errors.set("websocket handshake write failed", .{});
            return false;
        }
        var response = Buffer{};
        defer response.deinit();
        var chunk: [1024]u8 = undefined;
        const deadline = common.now() + @as(f64, @floatFromInt(timeout)) / 1000;
        while (std.mem.find(u8, common.visible(response.bytes()), "\r\n\r\n") == null) {
            if (common.cancel(self.web, errors)) return false;
            const current = common.now();
            if (current >= deadline) {
                errors.set("websocket handshake read failed", .{});
                return false;
            }
            const remaining: c_int = @intFromFloat((deadline - current) * 1000);
            const size = readSome(fd, &chunk, @min(100, @max(1, remaining)));
            if (size < 0) {
                errors.set("websocket handshake read failed", .{});
                return false;
            }
            if (size == 0) continue;
            response.add(chunk[0..@intCast(size)]);
            if (response.bytes().len > 8192) break;
        }
        if (std.mem.find(u8, common.visible(response.bytes()), " 101 ") == null) {
            errors.set("websocket handshake rejected", .{});
            return false;
        }
        self.fd = fd;
        self.next_id = 1;
        keep = true;
        return true;
    }
    fn readExact(self: *Connection, output: []u8, errors: Errors) bool {
        const deadline = common.now() + @as(f64, @floatFromInt(c.DS4_WEB_CDP_TIMEOUT_MS)) / 1000;
        var position: usize = 0;
        while (position < output.len) {
            if (common.cancel(self.web, errors)) return false;
            const current = common.now();
            if (current >= deadline) {
                errors.set("websocket read timeout", .{});
                return false;
            }
            const remaining: c_int = @intFromFloat((deadline - current) * 1000);
            const size = readSome(self.fd, output[position..], @min(100, @max(1, remaining)));
            if (size < 0) {
                errors.set("websocket frame read failed", .{});
                return false;
            }
            if (size == 0) continue;
            position += @intCast(size);
        }
        return true;
    }
    fn sendText(self: *Connection, text: []const u8, errors: Errors) bool {
        var frame = Buffer{};
        defer frame.deinit();
        frame.byte(0x81);
        if (text.len < 126) frame.byte(0x80 | @as(u8, @intCast(text.len))) else if (text.len <= 0xffff) {
            frame.add(&.{ 0x80 | 126, @truncate(text.len >> 8), @truncate(text.len) });
        } else {
            frame.byte(0x80 | 127);
            var length: [8]u8 = undefined;
            std.mem.writeInt(u64, &length, text.len, .big);
            frame.add(&length);
        }
        var mask: [4]u8 = undefined;
        common.random(&mask);
        frame.add(&mask);
        for (text, 0..) |byte, i| frame.byte(byte ^ mask[i & 3]);
        if (!writeAll(self.fd, frame.bytes())) {
            errors.set("websocket write failed: %s", .{c.strerror(c.__error().*)});
            return false;
        }
        return true;
    }
    fn pong(self: *Connection, payload: []const u8) void {
        const count = @min(payload.len, 125);
        var frame: [131]u8 = undefined;
        frame[0] = 0x8a;
        frame[1] = 0x80 | @as(u8, @intCast(count));
        common.random(frame[2..6]);
        for (payload[0..count], 0..) |byte, i| frame[6 + i] = byte ^ frame[2 + (i & 3)];
        _ = writeAll(self.fd, frame[0 .. 6 + count]);
    }
    fn message(self: *Connection, errors: Errors) ?[:0]u8 {
        var output = Buffer{};
        defer output.deinit();
        while (true) {
            var header: [2]u8 = undefined;
            if (!self.readExact(&header, errors)) return null;
            const final = header[0] & 0x80 != 0;
            const opcode = header[0] & 15;
            const masked = header[1] & 0x80 != 0;
            var length: u64 = header[1] & 127;
            if (length == 126) {
                var bytes: [2]u8 = undefined;
                if (!self.readExact(&bytes, errors)) return null;
                length = std.mem.readInt(u16, &bytes, .big);
            } else if (length == 127) {
                var bytes: [8]u8 = undefined;
                if (!self.readExact(&bytes, errors)) return null;
                length = std.mem.readInt(u64, &bytes, .big);
            }
            var mask: [4]u8 = @splat(0);
            if (masked and !self.readExact(&mask, errors)) return null;
            if (length > c.DS4_WEB_MAX_RESULT_BYTES * 4) {
                errors.set("websocket message too large", .{});
                return null;
            }
            const payload = allocator.alloc(u8, @as(usize, @intCast(length)) + 1) catch common.fatal("ds4_web: malloc");
            defer allocator.free(payload);
            if (length != 0 and !self.readExact(payload[0..@intCast(length)], errors)) return null;
            if (masked) for (payload[0..@intCast(length)], 0..) |*byte, i| {
                byte.* ^= mask[i & 3];
            };
            payload[@intCast(length)] = 0;
            if (opcode == 8) {
                errors.set("websocket closed", .{});
                return null;
            } else if (opcode == 9) {
                self.pong(payload[0..@intCast(length)]);
            } else if (opcode == 1 or opcode == 0) {
                output.add(payload[0..@intCast(length)]);
                if (final) return output.take();
            }
        }
    }
    pub fn call(self: *Connection, method: []const u8, parameters: []const u8, errors: Errors) ?[:0]u8 {
        if (common.cancel(self.web, errors)) return null;
        const id = self.next_id;
        self.next_id +%= 1;
        var request = Buffer{};
        defer request.deinit();
        var head: [256]u8 = undefined;
        _ = c.snprintf(&head, head.len, "{\"id\":%d,\"method\":", id);
        request.add(common.text(&head));
        const quoted = common.quote(method);
        defer allocator.free(quoted);
        request.add(quoted);
        if (parameters.len != 0) {
            request.add(",\"params\":");
            request.add(parameters);
        }
        request.byte('}');
        if (!self.sendText(common.visible(request.bytes()), errors)) return null;
        while (true) {
            if (common.cancel(self.web, errors)) return null;
            const response = self.message(errors) orelse return null;
            const bytes = common.visible(response);
            if (std.mem.find(u8, bytes, "\"id\"")) |field| {
                if (std.mem.findScalarPos(u8, bytes, field, ':')) |colon| {
                    // atoi retains the existing spaces/sign/numeric grammar.
                    if (c.atoi(response.ptr + colon + 1) == id) return response;
                }
            }
            allocator.free(response);
        }
    }
    pub fn optional(self: *Connection, method: []const u8, parameters: []const u8) void {
        var errors: [160]u8 = @splat(0);
        if (self.call(method, parameters, .{ .bytes = &errors })) |response| allocator.free(response);
    }
    pub fn evaluate(self: *Connection, expression: []const u8, errors: Errors) ?[:0]u8 {
        const quoted = common.quote(expression);
        defer allocator.free(quoted);
        var parameters = Buffer{};
        defer parameters.deinit();
        parameters.add("{\"expression\":");
        parameters.add(quoted);
        parameters.add(",\"returnByValue\":true,\"awaitPromise\":true,\"includeCommandLineAPI\":true}");
        const response = self.call("Runtime.evaluate", parameters.bytes(), errors) orelse return null;
        defer allocator.free(response);
        const bytes = common.visible(response);
        if (std.mem.find(u8, bytes, "\"exceptionDetails\"") != null) {
            errors.set("JavaScript evaluation failed", .{});
            return null;
        }
        const value = common.string(bytes, "value");
        if (value == null) errors.set("Runtime.evaluate did not return a string", .{});
        return value;
    }
};
