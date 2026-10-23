//! The server's one-request HTTP connection: framing, temporary buffers and
//! socket policy. Endpoint JSON, queueing and model state are separate owners.
//! The descriptor is borrowed; only the accept loop closes it. Request bodies
//! transfer to the caller as libc allocations, including an extra NUL byte.
const std = @import("std");
const c = @import("server_http_c");
const allocator = std.heap.c_allocator;
pub const Request = c.ds4_server_http_request;
const io_timeout_seconds = 10;
const send_stall_milliseconds = 2000;
const max_header_bytes = 64 * 1024;
const max_body_bytes = 64 * 1024 * 1024;
const cors_headers =
    "Access-Control-Allow-Origin: *\r\n" ++
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
    "Access-Control-Allow-Headers: *\r\n";

fn fatal(comptime message: [*:0]const u8) noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: %s\n", message);
    c.exit(1);
}
fn errnoValue() c_int {
    return c.__error().*;
}
fn wallMilliseconds() i64 {
    var time: c.struct_timeval = undefined;
    _ = c.gettimeofday(&time, null);
    return @as(i64, @intCast(time.tv_sec)) * 1000 + @divTrunc(time.tv_usec, 1000);
}
fn append(buffer: *std.ArrayList(u8), bytes: []const u8) void {
    if (bytes.len > std.math.maxInt(usize) - buffer.items.len - 1) fatal("buffer overflow");
    buffer.appendSlice(allocator, bytes) catch fatal("out of memory");
}
fn terminate(buffer: *std.ArrayList(u8)) void {
    buffer.ensureUnusedCapacity(allocator, 1) catch fatal("out of memory");
    buffer.items.ptr[buffer.items.len] = 0;
}
fn text(pointer: [*c]const u8) []const u8 {
    return std.mem.span(pointer);
}

pub const Connection = struct {
    descriptor: c_int,
    stop_requested: *const volatile c.sig_atomic_t,

    /// The deadline measures lack of progress, not the total response duration.
    /// Poll in short intervals so shutdown can cancel a slow streaming client.
    pub fn sendAll(self: Connection, bytes: []const u8) bool {
        var remaining = bytes;
        var deadline = wallMilliseconds() + send_stall_milliseconds;
        while (remaining.len != 0) {
            if (self.stop_requested.* != 0) return false;
            const written = c.send(self.descriptor, remaining.ptr, remaining.len, 0);
            if (written < 0) {
                const failure = errnoValue();
                if (failure == c.EINTR) continue;
                if (failure == c.EAGAIN or failure == c.EWOULDBLOCK) {
                    const wait = deadline - wallMilliseconds();
                    if (wait <= 0) return false;
                    var descriptor = std.mem.zeroes(c.struct_pollfd);
                    descriptor.fd = self.descriptor;
                    descriptor.events = c.POLLOUT;
                    const timeout: c_int = @intCast(@min(wait, 50));
                    const polled = while (true) {
                        const result = c.poll(&descriptor, 1, timeout);
                        if (result < 0 and errnoValue() == c.EINTR) continue;
                        break result;
                    };
                    if (polled < 0 or pollDisconnected(descriptor.revents)) return false;
                    continue;
                }
            }
            if (written <= 0) return false;
            remaining = remaining[@intCast(written)..];
            deadline = wallMilliseconds() + send_stall_milliseconds;
        }
        return true;
    }

    pub fn respond(self: Connection, enable_cors: bool, status: c_int, content_type: ?[]const u8, body: []const u8) bool {
        const reason: []const u8 = switch (status) {
            200 => "OK",
            204 => "No Content",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            500 => "Internal Server Error",
            else => "Error",
        };
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(allocator);
        header.print(allocator, "HTTP/1.1 {d} {s}\r\nContent-Length: {d}\r\n", .{ status, reason, body.len }) catch fatal("out of memory");
        if (content_type) |kind| {
            if (kind.len != 0) {
                append(&header, "Content-Type: ");
                append(&header, kind);
                append(&header, "\r\n");
            }
        }
        if (enable_cors) append(&header, cors_headers);
        append(&header, "Connection: close\r\n\r\n");
        return self.sendAll(header.items) and (body.len == 0 or self.sendAll(body));
    }

    pub fn sseHeaders(self: Connection, enable_cors: bool) bool {
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(allocator);
        append(&header, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n");
        if (enable_cors) append(&header, cors_headers);
        append(&header, "Connection: close\r\n\r\n");
        return self.sendAll(header.items);
    }

    pub fn sseEvent(self: Connection, event: []const u8, data: []const u8) bool {
        var frame: std.ArrayList(u8) = .empty;
        defer frame.deinit(allocator);
        append(&frame, "event: ");
        append(&frame, event);
        append(&frame, "\ndata: ");
        append(&frame, data);
        append(&frame, "\n\n");
        return self.sendAll(frame.items);
    }
};

/// Search all CRLF terminators before accepting LF-only framing, matching the
/// existing server even if an earlier LF-only terminator is also present.
pub fn headerEnd(bytes: []const u8) ?usize {
    if (std.mem.indexOf(u8, bytes, "\r\n\r\n")) |index| return index + 4;
    if (std.mem.indexOf(u8, bytes, "\n\n")) |index| return index + 2;
    return null;
}

/// First matching header wins. The caller keeps the whole received buffer NUL
/// terminated; strtol deliberately preserves the legacy permissive grammar.
pub fn contentLength(header: []const u8) c_long {
    var start: usize = 0;
    while (start < header.len) {
        const end = std.mem.indexOfScalarPos(u8, header, start, '\n') orelse header.len;
        const line_end = if (end > start and header[end - 1] == '\r') end - 1 else end;
        if (line_end - start >= 15 and c.strncasecmp(header.ptr + start, "Content-Length:", 15) == 0) {
            var value = start + 15;
            while (value < line_end and c.isspace(header[value]) != 0) value += 1;
            return c.strtol(header.ptr + value, null, 10);
        }
        start = if (end < header.len) end + 1 else end;
    }
    return 0;
}

/// Read exactly one supported request and discard any over-read pipeline data.
/// The last receive may cross the header limit if it also finds a terminator;
/// this is the existing server's limit behavior, not a stricter HTTP parser.
/// On failure only temporary storage is released; method/path may be populated.
pub fn readRequest(descriptor: c_int, request: *Request) bool {
    var received: std.ArrayList(u8) = .empty;
    defer received.deinit(allocator);
    var header_end: ?usize = null;
    while (header_end == null and received.items.len < max_header_bytes) {
        var chunk: [4096]u8 = undefined;
        const count = c.recv(descriptor, &chunk, chunk.len, 0);
        if (count < 0 and errnoValue() == c.EINTR) continue;
        if (count <= 0) return false;
        append(&received, chunk[0..@intCast(count)]);
        terminate(&received);
        header_end = headerEnd(received.items);
    }
    const body_start = header_end orelse return false;

    // sscanf's bounded tokens and ignored HTTP version are part of the current
    // accepted request grammar, including a truncated 511-byte request line.
    var line: [512]u8 = undefined;
    var line_length: usize = 0;
    while (line_length < received.items.len and received.items[line_length] != '\n' and line_length + 1 < line.len) : (line_length += 1) {
        line[line_length] = received.items[line_length];
    }
    line[line_length] = 0;
    if (c.sscanf(&line, "%7s %255s", &request.method, &request.path) != 2) return false;
    const path = text(&request.path);
    if (std.mem.indexOfScalar(u8, path, '?')) |query| {
        // Conversation item pagination consumes the query. Existing other
        // routes retain their path-only behavior (including prefix matching).
        if (!std.mem.startsWith(u8, path, "/v1/conversations")) request.path[query] = 0;
    }

    const content_length = contentLength(received.items[0..body_start]);
    if (content_length < 0 or content_length > max_body_bytes) return false;
    const body_length: usize = @intCast(content_length);
    while (received.items.len < body_start + body_length) {
        var chunk: [8192]u8 = undefined;
        const count = c.recv(descriptor, &chunk, chunk.len, 0);
        if (count < 0 and errnoValue() == c.EINTR) continue;
        if (count <= 0) return false;
        append(&received, chunk[0..@intCast(count)]);
    }
    request.body_len = body_length;
    const body: [*]u8 = @ptrCast(c.malloc(body_length + 1) orelse fatal("out of memory"));
    @memcpy(body[0..body_length], received.items[body_start..][0..body_length]);
    body[body_length] = 0;
    request.body = body;
    request.headers = (allocator.dupeSentinel(u8, received.items[0..body_start], 0) catch fatal("out of memory")).ptr;
    request.headers_len = body_start;
    const remaining = received.items[body_start + body_length ..];
    if (remaining.len != 0) {
        request.prefetched = (allocator.dupeSentinel(u8, remaining, 0) catch fatal("out of memory")).ptr;
        request.prefetched_len = remaining.len;
    }
    return true;
}

pub fn freeRequest(request: *Request) void {
    c.free(request.body);
    c.free(request.headers);
    c.free(request.prefetched);
    request.* = std.mem.zeroes(Request);
}
pub fn pollDisconnected(revents: c_short) bool {
    return revents & (c.POLLERR | c.POLLHUP | c.POLLNVAL) != 0;
}
pub fn recvDisconnected(failure: c_int) bool {
    return failure != c.EINTR and failure != c.EAGAIN and failure != c.EWOULDBLOCK;
}

/// The request body has been consumed and responses use Connection: close, so
/// EOF means cancellation. Drain unsupported pipeline bytes nonblockingly;
/// otherwise readable bytes could hide the peer's FIN indefinitely.
pub fn disconnected(descriptor: c_int) bool {
    var probe = std.mem.zeroes(c.struct_pollfd);
    probe.fd = descriptor;
    probe.events = c.POLLIN;
    const result = while (true) {
        const polled = c.poll(&probe, 1, 0);
        if (polled < 0 and errnoValue() == c.EINTR) continue;
        break polled;
    };
    if (result < 0) return recvDisconnected(errnoValue());
    if (result == 0) return false;
    if (pollDisconnected(probe.revents)) return true;
    if (probe.revents & c.POLLIN == 0) return false;
    var discard: [256]u8 = undefined;
    while (true) {
        const count = c.recv(descriptor, &discard, discard.len, 0);
        if (count > 0) continue;
        if (count == 0) return true;
        const failure = errnoValue();
        if (failure == c.EINTR) continue;
        return recvDisconnected(failure);
    }
}

pub fn listen(host: [:0]const u8, port: c_int) c_int {
    const descriptor = c.socket(c.AF_INET, c.SOCK_STREAM, 0);
    if (descriptor < 0) return -1;
    const yes: c_int = 1;
    _ = c.setsockopt(descriptor, c.SOL_SOCKET, c.SO_REUSEADDR, &yes, @sizeOf(c_int));
    var address = std.mem.zeroes(c.struct_sockaddr_in);
    address.sin_family = c.AF_INET;
    // Original C narrows to uint16_t before htons; option policy lives above us.
    address.sin_port = c.htons(@as(u16, @truncate(@as(c_uint, @bitCast(port)))));
    const literal: [:0]const u8 = if (std.mem.eql(u8, host, "localhost")) "127.0.0.1" else host;
    if (c.inet_pton(c.AF_INET, literal.ptr, &address.sin_addr) != 1) {
        _ = c.close(descriptor);
        c.__error().* = c.EINVAL;
        return -1;
    }
    if (c.bind(descriptor, @ptrCast(&address), @sizeOf(c.struct_sockaddr_in)) != 0) {
        _ = c.close(descriptor);
        return -1;
    }
    if (c.listen(descriptor, 128) != 0) {
        _ = c.close(descriptor);
        return -1;
    }
    return descriptor;
}
pub fn configure(descriptor: c_int) void {
    var timeout = std.mem.zeroes(c.struct_timeval);
    timeout.tv_sec = io_timeout_seconds;
    _ = c.setsockopt(descriptor, c.SOL_SOCKET, c.SO_RCVTIMEO, &timeout, @sizeOf(c.struct_timeval));
    _ = c.setsockopt(descriptor, c.SOL_SOCKET, c.SO_SNDTIMEO, &timeout, @sizeOf(c.struct_timeval));
}
pub fn nonblocking(descriptor: c_int) void {
    // The inference worker writes streams itself. A blocked socket would hold
    // every other session, so fail stalled clients rather than back-pressure it.
    const flags = c.fcntl(descriptor, c.F_GETFL, @as(c_int, 0));
    if (flags >= 0) _ = c.fcntl(descriptor, c.F_SETFL, flags | c.O_NONBLOCK);
}

// Narrow C facade: no endpoint policy, scheduler, engine or session imports.
export fn ds4_server_http_send_all(fd: c_int, bytes: ?*const anyopaque, length: usize, stop: *const volatile c.sig_atomic_t) bool {
    const payload: []const u8 = if (length == 0) "" else @as([*]const u8, @ptrCast(bytes.?))[0..length];
    return (Connection{ .descriptor = fd, .stop_requested = stop }).sendAll(payload);
}
export fn ds4_server_http_response(fd: c_int, cors: bool, status: c_int, kind: [*c]const u8, body: [*c]const u8, stop: *const volatile c.sig_atomic_t) bool {
    return (Connection{ .descriptor = fd, .stop_requested = stop }).respond(cors, status, if (kind == null) null else text(kind), if (body == null) "" else text(body));
}
export fn ds4_server_http_sse_headers(fd: c_int, cors: bool, stop: *const volatile c.sig_atomic_t) bool {
    return (Connection{ .descriptor = fd, .stop_requested = stop }).sseHeaders(cors);
}
export fn ds4_server_http_sse_event(fd: c_int, event: [*c]const u8, data: [*c]const u8, stop: *const volatile c.sig_atomic_t) bool {
    return (Connection{ .descriptor = fd, .stop_requested = stop }).sseEvent(text(event), text(data));
}
export fn ds4_server_http_read_request(fd: c_int, request: *Request) bool {
    return readRequest(fd, request);
}
export fn ds4_server_http_request_free(request: *Request) void {
    freeRequest(request);
}
export fn ds4_server_http_poll_disconnected(revents: c_short) bool {
    return pollDisconnected(revents);
}
export fn ds4_server_http_recv_disconnected(failure: c_int) bool {
    return recvDisconnected(failure);
}
export fn ds4_server_http_disconnected(fd: c_int) bool {
    return disconnected(fd);
}
export fn ds4_server_http_listen(host: [*c]const u8, port: c_int) c_int {
    return listen(std.mem.span(host), port);
}
export fn ds4_server_http_configure(fd: c_int) void {
    configure(fd);
}
export fn ds4_server_http_nonblocking(fd: c_int) void {
    nonblocking(fd);
}
