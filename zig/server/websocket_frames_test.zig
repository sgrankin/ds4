//! Small server framing fixture: actual socket I/O, fragmented UTF-8 with a
//! control frame in flight, and the protocol boundaries used by Responses.
const std = @import("std");
const c = @import("server_http_c");
const Ws = @import("websocket_frames.zig");
const allocator = std.heap.page_allocator;

fn require(value: bool) !void {
    if (!value) return error.CheckFailed;
}
fn equal(expected: []const u8, actual: []const u8) !void {
    if (!std.mem.eql(u8, expected, actual)) {
        std.debug.print("expected: {s}\nactual: {s}\n", .{ expected, actual });
        return error.BytesChanged;
    }
}

const Pair = struct {
    sockets: [2]c_int,
    stop: c.sig_atomic_t = 0,

    fn init() !Pair {
        var value: Pair = .{ .sockets = undefined };
        if (c.socketpair(c.AF_UNIX, c.SOCK_STREAM, 0, &value.sockets) != 0) return error.SocketPairFailed;
        return value;
    }
    fn deinit(self: *Pair) void {
        for (self.sockets) |fd| _ = c.close(fd);
    }
    fn connection(self: *Pair) Ws.Connection {
        return .{ .descriptor = self.sockets[0], .stop_requested = &self.stop };
    }
    fn send(self: *Pair, opcode: u8, final: bool, text: []const u8) !void {
        try require(text.len < 126);
        const mask = [_]u8{ 0x37, 0xfa, 0x21, 0x3d };
        var bytes: [131]u8 = undefined;
        bytes[0] = opcode | (if (final) @as(u8, 128) else 0);
        bytes[1] = @as(u8, @intCast(text.len)) | 128;
        @memcpy(bytes[2..6], &mask);
        for (text, 0..) |byte, position| bytes[position + 6] = byte ^ mask[position & 3];
        try require(c.send(self.sockets[1], &bytes, text.len + 6, 0) == text.len + 6);
    }
};

fn handshakeAndHeaders() !void {
    const accepted = try Ws.acceptKey("dGhlIHNhbXBsZSBub25jZQ==");
    try equal("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &accepted);
    const header = "GET /v1/responses HTTP/1.1\r\nConnection: keep-alive, Upgrade\r\nupgrade: websocket\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    try require(Ws.hasToken(Ws.headerValue(header, "Connection").?, "upgrade"));
    try equal("websocket", Ws.headerValue(header, "Upgrade").?);
    try require(!Ws.hasToken("not-upgrade", "upgrade"));
    if (Ws.acceptKey("not a key")) |_| return error.AcceptedBadKey else |failure| try require(failure == error.BadHandshake);
    var pair = try Pair.init();
    defer pair.deinit();
    try require(pair.connection().accept("dGhlIHNhbXBsZSBub25jZQ=="));
    var bytes: [256]u8 = undefined;
    const count = c.recv(pair.sockets[1], &bytes, bytes.len, 0);
    try require(count > 0);
    try equal("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n", bytes[0..@intCast(count)]);
}

fn fragmentedTextAndPing() !void {
    var pair = try Pair.init();
    defer pair.deinit();
    // Split a Unicode scalar over fragments; UTF-8 is checked on the complete
    // message, while ping remains visible to the caller between fragments.
    try pair.send(1, false, "hello \xe2");
    try pair.send(9, true, "probe");
    try pair.send(0, true, "\x82\xac");
    var reader = Ws.Reader.init(allocator);
    defer reader.deinit();
    const ping = try reader.read(pair.connection());
    defer ping.deinit(allocator);
    switch (ping) {
        .ping => |bytes| try equal("probe", bytes),
        else => return error.ExpectedPing,
    }
    try require(pair.connection().sendPong(ping.payload()));
    var pong: [16]u8 = undefined;
    const count = c.recv(pair.sockets[1], &pong, pong.len, 0);
    try require(count == 7);
    try equal("\x8a\x05probe", pong[0..7]);
    const text = try reader.read(pair.connection());
    defer text.deinit(allocator);
    switch (text) {
        .text => |bytes| try equal("hello €", bytes),
        else => return error.ExpectedText,
    }
}

fn prefixAndFrameLengths() !void {
    var pair = try Pair.init();
    defer pair.deinit();
    var reader = Ws.Reader.init(allocator);
    defer reader.deinit();
    // Header reads may have prefetched a complete next frame.
    reader.feed("\x81\x82\x01\x02\x03\x04ik");
    const message = try reader.read(pair.connection());
    defer message.deinit(allocator);
    try equal("hi", message.payload());
    const short = Ws.frameHeader(.text, 125);
    try equal("\x81\x7d", short.bytes[0..short.len]);
    const medium = Ws.frameHeader(.text, 126);
    try equal("\x81\x7e\x00\x7e", medium.bytes[0..medium.len]);
    const long = Ws.frameHeader(.text, 65536);
    try equal("\x81\x7f\x00\x00\x00\x00\x00\x01\x00\x00", long.bytes[0..long.len]);
}

fn rejectsProtocolViolationsAndStops() !void {
    var pair = try Pair.init();
    defer pair.deinit();
    var unmasked = Ws.Reader.init(allocator);
    defer unmasked.deinit();
    unmasked.feed("\x81\x02hi");
    if (unmasked.read(pair.connection())) |_| return error.AcceptedUnmaskedFrame else |failure| try require(failure == error.Protocol);
    var invalid_utf8 = Ws.Reader.init(allocator);
    defer invalid_utf8.deinit();
    invalid_utf8.feed("\x81\x81\x00\x00\x00\x00\xff");
    if (invalid_utf8.read(pair.connection())) |_| return error.AcceptedInvalidUtf8 else |failure| try require(failure == error.InvalidUtf8);
    var oversized = Ws.Reader.init(allocator);
    defer oversized.deinit();
    oversized.feed("\x81\xff\x00\x00\x00\x00\x04\x00\x00\x01");
    if (oversized.read(pair.connection())) |_| return error.AcceptedOversizedMessage else |failure| try require(failure == error.MessageTooLarge);
    var stopped = Ws.Reader.init(allocator);
    defer stopped.deinit();
    pair.stop = 1;
    if (stopped.read(pair.connection())) |_| return error.IgnoredStop else |failure| try require(failure == error.Stopped);
    pair.stop = 0;
    var expired = pair.connection();
    expired.expires_at_ms = Ws.milliseconds() - 1;
    if (stopped.read(expired)) |_| return error.IgnoredExpiry else |failure| try require(failure == error.Expired);
}

fn closeAndServerText() !void {
    var pair = try Pair.init();
    defer pair.deinit();
    try pair.send(8, true, "\x03\xe8done");
    var reader = Ws.Reader.init(allocator);
    defer reader.deinit();
    const closed = try reader.read(pair.connection());
    defer closed.deinit(allocator);
    switch (closed) {
        .close => |bytes| try equal("\x03\xe8done", bytes),
        else => return error.ExpectedClose,
    }
    try require(pair.connection().sendText("{}"));
    var text: [16]u8 = undefined;
    const count = c.recv(pair.sockets[1], &text, text.len, 0);
    try require(count == 4);
    try equal("\x81\x02{}", text[0..4]);
}

pub fn main() !void {
    // Supply the existing send primitive once for this standalone fixture.
    _ = @import("http.zig");
    try handshakeAndHeaders();
    try fragmentedTextAndPing();
    try prefixAndFrameLengths();
    try rejectsProtocolViolationsAndStops();
    try closeAndServerText();
    std.debug.print("PASS: server WebSocket framing (5 cases)\n", .{});
}
