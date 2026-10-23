//! Sequential OpenResponses turns over one socket. Other sockets share the
//! ordinary server queue/slots. Only transport and connection-local response
//! history live here; preparation, inference and durable publication are shared.
const std = @import("std");
const c = @import("server_serving_c");
const net = @import("server_http_c");
const Frames = @import("websocket_frames.zig");
const Serving = @import("serving.zig");
const allocator = std.heap.c_allocator;
const Value = std.json.Value;
const connection_ms = 60 * 60 * 1000;

fn text(value: [*c]const u8) []const u8 {
    return if (value == null) "" else std.mem.span(value);
}
fn string(value: ?Value) ?[]const u8 {
    return if (value) |item| (if (item == .string) item.string else null) else null;
}
fn duplicate(value: []const u8) [:0]u8 {
    return allocator.dupeSentinel(u8, value, 0) catch fatal();
}
fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}

const Parent = struct {
    id: []u8,
    response: []u8,
    replay: []u8,
    fn deinit(self: Parent) void {
        allocator.free(self.id);
        allocator.free(self.response);
        allocator.free(self.replay);
    }
};

const Session = struct {
    runtime: *c.server,
    connection: Frames.Connection,
    reader: Frames.Reader,
    mutex: c.pthread_mutex_t = undefined,
    sink: c.server_response_sink = undefined,
    closing: bool = false,
    terminal: ?[]u8 = null,
    failed: bool = false,
    parent: ?Parent = null,
    active: ?*c.job = null,
    pending: ?[]u8 = null,

    fn lock(self: *Session) void {
        _ = c.pthread_mutex_lock(&self.mutex);
    }
    fn unlock(self: *Session) void {
        _ = c.pthread_mutex_unlock(&self.mutex);
    }
    fn emit(context: ?*anyopaque, raw: [*c]const u8, length: usize) callconv(.c) bool {
        const self: *Session = @ptrCast(@alignCast(context.?));
        self.lock();
        defer self.unlock();
        if (self.closing) return false;
        const bytes = raw[0..length];
        var parsed = std.json.parseFromSlice(Value, allocator, bytes, .{}) catch return false;
        defer parsed.deinit();
        if (parsed.value != .object) return false;
        const kind = string(parsed.value.object.get("type")) orelse return false;
        if (std.mem.eql(u8, kind, "response.completed")) {
            if (parsed.value.object.get("response")) |response| {
                if (self.terminal) |old| allocator.free(old);
                self.terminal = std.json.Stringify.valueAlloc(allocator, response, .{}) catch fatal();
            }
        } else if (std.mem.eql(u8, kind, "error") or std.mem.eql(u8, kind, "response.failed")) {
            self.failed = true;
        }
        if (!self.connection.sendText(bytes)) {
            self.closing = true;
            return false;
        }
        return true;
    }

    fn load(context: ?*anyopaque, id: [*c]const u8, response: [*c][*c]u8, replay: [*c][*c]u8) callconv(.c) c_int {
        const self: *Session = @ptrCast(@alignCast(context.?));
        const parent = self.parent orelse return 1;
        if (!std.mem.eql(u8, text(id), parent.id)) return 1;
        response.* = duplicate(parent.response).ptr;
        replay.* = duplicate(parent.replay).ptr;
        return 0;
    }

    fn evict(self: *Session, id: ?[]const u8) void {
        if (id) |wanted| if (self.parent) |parent| {
            if (std.mem.eql(u8, wanted, parent.id)) {
                parent.deinit();
                self.parent = null;
            }
        };
    }

    fn sendError(self: *Session, status: c_int, code: []const u8, message: []const u8, parameter: ?[]const u8) void {
        const body = std.json.Stringify.valueAlloc(allocator, .{
            .type = "error",
            .status = status,
            .@"error" = .{
                .type = if (status >= 500) "server_error" else "invalid_request_error",
                .code = code,
                .message = message,
                .param = parameter,
            },
        }, .{}) catch fatal();
        defer allocator.free(body);
        self.lock();
        defer self.unlock();
        if (!self.closing and !self.connection.sendText(body)) self.closing = true;
    }

    fn close(self: *Session, code: u16, reason: []const u8) void {
        self.lock();
        defer self.unlock();
        if (!self.closing) {
            self.closing = true;
            _ = self.connection.sendClose(code, reason);
        }
    }

    fn done(self: *Session) bool {
        const job = self.active orelse return true;
        _ = c.pthread_mutex_lock(&job.mu);
        defer _ = c.pthread_mutex_unlock(&job.mu);
        return job.done;
    }

    /// Completion retires every callback before request/connection memory can
    /// be released. Capture hidden logical replay before freeing the request.
    fn collect(self: *Session) void {
        const job = self.active orelse return;
        std.debug.assert(self.done());
        self.lock();
        const terminal = self.terminal;
        self.terminal = null;
        const failed = self.failed;
        self.failed = false;
        self.unlock();
        defer if (terminal) |value| allocator.free(value);
        if (failed) self.evict(if (job.req.previous_response_id == null) null else text(job.req.previous_response_id));
        if (terminal != null and !failed and job.req.responses_replay != null and job.req.image_count == 0) {
            var parsed = std.json.parseFromSlice(Value, allocator, terminal.?, .{}) catch fatal();
            defer parsed.deinit();
            const id = string(parsed.value.object.get("id")) orelse unreachable;
            if (self.parent) |old| old.deinit();
            self.parent = .{
                .id = allocator.dupe(u8, id) catch fatal(),
                .response = allocator.dupe(u8, terminal.?) catch fatal(),
                .replay = allocator.dupe(u8, text(job.req.responses_replay)) catch fatal(),
            };
        }
        _ = c.pthread_cond_destroy(&job.cv);
        _ = c.pthread_mutex_destroy(&job.mu);
        c.ds4_server_input_request_free(&job.req);
        allocator.destroy(job);
        self.active = null;
    }

    fn reject(self: *Session, parent: ?[]const u8, code: []const u8, message: []const u8, parameter: ?[]const u8) void {
        self.evict(parent);
        self.sendError(400, code, message, parameter);
    }

    fn queueFull(self: *Session, raw: []const u8) void {
        if (std.json.parseFromSlice(Value, allocator, raw, .{})) |parsed| {
            defer parsed.deinit();
            if (parsed.value == .object) self.evict(string(parsed.value.object.get("previous_response_id")));
        } else |_| {}
        self.sendError(429, "too_many_requests", "Only one queued WebSocket turn is supported", null);
    }

    fn create(self: *Session, raw: []const u8) void {
        var parsed = std.json.parseFromSlice(Value, allocator, raw, .{ .duplicate_field_behavior = .use_last }) catch {
            self.sendError(400, "invalid_json", "WebSocket messages must be JSON objects", null);
            return;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            self.sendError(400, "invalid_request", "WebSocket messages must be JSON objects", null);
            return;
        }
        const object = &parsed.value.object;
        const parent = string(object.get("previous_response_id"));
        const kind = string(object.get("type")) orelse "";
        if (!std.mem.eql(u8, kind, "response.create")) {
            self.reject(parent, "unsupported_event_type", "Only response.create is supported", "type");
            return;
        }
        for ([_][]const u8{ "stream", "stream_options", "background", "stream_id", "generate" }) |field| {
            if (object.contains(field)) {
                self.reject(parent, "unsupported_parameter", "This parameter is not supported by the OpenResponses WebSocket transport", field);
                return;
            }
        }
        _ = object.swapRemove("type");
        object.put(parsed.arena.allocator(), "stream", .{ .bool = true }) catch fatal();
        const encoded = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch fatal();
        defer allocator.free(encoded);
        const body = duplicate(encoded);
        var req = std.mem.zeroes(c.request);
        var errors: [256]u8 = @splat(0);
        if (!Serving.decodeWithSink(self.runtime.engine, self.runtime, c.DS4_SERVER_INPUT_RESPONSES, body.ptr, self.runtime.default_tokens, self.runtime.ctx_size, &self.sink, &req, &errors)) {
            allocator.free(body);
            const missing = std.mem.find(u8, text(&errors), "previous_response_id was not found") != null;
            self.reject(parent, if (missing) "previous_response_not_found" else "invalid_request", text(&errors), if (missing) "previous_response_id" else null);
            return;
        }
        req.raw_body = body.ptr;
        if (!req.model_from_request) {
            c.free(req.model);
            req.model = duplicate(std.mem.span(Serving.modelId(self.runtime.engine))).ptr;
        }
        if (Serving.exceedsContext(&req, self.runtime.ctx_size)) {
            c.ds4_server_input_request_free(&req);
            self.reject(parent, "context_length_exceeded", "The request exceeds the configured model context", "input");
            return;
        }
        const job = allocator.create(c.job) catch fatal();
        job.* = std.mem.zeroes(c.job);
        job.fd = self.connection.descriptor;
        job.req = req;
        _ = c.pthread_mutex_init(&job.mu, null);
        _ = c.pthread_cond_init(&job.cv, null);
        self.active = job;
        if (!Serving.enqueue(self.runtime, job)) {
            job.done = true; // No worker acquired this request.
            self.sendError(503, "server_unavailable", "Server is shutting down", null);
            self.evict(parent);
            self.collect();
        }
    }

    fn deinit(self: *Session) void {
        self.close(1001, "connection closing");
        if (self.active) |job| {
            Serving.cancel(self.runtime, job);
            Serving.waitForJob(self.runtime, job);
            self.collect();
        }
        if (self.parent) |parent| parent.deinit();
        if (self.pending) |pending| allocator.free(pending);
        if (self.terminal) |terminal| allocator.free(terminal);
        self.reader.deinit();
        _ = c.pthread_mutex_destroy(&self.mutex);
    }

    fn run(self: *Session) void {
        while (self.connection.stop_requested.* == 0) {
            if (self.active != null and self.done()) self.collect();
            if (self.active == null) if (self.pending) |pending| {
                self.pending = null;
                self.create(pending);
                allocator.free(pending);
            };
            self.lock();
            const closing = self.closing;
            self.unlock();
            if (closing) return;
            if (Frames.milliseconds() >= self.connection.expires_at_ms.?) {
                self.sendError(400, "websocket_connection_limit_reached", "WebSocket connection limit reached (60 minutes)", null);
                return;
            }
            if (self.reader.initial_pos == self.reader.initial.len) {
                var descriptor = std.mem.zeroes(net.struct_pollfd);
                descriptor.fd = self.connection.descriptor;
                descriptor.events = net.POLLIN;
                const ready = net.poll(&descriptor, 1, 50);
                if (ready < 0) {
                    if (net.__error().* == net.EINTR) continue;
                    return;
                }
                if (ready == 0) continue;
                if ((descriptor.revents & (net.POLLERR | net.POLLNVAL)) != 0) return;
            }
            const incoming = self.reader.read(self.connection) catch |err| {
                switch (err) {
                    error.Expired => self.sendError(400, "websocket_connection_limit_reached", "WebSocket connection limit reached (60 minutes)", null),
                    error.InvalidUtf8 => self.close(1007, "invalid UTF-8"),
                    error.MessageTooLarge => self.close(1009, "message too large"),
                    error.Protocol => self.close(1002, "invalid WebSocket frame"),
                    else => {},
                }
                return;
            };
            defer incoming.deinit(allocator);
            switch (incoming) {
                .text => |message| {
                    // A terminal event can reach the client just before job
                    // retirement. Queue one next turn across that boundary.
                    if (self.active != null and self.done()) self.collect();
                    if (self.active == null) self.create(message) else if (self.pending == null) {
                        self.pending = allocator.dupe(u8, message) catch fatal();
                    } else self.queueFull(message);
                },
                .ping => |payload| {
                    self.lock();
                    if (!self.closing and !self.connection.sendPong(payload)) self.closing = true;
                    self.unlock();
                },
                .pong => {},
                .binary => {
                    self.close(1003, "Responses requires JSON text messages");
                    return;
                },
                .close => |payload| {
                    self.lock();
                    if (!self.closing) {
                        self.closing = true;
                        _ = self.connection.send(.close, payload);
                    }
                    self.unlock();
                    return;
                },
            }
        }
    }
};

pub fn serve(runtime: *c.server, descriptor: c_int, request: *const c.ds4_server_http_request, stop: *const volatile c.sig_atomic_t) void {
    const headers = if (request.headers == null) "" else request.headers[0..request.headers_len];
    const upgrade = Frames.headerValue(headers, "Upgrade") orelse "";
    const connection = Frames.headerValue(headers, "Connection") orelse "";
    const version = Frames.headerValue(headers, "Sec-WebSocket-Version") orelse "";
    const key = Frames.headerValue(headers, "Sec-WebSocket-Key") orelse "";
    const valid_key = if (Frames.acceptKey(key)) |_| true else |_| false;
    if (!Frames.hasToken(upgrade, "websocket") or !Frames.hasToken(connection, "Upgrade") or !std.mem.eql(u8, version, "13") or !valid_key or request.body_len != 0) {
        _ = c.ds4_server_http_response(descriptor, runtime.enable_cors, 400, "application/json", "{\"error\":{\"message\":\"invalid WebSocket upgrade\",\"type\":\"invalid_request_error\"}}", stop);
        return;
    }
    c.ds4_server_http_nonblocking(descriptor);
    var session: Session = .{
        .runtime = runtime,
        .connection = .{ .descriptor = descriptor, .stop_requested = stop, .expires_at_ms = Frames.milliseconds() + connection_ms },
        .reader = Frames.Reader.init(allocator),
    };
    _ = c.pthread_mutex_init(&session.mutex, null);
    session.sink = .{ .context = &session, .emit = Session.emit, .load_parent = Session.load };
    defer session.deinit();
    if (!session.connection.accept(key)) return;
    if (request.prefetched != null) session.reader.feed(request.prefetched[0..request.prefetched_len]);
    session.run();
}
