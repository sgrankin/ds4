//! One bounded JavaScript program over request-owned inputs and native tools.
//! QuickJS has no libc module/loader here. Host decisions run synchronously;
//! JavaScript await works, while Promise.all does not imply parallel inference.
const std = @import("std");
const c = @import("quickjs_c");
const allocator = std.heap.c_allocator;
// Translate-C cannot initialize QuickJS's union-valued compound-literal
// macros. These use the generated ABI type and its upstream tag constants.
const js_undefined: c.JSValue = .{ .u = .{ .float64 = 0 }, .tag = c.JS_TAG_UNDEFINED };
const js_exception: c.JSValue = .{ .u = .{ .float64 = 0 }, .tag = c.JS_TAG_EXCEPTION };

pub const Host = struct {
    context: ?*anyopaque,
    call: *const fn (?*anyopaque, []const u8) anyerror![:0]u8,
    cancelled: ?*const fn (?*anyopaque) bool = null,
};

pub const Limits = struct {
    memory_bytes: usize = 64 * 1024 * 1024,
    timeout_ms: u64 = 10_000,
    max_output_bytes: usize = 65_536,
};

pub const Status = enum { completed, failed, cancelled };
pub const Result = struct {
    status: Status,
    output: [:0]u8,

    pub fn deinit(self: *Result) void {
        allocator.free(self.output);
    }
};

/// Shared with the native host callback so helper work uses the same deadline.
pub fn nowMillis() u64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(u64, @intCast(stamp.tv_sec)) * 1000 + @as(u64, @intCast(stamp.tv_nsec)) / 1_000_000;
}

const Halt = enum { cancelled, timeout, output_limit };
const State = struct {
    host: Host,
    limits: Limits,
    deadline: u64,
    halt: ?Halt = null,
    emitted: std.ArrayList(u8) = .empty,
    count: usize = 0,

    fn stopped(self: *State) bool {
        if (self.halt != null) return true;
        if (self.host.cancelled) |cancelled| if (cancelled(self.host.context)) {
            self.halt = .cancelled;
            return true;
        };
        if (nowMillis() >= self.deadline) {
            self.halt = .timeout;
            return true;
        }
        return false;
    }

    fn finish(self: *State, code: ?[]const u8, message: []const u8) !Result {
        var final_code = code;
        var final_message = message;
        var status: Status = if (code == null) .completed else .failed;
        if (self.halt) |halt| {
            status = if (halt == .cancelled) .cancelled else .failed;
            final_code = @tagName(halt);
            final_message = switch (halt) {
                .cancelled => "Program cancelled",
                .timeout => "Program deadline exceeded",
                .output_limit => "Program output limit exceeded",
            };
        }
        const short = utf8Prefix(final_message, @min(@as(usize, 256), (self.limits.max_output_bytes - 128) / 6));
        const detail = if (final_code) |value|
            try std.json.Stringify.valueAlloc(allocator, .{ .code = value, .message = short }, .{})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(detail);
        var output = try std.fmt.allocPrintSentinel(allocator, "{{\"output\":[{s}],\"error\":{s}}}", .{ self.emitted.items, detail }, 0);
        // An error must still fit when earlier successful text filled the cap.
        if (output.len > self.limits.max_output_bytes) {
            allocator.free(output);
            output = try std.fmt.allocPrintSentinel(allocator, "{{\"output\":[],\"error\":{s}}}", .{detail}, 0);
        }
        return .{ .status = status, .output = output };
    }

    fn failure(self: *State, ctx: ?*c.JSContext, value: c.JSValue) !Result {
        if (self.stopped()) return self.finish(null, "");
        var length: usize = 0;
        const message = c.JS_ToCStringLen2(ctx, &length, value, false);
        if (message == null) return self.finish("javascript_error", "JavaScript execution failed (possibly out of memory)");
        defer c.JS_FreeCString(ctx, message);
        return self.finish("javascript_error", message[0..length]);
    }

    fn exception(self: *State, ctx: ?*c.JSContext) !Result {
        const value = c.JS_GetException(ctx);
        defer c.JS_FreeValue(ctx, value);
        return self.failure(ctx, value);
    }
};

fn utf8Prefix(value: []const u8, cap: usize) []const u8 {
    if (value.len <= cap) return value;
    var end = cap;
    while (end != 0 and (value[end] & 0xc0) == 0x80) end -= 1;
    return value[0..end];
}

fn stateOf(ctx: ?*c.JSContext) *State {
    return @ptrCast(@alignCast(c.JS_GetContextOpaque(ctx).?));
}

fn interrupted(_: ?*c.JSRuntime, context: ?*anyopaque) callconv(.c) c_int {
    const state: *State = @ptrCast(@alignCast(context.?));
    return @intFromBool(state.stopped());
}

fn stoppedException(ctx: ?*c.JSContext) c.JSValue {
    return c.JS_ThrowInternalError(ctx, "Program interrupted");
}

fn emit(ctx: ?*c.JSContext, _: c.JSValue, argc: c_int, argv: [*c]c.JSValue) callconv(.c) c.JSValue {
    const state = stateOf(ctx);
    if (state.stopped()) return stoppedException(ctx);
    if (argc != 1) return c.JS_ThrowTypeError(ctx, "text expects one JSON value");
    const encoded = c.JS_JSONStringify(ctx, argv[0], js_undefined, js_undefined);
    defer c.JS_FreeValue(ctx, encoded);
    if (c.JS_IsException(encoded)) return js_exception;
    if (c.JS_IsUndefined(encoded)) return c.JS_ThrowTypeError(ctx, "text expects a JSON-serializable value");
    var length: usize = 0;
    const raw = c.JS_ToCStringLen2(ctx, &length, encoded, false);
    if (raw == null) return js_exception;
    defer c.JS_FreeCString(ctx, raw);
    if (state.stopped()) return stoppedException(ctx);
    const extra = length + @intFromBool(state.count != 0);
    if (extra > state.limits.max_output_bytes - 32 or state.emitted.items.len > state.limits.max_output_bytes - 32 - extra) {
        state.halt = .output_limit;
        return c.JS_ThrowRangeError(ctx, "Program output limit exceeded");
    }
    if (state.count != 0) state.emitted.append(allocator, ',') catch return c.JS_ThrowOutOfMemory(ctx);
    state.emitted.appendSlice(allocator, raw[0..length]) catch return c.JS_ThrowOutOfMemory(ctx);
    state.count += 1;
    return js_undefined;
}

fn decide(ctx: ?*c.JSContext, _: c.JSValue, argc: c_int, argv: [*c]c.JSValue) callconv(.c) c.JSValue {
    const state = stateOf(ctx);
    if (state.stopped()) return stoppedException(ctx);
    if (argc != 1) return c.JS_ThrowTypeError(ctx, "tools.ds4_decide expects one argument object");
    const encoded = c.JS_JSONStringify(ctx, argv[0], js_undefined, js_undefined);
    defer c.JS_FreeValue(ctx, encoded);
    if (c.JS_IsException(encoded)) return js_exception;
    if (c.JS_IsUndefined(encoded)) return c.JS_ThrowTypeError(ctx, "tools.ds4_decide expects a JSON argument object");
    var length: usize = 0;
    const raw = c.JS_ToCStringLen2(ctx, &length, encoded, false);
    if (raw == null) return js_exception;
    defer c.JS_FreeCString(ctx, raw);
    if (state.stopped()) return stoppedException(ctx);
    const output = state.host.call(state.host.context, raw[0..length]) catch |err| {
        if (state.stopped()) return stoppedException(ctx);
        return c.JS_ThrowInternalError(ctx, "Native decision failed: %s", @as([*:0]const u8, @errorName(err)));
    };
    defer allocator.free(output);
    if (state.stopped()) return stoppedException(ctx);
    const result = c.JS_ParseJSON(ctx, output.ptr, output.len, "<decision-result>");
    if (c.JS_IsException(result)) return js_exception;
    const native_error = c.JS_GetPropertyStr(ctx, result, "error");
    defer c.JS_FreeValue(ctx, native_error);
    if (c.JS_IsException(native_error)) {
        c.JS_FreeValue(ctx, result);
        return js_exception;
    }
    if (!c.JS_IsUndefined(native_error) and !c.JS_IsNull(native_error)) {
        defer c.JS_FreeValue(ctx, result);
        const message = c.JS_GetPropertyStr(ctx, native_error, "message");
        defer c.JS_FreeValue(ctx, message);
        if (c.JS_IsException(message)) return js_exception;
        const detail = if (c.JS_IsString(message)) c.JS_DupValue(ctx, message) else c.JS_JSONStringify(ctx, native_error, js_undefined, js_undefined);
        if (c.JS_IsException(detail)) return js_exception;
        const exception = c.JS_NewError(ctx);
        if (c.JS_IsException(exception)) {
            c.JS_FreeValue(ctx, detail);
            return js_exception;
        }
        if (!set(ctx, exception, "message", detail)) {
            c.JS_FreeValue(ctx, exception);
            return js_exception;
        }
        return c.JS_Throw(ctx, exception);
    }
    return result;
}

fn set(ctx: ?*c.JSContext, object: c.JSValue, name: [*:0]const u8, value: c.JSValue) bool {
    if (c.JS_IsException(value)) return false;
    return c.JS_SetPropertyStr(ctx, object, name, value) >= 0;
}

/// The only returned data is captured by text(value); a program's ordinary
/// return value is ignored. The Result and Host.call results are libc-owned.
pub fn run(code: []const u8, bindings_json: []const u8, host: Host, limits: Limits) !Result {
    if (limits.memory_bytes == 0 or limits.timeout_ms == 0 or limits.max_output_bytes < 128) return error.InvalidLimits;
    var state = State{ .host = host, .limits = limits, .deadline = nowMillis() +| limits.timeout_ms };
    defer state.emitted.deinit(allocator);
    if (state.stopped()) return state.finish(null, "");
    const runtime = c.JS_NewRuntime() orelse return state.finish("memory_limit", "Could not create JavaScript runtime");
    defer c.JS_FreeRuntime(runtime);
    c.JS_SetMemoryLimit(runtime, limits.memory_bytes);
    c.JS_SetMaxStackSize(runtime, 512 * 1024);
    c.JS_SetCanBlock(runtime, false);
    c.JS_SetInterruptHandler(runtime, interrupted, &state);
    const ctx = c.JS_NewContext(runtime) orelse return state.finish("memory_limit", "Could not create JavaScript context within memory limit");
    defer c.JS_FreeContext(ctx);
    c.JS_SetContextOpaque(ctx, &state);

    const bindings_source = try allocator.dupeSentinel(u8, bindings_json, 0);
    defer allocator.free(bindings_source);
    const bindings = c.JS_ParseJSON(ctx, bindings_source.ptr, bindings_source.len, "<program-bindings>");
    defer c.JS_FreeValue(ctx, bindings);
    if (c.JS_IsException(bindings)) return state.exception(ctx);
    if (!c.JS_IsObject(bindings)) return state.finish("invalid_bindings", "Program bindings must be an object");
    const global = c.JS_GetGlobalObject(ctx);
    defer c.JS_FreeValue(ctx, global);
    // No shared-memory/thread coordination is needed for orchestration. Keep
    // can_block=false as well, so no recovered atomic wait could block a worker.
    for ([_][*:0]const u8{ "Atomics", "SharedArrayBuffer" }) |name| {
        const atom = c.JS_NewAtom(ctx, name);
        if (atom == c.JS_ATOM_NULL) return state.exception(ctx);
        defer c.JS_FreeAtom(ctx, atom);
        const removed = c.JS_DeleteProperty(ctx, global, atom, 0);
        if (removed < 0) return state.exception(ctx);
        if (removed == 0) return state.finish("runtime_error", "Could not remove a blocking JavaScript API");
    }
    const tools = c.JS_NewObject(ctx);
    defer c.JS_FreeValue(ctx, tools);
    if (c.JS_IsException(tools) or
        !set(ctx, global, "inputs", c.JS_GetPropertyStr(ctx, bindings, "inputs")) or
        !set(ctx, global, "schemas", c.JS_GetPropertyStr(ctx, bindings, "schemas")) or
        !set(ctx, tools, "ds4_decide", c.JS_NewCFunction2(ctx, decide, "ds4_decide", 1, c.JS_CFUNC_generic, 0)) or
        !set(ctx, global, "tools", c.JS_DupValue(ctx, tools)) or
        !set(ctx, global, "text", c.JS_NewCFunction2(ctx, emit, "text", 1, c.JS_CFUNC_generic, 0))) return state.exception(ctx);

    const source = try std.fmt.allocPrintSentinel(allocator, "(async () => {{\n\"use strict\";\n{s}\n}})()", .{code}, 0);
    defer allocator.free(source);
    const promise = c.JS_Eval(ctx, source.ptr, source.len, "<ds4-program>", c.JS_EVAL_TYPE_GLOBAL);
    defer c.JS_FreeValue(ctx, promise);
    if (c.JS_IsException(promise)) return state.exception(ctx);
    while (c.JS_PromiseState(ctx, promise) == c.JS_PROMISE_PENDING) {
        if (state.stopped()) return state.finish(null, "");
        var job_context: ?*c.JSContext = null;
        const jobs = c.JS_ExecutePendingJob(runtime, &job_context);
        if (jobs < 0) return state.exception(job_context orelse ctx);
        if (jobs == 0 and c.JS_PromiseState(ctx, promise) == c.JS_PROMISE_PENDING)
            return state.finish("unresolved_promise", "Program is awaiting a promise with no remaining work");
    }
    if (state.stopped()) return state.finish(null, "");
    if (c.JS_PromiseState(ctx, promise) == c.JS_PROMISE_REJECTED) {
        const rejected = c.JS_PromiseResult(ctx, promise);
        defer c.JS_FreeValue(ctx, rejected);
        return state.failure(ctx, rejected);
    }
    return state.finish(null, "");
}

const TestHost = struct {
    calls: usize = 0,
    cancel: bool = false,

    fn call(context: ?*anyopaque, raw: []const u8) ![:0]u8 {
        const self: *TestHost = @ptrCast(@alignCast(context.?));
        self.calls += 1;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
        defer parsed.deinit();
        if (parsed.value.object.get("cancel")) |flag| {
            if (flag.bool) self.cancel = true;
        }
        return allocator.dupeSentinel(u8, "{\"choice\":\"billing\"}", 0);
    }
    fn cancelled(context: ?*anyopaque) bool {
        const self: *TestHost = @ptrCast(@alignCast(context.?));
        return self.cancel;
    }
    fn host(self: *TestHost) Host {
        return .{ .context = self, .call = call, .cancelled = cancelled };
    }
};

test "async program uses bound data and emits typed snapshots" {
    var host = TestHost{};
    var result = try run("const r = await tools.ds4_decide({input:inputs.evidence, questions:schemas.route}); text({choice:r.choice}); text([true,3,null]);", "{\"inputs\":{\"evidence\":\"bound invoice\"},\"schemas\":{\"route\":[]}}", host.host(), .{});
    defer result.deinit();
    try std.testing.expectEqual(Status.completed, result.status);
    try std.testing.expectEqual(@as(usize, 1), host.calls);
    try std.testing.expectEqualStrings("{\"output\":[{\"choice\":\"billing\"},[true,3,null]],\"error\":null}", result.output);
}

test "program context exposes no ambient filesystem network or process APIs" {
    var host = TestHost{};
    var result = try run("text([typeof process,typeof require,typeof fetch,typeof std,typeof os,typeof console,typeof Atomics,typeof SharedArrayBuffer]);", "{}", host.host(), .{});
    defer result.deinit();
    try std.testing.expectEqual(Status.completed, result.status);
    try std.testing.expectEqualStrings("{\"output\":[[\"undefined\",\"undefined\",\"undefined\",\"undefined\",\"undefined\",\"undefined\",\"undefined\",\"undefined\"]],\"error\":null}", result.output);
}

test "CPU microtask and regex loops stop at the program deadline" {
    for ([_][]const u8{ "while(true){}", "const again=()=>Promise.resolve().then(again); await again();", "/(a|aa)+$/.test('a'.repeat(100)+'!');" }) |code| {
        var host = TestHost{};
        const start = nowMillis();
        var result = try run(code, "{}", host.host(), .{ .timeout_ms = 20 });
        defer result.deinit();
        try std.testing.expectEqual(Status.failed, result.status);
        try std.testing.expect(std.mem.indexOf(u8, result.output, "\"code\":\"timeout\"") != null);
        try std.testing.expect(nowMillis() - start < 2000);
    }
}

test "cancellation is checked before native work and after a callback" {
    for ([_]bool{ false, true }) |initial| {
        var host = TestHost{ .cancel = initial };
        var result = try run("await tools.ds4_decide({cancel:true}); text('too late');", "{}", host.host(), .{});
        defer result.deinit();
        try std.testing.expectEqual(Status.cancelled, result.status);
        try std.testing.expectEqual(@as(usize, if (initial) 0 else 1), host.calls);
        try std.testing.expect(std.mem.indexOf(u8, result.output, "too late") == null);
    }
}

test "heap and stack exhaustion return program errors" {
    const cases = .{
        .{ "const x=[]; while(true) x.push('x'.repeat(8192));", "memory" },
        .{ "function recurse(){return 1+recurse();} recurse();", "stack" },
    };
    inline for (cases) |case| {
        var host = TestHost{};
        var result = try run(case[0], "{}", host.host(), .{ .memory_bytes = 1024 * 1024 });
        defer result.deinit();
        try std.testing.expectEqual(Status.failed, result.status);
        try std.testing.expect(std.mem.indexOf(u8, result.output, case[1]) != null);
    }
}

test "output cap and stalled promise fail clearly" {
    var host = TestHost{};
    var output = try run("text('x'.repeat(4096));", "{}", host.host(), .{ .max_output_bytes = 256 });
    defer output.deinit();
    try std.testing.expectEqual(Status.failed, output.status);
    try std.testing.expect(output.output.len <= 256);
    try std.testing.expect(std.mem.indexOf(u8, output.output, "output_limit") != null);
    var stalled = try run("await new Promise(()=>{});", "{}", host.host(), .{});
    defer stalled.deinit();
    try std.testing.expectEqual(Status.failed, stalled.status);
    try std.testing.expect(std.mem.indexOf(u8, stalled.output, "unresolved_promise") != null);
}

test "script errors retain bounded valid JSON messages" {
    var host = TestHost{};
    var syntax = try run("await (", "{}", host.host(), .{});
    defer syntax.deinit();
    try std.testing.expectEqual(Status.failed, syntax.status);
    try std.testing.expect(std.mem.indexOf(u8, syntax.output, "SyntaxError") != null);
    var thrown = try run("throw new Error('🙂'.repeat(10000));", "{}", host.host(), .{ .max_output_bytes = 256 });
    defer thrown.deinit();
    try std.testing.expectEqual(Status.failed, thrown.status);
    try std.testing.expect(thrown.output.len <= 256);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, thrown.output, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("javascript_error", parsed.value.object.get("error").?.object.get("code").?.string);
}

test "native decision errors throw their useful message and can be caught" {
    const NativeFailure = struct {
        fn call(_: ?*anyopaque, _: []const u8) ![:0]u8 {
            return allocator.dupeSentinel(u8, "{\"error\":{\"code\":\"invalid_input\",\"message\":\"Input must be text\"},\"status\":400}", 0);
        }
    };
    const host = Host{ .context = null, .call = NativeFailure.call };
    var unhandled = try run("await tools.ds4_decide({input:7});", "{}", host, .{});
    defer unhandled.deinit();
    try std.testing.expectEqual(Status.failed, unhandled.status);
    try std.testing.expect(std.mem.indexOf(u8, unhandled.output, "Input must be text") != null);
    var caught = try run("try { await tools.ds4_decide({input:7}); } catch(error) { text({message:error.message}); }", "{}", host, .{});
    defer caught.deinit();
    try std.testing.expectEqual(Status.completed, caught.status);
    try std.testing.expectEqualStrings("{\"output\":[{\"message\":\"Input must be text\"}],\"error\":null}", caught.output);
}
