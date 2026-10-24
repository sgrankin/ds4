//! Public-API qualification of live steering ownership on one installed model.
//! Model execution stays behind the public ABI; this fixture owns its engines,
//! sessions and logit buffers and never downloads weights.
const std = @import("std");
const c = @import("steering_c");
const allocator = std.heap.c_allocator;

var error_text: [512]u8 = @splat(0);

fn require(condition: bool, message: []const u8) !void {
    if (!condition) {
        std.debug.print("session steering: {s}: {s}\n", .{ message, std.mem.sliceTo(&error_text, 0) });
        return error.CheckFailed;
    }
}

fn exact(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

fn gap(a: []const f32, b: []const f32) f64 {
    var maximum: f64 = 0;
    for (a, b) |left, right| maximum = @max(maximum, @abs(@as(f64, left) - right));
    return maximum;
}

const Engine = struct {
    handle: *c.ds4_engine,

    fn open(options: *const c.ds4_engine_options) !Engine {
        var handle: ?*c.ds4_engine = null;
        try require(c.ds4_engine_open(&handle, options) == 0, "open model with zero-preloaded bank");
        return .{ .handle = handle.? };
    }

    fn close(self: Engine) void {
        c.ds4_engine_close(self.handle);
    }
};

const Session = struct {
    handle: ?*c.ds4_session = null,

    fn create(engine: Engine) !Session {
        var session = Session{};
        try require(c.ds4_session_create(&session.handle, engine.handle, 256) == 0, "create session");
        return session;
    }

    fn close(self: *Session) void {
        if (self.handle) |handle| c.ds4_session_free(handle);
        self.handle = null;
    }

    fn ptr(self: Session) *c.ds4_session {
        return self.handle.?;
    }

    fn position(self: Session) c_int {
        return c.ds4_session_pos(self.ptr());
    }

    fn scale(self: Session) f32 {
        return c.ds4_session_directional_steering_ffn(self.ptr());
    }

    fn setScale(self: Session, value: f32) !void {
        try require(c.ds4_session_set_directional_steering_ffn(self.ptr(), value) == 0, "set live FFN scale");
        try require(self.scale() == value, "get actual session scale");
    }

    fn sync(self: Session, prompt: *const c.ds4_tokens) !void {
        try require(c.ds4_session_sync(self.ptr(), prompt, &error_text, error_text.len) == 0, "synchronize prompt");
        try require(self.position() == prompt.len, "prompt position");
    }

    fn eval(self: Session, token: c_int) !void {
        const before = self.position();
        try require(c.ds4_session_eval(self.ptr(), token, &error_text, error_text.len) == 0, "evaluate fixed token");
        try require(self.position() == before + 1, "one committed input");
    }

    fn copyLogits(self: Session, out: []f32) !void {
        const vocabulary: c_int = @intCast(out.len);
        try require(c.ds4_session_checkpoint_valid(self.ptr()), "valid checkpoint");
        try require(c.ds4_session_copy_logits(self.ptr(), out.ptr, vocabulary) == vocabulary, "copy all vocabulary logits");
        for (out) |value| try require(std.math.isFinite(value), "finite logits");
    }

    fn tokens(self: Session) ![]const c_int {
        const history = c.ds4_session_tokens(self.ptr());
        try require(history != null, "published token history");
        return history.*.v[0..@intCast(history.*.len)];
    }
};

fn syncPair(sessions: *const [2]Session, prompt: *const c.ds4_tokens) !void {
    for (sessions) |session| {
        try session.setScale(0);
        try session.sync(prompt);
    }
}

fn checkBatch(sessions: *const [2]Session, prompt: *const c.ds4_tokens, tokens: []const c_int, reference: [2][]f32, actual: []f32, mixed: bool) !void {
    try syncPair(sessions, prompt);
    try sessions[0].setScale(1);
    try sessions[1].setScale(if (mixed) 0 else 1);
    for (tokens[0..2]) |token| for (sessions) |session| try session.eval(token);
    for (sessions, reference) |session, logits| try session.copyLogits(logits);

    try syncPair(sessions, prompt);
    try sessions[0].setScale(1);
    try sessions[1].setScale(if (mixed) 0 else 1);
    for (tokens[0..2], 0..) |token, step| {
        var items = [_]c.ds4_decode_item{
            .{ .session = sessions[0].ptr(), .token = token },
            .{ .session = sessions[1].ptr(), .token = token },
        };
        try require(c.ds4_sessions_eval_batch(&items, items.len, &error_text, error_text.len) == 0, "homogeneous/mixed steering batch");
        for (sessions, 0..) |session, index| {
            const expected = prompt.len + @as(c_int, @intCast(step)) + 1;
            try require(session.position() == expected, "batch committed frontier");
            try require(session.scale() == (if (index == 1 and mixed) @as(f32, 0) else 1), "batch retained independent scales");
            const history = try session.tokens();
            const committed: usize = @intCast(expected);
            try require(history.len == committed and history[history.len - 1] == token, "batch committed the requested input");
        }
    }
    const kind: [*:0]const u8 = if (mixed) "mixed" else "homogeneous";
    for (sessions, reference, 0..) |session, logits, index| {
        try session.copyLogits(actual);
        // Batched arithmetic may differ from scalar replay. Record its gap;
        // exact identity belongs to zero/scalar checks, not a different graph.
        _ = c.printf("batch %s member%zu replay_max_delta=%.9g\n", kind, index, gap(logits, actual));
    }
}

fn checkSpeculative(sessions: *const [2]Session, prompt: *const c.ds4_tokens, a: []f32, b: []f32, limit: c_int) !void {
    const vocabulary: c_int = @intCast(a.len);
    try syncPair(sessions, prompt);
    for (sessions) |session| try session.setScale(1);
    for (0..3) |cycle| {
        const token = c.ds4_session_argmax(sessions[0].ptr());
        try require(token >= 0, "speculative first token");
        var accepted: [8]c_int = @splat(0);
        const before = sessions[0].position();
        const count = c.ds4_session_eval_speculative_argmax(sessions[0].ptr(), token, limit, -1, &accepted, accepted.len, &error_text, error_text.len);
        try require(count >= 1 and count <= limit and accepted[0] == token, "speculative committed inputs");
        try require(sessions[0].position() == before + count, "speculative committed frontier");
        const history = try sessions[0].tokens();
        const committed: usize = @intCast(before + count);
        try require(history.len == committed, "speculative token history length");
        for (accepted[0..@intCast(count)], 0..) |id, index| {
            try require(id >= 0 and id < vocabulary, "accepted token range");
            try require(history[@as(usize, @intCast(before)) + index] == id, "speculative published token history");
            try sessions[1].eval(id);
        }
        try require(sessions[0].position() == sessions[1].position(), "speculative replay position");
        try require(sessions[0].scale() == 1 and sessions[1].scale() == 1, "speculation retained scales");
        try sessions[0].copyLogits(a);
        try sessions[1].copyLogits(b);
        _ = c.printf("speculative cycle%zu committed=%d replay_max_delta=%.9g\n", cycle, count, gap(a, b));
    }
}

const Options = struct {
    engine: c.ds4_engine_options,
    batch: bool = false,
    speculative: bool = false,
};

fn usage(program: []const u8) noreturn {
    std.debug.print("usage: {s} MODEL DIRECTIONS [--mode project|add] [--prefill-chunk N] [--ssd] [--cache-gb N] [--batch] [--mtp|--packet]\n", .{program});
    std.process.exit(2);
}

fn parse(args: std.process.Args) Options {
    var iterator = args.iterate();
    const program = iterator.next() orelse "test-session-steering";
    const model = iterator.next() orelse usage(program);
    const directions = iterator.next() orelse usage(program);
    var options = Options{ .engine = std.mem.zeroes(c.ds4_engine_options) };
    options.engine.model_path = model.ptr;
    options.engine.directional_steering_file = directions.ptr;
    options.engine.backend = c.DS4_BACKEND_METAL;
    options.engine.context_size = 256;
    options.engine.prefill_chunk = 64;
    options.engine.power_percent = 100;
    options.engine.placement_session_count_hint = 2;
    options.engine.share_session_prefill_workspace = true;
    while (iterator.next()) |argument| {
        if (std.mem.eql(u8, argument, "--ssd")) {
            options.engine.ssd_streaming = true;
        } else if (std.mem.eql(u8, argument, "--batch")) {
            options.batch = true;
        } else if (std.mem.eql(u8, argument, "--mtp")) {
            options.engine.glm_mtp = true;
            options.speculative = true;
        } else if (std.mem.eql(u8, argument, "--packet")) {
            options.speculative = true;
        } else if (std.mem.eql(u8, argument, "--mode")) {
            const mode = iterator.next() orelse usage(program);
            options.engine.directional_steering_mode = if (std.mem.eql(u8, mode, "project")) c.DS4_STEERING_PROJECT else if (std.mem.eql(u8, mode, "add")) c.DS4_STEERING_ADD else usage(program);
        } else if (std.mem.eql(u8, argument, "--cache-gb")) {
            const value = iterator.next() orelse usage(program);
            const gb = std.fmt.parseInt(u64, value, 10) catch usage(program);
            options.engine.ssd_streaming_cache_bytes = std.math.mul(u64, gb, 1 << 30) catch usage(program);
        } else if (std.mem.eql(u8, argument, "--prefill-chunk")) {
            const value = iterator.next() orelse usage(program);
            options.engine.prefill_chunk = std.fmt.parseInt(u32, value, 10) catch usage(program);
        } else {
            std.debug.print("unknown argument: {s}\n", .{argument});
            usage(program);
        }
    }
    return options;
}

fn run(options: Options) !void {
    const engine = try Engine.open(&options.engine);
    defer engine.close();
    const vocabulary = c.ds4_engine_vocab_size(engine.handle);
    try require(vocabulary > 0, "vocabulary");
    const a = try allocator.alloc(f32, @intCast(vocabulary));
    defer allocator.free(a);
    const b = try allocator.alloc(f32, @intCast(vocabulary));
    defer allocator.free(b);
    const saved = try allocator.alloc(f32, @intCast(vocabulary));
    defer allocator.free(saved);
    var sessions = [_]Session{ .{}, .{} };
    defer for (&sessions) |*session| session.close();
    for (&sessions) |*session| {
        session.* = try Session.create(engine);
        try require(session.scale() == 0, "initial zero scale");
    }
    var prompt = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&prompt);
    c.ds4_encode_chat_prompt(engine.handle, null, "Count from one to one hundred, writing every number.", c.DS4_THINK_NONE, &prompt);
    try require(prompt.len > 0 and prompt.len < 224, "bounded prompt");
    try syncPair(&sessions, &prompt);
    try sessions[0].copyLogits(a);
    try sessions[1].copyLogits(b);
    try require(exact(a, b), "zero prefill exact FP32 identity");
    var tokens: [3]c_int = undefined;
    tokens[0] = c.ds4_session_argmax(sessions[1].ptr());
    try require(tokens[0] >= 0, "fixed decode token");
    for (sessions) |session| try session.eval(tokens[0]);
    try sessions[0].copyLogits(a);
    try sessions[1].copyLogits(b);
    try require(exact(a, b), "zero decode exact FP32 identity");
    _ = c.puts("zero prefill/decode: all FP32 logits identical");

    const position = sessions[0].position();
    try sessions[0].setScale(1);
    try require(sessions[1].scale() == 0, "independent session scale");
    try sessions[0].copyLogits(saved);
    try require(exact(a, saved) and sessions[0].position() == position, "scale change leaves existing frontier/logits intact");
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 101, -101 }) |invalid| {
        try require(c.ds4_session_set_directional_steering_ffn(sessions[0].ptr(), invalid) != 0, "reject invalid scale");
        try require(sessions[0].scale() == 1 and sessions[1].scale() == 0 and sessions[0].position() == position, "invalid setter preserves ownership/frontier");
        try sessions[0].copyLogits(a);
        try require(exact(a, saved), "invalid setter preserves logits");
    }
    // New sessions inherit the published default. Reuse the other session's
    // slot so this check never keeps more than two GPU sessions alive.
    sessions[1].close();
    sessions[1] = try Session.create(engine);
    try require(sessions[1].scale() == 1, "new session inherits published default");
    try sessions[1].setScale(0);
    try sessions[1].sync(&prompt);
    try sessions[1].eval(tokens[0]);
    try sessions[1].copyLogits(b);
    try require(exact(b, saved), "independent zero-scale replay identity");
    try require(sessions[0].scale() == 1, "new-session defaults do not mutate existing scale");
    tokens[1] = c.ds4_session_argmax(sessions[1].ptr());
    try require(tokens[1] >= 0, "next fixed decode token");
    for (sessions) |session| try session.eval(tokens[1]);
    try sessions[0].copyLogits(a);
    try sessions[1].copyLogits(b);
    try require(!exact(a, b), "nonzero steering changes actual decode");
    _ = c.printf("live nonzero decode: max_delta=%.9g\n", gap(a, b));
    const steered_position = sessions[0].position();
    try sessions[0].setScale(0);
    try sessions[0].copyLogits(saved);
    try require(exact(a, saved) and sessions[0].position() == steered_position, "restoring zero retains the existing steered state");
    const history = try sessions[0].tokens();
    const prompt_length: usize = @intCast(prompt.len);
    try require(history.len == prompt_length + 2 and c.ds4_tokens_starts_with(c.ds4_session_tokens(sessions[0].ptr()), &prompt) and
        history[prompt_length] == tokens[0] and history[prompt_length + 1] == tokens[1], "scale restoration retains exact committed token history");
    tokens[2] = c.ds4_session_argmax(sessions[1].ptr());
    try require(tokens[2] >= 0, "post-restore fixed token");
    for (sessions) |session| try session.eval(tokens[2]);
    try sessions[0].copyLogits(a);
    try sessions[1].copyLogits(b);
    _ = c.printf("restored-zero continued decode: historical_state_max_delta=%.9g\n", gap(a, b));
    _ = c.puts("live ownership, invalid scales and future-only publication: OK");

    if (options.batch) {
        try checkBatch(&sessions, &prompt, &tokens, .{ a, b }, saved, false);
        try checkBatch(&sessions, &prompt, &tokens, .{ a, b }, saved, true);
    }
    if (options.speculative) try checkSpeculative(&sessions, &prompt, a, b, std.math.clamp(c.ds4_engine_decode_packet_size(engine.handle), 1, 8));
    _ = c.puts("session steering: PASS");
}

pub fn main(init: std.process.Init.Minimal) !void {
    run(parse(init.args)) catch |err| {
        if (err != error.CheckFailed) std.debug.print("session steering: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}
