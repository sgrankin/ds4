//! Real public-API owner qualification: two native DS4.1 engines share one
//! checkpoint's file pages, while each owns its cache, sessions and GPU state.
const std = @import("std");
const c = @import("runtime_engine_c");
const a = std.heap.c_allocator;
const context = 512;
const steps = 3;
const query_b = "Name the country whose capital is Tokyo. Answer briefly.";
var error_text: [512]u8 = @splat(0);

fn require(ok: bool, stage: []const u8) !void {
    if (!ok) {
        std.debug.print("two native engines: {s}: {s}\n", .{ stage, std.mem.sliceTo(&error_text, 0) });
        return error.OwnerInvariant;
    }
}

fn processAdmission(held: bool) !void {
    const setting = c.getenv("DS4_LOCK_FILE");
    const path: [*c]const u8 = if (setting != null and setting[0] != 0) setting else "/tmp/ds4.lock";
    const fd = c.open(path, c.O_RDWR | c.O_CREAT, @as(c_uint, 0o600));
    try require(fd >= 0, "open independent process-lock probe");
    defer _ = c.close(fd);
    const acquired = c.flock(fd, c.LOCK_EX | c.LOCK_NB) == 0;
    if (acquired) {
        try require(c.flock(fd, c.LOCK_UN) == 0, "release independent process-lock probe");
    } else {
        try require(c.__error().* == c.EWOULDBLOCK, "process-lock probe fails only because admission is held");
    }
    try require(acquired != held, "process admission follows actual remaining native owners");
}

const Engine = struct {
    handle: ?*c.ds4_engine = null,
    fn open(path: [*:0]const u8) !Engine {
        var options = std.mem.zeroes(c.ds4_engine_options);
        options.model_path = path;
        options.backend = c.DS4_BACKEND_METAL;
        options.n_threads = 8;
        options.context_size = context;
        options.prefill_chunk = 64;
        options.power_percent = 100;
        options.ssd_streaming = true;
        options.ssd_streaming_cache_bytes = 8 * (1 << 30);
        options.ssd_streaming_full_layers_set = true;
        options.ssd_streaming_full_layers = 0;
        options.placement_session_count_hint = 1;
        var self = Engine{};
        try require(c.ds4_engine_open(&self.handle, &options) == 0, "open native SSD-streaming engine");
        errdefer self.close();
        try require(self.handle != null and c.ds4_engine_is_deepseek41(self.handle), "DS4.1 family");
        return self;
    }
    fn close(self: *Engine) void {
        if (self.handle) |handle| c.ds4_engine_close(handle);
        self.handle = null;
    }
    fn prompt(self: Engine, text: [*:0]const u8) !c.ds4_tokens {
        var out = std.mem.zeroes(c.ds4_tokens);
        c.ds4_encode_chat_prompt(self.handle, null, text, c.DS4_THINK_NONE, &out);
        errdefer c.ds4_tokens_free(&out);
        try require(out.len > 0 and out.len + steps < context, "bounded public chat prompt");
        return out;
    }
};

const Session = struct {
    handle: ?*c.ds4_session = null,
    fn create(engine: Engine) !Session {
        var self = Session{};
        try require(c.ds4_session_create(&self.handle, engine.handle, context) == 0, "create session");
        return self;
    }
    fn close(self: *Session) void {
        if (self.handle) |handle| c.ds4_session_free(handle);
        self.handle = null;
    }
    fn sync(self: Session, prompt: *const c.ds4_tokens) !void {
        try require(c.ds4_session_sync(self.handle, prompt, &error_text, error_text.len) == 0, "prefill independent prompt");
        try require(c.ds4_session_pos(self.handle) == prompt.len, "prefill position");
    }
    fn eval(self: Session, token: c_int) !void {
        const before = c.ds4_session_pos(self.handle);
        try require(c.ds4_session_eval(self.handle, token, &error_text, error_text.len) == 0, "decode fixed input token");
        try require(c.ds4_session_pos(self.handle) == before + 1, "decode position");
    }
    fn logits(self: Session, output: []f32) !void {
        try require(c.ds4_session_checkpoint_valid(self.handle), "live checkpoint");
        const count: c_int = @intCast(output.len);
        try require(c.ds4_session_copy_logits(self.handle, output.ptr, count) == count, "copy all frozen logits");
        for (output) |v| try require(std.math.isFinite(v), "finite logits");
    }
    fn history(self: Session, prompt: *const c.ds4_tokens, decoded: []const c_int) !void {
        const actual = c.ds4_session_tokens(self.handle);
        try require(actual != null and actual.*.len == prompt.len + @as(c_int, @intCast(decoded.len)), "committed token count");
        const before: usize = @intCast(prompt.len);
        try require(std.mem.eql(c_int, actual.*.v[0..before], prompt.v[0..before]), "unchanged prompt tokens");
        try require(std.mem.eql(c_int, actual.*.v[before..][0..decoded.len], decoded), "committed fixed decode tokens");
    }
};

const Frozen = struct {
    vocabulary: usize,
    logits: []f32,
    tokens: [steps]c_int,
    fn capture(engine: Engine, prompt: *const c.ds4_tokens) !Frozen {
        const vocabulary = c.ds4_engine_vocab_size(engine.handle);
        try require(vocabulary > 0, "vocabulary size");
        var self = Frozen{
            .vocabulary = @intCast(vocabulary),
            .logits = try a.alloc(f32, @as(usize, @intCast(vocabulary)) * (steps + 1)),
            .tokens = undefined,
        };
        errdefer self.deinit();
        var session = try Session.create(engine);
        defer session.close();
        try session.sync(prompt);
        for (0..steps + 1) |step| {
            try session.logits(self.row(step));
            if (step != steps) {
                const token = c.ds4_session_argmax(session.handle);
                try require(token >= 0 and token < vocabulary, "frozen argmax token");
                self.tokens[step] = token;
                try session.eval(token);
            }
        }
        try session.history(prompt, &self.tokens);
        return self;
    }
    fn row(self: Frozen, step: usize) []f32 {
        return self.logits[step * self.vocabulary ..][0..self.vocabulary];
    }
    fn deinit(self: Frozen) void {
        a.free(self.logits);
    }
    fn check(self: Frozen, session: Session, prompt: *const c.ds4_tokens, step: usize, scratch: []f32, stage: []const u8) !void {
        try session.logits(scratch);
        const expected = self.row(step);
        var maximum: f64 = 0;
        for (scratch, expected) |actual, reference| maximum = @max(maximum, @abs(@as(f64, actual) - reference));
        std.debug.print("two native engines: {s}, step={d}, max_logit_delta={d:.9}\n", .{ stage, step, maximum });
        try require(std.mem.eql(u8, std.mem.sliceAsBytes(scratch), std.mem.sliceAsBytes(expected)), stage);
        if (step < steps) try require(c.ds4_session_argmax(session.handle) == self.tokens[step], "unchanged next token");
        try session.history(prompt, self.tokens[0..step]);
    }
};

fn run(path: [*:0]const u8) !void {
    const users = c.ds4_runtime_metal_users();
    const selected = c.ds4_metal_context_selected();
    var first = try Engine.open(path);
    defer first.close();
    var prompt_a = try first.prompt("Continue this sequence with the next number: 2, 4, 6,");
    defer c.ds4_tokens_free(&prompt_a);
    const frozen_a = try Frozen.capture(first, &prompt_a);
    defer frozen_a.deinit();
    const scratch = try a.alloc(f32, frozen_a.vocabulary);
    defer a.free(scratch);
    var session_a = try Session.create(first);
    defer session_a.close();
    try session_a.sync(&prompt_a);
    try frozen_a.check(session_a, &prompt_a, 0, scratch, "A before B open");
    try require(c.ds4_runtime_metal_users() == users + 1, "first runtime owner");

    var second = try Engine.open(path);
    defer second.close();
    try require(c.ds4_runtime_metal_users() == users + 2, "two actual native engine owners");
    try processAdmission(true);
    try require(c.ds4_engine_vocab_size(second.handle) == @as(c_int, @intCast(frozen_a.vocabulary)), "same installed vocabulary");
    var prompt_b = try second.prompt(query_b);
    defer c.ds4_tokens_free(&prompt_b);
    const frozen_b = try Frozen.capture(second, &prompt_b);
    defer frozen_b.deinit();
    var session_b = try Session.create(second);
    defer session_b.close();
    try session_b.sync(&prompt_b);
    try frozen_b.check(session_b, &prompt_b, 0, scratch, "B before interleave");
    try frozen_a.check(session_a, &prompt_a, 0, scratch, "A after B load/prefill");

    try session_a.eval(frozen_a.tokens[0]);
    try frozen_a.check(session_a, &prompt_a, 1, scratch, "A after B");
    try session_b.eval(frozen_b.tokens[0]);
    try frozen_b.check(session_b, &prompt_b, 1, scratch, "B after A");
    try session_a.eval(frozen_a.tokens[1]);
    try frozen_a.check(session_a, &prompt_a, 2, scratch, "A after B again");
    session_a.close();
    first.close();
    try require(c.ds4_runtime_metal_users() == users + 1, "closing A preserves B runtime");
    try processAdmission(true);
    try session_b.eval(frozen_b.tokens[1]);
    try frozen_b.check(session_b, &prompt_b, 2, scratch, "B after A retirement");
    try session_b.eval(frozen_b.tokens[2]);
    try frozen_b.check(session_b, &prompt_b, 3, scratch, "B surviving continuation");
    session_b.close();
    var encoded_again = try second.prompt(query_b);
    defer c.ds4_tokens_free(&encoded_again);
    try require(encoded_again.len == prompt_b.len and std.mem.eql(c_int, encoded_again.v[0..@intCast(encoded_again.len)], prompt_b.v[0..@intCast(prompt_b.len)]), "B vocabulary and renderer survive A close");
    // Allocate fresh graph resources after A has retired its registries/PSOs.
    session_b = try Session.create(second);
    try session_b.sync(&prompt_b);
    try frozen_b.check(session_b, &prompt_b, 0, scratch, "fresh B graph after A close");
    session_b.close();
    second.close();
    try processAdmission(false);
    try require(c.ds4_runtime_metal_users() == users, "all runtime borrows returned");
    try require(c.ds4_metal_context_selected() == selected, "public entries restore ambient selection");
    std.debug.print("Two native DS4.1 engines: frozen FP32 logits/tokens, A→B→A, A close/B continuation and fresh B allocation passed.\n", .{});
}

export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc != 2) {
        std.debug.print("Usage: test-runtime-two-engines DS41_GGUF\n", .{});
        return 2;
    }
    run(std.mem.span(argv[1]).ptr) catch |err| {
        std.debug.print("two native engines: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
