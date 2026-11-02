//! One next-chunk PLE read overlaps the current chunk's GPU work. The worker
//! owns its copied n-gram history and output; it never touches live graph state.
const std = @import("std");
const c = @import("gguf_c");
const ngrams = @import("ngrams.zig");
const input = @import("input.zig");
const ops = @import("ops.zig");
const Config = @import("config.zig").Config;

pub const Prefetch = struct {
    rows: []f32 = &.{},
    model: *const c.ds4_model = undefined,
    config: Config = undefined,
    tokens: []const c_int = &.{},
    previous: [c.DS4_MAX_PLE_NGRAM]c_int = undefined,
    thread: c.pthread_t = undefined,
    reading: bool = false,
    ready: bool = false,
    err: c_int = 0,
    elapsed: f64 = 0,

    pub fn init(capacity: u32, width: u32, remaining: usize) Prefetch {
        const count = @as(usize, capacity) * width;
        if (c.getenv("DS4_QWEN4_NO_PLE_PREFETCH") != null or remaining <= capacity or
            capacity < 256 or count * @sizeOf(f32) > 128 * 1024 * 1024) return .{};
        const rows = std.heap.c_allocator.alloc(f32, count) catch {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen PLE prefetch buffer allocation failed\n");
            return .{};
        };
        return .{ .rows = rows };
    }

    pub fn deinit(self: *Prefetch) void {
        self.join(0);
        std.heap.c_allocator.free(self.rows);
        self.* = .{};
    }

    pub fn start(self: *Prefetch, graph: *const c.ds4_qwen4_gpu_graph, model: *const c.ds4_model, tokens: []const c_int) void {
        std.debug.assert(!self.reading);
        self.ready = false;
        if (tokens.len == 0) return;
        self.model = model;
        self.config = graph.config;
        self.previous = graph.ple_prev;
        self.tokens = tokens;
        self.err = 0;
        self.reading = c.pthread_create(&self.thread, null, run, self) == 0;
        if (!self.reading) _ = c.fprintf(c.__stderrp, "ds4: Qwen PLE prefetch thread unavailable; retrying synchronously\n");
    }

    /// Join before progress/cancellation can expose the current frontier. A
    /// failed read remains unready and the next chunk retries synchronously.
    pub fn join(self: *Prefetch, position: c_int) void {
        if (!self.reading) return;
        const started = ops.now();
        if (c.pthread_join(self.thread, null) != 0) c.ds4_engine_model_die("cannot join Qwen PLE prefetch");
        self.reading = false;
        if (!self.ready) _ = c.fprintf(c.__stderrp, "ds4: Qwen PLE prefetch failed at token %d: %s; retrying synchronously\n", position, c.strerror(self.err));
        if (c.getenv("DS4_QWEN4_PLE_PREFETCH_TRACE") != null) _ = c.fprintf(c.__stderrp, "ds4: Qwen PLE prefetch pos=%d count=%zu read_ms=%.3f join_ms=%.3f ready=%d\n", position, self.tokens.len, self.elapsed * 1e3, (ops.now() - started) * 1e3, @as(c_int, @intFromBool(self.ready)));
    }

    fn run(context: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Prefetch = @ptrCast(@alignCast(context.?));
        const started = ops.now();
        defer self.elapsed = ops.now() - started;
        if (!input.validTokens(self.tokens, self.config.shape.n_vocab)) {
            self.err = c.EINVAL;
            return null;
        }
        var ids: [256 * c.DS4_MAX_PLE_HEADS]u32 = undefined;
        const heads = self.config.ple.n_heads;
        const width = self.config.shape.n_embd;
        for (self.tokens, 0..) |token, index| {
            ngrams.step(&self.config, token, &self.previous, ids[index % 256 * heads ..][0..heads]);
            if (index % 256 == 255 or index + 1 == self.tokens.len) {
                const first = index / 256 * 256;
                const count = index - first + 1;
                if (!ngrams.read(self.model, ids[0 .. count * heads], self.rows[first * width ..][0 .. count * width])) {
                    self.err = if (c.__error().* != 0) c.__error().* else c.EIO;
                    return null;
                }
            }
        }
        self.ready = true;
        return null;
    }
};
