//! A decision engine owns one immutable model and bounded reusable scratch.
//! Each request borrows outputs only while this engine's mutex is held.
const std = @import("std");
const c = @import("clef_runtime_c");
const Backend = @import("backend.zig");
const Head = @import("head.zig").Head;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Record = @import("record.zig");
const Media = @import("media_prepare.zig").Plan;
const types = @import("types.zig");
const a = std.heap.c_allocator;

fn lockCancelable(mutex: *c.pthread_mutex_t, cancel: Backend.Cancel, context: ?*anyopaque) !void {
    const check = cancel orelse {
        if (c.pthread_mutex_lock(mutex) != 0) return error.MutexLock;
        return;
    };
    while (true) {
        if (check(context)) return error.Cancelled;
        const result = c.pthread_mutex_trylock(mutex);
        if (result == 0) return;
        if (result != @backingInt(std.c.E.BUSY)) return error.MutexLock;
        // A queued native tool must still observe its request deadline while
        // another request owns this engine. The uncontended path never sleeps.
        const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 2_000_000 };
        _ = c.nanosleep(&delay, null);
    }
}

pub const Engine = struct {
    backend: *Backend.Engine,
    head: ?Head,
    tokenizer: Tokenizer,
    mutex: c.pthread_mutex_t,
    max_tokens: usize,
    timing: bool,
    gpu_head: bool,
    full_gpu_head: bool,

    pub fn open(directory: []const u8, max_tokens: usize) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        if (c.pthread_mutex_init(&self.mutex, null) != 0) return error.MutexInitialization;
        errdefer _ = c.pthread_mutex_destroy(&self.mutex);
        self.backend = try Backend.Engine.open(directory);
        errdefer self.backend.close();
        // Keep one head owner. The qualified GPU path is ordinary serving;
        // CPU/partial paths remain useful when isolating numerical changes.
        self.gpu_head = enabled("DS4_CLEF_HEAD_GPU");
        const full_gpu_head = c.getenv("DS4_CLEF_FULL_HEAD_GPU");
        self.full_gpu_head = if (full_gpu_head != null)
            std.mem.eql(u8, std.mem.span(full_gpu_head), "1")
        else
            !self.gpu_head;
        self.head = null;
        if (self.full_gpu_head) try self.backend.prepareGpuHead() else self.head = try Head.open(a, self.backend);
        errdefer if (self.head) |*head| head.deinit();
        const path = try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{directory});
        defer a.free(path);
        self.tokenizer = try Tokenizer.open(a, path);
        errdefer self.tokenizer.deinit();
        const limit: usize = if (self.backend.config.family == .omni) 64000 else 16384;
        self.max_tokens = @min(if (max_tokens == 0) limit else max_tokens, limit);
        self.timing = enabled("DS4_CLEF_TIMING");
        return self;
    }

    pub fn close(self: *Engine) void {
        self.tokenizer.deinit();
        if (self.head) |*head| head.deinit();
        self.backend.close();
        _ = c.pthread_mutex_destroy(&self.mutex);
        a.destroy(self);
    }

    pub fn name(self: *const Engine) [:0]const u8 {
        return if (self.backend.config.family == .omni) "clef-omni" else if (self.backend.config.layers == 32) "clef-flash" else "clef";
    }

    pub fn decide(self: *Engine, allocator: std.mem.Allocator, request: types.Request) !types.Result {
        return self.decideCancelable(allocator, request, null, null);
    }

    pub fn decideCancelable(self: *Engine, allocator: std.mem.Allocator, request: types.Request, cancel: Backend.Cancel, context: ?*anyopaque) !types.Result {
        const started = if (self.timing) now() else 0;
        const media = try Media.prepare(allocator, request.media, self.backend.config.family == .omni, cancel, context);
        const queued = if (self.timing) now() else 0;
        try lockCancelable(&self.mutex, cancel, context);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        const acquired = if (self.timing) now() else 0;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        var record = try Record.encodeMedia(&self.tokenizer, allocator, request, self.max_tokens, self.backend.config.family == .omni, media.specs);
        defer record.deinit();
        const prepared = if (self.timing) now() else 0;
        const injections = try media.embeddings(allocator, self.backend, &record, cancel, context);
        const towers = if (self.timing) now() else 0;
        const hidden = if (request.media.len == 0)
            try self.backend.inferText(record.ids, record.schema_start, cancel, context)
        else
            try self.backend.inferMedia(record.ids, record.positions, injections, cancel, context);
        const inferred = if (self.timing) now() else 0;
        const lexical = try self.backend.embeddingRows(record.embedding_ids);
        defer a.free(lexical);
        const embedded = if (self.timing) now() else 0;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        const result = if (self.full_gpu_head)
            try self.backend.decideGpu(allocator, &record, hidden, lexical, cancel, context)
        else if (self.gpu_head)
            try self.head.?.runPrepared(allocator, &record, try self.backend.headFeaturesCancelable(hidden, cancel, context), lexical)
        else
            try self.head.?.run(allocator, &record, hidden, lexical);
        if (self.timing) {
            std.debug.print("clef-timing: tokens={d} media_ms={d:.3} queue_ms={d:.3} prepare_ms={d:.3} towers_ms={d:.3} backbone_ms={d:.3} lexical_ms={d:.3} head_ms={d:.3}\n", .{
                record.ids.len,               (queued - started) * 1000,  (acquired - queued) * 1000,
                (prepared - acquired) * 1000, (towers - prepared) * 1000, (inferred - towers) * 1000,
                (embedded - inferred) * 1000, (now() - embedded) * 1000,
            });
        }
        return result;
    }
};

fn enabled(name_: [*:0]const u8) bool {
    const value = c.getenv(name_);
    return value != null and std.mem.eql(u8, std.mem.span(value), "1");
}

fn now() f64 {
    var value: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &value);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_nsec)) / 1e9;
}

test "queued decision deadline cancels before the engine mutex is released" {
    var mutex: c.pthread_mutex_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.pthread_mutex_init(&mutex, null));
    defer _ = c.pthread_mutex_destroy(&mutex);
    const Waiter = struct {
        mutex: *c.pthread_mutex_t,
        deadline: f64,
        finished: std.atomic.Value(bool) = .init(false),
        failure: ?anyerror = null,

        fn cancelled(context: ?*anyopaque) callconv(.c) bool {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            return now() >= self.deadline;
        }
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            lockCancelable(self.mutex, cancelled, self) catch |err| {
                self.failure = err;
                return;
            };
            _ = c.pthread_mutex_unlock(self.mutex);
        }
    };
    try std.testing.expectEqual(@as(c_int, 0), c.pthread_mutex_lock(&mutex));
    var held = true;
    defer {
        if (held) _ = c.pthread_mutex_unlock(&mutex);
    }
    var waiter = Waiter{ .mutex = &mutex, .deadline = now() + 0.02 };
    const thread = try std.Thread.spawn(.{}, Waiter.run, .{&waiter});
    const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 200_000_000 };
    _ = c.nanosleep(&delay, null);
    const retired_while_locked = waiter.finished.load(.acquire);
    // Release even on regression, so an unconditional blocking lock makes
    // this test fail instead of hanging the complete host suite.
    _ = c.pthread_mutex_unlock(&mutex);
    held = false;
    thread.join();
    try std.testing.expect(retired_while_locked);
    try std.testing.expectEqual(@as(anyerror, error.Cancelled), waiter.failure.?);
}
