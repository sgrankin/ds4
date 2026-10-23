//! Flint model/session lifecycle. The shared server owns scheduling and the
//! protocol; this module owns model arithmetic and model-specific KV validity.
const std = @import("std");
const c = @import("flint_c");
const weights = @import("weights.zig");
const decode = @import("decode.zig");
const prefill = @import("prefill.zig");
const a = std.heap.c_allocator;

pub const Engine = struct {
    // Resident kernels require all threadgroups to be schedulable together.
    // Serialize complete encode/submit/wait/readback phases across sessions.
    mutex: c.pthread_mutex_t,
    model: weights.Weights,
    decoder: decode.Decoder,
    tokenizer: *c.flint_tokenizer,
    chunk: u32,
    stops: [4]u32,

    pub fn open(directory: []const u8, chunk: u32) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        if (c.pthread_mutex_init(&self.mutex, null) != 0) return error.MutexInitialization;
        errdefer _ = c.pthread_mutex_destroy(&self.mutex);
        self.model = try weights.load(directory);
        errdefer self.model.deinit();
        const source = try prefill.shaders.source(a);
        defer a.free(source);
        var compiler_error: [4096]u8 = @splat(0);
        if (!c.flint_gpu_compile(self.model.gpu, source.ptr, source.len, &compiler_error, compiler_error.len)) {
            std.debug.print("flint: prefill compilation: {s}\n", .{std.mem.sliceTo(&compiler_error, 0)});
            return error.CompilePrefill;
        }
        self.decoder = try decode.Decoder.init(&self.model);
        errdefer self.decoder.deinit();
        const tokenizer_path = try std.fmt.allocPrintSentinel(a, "{s}/tokenizer.json", .{directory}, 0);
        defer a.free(tokenizer_path);
        self.tokenizer = c.flint_tokenizer_load(tokenizer_path) orelse return error.LoadTokenizer;
        errdefer c.flint_tokenizer_free(self.tokenizer);
        if (c.flint_tokenizer_vocab_size(self.tokenizer) != decode.vocab) return error.VocabularyMismatch;
        for ([_][:0]const u8{ "<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>" }, &self.stops) |spelling, *id| {
            const found = c.flint_tokenizer_special_id(self.tokenizer, spelling);
            if (found < 0) return error.MissingStopToken;
            id.* = @intCast(found);
        }
        self.chunk = if (chunk == 0) 4096 else @min(chunk, 4096);
        return self;
    }

    pub fn close(self: *Engine) void {
        c.flint_tokenizer_free(self.tokenizer);
        self.decoder.deinit();
        self.model.deinit();
        _ = c.pthread_mutex_destroy(&self.mutex);
        a.destroy(self);
    }
};

const global_bytes = @as(usize, 14) * decode.block_tokens * 1024 * 2;
const local_bytes = @as(usize, 35) * 513 * 1024 * 2;

pub const Session = struct {
    engine: *Engine,
    native: decode.Session,
    tokens: std.ArrayList(c_int) = .empty,
    layout: prefill.Layout,
    scratch: ?*anyopaque = null,
    inputs: *anyopaque,
    logits: *anyopaque,
    valid: bool = false,
    // Decode is deliberately recomputed through prefill before extending a
    // reusable prompt, matching QATSession's different prefill/decode arithmetic.
    prefill_position: u32 = 0,
    saved_local: ?[]u8 = null,
    saved_logits: ?[]f32 = null,
    sampling_seed: ?u64 = null,

    pub fn create(engine: *Engine, context: u32) !*Session {
        const self = try a.create(Session);
        errdefer a.destroy(self);
        self.* = .{
            .engine = engine,
            .native = try decode.Session.init(&engine.decoder, context),
            .layout = try prefill.Layout.init(@min(engine.chunk, context), context),
            .inputs = undefined,
            .logits = undefined,
        };
        errdefer self.native.deinit();
        self.inputs = c.flint_gpu_buffer(engine.model.gpu, @as(usize, self.layout.capacity) * 4) orelse return error.GpuAllocation;
        errdefer c.flint_gpu_buffer_free(self.inputs);
        self.logits = c.flint_gpu_buffer(engine.model.gpu, decode.vocab * 4) orelse return error.GpuAllocation;
        @memset(self.logitValues(), 0);
        return self;
    }

    pub fn destroy(self: *Session) void {
        self.native.deinit();
        if (self.scratch) |buffer| c.flint_gpu_buffer_free(buffer);
        c.flint_gpu_buffer_free(self.inputs);
        c.flint_gpu_buffer_free(self.logits);
        if (self.saved_local) |bytes| a.free(bytes);
        if (self.saved_logits) |values| a.free(values);
        self.tokens.deinit(a);
        a.destroy(self);
    }

    pub fn logitValues(self: *Session) []f32 {
        return decode.contents(f32, self.logits, decode.vocab);
    }

    pub fn reset(self: *Session) void {
        // Old append-only global entries are hidden by the new logical length.
        self.native.position = 0;
        self.native.initial_position = 0;
        self.native.usable = true;
        self.tokens.clearRetainingCapacity();
        self.prefill_position = 0;
        self.valid = false;
        self.sampling_seed = null;
    }

    fn prepare(self: *Session, token_count: usize) !void {
        try self.tokens.ensureTotalCapacity(a, token_count);
        if (self.scratch == null) self.scratch = c.flint_gpu_buffer(self.engine.model.gpu, @intCast(self.layout.bytes)) orelse return error.GpuAllocation;
        if (self.saved_local == null) self.saved_local = try a.alloc(u8, local_bytes);
        if (self.saved_logits == null) self.saved_logits = try a.alloc(f32, decode.vocab);
    }

    fn savePrefill(self: *Session) void {
        const kv = decode.contents(u8, self.native.kv[0], global_bytes + local_bytes);
        @memcpy(self.saved_local.?, kv[global_bytes..][0..local_bytes]);
        @memcpy(self.saved_logits.?, self.logitValues());
        self.prefill_position = self.native.position;
    }

    fn restorePrefill(self: *Session) void {
        const kv = decode.contents(u8, self.native.kv[0], global_bytes + local_bytes);
        @memcpy(kv[global_bytes..][0..local_bytes], self.saved_local.?);
        @memcpy(self.logitValues(), self.saved_logits.?);
        self.native.position = self.prefill_position;
        self.native.initial_position = self.prefill_position;
        self.native.usable = true;
        self.tokens.shrinkRetainingCapacity(self.prefill_position);
        self.valid = true;
    }

    pub fn sync(self: *Session, prompt: []const c_int, progress: c.ds4_session_progress_fn, progress_ud: ?*anyopaque, cancel: c.ds4_session_cancel_fn, cancel_ud: ?*anyopaque) !bool {
        if (prompt.len == 0 or prompt.len > self.native.limit) return error.ContextLimit;
        for (prompt) |token| if (token < 0 or token >= decode.vocab) return error.InvalidToken;
        self.sampling_seed = null;
        try self.prepare(prompt.len);
        var common: usize = 0;
        while (common < @min(prompt.len, self.tokens.items.len) and prompt[common] == self.tokens.items[common]) : (common += 1) {}
        if (self.valid and self.native.usable and common == prompt.len and prompt.len == self.tokens.items.len and self.prefill_position == self.native.position)
            return true;
        if (!self.valid or !self.native.usable or common < self.prefill_position) {
            self.reset();
        } else if (self.native.position != self.prefill_position) {
            // Global prefix rows are immutable; only the bounded local rings and
            // logits need restoring before continuation is recomputed.
            self.restorePrefill();
        } else if (common < self.tokens.items.len) {
            self.reset();
        }
        if (progress) |callback| callback(progress_ud, "prefill_chunk", @intCast(self.native.position), @intCast(prompt.len));
        while (self.native.position < prompt.len) {
            if (cancel) |callback| if (callback(cancel_ud)) return false;
            const begin = self.native.position;
            const count: u32 = @intCast(@min(prompt.len - begin, self.layout.capacity));
            try self.native.grow(begin + count);
            const input = decode.contents(u32, self.inputs, count);
            for (input, prompt[begin..][0..count]) |*out, token| out.* = @intCast(token);
            const gpu = self.engine.model.gpu;
            const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
            var bank_views: [13]prefill.View = undefined;
            for (self.engine.model.banks, &bank_views) |buffer, *view| view.* = .{ .handle = buffer };
            var kv_views: [4]prefill.View = undefined;
            for (self.native.kv, &kv_views) |buffer, *view| view.* = .{ .handle = buffer };
            prefill.encode(prefill.Encoder{ .context = @ptrCast(gpu), .command = command }, .{ .banks = bank_views, .metadata = &self.native.metadata }, .{ .banks = kv_views, .initial_position = self.native.initial_position }, self.layout.bind(.{ .handle = self.scratch }), .{ .handle = self.inputs }, begin, count, .{ .handle = self.logits }) catch |err| {
                _ = c.flint_gpu_command_end(gpu, command, true);
                self.native.usable = false;
                self.valid = false;
                return err;
            };
            if (!c.flint_gpu_command_end(gpu, command, true)) {
                self.native.usable = false;
                self.valid = false;
                return error.GpuCommand;
            }
            self.native.position = begin + count;
            self.native.initial_position = self.native.position;
            self.tokens.appendSliceAssumeCapacity(prompt[begin..][0..count]);
            self.valid = true;
            self.savePrefill();
            if (progress) |callback| callback(progress_ud, "prefill_chunk", @intCast(self.native.position), @intCast(prompt.len));
        }
        if (cancel) |callback| if (callback(cancel_ud)) return false;
        return true;
    }

    pub fn eval(self: *Session, token: c_int) !void {
        if (!self.valid or token < 0 or token >= decode.vocab) return error.InvalidToken;
        try self.tokens.ensureUnusedCapacity(a, 1);
        const logits = self.native.forward(@intCast(token)) catch |err| {
            self.valid = false;
            return err;
        };
        for (logits, self.logitValues()) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
        self.tokens.appendAssumeCapacity(token);
    }

    pub fn packet(self: *Session, first: c_int, maximum: u32, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: ?*u64, accepted: []c_int) !u32 {
        if (!self.valid or first < 0 or first >= decode.vocab or maximum == 0 or maximum > decode.packet_width or accepted.len < maximum)
            return error.InvalidPacket;
        const count = @min(maximum, self.native.limit - self.native.position);
        if (count == 0) return error.ContextLimit;
        if (temperature != 0 and (top_k < 0 or top_k > 128 or (top_k == 0 and ((top_p > 0 and top_p < 1) or min_p > 0)))) {
            // Unimplemented sampling configurations keep the ordinary shared
            // CPU sampler and take an honest scalar step.
            try self.eval(first);
            accepted[0] = first;
            return 1;
        }
        try self.tokens.ensureUnusedCapacity(a, count);
        if (self.sampling_seed == null) self.sampling_seed = if (rng) |state| state.* else 0;
        const result = self.native.sampled(@intCast(first), count, .{
            .temperature = temperature,
            .top_k = if (top_k < 0) 0 else @intCast(top_k),
            .top_p = top_p,
            .min_p = min_p,
            .seed = self.sampling_seed.?,
            .eos = &self.engine.stops,
        }) catch |err| {
            self.valid = false;
            return err;
        };
        accepted[0] = first;
        for (1..result.len) |i| accepted[i] = @intCast(result.tokens[i - 1]);
        self.tokens.appendSliceAssumeCapacity(accepted[0..result.len]);
        for (self.native.lastLogits(), self.logitValues()) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
        return result.len;
    }
};

fn getEngine(handle: *c.flint_engine) *Engine {
    return @ptrCast(@alignCast(handle));
}
fn getSession(handle: *c.flint_session) *Session {
    return @ptrCast(@alignCast(handle));
}
fn report(buffer: [*c]u8, size: usize, err: anyerror) void {
    if (buffer == null or size == 0) return;
    const text = @errorName(err);
    const count = @min(text.len, size - 1);
    @memcpy(buffer[0..count], text[0..count]);
    buffer[count] = 0;
}

export fn flint_engine_detect(directory: [*c]const u8) bool {
    if (directory == null) return false;
    const path = std.fmt.allocPrintSentinel(a, "{s}/flint-int4.json", .{std.mem.span(directory)}, 0) catch return false;
    defer a.free(path);
    return c.access(path, c.F_OK) == 0;
}
export fn flint_engine_open(directory: [*c]const u8, chunk: u32, error_text: [*c]u8, error_size: usize) ?*c.flint_engine {
    if (directory == null) return null;
    return @ptrCast(Engine.open(std.mem.span(directory), chunk) catch |err| {
        report(error_text, error_size, err);
        return null;
    });
}
export fn flint_engine_close(handle: ?*c.flint_engine) void {
    if (handle) |value| getEngine(value).close();
}
export fn flint_engine_tokenizer(handle: *c.flint_engine) ?*const c.flint_tokenizer {
    return getEngine(handle).tokenizer;
}
export fn flint_engine_prefill_chunk(handle: *c.flint_engine) u32 {
    return getEngine(handle).chunk;
}
export fn flint_engine_weight_bytes(handle: *c.flint_engine) u64 {
    var bytes: u64 = 0;
    for (getEngine(handle).model.banks) |buffer| bytes += c.flint_gpu_buffer_size(buffer);
    return bytes;
}
export fn flint_engine_context_memory(handle: *c.flint_engine, context: u32) c.ds4_context_memory {
    const engine = getEngine(handle);
    const layout = prefill.Layout.init(@min(engine.chunk, context), context) catch return std.mem.zeroes(c.ds4_context_memory);
    const blocks = (context + decode.block_tokens - 1) / decode.block_tokens;
    const raw = @as(u64, blocks) * global_bytes + local_bytes;
    // Planned per-session allocations, including lazy prefill buffers and the
    // local-ring/logit checkpoint. Weights and the shared SiLU table are separate.
    const scratch = layout.bytes + decode.session_scratch_bytes +
        @as(u64, layout.capacity) * 4 + decode.vocab * 4 +
        local_bytes + decode.vocab * 4 + @as(u64, context) * @sizeOf(c_int);
    return .{
        .total_bytes = raw + scratch,
        .raw_bytes = raw,
        .compressed_bytes = 0,
        .scratch_bytes = scratch,
        .prefill_cap = layout.capacity,
        .raw_cap = blocks * decode.block_tokens,
        .comp_cap = 0,
    };
}
export fn flint_session_create(handle: *c.flint_engine, context: u32) ?*c.flint_session {
    return @ptrCast(Session.create(getEngine(handle), context) catch |err| {
        std.debug.print("flint: create session: {s}\n", .{@errorName(err)});
        return null;
    });
}
export fn flint_session_free(handle: ?*c.flint_session) void {
    if (handle) |value| getSession(value).destroy();
}
export fn flint_session_logits(handle: *c.flint_session) [*c]f32 {
    return getSession(handle).logitValues().ptr;
}
export fn flint_session_position(handle: *c.flint_session) u32 {
    return getSession(handle).native.position;
}
export fn flint_session_prefill_position(handle: *c.flint_session) u32 {
    const session = getSession(handle);
    return if (session.valid and session.native.usable) session.prefill_position else 0;
}
export fn flint_session_sync(handle: *c.flint_session, tokens: [*c]const c.ds4_tokens, progress: c.ds4_session_progress_fn, progress_ud: ?*anyopaque, cancel: c.ds4_session_cancel_fn, cancel_ud: ?*anyopaque, error_text: [*c]u8, error_size: usize) c_int {
    const session = getSession(handle);
    _ = c.pthread_mutex_lock(&session.engine.mutex);
    defer _ = c.pthread_mutex_unlock(&session.engine.mutex);
    if (tokens == null or tokens.*.len <= 0 or tokens.*.v == null) {
        report(error_text, error_size, error.InvalidToken);
        return 1;
    }
    const complete = session.sync(tokens.*.v[0..@intCast(tokens.*.len)], progress, progress_ud, cancel, cancel_ud) catch |err| {
        report(error_text, error_size, err);
        return 1;
    };
    return if (complete) 0 else c.DS4_SESSION_SYNC_INTERRUPTED;
}
export fn flint_session_eval(handle: *c.flint_session, token: c_int, error_text: [*c]u8, error_size: usize) c_int {
    const session = getSession(handle);
    _ = c.pthread_mutex_lock(&session.engine.mutex);
    defer _ = c.pthread_mutex_unlock(&session.engine.mutex);
    session.eval(token) catch |err| {
        report(error_text, error_size, err);
        return 1;
    };
    return 0;
}
export fn flint_session_packet(handle: *c.flint_session, first: c_int, maximum: u32, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, accepted: [*c]c_int, error_text: [*c]u8, error_size: usize) c_int {
    const session = getSession(handle);
    _ = c.pthread_mutex_lock(&session.engine.mutex);
    defer _ = c.pthread_mutex_unlock(&session.engine.mutex);
    if (accepted == null or maximum > decode.packet_width) {
        report(error_text, error_size, error.InvalidPacket);
        return -1;
    }
    const count = session.packet(first, maximum, temperature, top_k, top_p, min_p, if (rng == null) null else @ptrCast(rng), accepted[0..maximum]) catch |err| {
        report(error_text, error_size, err);
        return -1;
    };
    return @intCast(count);
}
export fn flint_session_reset(handle: *c.flint_session) void {
    const session = getSession(handle);
    _ = c.pthread_mutex_lock(&session.engine.mutex);
    defer _ = c.pthread_mutex_unlock(&session.engine.mutex);
    session.reset();
}
