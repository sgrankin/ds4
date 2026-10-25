//! One exact text prefix per dense Clef engine. Questions remain fresh: the
//! retained boundary precedes the schema and the ordinary learned head runs
//! over retained prefix rows plus newly computed suffix rows.
//!
//! The causal state layout and attention offset contract follow Cleffa's MIT
//! implementation (zknpr/cleffa, 24fc91e); see src/metal/clef/LICENSE.cleffa.
//! This deliberately has no multi-entry policy, media identity or checkpoints.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const a = std.heap.c_allocator;
const View = b.View;

const Rows = extern struct { rows: u32, width: u32 };
const Norm = extern struct { width: u32, epsilon: f32 };
const AttentionPrep = extern struct { heads: u32, kv: u32, dim: u32, rotary: u32, row: u32, epsilon: f32, capacity: u32, offset: u32 };
const Attention = extern struct { heads: u32, kv: u32, dim: u32, rows: u32, scale: f32, capacity: u32, offset: u32 };
const Conv = extern struct { channels: u32, row: u32, kernel: u32 };
const Tail = extern struct { channels: u32, row: u32, first: u32 };
const GdnPrep = extern struct { hk: u32, hv: u32, dk: u32, dv: u32, row: u32, channels: u32 };
const Gdn = extern struct { hk: u32, hv: u32, dk: u32, dv: u32, channels: u32 };
const GdnOut = extern struct { hv: u32, dv: u32, row: u32, offset: u32, epsilon: f32 };

const Layer = union(enum) {
    attention: struct { key: View, value: View },
    delta: struct { state: View, tail: View },
};

/// Align to GEMM's 64-row tiles, attention's 32-query tiles and GDN blocks.
/// Requiring 256 prefix rows also keeps all eligible records in the ordinary
/// chunked-GDN class, even when a hit computes fewer than 256 new rows.
pub fn boundary(schema_start: usize) u32 {
    const rows = schema_start / 64 * 64;
    return if (rows >= 256 and rows <= 16384) @intCast(rows) else 0;
}

pub const State = struct {
    gpu: *c.flint_gpu,
    capacity: u32 = 0,
    length: u32 = 0,
    ids: []u32 = &.{},
    hidden: []f32 = &.{},
    layers: []Layer = &.{},
    buffers: std.ArrayList(b.Buffer) = .empty,
    save: View = undefined,
    load: View = undefined,
    prefix_length: View = undefined,
    suffix_length: View = undefined,
    last_reused: u32 = 0,

    pub fn open(engine: *b.Engine) !State {
        try engine.compile(@embedFile("../../metal/clef/prefix_attention.metal"));
        for ([_][:0]const u8{
            "clef_attn_prep_prefix",   "clef_attention_prefix_64",
            "ssm_conv_tail",           "ssm_tail_save",
            "gdn_chunk_scan_st_32_32",
        }) |name| if (!c.flint_gpu_prepare(engine.gpu, name)) return error.GpuPipeline;
        return .{ .gpu = engine.gpu };
    }

    pub fn clear(self: *State) void {
        self.length = 0;
        self.last_reused = 0;
    }

    pub fn deinit(self: *State) void {
        for (self.buffers.items) |buffer| c.flint_gpu_buffer_free(buffer);
        self.buffers.deinit(a);
        a.free(self.layers);
        a.free(self.ids);
        a.free(self.hidden);
        self.* = .{ .gpu = self.gpu };
    }

    pub fn bytes(self: *const State, cfg: b.Config) usize {
        if (self.capacity == 0) return 0;
        var total = self.hidden.len * 4 + self.ids.len * 4 + 24;
        for (cfg.full) |full| {
            total += if (full)
                2 * (@as(usize, cfg.kv_heads) * self.capacity + 64) * cfg.head_dim * 4
            else
                (@as(usize, cfg.Hv) * cfg.dk * cfg.dv + (cfg.conv - 1) * channels(cfg)) * 4;
        }
        return total;
    }

    fn alloc(self: *State, size: usize) !View {
        const buffer = c.flint_gpu_buffer(self.gpu, size) orelse return error.GpuAllocation;
        self.buffers.append(a, buffer) catch |err| {
            c.flint_gpu_buffer_free(buffer);
            return err;
        };
        return .{ .buffer = buffer };
    }

    fn reserve(self: *State, cfg: b.Config, rows: u32) !void {
        if (self.capacity >= rows) return;
        // Leave room for a modest schema edit without discarding a valid state
        // prefix just because its fresh questions occupy a few more rows.
        const capacity = @min(16384, std.mem.alignForward(u32, rows + 512, 64));
        var next = State{ .gpu = self.gpu, .capacity = capacity };
        errdefer next.deinit();
        next.layers = try a.alloc(Layer, cfg.layers);
        next.ids = try a.alloc(u32, next.capacity);
        next.hidden = try a.alloc(f32, @as(usize, next.capacity) * cfg.H);
        next.save = try next.alloc(8);
        next.load = try next.alloc(8);
        next.prefix_length = try next.alloc(4);
        next.suffix_length = try next.alloc(4);
        @memcpy(b.contents(i32, next.save, 2), &[_]i32{ 0, 1 });
        @memcpy(b.contents(i32, next.load, 2), &[_]i32{ 1, 0 });
        for (cfg.full, next.layers) |full, *layer| {
            if (full) {
                const size = (@as(usize, cfg.kv_heads) * next.capacity + 64) * cfg.head_dim * 4;
                const key = try next.alloc(size);
                const value = try next.alloc(size);
                @memset(b.contents(u8, key, size), 0);
                @memset(b.contents(u8, value, size), 0);
                layer.* = .{ .attention = .{ .key = key, .value = value } };
            } else {
                layer.* = .{ .delta = .{
                    .state = try next.alloc(@as(usize, cfg.Hv) * cfg.dk * cfg.dv * 4),
                    .tail = try next.alloc(@as(usize, cfg.conv - 1) * channels(cfg) * 4),
                } };
            }
        }
        self.deinit();
        self.* = next;
    }

    /// Input is text only. Returned rows are owned by this entry and borrowed
    /// until its next infer/reserve; the caller holds the engine's mutex.
    pub fn infer(self: *State, engine: *b.Engine, ids: []const u32, schema_start: usize, cancel: b.Cancel, context: ?*anyopaque) ![]const f32 {
        const cfg = engine.config;
        const wanted = boundary(schema_start);
        if (cfg.family != .dense or wanted == 0 or wanted >= ids.len or ids.len > 16384) return error.InvalidPrefixBoundary;
        for (ids) |id| if (id >= cfg.vocab) return error.TokenOutOfRange;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        try self.reserve(cfg, @intCast(ids.len));
        const hit = self.length != 0 and self.length <= wanted and
            std.mem.eql(u32, self.ids[0..self.length], ids[0..self.length]);
        const reused = if (hit) self.length else 0;
        const snapshot = if (hit) self.length else wanted;
        const total: u32 = @intCast(ids.len);
        const T = total - reused;
        self.clear();
        errdefer self.clear();
        const s = try engine.scratchFor(T);
        @memcpy(b.contents(u32, s.ids, T), ids[reused..]);
        const position = b.contents(u32, s.position, @as(usize, T) * 3);
        for (0..T) |row| @memset(position[row * 3 ..][0..3], @as(u32, @intCast(row)) + reused);
        @memset(b.contents(u32, s.starts, T), 0);
        @memset(b.contents(u32, s.overflow, T), 0);
        b.contents(u32, s.length, 1)[0] = T;
        b.contents(u32, self.prefix_length, 1)[0] = snapshot;
        b.contents(u32, self.suffix_length, 1)[0] = if (hit) T else T - snapshot;
        self.clearTail(cfg, total);
        // Any speculative final-head Q loads must remain finite.
        @memset(b.contents(u8, s.query.at(@as(usize, cfg.heads) * T * cfg.head_dim * 4), 8 * cfg.head_dim * 4), 0);
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var e = b.Encoder{ .gpu = engine.gpu, .command = command };
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, e.command, true);
        };
        try e.dispatch("clef_embed", extern struct { rows: u32, width: u32, bits: u32, pad: u32 }{ .rows = T, .width = cfg.H, .bits = engine.embedding.bits, .pad = 0 }, &.{ s.ids.bind(1), engine.embedding.w.bind(2), engine.embedding.scales.bind(3), engine.embedding.biases.bind(4), s.hidden.bind(5) }, .{ ceil(cfg.H, 256), T, 1 }, 256);
        for (engine.layer, self.layers, 0..) |layer, retained, index| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try engine.norm(e, layer.norm, s.hidden, s.normal, T);
            try engine.linear(e, layer.in, s.normal, s.project16, T);
            try e.dispatch("clef_widen", Rows{ .rows = T, .width = layer.in.rows }, &.{ s.project16.bind(1), s.project.bind(2) }, .{ ceil(T * layer.in.rows, 256), 1, 1 }, 256);
            switch (retained) {
                .attention => |kv| {
                    try e.dispatch("clef_attn_prep_prefix", AttentionPrep{ .heads = cfg.heads, .kv = cfg.kv_heads, .dim = cfg.head_dim, .rotary = cfg.rotary, .row = layer.in.rows, .epsilon = cfg.eps, .capacity = self.capacity, .offset = reused }, &.{ s.project.bind(1), layer.qnorm.bind(2), layer.knorm.bind(3), s.position.bind(4), engine.inv_freq.bind(5), s.query.bind(6), kv.key.bind(7), kv.value.bind(8), s.gate.bind(9), s.length.bind(10) }, .{ T, cfg.heads + cfg.kv_heads, 1 }, 32);
                    try e.dispatch("clef_attention_prefix_64", Attention{ .heads = cfg.heads, .kv = cfg.kv_heads, .dim = cfg.head_dim, .rows = T, .scale = 1 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))), .capacity = self.capacity, .offset = reused }, &.{ s.query.bind(1), kv.key.bind(2), kv.value.bind(3), s.gate.bind(4), s.starts.bind(5), s.activation.bind(6), s.act.bind(7), s.overflow.bind(8) }, .{ ceil(T, 32), cfg.heads, 1 }, 128);
                },
                .delta => |delta| {
                    const cp = Conv{ .channels = channels(cfg), .row = layer.in.rows, .kernel = cfg.conv };
                    if (hit) {
                        try e.dispatch("ssm_conv_tail", cp, &.{ s.project.bind(1), layer.conv.bind(2), delta.tail.bind(3), s.convolution.bind(4) }, .{ ceil(channels(cfg), 256), T, 1 }, 256);
                    } else {
                        try e.dispatch("ssm_conv", cp, &.{ s.project.bind(1), layer.conv.bind(2), s.starts.bind(3), s.convolution.bind(4) }, .{ ceil(channels(cfg), 256), T, 1 }, 256);
                        try e.dispatch("ssm_tail_save", Tail{ .channels = channels(cfg), .row = layer.in.rows, .first = snapshot - (cfg.conv - 1) }, &.{ s.project.bind(1), delta.tail.bind(2) }, .{ ceil(channels(cfg), 256), cfg.conv - 1, 1 }, 256);
                    }
                    try e.dispatch("gdn_prep", GdnPrep{ .hk = cfg.Hk, .hv = cfg.Hv, .dk = cfg.dk, .dv = cfg.dv, .row = layer.in.rows, .channels = channels(cfg) }, &.{ s.convolution.bind(1), s.project.bind(2), layer.decay.bind(3), layer.dt.bind(4), s.beta.bind(5), s.decay.bind(6) }, .{ T, 2 * cfg.Hk + cfg.Hv, 1 }, 32);
                    if (hit) {
                        try self.scan(e, cfg, s, delta.state, 0, T, self.suffix_length, self.load);
                    } else {
                        try self.scan(e, cfg, s, delta.state, 0, snapshot, self.prefix_length, self.save);
                        try self.scan(e, cfg, s, delta.state, snapshot, T - snapshot, self.suffix_length, self.load);
                    }
                    try e.dispatch("gdn_out", GdnOut{ .hv = cfg.Hv, .dv = cfg.dv, .row = layer.in.rows, .offset = channels(cfg), .epsilon = cfg.eps }, &.{ s.output.bind(1), s.project.bind(2), layer.gated_norm.bind(3), s.activation.bind(4), s.act.bind(5), s.starts.bind(6), s.overflow.bind(7) }, .{ T, cfg.Hv, 1 }, 32);
                },
            }
            try engine.linear(e, layer.out, s.activation, s.project16, T);
            try e.dispatch("clef_residual", Rows{ .rows = T, .width = cfg.H }, &.{ s.hidden.bind(1), s.project16.bind(2) }, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
            try engine.norm(e, layer.ffn_norm, s.hidden, s.normal, T);
            try engine.linear(e, layer.gate_up, s.normal, s.project16, T);
            if (!engine.ffn_bf16) try e.dispatch("clef_widen", Rows{ .rows = T, .width = layer.gate_up.rows }, &.{ s.project16.bind(1), s.project.bind(2) }, .{ ceil(T * layer.gate_up.rows, 256), 1, 1 }, 256);
            try e.dispatch(if (engine.ffn_bf16) "clef_swiglu_bf16" else "swiglu", extern struct { width: u32 }{ .width = cfg.ffn }, &.{ (if (engine.ffn_bf16) s.project16 else s.project).bind(1), s.activation.bind(2), s.act.bind(3), s.starts.bind(4), s.overflow.bind(5) }, .{ ceil(cfg.ffn, 256), T, 1 }, 256);
            try engine.linear(e, layer.down, s.activation, s.project16, T);
            try e.dispatch("clef_residual", Rows{ .rows = T, .width = cfg.H }, &.{ s.hidden.bind(1), s.project16.bind(2) }, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
            try engine.yieldAfterLayer(&e, index, engine.layer.len, &pending, cancel, context);
        }
        try e.dispatch("rmsnorm_f32", Norm{ .width = cfg.H, .epsilon = cfg.eps }, &.{ s.hidden.bind(1), engine.final_norm.bind(2), s.final.bind(3) }, .{ T, 1, 1 }, 256);
        try e.dispatch("clef_round", Rows{ .rows = T, .width = cfg.H }, &.{s.final.bind(1)}, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
        pending = false;
        if (!c.flint_gpu_command_end(engine.gpu, e.command, true)) return error.GpuExecution;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        @memcpy(self.hidden[@as(usize, reused) * cfg.H ..][0 .. @as(usize, T) * cfg.H], b.contents(f32, s.final, @as(usize, T) * cfg.H));
        @memcpy(self.ids[0..snapshot], ids[0..snapshot]);
        self.length = snapshot;
        self.last_reused = reused;
        return self.hidden[0 .. @as(usize, total) * cfg.H];
    }

    fn scan(self: *State, e: b.Encoder, cfg: b.Config, s: *b.Scratch, state: View, first: u32, rows: u32, length: View, flags: View) !void {
        _ = self;
        const params = Gdn{ .hk = cfg.Hk, .hv = cfg.Hv, .dk = cfg.dk, .dv = cfg.dv, .channels = channels(cfg) };
        const binds = &.{
            s.convolution.at(@as(usize, first) * channels(cfg) * 4).bind(1),
            s.beta.at(@as(usize, first) * cfg.Hv * 4).bind(2),
            s.decay.at(@as(usize, first) * cfg.Hv * 4).bind(3),
            s.output.at(@as(usize, first) * cfg.Hv * cfg.dv * 4).bind(5),
            s.gdn_w.bind(6),
            s.gdn_u.bind(7),
            s.gdn_ke.bind(8),
            s.gdn_a.bind(9),
            s.gdn_e.bind(10),
            length.bind(11),
            state.bind(12),
            flags.bind(13),
            state.bind(14),
        };
        try e.dispatch("gdn_chunk_prep_32", params, binds, .{ ceil(rows, 32), cfg.Hv, 1 }, 128);
        try e.dispatch("gdn_chunk_scan_st_32_32", params, binds, .{ cfg.Hv, 4, 1 }, 128);
    }

    fn clearTail(self: *State, cfg: b.Config, end: u32) void {
        const count = @min(64, self.capacity - end);
        for (self.layers) |layer| switch (layer) {
            .attention => |kv| for ([_]View{ kv.key, kv.value }) |view| {
                for (0..cfg.kv_heads) |head| {
                    const offset = (head * self.capacity + end) * cfg.head_dim * 4;
                    @memset(b.contents(u8, view.at(offset), @as(usize, count) * cfg.head_dim * 4), 0);
                }
                @memset(b.contents(u8, view.at(@as(usize, cfg.kv_heads) * self.capacity * cfg.head_dim * 4), 64 * cfg.head_dim * 4), 0);
            },
            .delta => {},
        };
    }
};

fn channels(cfg: b.Config) u32 {
    return 2 * cfg.Hk * cfg.dk + cfg.Hv * cfg.dv;
}
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}
