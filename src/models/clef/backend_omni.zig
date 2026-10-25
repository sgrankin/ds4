//! Clef-Omni's sparse thinker. This owns its graph and scratch; the shared
//! catalogue/affine layout and device bridge are the same as dense Clef.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const Prefill = @import("../flint/prefill.zig");
const Media = @import("backend_media.zig");
const a = std.heap.c_allocator;
const View = b.View;
const Matrix = b.Matrix;
const Rows = extern struct { rows: u32, width: u32 };
const Matmul = extern struct { m: i32, n: i32, k: i32, pad: i32 = 0 };
const Norm = extern struct { width: u32, epsilon: f32 };
const Route = extern struct { rows: u32, experts: u32 = 128, top_k: u32 = 8, width: u32 = 2048 };
const Prepare = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32 = 128, eps: f32 };
const Tiles = extern struct { rows: u32, tile_rows: u32 = 32, tiles: u32, experts: u32 = 128 };

const Layer = struct {
    norm: View,
    ffn_norm: View,
    query_norm: View,
    key_norm: View,
    qkv: Matrix,
    out: Matrix,
    router: Matrix,
    gate_up: Matrix,
    down: Matrix,
};
pub const State = struct {
    layers: []Layer,
    scratch: ?Scratch = null,

    pub fn open(engine: *b.Engine) !State {
        const cfg = engine.config;
        const layers = try a.alloc(Layer, cfg.layers);
        errdefer a.free(layers);
        for (layers, 0..) |*layer, i| {
            var storage: [256]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&storage, "thinker.language_model.model.layers.{d}.", .{i});
            const input_norm = try key(prefix, "input_layernorm.weight");
            defer a.free(input_norm);
            const ffn_norm = try key(prefix, "post_attention_layernorm.weight");
            defer a.free(ffn_norm);
            const query_norm = try key(prefix, "self_attn.q_norm.weight");
            defer a.free(query_norm);
            const key_norm = try key(prefix, "self_attn.k_norm.weight");
            defer a.free(key_norm);
            layer.norm = try engine.vector(input_norm, 0, false);
            layer.ffn_norm = try engine.vector(ffn_norm, 0, false);
            layer.query_norm = try engine.vector(query_norm, 0, false);
            layer.key_norm = try engine.vector(key_norm, 0, false);
            const q = try key(prefix, "self_attn.q_proj");
            defer a.free(q);
            const k = try key(prefix, "self_attn.k_proj");
            defer a.free(k);
            const v = try key(prefix, "self_attn.v_proj");
            defer a.free(v);
            layer.qkv = try engine.matrix(&.{ q, k, v });
            const out = try key(prefix, "self_attn.o_proj");
            defer a.free(out);
            layer.out = try engine.matrix(&.{out});
            const router = try key(prefix, "mlp.gate");
            defer a.free(router);
            layer.router = try engine.matrix(&.{router});
            const gate = try key(prefix, "mlp.switch_mlp.gate_proj");
            defer a.free(gate);
            const up = try key(prefix, "mlp.switch_mlp.up_proj");
            defer a.free(up);
            layer.gate_up = try engine.matrix(&.{ gate, up });
            const down = try key(prefix, "mlp.switch_mlp.down_proj");
            defer a.free(down);
            layer.down = try engine.matrix(&.{down});
            if (layer.router.rows != 128 or layer.gate_up.experts != 128 or layer.down.experts != 128 or
                layer.gate_up.bits != 4 or layer.down.bits != 4)
                return error.UnsupportedExperts;
        }
        return .{ .layers = layers };
    }
    pub fn deinit(self: *State) void {
        if (self.scratch) |*scratch| scratch.deinit();
        a.free(self.layers);
    }
    pub fn infer(self: *State, engine: *b.Engine, ids: []const u32, positions: []const u32, injections: []const b.Injection, cancel: b.Cancel, context: ?*anyopaque) ![]const f32 {
        const cfg = engine.config;
        if (ids.len == 0 or ids.len > 64000) return error.InvalidContextLength;
        for (ids) |id| if (id >= cfg.vocab) return error.TokenOutOfRange;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        const T: u32 = @intCast(ids.len);
        if (self.scratch == null or self.scratch.?.capacity < T) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(engine.gpu, cfg, T);
        }
        const s = &self.scratch.?;
        @memcpy(b.contents(u32, s.ids, T), ids);
        if (positions.len != 0) {
            @memcpy(b.contents(u32, s.position, T * 3), positions);
        } else {
            const position = b.contents(i32, s.position, T * 3);
            for (0..T) |i| for (0..3) |axis| {
                position[i * 3 + axis] = @intCast(i);
            };
        }
        var media = try Media.Prepared.init(engine.gpu, injections, T, cfg.H);
        defer media.deinit();
        @memset(b.contents(u32, s.starts, T), 0);
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var e = b.Encoder{ .gpu = engine.gpu, .command = command };
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, e.command, true);
        };
        const tile_setting = c.getenv("DS4_CLEF_OMNI_EXPERT_TILES");
        const tiled = tile_setting == null or std.mem.eql(u8, std.mem.span(tile_setting), "1");
        const tile_rows: u32 = if (T * 8 / 128 < 64) 32 else 64;
        const tile_count = @min(T * 8, ceil(T * 8, tile_rows) + 127);
        const trace_path = c.getenv("DS4_CLEF_TRACE_LAYERS");
        const trace_rows = @min(T, 16);
        if (trace_path != null and s.trace == null) s.trace = try s.alloc(@as(usize, trace_rows) * cfg.H * (cfg.layers + 2) * 4);
        try e.dispatch("clef_embed", extern struct { rows: u32, width: u32, bits: u32, pad: u32 }{
            .rows = T,
            .width = cfg.H,
            .bits = engine.embedding.bits,
            .pad = 0,
        }, &.{ s.ids.bind(1), engine.embedding.w.bind(2), engine.embedding.scales.bind(3), engine.embedding.biases.bind(4), s.hidden.bind(5) }, .{ ceil(cfg.H, 256), T, 1 }, 256);
        try media.inject(e, s.hidden, cfg.H, null);
        if (trace_path != null) try capture(e, s, 0, T, trace_rows, cfg.H, s.hidden);
        for (self.layers, 0..) |layer, index| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try norm(e, cfg, s, layer.norm, T);
            try dense(e, layer.qkv, s.normal, s.qkv, T);
            try e.dispatch("clef_omni_prepare", Prepare{ .rows = T, .heads = cfg.heads, .kv_heads = cfg.kv_heads, .eps = cfg.eps }, &.{
                s.qkv.bind(1), layer.query_norm.bind(2), layer.key_norm.bind(3), s.position.bind(4), engine.inv_freq.bind(5), s.query.bind(6), s.key.bind(7), s.value.bind(8),
            }, .{ T, cfg.heads + cfg.kv_heads, 1 }, 32);
            const attention = Prefill.Attention{ .base = .{
                .heads = @intCast(cfg.heads),
                .dim = 128,
                .queries = @intCast(T),
                .keys = @intCast(T),
                .gqa_factor = @intCast(cfg.heads / cfg.kv_heads),
                .scale = 1 / @sqrt(@as(f32, 128)),
                .query_blocks = @intCast(ceil(T, 64)),
                .key_blocks = @intCast(ceil(T, 64)),
                .query_aligned = @intCast(T / 64),
                .key_aligned = @intCast(T / 64),
                .query_remainder = @intCast(T % 64),
                .key_remainder = @intCast(T % 64),
                .query_offset = 0,
                .query_strides = .{ @as(i64, T) * cfg.heads * 128, @as(i64, T) * 128, 128 },
                .key_strides = .{ @as(i64, T) * cfg.kv_heads * 128, @as(i64, T) * 128, 128 },
                .value_strides = .{ @as(i64, T) * cfg.kv_heads * 128, @as(i64, T) * 128, 128 },
                .output_strides = .{ @as(i64, T) * cfg.heads * 128, 128, @as(i64, cfg.heads) * 128 },
            }, .window = 0 };
            try e.dispatch("clef_omni_attention", attention, &.{ s.query.bind(1), s.key.bind(2), s.value.bind(3), s.attention.bind(4), s.dummy.bind(5) }, .{ ceil(T, 64), cfg.heads, 1 }, 128);
            try dense(e, layer.out, s.attention, s.projection, T);
            try residual(e, s.hidden, s.projection, T, cfg.H);
            try norm(e, cfg, s, layer.ffn_norm, T);
            try dense(e, layer.router, s.normal, s.router, T);
            try e.dispatch("clef_omni_route_top8", Route{ .rows = T }, &.{ s.router.bind(1), s.expert_ids.bind(2), s.weights.bind(3) }, .{ T, 1, 1 }, 32);
            try e.dispatch("flint_prefill_zero_counts", Route{ .rows = T }, &.{s.counts.bind(1)}, .{ 1, 1, 1 }, 256);
            try e.dispatch("clef_omni_histogram", Route{ .rows = T }, &.{ s.expert_ids.bind(1), s.counts.bind(2) }, .{ ceil(T * 8, 256), 1, 1 }, 256);
            try e.dispatch("flint_prefill_offsets", Route{ .rows = T }, &.{ s.counts.bind(1), s.offsets.bind(2), s.cursors.bind(3) }, .{ 1, 1, 1 }, 256);
            try e.dispatch("clef_omni_scatter", Route{ .rows = T }, &.{ s.normal.bind(1), s.expert_ids.bind(2), s.offsets.bind(3), s.cursors.bind(4), s.permutation.bind(5), s.sorted_ids.bind(6), s.sorted_input.bind(7) }, .{ T * 8, 1, 1 }, 128);
            if (tiled) try e.dispatch("clef_omni_expert_tiles", Tiles{ .rows = T * 8, .tile_rows = tile_rows, .tiles = tile_count }, &.{ s.counts.bind(1), s.offsets.bind(2), s.expert_tiles.bind(3) }, .{ 1, 1, 1 }, 128);
            try gather(e, layer.gate_up, s.sorted_input, s.gate_up, s.sorted_ids, T * 8, if (tiled) s.expert_tiles else null, tile_count, tile_rows);
            try e.dispatch("clef_omni_swiglu", Rows{ .rows = T * 8, .width = cfg.ffn }, &.{ s.gate_up.bind(1), s.activation.bind(2) }, .{ ceil(T * 8 * cfg.ffn, 256), 1, 1 }, 256);
            try gather(e, layer.down, s.activation, s.expert_output, s.sorted_ids, T * 8, if (tiled) s.expert_tiles else null, tile_count, tile_rows);
            try e.dispatch("clef_omni_combine", Rows{ .rows = T, .width = cfg.H }, &.{ s.expert_output.bind(1), s.permutation.bind(2), s.weights.bind(3), s.hidden.bind(4) }, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
            if (index < 3) try media.inject(e, s.hidden, cfg.H, @intCast(index));
            if (trace_path != null) try capture(e, s, @intCast(index + 1), T, trace_rows, cfg.H, s.hidden);
            try engine.yieldAfterLayer(&e, index, self.layers.len, &pending, cancel, context);
        }
        try e.dispatch("rmsnorm_f32", Norm{ .width = cfg.H, .epsilon = cfg.eps }, &.{ s.hidden.bind(1), engine.final_norm.bind(2), s.final.bind(3) }, .{ T, 1, 1 }, 256);
        try e.dispatch("clef_round", Rows{ .rows = T, .width = cfg.H }, &.{s.final.bind(1)}, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
        if (trace_path != null) try capture(e, s, cfg.layers + 1, T, trace_rows, cfg.H, s.final);
        pending = false;
        if (!c.flint_gpu_command_end(engine.gpu, e.command, true)) return error.GpuExecution;
        if (trace_path != null) {
            const file = c.fopen(trace_path, "wb") orelse return error.WriteTrace;
            defer _ = c.fclose(file);
            const values = b.contents(f32, s.trace.?, @as(usize, trace_rows) * cfg.H * (cfg.layers + 2));
            if (c.fwrite(values.ptr, 4, values.len, file) != values.len) return error.WriteTrace;
        }
        if (cancel) |check| if (check(context)) return error.Cancelled;
        return b.contents(f32, s.final, @as(usize, T) * cfg.H);
    }
};
fn capture(e: b.Encoder, s: *Scratch, stage: u32, T: u32, R: u32, H: u32, input: View) !void {
    const source = input.at(@as(usize, T - R) * H * 4);
    const output = s.trace.?.at(@as(usize, stage) * R * H * 4);
    try e.dispatch("clef_trace_rows", Rows{ .rows = R, .width = H }, &.{ source.bind(1), output.bind(2) }, .{ ceil(R * H, 256), 1, 1 }, 256);
}
fn key(prefix: []const u8, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
}
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}
fn norm(e: b.Encoder, cfg: b.Config, s: *const Scratch, weight: View, T: u32) !void {
    try e.dispatch("rmsnorm_act", Norm{ .width = cfg.H, .epsilon = cfg.eps }, &.{ s.hidden.bind(1), weight.bind(2), s.normal.bind(3), s.act.bind(4), s.starts.bind(5), s.dummy.bind(6) }, .{ T, 1, 1 }, 256);
}
fn residual(e: b.Encoder, hidden: View, projection: View, T: u32, H: u32) !void {
    try e.dispatch("clef_residual", Rows{ .rows = T, .width = H }, &.{ hidden.bind(1), projection.bind(2) }, .{ ceil(T * H, 256), 1, 1 }, 256);
}
fn dense(e: b.Encoder, m: Matrix, input: View, out: View, T: u32) !void {
    const narrow = T <= 8 and m.bits == 4;
    const name: [:0]const u8 = if (m.bits == 8) "flint_prefill_qmm8" else if (narrow) "flint_prefill_qmv4_packed" else "flint_prefill_qmm4";
    try e.dispatch(name, Matmul{ .m = @intCast(T), .n = @intCast(m.rows), .k = @intCast(m.columns) }, &.{ m.w.bind(1), m.scales.bind(2), m.biases.bind(3), input.bind(4), out.bind(5) }, if (narrow) .{ ceil(m.rows, 16), T, 1 } else .{ ceil(m.rows, 64), ceil(T, 64), 1 }, 128);
}
fn gather(e: b.Encoder, m: Matrix, input: View, out: View, ids: View, T: u32, tiles: ?View, tile_count: u32, tile_rows: u32) !void {
    // Flint's packed vector kernel reads complete 512-column blocks. Omni's
    // FFN down projection has K=768, so it must use a matrix path even when
    // the routed batch is small. Otherwise it reads into the next sorted row.
    const narrow = T <= 768 and m.columns % 512 == 0;
    if (tiles != null and !narrow) {
        const map = tiles.?;
        const name: [:0]const u8 = if (tile_rows == 32) "clef_omni_expert_qmm32" else "clef_omni_expert_qmm64";
        try e.dispatch(name, Matmul{ .m = @intCast(T), .n = @intCast(m.rows), .k = @intCast(m.columns) }, &.{ input.bind(1), m.w.bind(2), m.scales.bind(3), m.biases.bind(4), map.bind(5), out.bind(6) }, .{ ceil(m.rows, 64), tile_count, 1 }, 128);
        return;
    }
    const name: [:0]const u8 = if (narrow) "flint_prefill_gather_qmv4_packed" else "flint_prefill_gather_qmm4";
    try e.dispatch(name, Matmul{ .m = @intCast(T), .n = @intCast(m.rows), .k = @intCast(m.columns) }, &.{ input.bind(1), m.w.bind(2), m.scales.bind(3), m.biases.bind(4), ids.bind(5), out.bind(6) }, if (narrow) .{ ceil(m.rows, 16), T, 1 } else .{ ceil(m.rows, 64), ceil(T, 64), 1 }, 128);
}
const Scratch = struct {
    gpu: *c.flint_gpu,
    capacity: u32,
    buffers: std.ArrayList(b.Buffer) = .empty,
    ids: View = undefined,
    position: View = undefined,
    starts: View = undefined,
    hidden: View = undefined,
    normal: View = undefined,
    qkv: View = undefined,
    query: View = undefined,
    key: View = undefined,
    value: View = undefined,
    attention: View = undefined,
    projection: View = undefined,
    router: View = undefined,
    expert_ids: View = undefined,
    weights: View = undefined,
    counts: View = undefined,
    offsets: View = undefined,
    cursors: View = undefined,
    permutation: View = undefined,
    sorted_ids: View = undefined,
    expert_tiles: View = undefined,
    sorted_input: View = undefined,
    gate_up: View = undefined,
    activation: View = undefined,
    expert_output: View = undefined,
    final: View = undefined,
    act: View = undefined,
    dummy: View = undefined,
    trace: ?View = null,
    fn alloc(self: *Scratch, size: usize) !View {
        const buffer = c.flint_gpu_buffer(self.gpu, size) orelse return error.GpuAllocation;
        self.buffers.append(a, buffer) catch |err| {
            c.flint_gpu_buffer_free(buffer);
            return err;
        };
        return .{ .buffer = buffer };
    }
    fn init(gpu: *c.flint_gpu, cfg: b.Config, T: u32) !Scratch {
        var s = Scratch{ .gpu = gpu, .capacity = T };
        errdefer s.deinit();
        const rows: usize = T;
        s.ids = try s.alloc(rows * 4);
        s.position = try s.alloc(rows * 3 * 4);
        s.starts = try s.alloc(rows * 4);
        s.hidden = try s.alloc(rows * cfg.H * 4);
        s.normal = try s.alloc(rows * cfg.H * 2);
        s.qkv = try s.alloc(rows * (cfg.heads + 2 * cfg.kv_heads) * 128 * 2);
        s.query = try s.alloc(rows * cfg.heads * 128 * 2);
        s.key = try s.alloc(rows * cfg.kv_heads * 128 * 2);
        s.value = try s.alloc(rows * cfg.kv_heads * 128 * 2);
        s.attention = try s.alloc(rows * cfg.heads * 128 * 2);
        s.projection = try s.alloc(rows * cfg.H * 2);
        s.router = try s.alloc(rows * 128 * 2);
        s.expert_ids = try s.alloc(rows * 8 * 4);
        s.weights = try s.alloc(rows * 8 * 4);
        s.counts = try s.alloc(256 * 4);
        s.offsets = try s.alloc(256 * 4);
        s.cursors = try s.alloc(256 * 4);
        s.permutation = try s.alloc(rows * 8 * 4);
        s.sorted_ids = try s.alloc(rows * 8 * 4);
        s.expert_tiles = try s.alloc(@min(rows * 8, ceil(T * 8, 32) + 127) * 16);
        s.sorted_input = try s.alloc(rows * 8 * cfg.H * 2);
        s.gate_up = try s.alloc(rows * 8 * 2 * cfg.ffn * 2);
        s.activation = try s.alloc(rows * 8 * cfg.ffn * 2);
        s.expert_output = try s.alloc(rows * 8 * cfg.H * 2);
        s.final = try s.alloc(rows * cfg.H * 4);
        s.act = try s.alloc(8);
        @memset(b.contents(u8, s.act, 8), 0);
        s.dummy = try s.alloc(16);
        @memset(b.contents(u8, s.dummy, 16), 0);
        return s;
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |buffer| c.flint_gpu_buffer_free(buffer);
        self.buffers.deinit(a);
    }
};
