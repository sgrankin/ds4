//! Native, chunked Flint prefill. One command contains the whole chunk; no
//! tensor readback, MLX graph or Python callback occurs between operations.
const std = @import("std");
pub const shaders = @import("prefill_shaders.zig");

pub const View = extern struct {
    handle: ?*anyopaque,
    offset: u64 = 0,

    fn at(self: View, bytes: u64) View {
        return .{ .handle = self.handle, .offset = self.offset + bytes };
    }
    fn binding(self: View, index: u32) Binding {
        return .{ .handle = self.handle, .offset = self.offset, .index = index };
    }
};
pub const Binding = extern struct { handle: ?*anyopaque, offset: u64, index: u32 };

extern fn flint_gpu_dispatch(
    context: ?*anyopaque,
    command: ?*anyopaque,
    kernel: [*:0]const u8,
    params: ?*const anyopaque,
    params_bytes: usize,
    bindings: [*]const Binding,
    binding_count: usize,
    groups: *const [3]u32,
    threads: *const [3]u32,
) bool;

pub const Encoder = struct {
    context: ?*anyopaque,
    command: ?*anyopaque,

    pub fn dispatch(self: Encoder, name: [:0]const u8, params: anytype, bindings: []const Binding, grid: [3]u32, threads: u32) !void {
        const block = [3]u32{ threads, 1, 1 };
        if (!flint_gpu_dispatch(self.context, self.command, name, &params, @sizeOf(@TypeOf(params)), bindings.ptr, bindings.len, &grid, &block))
            return error.GpuDispatchFailed;
    }
};

pub const Model = struct {
    banks: [13]View,
    metadata: *const [50][64]u32,
    routed_bits: u32 = 4,

    fn tensor(self: Model, row: *const [64]u32, column: usize) View {
        return self.banks[row[0]].at(row[column]);
    }
};

pub const Cache = struct {
    banks: [4]View,
    /// Decode's insertion cursor is relative to the last chronological prefill.
    initial_position: u32,
};

const Slot = enum {
    hidden,
    norm,
    q,
    k,
    v,
    attention,
    projection,
    gate,
    up,
    activation,
    shared,
    sorted_x,
    experts,
    router,
    ids,
    weights,
    permutation,
    sorted_ids,
    counts,
    offsets,
    cursors,
    kv_keys,
    kv_values,
    head,
};

pub const Layout = struct {
    capacity: u32,
    context: u32,
    kv_capacity: u32,
    offsets: [@typeInfo(Slot).@"enum".field_names.len]u64,
    bytes: u64,

    pub fn init(capacity: u32, context: u32) !Layout {
        if (capacity == 0 or capacity > context or context > 262144) return error.InvalidPrefillCapacity;
        const kv_capacity = if (context <= 65536) capacity + 513 else context;
        var result = Layout{ .capacity = capacity, .context = context, .kv_capacity = kv_capacity, .offsets = undefined, .bytes = 0 };
        const sizes = [_]u64{
            @as(u64, capacity) * 2048 * 2, // hidden
            @as(u64, capacity) * 2048 * 2, // norm
            @as(u64, capacity) * 4096 * 2, // q
            @as(u64, capacity) * 512 * 2, // k
            @as(u64, capacity) * 512 * 2, // v
            @as(u64, capacity) * 4096 * 2, // attention
            @as(u64, capacity) * 2048 * 2, // projection
            @as(u64, capacity) * 6144 * 2, // dense6144 or six routed512 gate
            @as(u64, capacity) * 6144 * 2, // up
            @as(u64, capacity) * 6144 * 2, // activation
            @as(u64, capacity) * 2048 * 2, // shared output
            @as(u64, capacity) * 6 * 2048 * 2, // sorted routed inputs
            @as(u64, capacity) * 6 * 2048 * 2, // routed outputs
            @as(u64, capacity) * 256 * 2, // router logits
            @as(u64, capacity) * 6 * 4, // selected experts
            @as(u64, capacity) * 6 * 2, // BF16 mixture weights
            @as(u64, capacity) * 6 * 4, // inverse counting-sort permutation
            @as(u64, capacity) * 6 * 4, // sorted expert ids
            256 * 4,
            256 * 4,
            256 * 4,
            @as(u64, kv_capacity) * 512 * 2,
            @as(u64, kv_capacity) * 512 * 2,
            150144 * 2,
        };
        for (sizes, 0..) |bytes, i| {
            result.bytes = std.mem.alignForward(u64, result.bytes, 256);
            result.offsets[i] = result.bytes;
            result.bytes += bytes;
        }
        return result;
    }

    pub fn bind(self: Layout, buffer: View) Scratch {
        return .{ .layout = self, .buffer = buffer };
    }
};

pub const Scratch = struct {
    layout: Layout,
    buffer: View,
    fn view(self: Scratch, slot: Slot) View {
        return self.buffer.at(self.layout.offsets[@backingInt(slot)]);
    }
};

const Rows = extern struct { rows: u32, width: u32, epsilon: f32 = 1e-6, pad: u32 = 0 };
const Embedding = extern struct { rows: u32, width: u32 = 2048, vocab: u32 = 150144, pad: u32 = 0 };
const Qk = extern struct { rows: u32, heads: u32, position: u32, rope: u32, epsilon: f32 = 1e-6, theta: f32 = 10000 };
const Route = extern struct { rows: u32, experts: u32 = 256, top_k: u32 = 6, pad: u32 = 0 };
const Scatter = extern struct { rows: u32, width: u32 = 2048, top_k: u32 = 6, pad: u32 = 0 };
const Matmul = extern struct { m: i32, n: i32, k: i32, pad: i32 = 0 };
pub const Kv = extern struct { offset: u64, capacity: u32, position: u32, rows: u32, is_global: u32, prior: u32, total: u32, initial_position: u32, pad: u32 = 0 };
const AttentionBase = extern struct {
    batch: i32 = 1,
    heads: i32 = 32,
    dim: i32 = 128,
    queries: i32,
    keys: i32,
    gqa_factor: i32 = 8,
    scale: f32 = 0.08838834764831845,
    query_blocks: i32,
    key_blocks: i32,
    query_aligned: i32,
    key_aligned: i32,
    query_remainder: i32,
    key_remainder: i32,
    query_offset: i32,
    query_strides: [3]i64,
    key_strides: [3]i64,
    value_strides: [3]i64,
    output_strides: [3]i64,
};
const Attention = extern struct { base: AttentionBase, window: i32, pad: i32 = 0 };

fn groups(items: u32, width: u32) [3]u32 {
    return .{ std.math.divCeil(u32, items, width) catch unreachable, 1, 1 };
}
fn elementwise(encoder: anytype, name: [:0]const u8, params: Rows, bindings: []const Binding) !void {
    try encoder.dispatch(name, params, bindings, groups(params.rows * params.width, 256), 256);
}
fn linear(encoder: anytype, model: Model, row: *const [64]u32, column: usize, input: View, output: View, count: u32) !void {
    const p = Matmul{ .m = @intCast(count), .n = @intCast(row[column + 3]), .k = @intCast(row[column + 4]) };
    const narrow = count <= 8;
    try encoder.dispatch(if (narrow) "flint_prefill_qmv4_packed" else "flint_prefill_qmm4", p, &.{
        model.tensor(row, column).binding(1),     model.tensor(row, column + 1).binding(2),
        model.tensor(row, column + 2).binding(3), input.binding(4),
        output.binding(5),
    }, if (narrow) .{ @intCast(@divTrunc(p.n + 15, 16)), count, 1 } else .{ @intCast(@divExact(p.n, 64)), @intCast(@divTrunc(p.m + 63, 64)), 1 }, 128);
}
fn gatherLinear(encoder: anytype, model: Model, row: *const [64]u32, column: usize, input: View, output: View, ids: View, count: u32) !void {
    const p = Matmul{ .m = @intCast(count), .n = @intCast(row[column + 3]), .k = @intCast(row[column + 4]) };
    const narrow = count <= 768 and model.routed_bits == 4;
    const name: [:0]const u8 = if (narrow) "flint_prefill_gather_qmv4_packed" else if (model.routed_bits == 4) "flint_prefill_gather_qmm4" else "flint_prefill_gather_qmm3";
    try encoder.dispatch(name, p, &.{
        input.binding(1),                         model.tensor(row, column).binding(2), model.tensor(row, column + 1).binding(3),
        model.tensor(row, column + 2).binding(4), ids.binding(5),                       output.binding(6),
    }, if (narrow) .{ @intCast(@divTrunc(p.n + 15, 16)), count, 1 } else .{ @intCast(@divExact(p.n, 64)), @intCast(@divTrunc(p.m + 63, 64)), 1 }, 128);
}
fn mlp(encoder: anytype, model: Model, row: *const [64]u32, column: usize, scratch: Scratch, output: View, count: u32) !void {
    try linear(encoder, model, row, column, scratch.view(.norm), scratch.view(.gate), count);
    try linear(encoder, model, row, column + 5, scratch.view(.norm), scratch.view(.up), count);
    try elementwise(encoder, "flint_prefill_swiglu", .{ .rows = count, .width = row[column + 3] }, &.{
        scratch.view(.gate).binding(1), scratch.view(.up).binding(2), scratch.view(.activation).binding(3),
    });
    try linear(encoder, model, row, column + 10, scratch.view(.activation), output, count);
}
fn publishKv(encoder: anytype, p: Kv, banks: [4]View, scratch: Scratch) !void {
    const keys = scratch.view(if (p.is_global != 0) .k else .kv_keys);
    const values = scratch.view(if (p.is_global != 0) .v else .kv_values);
    try encoder.dispatch("flint_prefill_kv_publish", p, &.{
        banks[0].binding(1), banks[1].binding(2), banks[2].binding(3), banks[3].binding(4),
        keys.binding(5),     values.binding(6),
    }, groups((if (p.is_global != 0) p.rows else @min(p.total, p.capacity)) * 512, 256), 256);
}
fn assembleKv(encoder: anytype, p: Kv, banks: [4]View, scratch: Scratch) !void {
    try encoder.dispatch("flint_prefill_kv_assemble", p, &.{
        banks[0].binding(1),         banks[1].binding(2),         banks[2].binding(3),               banks[3].binding(4),
        scratch.view(.k).binding(5), scratch.view(.v).binding(6), scratch.view(.kv_keys).binding(7), scratch.view(.kv_values).binding(8),
    }, groups(p.total * 512, 256), 256);
}

/// Metadata and KV banks use FullModel's existing layouts. d[60] must reference
/// the BF16 router copy; d[61] is an element offset and d[62] the KV bank stride.
/// The caller owns allocation, model/session lifetime and command submission.
pub fn encode(encoder: anytype, model: Model, cache: Cache, scratch: Scratch, tokens: View, position: u32, count: u32, logits_f32: View) !void {
    if (count == 0 or count > scratch.layout.capacity or position > scratch.layout.context - count)
        return error.InvalidPrefillRange;
    if (model.routed_bits != 3 and model.routed_bits != 4) return error.UnsupportedRoutedBits;
    const tail = &model.metadata[49];
    try encoder.dispatch("flint_prefill_embedding", Embedding{ .rows = count }, &.{
        tokens.binding(1),                 model.tensor(tail, 10).binding(2), model.tensor(tail, 11).binding(3),
        model.tensor(tail, 12).binding(4), scratch.view(.hidden).binding(5),
    }, groups(count * 2048, 256), 256);

    for (model.metadata[0..49]) |*row| {
        const is_global = row[1] != 0;
        try encoder.dispatch("flint_prefill_rms", Rows{ .rows = count, .width = 2048 }, &.{
            scratch.view(.hidden).binding(1), model.tensor(row, 3).binding(2), scratch.view(.norm).binding(3),
        }, .{ count, 1, 1 }, 256);
        try linear(encoder, model, row, 10, scratch.view(.norm), scratch.view(.q), count);
        try linear(encoder, model, row, 15, scratch.view(.norm), scratch.view(.k), count);
        try linear(encoder, model, row, 20, scratch.view(.norm), scratch.view(.v), count);
        for ([_]struct { slot: Slot, norm: usize, heads: u32 }{
            .{ .slot = .q, .norm = 5, .heads = 32 }, .{ .slot = .k, .norm = 6, .heads = 4 },
        }) |part| try encoder.dispatch("flint_prefill_qk_norm_rope", Qk{
            .rows = count,
            .heads = part.heads,
            .position = position,
            .rope = @intFromBool(!is_global),
        }, &.{ scratch.view(part.slot).binding(1), model.tensor(row, part.norm).binding(2), scratch.view(part.slot).binding(3) }, .{ count * part.heads, 1, 1 }, 64);

        const prior = if (is_global) position else @min(position, row[62]);
        const kv = Kv{ .offset = row[61], .capacity = row[62], .position = position, .rows = count, .is_global = @intFromBool(is_global), .prior = prior, .total = prior + count, .initial_position = cache.initial_position };
        if (kv.capacity == 0 or kv.total > scratch.layout.kv_capacity and (!is_global or kv.total > kv.capacity))
            return error.InvalidKvCapacity;
        var keys: View = undefined;
        var values: View = undefined;
        var stride: u32 = undefined;
        if (is_global) {
            try publishKv(encoder, kv, cache.banks, scratch);
            if (kv.total <= kv.capacity) {
                keys = cache.banks[0].at(kv.offset * 2);
                values = cache.banks[0].at((kv.offset + @as(u64, kv.capacity) * 512) * 2);
                stride = kv.capacity;
            } else {
                try assembleKv(encoder, kv, cache.banks, scratch);
                keys = scratch.view(.kv_keys);
                values = scratch.view(.kv_values);
                stride = kv.total;
            }
        } else {
            // The old ring remains intact until attention has consumed it.
            try assembleKv(encoder, kv, cache.banks, scratch);
            keys = scratch.view(.kv_keys);
            values = scratch.view(.kv_values);
            stride = kv.total;
        }
        const q_stride = [3]i64{ @as(i64, count) * 4096, 128, 4096 };
        const kv_stride = [3]i64{ @as(i64, stride) * 512, @as(i64, stride) * 128, 128 };
        const attention = Attention{ .base = .{
            .queries = @intCast(count),
            .keys = @intCast(kv.total),
            .query_blocks = @intCast((count + 63) / 64),
            .key_blocks = @intCast((kv.total + 63) / 64),
            .query_aligned = @intCast(count / 64),
            .key_aligned = @intCast(kv.total / 64),
            .query_remainder = @intCast(count % 64),
            .key_remainder = @intCast(kv.total % 64),
            .query_offset = @intCast(prior),
            .query_strides = q_stride,
            .key_strides = kv_stride,
            .value_strides = kv_stride,
            .output_strides = q_stride,
        }, .window = if (is_global) 0 else 513 };
        try encoder.dispatch("flint_prefill_attention", attention, &.{
            scratch.view(.q).binding(1),         keys.binding(2),                 values.binding(3),
            scratch.view(.attention).binding(4), model.tensor(row, 7).binding(5),
        }, .{ (count + 63) / 64, 32, 1 }, 128);
        if (!is_global) try publishKv(encoder, kv, cache.banks, scratch);
        try linear(encoder, model, row, 25, scratch.view(.attention), scratch.view(.projection), count);
        try elementwise(encoder, "flint_prefill_add", .{ .rows = count, .width = 2048 }, &.{
            scratch.view(.hidden).binding(1), scratch.view(.projection).binding(2), scratch.view(.hidden).binding(3),
        });
        try encoder.dispatch("flint_prefill_rms", Rows{ .rows = count, .width = 2048 }, &.{
            scratch.view(.hidden).binding(1), model.tensor(row, 4).binding(2), scratch.view(.norm).binding(3),
        }, .{ count, 1, 1 }, 256);
        if (row[2] == 0) {
            try mlp(encoder, model, row, 30, scratch, scratch.view(.projection), count);
            try elementwise(encoder, "flint_prefill_add", .{ .rows = count, .width = 2048 }, &.{
                scratch.view(.hidden).binding(1), scratch.view(.projection).binding(2), scratch.view(.hidden).binding(3),
            });
        } else {
            try mlp(encoder, model, row, 45, scratch, scratch.view(.shared), count);
            try encoder.dispatch("flint_prefill_router_mm", Matmul{ .m = @intCast(count), .n = 256, .k = 2048 }, &.{
                scratch.view(.norm).binding(1), model.tensor(row, 60).binding(2), scratch.view(.router).binding(3),
            }, .{ 4, (count + 63) / 64, 1 }, 128);
            try encoder.dispatch("flint_prefill_route", Route{ .rows = count }, &.{
                scratch.view(.router).binding(1), model.tensor(row, 9).binding(2),
                scratch.view(.ids).binding(3),    scratch.view(.weights).binding(4),
            }, .{ count, 1, 1 }, 32);
            try encoder.dispatch("flint_prefill_zero_counts", Route{ .rows = count }, &.{scratch.view(.counts).binding(1)}, .{ 1, 1, 1 }, 256);
            try encoder.dispatch("flint_prefill_histogram", Route{ .rows = count }, &.{ scratch.view(.ids).binding(1), scratch.view(.counts).binding(2) }, groups(count * 6, 256), 256);
            try encoder.dispatch("flint_prefill_offsets", Route{ .rows = count }, &.{
                scratch.view(.counts).binding(1), scratch.view(.offsets).binding(2), scratch.view(.cursors).binding(3),
            }, .{ 1, 1, 1 }, 256);
            try encoder.dispatch("flint_prefill_scatter", Scatter{ .rows = count * 6 }, &.{
                scratch.view(.norm).binding(1),     scratch.view(.ids).binding(2),         scratch.view(.offsets).binding(3),
                scratch.view(.cursors).binding(4),  scratch.view(.permutation).binding(5), scratch.view(.sorted_ids).binding(6),
                scratch.view(.sorted_x).binding(7),
            }, .{ count * 6, 1, 1 }, 128);
            try gatherLinear(encoder, model, row, 30, scratch.view(.sorted_x), scratch.view(.gate), scratch.view(.sorted_ids), count * 6);
            try gatherLinear(encoder, model, row, 35, scratch.view(.sorted_x), scratch.view(.up), scratch.view(.sorted_ids), count * 6);
            try elementwise(encoder, "flint_prefill_swiglu", .{ .rows = count * 6, .width = 512 }, &.{
                scratch.view(.gate).binding(1), scratch.view(.up).binding(2), scratch.view(.activation).binding(3),
            });
            try gatherLinear(encoder, model, row, 40, scratch.view(.activation), scratch.view(.experts), scratch.view(.sorted_ids), count * 6);
            try elementwise(encoder, "flint_prefill_combine", .{ .rows = count, .width = 2048 }, &.{
                scratch.view(.experts).binding(1), scratch.view(.permutation).binding(2), scratch.view(.weights).binding(3),
                scratch.view(.shared).binding(4),  scratch.view(.hidden).binding(5),
            });
        }
    }
    try encoder.dispatch("flint_prefill_rms", Rows{ .rows = 1, .width = 2048 }, &.{
        scratch.view(.hidden).at(@as(u64, count - 1) * 2048 * 2).binding(1),
        model.tensor(tail, 3).binding(2),
        scratch.view(.norm).binding(3),
    }, .{ 1, 1, 1 }, 256);
    try linear(encoder, model, tail, 30, scratch.view(.norm), scratch.view(.head), 1);
    try encoder.dispatch("flint_prefill_logits_f32", Rows{ .rows = 1, .width = 150144 }, &.{
        scratch.view(.head).binding(1), logits_f32.binding(2),
    }, groups(150144, 256), 256);
}

test "prefill layout uses disjoint aligned views and bounded local KV scratch" {
    const short = try Layout.init(4096, 32768);
    const long = try Layout.init(4096, 262144);
    try std.testing.expectEqual(@as(u32, 4609), short.kv_capacity);
    try std.testing.expectEqual(@as(u32, 262144), long.kv_capacity);
    for (short.offsets[1..], short.offsets[0 .. short.offsets.len - 1]) |next, previous| {
        try std.testing.expect(next > previous);
        try std.testing.expectEqual(@as(u64, 0), next % 256);
    }
    try std.testing.expect(long.bytes > short.bytes);
    try std.testing.expectError(error.InvalidPrefillCapacity, Layout.init(0, 1));
    try std.testing.expectEqual(@as(usize, 152), @sizeOf(AttentionBase));
    try std.testing.expectEqual(@as(usize, 160), @sizeOf(Attention));
}

test "the full graph encodes dense, routed and crossing-bank attention without readback" {
    const Recorder = struct {
        attention: u32 = 0,
        routing: u32 = 0,
        assembled: u32 = 0,
        calls: u32 = 0,
        pub fn dispatch(self: *@This(), name: [:0]const u8, params: anytype, bindings: []const Binding, grid: [3]u32, threads: u32) !void {
            _ = params;
            try std.testing.expect(grid[0] > 0 and grid[1] > 0 and grid[2] > 0 and threads > 0);
            for (bindings) |binding| try std.testing.expect(binding.handle != null);
            self.calls += 1;
            if (std.mem.eql(u8, name, "flint_prefill_attention")) self.attention += 1;
            if (std.mem.eql(u8, name, "flint_prefill_route")) self.routing += 1;
            if (std.mem.eql(u8, name, "flint_prefill_kv_assemble")) self.assembled += 1;
        }
    };
    var metadata: [50][64]u32 = @splat(@splat(0));
    for (metadata[0..49], 0..) |*row, i| {
        row[1] = @intFromBool(i % 4 == 0);
        row[2] = @intFromBool(i >= 2);
        row[62] = if (row[1] != 0) 65536 else 513;
        for ([_]struct { at: usize, n: u32, k: u32 }{
            .{ .at = 10, .n = 4096, .k = 2048 },                     .{ .at = 15, .n = 512, .k = 2048 },
            .{ .at = 20, .n = 512, .k = 2048 },                      .{ .at = 25, .n = 2048, .k = 4096 },
            .{ .at = 30, .n = if (i < 2) 6144 else 512, .k = 2048 }, .{ .at = 35, .n = if (i < 2) 6144 else 512, .k = 2048 },
            .{ .at = 40, .n = 2048, .k = if (i < 2) 6144 else 512 }, .{ .at = 45, .n = 1024, .k = 2048 },
            .{ .at = 50, .n = 1024, .k = 2048 },                     .{ .at = 55, .n = 2048, .k = 1024 },
        }) |linear_shape| {
            row[linear_shape.at + 3] = linear_shape.n;
            row[linear_shape.at + 4] = linear_shape.k;
        }
    }
    metadata[49][33] = 150144;
    metadata[49][34] = 2048;
    const dummy = View{ .handle = @ptrFromInt(1) };
    const layout = try Layout.init(64, 100000);
    var recorder = Recorder{};
    try encode(&recorder, .{ .banks = @splat(dummy), .metadata = &metadata }, .{ .banks = @splat(dummy), .initial_position = 65500 }, layout.bind(dummy), dummy, 65520, 32, dummy);
    try std.testing.expectEqual(@as(u32, 49), recorder.attention);
    try std.testing.expectEqual(@as(u32, 47), recorder.routing);
    try std.testing.expectEqual(@as(u32, 49), recorder.assembled);
    try std.testing.expect(recorder.calls > 1000);
}
