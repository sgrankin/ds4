//! Model-free GPU qualification of the portable prefill primitives. Small exact
//! affine products exercise tails/expert runs; SDPA checks causal/window sinks.
const std = @import("std");
const c = @import("flint_c");
const p = @import("prefill.zig");

fn allocate(gpu: *c.flint_gpu, bytes: usize) !*anyopaque {
    return c.flint_gpu_buffer(gpu, bytes) orelse error.PrefillFixtureAllocation;
}
fn values(comptime T: type, buffer: *anyopaque, count: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(c.flint_gpu_contents(buffer))))[0..count];
}
fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn float(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn finish(gpu: *c.flint_gpu, command: *anyopaque, dispatched: bool) !void {
    const completed = c.flint_gpu_command_end(gpu, command, true);
    if (!dispatched or !completed) {
        std.debug.print("flint portable prefill fixture: {s}\n", .{c.flint_gpu_error(gpu)});
        return error.PrefillFixtureDispatch;
    }
}

const rows = 33;
const columns = 64;
const width = 512;
const experts = 3;

fn inputValue(row: usize, col: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((row * 3 + col) % 9)) - 4)) * 0.25;
}
fn scaleValue(expert: usize) f32 {
    return @as(f32, @floatFromInt(expert + 1)) * 0.0625;
}
fn biasValue(out: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(out % 3)) - 1)) * 0.125;
}
fn codeValue(expert: usize, out: usize, col: usize, bits: u32) u8 {
    return @intCast((expert * 3 + out + col) % (@as(usize, 1) << @intCast(bits)));
}
fn selectedExpert(row: usize) u32 {
    return if (row < 5) 0 else if (row < 21) 1 else 2;
}

fn quantized(gpu: *c.flint_gpu, bits: u32, gather: bool, narrow: bool) !void {
    const weight_bytes = experts * columns * width * bits / 8;
    const weight = try allocate(gpu, weight_bytes);
    defer c.flint_gpu_buffer_free(weight);
    const scales = try allocate(gpu, experts * columns * (width / 64) * 2);
    defer c.flint_gpu_buffer_free(scales);
    const biases = try allocate(gpu, experts * columns * (width / 64) * 2);
    defer c.flint_gpu_buffer_free(biases);
    const x = try allocate(gpu, rows * width * 2);
    defer c.flint_gpu_buffer_free(x);
    const y = try allocate(gpu, rows * columns * 2);
    defer c.flint_gpu_buffer_free(y);
    const ids = try allocate(gpu, rows * 4);
    defer c.flint_gpu_buffer_free(ids);
    const packed_bytes = values(u8, weight, weight_bytes);
    @memset(packed_bytes, 0);
    for (0..experts) |expert| for (0..columns) |out| {
        for (0..width) |col| {
            const index = (expert * columns + out) * width + col;
            const code = codeValue(expert, out, col, bits);
            if (bits == 4) packed_bytes[index / 2] |= code << @intCast((index % 2) * 4) else packed_bytes[index] = code;
        }
        for (0..width / 64) |group| {
            const index = (expert * columns + out) * (width / 64) + group;
            values(u16, scales, experts * columns * (width / 64))[index] = bf16(scaleValue(expert));
            values(u16, biases, experts * columns * (width / 64))[index] = bf16(biasValue(out));
        }
    };
    for (0..rows) |row| {
        values(u32, ids, rows)[row] = selectedExpert(row);
        for (0..width) |col| values(u16, x, rows * width)[row * width + col] = bf16(inputValue(row, col));
    }
    const params = p.Matmul{ .m = rows, .n = columns, .k = width };
    const name: [*:0]const u8 = if (gather)
        (if (narrow) "flint_prefill_gather_qmv4_packed_simd" else "flint_prefill_gather_qmm4_simd")
    else
        (if (narrow) "flint_prefill_qmv4_packed_simd" else if (bits == 8) "flint_prefill_qmm8_simd" else "flint_prefill_qmm4_simd");
    const dense_bindings = [_]c.flint_gpu_binding{
        .{ .handle = weight, .offset = 0, .index = 1 }, .{ .handle = scales, .offset = 0, .index = 2 },
        .{ .handle = biases, .offset = 0, .index = 3 }, .{ .handle = x, .offset = 0, .index = 4 },
        .{ .handle = y, .offset = 0, .index = 5 },
    };
    const gather_bindings = [_]c.flint_gpu_binding{
        .{ .handle = x, .offset = 0, .index = 1 },      .{ .handle = weight, .offset = 0, .index = 2 },
        .{ .handle = scales, .offset = 0, .index = 3 }, .{ .handle = biases, .offset = 0, .index = 4 },
        .{ .handle = ids, .offset = 0, .index = 5 },    .{ .handle = y, .offset = 0, .index = 6 },
    };
    const bindings: []const c.flint_gpu_binding = if (gather) &gather_bindings else &dense_bindings;
    const grid = if (narrow) [_]u32{ columns / 16, rows, 1 } else [_]u32{ columns / 32, (rows + 31) / 32, 1 };
    const command = c.flint_gpu_command_begin(gpu) orelse return error.PrefillFixtureCommand;
    try finish(gpu, command, c.flint_gpu_dispatch(gpu, command, name, &params, @sizeOf(p.Matmul), bindings.ptr, bindings.len, &grid, &[_]u32{ 128, 1, 1 }));
    const actual = values(u16, y, rows * columns);
    for (0..rows) |row| for (0..columns) |out| {
        const expert = if (gather) selectedExpert(row) else 0;
        var sum: f32 = 0;
        for (0..width) |col| {
            const w = float(bf16(scaleValue(expert) * @as(f32, @floatFromInt(codeValue(expert, out, col, bits))) + biasValue(out)));
            sum += inputValue(row, col) * w;
        }
        if (actual[row * columns + out] != bf16(sum)) {
            std.debug.print("{s} row={d} col={d}: {d} != {d}\n", .{ name, row, out, float(actual[row * columns + out]), float(bf16(sum)) });
            return error.PrefillFixtureMatmul;
        }
    };
}

fn attention(gpu: *c.flint_gpu, window: i32) !void {
    const queries = 33;
    const keys = 70;
    const heads = 32;
    const dim = 128;
    const query = try allocate(gpu, queries * heads * dim * 2);
    defer c.flint_gpu_buffer_free(query);
    const key = try allocate(gpu, keys * 4 * dim * 2);
    defer c.flint_gpu_buffer_free(key);
    const value = try allocate(gpu, keys * 4 * dim * 2);
    defer c.flint_gpu_buffer_free(value);
    const out = try allocate(gpu, queries * heads * dim * 2);
    defer c.flint_gpu_buffer_free(out);
    const sinks = try allocate(gpu, heads * 2);
    defer c.flint_gpu_buffer_free(sinks);
    @memset(values(u16, query, queries * heads * dim), 0);
    @memset(values(u16, key, keys * 4 * dim), 0);
    @memset(values(u16, sinks, heads), 0);
    for (0..4) |head| for (0..keys) |position| for (0..dim) |col| {
        const v = @as(f32, @floatFromInt(head + 1)) * 0.25 + @as(f32, @floatFromInt(position % 5)) * 0.125;
        values(u16, value, keys * 4 * dim)[(head * keys + position) * dim + col] = bf16(v);
    };
    const q_stride = [3]i64{ queries * heads * dim, dim, heads * dim };
    const kv_stride = [3]i64{ keys * 4 * dim, keys * dim, dim };
    const params = p.Attention{ .base = .{
        .queries = queries,
        .keys = keys,
        .query_blocks = (queries + 31) / 32,
        .key_blocks = (keys + 31) / 32,
        .query_aligned = queries / 32,
        .key_aligned = keys / 32,
        .query_remainder = queries % 32,
        .key_remainder = keys % 32,
        .query_offset = keys - queries,
        .query_strides = q_stride,
        .key_strides = kv_stride,
        .value_strides = kv_stride,
        .output_strides = q_stride,
    }, .window = window };
    const bindings = [_]c.flint_gpu_binding{
        .{ .handle = query, .offset = 0, .index = 1 }, .{ .handle = key, .offset = 0, .index = 2 },
        .{ .handle = value, .offset = 0, .index = 3 }, .{ .handle = out, .offset = 0, .index = 4 },
        .{ .handle = sinks, .offset = 0, .index = 5 },
    };
    const command = c.flint_gpu_command_begin(gpu) orelse return error.PrefillFixtureCommand;
    try finish(gpu, command, c.flint_gpu_dispatch(gpu, command, "flint_prefill_attention_simd", &params, @sizeOf(p.Attention), &bindings, bindings.len, &[_]u32{ (queries + 31) / 32, heads, 1 }, &[_]u32{ 128, 1, 1 }));
    const actual = values(u16, out, queries * heads * dim);
    for (0..queries) |row| for (0..heads) |head| {
        const absolute: usize = keys - queries + row;
        const begin = if (window == 0) 0 else (absolute + 1) -| @as(usize, @intCast(window));
        var sum: f32 = 0;
        for (begin..absolute + 1) |position| sum += @as(f32, @floatFromInt(head / 8 + 1)) * 0.25 + @as(f32, @floatFromInt(position % 5)) * 0.125;
        const expected = bf16(sum / @as(f32, @floatFromInt(absolute - begin + 2))); // Zero-valued sink has weight one.
        for (0..dim) |col| {
            const observed = actual[(row * heads + head) * dim + col];
            const difference = @as(i32, observed) - @as(i32, expected);
            if (!std.math.isFinite(float(observed)) or @abs(difference) > 1) {
                std.debug.print("portable SDPA window={d} row={d} head={d}: {d} != {d}\n", .{ window, row, head, float(observed), float(expected) });
                return error.PrefillFixtureAttention;
            }
        }
    };
}

fn router(gpu: *c.flint_gpu) !void {
    const x = try allocate(gpu, rows * width * 2);
    defer c.flint_gpu_buffer_free(x);
    const weight = try allocate(gpu, columns * width * 2);
    defer c.flint_gpu_buffer_free(weight);
    const y = try allocate(gpu, rows * columns * 2);
    defer c.flint_gpu_buffer_free(y);
    for (0..rows) |row| for (0..width) |col| {
        values(u16, x, rows * width)[row * width + col] = bf16(inputValue(row, col));
    };
    for (0..columns) |out| for (0..width) |col| {
        const v = @as(f32, @floatFromInt(@as(i32, @intCast((out + col) % 7)) - 3)) * 0.125;
        values(u16, weight, columns * width)[out * width + col] = bf16(v);
    };
    const params = p.Matmul{ .m = rows, .n = columns, .k = width };
    const bindings = [_]c.flint_gpu_binding{
        .{ .handle = x, .offset = 0, .index = 1 }, .{ .handle = weight, .offset = 0, .index = 2 },
        .{ .handle = y, .offset = 0, .index = 3 },
    };
    const command = c.flint_gpu_command_begin(gpu) orelse return error.PrefillFixtureCommand;
    try finish(gpu, command, c.flint_gpu_dispatch(gpu, command, "flint_prefill_router_mm_simd", &params, @sizeOf(p.Matmul), &bindings, bindings.len, &[_]u32{ columns / 32, (rows + 31) / 32, 1 }, &[_]u32{ 128, 1, 1 }));
    for (0..rows) |row| for (0..columns) |out| {
        var sum: f32 = 0;
        for (0..width) |col| sum += inputValue(row, col) * float(values(u16, weight, columns * width)[out * width + col]);
        if (values(u16, y, rows * columns)[row * columns + out] != bf16(sum)) return error.PrefillFixtureRouter;
    };
}

pub fn run(gpu: *c.flint_gpu) !void {
    try quantized(gpu, 4, false, false);
    try quantized(gpu, 8, false, false);
    try quantized(gpu, 4, true, false);
    try quantized(gpu, 4, false, true);
    try quantized(gpu, 4, true, true);
    try router(gpu);
    try attention(gpu, 0);
    try attention(gpu, 13);
    std.debug.print("flint: portable SIMD prefill matrix/expert tails, causal/window SDPA and sinks passed\n", .{});
}
