//! Bounded comparison for the prefill index query's 1280 -> 4096 projection.
//! Matrix reductions may differ from the exact row matvec. Report that drift
//! before and after the existing BF16 boundary; published-key tests stay exact.
const std = @import("std");
const c = @import("prefill_f16_query_gpu_c");
extern "c" fn getpagesize() c_int;
extern "c" fn posix_memalign(*?*anyopaque, usize, usize) c_int;
extern "c" fn free(?*anyopaque) void;

const allocator = std.heap.c_allocator;
const Tensor = *c.ds4_gpu_tensor;
const width = 1280;
const outputs = 4096;
const guard: f32 = 12345.25;

fn require(ok: bool, message: []const u8) !void {
    if (!ok) {
        std.debug.print("prefill F16 query: {s}\n", .{message});
        return error.PrefillF16QueryCheckFailed;
    }
}

fn sample(seed: *u32) f32 {
    seed.* ^= seed.* << 13;
    seed.* ^= seed.* >> 17;
    seed.* ^= seed.* << 5;
    return @as(f32, @floatFromInt(@as(i32, @intCast(seed.* % 65537)) - 32768)) / 8192;
}

fn bits(value: f32) u32 {
    return @bitCast(value);
}

fn bf16(value: f32) f32 {
    const raw = bits(value);
    return @bitCast((raw +% 0x7fff +% ((raw >> 16) & 1)) & 0xffff0000);
}

fn allocate(count: usize) !Tensor {
    return c.ds4_gpu_tensor_alloc(count * @sizeOf(f32)) orelse error.GpuAllocation;
}

fn contents(tensor: Tensor, count: usize) ![]const f32 {
    return @as([*]const f32, @ptrCast(@alignCast(c.ds4_gpu_tensor_contents(tensor) orelse return error.GpuContents)))[0..count];
}

fn compare(expected: []const f32, actual: []const f32, rows: u32, rounded: bool) !void {
    try require(bits(expected[expected.len - 1]) == bits(guard) and bits(actual[actual.len - 1]) == bits(guard), "trailing sentinel preserved");
    var changed: usize = 0;
    var maximum: f64 = 0;
    var squared: f64 = 0;
    var norm: f64 = 0;
    const count = expected.len - 1;
    for (expected[0..count], actual[0..count]) |a, b| {
        try require(std.math.isFinite(a) and std.math.isFinite(b), "nonfinite or unwritten output");
        changed += @intFromBool(bits(a) != bits(b));
        const delta = @as(f64, b) - a;
        maximum = @max(maximum, @abs(delta));
        squared += delta * delta;
        norm += @as(f64, a) * a;
    }
    std.debug.print("prefill_f16_query rows={d} bf16={} changed={d}/{d} max_abs={e} rms={e} relative_rms={e}\n", .{
        rows, rounded, changed, count, maximum, @sqrt(squared / @as(f64, @floatFromInt(count))), @sqrt(squared / @max(norm, 1e-30)),
    });
}

fn oracle(weights: []const f16, input: []const f32, result: []const f32, rows: u32, arm: []const u8) !void {
    var maximum: f64 = 0;
    for (0..8) |sample_index| {
        const row = sample_index * (rows - 1) / 7;
        const column = sample_index * (outputs - 1) / 7;
        var expected: f64 = 0;
        var magnitude: f64 = 0;
        for (0..width) |k| {
            const term = @as(f64, weights[column * width + k]) * input[row * width + k];
            expected += term;
            magnitude += @abs(term);
        }
        const actual = result[row * outputs + column];
        const delta = @abs(@as(f64, actual) - expected);
        maximum = @max(maximum, delta);
        // An L1-scaled bound tolerates changed F32 reduction order even at
        // cancellation, while catching wrong layouts or element types.
        const limit = 1e-4 * @max(magnitude, 1);
        if (delta > limit) std.debug.print("  {s} row={d} column={d} actual={e} expected={e} error={e} limit={e}\n", .{
            arm, row, column, actual, expected, delta, limit,
        });
        try require(std.math.isFinite(actual) and delta <= limit, "bounded double-precision dot oracle");
    }
    std.debug.print("prefill_f16_query rows={d} arm={s} cpu_dots=8 max_abs={e}\n", .{ rows, arm, maximum });
}

fn checkRows(model: []const u8, weights: []const f16, rows: u32) !void {
    errdefer std.debug.print("  F16 query width={d} outputs={d} rows={d}\n", .{ width, outputs, rows });
    const input_count = @as(usize, rows) * width;
    const count = @as(usize, rows) * outputs;
    const input = try allocator.alloc(f32, input_count);
    defer allocator.free(input);
    var seed: u32 = 7919;
    // The production input is a normalized BF16 value stored in F32. These
    // bounded inputs also fit F16 exactly, isolating reduction differences.
    for (input) |*value| value.* = bf16(sample(&seed));
    const x = try allocate(input_count);
    defer c.ds4_gpu_tensor_free(x);
    const baseline = try allocate(count + 1);
    defer c.ds4_gpu_tensor_free(baseline);
    const candidate = try allocate(count + 1);
    defer c.ds4_gpu_tensor_free(candidate);
    try require(c.ds4_gpu_tensor_write(x, 0, input.ptr, input_count * 4) != 0, "write BF16 input");
    for ([_]Tensor{ baseline, candidate }) |tensor| {
        try require(c.ds4_gpu_tensor_fill_f32(tensor, std.math.nan(f32), count + 1) != 0, "initialize output");
        try require(c.ds4_gpu_tensor_write(tensor, count * 4, &guard, 4) != 0, "write output sentinel");
    }

    try require(c.ds4_gpu_begin_commands() != 0, "begin projection comparison");
    const exact_queued = c.ds4_gpu_dsv41_projection_rows(baseline, model.ptr, model.len, 0, width, outputs, rows, x);
    const matrix_queued = c.ds4_gpu_matmul_f16_tensor(candidate, model.ptr, model.len, 0, width, outputs, x, rows);
    const finished = c.ds4_gpu_end_commands();
    try require(exact_queued != 0 and matrix_queued != 0 and finished != 0 and c.ds4_gpu_synchronize() != 0, "execute projection comparison");
    // Inspect mapped outputs only after completion; no second pair of large
    // host output arrays is needed for this model-free fixture.
    const expected = try contents(baseline, count + 1);
    const actual = try contents(candidate, count + 1);
    try compare(expected, actual, rows, false);
    try oracle(weights, input, expected, rows, "exact");
    try oracle(weights, input, actual, rows, "matrix");

    try require(c.ds4_gpu_begin_commands() != 0, "begin BF16 boundary");
    const exact_rounded = c.ds4_gpu_dsv41_quantize(baseline, outputs, rows, c.DS4_V41_BF16);
    const matrix_rounded = c.ds4_gpu_dsv41_quantize(candidate, outputs, rows, c.DS4_V41_BF16);
    const rounded_finished = c.ds4_gpu_end_commands();
    try require(exact_rounded != 0 and matrix_rounded != 0 and rounded_finished != 0 and c.ds4_gpu_synchronize() != 0, "execute BF16 boundary");
    try compare(expected, actual, rows, true);
}

pub fn main() !void {
    const weight_bytes = width * outputs * @sizeOf(f16);
    const page: usize = @intCast(getpagesize());
    const model_bytes = std.mem.alignForward(usize, weight_bytes, page);
    var model_pointer: ?*anyopaque = null;
    try require(posix_memalign(&model_pointer, page, model_bytes) == 0, "allocate mapped F16 weights");
    // No-copy Metal views retain this backing until GPU cleanup below.
    defer free(model_pointer);
    const model = @as([*]u8, @ptrCast(model_pointer.?))[0..model_bytes];
    @memset(model, 0);
    const weights = @as([*]f16, @ptrCast(@alignCast(model.ptr)))[0 .. width * outputs];
    var seed: u32 = 41;
    for (weights) |*weight| weight.* = @floatCast(sample(&seed) / 128);

    try require(c.ds4_gpu_init() != 0, "initialize GPU");
    defer c.ds4_gpu_cleanup();
    c.ds4_gpu_set_quality(false);
    try require(c.ds4_gpu_set_model_map(model.ptr, model.len) != 0, "map F16 weights");
    std.debug.print("prefill_f16_query tensor_ops_available={}\n", .{c.ds4_gpu_dsv41_tensor_ops_available() != 0});
    try checkRows(model, weights, 32);
    try checkRows(model, weights, 8192);
    std.debug.print("prefill F16 query: finite outputs, sentinels and 32 bounded CPU dots PASS; numerical differences reported\n", .{});
}
