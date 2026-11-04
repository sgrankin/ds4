//! Model-free short-Q8 query oracle. Compare inline cos/sin with the shared
//! sincos table; report all bit deltas before deciding whether parity passed.
const std = @import("std");
const c = @import("query_gpu_c");
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
extern "c" fn getpagesize() c_int;
extern "c" fn posix_memalign(*?*anyopaque, usize, usize) c_int;
extern "c" fn free(?*anyopaque) void;

const allocator = std.heap.c_allocator;
const Tensor = *c.ds4_gpu_tensor;
const width = 1280;
const outputs = 32768;
const baseline_kernel = "kernel_dsv41_query_short_q8";
const candidate_kernel = "kernel_dsv41_query_short_q8_tab";
const Q8Block = extern struct { d: u16, qs: [32]i8 };

fn require(ok: bool, message: []const u8) !void {
    if (!ok) {
        std.debug.print("query RoPE table: {s}\n", .{message});
        return error.QueryRopeTableCheckFailed;
    }
}

// Match the existing C short-query fixture's bounded xorshift input and
// Q8_0 distribution: signed bytes and half scales near 2^-8.
fn sample(seed: *u32) i32 {
    seed.* ^= seed.* << 13;
    seed.* ^= seed.* >> 17;
    seed.* ^= seed.* << 5;
    return @as(i32, @intCast(seed.* % 65537)) - 32768;
}

fn allocate(bytes: usize) !Tensor {
    return c.ds4_gpu_tensor_alloc(bytes) orelse error.GpuAllocation;
}

fn write(tensor: Tensor, values: []const f32) !void {
    try require(c.ds4_gpu_tensor_write(tensor, 0, values.ptr, values.len * 4) != 0, "write tensor");
}

fn read(tensor: Tensor, values: []f32) !void {
    try require(c.ds4_gpu_tensor_read(tensor, 0, values.ptr, values.len * 4) != 0, "read tensor");
}

fn compare(expected: []const f32, actual: []const f32, position: u32, compressed: bool) !usize {
    var different: usize = 0;
    var unrotated: usize = 0;
    var first: usize = 0;
    var max_absolute: f64 = 0;
    var max_relative: f64 = 0;
    var squared: f64 = 0;
    for (expected, actual, 0..) |a, b, i| {
        try require(std.math.isFinite(a) and std.math.isFinite(b), "nonfinite or unwritten output");
        const a_bits: u32 = @bitCast(a);
        const b_bits: u32 = @bitCast(b);
        if (a_bits == b_bits) continue;
        if (different == 0) first = i;
        different += 1;
        unrotated += @intFromBool(i % 512 < 448);
        const delta = @abs(@as(f64, a) - @as(f64, b));
        const magnitude = @max(@max(@abs(@as(f64, a)), @abs(@as(f64, b))), 1e-30);
        max_absolute = @max(max_absolute, delta);
        max_relative = @max(max_relative, delta / magnitude);
        squared += delta * delta;
    }
    std.debug.print("query_rope_table position={d} compressed={} different={d}/{d} unrotated={d} max_abs={e} max_rel={e} rms={e}\n", .{
        position, compressed, different, expected.len, unrotated, max_absolute, max_relative, @sqrt(squared / @as(f64, @floatFromInt(expected.len))),
    });
    if (different != 0) std.debug.print("  first[{d}] inline={d} (0x{x}) table={d} (0x{x})\n", .{
        first, expected[first], @as(u32, @bitCast(expected[first])), actual[first], @as(u32, @bitCast(actual[first])),
    });
    return different;
}

fn checkTable(table: []const f32, position: u32) !void {
    for (table) |value| try require(std.math.isFinite(value), "nonfinite or unwritten coefficient");
    if (position == 0) {
        for (0..64) |i| try require(table[2 * i] == 1 and table[2 * i + 1] == 0, "position-zero coefficients");
    }
    if (position == 1) {
        // Both frequency banks begin at frequency 1. The generator stores a
        // positive sine even though its host packet carries inverse=1. Query
        // rotation is forward, so it consumes that sine without negation.
        for ([_]usize{ 0, 64 }) |offset| try require(
            @abs(table[offset] - @cos(@as(f32, 1))) < 0.000002 and
                @abs(table[offset + 1] - @sin(@as(f32, 1))) < 0.000002,
            "forward cosine/sine convention",
        );
    }
}

pub fn main() !void {
    comptime std.debug.assert(@sizeOf(Q8Block) == 34);
    try require(setenv("DS4_METAL_V41_SHORT_Q8_QUERY", "1", 1) == 0, "enable short query");
    try require(unsetenv("DS4_METAL_Q8_MV_NSG") == 0, "clear explicit NSG override");

    const weight_bytes = outputs * (width / 32) * @sizeOf(Q8Block);
    const page: usize = @intCast(getpagesize());
    const model_bytes = std.mem.alignForward(usize, weight_bytes, page);
    var model_pointer: ?*anyopaque = null;
    try require(posix_memalign(&model_pointer, page, model_bytes) == 0, "allocate mapped weights");
    // No-copy model views retain this backing. GPU cleanup must run before
    // free, including on a failed dispatch or comparison.
    defer free(model_pointer);
    const model = @as([*]u8, @ptrCast(model_pointer.?))[0..model_bytes];
    @memset(model, 0);
    const blocks = @as([*]Q8Block, @ptrCast(@alignCast(model.ptr)))[0 .. weight_bytes / @sizeOf(Q8Block)];
    var seed: u32 = 7919;
    for (blocks) |*block| {
        block.d = @intCast(0x1c00 + @rem(sample(&seed), 1024));
        for (&block.qs) |*value| value.* = @intCast(@rem(sample(&seed), 127));
    }
    var input: [width]f32 = undefined;
    for (&input) |*value| value.* = @as(f32, @floatFromInt(sample(&seed))) / 8192;

    try require(c.ds4_gpu_init() != 0, "initialize GPU");
    defer c.ds4_gpu_cleanup();
    try require(c.ds4_gpu_set_model_map(model.ptr, model.len) != 0, "map Q8 weights");
    const x = try allocate(width * 4);
    defer c.ds4_gpu_tensor_free(x);
    const inline_out = try allocate(outputs * 4);
    defer c.ds4_gpu_tensor_free(inline_out);
    const table_out = try allocate(outputs * 4);
    defer c.ds4_gpu_tensor_free(table_out);
    const table = try allocate(512);
    defer c.ds4_gpu_tensor_free(table);
    try write(x, &input);
    const expected = try allocator.alloc(f32, outputs);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, outputs);
    defer allocator.free(actual);
    var coefficients: [128]f32 = undefined;
    c.ds4_gpu_test_set_flags(c.DS4_GPU_TEST_COUNT_KERNELS);
    defer c.ds4_gpu_test_set_flags(0);

    // Keep the same input and table allocation across forward and backward
    // position changes. A stale table or retained position now changes the
    // result, rather than being hidden by replacing the table each time.
    const positions = [_]u32{ 0, 1, 7, 8193, 32768, 1048575, 8193, 1, 0 };
    var cases: usize = 0;
    var different: usize = 0;
    for ([_]bool{ false, true }) |compressed| {
        for (positions) |position| {
            errdefer std.debug.print("query RoPE table case: position={d} compressed={}\n", .{ position, compressed });
            @memset(expected, std.math.nan(f32));
            @memset(&coefficients, std.math.nan(f32));
            try write(inline_out, expected);
            try write(table_out, expected);
            try write(table, &coefficients);
            const baseline_runs = c.ds4_gpu_test_kernel_runs(baseline_kernel);
            const candidate_runs = c.ds4_gpu_test_kernel_runs(candidate_kernel);
            try require(c.ds4_gpu_begin_commands() != 0, "begin commands");
            const table_queued = c.ds4_gpu_dsv41_rope_coeff_table(table, position);
            const baseline_queued = c.ds4_gpu_dsv41_project_q(inline_out, model.ptr, model.len, 0, width, outputs, x, position, compressed);
            const candidate_queued = c.ds4_gpu_dsv41_project_q_tab(table_out, model.ptr, model.len, 0, width, outputs, x, position, compressed, table);
            const finished = c.ds4_gpu_end_commands();
            try require(table_queued != 0 and baseline_queued != 0 and candidate_queued != 0 and finished != 0, "enqueue and finish query paths");
            try require(c.ds4_gpu_test_kernel_runs(baseline_kernel) == baseline_runs + 1, "baseline short-query dispatch count");
            try require(c.ds4_gpu_test_kernel_runs(candidate_kernel) == candidate_runs + 1, "table short-query dispatch count");
            try read(inline_out, expected);
            try read(table_out, actual);
            try read(table, &coefficients);
            try checkTable(&coefficients, position);
            different += try compare(expected, actual, position, compressed);
            cases += 1;
        }
    }
    std.debug.print("query RoPE table: {s}, {d} comparisons, {d} differing output values\n", .{ if (different == 0) "bitwise PASS" else "DIFFERENCES", cases, different });
    // A sincos/cos/sin difference is experimental evidence, not an implicit
    // tolerance. Preserve every case's deltas and require an explicit decision.
    if (different != 0) return error.QueryRopeTableMismatch;
}
