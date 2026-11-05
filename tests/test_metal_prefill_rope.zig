//! Model-free oracle for prefill's BF16 rounding plus inverse RoPE fusion.
//! Compare every output bit with the two-dispatch path, then check selected
//! heads independently so a shared row/head indexing mistake cannot hide.
const std = @import("std");
const c = @import("prefill_rope_gpu_c");
extern "c" fn unsetenv([*:0]const u8) c_int;
const allocator = std.heap.c_allocator;
const Tensor = *c.ds4_gpu_tensor;
const width = 512;
const guard: f32 = 12345.25;

const Case = struct { heads: u32, rows: u32, start: u32 };

fn require(ok: bool, message: []const u8) !void {
    if (!ok) {
        std.debug.print("prefill RoPE: {s}\n", .{message});
        return error.PrefillRopeCheckFailed;
    }
}

fn bits(value: f32) u32 {
    return @bitCast(value);
}

fn bf16Bits(value: f32) u32 {
    const raw = bits(value);
    return (raw +% 0x7fff +% ((raw >> 16) & 1)) & 0xffff0000;
}

fn fill(values: []f32) void {
    // Both halfway directions, their immediate neighbors, signed zeros,
    // subnormals and finite extremes. Largest f32 rounds to infinity in the
    // unrotated prefix; rotated values are bounded to avoid overflow there.
    const boundaries = [_]u32{
        0x00000000, 0x80000000, 0x3f807fff, 0x3f808000,
        0x3f808001, 0x3f817fff, 0x3f818000, 0x3f818001,
        0xbf808000, 0xbf818000, 0x00000001, 0x80000001,
        0x007fffff, 0x00800000, 0x7f7f0000, 0xff7f0000,
        0x7f7fffff, 0xff7fffff,
    };
    for (values, 0..) |*value, i| {
        const head = i / width;
        const column = i % width;
        if (column % 4 == 0) {
            var raw = boundaries[(column / 4 + head * 7) % boundaries.len];
            if (column >= width - 64 and raw & 0x7fffffff > 0x7d7f0000)
                raw = (raw & 0x80000000) | 0x7d7f0000;
            value.* = @bitCast(raw);
        } else {
            // Different heads and rows have different ordinary values, while
            // the low mantissa bits force the fused prefix to do real work.
            const sample: i32 = @intCast((i *% 1664525 +% 1013904223) % 65537);
            value.* = @as(f32, @floatFromInt(sample - 32768)) / 8192;
        }
    }
}

fn write(tensor: Tensor, values: []const f32) !void {
    try require(c.ds4_gpu_tensor_write(tensor, 0, values.ptr, values.len * @sizeOf(f32)) != 0, "write tensor");
}

fn read(tensor: Tensor, values: []f32) !void {
    try require(c.ds4_gpu_tensor_read(tensor, 0, values.ptr, values.len * @sizeOf(f32)) != 0, "read tensor");
}

fn allocate(count: usize) !Tensor {
    return c.ds4_gpu_tensor_alloc(count * @sizeOf(f32)) orelse error.GpuAllocation;
}

fn separate(tensor: Tensor, heads: u32, rows: u32, start: u32, compressed: bool) bool {
    return c.ds4_gpu_dsv41_quantize(tensor, width * heads, rows, c.DS4_V41_BF16) != 0 and
        c.ds4_gpu_dsv41_rope(tensor, width, heads, rows, start, compressed, true) != 0;
}

fn checkCase(shape: Case, compressed: bool) !void {
    errdefer std.debug.print("  heads={d} rows={d} start={d} compressed={}\n", .{ shape.heads, shape.rows, shape.start, compressed });
    const count = @as(usize, width) * shape.heads * shape.rows;
    const input = try allocator.alloc(f32, count + 1);
    defer allocator.free(input);
    const expected = try allocator.alloc(f32, count + 1);
    defer allocator.free(expected);
    const actual = try allocator.alloc(f32, count + 1);
    defer allocator.free(actual);
    fill(input[0..count]);
    input[count] = guard;
    const baseline = try allocate(count + 1);
    defer c.ds4_gpu_tensor_free(baseline);
    const fused = try allocate(count + 1);
    defer c.ds4_gpu_tensor_free(fused);
    try write(baseline, input);
    try write(fused, input);

    try require(c.ds4_gpu_begin_commands() != 0, "begin comparison");
    const baseline_queued = separate(baseline, shape.heads, shape.rows, shape.start, compressed);
    const fused_queued = c.ds4_gpu_dsv41_rope_bf16(fused, width, shape.heads, shape.rows, shape.start, compressed, true);
    const finished = c.ds4_gpu_end_commands();
    try require(baseline_queued and fused_queued != 0 and finished != 0, "execute comparison");
    try read(baseline, expected);
    try read(fused, actual);
    try require(bits(expected[count]) == bits(guard) and bits(actual[count]) == bits(guard), "trailing sentinel preserved");

    var different: usize = 0;
    var prefix_different: usize = 0;
    var first: usize = 0;
    for (expected[0..count], actual[0..count], input[0..count], 0..) |a, b, original, i| {
        if (bits(a) != bits(b)) {
            if (different == 0) first = i;
            different += 1;
        }
        if (i % width < width - 64) prefix_different += @intFromBool(bits(b) != bf16Bits(original));
    }
    std.debug.print("prefill_rope width={d} heads={d} rows={d} start={d} compressed={} different={d}/{d} prefix_different={d}\n", .{
        width, shape.heads, shape.rows, shape.start, compressed, different, count, prefix_different,
    });
    if (different != 0) std.debug.print("  first row={d} head={d} column={d}: separate=0x{x} fused=0x{x}\n", .{
        first / (width * shape.heads), (first / width) % shape.heads, first % width, bits(expected[first]), bits(actual[first]),
    });
    try require(different == 0 and prefix_different == 0, "bitwise output and prefix parity");

    const single = try allocate(width);
    defer c.ds4_gpu_tensor_free(single);
    var head_output: [width]f32 = undefined;
    const selected = [_][2]u32{
        .{ 0, 0 },
        .{ shape.rows / 2, shape.heads / 2 },
        .{ shape.rows - 1, shape.heads - 1 },
    };
    for (selected) |location| {
        const offset = (@as(usize, location[0]) * shape.heads + location[1]) * width;
        try write(single, input[offset..][0..width]);
        try require(c.ds4_gpu_begin_commands() != 0, "begin single-head reference");
        const queued = separate(single, 1, 1, shape.start + location[0], compressed);
        const head_finished = c.ds4_gpu_end_commands();
        try require(queued and head_finished != 0, "execute single-head reference");
        try read(single, &head_output);
        try require(std.mem.eql(u8, std.mem.asBytes(&head_output), std.mem.sliceAsBytes(actual[offset..][0..width])), "independent row/head indexing");
    }
}

pub fn main() !void {
    try require(unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16") == 0, "enable the ordinary linear BF16 path");
    try require(c.ds4_gpu_init() != 0, "initialize GPU");
    defer c.ds4_gpu_cleanup();
    const cases = [_]Case{
        .{ .heads = 32, .rows = 1, .start = 0 },
        .{ .heads = 64, .rows = 2, .start = 8192 },
        .{ .heads = 32, .rows = 33, .start = 1048576 - 33 },
        .{ .heads = 64, .rows = 129, .start = 8192 },
    };
    for ([_]bool{ false, true }) |compressed| {
        for (cases) |shape| try checkCase(shape, compressed);
    }
    std.debug.print("prefill RoPE: bitwise PASS, 8 comparisons, 24 independent head checks\n", .{});
}
