//! Exact model-free oracle for the opt-in SIMD router. The existing fused
//! bitonic router owns tied-ID order as well as probability/weight arithmetic.
const std = @import("std");
const c = @import("router_gpu_c");
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn getpagesize() c_int;
extern "c" fn posix_memalign(*?*anyopaque, usize, usize) c_int;
extern "c" fn free(?*anyopaque) void;

const Tensor = *c.ds4_gpu_tensor;
const max_experts = 384;
const max_rows = 2;
const max_top_k = 6;
const bias_offset = 16;
const flag = "DS4_METAL_V41_ROUTER_SIMD";
const Entry = enum { one, rows };

fn require(ok: bool) !void {
    if (!ok) return error.RouterCheckFailed;
}

const Result = struct {
    probs: [max_experts * max_rows]f32 = @splat(std.math.nan(f32)),
    selected: [max_top_k * max_rows]i32 = @splat(-1),
    weights: [max_top_k * max_rows]f32 = @splat(std.math.nan(f32)),
};

const Shape = struct {
    experts: u32 = max_experts,
    top_k: u32 = max_top_k,
    rows: u32 = 1,
    scale: f32 = 1.1875,
};

fn allocate(count: usize) !Tensor {
    return c.ds4_gpu_tensor_alloc(count * 4) orelse error.GpuAllocation;
}

fn write(tensor: Tensor, bytes: []const u8) !void {
    try require(c.ds4_gpu_tensor_write(tensor, 0, bytes.ptr, bytes.len) != 0);
}

fn read(tensor: Tensor, bytes: []u8) !void {
    try require(c.ds4_gpu_tensor_read(tensor, 0, bytes.ptr, bytes.len) != 0);
}

const Fixture = struct {
    logits: Tensor,
    probs: Tensor,
    selected: Tensor,
    weights: Tensor,
    model: []u8,

    fn init(model: []u8) !Fixture {
        const logits = try allocate(max_experts * max_rows);
        errdefer c.ds4_gpu_tensor_free(logits);
        const probs = try allocate(max_experts * max_rows);
        errdefer c.ds4_gpu_tensor_free(probs);
        const selected = try allocate(max_top_k * max_rows);
        errdefer c.ds4_gpu_tensor_free(selected);
        const weights = try allocate(max_top_k * max_rows);
        return .{ .logits = logits, .probs = probs, .selected = selected, .weights = weights, .model = model };
    }

    fn deinit(self: Fixture) void {
        c.ds4_gpu_tensor_free(self.logits);
        c.ds4_gpu_tensor_free(self.probs);
        c.ds4_gpu_tensor_free(self.selected);
        c.ds4_gpu_tensor_free(self.weights);
    }

    fn run(self: Fixture, entry: Entry, enabled: bool, shape: Shape, logits: []const f32) !Result {
        const probabilities: usize = shape.experts * shape.rows;
        const selections: usize = shape.top_k * shape.rows;
        std.debug.assert(logits.len == probabilities);
        try require(setenv(flag, if (enabled) "1" else "0", 1) == 0);
        try write(self.logits, std.mem.sliceAsBytes(logits));
        // Poison every output before each dispatch: a missing candidate store
        // must not silently reuse the preceding baseline's result.
        var result = Result{};
        try write(self.probs, std.mem.sliceAsBytes(result.probs[0..probabilities]));
        try write(self.selected, std.mem.sliceAsBytes(result.selected[0..selections]));
        try write(self.weights, std.mem.sliceAsBytes(result.weights[0..selections]));
        try require(c.ds4_gpu_begin_commands() != 0);
        const queued = switch (entry) {
            .one => c.ds4_gpu_dsv41_router_one(self.selected, self.weights, self.probs, self.logits, self.model.ptr, self.model.len, bias_offset, shape.experts, shape.top_k, shape.scale),
            .rows => c.ds4_gpu_dsv41_router_rows(self.selected, self.weights, self.probs, self.logits, self.model.ptr, self.model.len, bias_offset, shape.experts, shape.top_k, shape.scale, shape.rows),
        };
        // Always finish the command scope before inspecting or changing the
        // no-copy bias page, including when dispatch reports a failure.
        const finished = c.ds4_gpu_end_commands();
        try require(queued != 0 and finished != 0);
        try read(self.probs, std.mem.sliceAsBytes(result.probs[0..probabilities]));
        try read(self.selected, std.mem.sliceAsBytes(result.selected[0..selections]));
        try read(self.weights, std.mem.sliceAsBytes(result.weights[0..selections]));
        for (result.probs[0..probabilities]) |value| try require(std.math.isFinite(value) and value >= 0);
        for (result.weights[0..selections]) |value| try require(std.math.isFinite(value) and value >= 0);
        for (result.selected[0..selections]) |index| try require(index >= 0 and @as(u32, @intCast(index)) < shape.experts);
        return result;
    }
};

fn exact(comptime T: type, field: []const u8, expected: []const T, actual: []const T) !void {
    if (std.mem.eql(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual))) return;
    for (expected, actual, 0..) |a, b, i| {
        const a_bits: u32 = @bitCast(a);
        const b_bits: u32 = @bitCast(b);
        if (a_bits != b_bits) {
            std.debug.print("router {s}[{d}]: baseline={d} (0x{x}) SIMD={d} (0x{x})\n", .{ field, i, a, a_bits, b, b_bits });
            break;
        }
    }
    return error.RouterMismatch;
}

fn check(fixture: Fixture, label: []const u8, entry: Entry, shape: Shape, logits: []const f32, cases: *usize) !void {
    errdefer std.debug.print("router case: {s}, entry={t}, experts={d}, top_k={d}, rows={d}, scale={d}\n", .{ label, entry, shape.experts, shape.top_k, shape.rows, shape.scale });
    const baseline = try fixture.run(entry, false, shape, logits);
    const candidate = try fixture.run(entry, true, shape, logits);
    const probabilities: usize = shape.experts * shape.rows;
    const selections: usize = shape.top_k * shape.rows;
    try exact(f32, "probabilities", baseline.probs[0..probabilities], candidate.probs[0..probabilities]);
    try exact(i32, "ordered IDs", baseline.selected[0..selections], candidate.selected[0..selections]);
    try exact(f32, "weights", baseline.weights[0..selections], candidate.weights[0..selections]);
    cases.* += 1;
}

fn checkBoth(fixture: Fixture, label: []const u8, scale: f32, logits: []const f32, cases: *usize) !void {
    for ([_]Entry{ .one, .rows }) |entry| try check(fixture, label, entry, .{ .scale = scale }, logits, cases);
}

fn varied(logits: []f32, bias: []f32) void {
    for (logits, 0..) |*value, i| {
        const sample: i32 = @intCast((i * 73 + 19) % 257);
        value.* = @as(f32, @floatFromInt(sample - 128)) / 16;
    }
    for (bias, 0..) |*value, i| {
        const sample: i32 = @intCast((i * 29 + 7) % 61);
        value.* = @as(f32, @floatFromInt(sample - 30)) / 97;
    }
}

pub fn main() !void {
    // Model views borrow host storage. Keep the page-aligned allocation alive
    // until after GPU cleanup releases every no-copy view.
    const page: usize = @intCast(getpagesize());
    const model_bytes = std.mem.alignForward(usize, bias_offset + max_experts * 4, page);
    var model_pointer: ?*anyopaque = null;
    try require(posix_memalign(&model_pointer, page, model_bytes) == 0);
    defer free(model_pointer);
    const model = @as([*]u8, @ptrCast(model_pointer.?))[0..model_bytes];
    @memset(model, 0);
    const bias = @as([*]f32, @ptrCast(@alignCast(model.ptr + bias_offset)))[0..max_experts];
    try require(c.ds4_gpu_init() != 0);
    defer c.ds4_gpu_cleanup();
    try require(c.ds4_gpu_set_model_map(model.ptr, model.len) != 0);
    const fixture = try Fixture.init(model);
    defer fixture.deinit();
    var logits: [max_experts * max_rows]f32 = undefined;
    const row = logits[0..max_experts];
    var cases: usize = 0;

    varied(row, bias);
    // Non-power-of-two scales expose quotient rounding before the scale
    // multiply; the varied probabilities also exercise normalization order.
    for ([_]f32{ 0.1, 1.0000001192092896, 2.7182817 }) |scale|
        try checkBoth(fixture, "varied logits and biases", scale, row, &cases);

    const boundaries = [_]f32{ -90, -60, -25, -4, -3.4657359, -3.4, -0.001, 0, 0.001, 20, 20.000002, 80 };
    for (row, 0..) |*value, i| value.* = boundaries[i % boundaries.len];
    try checkBoth(fixture, "probability branch boundaries", 1.3, row, &cases);

    @memset(row, -4);
    @memset(bias, 0);
    const tied_ids = [_]usize{ 1, 17, 31, 32, 63, 64, 95, 127, 128, 191, 255, 319, 383 };
    for (tied_ids) |index| row[index] = 3;
    try checkBoth(fixture, "ties across SIMD groups", 1.1875, row, &cases);
    for ([_]usize{ 2, 98, 200, 382 }, 0..) |index, i| row[index] = 9 - @as(f32, @floatFromInt(i));
    try checkBoth(fixture, "ties at top-six boundary", 1.1875, row, &cases);

    @memset(row, 0.375);
    try checkBoth(fixture, "all equal scores", 1.3, row, &cases);

    varied(row, bias);
    const measured = try fixture.run(.one, false, .{}, row);
    try require(measured.probs[0] != measured.probs[1]);
    // Exact cancellation makes every biased score zero while retaining
    // unequal probabilities. Tied-ID order now directly affects weights.
    for (bias, measured.probs[0..max_experts]) |*value, probability| value.* = -probability;
    try checkBoth(fixture, "equal biased scores, unequal probabilities", 1.3, row, &cases);

    @memset(bias, 0);
    for (row, 0..) |*value, i| value.* = -60 + @as(f32, @floatFromInt(i % 19));
    const tiny = try fixture.run(.one, false, .{}, row);
    var selected_sum: f32 = 0;
    for (tiny.selected[0..max_top_k]) |index| selected_sum += tiny.probs[@intCast(index)];
    try require(selected_sum > 0 and selected_sum < 6.103515625e-5);
    try checkBoth(fixture, "denominator floor", 2.7182817, row, &cases);

    varied(&logits, bias);
    try check(fixture, "multirow fallback", .rows, .{ .rows = 2 }, &logits, &cases);
    try check(fixture, "expert-count fallback", .one, .{ .experts = 256 }, logits[0..256], &cases);
    try check(fixture, "top-k fallback", .one, .{ .top_k = 5 }, row, &cases);
    std.debug.print("router SIMD: PASS {d} exact comparisons (all probabilities, ordered IDs and weights)\n", .{cases});
}
