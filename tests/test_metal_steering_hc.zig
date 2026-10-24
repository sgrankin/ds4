//! Model-free residual oracle: publish with the established kernels, then
//! independently project/add. A block-only intervention fails these cases.
const std = @import("std");
const c = @import("steering_gpu_c");
const allocator = std.heap.c_allocator;
const Tensor = *c.ds4_gpu_tensor;
const bank_layer = 2;
const hc_streams = 4;

const Mode = enum(c_int) {
    project,
    add,

    fn id(self: Mode) c_int {
        return @backingInt(self);
    }
};

fn require(ok: bool) !void {
    if (!ok) return error.GpuCallFailed;
}

fn values(count: usize, seed: u32) ![]f32 {
    const result = try allocator.alloc(f32, count);
    const seeded = seed *% 1013904223;
    for (result, 0..) |*value, i| {
        const sample: i32 = @intCast((i * 1664525 + seeded) % 1023);
        value.* = @as(f32, @floatFromInt(sample - 511)) / 1024;
    }
    return result;
}

fn directions(width: u32) ![]f32 {
    const result = try values(3 * width, 17);
    for (0..3) |layer| {
        const row = result[layer * width ..][0..width];
        var square: f64 = 0;
        for (row) |value| square += @as(f64, value) * value;
        const norm: f32 = @floatCast(@sqrt(square));
        for (row) |*value| value.* /= norm;
    }
    return result;
}

fn write(tensor: Tensor, data: []const f32) !void {
    try require(c.ds4_gpu_tensor_write(tensor, 0, data.ptr, data.len * @sizeOf(f32)) != 0);
}

fn read(tensor: Tensor, data: []f32) !void {
    try require(c.ds4_gpu_tensor_read(tensor, 0, data.ptr, data.len * @sizeOf(f32)) != 0);
}

/// Fixture-scoped tensors, including the short row views used by norm checks.
const Tensors = struct {
    items: std.ArrayList(Tensor) = .empty,

    fn deinit(self: *Tensors) void {
        for (self.items.items) |tensor| c.ds4_gpu_tensor_free(tensor);
        self.items.deinit(allocator);
    }

    fn upload(self: *Tensors, data: []const f32) !Tensor {
        const tensor = c.ds4_gpu_tensor_alloc(data.len * @sizeOf(f32)) orelse return error.GpuAllocation;
        errdefer c.ds4_gpu_tensor_free(tensor);
        try write(tensor, data);
        try self.items.append(allocator, tensor);
        return tensor;
    }

    fn view(self: *Tensors, tensor: Tensor, row: usize, width: u32) !Tensor {
        const bytes: u64 = @as(u64, width) * @sizeOf(f32);
        const result = c.ds4_gpu_tensor_view(tensor, row * bytes, bytes) orelse return error.GpuView;
        errdefer c.ds4_gpu_tensor_free(result);
        try self.items.append(allocator, result);
        return result;
    }
};

fn compare(expected: Tensor, actual: Tensor, count: usize, tolerance: f32, exact: bool, maximum: *f64) !void {
    const a = try allocator.alloc(f32, count);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, count);
    defer allocator.free(b);
    try read(expected, a);
    try read(actual, b);
    var different: usize = 0;
    var worst: f64 = 0;
    for (a, b) |reference_value, actual_value| {
        if (!std.math.isFinite(reference_value) or !std.math.isFinite(actual_value)) return error.NonFiniteResult;
        worst = @max(worst, @abs(@as(f64, reference_value) - actual_value));
        const reference_bits: u32 = @bitCast(reference_value);
        const actual_bits: u32 = @bitCast(actual_value);
        different += @intFromBool(reference_bits != actual_bits);
    }
    if (worst > tolerance or (exact and different != 0)) {
        std.debug.print("steering mismatch: n={d} different={d} max={d} tolerance={d} exact={}\n", .{ count, different, worst, tolerance, exact });
        return error.SteeringMismatch;
    }
    maximum.* = @max(maximum.*, worst);
}

/// Projection uses the established standalone kernel. Addition is a plain CPU
/// offset, independent of every fused shader being checked here.
fn reference(tensor: Tensor, bank: Tensor, bank_values: []const f32, width: u32, rows: u32, scale: f32, mode: Mode) !void {
    if (scale == 0) return;
    if (mode == .project) {
        try require(c.ds4_gpu_directional_steering_project_tensor(tensor, bank, bank_layer, width, rows, scale) != 0);
        return;
    }
    try require(c.ds4_gpu_end_commands() != 0);
    const data = try allocator.alloc(f32, @as(usize, width) * rows);
    defer allocator.free(data);
    try read(tensor, data);
    const direction = bank_values[bank_layer * width ..][0..width];
    for (data, 0..) |*value, i| value.* += scale * direction[i % width];
    try write(tensor, data);
    try require(c.ds4_gpu_begin_commands() != 0);
}

const Layout = enum { separate, split, split_bf16 };

fn expand(width: u32, rows: u32, layout: Layout, scale: f32, alias: bool, mode: Mode, maximum: *f64) !void {
    errdefer std.debug.print("HC case: width={d} rows={d} layout={t} scale={d} alias={} mode={t}\n", .{ width, rows, layout, scale, alias, mode });
    const count: usize = width * rows;
    const stream_count = count * hc_streams;
    const block = try values(count, 1);
    defer allocator.free(block);
    const residual = try values(stream_count, 2);
    defer allocator.free(residual);
    const post = try values(rows * hc_streams, 3);
    defer allocator.free(post);
    const combination = try values(rows * 16, 4);
    defer allocator.free(combination);
    const split = try values(rows * 24, 5);
    defer allocator.free(split);
    const bank_values = try directions(width);
    defer allocator.free(bank_values);
    const direction = bank_values[bank_layer * width ..][0..width];
    // Incoming streams already carry the direction. Projecting only the new
    // block contribution cannot remove it from the published residual.
    for (residual, 0..) |*value, i| value.* += @as(f32, @floatFromInt(4 + (i / width) % hc_streams)) * direction[i % width];
    for (0..rows) |row| {
        @memcpy(split[row * 24 + 4 ..][0..4], post[row * hc_streams ..][0..hc_streams]);
        @memcpy(split[row * 24 + 8 ..][0..16], combination[row * 16 ..][0..16]);
    }
    var tensors = Tensors{};
    defer tensors.deinit();
    const block_a = try tensors.upload(block);
    const block_b = try tensors.upload(block);
    const residual_a = try tensors.upload(residual);
    const residual_b = try tensors.upload(residual);
    const output_a = if (alias) residual_a else try tensors.upload(residual);
    const output_b = if (alias) residual_b else try tensors.upload(residual);
    const post_tensor = try tensors.upload(post);
    const combination_tensor = try tensors.upload(combination);
    const split_tensor = try tensors.upload(split);
    const bank = try tensors.upload(bank_values);
    try require(c.ds4_gpu_begin_commands() != 0);
    if (layout == .separate) {
        try require(c.ds4_gpu_hc_expand_tensor(output_a, block_a, residual_a, post_tensor, combination_tensor, width, hc_streams) != 0);
    } else {
        try require(c.ds4_gpu_hc_expand_split_tensor(output_a, block_a, residual_a, split_tensor, width, hc_streams) != 0);
    }
    const bf16 = layout == .split_bf16;
    if (bf16) try require(c.ds4_gpu_dsv41_quantize(output_a, width * hc_streams, rows, c.DS4_V41_BF16) != 0);
    try reference(output_a, bank, bank_values, width, rows * hc_streams, scale, mode);
    if (bf16) try require(c.ds4_gpu_dsv41_quantize(output_a, width * hc_streams, rows, c.DS4_V41_BF16) != 0);
    if (layout == .separate) {
        try require(c.ds4_gpu_hc_expand_steering_tensor(
            output_b,
            block_b,
            residual_b,
            post_tensor,
            combination_tensor,
            bank,
            bank_layer,
            width,
            hc_streams,
            scale,
            mode.id(),
        ) != 0);
    } else {
        try require(c.ds4_gpu_hc_expand_split_steering_tensor(
            output_b,
            block_b,
            residual_b,
            split_tensor,
            bank,
            bank_layer,
            width,
            hc_streams,
            scale,
            bf16,
            mode.id(),
        ) != 0);
    }
    try require(c.ds4_gpu_end_commands() != 0);
    try compare(block_a, block_b, count, 0.00001, scale == 0, maximum);
    try compare(output_a, output_b, stream_count, if (bf16) 0.00390625 else 0.00001, scale == 0, maximum);
}

fn qwen(rows: u32, scale: f32, shared_source: u32, mode: Mode, maximum: *f64) !void {
    errdefer std.debug.print("Qwen case: rows={d} scale={d} shared={d} mode={t}\n", .{ rows, scale, shared_source, mode });
    const width = 2560;
    const slots = 3;
    const stride = 4;
    const count: usize = width * rows;
    const block = try values(count, 11);
    defer allocator.free(block);
    const residual = try values(count * hc_streams, 12);
    defer allocator.free(residual);
    const injection = try values(rows * 128, 13);
    defer allocator.free(injection);
    const partial = try values(count * stride, 14);
    defer allocator.free(partial);
    const weights = try values(rows * slots, 15);
    defer allocator.free(weights);
    const gate = try values(rows, 16);
    defer allocator.free(gate);
    const shared = try values(count, 17);
    defer allocator.free(shared);
    const bank_values = try directions(width);
    defer allocator.free(bank_values);
    var tensors = Tensors{};
    defer tensors.deinit();
    const block_a = try tensors.upload(block);
    const block_b = try tensors.upload(block);
    const residual_a = try tensors.upload(residual);
    const residual_b = try tensors.upload(residual);
    const inject = try tensors.upload(injection);
    const partials = try tensors.upload(partial);
    const routed_weights = try tensors.upload(weights);
    const shared_gate = try tensors.upload(gate);
    const shared_block = try tensors.upload(shared);
    const bank = try tensors.upload(bank_values);
    try require(c.ds4_gpu_begin_commands() != 0);
    try require(c.ds4_gpu_qwen4_hc_combine_tensor(residual_a, block_a, inject, rows, width, hc_streams) != 0);
    try reference(residual_a, bank, bank_values, width, rows * hc_streams, scale, mode);
    try require(c.ds4_gpu_qwen4_hc_combine_steering_tensor(
        residual_b,
        block_b,
        inject,
        bank,
        bank_layer,
        rows,
        width,
        hc_streams,
        scale,
        mode.id(),
    ) != 0);
    try require(c.ds4_gpu_end_commands() != 0);
    try compare(block_a, block_b, count, 0.00001, scale == 0, maximum);
    try compare(residual_a, residual_b, count * hc_streams, 0.00001, scale == 0, maximum);
    try write(residual_a, residual);
    try write(residual_b, residual);
    const gate_tensor = if (shared_source != 0) shared_gate else null;
    const shared_tensor = if (shared_source == 2) shared_block else null;
    try require(c.ds4_gpu_begin_commands() != 0);
    try require(c.ds4_gpu_qwen4_moe_reduce_tensor(
        block_a,
        partials,
        routed_weights,
        gate_tensor,
        shared_tensor,
        residual_a,
        inject,
        rows,
        slots,
        stride,
        width,
        hc_streams,
    ) != 0);
    try reference(residual_a, bank, bank_values, width, rows * hc_streams, scale, mode);
    try require(c.ds4_gpu_qwen4_moe_reduce_steering_tensor(
        block_b,
        partials,
        routed_weights,
        gate_tensor,
        shared_tensor,
        residual_b,
        inject,
        bank,
        bank_layer,
        rows,
        slots,
        stride,
        width,
        hc_streams,
        scale,
        mode.id(),
    ) != 0);
    try require(c.ds4_gpu_end_commands() != 0);
    try compare(block_a, block_b, count, 0.00001, true, maximum);
    try compare(residual_a, residual_b, count * hc_streams, 0.00001, scale == 0, maximum);
}

const Alias = enum { none, block, residual };

fn plainResidual(rows: u32, scale: f32, with_shared: bool, alias: Alias, model: []const u8, mode: Mode, maximum: *f64) !void {
    errdefer std.debug.print("plain residual case: rows={d} scale={d} shared={} alias={t} mode={t}\n", .{ rows, scale, with_shared, alias, mode });
    const width = 6144;
    const count: usize = width * rows;
    const block = try values(count, 21);
    defer allocator.free(block);
    const residual = try values(count, 22);
    defer allocator.free(residual);
    const shared = try values(count, 23);
    defer allocator.free(shared);
    const bank_values = try directions(width);
    defer allocator.free(bank_values);
    var tensors = Tensors{};
    defer tensors.deinit();
    const block_a = try tensors.upload(block);
    const block_b = try tensors.upload(block);
    const residual_a = try tensors.upload(residual);
    const residual_b = try tensors.upload(residual);
    const shared_block = try tensors.upload(shared);
    const bank = try tensors.upload(bank_values);
    const separate_a = try tensors.upload(block);
    const separate_b = try tensors.upload(block);
    const output_a = switch (alias) {
        .none => separate_a,
        .block => block_a,
        .residual => residual_a,
    };
    const output_b = switch (alias) {
        .none => separate_b,
        .block => block_b,
        .residual => residual_b,
    };
    try require(c.ds4_gpu_begin_commands() != 0);
    if (with_shared) try require(c.ds4_gpu_add_tensor(block_a, block_a, shared_block, @intCast(count)) != 0);
    try require(c.ds4_gpu_add_tensor(output_a, residual_a, block_a, @intCast(count)) != 0);
    try reference(output_a, bank, bank_values, width, rows, scale, mode);
    try require(c.ds4_gpu_residual_steering_tensor(
        output_b,
        block_b,
        if (with_shared) shared_block else null,
        residual_b,
        bank,
        bank_layer,
        width,
        rows,
        scale,
        mode.id(),
    ) != 0);
    try require(c.ds4_gpu_end_commands() != 0);
    try compare(output_a, output_b, count, 0.00001, scale == 0, maximum);

    // Weighted normalization also publishes the steered sum. At zero strength,
    // use the old fused add+norm to retain its exact ordinary accumulation.
    if (!with_shared and alias == .none) {
        try write(block_a, block);
        try write(block_b, block);
        try require(c.ds4_gpu_begin_commands() != 0);
        try require(c.ds4_gpu_add_tensor(separate_a, residual_b, block_a, @intCast(count)) != 0);
        try reference(separate_a, bank, bank_values, width, rows, scale, mode);
        for (0..rows) |row| {
            const sum = try tensors.view(separate_a, row, width);
            const norm = try tensors.view(residual_a, row, width);
            if (scale == 0) {
                const x = try tensors.view(block_a, row, width);
                const r = try tensors.view(residual_b, row, width);
                try require(c.ds4_gpu_add_rms_norm_weight_tensor(norm, sum, r, x, model.ptr, model.len, 16, width, 0.000001) != 0);
            } else {
                try require(c.ds4_gpu_rms_norm_weight_tensor(norm, sum, model.ptr, model.len, 16, width, 0.000001) != 0);
            }
        }
        try require(c.ds4_gpu_residual_norm_steering_tensor(
            shared_block,
            separate_b,
            block_b,
            residual_b,
            bank,
            model.ptr,
            model.len,
            16,
            bank_layer,
            width,
            rows,
            scale,
            0.000001,
            mode.id(),
        ) != 0);
        try require(c.ds4_gpu_end_commands() != 0);
        try compare(separate_a, separate_b, count, 0.00001, scale == 0, maximum);
        try compare(residual_a, shared_block, count, 0.00001, scale == 0, maximum);
    }
}

fn transform(rows: u32, scale: f32, mode: Mode, maximum: *f64) !void {
    errdefer std.debug.print("transform case: rows={d} scale={d} mode={t}\n", .{ rows, scale, mode });
    const width = 5120;
    const count: usize = width * rows;
    const raw = try values(count, 42);
    defer allocator.free(raw);
    const bank_values = try directions(width);
    defer allocator.free(bank_values);
    var tensors = Tensors{};
    defer tensors.deinit();
    const a = try tensors.upload(raw);
    const b = try tensors.upload(raw);
    const bank = try tensors.upload(bank_values);
    try require(c.ds4_gpu_begin_commands() != 0);
    try reference(a, bank, bank_values, width, rows, scale, mode);
    try require(c.ds4_gpu_directional_steering_tensor(b, bank, bank_layer, width, rows, scale, mode.id()) != 0);
    try require(c.ds4_gpu_end_commands() != 0);
    try compare(a, b, count, 0.00001, scale == 0, maximum);
}

fn seconds() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}

/// Wall time includes encoding and the final GPU wait. Fixed tensors remove
/// expert/SSD effects. Resetting the forty independent inputs is untimed.
fn timings() !void {
    try require(c.ds4_gpu_init() != 0);
    defer c.ds4_gpu_cleanup();
    const width = 5120;
    const sublayers = 40;
    for ([_]u32{ 1, 256 }) |rows| {
        const count: usize = width * rows;
        const raw = try values(count, 31);
        defer allocator.free(raw);
        const residual = try values(count * hc_streams, 32);
        defer allocator.free(residual);
        const split = try values(rows * 24, 33);
        defer allocator.free(split);
        const bank_values = try directions(width);
        defer allocator.free(bank_values);
        var tensors = Tensors{};
        defer tensors.deinit();
        var blocks: [sublayers]Tensor = undefined;
        for (&blocks) |*block| block.* = try tensors.upload(raw);
        const residual_tensor = try tensors.upload(residual);
        const output = try tensors.upload(residual);
        const mix = try tensors.upload(split);
        const bank = try tensors.upload(bank_values);
        // One warmup bracket followed by three individual ABBA brackets.
        for (0..4) |repeat| {
            for (0..4) |arm| {
                const active = arm == 1 or arm == 2;
                for (blocks) |block| try write(block, raw);
                const start = seconds();
                try require(c.ds4_gpu_begin_commands() != 0);
                for (blocks) |block| {
                    if (active) {
                        try require(c.ds4_gpu_hc_expand_split_steering_tensor(
                            output,
                            block,
                            residual_tensor,
                            mix,
                            bank,
                            bank_layer,
                            width,
                            hc_streams,
                            0.2,
                            true,
                            Mode.project.id(),
                        ) != 0);
                    } else {
                        try require(c.ds4_gpu_hc_expand_split_tensor(output, block, residual_tensor, mix, width, hc_streams) != 0);
                        try require(c.ds4_gpu_dsv41_quantize(output, width * hc_streams, rows, c.DS4_V41_BF16) != 0);
                    }
                }
                try require(c.ds4_gpu_end_commands() != 0);
                const elapsed = seconds() - start;
                if (repeat != 0) _ = c.printf(
                    "hc_timing rows=%u repeat=%zu arm=%zu active=%d sublayers=%u" ++
                        " elapsed_ms=%.6f us_per_sublayer=%.6f\n",
                    rows,
                    repeat,
                    arm,
                    @as(c_int, @intFromBool(active)),
                    @as(c_uint, sublayers),
                    elapsed * 1000,
                    elapsed * 1e6 / sublayers,
                );
            }
        }
    }
}

fn oracle() !void {
    var model_pointer: ?*anyopaque = null;
    try require(c.ds4_gpu_init() != 0);
    defer {
        c.ds4_gpu_cleanup();
        c.free(model_pointer);
    }
    const modes = [_]Mode{ .project, .add };
    const widths = [_]u32{ 3072, 4096, 5120 };
    const rows = [_]u32{ 1, 3, 17 };
    const scales = [_]f32{ -0.7, 0, 1 };
    var maximum: f64 = 0;
    var cases: usize = 0;
    for (modes) |mode| for (widths) |width| for (rows) |row_count| for (scales) |scale| for ([_]Layout{ .separate, .split, .split_bf16 }) |layout| {
        try expand(width, row_count, layout, scale, cases % 2 != 0, mode, &maximum);
        cases += 1;
    };
    for (modes) |mode| for (rows) |row_count| for (scales) |scale| for (0..3) |shared| {
        try qwen(row_count, scale, @intCast(shared), mode, &maximum);
        cases += 1;
    };

    // Model mapping requires a page-aligned host allocation. Offset16 leaves a
    // small prefix before the 6144-element F32 norm weight, as in real GGUF.
    const page: usize = @intCast(c.sysconf(c._SC_PAGESIZE));
    const model_bytes = std.mem.alignForward(usize, 6144 * @sizeOf(f32) + 16, page);
    try require(c.posix_memalign(&model_pointer, page, model_bytes) == 0);
    const model: []u8 = @as([*]u8, @ptrCast(model_pointer.?))[0..model_bytes];
    @memset(model, 0);
    const weight: []f32 = @as([*]f32, @ptrCast(@alignCast(model.ptr + 16)))[0..6144];
    for (weight, 0..) |*value, i| value.* = 0.5 + @as(f32, @floatFromInt(i % 53)) / 53;
    try require(c.ds4_gpu_set_model_map(model.ptr, model.len) != 0);
    for (modes) |mode| for (rows) |row_count| for (scales) |scale| for ([_]bool{ false, true }) |shared| for ([_]Alias{ .none, .block, .residual }) |alias| {
        try plainResidual(row_count, scale, shared, alias, model, mode, &maximum);
        cases += 1;
    };
    for (modes) |mode| for (rows) |row_count| for (scales) |scale| {
        try transform(row_count, scale, mode, &maximum);
        cases += 1;
    };
    _ = c.printf("steered HC: PASS %zu cases, maximum absolute drift %.9g; zero-scale exact\n", cases, maximum);
}

pub fn main(init: std.process.Init.Minimal) !void {
    var args = std.process.Args.Iterator.init(init.args);
    defer args.deinit();
    const program = args.next() orelse "test-metal-steering-hc";
    if (args.next()) |option| {
        if (std.mem.eql(u8, option, "--timings") and args.next() == null) return timings();
        std.debug.print("usage: {s} [--timings]\n", .{program});
        std.process.exit(2);
    }
    try oracle();
}
