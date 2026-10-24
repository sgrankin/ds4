//! Model-free numerical check of fused prefill residuals and the exact helper
//! used by every resident decode profile. No weights, KV or serving state.
const std = @import("std");
const c = @import("flint_c");
const decode = @import("decode.zig");

const width = 2048;
const rows = 3;
const count = rows * width;
const sizes = [_]usize{ count * 2, count * 2, width * 4, count * 2, count * 6 * 2, rows * 6 * 4, rows * 6 * 2, count * 2, width * 4 };

fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn float(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn hiddenValue(i: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.125;
}
fn moduleValue(i: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.25;
}
fn combinedValue(experts: []const u16, permutation: []const u32, weights: []const u16, shared: []const u16, row: usize, col: usize) u16 {
    var sum = bf16(float(experts[permutation[row * 6] * width + col]) * float(weights[row * 6]));
    for (1..6) |rank| {
        const product = bf16(float(experts[permutation[row * 6 + rank] * width + col]) * float(weights[row * 6 + rank]));
        sum = bf16(float(sum) + float(product));
    }
    return bf16(float(sum) + float(shared[row * width + col]));
}

pub fn run(gpu: *c.flint_gpu) !void {
    var storage: [sizes.len]*anyopaque = undefined;
    var allocated: usize = 0;
    defer for (storage[0..allocated]) |buffer| c.flint_gpu_buffer_free(buffer);
    for (&storage, sizes) |*buffer, bytes| {
        buffer.* = c.flint_gpu_buffer(gpu, bytes) orelse return error.SteeringFixtureAllocation;
        allocated += 1;
    }
    const hidden = decode.contents(u16, storage[0], count);
    const module = decode.contents(u16, storage[1], count);
    const direction = decode.contents(f32, storage[2], width);
    const output = decode.contents(u16, storage[3], count);
    const experts = decode.contents(u16, storage[4], count * 6);
    const permutation = decode.contents(u32, storage[5], rows * 6);
    const weights = decode.contents(u16, storage[6], rows * 6);
    const shared = decode.contents(u16, storage[7], count);
    const captured = decode.contents(f32, storage[8], width);
    for (direction, 0..) |*value, i| value.* = if (i < 1024) 1.0 / 32.0 else 0;
    for (module, shared, 0..) |*value, *shared_value, i| {
        value.* = bf16(moduleValue(i));
        shared_value.* = bf16(moduleValue(i + 4) * 0.5);
    }
    const mixture = [_]f32{ 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125 };
    for (0..rows) |row| for (0..6) |rank| {
        permutation[row * 6 + rank] = @intCast(row * 6 + 5 - rank);
        weights[row * 6 + rank] = bf16(mixture[rank]);
        for (0..width) |col| experts[(row * 6 + rank) * width + col] = bf16(moduleValue(col + rank * 2 + row));
    };
    const Params = extern struct { rows: u32, width: u32, scale: f32, pad: u32 = 0 };
    const add_bindings = [_]c.flint_gpu_binding{
        .{ .handle = storage[0], .offset = 0, .index = 1 }, .{ .handle = storage[1], .offset = 0, .index = 2 },
        .{ .handle = storage[3], .offset = 0, .index = 3 }, .{ .handle = storage[2], .offset = 0, .index = 4 },
    };
    const combine_bindings = [_]c.flint_gpu_binding{
        .{ .handle = storage[4], .offset = 0, .index = 1 }, .{ .handle = storage[5], .offset = 0, .index = 2 },
        .{ .handle = storage[6], .offset = 0, .index = 3 }, .{ .handle = storage[7], .offset = 0, .index = 4 },
        .{ .handle = storage[0], .offset = 0, .index = 5 }, .{ .handle = storage[2], .offset = 0, .index = 6 },
    };
    const resident_bindings = [_]c.flint_gpu_binding{
        .{ .handle = storage[1], .offset = 0, .index = 1 }, .{ .handle = storage[2], .offset = 0, .index = 2 },
        .{ .handle = storage[3], .offset = 0, .index = 3 },
    };
    var add_capture_bindings: [5]c.flint_gpu_binding = undefined;
    @memcpy(add_capture_bindings[0..4], &add_bindings);
    add_capture_bindings[4] = .{ .handle = storage[8], .offset = 0, .index = 5 };
    var combine_capture_bindings: [7]c.flint_gpu_binding = undefined;
    @memcpy(combine_capture_bindings[0..6], &combine_bindings);
    combine_capture_bindings[6] = .{ .handle = storage[8], .offset = 0, .index = 7 };
    for ([_]bool{ false, true }) |additive| {
        for ([_]f32{ 0, 0.5, 1, -1, 2 }) |scale| {
            const params = Params{ .rows = rows, .width = width, .scale = scale, .pad = @intFromBool(additive) };
            for ([_]bool{ false, true }) |combine| {
                for ([_]bool{ false, true }) |capture| {
                    for (hidden, 0..) |*value, i| value.* = bf16(hiddenValue(i));
                    const bindings: []const c.flint_gpu_binding = if (combine)
                        (if (capture) &combine_capture_bindings else &combine_bindings)
                    else
                        (if (capture) &add_capture_bindings else &add_bindings);
                    const kernel: [*:0]const u8 = if (combine)
                        (if (capture) "flint_prefill_combine_capture" else if (additive) "flint_prefill_combine_additive" else "flint_prefill_combine_steer")
                    else
                        (if (capture) "flint_prefill_add_capture" else if (additive) "flint_prefill_add_additive" else "flint_prefill_add_steer");
                    const grid = [_]u32{ if (additive and !capture) count / 256 else rows, 1, 1 };
                    const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
                    const dispatched = c.flint_gpu_dispatch(gpu, command, kernel, &params, @sizeOf(Params), bindings.ptr, bindings.len, &grid, &[_]u32{ 256, 1, 1 });
                    const done = c.flint_gpu_command_end(gpu, command, true);
                    if (!dispatched or !done) return error.SteeringFixtureDispatch;
                    for (0..rows) |row| {
                        var residual: [width]u16 = undefined;
                        var dot: f32 = 0;
                        for (0..width) |col| {
                            const i = row * width + col;
                            const value = if (combine) combinedValue(experts, permutation, weights, shared, row, col) else module[i];
                            residual[col] = bf16(hiddenValue(i) + float(value));
                            dot += float(residual[col]) * direction[col];
                        }
                        for (0..width) |col| {
                            const i = row * width + col;
                            const expected = if (additive)
                                bf16(float(residual[col]) + scale * direction[col])
                            else
                                bf16(float(residual[col]) - scale * direction[col] * dot);
                            if ((if (combine) hidden[i] else output[i]) != expected) return error.SteeringResidualMismatch;
                            if (capture and row == rows - 1 and captured[col] != float(residual[col])) return error.SteeringResidualCaptureMismatch;
                        }
                    }
                }
            }
            // Feed an already-added residual to the exact resident helper.
            // The generated body uses it at both attention and FFN boundaries.
            const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
            const dispatched = c.flint_gpu_dispatch(gpu, command, if (additive) "flint_additive_fixture" else "flint_steering_fixture", &scale, @sizeOf(f32), &resident_bindings, resident_bindings.len, &[_]u32{ if (additive) width / 256 else 1, 1, 1 }, &[_]u32{ 256, 1, 1 });
            const done = c.flint_gpu_command_end(gpu, command, true);
            if (!dispatched or !done) return error.SteeringFixtureDispatch;
            var dot: f32 = 0;
            for (module[0..width], direction) |value, d| dot += float(value) * d;
            for (output[0..width], module[0..width], direction) |value, input, d| {
                const expected = if (additive) bf16(float(input) + scale * d) else bf16(float(input) - scale * d * dot);
                if (value != expected) return error.SteeringResidentMismatch;
            }
        }
    }
    std.debug.print("flint: residual project/add production, resident and capture oracles passed (zero, positive, negative scales)\n", .{});
}
