//! V4.1 activation quantization, Engram gating and two-row KV pooling.
//! Shape and carry policy live here; ObjC only binds retained Metal resources.
const std = @import("std");
const api = @import("bridge.zig");
const Tensor = api.Tensor;

// Match the existing tensor check: buffer resolution happens during encode.
fn hasFloats(tensor: ?*const Tensor, count: u64) bool {
    return tensor != null and count <= std.math.maxInt(u64) / 4 and
        api.tensorBytes(tensor) >= count * 4;
}

pub export fn ds4_gpu_dsv41_quantize(x: ?*Tensor, width: u32, rows: u32, format: api.ActivationFormat) callconv(.c) c_int {
    const block: u32 = if (format == api.fp4) 16 else 32;
    const count = @as(u64, width) * rows;
    if (width == 0 or rows == 0 or format < api.bf16 or format > api.fp4 or
        (format != api.bf16 and width % block != 0) or !hasFloats(x, count)) return 0;
    if (api.init() == 0) return 0;
    const linear = format == api.bf16 and count <= std.math.maxInt(u32) and
        api.tensorOffset(x) % 16 == 0 and api.envPresent("DS4_METAL_DISABLE_V41_LINEAR_BF16") == 0;
    const launch = api.QuantLaunch{
        .linear = @intFromBool(linear),
        .args = .{ .width = width, .rows = rows, .format = @intCast(format) },
        // Current C publishes all eight bytes, although the shader reads the
        // low word. Keep the actual packet, rather than the old port's u32.
        .count = count,
        .x = x,
        .groups_x = if (linear) (count + 1023) / 1024 else (@as(u64, width) + block - 1) / block,
        .groups_y = if (linear) 1 else rows,
        .threads = if (linear) 256 else 32,
    };
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    const ok = api.quantEncode(command, &launch);
    return api.release(command, owned, ok, "V4.1 activation quantization");
}

pub export fn ds4_gpu_dsv41_engram_add(residual: ?*Tensor, kv: ?*const Tensor, q_weight: ?*const Tensor, k_weight: ?*const Tensor, mask: ?*const Tensor, width: u32, rows: u32, eps: f32) callconv(.c) c_int {
    const count = @as(u64, width) * rows;
    if (width == 0 or rows == 0 or !std.math.isFinite(eps) or eps <= 0 or count > std.math.maxInt(u64) / 5 or
        !hasFloats(residual, count * 4) or !hasFloats(kv, count * 5) or
        !hasFloats(q_weight, @as(u64, width) * 4) or !hasFloats(k_weight, @as(u64, width) * 4) or
        (mask != null and api.tensorBytes(mask) < rows)) return 0;
    if (api.init() == 0) return 0;
    const launch = api.EngramLaunch{
        .args = .{ .width = width, .rows = rows, .eps = eps, .masked = @intFromBool(mask != null) },
        .residual = residual,
        .kv = kv,
        .q_weight = q_weight,
        .k_weight = k_weight,
        .mask = if (mask != null) mask else residual,
        .groups_x = rows,
        .groups_y = 4,
        .threads = 32,
    };
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    const ok = api.engramEncode(command, &launch);
    return api.release(command, owned, ok, "V4.1 Engram gate");
}

pub export fn ds4_gpu_dsv41_pool2(out: ?*Tensor, kv: ?*const Tensor, scores: ?*const Tensor, previous_kv: ?*Tensor, previous_scores: ?*Tensor, width: u32, rows: u32, start: u32) callconv(.c) c_int {
    const count = @as(u64, width) * rows;
    const pairs: u32 = @intCast((@as(u64, rows) + (start & 1)) / 2);
    if (width == 0 or rows == 0 or rows > std.math.maxInt(u32) - start or
        !hasFloats(kv, count) or !hasFloats(scores, count) or
        !hasFloats(previous_kv, width) or !hasFloats(previous_scores, width) or
        (pairs != 0 and !hasFloats(out, @as(u64, width) * pairs))) return 0;
    if (api.init() == 0) return 0;
    if (pairs != 0) {
        const launch = api.PoolLaunch{
            .args = .{ .width = width, .pairs = pairs, .odd = start & 1 },
            .out = out,
            .kv = kv,
            .scores = scores,
            .previous_kv = previous_kv,
            .previous_scores = previous_scores,
            .grid_x = width,
            .grid_y = pairs,
            .threads = @min(width, 256),
        };
        var owned: c_int = 0;
        const command = api.acquire(&owned) orelse return 0;
        const ok = api.poolEncode(command, &launch);
        if (api.release(command, owned, ok, "V4.1 KV pair pooling") == 0) return 0;
    }
    // Keep the last even-position input, as the row-at-a-time path does.
    // It is live only for an odd frontier, but matching it also makes
    // snapshots independent of how the prefix was partitioned.
    const last_even = (start + rows - 1) & ~@as(u32, 1);
    if (last_even >= start) {
        const bytes = @as(u64, width) * 4;
        const offset = @as(u64, last_even - start) * bytes;
        if (api.tensorCopy(previous_kv, 0, kv, offset, bytes) == 0 or
            api.tensorCopy(previous_scores, 0, scores, offset, bytes) == 0) return 0;
    }
    return 1;
}
