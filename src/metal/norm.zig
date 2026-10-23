//! Stateless RMS host control for ordinary Metal decode and prefill.
//! Typed shader packets come from the project header. ObjC retains mapped
//! buffers, commands and encoders; no deferred KV/task state crosses this seam.
const api = @import("bridge.zig");
const Tensor = api.Tensor;

fn hasBytes(tensor: ?*const Tensor, bytes: u64) bool {
    return tensor != null and api.tensorHasBuffer(tensor) != 0 and api.tensorBytes(tensor) >= bytes;
}
fn range(size: u64, offset: u64, bytes: u64) bool {
    return offset <= size and bytes <= size - offset;
}
fn threads(width: u32) u64 {
    const width4: u64 = width / 4;
    var count: u64 = 32;
    while (count < width4 and count < 1024) count *= 2;
    return @max(1, @min(count, width4));
}
fn rmsArgs(width: u32, rows: u32, eps: f32) api.RmsArgs {
    const row = @as(u64, width) * 4;
    // These products and signed field conversions preserve the original C
    // unsigned arithmetic, including wraparound outside ordinary model sizes.
    const bytes = row *% rows;
    return .{
        .ne00 = @bitCast(width),
        .ne00_t = @intCast(width / 4),
        .nb1 = row,
        .nb2 = bytes,
        .nb3 = bytes,
        .eps = eps,
        .nef1 = .{ @bitCast(rows), 1, 1 },
        .nef2 = .{ 1, 1, 1 },
        .nef3 = .{ 1, 1, 1 },
        .nbf1 = .{ row, row, row },
        .nbf2 = .{ bytes, row, row },
        .nbf3 = .{ bytes, row, row },
    };
}
fn rmsLaunch(mode: u32, out: ?*Tensor, x: ?*const Tensor, width: u32, rows: u32, eps: f32) api.NormLaunch {
    return .{
        .mode = mode,
        .args = .{ .rms = rmsArgs(width, rows, eps) },
        .x = x,
        .y = null,
        .out = out,
        .aux_out = null,
        .weights0 = null,
        .weights1 = null,
        .weight_offset0 = 0,
        .weight_offset1 = 0,
        .rows = rows,
        .tasks = 1,
        .threads = threads(width),
        .shared_bytes = 32 * 4,
    };
}
fn dispatch(launch: *const api.NormLaunch, label: [*:0]const u8) c_int {
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    const ok = api.normEncode(command, launch);
    return api.release(command, owned, ok, label);
}

pub export fn ds4_gpu_rms_norm_plain_tensor(out: ?*Tensor, x: ?*const Tensor, width: u32, eps: f32) callconv(.c) c_int {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, width, 1, eps);
}
pub export fn ds4_gpu_rms_norm_plain_rows_tensor(out: ?*Tensor, x: ?*const Tensor, width: u32, rows: u32, eps: f32) callconv(.c) c_int {
    if (api.init() == 0) return 0;
    if (width == 0 or rows == 0 or width & 3 != 0) return 0;
    const bytes = @as(u64, width) *% rows *% 4;
    if (!hasBytes(x, bytes) or !hasBytes(out, bytes))
        return api.reportError("ds4: Metal plain RMS norm received undersized activation buffers");
    const launch = rmsLaunch(api.normPlain, out, x, width, rows, eps);
    return dispatch(&launch, "plain RMS norm");
}
fn weighted(out: ?*Tensor, x: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rows: u32, eps: f32, bf16: bool) c_int {
    if (api.init() == 0) return 0;
    if (width == 0 or rows == 0 or width & 3 != 0) return 0;
    const row = @as(u64, width) * 4;
    if (!hasBytes(x, row *% rows) or !hasBytes(out, row *% rows))
        return api.reportError("ds4: Metal weighted RMS norm received undersized activation buffers");
    if (!range(size, offset, row))
        return api.reportError("ds4: Metal weighted RMS norm range is outside the mapped model");
    var inner: u64 = 0;
    const weights = api.retainModelRange(map, size, offset, row, &inner) orelse return 0;
    defer api.releaseModelBuffer(weights);
    var launch = rmsLaunch(if (bf16) api.normWeightedBf16 else api.normWeighted, out, x, width, rows, eps);
    launch.weights0 = weights;
    launch.weight_offset0 = inner;
    return dispatch(&launch, "weighted RMS norm");
}
pub export fn ds4_gpu_rms_norm_weight_tensor(out: ?*Tensor, x: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, eps: f32) callconv(.c) c_int {
    return weighted(out, x, map, size, offset, width, 1, eps, false);
}
pub export fn ds4_gpu_rms_norm_weight_rows_tensor(out: ?*Tensor, x: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rows: u32, eps: f32) callconv(.c) c_int {
    return weighted(out, x, map, size, offset, width, rows, eps, false);
}
pub export fn ds4_gpu_dsv41_rms_norm_weight_bf16_rows(out: ?*Tensor, x: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rows: u32, eps: f32) callconv(.c) c_int {
    return weighted(out, x, map, size, offset, width, rows, eps, true);
}
pub export fn ds4_gpu_add_rms_norm_weight_tensor(out: ?*Tensor, sum: ?*Tensor, a: ?*const Tensor, b: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, eps: f32) callconv(.c) c_int {
    if (api.init() == 0) return 0;
    if (out == null or sum == null or a == null or b == null or width == 0 or width & 3 != 0) return 0;
    const row = @as(u64, width) * 4;
    if (!hasBytes(a, row) or !hasBytes(b, row) or !hasBytes(sum, row) or !hasBytes(out, row))
        return api.reportError("ds4: Metal add+RMS norm received undersized activation buffers");
    if (!range(size, offset, row))
        return api.reportError("ds4: Metal add+RMS norm range is outside the mapped model");
    var inner: u64 = 0;
    const weights = api.retainModelRange(map, size, offset, row, &inner) orelse return 0;
    defer api.releaseModelBuffer(weights);
    var launch = rmsLaunch(api.normAdd, out, a, width, 1, eps);
    launch.y = b;
    launch.aux_out = sum;
    launch.weights0 = weights;
    launch.weight_offset0 = inner;
    return dispatch(&launch, "add+RMS norm");
}
pub export fn ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(q_out: ?*Tensor, q: ?*const Tensor, map: ?*const anyopaque, size: u64, q_offset: u64, q_width: u32, kv_out: ?*Tensor, kv: ?*const Tensor, kv_offset: u64, kv_width: u32, rows: u32, eps: f32) callconv(.c) c_int {
    if (api.init() == 0) return 0;
    if (q_out == null or q == null or kv_out == null or kv == null or
        q_width == 0 or kv_width == 0 or rows == 0 or q_width & 3 != 0 or kv_width & 3 != 0) return 0;
    const q_row = @as(u64, q_width) * 4;
    const kv_row = @as(u64, kv_width) * 4;
    if (!hasBytes(q, q_row *% rows) or !hasBytes(q_out, q_row *% rows) or
        !hasBytes(kv, kv_row *% rows) or !hasBytes(kv_out, kv_row *% rows))
        return api.reportError("ds4: Metal fused q/kv RMS norm received undersized activation buffers");
    if (!range(size, q_offset, q_row) or !range(size, kv_offset, kv_row))
        return api.reportError("ds4: Metal fused q/kv RMS norm weight range is outside the mapped model");
    var q_inner: u64 = 0;
    const q_weights = api.retainModelRange(map, size, q_offset, q_row, &q_inner) orelse return 0;
    defer api.releaseModelBuffer(q_weights);
    var kv_inner: u64 = 0;
    const kv_weights = api.retainModelRange(map, size, kv_offset, kv_row, &kv_inner) orelse return 0;
    defer api.releaseModelBuffer(kv_weights);
    const args = api.QkvRmsArgs{
        .q_n = @bitCast(q_width),
        .q_n4 = @intCast(q_width / 4),
        .kv_n = @bitCast(kv_width),
        .kv_n4 = @intCast(kv_width / 4),
        .q_row_stride = q_row,
        .kv_row_stride = kv_row,
        .eps = eps,
        .tasks = 0,
    };
    const launch = api.NormLaunch{
        .mode = api.normQkv,
        .args = .{ .qkv = args },
        .x = q,
        .y = kv,
        .out = q_out,
        .aux_out = kv_out,
        .weights0 = q_weights,
        .weights1 = kv_weights,
        .weight_offset0 = q_inner,
        .weight_offset1 = kv_inner,
        .rows = rows,
        .tasks = 2,
        .threads = threads(q_width),
        .shared_bytes = 32 * 4,
    };
    return dispatch(&launch, "fused q/kv RMS norm");
}
