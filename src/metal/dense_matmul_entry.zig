//! Sole C facade; the ordinary dense owner is a pure native import.
const c = @import("metal_c");
const owner = @import("dense_matmul.zig");
const Tensor = c.ds4_gpu_tensor;
fn request(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) owner.Request {
    return .{ .out = out, .map = map, .size = size, .offset = offset, .width = width, .outputs = outputs, .x = x, .rows = rows };
}
export fn ds4_native_metal_q8_legacy(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64, prefer_mpp: bool, default_unpack: bool) c_int {
    return owner.q8Legacy(request(out, map, size, offset, width, outputs, x, rows), prefer_mpp, default_unpack);
}
export fn ds4_gpu_matmul_q8_0_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.projectQ8(request(out, map, size, offset, width, outputs, x, rows), false);
}
export fn ds4_gpu_qwen4_matmul_q8_0_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.projectQ8(request(out, map, size, offset, width, outputs, x, rows), true);
}
export fn ds4_gpu_matmul_q8_0_decode_mpp_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.q8Legacy(request(out, map, size, offset, width, outputs, x, rows), true, false);
}
export fn ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.q8Legacy(request(out, map, size, offset, width, outputs, x, rows), true, false);
}
export fn ds4_gpu_matmul_q8_0_rows_scalar_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.q8Rows(request(out, map, size, offset, width, outputs, x, rows), false, true);
}
export fn ds4_gpu_matmul_f16_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.half(request(out, map, size, offset, width, outputs, x, rows), false);
}
export fn ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u32) c_int {
    return owner.q8Rows(request(out, map, size, offset, width, outputs, x, rows), false, false);
}
export fn ds4_gpu_dsv41_matmul_q8_0_bf16_rows(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u32) c_int {
    return owner.q8Rows(request(out, map, size, offset, width, outputs, x, rows), true, false);
}
export fn ds4_gpu_dsv41_projection_rows(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, rows: u32, x: ?*const Tensor) c_int {
    if (width == 0 or outputs == 0 or rows == 0 or rows > 8192 or map == null) return 0;
    return owner.half(request(out, map, size, offset, width, outputs, x, rows), true);
}
export fn ds4_gpu_matmul_q8_0_pair_tensor(out0: ?*Tensor, out1: ?*Tensor, map: ?*const anyopaque, size: u64, offset0: u64, offset1: u64, width: u64, outputs0: u64, outputs1: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.q8Pair(.{ .first = request(out0, map, size, offset0, width, outputs0, x, rows), .out1 = out1, .offset1 = offset1, .outputs1 = outputs1 }, false);
}
export fn ds4_gpu_qwen4_q8_pair_tensor(out0: ?*Tensor, out1: ?*Tensor, map: ?*const anyopaque, size: u64, offset0: u64, offset1: u64, width: u64, outputs0: u64, outputs1: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.q8Pair(.{ .first = request(out0, map, size, offset0, width, outputs0, x, rows), .out1 = out1, .offset1 = offset1, .outputs1 = outputs1 }, true);
}
export fn ds4_gpu_matmul_f16_pair_tensor(out0: ?*Tensor, out1: ?*Tensor, map: ?*const anyopaque, size: u64, offset0: u64, offset1: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) c_int {
    return owner.halfPair(.{ .first = request(out0, map, size, offset0, width, outputs, x, rows), .out1 = out1, .offset1 = offset1, .outputs1 = outputs });
}
