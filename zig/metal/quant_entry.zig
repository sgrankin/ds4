//! Sole existing GeneralQuant ABI facade.
const c = @import("metal_c");
const owner = @import("quant.zig");
const Tensor = c.ds4_gpu_tensor;
fn request(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    width: u64,
    outputs: u64,
    input: ?*const Tensor,
    rows: u64,
) owner.Request {
    return .{
        .out = out,
        .map = model,
        .size = size,
        .offset = offset,
        .width = width,
        .outputs = outputs,
        .x = input,
        .rows = rows,
    };
}
export fn ds4_gpu_matmul_quant_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    width: u64,
    outputs: u64,
    input: ?*const Tensor,
    rows: u64,
) c_int {
    return @intFromBool(owner.run(request(out, model, size, offset, width, outputs, input, rows), format));
}
export fn ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    width: u64,
    outputs: u64,
    input: ?*const Tensor,
    rows: u64,
) c_int {
    // The original decode entry deliberately prefers classic Q8 matvec at one row.
    return @intFromBool(owner.run(request(out, model, size, offset, width, outputs, input, rows), format));
}
export fn ds4_gpu_matmul_quant_rows_scalar_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    width: u64,
    outputs: u64,
    input: ?*const Tensor,
    rows: u64,
) c_int {
    return @intFromBool(owner.run(request(out, model, size, offset, width, outputs, input, rows), format));
}
export fn ds4_gpu_matmul_quant_kslice_tensor(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    format: u32,
    width: u64,
    first: u64,
    count: u64,
    outputs: u64,
    input: ?*const Tensor,
    input_offset: u64,
) c_int {
    return @intFromBool(owner.slice(request(out, model, size, offset, width, outputs, input, 1), format, first, count, input_offset));
}
