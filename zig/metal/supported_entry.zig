//! Sole existing supported-primitive GPU facade.
const c = @import("metal_c");
const owner = @import("supported_primitives.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_dsv41_carry_copy(
    compact: ?*Tensor,
    row: u32,
    plain: ?*Tensor,
    width: u32,
    rows: u32,
    format: u32,
    pack: bool,
) c_int {
    return @intFromBool(owner.carry(compact, row, plain, width, rows, format, pack, false));
}
export fn ds4_gpu_dsv41_prerope_copy(
    compact: ?*Tensor,
    row: u32,
    plain: ?*Tensor,
    width: u32,
    rows: u32,
    pack: bool,
) c_int {
    return @intFromBool(owner.carry(compact, row, plain, width, rows, 0, pack, true));
}
export fn ds4_gpu_dsv41_prerope_quantize(
    x: ?*Tensor,
    width: u32,
    rows: u32,
    format: c.ds4_v41_activation_format,
) c_int {
    return @intFromBool(owner.quantize(x, width, rows, format));
}
export fn ds4_gpu_dsv41_candidate_blocks(
    blocks: ?*Tensor,
    scores: ?*const Tensor,
    width: u32,
    rows: u32,
    start: u32,
    ratio: u32,
) c_int {
    return @intFromBool(owner.candidates(blocks, scores, null, width, rows, start, ratio, false));
}
export fn ds4_gpu_dsv41_candidate_filter(
    scores: ?*Tensor,
    mask: ?*const Tensor,
    width: u32,
    rows: u32,
    start: u32,
    ratio: u32,
) c_int {
    return @intFromBool(owner.candidates(scores, scores, mask, width, rows, start, ratio, true));
}
export fn ds4_gpu_dsv41_tensor_ops_available() c_int {
    return @intFromBool(owner.tensorOpsAvailable());
}
export fn ds4_gpu_glm53_matmul_bf16(
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    width: u32,
    outputs: u32,
    x: ?*const Tensor,
    rows: u32,
) c_int {
    return @intFromBool(owner.bf16(.{ .out = out, .map = model, .size = size, .offset = offset, .width = width, .outputs = outputs, .x = x, .rows = rows }));
}
