//! Remaining supported DS41 carry/originals/candidate transactions and the
//! shared BF16-weight projection used by the actual 1017 vision encoder.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("routed_batch.zig").Command;
const Dense = @import("dense_matmul.zig");
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
fn program(name: [*:0]const u8) ?*const c.ds4_metal_hc_pipeline {
    return c.ds4_metal_routed_pipeline(0, name, 0);
}
fn floats(buffer: Buffer, count: u64) bool {
    // Mirrors dsv41_tensor_has_floats: pointer/count/bytes admission.
    return count <= max64 / 4 and buffer.view.bytes >= count * 4;
}
fn report(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
fn plan(kind: u32, name: [*:0]const u8) c.ds4_metal_supported_launch {
    var result = std.mem.zeroes(c.ds4_metal_supported_launch);
    result.kind = kind;
    result.pipeline = program(name);
    result.y = 1;
    return result;
}
fn dispatch(launch: *const c.ds4_metal_supported_launch, label: [*:0]const u8) bool {
    if (launch.pipeline == null) return false;
    var command: Command = .{ .label = label };
    defer command.drop();
    return command.acquire() and c.ds4_metal_supported_encode(command.token, launch) != 0 and command.finish();
}
/// Admission precedes initialization, including tensor view bounds, as in C.
pub fn carry(
    compact_tensor: ?*api.Tensor,
    row: u32,
    plain: ?*api.Tensor,
    width: u32,
    rows: u32,
    format: u32,
    pack: bool,
    originals: bool,
) bool {
    if (rows == 0 or rows > max32 - row or compact_tensor == plain or (if (originals) width < 64 else width == 0 or format > c.DS4_V41_CARRY_MASK)) return false;
    const words: u64 = if (format == c.DS4_V41_CARRY_BF16) (@as(u64, width) + 1) / 2 else (@as(u64, width) + 31) / 32;
    const packed_row: u64 = if (originals) 128 else words * 4;
    const compact = Buffer.tensor(compact_tensor);
    defer compact.deinit();
    const expanded = Buffer.tensor(plain);
    defer expanded.deinit();
    const packed_bytes = @as(u64, row + rows) *% packed_row;
    if ((if (originals) compact.view.bytes < packed_bytes else !floats(compact, @as(u64, row + rows) * words)) or !floats(expanded, @as(u64, rows) * width)) return false;
    if (api.init() == 0) return false;
    var launch = plan(if (originals) c.DS4_TAIL_PREROPE else c.DS4_TAIL_CARRY, if (originals) "kernel_dsv41_prerope_copy" else "kernel_dsv41_carry_copy");
    launch.first = compact.view;
    launch.first.offset +%= @as(u64, row) * packed_row;
    launch.second = expanded.view;
    launch.args = if (originals) .{ width, rows, @intFromBool(pack), 0, 0 } else .{ width, rows, @intCast(words), format, @intFromBool(pack) };
    launch.x = if (originals) 64 else (@as(u64, width) + 127) / 128;
    launch.y = rows;
    launch.threads = if (originals) 64 else 128;
    launch.dispatch_threads = originals;
    return dispatch(&launch, if (originals) "V4.1 pre-RoPE originals" else "V4.1 compact prefill carry");
}
pub fn quantize(input: ?*api.Tensor, width: u32, rows: u32, format: c.ds4_v41_activation_format) bool {
    if (width < 64 or rows == 0 or format < c.DS4_V41_FP8_E8M0 or format > c.DS4_V41_FP4_E4M3) return false;
    const x = Buffer.tensor(input);
    defer x.deinit();
    if (!floats(x, @as(u64, width) * rows) or api.init() == 0) return false;
    var launch = plan(c.DS4_TAIL_QUANTIZE, "kernel_dsv41_prerope_quantize");
    launch.args = .{ width, rows, @intCast(format), 0, 0 };
    launch.first = x.view;
    launch.x = 64 / @as(u64, if (format == c.DS4_V41_FP4_E4M3) 16 else 32);
    launch.y = rows;
    launch.threads = 32;
    return dispatch(&launch, "V4.1 pre-RoPE tail quantization");
}
pub fn candidates(
    out_tensor: ?*api.Tensor,
    scores_tensor: ?*const api.Tensor,
    mask_tensor: ?*const api.Tensor,
    width: u32,
    rows: u32,
    start: u32,
    ratio: u32,
    filter: bool,
) bool {
    if (width == 0 or width > max32 - 7 or rows == 0 or ratio == 0 or rows > max32 - start or (filter and mask_tensor == null)) return false;
    const blocks = (width + 7) / 8;
    const output_width = if (filter) width else blocks;
    const scores = Buffer.tensor(scores_tensor);
    defer scores.deinit();
    const out = Buffer.tensor(out_tensor);
    defer out.deinit();
    const mask = Buffer.tensor(mask_tensor);
    defer mask.deinit();
    if (!floats(scores, @as(u64, width) * rows) or !floats(out, @as(u64, output_width) * rows) or (filter and !floats(mask, @as(u64, blocks) * rows))) return false;
    if (api.init() == 0) return false;
    var launch = plan(if (filter) c.DS4_TAIL_FILTER else c.DS4_TAIL_BLOCKS, if (filter) "kernel_dsv41_candidate_filter" else "kernel_dsv41_candidate_blocks");
    launch.args = .{ width, rows, start, ratio, 0 };
    launch.first = scores.view;
    launch.second = out.view;
    launch.third = if (filter) mask.view else scores.view;
    launch.x = output_width;
    launch.y = rows;
    launch.threads = @min(output_width, 256);
    launch.dispatch_threads = true;
    return dispatch(&launch, "V4.1 candidate selection");
}
pub fn tensorOpsAvailable() bool {
    const facts = c.ds4_metal_dense_facts(false);
    return facts & c.DS4_DENSE_MPP != 0 and facts & c.DS4_DENSE_QUALITY == 0;
}
pub fn bf16(request: Dense.Request) bool {
    if (api.init() == 0) return false;
    const input = Buffer.tensor(request.x);
    defer input.deinit();
    const out = Buffer.tensor(request.out);
    defer out.deinit();
    // Public dimensions are u32, so these element products fit u64. Byte
    // bounds keep the original glm53_tensor_has overflow check before multiply.
    const inputs = request.width * request.rows;
    const outputs = request.outputs * request.rows;
    if (request.width == 0 or request.outputs == 0 or request.rows == 0 or inputs > max64 / 4 or outputs > max64 / 4 or !input.fits(inputs * 4) or !out.fits(outputs * 4)) return report("ds4: GLM-5.3 BF16 matmul received invalid buffers");
    const bytes = request.width *% request.outputs *% 2;
    if (request.map == null or request.offset > request.size or bytes > request.size - request.offset) return report("ds4: GLM-5.3 BF16 matrix range is outside the mapped model");
    var inner: u64 = 0;
    const pointer = api.retainModelRange(request.map, request.size, request.offset, bytes, &inner) orelse return false;
    defer api.releaseModelBuffer(pointer);
    const vector = request.rows <= 8;
    const bc_input = request.width % 32 != 0;
    const bc_output = request.outputs % 64 != 0 or request.rows % 32 != 0;
    const pso = if (vector) program("kernel_glm53_mul_mv_bf16_f32") else c.ds4_metal_dense_pipeline(c.DS4_DENSE_MM, "kernel_glm53_mul_mm_bf16_f32", 0, 0, bc_input, bc_output);
    if (pso == null) return false;
    var command: Command = .{ .label = "GLM-5.3 BF16 matmul" };
    defer command.drop();
    if (!command.acquire()) return false;
    if (vector) {
        const facts = c.ds4_metal_dense_facts(false);
        const groups: u32 = if (facts & c.DS4_DENSE_M3_ULTRA != 0 and c.getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE") == null and c.getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_BF16_NSG4") == null) 4 else 8;
        var launch = std.mem.zeroes(c.ds4_metal_embedding_launch);
        launch.kind = c.DS4_EMBED_BF16;
        launch.pipeline = pso;
        launch.args.bf16 = .{ .in_dim = @intCast(request.width), .out_dim = @intCast(request.outputs), .n_rows = @intCast(request.rows) };
        launch.source = .{ .buffer = pointer, .offset = inner, .bytes = bytes };
        launch.ids = input.view;
        launch.out = out.view;
        launch.groups_x = (@as(u32, @intCast(request.outputs)) +% groups -% 1) / groups;
        launch.groups_y = request.rows;
        launch.threads = 32 * groups;
        if (c.ds4_metal_embedding_encode(command.token, &launch) == 0) return false;
    } else {
        var launch = std.mem.zeroes(c.ds4_metal_dense_launch);
        launch.kind = c.DS4_DENSE_MM;
        launch.pipeline = pso;
        launch.args.mm = Dense.mmArgs(request, request.rows, request.width * 2);
        launch.weights = pointer;
        launch.weight_offset = inner;
        launch.x = request.x;
        launch.out = request.out;
        launch.groups_x = (@as(u32, @intCast(request.rows)) +% 31) / 32;
        launch.groups_y = (@as(u32, @intCast(request.outputs)) +% 63) / 64;
        launch.threads_x = 128;
        launch.threads_y = 1;
        launch.repeats = 1;
        launch.shared_bytes = if (bc_output) 8192 else 6144;
        if (c.ds4_metal_dense_encode(command.token, &launch) == 0) return false;
    }
    return command.finish();
}
