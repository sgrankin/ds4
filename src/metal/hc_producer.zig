//! Decode HC producer fusions. Admission returns zero; a failed submitted compound
//! command returns -1, so the caller cannot replay an already-mutating operation.
const std = @import("std");
const h = @import("hc_control.zig");
const c = h.c;
const Buffer = h.Buffer;
const Inputs = h.Inputs;
const Model = h.Model;
const Shape = h.Shape;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
var norm_mix_available: ?bool = null;
pub fn available() c_int {
    if (!h.init()) return 0;
    if (norm_mix_available == null) norm_mix_available = h.pipeline(c.DS4_HC_NORM_MIX).handle != null;
    return @intFromBool(norm_mix_available.?);
}
// The fused reduction reproduces standalone RMSNorm/matvec reduction trees
// bit-exactly (metal/dsv4_hc.metal); this only removes a dispatch boundary.
pub fn normMix(inputs: Inputs, model: Model, n: u32, out_dim: u32, eps: f32) c_int {
    if (!h.init() or inputs.out == null or inputs.input == null or model.map == null or n != 16384 or out_dim != 24) return 0;
    const weight_bytes = @as(u64, n) * 2 * out_dim;
    if (!model.range(model.weight, weight_bytes)) return h.fail("ds4: Metal fused HC norm/mix weight range is outside the mapped model");
    const buffers = h.Buffers.from(inputs);
    defer buffers.deinit();
    if (!buffers.input.fits(@as(u64, n) * 4) or !buffers.out.fits(@as(u64, out_dim) * 4)) return h.fail("ds4: Metal fused HC norm/mix received undersized activation buffers");
    const weights = model.buffer(model.weight, weight_bytes) orelse return 0;
    defer weights.deinit();
    const facts = c.ds4_metal_attention_get_facts();
    const cluster = facts.flags & c.DS4_ATTN_M5 != 0 and !h.present("DS4_METAL_DISABLE_M5_HC_NORM_MIX_CLUSTER2");
    const pso = h.pipeline(if (cluster) c.DS4_HC_NORM_MIX_CLUSTER else c.DS4_HC_NORM_MIX);
    if (pso.handle == null or (cluster and pso.threads < 512)) return 0;
    var launch = h.plan(c.DS4_HC_NORM_MIX, pso);
    buffers.fill(&launch);
    launch.norm_mix_args = .{ .n = h.narrow(n), .out_dim = h.narrow(out_dim), .eps = eps };
    launch.weights = weights.view;
    launch.shared_bytes = (if (cluster) @as(u64, 160) else 96) * 4;
    launch.gx = if (cluster) (out_dim + 3) / 4 else (out_dim + 1) / 2;
    launch.tx = 32;
    launch.ty = if (cluster) 16 else 8;
    return h.run(launch, "fused HC norm/mix");
}
pub fn compound(inputs: Inputs, model: Model, shape: Shape, n: u32, mix_dim: u32, rms_eps: f32, expanded: bool) c_int {
    if (!h.init() or inputs.mix == null or inputs.out == null or inputs.norm_out == null or inputs.split == null or inputs.residual == null or model.map == null or n != 16384 or mix_dim != 24 or shape.width != 4096 or shape.hc != 4) return 0;
    if (expanded and (inputs.block == null or inputs.block_add == null or inputs.previous == null or inputs.post == null or inputs.comb == null)) return 0;
    const weight_bytes = @as(u64, n) * mix_dim * 2;
    const mix_bytes = @as(u64, mix_dim) * 4;
    const residual_bytes = @as(u64, n) * 4;
    const out_bytes = @as(u64, shape.width) * 4;
    if (!model.range(model.weight, weight_bytes) or !model.range(model.scale, 12) or !model.range(model.base, mix_bytes) or !model.range(model.norm, out_bytes)) return h.fail("ds4: Metal compound HC producer parameter range is outside the mapped model");
    const buffers = h.Buffers.from(inputs);
    defer buffers.deinit();
    if (!buffers.residual.fits(residual_bytes) or !buffers.mix.fits(mix_bytes) or !buffers.split.fits(mix_bytes) or !buffers.out.fits(out_bytes) or !buffers.norm_out.fits(out_bytes)) return h.fail("ds4: Metal compound HC producer received undersized activation buffers");
    // All four wrappers run before testing any result, preserving cache/failure order.
    const weights = model.buffer(model.weight, weight_bytes);
    const scale = model.buffer(model.scale, 12);
    const base = model.buffer(model.base, mix_bytes);
    const norm = model.buffer(model.norm, out_bytes);
    defer if (weights) |b| b.deinit();
    defer if (scale) |b| b.deinit();
    defer if (base) |b| b.deinit();
    defer if (norm) |b| b.deinit();
    if (weights == null or scale == null or base == null or norm == null) return 0;
    const kind: u32 = if (expanded) c.DS4_HC_EXPAND_PRODUCER else c.DS4_HC_PRODUCER;
    const pso = h.pipeline(kind);
    if (expanded and (!buffers.block.fits(out_bytes) or !buffers.block_add.fits(out_bytes) or !buffers.previous.fits(residual_bytes) or !buffers.post.fits(@as(u64, shape.hc) * 4) or !buffers.comb.fits(@as(u64, shape.hc) * shape.hc * 4))) return h.fail("ds4: Metal fused HC expand/producer received undersized expand buffers");
    if (pso.handle == null or pso.threads < 512) return 0;
    var completion = std.mem.zeroes(c.ds4_metal_hc_completion);
    if (!c.ds4_metal_hc_control_completion(buffers.mix.view, expanded, &completion)) return 0;
    defer (Buffer{ .view = completion.completion }).deinit();
    defer (Buffer{ .view = completion.barrier }).deinit();
    var launch = h.plan(kind, pso);
    buffers.fill(&launch);
    if (expanded) {
        // Monotonic per-mix counter. Preserve u32 generation and target wrap,
        // and publish before shared-memory/command admission as the C path did.
        const generation = completion.generation +% 1;
        c.ds4_metal_hc_control_generation(buffers.mix.view, generation);
        launch.barrier_target = 6 *% generation;
        launch.expand_args = h.expandPacket(shape, 1, false, true);
    }
    launch.completion = completion.completion;
    launch.barrier = completion.barrier;
    launch.norm_mix_args = .{ .n = h.narrow(n), .out_dim = h.narrow(mix_dim), .eps = rms_eps };
    launch.split_norm_args = h.splitPacket(shape, 1);
    launch.weights = weights.?.view;
    launch.scale = scale.?.view;
    launch.base = base.?.view;
    launch.norm_weights = norm.?.view;
    launch.shared_bytes = (32 + 4 * 32 + 4 + 32) * 4;
    const max_shared = c.ds4_metal_attention_get_facts().max_shared_bytes;
    if (max_shared != 0 and launch.shared_bytes > max_shared) return 0;
    launch.gx = 6;
    launch.tx = 32;
    launch.ty = 16;
    var command = h.Command{ .label = if (expanded) "fused HC expand/producer/pre-norm" else "compound HC producer/pre-norm" };
    defer command.drop();
    if (!command.acquire()) return 0;
    if (c.ds4_metal_hc_control_encode(command.token.?, &launch) == 0) return -1;
    return if (command.finish()) 1 else -1;
}
fn binaryArgs(width: u32, rows: u32, rhs: u32) c.ds4_gpu_bin_args {
    var packet = std.mem.zeroes(c.ds4_gpu_bin_args);
    const row_bytes = @as(u64, width) * 4;
    const rhs_bytes = @as(u64, rhs) * 4;
    packet.ne00 = h.narrow(width);
    packet.ne01 = h.narrow(rows);
    packet.ne02 = 1;
    packet.ne03 = 1;
    packet.nb00 = 4;
    packet.nb01 = row_bytes;
    packet.nb02 = row_bytes;
    packet.nb03 = row_bytes;
    packet.ne10 = h.narrow(rhs);
    packet.ne11 = 1;
    packet.ne12 = 1;
    packet.ne13 = 1;
    packet.nb10 = 4;
    packet.nb11 = rhs_bytes;
    packet.nb12 = rhs_bytes;
    packet.nb13 = rhs_bytes;
    packet.ne0 = h.narrow(width);
    packet.ne1 = h.narrow(rows);
    packet.ne2 = 1;
    packet.ne3 = 1;
    packet.nb0 = 4;
    packet.nb1 = row_bytes;
    packet.nb2 = row_bytes;
    packet.nb3 = row_bytes;
    return packet;
}
fn unaryArgs(width: u32, rows: u32, scale: f32, bias: f32) c.ds4_metal_attention_unary_args {
    var packet = std.mem.zeroes(c.ds4_metal_attention_unary_args);
    const row_bytes = @as(u64, width) * 4;
    packet.ne00 = h.narrow(width / 4);
    packet.ne01 = h.narrow(rows);
    packet.ne02 = 1;
    packet.ne03 = 1;
    packet.nb00 = 4;
    packet.nb01 = row_bytes;
    packet.nb02 = row_bytes;
    packet.nb03 = row_bytes;
    packet.ne0 = h.narrow(width / 4);
    packet.ne1 = h.narrow(rows);
    packet.ne2 = 1;
    packet.ne3 = 1;
    packet.nb0 = 4;
    packet.nb1 = row_bytes;
    packet.nb2 = row_bytes;
    packet.nb3 = row_bytes;
    packet.scale = scale;
    packet.bias = bias;
    return packet;
}
fn binaryThreads(width: u32, pso: h.Pipeline) u64 {
    var threads: u64 = 1;
    while (2 * threads < width and threads < @min(256, pso.threads)) threads *= 2;
    return threads;
}
pub fn outputWeights(inputs: Inputs, model: Model, hc: u32, eps: f32) c_int {
    if (!h.init() or inputs.out == null or inputs.input == null or model.map == null or hc == 0) return 0;
    if (hc % 4 != 0) return h.fail("ds4: Metal output HC weights requires a multiple-of-4 HC width");
    const buffers = h.Buffers.from(inputs);
    defer buffers.deinit();
    const row_bytes = @as(u64, hc) * 4;
    if (buffers.out.view.bytes < row_bytes or buffers.out.view.bytes % row_bytes != 0) return h.fail("ds4: Metal output HC weights size is not a whole token row");
    const rows = buffers.out.view.bytes / row_bytes;
    if (rows == 0 or rows > max32 or rows > max64 / row_bytes) return h.fail("ds4: Metal output HC weights token count is outside supported range");
    if (!buffers.input.fits(rows * row_bytes) or !buffers.out.fits(rows * row_bytes)) return h.fail("ds4: Metal output HC weights received undersized buffers");
    const scale = model.buffer(model.scale, 4);
    const base = model.buffer(model.base, row_bytes);
    defer if (scale) |b| b.deinit();
    defer if (base) |b| b.deinit();
    if (scale == null or base == null) return 0;
    const facts = c.ds4_metal_attention_get_facts();
    const fast = h.pipeline(c.DS4_HC_OUTPUT4);
    const fast_shape = hc == 4 and rows == 1;
    const use_fast = fast_shape and facts.flags & c.DS4_ATTN_QUALITY == 0 and fast.handle != null and (facts.flags & c.DS4_ATTN_M3 != 0 or c.ds4_metal_hc_control_output4_test()) and fast.threads >= 2;
    if (h.present("DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4") and fast_shape and !use_fast) return h.fail("ds4: required Metal output HC weights4 kernel was not selected");
    var launch = h.plan(if (use_fast) c.DS4_HC_OUTPUT4 else c.DS4_HC_OUTPUT_STAGES, fast);
    buffers.fill(&launch);
    launch.scale = scale.?.view;
    launch.base = base.?.view;
    if (use_fast) {
        launch.output4_args = .{ .post_scale = 1, .eps = eps };
        launch.tx = 2;
    } else {
        // These four dependent stages share ONE encoder, as in the original.
        launch.mul_args = binaryArgs(hc, @intCast(rows), 1);
        launch.add_args = binaryArgs(hc, @intCast(rows), hc);
        launch.sigmoid_args = unaryArgs(hc, @intCast(rows), 0, 0);
        launch.scale_args = unaryArgs(hc, @intCast(rows), 1, eps);
        launch.mul = h.pipeline(c.DS4_HC_OUTPUT_MUL);
        launch.add_pipeline = h.pipeline(c.DS4_HC_OUTPUT_ADD);
        launch.sigmoid = h.pipeline(c.DS4_HC_OUTPUT_SIGMOID);
        launch.scale_pipeline = h.pipeline(c.DS4_HC_OUTPUT_SCALE);
        launch.mul_threads = binaryThreads(hc, launch.mul);
        launch.add_threads = binaryThreads(hc, launch.add_pipeline);
        launch.unary_threads = @max(1, @min(hc / 4, @min(256, launch.sigmoid.threads)));
        // The shader row field is signed; preserve the C NSUInteger conversion.
        const shader_rows: u64 = @bitCast(@as(i64, launch.sigmoid_args.ne01));
        launch.unary_groups = ((@as(u64, hc / 4) + launch.unary_threads - 1) / launch.unary_threads) *% shader_rows;
    }
    return h.run(launch, if (use_fast) "output HC weights4" else "output HC weights");
}
pub fn q8Expand(inputs: Inputs, model: Model, shape: Shape, input_width: u64, output_width: u64, shared: bool) c_int {
    if (!h.init() or inputs.out == null or inputs.block == null or inputs.input == null or inputs.residual == null or inputs.split == null or model.map == null or shape.width == 0 or shape.hc != 4 or output_width != shape.width or input_width & 31 != 0 or input_width > max32 or output_width > max32 or (shared and inputs.block_add == null)) return 0;
    const buffers = h.Buffers.from(inputs);
    defer buffers.deinit();
    const weight_row = (input_width / 32) * 34;
    // Original u64 product wraps after the u32 dimension admission.
    const weight_bytes = output_width *% weight_row;
    if (!model.range(model.weight, weight_bytes)) return h.fail(if (shared) "ds4: Metal shared-down HC fusion weight range is outside the mapped model" else "ds4: Metal Q8 HC fusion weight range is outside the mapped model");
    if (!buffers.input.fits(input_width * 4) or !buffers.block.fits(output_width * 4) or !buffers.residual.fits(@as(u64, shape.hc) * shape.width * 4) or !buffers.split.fits(shape.mix() * 4) or !buffers.out.fits(@as(u64, shape.hc) * shape.width * 4) or (shared and !buffers.block_add.fits(output_width * 4))) return h.fail(if (shared) "ds4: Metal shared-down HC fusion received undersized buffers" else "ds4: Metal Q8 HC fusion received undersized buffers");
    const weights = model.buffer(model.weight, weight_bytes) orelse return 0;
    defer weights.deinit();
    const facts = c.ds4_metal_attention_get_facts();
    const fallback: u64 = if (c.ds4_metal_hc_control_tp_two()) 2 else 4;
    const nsg: i16 = @intCast(c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", fallback, 1, 8));
    const vector = !shared and buffers.split.view.offset & 15 == 0 and !h.present("DS4_METAL_DISABLE_M5_Q8_HC_VEC") and facts.flags & c.DS4_ATTN_M5 != 0;
    const name: [*:0]const u8 = if (shared) "kernel_dsv4_shared_down_hc_expand4_q8_0" else if (vector) "kernel_dsv4_q8_hc_expand4_q8_0_vec_hc" else "kernel_dsv4_q8_hc_expand4_q8_0";
    var pso = c.ds4_metal_routed_pipeline(1, name, nsg);
    if (pso == null and vector) pso = c.ds4_metal_routed_pipeline(1, "kernel_dsv4_q8_hc_expand4_q8_0", nsg);
    if (pso == null) return 0;
    var launch = h.plan(if (shared) c.DS4_HC_SHARED_DOWN else c.DS4_HC_MATMUL, .{ .handle = pso, .threads = 0, .simd_width = 0 });
    buffers.fill(&launch);
    launch.mv_args = .{
        .ne00 = h.narrow(input_width),
        .ne01 = h.narrow(output_width),
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = weight_row,
        .nb02 = weight_bytes,
        .nb03 = weight_bytes,
        .ne10 = h.narrow(input_width),
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = input_width * 4,
        .nb12 = input_width * 4,
        .nb13 = input_width * 4,
        .ne0 = h.narrow(output_width),
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
    launch.expand_args = h.expandPacket(shape, 1, true, shared);
    launch.weights = weights.view;
    launch.post = h.at(buffers.split.view, @as(u64, shape.hc) * 4);
    launch.comb = h.at(buffers.split.view, @as(u64, 2 *% shape.hc) * 4);
    launch.shared_bytes = 256;
    launch.gx = (output_width + 1) / 2;
    launch.tx = 32;
    launch.ty = @intCast(nsg);
    return h.run(launch, if (shared) "shared-down HC expand fused" else "Q8 HC expand fused");
}
