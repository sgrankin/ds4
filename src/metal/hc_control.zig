//! Hyper-Connections packets, shape admission and dispatch sequencing. The MSL
//! reduction trees remain unchanged; fixed ARC leaves only bind named resources.
const std = @import("std");
pub const c = @import("metal_c");
pub const api = @import("bridge.zig");
pub const Buffer = @import("attention_decode.zig").Buffer;
pub const Command = @import("routed_batch.zig").Command;
pub const Tensor = api.Tensor;
pub const View = c.ds4_metal_attention_view;
pub const Pipeline = c.ds4_metal_attention_pipeline;
pub const Launch = c.ds4_metal_hc_control_launch;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
pub fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn fail(line: [*:0]const u8) c_int {
    _ = api.reportError(line);
    return 0;
}
pub fn narrow(value: u64) i32 {
    return @bitCast(@as(u32, @truncate(value)));
}
pub fn at(view: View, offset: u64) View {
    var result = view;
    result.offset +%= offset;
    return result;
}
pub fn init() bool {
    return api.init() != 0;
}
pub fn pipeline(kind: u32) Pipeline {
    return c.ds4_metal_hc_control_pipeline(kind);
}
pub fn linear(elements: u64) u64 {
    return @min(256, @max(1, elements));
}
pub fn plan(kind: u32, pso: Pipeline) Launch {
    var result = std.mem.zeroes(Launch);
    result.kind = kind;
    result.pipeline = pso;
    result.gx = 1;
    result.gy = 1;
    result.gz = 1;
    result.tx = 1;
    result.ty = 1;
    return result;
}
pub fn run(launch: Launch, label: [*:0]const u8) c_int {
    var command = Command{ .label = label };
    defer command.drop();
    if (!command.acquire()) return 0;
    if (c.ds4_metal_hc_control_encode(command.token.?, &launch) == 0) return 0;
    return @intFromBool(command.finish());
}
pub const Model = struct {
    map: ?*const anyopaque = null,
    size: u64 = 0,
    scale: u64 = 0,
    base: u64 = 0,
    norm: u64 = 0,
    weight: u64 = 0,
    pub fn range(self: Model, offset: u64, bytes: u64) bool {
        return offset <= self.size and bytes <= self.size - offset;
    }
    pub fn buffer(self: Model, offset: u64, bytes: u64) ?Buffer {
        var inner: u64 = 0;
        const pointer = api.retainModelRange(self.map, self.size, offset, bytes, &inner) orelse return null;
        return .{ .view = .{ .buffer = pointer, .offset = inner, .bytes = bytes } };
    }
};
pub const Shape = struct {
    width: u32 = 0,
    hc: u32 = 0,
    iterations: u32 = 0,
    eps: f32 = 0,
    norm_eps: f32 = 0,
    pub fn mix(self: Shape) u64 {
        return 2 * @as(u64, self.hc) + @as(u64, self.hc) * self.hc;
    }
};
pub const Inputs = struct {
    out: ?*const Tensor = null,
    norm_out: ?*const Tensor = null,
    mix: ?*const Tensor = null,
    split: ?*const Tensor = null,
    residual: ?*const Tensor = null,
    block: ?*const Tensor = null,
    block_add: ?*const Tensor = null,
    previous: ?*const Tensor = null,
    post: ?*const Tensor = null,
    comb: ?*const Tensor = null,
    input: ?*const Tensor = null,
};
pub const Buffers = struct {
    out: Buffer,
    norm_out: Buffer,
    mix: Buffer,
    split: Buffer,
    residual: Buffer,
    block: Buffer,
    block_add: Buffer,
    previous: Buffer,
    post: Buffer,
    comb: Buffer,
    input: Buffer,
    pub fn from(inputs: Inputs) Buffers {
        var result: Buffers = undefined;
        inline for (comptime std.meta.fieldNames(Buffers)) |field| @field(result, field) = if (@field(inputs, field)) |tensor| Buffer.tensor(tensor) else .{ .view = std.mem.zeroes(View) };
        return result;
    }
    pub fn deinit(self: Buffers) void {
        inline for (comptime std.meta.fieldNames(Buffers)) |field| @field(self, field).deinit();
    }
    pub fn fill(self: Buffers, launch: *Launch) void {
        inline for (comptime std.meta.fieldNames(Buffers)) |field| @field(launch, field) = @field(self, field).view;
    }
};
pub fn splitPacket(shape: Shape, rows: u64) c.ds4_gpu_hc_split_weighted_sum_norm_args {
    const row_bytes = @as(u64, shape.width) * 4;
    return .{
        .n_embd = shape.width,
        .n_hc = narrow(shape.hc),
        .sinkhorn_iters = narrow(shape.iterations),
        .n_rows = @intCast(rows),
        .mix_hc = @intCast(shape.mix()),
        .nb_mix1 = shape.mix() * 4,
        .nb_split1 = shape.mix() * 4,
        .nb_x0 = 4,
        .nb_x1 = row_bytes,
        .nb_x2 = @as(u64, shape.hc) * row_bytes,
        .nb0 = 4,
        .nb1 = row_bytes,
        .nb_norm1 = row_bytes,
        .eps = shape.eps,
        .norm_eps = shape.norm_eps,
    };
}
pub fn expandPacket(shape: Shape, rows: u64, split: bool, add: bool) c.ds4_gpu_hc_expand_args {
    const width_bytes = @as(u64, shape.width) * 4;
    const hc_bytes = @as(u64, shape.hc) * width_bytes;
    const post_bytes = @as(u64, shape.hc) * 4;
    const mix_bytes = shape.mix() * 4;
    return .{
        .n_embd = shape.width,
        .n_hc = shape.hc,
        .n_tokens = @intCast(rows),
        .nb_block0 = 4,
        .nb_block1 = width_bytes,
        .nb_add0 = 4,
        .nb_add1 = width_bytes,
        .nb_res0 = 4,
        .nb_res1 = width_bytes,
        .nb_res2 = hc_bytes,
        .nb_post0 = 4,
        .nb_post1 = if (split) mix_bytes else post_bytes,
        .nb_comb0 = 4,
        .nb_comb1 = post_bytes,
        .nb_comb2 = if (split) mix_bytes else @as(u64, shape.hc) * post_bytes,
        .nb0 = 4,
        .nb1 = width_bytes,
        .nb2 = hc_bytes,
        .has_add = @intFromBool(add),
    };
}
pub fn sinkhorn(inputs: Inputs, model: Model, shape: Shape) c_int {
    if (!init() or shape.hc == 0 or shape.hc > 16) return 0;
    const buffers = Buffers.from(inputs);
    defer buffers.deinit();
    const mix_bytes = shape.mix() * 4;
    if (!buffers.mix.fits(mix_bytes) or !buffers.out.fits(mix_bytes)) return fail("ds4: Metal HC split received undersized activation buffers");
    if (!model.range(model.scale, 12) or !model.range(model.base, mix_bytes)) return fail("ds4: Metal HC split parameter range is outside the mapped model");
    const scale = model.buffer(model.scale, 12);
    const base = model.buffer(model.base, mix_bytes);
    defer if (scale) |buffer| buffer.deinit();
    defer if (base) |buffer| buffer.deinit();
    if (scale == null or base == null) return 0;
    const rows = @min(buffers.mix.view.bytes / mix_bytes, buffers.out.view.bytes / mix_bytes);
    if (rows == 0 or rows > max32) return fail("ds4: Metal HC split row count is outside supported range");
    var launch = plan(c.DS4_HC_SPLIT, pipeline(c.DS4_HC_SPLIT));
    buffers.fill(&launch);
    launch.split_args = .{ .n_hc = narrow(shape.hc), .sinkhorn_iters = narrow(shape.iterations), .n_rows = @intCast(rows), .mix_hc = @intCast(shape.mix()), .nb01 = mix_bytes, .nb1 = mix_bytes, .eps = shape.eps };
    launch.scale = scale.?.view;
    launch.base = base.?.view;
    launch.tx = linear(rows);
    launch.gx = (rows + launch.tx - 1) / launch.tx;
    return run(launch, "HC split/sinkhorn");
}
pub fn weighted(inputs: Inputs, shape: Shape, stride: u64, label: [*:0]const u8) c_int {
    if (!init() or inputs.out == null or inputs.residual == null or inputs.post == null or shape.width == 0 or shape.hc == 0 or stride < @as(u64, shape.hc) * 4) return 0;
    const buffers = Buffers.from(inputs);
    defer buffers.deinit();
    const row_bytes = @as(u64, shape.width) * 4;
    if (buffers.out.view.bytes < row_bytes or buffers.out.view.bytes % row_bytes != 0) return fail("ds4: Metal HC weighted sum output size is not a whole token row");
    const rows = buffers.out.view.bytes / row_bytes;
    if (rows == 0 or rows > max32) return fail("ds4: Metal HC weighted sum token count is outside supported range");
    const x_values = @as(u64, shape.hc) * shape.width;
    if (x_values == 0 or x_values > max64 / 4 or rows > max64 / (x_values * 4) or rows > max64 / (@as(u64, shape.hc) * 4)) return fail("ds4: Metal HC weighted sum activation size overflow");
    // Preserve the original unsigned final-row address calculation.
    const last = (rows - 1) *% stride +% @as(u64, shape.hc) * 4;
    if (!buffers.residual.fits(rows * x_values * 4) or !buffers.post.fits(last) or !buffers.out.valid()) return fail("ds4: Metal HC weighted sum received undersized activation buffers");
    var launch = plan(c.DS4_HC_SUM, pipeline(c.DS4_HC_SUM));
    buffers.fill(&launch);
    launch.sum_args = .{ .n_embd = shape.width, .n_hc = shape.hc, .n_tokens = @intCast(rows), .nb_x0 = 4, .nb_x1 = row_bytes, .nb_x2 = x_values * 4, .nb_w0 = 4, .nb_w1 = stride, .nb0 = 4, .nb1 = row_bytes };
    const elements = @as(u64, shape.width) * rows;
    launch.tx = linear(elements);
    launch.gx = (elements + launch.tx - 1) / launch.tx;
    return run(launch, label);
}
// HC-pre is a direct fused dispatch. The graph driver owns optional fallback.
// Weighted RMSNorm preserves the established single-row and batched formulas.
pub fn splitSum(inputs: Inputs, model: Model, shape: Shape, normalized: bool) c_int {
    if (!init() or inputs.out == null or inputs.split == null or inputs.mix == null or inputs.residual == null or model.map == null) return 0;
    if (normalized) {
        if (inputs.norm_out == null or shape.hc != 4 or shape.width & 3 != 0) return 0;
    } else {
        if (shape.width == 0 or shape.hc == 0) return 0;
        if (shape.hc != 4) return fail("ds4: Metal fused HC split/sum is specialized for HC=4");
    }
    const buffers = Buffers.from(inputs);
    defer buffers.deinit();
    const row_bytes = @as(u64, shape.width) * 4;
    const mix_bytes = shape.mix() * 4;
    const residual_bytes = @as(u64, shape.hc) * row_bytes;
    if (row_bytes == 0 or buffers.out.view.bytes < row_bytes or buffers.out.view.bytes % row_bytes != 0) return fail(if (normalized) "ds4: Metal fused HC split/sum/norm output size is not a whole token row" else "ds4: Metal fused HC split/sum output size is not a whole token row");
    const rows = buffers.out.view.bytes / row_bytes;
    if (rows == 0 or rows > max32 or rows > max64 / mix_bytes or rows > max64 / residual_bytes) return fail(if (normalized) "ds4: Metal fused HC split/sum/norm row count is outside supported range" else "ds4: Metal fused HC split/sum row count is outside supported range");
    if (!buffers.mix.fits(rows * mix_bytes) or !buffers.split.fits(rows * mix_bytes) or !buffers.residual.fits(rows * residual_bytes) or !buffers.out.valid() or (normalized and !buffers.norm_out.fits(rows * row_bytes))) return fail(if (normalized) "ds4: Metal fused HC split/sum/norm received undersized activation buffers" else "ds4: Metal fused HC split/sum received undersized activation buffers");
    if (!model.range(model.scale, 12) or !model.range(model.base, mix_bytes) or (normalized and !model.range(model.norm, row_bytes))) return fail(if (normalized) "ds4: Metal fused HC split/sum/norm parameter range is outside the mapped model" else "ds4: Metal fused HC split/sum parameter range is outside the mapped model");
    const scale = model.buffer(model.scale, 12);
    const base = model.buffer(model.base, mix_bytes);
    const norm = if (normalized) model.buffer(model.norm, row_bytes) else null;
    defer if (scale) |b| b.deinit();
    defer if (base) |b| b.deinit();
    defer if (norm) |b| b.deinit();
    if (scale == null or base == null or (normalized and norm == null)) return 0;
    const kind: u32 = if (normalized) c.DS4_HC_SPLIT_NORM else c.DS4_HC_SPLIT_SUM;
    const pso = pipeline(kind);
    if (normalized and pso.handle == null) return 0;
    var launch = plan(kind, pso);
    buffers.fill(&launch);
    const packet = splitPacket(shape, rows);
    if (normalized) launch.split_norm_args = packet else {
        inline for (comptime std.meta.fieldNames(c.ds4_gpu_hc_split_weighted_sum_args)) |field| @field(launch.split_sum_args, field) = @field(packet, field);
    }
    launch.scale = scale.?.view;
    launch.base = base.?.view;
    launch.gx = rows;
    if (normalized) {
        var threads: u64 = 32;
        while (threads < shape.width / 4 and threads < 1024) threads *= 2;
        launch.tx = @max(1, @min(threads, shape.width / 4));
        if (launch.tx > pso.threads) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal fused HC split/sum/norm requires %lu threads but pipeline supports %lu\n", launch.tx, pso.threads);
            return 0;
        }
        launch.shared_bytes = (@as(u64, shape.width) + 4 + 32) * 4;
        const max_shared = c.ds4_metal_attention_get_facts().max_shared_bytes;
        if (max_shared != 0 and launch.shared_bytes > max_shared) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal fused HC split/sum/norm requires %lu bytes of threadgroup memory but device supports %lu\n", launch.shared_bytes, max_shared);
            return 0;
        }
        launch.norm_weights = norm.?.view;
    } else {
        launch.tx = @max(1, @min(@min(pso.threads, 256), shape.width));
        launch.shared_bytes = @as(u64, shape.hc) * 4;
    }
    return run(launch, if (normalized) "HC split/sum/norm fused" else "HC split/sum fused");
}
// TP expand-add keeps canonical rank order: block first, then block_add.
pub fn expand(inputs: Inputs, shape: Shape, split: bool, add: bool) c_int {
    const verbose = split and !add;
    if (!init()) return if (verbose) fail("ds4: Metal HC expand split could not initialize the backend") else 0;
    if (shape.width == 0 or shape.hc == 0 or (add and !split and shape.hc != 4) or (split and (inputs.out == null or inputs.block == null or inputs.residual == null or inputs.split == null or (add and inputs.block_add == null)))) return if (verbose) fail("ds4: Metal HC expand split received invalid arguments") else 0;
    const buffers = Buffers.from(inputs);
    defer buffers.deinit();
    const hc_values = @as(u64, shape.hc) * shape.width;
    const row_bytes = hc_values *% 4;
    if (row_bytes == 0 or buffers.out.view.bytes < row_bytes or buffers.out.view.bytes % row_bytes != 0) return fail(if (split) (if (add) "ds4: Metal HC expand add split output size is not a whole HC token row" else "ds4: Metal HC expand split output size is not a whole HC token row") else "ds4: Metal HC expand output size is not a whole HC token row");
    const rows = buffers.out.view.bytes / row_bytes;
    if (rows == 0 or rows > max32) return fail(if (split) (if (add) "ds4: Metal HC expand add split token count is outside supported range" else "ds4: Metal HC expand split token count is outside supported range") else "ds4: Metal HC expand token count is outside supported range");
    const block_bytes = @as(u64, shape.width) * 4;
    const mixing_values = if (split) shape.mix() else @as(u64, shape.hc) * shape.hc;
    if (hc_values == 0 or hc_values > max64 / 4 or mixing_values > max64 / 4 or rows > max64 / block_bytes or rows > max64 / (hc_values * 4) or rows > max64 / (mixing_values * 4)) return fail(if (split) (if (add) "ds4: Metal HC expand add split activation size overflow" else "ds4: Metal HC expand split activation size overflow") else "ds4: Metal HC expand activation size overflow");
    if (!buffers.block.fits(rows * block_bytes) or !buffers.residual.fits(rows * hc_values * 4) or !buffers.out.valid() or (add and !buffers.block_add.fits(rows * block_bytes)) or (split and !buffers.split.fits(rows * mixing_values * 4)) or (!split and (!buffers.post.fits(rows * @as(u64, shape.hc) * 4) or !buffers.comb.fits(rows * mixing_values * 4)))) return fail(if (split) (if (add) "ds4: Metal HC expand add split received undersized activation buffers" else "ds4: Metal HC expand split received undersized activation buffers") else "ds4: Metal HC expand received undersized activation buffers");
    const pso = pipeline(if (shape.hc == 4) c.DS4_HC_EXPAND4 else c.DS4_HC_EXPAND);
    if (pso.handle == null) return if (verbose) fail("ds4: Metal HC expand split pipeline is unavailable") else 0;
    var launch = plan(c.DS4_HC_EXPAND, pso);
    buffers.fill(&launch);
    launch.expand_args = expandPacket(shape, rows, split, add);
    if (split) {
        launch.post = at(buffers.split.view, @as(u64, shape.hc) * 4);
        launch.comb = at(buffers.split.view, @as(u64, 2 *% shape.hc) * 4);
    }
    if (!add) launch.block_add = buffers.block.view;
    const elements = @as(u64, shape.width) *% rows *% (if (shape.hc == 4) @as(u64, 1) else shape.hc);
    launch.tx = linear(elements);
    launch.gx = (elements + launch.tx - 1) / launch.tx;
    if (verbose) {
        var command = Command{ .label = "HC expand split" };
        defer command.drop();
        if (!command.acquire()) return fail("ds4: Metal HC expand split could not obtain a command buffer");
        if (c.ds4_metal_hc_control_encode(command.token.?, &launch) == 0) return 0;
        return @intFromBool(command.finish());
    }
    return run(launch, if (split) "HC expand add split" else if (add) "HC expand add" else "HC expand");
}
