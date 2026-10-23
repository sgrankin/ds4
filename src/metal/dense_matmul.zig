//! Ordinary Q8/F16 dense projections. The immutable request describes tensor
//! views; every byte offset below is relative to those views until encoding.
//! Shader math is unchanged. Owned model/scratch references survive encoding
//! and any owned-command wait; a failed sequence never commits its owned CB.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Tensor = api.Tensor;
const Pso = c.ds4_metal_hc_pipeline;
const Args = c.ds4_metal_dense_args;
const Mv = c.ds4_metal_mv_args;
const Ext = c.ds4_metal_mv_ext_args;
const Mm = c.ds4_metal_mm_args;
const max_u32 = std.math.maxInt(u32);
const max_u64 = std.math.maxInt(u64);

pub const Request = struct {
    out: ?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    offset: u64,
    width: u64,
    outputs: u64,
    x: ?*const Tensor,
    rows: u64,

    fn shapesFit(self: Request) bool {
        return self.width <= max_u32 and self.outputs <= max_u32 and self.rows <= max_u32;
    }
    fn activations(self: Request, input: u64, output: u64, message: ?[*:0]const u8) bool {
        if (api.tensorHasBuffer(self.x) != 0 and api.tensorHasBuffer(self.out) != 0 and
            api.tensorBytes(self.x) >= input and api.tensorBytes(self.out) >= output) return true;
        if (message) |line| _ = api.reportError(line);
        return false;
    }
    fn weight(self: Request, bytes: u64, message: ?[*:0]const u8) ?Weight {
        if (self.offset > self.size or bytes > self.size - self.offset) {
            if (message) |line| _ = api.reportError(line);
            return null;
        }
        var inner: u64 = 0;
        const buffer = api.retainModelRange(self.map, self.size, self.offset, bytes, &inner) orelse return null;
        return .{ .buffer = buffer, .offset = inner };
    }
};
const Weight = struct {
    buffer: *api.ModelBuffer,
    offset: u64,
    fn deinit(self: Weight) void {
        api.releaseModelBuffer(self.buffer);
    }
};
const Command = struct {
    token: ?*anyopaque,
    owned: c_int,
    fn init() ?Command {
        var owned: c_int = 0;
        const token = api.acquire(&owned) orelse return null;
        return .{ .token = token, .owned = owned };
    }
    fn deinit(self: *Command) void {
        if (self.token) |token| _ = api.release(token, self.owned, 0, "dense matmul");
    }
    fn finish(self: *Command, label: [*:0]const u8) c_int {
        const token = self.token.?;
        self.token = null;
        return api.release(token, self.owned, 1, label);
    }
};
const Program = struct {
    kind: u32,
    name: [*:0]const u8,
    nsg: i16 = 0,
    nxpsg: i16 = 0,
    bc_in: bool = false,
    bc_out: bool = false,
    fn pipeline(self: Program) ?*const Pso {
        return c.ds4_metal_dense_pipeline(self.kind, self.name, self.nsg, self.nxpsg, self.bc_in, self.bc_out);
    }
};
const Grid = struct {
    x: u64,
    y: u64 = 1,
    threads_x: u64 = 32,
    threads_y: u64 = 1,
    shared: u32 = 0,
    x_offset: u64 = 0,
    out_offset: u64 = 0,
    repeats: u64 = 1,
    x_step: u64 = 0,
    out_step: u64 = 0,
};
fn encode(command: *const Command, request: Request, weights: Weight, kind: u32, pipeline: *const Pso, args: Args, grid: Grid) bool {
    const launch = c.ds4_metal_dense_launch{
        .kind = kind,
        .args = args,
        .pipeline = pipeline,
        .weights = weights.buffer,
        .weight_offset = weights.offset,
        .x = request.x,
        .out = request.out,
        .x_offset = grid.x_offset,
        .out_offset = grid.out_offset,
        .groups_x = grid.x,
        .groups_y = grid.y,
        .threads_x = grid.threads_x,
        .threads_y = grid.threads_y,
        .repeats = grid.repeats,
        .x_step = grid.x_step,
        .out_step = grid.out_step,
        .shared_bytes = grid.shared,
    };
    return c.ds4_metal_dense_encode(command.token.?, &launch) != 0;
}
fn dispatch(command: *const Command, request: Request, weights: Weight, program: Program, args: Args, grid: Grid) bool {
    const pipeline = program.pipeline() orelse return false;
    return encode(command, request, weights, program.kind, pipeline, args, grid);
}
fn narrow(value: u64) i32 {
    // C narrows admitted uint32 dimensions to the shader's int32 bit pattern.
    return @bitCast(@as(u32, @truncate(value)));
}
pub fn mvArgs(width: u64, outputs: u64, rows: u64, q8: bool, per_group: i32) Mv {
    const row = if (q8) (width / 32) * 34 else width * 2;
    const weights = row *% outputs;
    const input = width *% rows *% 4;
    return .{
        .ne00 = narrow(width),
        .ne01 = narrow(outputs),
        .ne02 = 1,
        .nb00 = if (q8) 34 else 2,
        .nb01 = row,
        .nb02 = weights,
        .nb03 = weights,
        .ne10 = narrow(width),
        .ne11 = narrow(rows),
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = width * 4,
        .nb12 = input,
        .nb13 = input,
        .ne0 = narrow(outputs),
        .ne1 = narrow(rows),
        .nr0 = per_group,
        .r2 = 1,
        .r3 = 1,
    };
}
pub fn extArgs(request: Request, q8: bool) Ext {
    const mv = mvArgs(request.width, request.outputs, request.rows, q8, 2);
    return .{
        .ne00 = mv.ne00,
        .ne01 = mv.ne01,
        .ne02 = 1,
        .nb00 = mv.nb00,
        .nb01 = mv.nb01,
        .nb02 = mv.nb02,
        .nb03 = mv.nb03,
        .ne10 = mv.ne10,
        .ne11 = mv.ne11,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = mv.nb11,
        .nb12 = mv.nb12,
        .nb13 = mv.nb13,
        .ne0 = mv.ne0,
        .ne1 = mv.ne1,
        .r2 = 1,
        .r3 = 1,
    };
}
pub fn mmArgs(request: Request, rows: u64, row_bytes: u64) Mm {
    const weights = row_bytes *% request.outputs;
    const input = request.width *% rows *% 4;
    return .{
        .ne00 = narrow(request.width),
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = weights,
        .nb03 = weights,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = request.width * 4,
        .nb12 = input,
        .nb13 = input,
        .ne0 = narrow(request.outputs),
        .ne1 = narrow(rows),
        .r2 = 1,
        .r3 = 1,
    };
}
const Matvec = struct { name: [*:0]const u8, nsg: i16, rows: i32 = 2, shared: u32 = 256 };
fn q8Dispatch(outputs: u64, facts: u32) Matvec {
    const fallback: u64 = if (facts & c.DS4_DENSE_TP_TWO != 0) 2 else 4;
    const groups = c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", fallback, 1, 8);
    return .{ .name = "kernel_mul_mv_q8_0_f32", .nsg = if (outputs > 65536) 8 else @intCast(groups) };
}
fn halfDispatch(width: u64) Matvec {
    if (width < 32) return .{ .name = "kernel_mul_mv_f16_f32_short", .nsg = 1, .rows = 32, .shared = 0 };
    return .{ .name = if (width % 4 == 0) "kernel_mul_mv_f16_f32_4" else "kernel_mul_mv_f16_f32", .nsg = @intCast(@min((width + 127) / 128, @as(u64, 8))) };
}
pub fn extRows(rows: u64) i16 {
    return switch (rows) {
        2 => 2,
        3, 6 => 3,
        4, 7, 8 => 4,
        5 => 5,
        else => if (rows > 8) 4 else 0,
    };
}
fn extName(q8: bool, rows: i16) ?[*:0]const u8 {
    return if (q8) switch (rows) {
        2 => "kernel_mul_mv_ext_q8_0_f32_r1_2",
        3 => "kernel_mul_mv_ext_q8_0_f32_r1_3",
        4 => "kernel_mul_mv_ext_q8_0_f32_r1_4",
        5 => "kernel_mul_mv_ext_q8_0_f32_r1_5",
        else => null,
    } else switch (rows) {
        2 => "kernel_mul_mv_ext_f16_f32_r1_2",
        3 => "kernel_mul_mv_ext_f16_f32_r1_3",
        4 => "kernel_mul_mv_ext_f16_f32_r1_4",
        5 => "kernel_mul_mv_ext_f16_f32_r1_5",
        else => null,
    };
}
fn fewRows(command: *const Command, request: Request, weights: Weight, q8: bool, facts: u32) bool {
    const nsg: i16 = @intCast(c.ds4_zig_metal_env_u64("DS4_METAL_MV_EXT_NSG", 2, 1, 8));
    const nxpsg: i16 = if (request.width % 256 == 0 and (request.rows < 3 or (request.rows == 3 and facts & c.DS4_DENSE_VERIFY_EXACT != 0))) 16 else if (request.width % 128 == 0) 8 else 4;
    const r1 = extRows(request.rows);
    const name = extName(q8, r1) orelse return false;
    const r0: u64 = @as(u64, @intCast(@divTrunc(32, nxpsg))) * @as(u64, @intCast(nsg));
    const token_rows: u64 = @intCast(r1);
    return dispatch(command, request, weights, .{ .kind = c.DS4_DENSE_EXT, .name = name, .nsg = nsg, .nxpsg = nxpsg }, .{ .ext = extArgs(request, q8) }, .{ .x = (request.outputs + r0 - 1) / r0, .y = (request.rows + token_rows - 1) / token_rows, .threads_y = @intCast(nsg) });
}
fn tile(rows: u64) u64 {
    return if (rows % 128 == 0) 128 else if (rows % 64 == 0) 64 else 32;
}
fn naxName(q8: bool, rows: u64) [*:0]const u8 {
    return if (q8) switch (rows) {
        128 => "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128",
        64 => "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64",
        else => "kernel_mul_mm_q8_0_f32_nax_direct_rhs",
    } else switch (rows) {
        128 => "kernel_mul_mm_f16_f32_mpp_direct_rhs_n128",
        64 => "kernel_mul_mm_f16_f32_mpp_direct_rhs_n64",
        else => "kernel_mul_mm_f16_f32_mpp_direct_rhs",
    };
}
fn matvec(command: *const Command, request: Request, weights: Weight, mode: Matvec, q8: bool) bool {
    const nr: u64 = @intCast(mode.rows);
    return dispatch(command, request, weights, .{ .kind = c.DS4_DENSE_MV, .name = mode.name, .nsg = mode.nsg }, .{ .mv = mvArgs(request.width, request.outputs, request.rows, q8, mode.rows) }, .{ .x = (request.outputs + nr - 1) / nr, .y = request.rows, .threads_y = @intCast(mode.nsg), .shared = mode.shared });
}

/// Whole Q8 backend used by public dense/MPP entries and the remaining general
/// quant dispatcher. Aligned TensorOps prefix + boundary-safe tail preserve the
/// original schedule; quality mode deliberately keeps one generic dispatch.
pub fn q8Legacy(request: Request, prefer_decode_mpp: bool, prefill_default: bool) c_int {
    if (api.init() == 0 or request.width & 31 != 0 or !request.shapesFit()) return 0;
    if (!request.activations(request.rows *% request.width *% 4, request.rows *% request.outputs *% 4, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers")) return 0;
    const row_bytes = (request.width / 32) * 34;
    const weights = request.weight(request.outputs *% row_bytes, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model") orelse return 0;
    defer weights.deinit();
    var command = Command.init() orelse return 0;
    defer command.deinit();
    const facts = c.ds4_metal_dense_facts(false);
    if (request.rows == 1) {
        if (facts & c.DS4_DENSE_MPP != 0 and prefer_decode_mpp and request.width % 64 == 0) {
            const program = Program{ .kind = c.DS4_DENSE_MM, .name = naxName(true, 32) };
            if (program.pipeline()) |pipeline| {
                if (!encode(&command, request, weights, c.DS4_DENSE_MM, pipeline, .{ .mm = mmArgs(request, 1, row_bytes) }, .{ .x = 1, .y = (request.outputs + 63) / 64, .threads_x = 128, .shared = 8192 })) return 0;
                return command.finish("Q8_0 decode MPP matmul");
            }
            c.ds4_metal_dense_warn_mpp();
        }
        if (!matvec(&command, request, weights, q8Dispatch(request.outputs, facts), true)) return 0;
        return command.finish("Q8_0 tensor matvec");
    }
    const ext_limit = c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", 16, 2, 128);
    if (request.rows <= ext_limit and request.width % 128 == 0) {
        if (!fewRows(&command, request, weights, true, facts)) return 0;
        return command.finish("Q8_0 tensor mul_mv_ext");
    }
    // Direct-RHS MPP dequantizes a 64x32 weight tile to half in shared memory.
    // Tiny uneven prompts do not amortize a second tail dispatch.
    const split_prefix = facts & c.DS4_DENSE_QUALITY == 0 and request.rows >= 192 and request.rows % 32 != 0;
    const prefix_rows = if (request.rows % 32 == 0) request.rows else if (split_prefix) request.rows - request.rows % 32 else 0;
    var generic_start: u64 = 0;
    var generic_rows = request.rows;
    if (facts & c.DS4_DENSE_MPP != 0 and prefix_rows >= 32 and request.width % 64 == 0 and request.outputs % 64 == 0) {
        const tile_rows = tile(prefix_rows);
        const program = Program{ .kind = c.DS4_DENSE_MM, .name = naxName(true, tile_rows) };
        if (program.pipeline()) |pipeline| {
            if (!encode(&command, request, weights, c.DS4_DENSE_MM, pipeline, .{ .mm = mmArgs(request, prefix_rows, row_bytes) }, .{ .x = prefix_rows / tile_rows, .y = request.outputs / 64, .threads_x = 128, .shared = 8192 })) return 0;
            if (prefix_rows == request.rows) return command.finish("Q8_0 NAX tensor matmul");
            generic_start = prefix_rows;
            generic_rows = request.rows - prefix_rows;
        } else c.ds4_metal_dense_warn_mpp();
    }
    const bc_in = request.width % 32 != 0;
    const bc_out = request.outputs % 64 != 0 or generic_rows % 32 != 0;
    // Qwen's M3 entry point scopes the default; explicit env override is shared.
    const forced = api.envBool("DS4_QWEN4_Q8_PREFILL_UNPACK");
    const unpack = generic_rows >= 32 and (if (forced >= 0) forced == 1 else prefill_default);
    const fits = request.width > 0 and request.outputs > 0 and request.outputs <= (64 << 20) / request.width;
    var scratch: ?*api.ModelBuffer = null;
    defer if (scratch) |buffer| api.releaseModelBuffer(buffer);
    var matrix = weights;
    var matrix_row = row_bytes;
    var name: [*:0]const u8 = "kernel_mul_mm_q8_0_f32";
    if (unpack and fits) {
        scratch = c.ds4_metal_dense_retain_scratch(request.width * request.outputs * 2) orelse return 0;
        const pipeline = c.ds4_metal_dense_unpack_pipeline() orelse return 0;
        const blocks: u32 = @intCast(request.width * request.outputs / 32);
        if (c.ds4_metal_dense_unpack(command.token.?, pipeline, weights.buffer, weights.offset, scratch.?, blocks, (@as(u64, blocks) * 8 + 255) / 256) == 0) return 0;
        matrix = .{ .buffer = scratch.?, .offset = 0 };
        matrix_row = request.width * 2;
        name = "kernel_mul_mm_f16_f32";
    }
    if (!dispatch(&command, request, matrix, .{ .kind = c.DS4_DENSE_MM, .name = name, .bc_in = bc_in, .bc_out = bc_out }, .{ .mm = mmArgs(request, generic_rows, matrix_row) }, .{ .x = (generic_rows + 31) / 32, .y = (request.outputs + 63) / 64, .threads_x = 128, .shared = if (bc_out) 8192 else 6144, .x_offset = generic_start *% request.width *% 4, .out_offset = generic_start *% request.outputs *% 4 })) return 0;
    return command.finish("Q8_0 tensor matmul");
}

pub fn projectQ8(request: Request, qwen: bool) c_int {
    @setFloatMode(.optimized); // Original TU policy for diagnostic timing only.
    if (api.init() == 0 or request.width & 31 != 0 or !request.shapesFit()) return 0;
    const profile_requested = request.rows > 8 and api.envBool("DS4_METAL_Q8_PREFILL_PROFILE") > 0;
    var label: [128]u8 = undefined;
    var profile = false;
    if (profile_requested) {
        _ = c.snprintf(&label, label.len, "q8 weight_off=%llu in=%llu out=%llu tok=%llu", @as(c_ulonglong, request.offset), @as(c_ulonglong, request.width), @as(c_ulonglong, request.outputs), @as(c_ulonglong, request.rows));
        const filter = c.getenv("DS4_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile = filter == null or filter[0] == 0 or std.mem.find(u8, std.mem.sliceTo(&label, 0), std.mem.span(filter)) != null;
    }
    var split_batch = false;
    if (profile and c.ds4_gpu_commands_active() != 0) {
        if (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0) return 0;
        split_batch = true;
    }
    const start = if (profile) c.ds4_metal_dense_now_ms() else 0;
    const default_unpack = qwen and c.ds4_metal_dense_facts(false) & c.DS4_DENSE_M3_ULTRA != 0;
    var result = q8Legacy(request, false, default_unpack);
    if (profile) {
        if (split_batch and c.ds4_gpu_end_commands() == 0) result = 0;
        const elapsed = c.ds4_metal_dense_now_ms() - start;
        _ = c.fprintf(c.__stderrp, "ds4: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n", &label, @as(c_ulonglong, request.width), @as(c_ulonglong, request.outputs), @as(c_ulonglong, request.rows), elapsed);
        if (split_batch and c.ds4_gpu_begin_commands() == 0) result = 0;
    }
    return result;
}

/// Batched decode keeps the single-row lane walk. BF16 is a shader variant,
/// not a new reduction schedule. Scalar mode repeats dispatches in ONE encoder.
pub fn q8Rows(request: Request, bf16: bool, scalar: bool) c_int {
    if (scalar and request.rows == 1) return projectQ8(request, false);
    if (api.init() == 0) return 0;
    if (request.out == null or request.x == null or request.map == null or request.rows == 0 or request.rows > std.math.maxInt(i32) or request.width & 31 != 0 or !request.shapesFit()) return 0;
    if (!scalar and (request.width == 0 or request.outputs == 0)) return 0;
    if (request.width > max_u64 / request.rows / 4 or request.outputs > max_u64 / request.rows / 4) return 0;
    const message: ?[*:0]const u8 = if (scalar) "ds4: Metal Q8_0 scalar-row matmul received undersized activation buffers" else null;
    if (!request.activations(request.rows * request.width * 4, request.rows * request.outputs * 4, message)) return 0;
    const row_bytes = (request.width / 32) * 34;
    if (request.outputs > max_u64 / row_bytes) return 0;
    const weights = request.weight(request.outputs * row_bytes, if (scalar) "ds4: Metal Q8_0 scalar-row matmul range is outside the mapped model" else null) orelse return 0;
    defer weights.deinit();
    const mode = q8Dispatch(request.outputs, c.ds4_metal_dense_facts(false));
    const program = Program{ .kind = c.DS4_DENSE_MV, .name = if (bf16) "kernel_mul_mv_q8_0_bf16_f32" else mode.name, .nsg = mode.nsg };
    const pipeline = program.pipeline() orelse return 0;
    var command = Command.init() orelse return 0;
    defer command.deinit();
    const nr: u64 = @intCast(mode.rows);
    const args = mvArgs(request.width, request.outputs, if (scalar) 1 else request.rows, true, mode.rows);
    const grid = Grid{ .x = (request.outputs + nr - 1) / nr, .y = if (scalar) 1 else request.rows, .threads_y = @intCast(mode.nsg), .shared = mode.shared, .repeats = if (scalar) request.rows else 1, .x_step = if (scalar) request.width * 4 else 0, .out_step = if (scalar) request.outputs * 4 else 0 };
    if (!encode(&command, request, weights, c.DS4_DENSE_MV, pipeline, .{ .mv = args }, grid)) return 0;
    return command.finish(if (scalar) "Q8_0 scalar-row matvecs" else "Q8_0 exact decode-row matvec");
}

pub const Pair = struct { first: Request, out1: ?*Tensor, offset1: u64, outputs1: u64 };
fn encodePair(command: *const Command, pair: Pair, w0: Weight, w1: Weight, pipeline: *const Pso, mode: Matvec, q8_kind: bool, concat: bool) bool {
    const request = pair.first;
    const args0 = mvArgs(request.width, request.outputs, 1, q8_kind, mode.rows);
    const args1 = mvArgs(request.width, pair.outputs1, 1, q8_kind, mode.rows);
    const maximum = if (concat) request.outputs + pair.outputs1 else @max(request.outputs, pair.outputs1);
    const nr: u64 = @intCast(mode.rows);
    const launch = c.ds4_metal_dense_pair_launch{ .args0 = args0, .args1 = args1, .two_args = q8_kind, .pipeline = pipeline, .weights0 = w0.buffer, .weights1 = w1.buffer, .weight_offset0 = w0.offset, .weight_offset1 = w1.offset, .x = request.x, .out0 = request.out, .out1 = pair.out1, .groups = (maximum + nr - 1) / nr, .simdgroups = @intCast(mode.nsg), .shared_bytes = if (q8_kind) 2 * mode.shared else mode.shared };
    return c.ds4_metal_dense_pair_encode(command.token.?, &launch) != 0;
}
pub fn q8Pair(pair: Pair, concat: bool) c_int {
    const request = pair.first;
    if (concat and (request.width == 0 or (request.outputs | pair.outputs1) & 1 != 0)) return 0;
    if (api.init() == 0) return 0;
    if (request.out == null or pair.out1 == null or request.map == null or request.x == null or request.rows != 1 or request.outputs == 0 or pair.outputs1 == 0 or request.width & 31 != 0 or !request.shapesFit() or pair.outputs1 > max_u32) return 0;
    if (!request.activations(request.width * 4, request.outputs * 4, null) or api.tensorHasBuffer(pair.out1) == 0 or api.tensorBytes(pair.out1) < pair.outputs1 * 4) return api.reportError("ds4: Metal paired Q8_0 matvec received undersized activation buffers");
    const row_bytes = (request.width / 32) * 34;
    const bytes0 = request.outputs *% row_bytes;
    const bytes1 = pair.outputs1 *% row_bytes;
    if (request.offset > request.size or bytes0 > request.size - request.offset or pair.offset1 > request.size or bytes1 > request.size - pair.offset1) return api.reportError("ds4: Metal paired Q8_0 matvec range is outside the mapped model");
    const first_weight = request.weight(bytes0, null);
    defer if (first_weight) |value| value.deinit();
    var second = request;
    second.offset = pair.offset1;
    const second_weight = second.weight(bytes1, null);
    defer if (second_weight) |value| value.deinit();
    // The C scope attempts both wraps before testing either result.
    const weights0 = first_weight orelse return 0;
    const weights1 = second_weight orelse return 0;
    const facts = c.ds4_metal_dense_facts(false);
    const mode = q8Dispatch(request.outputs, facts);
    const other = q8Dispatch(pair.outputs1, facts);
    // Mixed 4/8 SIMD extents cannot retain both standalone reduction trees.
    if (mode.nsg != other.nsg) return 0;
    const pipeline = (Program{ .kind = c.DS4_DENSE_MV, .name = if (concat) "kernel_qwen4_q8_concat" else "kernel_mul_mv_q8_0_f32_pair", .nsg = mode.nsg }).pipeline() orelse return 0;
    var command = Command.init() orelse return 0;
    defer command.deinit();
    if (!encodePair(&command, pair, weights0, weights1, pipeline, mode, true, concat)) return 0;
    return command.finish("paired Q8_0 matvec");
}

pub fn half(request: Request, exact_rows: bool) c_int {
    if (api.init() == 0 or !request.shapesFit()) return 0;
    if (!request.activations(request.rows *% request.width *% 4, request.rows *% request.outputs *% 4, "ds4: Metal F16 tensor matmul received undersized activation buffers")) return 0;
    const row_bytes = request.width * 2;
    const weights = request.weight(row_bytes *% request.outputs, "ds4: Metal F16 tensor matmul range is outside the mapped model") orelse return 0;
    defer weights.deinit();
    var command = Command.init() orelse return 0;
    defer command.deinit();
    const facts = c.ds4_metal_dense_facts(false);
    if (request.rows == 1 or exact_rows) {
        var mode = halfDispatch(request.width);
        if (facts & c.DS4_DENSE_QUALITY == 0 and (request.outputs == 512 or request.outputs == 1024) and request.width >= 4096) {
            mode.rows = 4;
            mode.shared = 512;
        }
        // Few-output M5 projections use one row per SIMD group; K traversal
        // and reduction are unchanged. Explicit 1/2 override works everywhere.
        const forced = c.ds4_zig_metal_env_u64("DS4_METAL_PLAIN_MV_NR0", 0, 0, 2);
        if ((forced != 0 and forced == 1) or (forced == 0 and request.outputs <= 1024 and facts & c.DS4_DENSE_M5 != 0)) {
            mode.rows = 1;
            mode.shared = 128;
        }
        if (!matvec(&command, request, weights, mode, false)) return 0;
        return command.finish("F16 tensor matvec");
    }
    if (request.rows <= 8 and request.width % 128 == 0) {
        if (!fewRows(&command, request, weights, false, facts)) return 0;
        return command.finish("F16 tensor mul_mv_ext");
    }
    // Keep the 128-token direct-RHS tile when aligned: the later 64 retest
    // was neutral/slower. Boundary rows stay in the generic safe kernel.
    if (facts & c.DS4_DENSE_MPP != 0 and request.rows >= 32 and request.width % 32 == 0 and request.outputs % 64 == 0 and request.rows % 32 == 0) {
        const tile_rows = tile(request.rows);
        const program = Program{ .kind = c.DS4_DENSE_MM, .name = naxName(false, tile_rows) };
        if (program.pipeline()) |pipeline| {
            if (!encode(&command, request, weights, c.DS4_DENSE_MM, pipeline, .{ .mm = mmArgs(request, request.rows, row_bytes) }, .{ .x = request.rows / tile_rows, .y = request.outputs / 64, .threads_x = 128, .shared = 8192 })) return 0;
            return command.finish("F16 NAX tensor matmul");
        }
        c.ds4_metal_dense_warn_mpp();
    }
    const bc_out = request.outputs % 64 != 0 or request.rows % 32 != 0;
    if (!dispatch(&command, request, weights, .{ .kind = c.DS4_DENSE_MM, .name = "kernel_mul_mm_f16_f32", .bc_in = request.width % 32 != 0, .bc_out = bc_out }, .{ .mm = mmArgs(request, request.rows, row_bytes) }, .{ .x = (request.rows + 31) / 32, .y = (request.outputs + 63) / 64, .threads_x = 128, .shared = if (bc_out) 8192 else 6144 })) return 0;
    return command.finish("F16 tensor matmul");
}
pub fn halfPair(pair: Pair) c_int {
    const request = pair.first;
    if (api.init() == 0 or !request.shapesFit() or request.rows != 1 or request.width & 3 != 0) return 0;
    if (!request.activations(request.width * 4, request.outputs * 4, null) or api.tensorHasBuffer(pair.out1) == 0 or api.tensorBytes(pair.out1) < request.outputs * 4) return api.reportError("ds4: Metal F16 paired matvec received undersized activation buffers");
    const bytes = request.width *% 2 *% request.outputs;
    if (request.offset > request.size or bytes > request.size - request.offset or pair.offset1 > request.size or bytes > request.size - pair.offset1) return api.reportError("ds4: Metal F16 paired matvec range is outside the mapped model");
    const first_weight = request.weight(bytes, null);
    defer if (first_weight) |value| value.deinit();
    var second = request;
    second.offset = pair.offset1;
    const second_weight = second.weight(bytes, null);
    defer if (second_weight) |value| value.deinit();
    // The C scope attempts both wraps before testing either result.
    const weights0 = first_weight orelse return 0;
    const weights1 = second_weight orelse return 0;
    var command = Command.init() orelse return 0;
    defer command.deinit();
    var mode = halfDispatch(request.width);
    if (c.ds4_metal_dense_facts(true) & c.DS4_DENSE_PAIR_NR4 != 0 and (request.outputs == 512 or request.outputs == 1024) and request.width >= 4096) {
        mode.rows = 4;
        mode.shared = 512;
    }
    const pipeline = (Program{ .kind = c.DS4_DENSE_MV, .name = "kernel_mul_mv_f16_f32_pair_4", .nsg = mode.nsg }).pipeline() orelse return 0;
    if (!encodePair(&command, pair, weights0, weights1, pipeline, mode, false, false)) return 0;
    return command.finish("F16 paired matvec");
}
