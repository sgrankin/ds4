//! General Q4 projection and K-slice policy. Q8 composes the complete native
//! Dense/AttentionOutput owners. The existing fixed dense ARC encoder only
//! binds the actual MV112/EXT112/MM88 packets and named buffers.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Dense = @import("dense_matmul.zig");
const Output = @import("attention_output.zig");
const rowBytes = @import("embedding.zig").rowBytes;
const Command = @import("routed_batch.zig").Command;
pub const Request = Dense.Request;
const q8 = c.DS4_METAL_TENSOR_Q8_0;
const q4 = c.DS4_METAL_TENSOR_Q4_K;
const q40 = c.DS4_METAL_TENSOR_Q4_0;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
const Pso = c.ds4_metal_hc_pipeline;
fn narrow(value: u64) i32 {
    return @bitCast(@as(u32, @truncate(value)));
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn report(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
const Weight = struct {
    buffer: *api.ModelBuffer,
    offset: u64,
    fn deinit(self: Weight) void {
        api.releaseModelBuffer(self.buffer);
    }
};
fn weight(request: Request, bytes: u64) ?Weight {
    var inner: u64 = 0;
    const buffer = api.retainModelRange(request.map, request.size, request.offset, bytes, &inner) orelse return null;
    return .{ .buffer = buffer, .offset = inner };
}
fn pipeline(kind: u32, name: [*:0]const u8, nx: i16, input: bool, output: bool) ?*const Pso {
    return c.ds4_metal_dense_pipeline(kind, name, 2, nx, input, output);
}
fn launch(request: Request, weights: Weight, kind: u32, pso: *const Pso) c.ds4_metal_dense_launch {
    return .{
        .kind = kind,
        .args = std.mem.zeroes(c.ds4_metal_dense_args),
        .pipeline = pso,
        .weights = weights.buffer,
        .weight_offset = weights.offset,
        .x = request.x,
        .out = request.out,
        .x_offset = 0,
        .out_offset = 0,
        .groups_x = 1,
        .groups_y = 1,
        .threads_x = 32,
        .threads_y = 1,
        .repeats = 1,
        .x_step = 0,
        .out_step = 0,
        .shared_bytes = 0,
    };
}
fn dispatch(command: *Command, p: *const c.ds4_metal_dense_launch, label: [*:0]const u8) bool {
    command.label = label;
    return c.ds4_metal_dense_encode(command.token, p) != 0 and command.finish();
}
fn classicArgs(width: u64, outputs: u64, rows: u64, row: u64) c.ds4_metal_mv_args {
    const weights = row *% outputs;
    const input = width *% rows *% 4;
    return .{
        .ne00 = narrow(width),
        .ne01 = narrow(outputs),
        .ne02 = 1,
        .nb00 = 1,
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
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}
fn extArgs(request: Request, row: u64) c.ds4_metal_mv_ext_args {
    const base = classicArgs(request.width, request.outputs, request.rows, row);
    return .{
        .ne00 = base.ne00,
        .ne01 = base.ne01,
        .ne02 = 1,
        // The original general-quant factory receives row_bytes for BOTH
        // elem_bytes and row_bytes; retain this packet exactly.
        .nb00 = row,
        .nb01 = row,
        .nb02 = base.nb02,
        .nb03 = base.nb03,
        .ne10 = base.ne10,
        .ne11 = base.ne11,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = base.nb11,
        .nb12 = base.nb12,
        .nb13 = base.nb13,
        .ne0 = base.ne0,
        .ne1 = base.ne1,
        .r2 = 1,
        .r3 = 1,
    };
}
fn mmArgs(request: Request, row: u64) c.ds4_metal_mm_args {
    return .{
        .ne00 = narrow(request.width),
        .ne02 = 1,
        .nb01 = row,
        .nb02 = row *% request.outputs,
        .nb03 = row *% request.outputs,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = request.width * 4,
        .nb12 = request.width *% request.rows *% 4,
        .nb13 = request.width *% request.rows *% 4,
        .ne0 = narrow(request.outputs),
        .ne1 = narrow(request.rows),
        .r2 = 1,
        .r3 = 1,
    };
}
fn extRows(rows: u64) i16 {
    return switch (rows) {
        1 => 1,
        2 => 2,
        3, 6 => 3,
        4, 7, 8 => 4,
        5 => 5,
        else => 0,
    };
}
fn extName(format: u32, rows: i16) ?[*:0]const u8 {
    return if (format == q4) switch (rows) {
        1 => "kernel_mul_mv_ext_q4_K_f32_r1_1",
        2 => "kernel_mul_mv_ext_q4_K_f32_r1_2",
        3 => "kernel_mul_mv_ext_q4_K_f32_r1_3",
        4 => "kernel_mul_mv_ext_q4_K_f32_r1_4",
        5 => "kernel_mul_mv_ext_q4_K_f32_r1_5",
        else => null,
    } else if (format == q40) switch (rows) {
        1 => "kernel_mul_mv_ext_q4_0_f32_r1_1",
        2 => "kernel_mul_mv_ext_q4_0_f32_r1_2",
        3 => "kernel_mul_mv_ext_q4_0_f32_r1_3",
        4 => "kernel_mul_mv_ext_q4_0_f32_r1_4",
        5 => "kernel_mul_mv_ext_q4_0_f32_r1_5",
        else => null,
    } else null;
}
fn mmName(format: u32) ?[*:0]const u8 {
    return switch (format) {
        q4 => "kernel_mul_mm_q4_K_f32",
        q40 => "kernel_mul_mm_q4_0_f32",
        else => null,
    };
}
fn naxName(format: u32, tile: u64) ?[*:0]const u8 {
    return if (format == q4) switch (tile) {
        128 => "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n128",
        64 => "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n64",
        else => "kernel_mul_mm_q4_K_f32_nax_direct_rhs",
    } else if (format == q40) switch (tile) {
        128 => "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n128",
        64 => "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n64",
        else => "kernel_mul_mm_q4_0_f32_nax_direct_rhs",
    } else null;
}
pub fn run(request: Request, format: u32) bool {
    if (format == q8) return Dense.q8Legacy(request, false, false) != 0;
    if (api.init() == 0 or request.out == null or request.x == null or request.map == null or
        request.width == 0 or request.outputs == 0 or request.rows == 0 or
        request.width > max32 or request.outputs > max32 or request.rows > max32) return false;
    const row = rowBytes(format, @intCast(request.width)) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: Metal quant matmul received unsupported type/dim (%u, in=%llu)\n", format, @as(c_ulonglong, request.width));
        return false;
    };
    if (api.tensorHasBuffer(request.x) == 0 or api.tensorHasBuffer(request.out) == 0 or
        api.tensorBytes(request.x) < request.rows *% request.width *% 4 or
        api.tensorBytes(request.out) < request.rows *% request.outputs *% 4)
        return report("ds4: Metal quant tensor matmul received undersized activation buffers");
    if (request.outputs > max64 / row) return false;
    const bytes = request.outputs * row;
    if (request.offset > request.size or bytes > request.size - request.offset)
        return report("ds4: Metal quant tensor matmul range is outside the mapped model");
    const weights = weight(request, bytes) orelse return false;
    defer weights.deinit();
    var command: Command = .{};
    defer command.drop();
    if (!command.acquire()) return false;

    // Small-batch Q4_K classic matvec streams 530–650 GB/s on the original M5
    // decode shapes, versus ~220 GB/s for ext. Fall through if its PSO is absent.
    if (format == q4 and request.rows <= 8 and request.width % 256 == 0 and !present("DS4_METAL_DISABLE_Q4_MV_CLASSIC")) {
        if (pipeline(c.DS4_DENSE_EXT, "kernel_mul_mv_q4_K_dense_f32", 8, false, false)) |pso| {
            var p = launch(request, weights, c.DS4_DENSE_MV, pso);
            p.args.mv = classicArgs(request.width, request.outputs, request.rows, row);
            p.groups_x = (request.outputs + 3) / 4;
            p.groups_y = request.rows;
            p.threads_y = 2;
            p.shared_bytes = 32;
            return dispatch(&command, &p, "Q4_K classic mul_mv");
        }
    }
    const facts = c.ds4_metal_dense_facts(false);
    if (request.rows <= 8 and request.width % 128 == 0) {
        const nx: i16 = if (request.width % 256 == 0 and (request.rows < 3 or (request.rows == 3 and facts & c.DS4_DENSE_VERIFY_EXACT != 0))) 16 else 8;
        const r1 = extRows(request.rows);
        if (extName(format, r1)) |name| {
            if (pipeline(c.DS4_DENSE_EXT, name, nx, false, false)) |pso| {
                const r0: u64 = @intCast(@divTrunc(32, nx) * 2);
                var p = launch(request, weights, c.DS4_DENSE_EXT, pso);
                p.args.ext = extArgs(request, row);
                p.groups_x = (request.outputs + r0 - 1) / r0;
                p.groups_y = (request.rows + @as(u64, @intCast(r1)) - 1) / @as(u64, @intCast(r1));
                p.threads_y = 2;
                return dispatch(&command, &p, "Q4 tensor mul_mv_ext");
            }
        }
    }
    if (facts & c.DS4_DENSE_MPP != 0 and request.rows >= 32 and request.width % 64 == 0 and request.outputs % 64 == 0 and request.rows % 32 == 0) {
        const tile: u64 = if (request.rows % 128 == 0) 128 else if (request.rows % 64 == 0) 64 else 32;
        if (naxName(format, tile)) |name| {
            if (pipeline(c.DS4_DENSE_MM, name, 0, false, false)) |pso| {
                var p = launch(request, weights, c.DS4_DENSE_MM, pso);
                p.args.mm = mmArgs(request, row);
                p.groups_x = request.rows / tile;
                p.groups_y = request.outputs / 64;
                p.threads_x = 128;
                p.shared_bytes = 8192;
                return dispatch(&command, &p, "Q4 NAX tensor matmul");
            }
            c.ds4_metal_dense_warn_mpp();
        }
    }
    const name = mmName(format) orelse return false;
    const bc_input = request.width % 32 != 0;
    const bc_output = request.outputs % 64 != 0 or request.rows % 32 != 0;
    const pso = pipeline(c.DS4_DENSE_MM, name, 0, bc_input, bc_output) orelse return false;
    var p = launch(request, weights, c.DS4_DENSE_MM, pso);
    p.args.mm = mmArgs(request, row);
    p.groups_x = (request.rows + 31) / 32;
    p.groups_y = (request.outputs + 63) / 64;
    p.threads_x = 128;
    p.shared_bytes = if (bc_output) 8192 else 6144;
    return dispatch(&command, &p, "Q4 tensor matmul");
}
pub fn slice(request: Request, format: u32, first: u64, count: u64, input_offset: u64) bool {
    if (format == q8) return (Output.KSlice{ .out = request.out, .model = request.map, .size = request.size, .offset = request.offset, .width = request.width, .outputs = request.outputs, .first = first, .count = count, .input = request.x, .input_offset = input_offset }).scalar();
    if (api.init() == 0 or request.out == null or request.x == null or request.map == null or request.width == 0 or count == 0 or request.outputs == 0 or first +% count > request.width or request.width > max32 or count > max32 or request.outputs > max32) return false;
    const elements: u64 = if (format == q4) 256 else if (format == q40) 32 else {
        _ = c.fprintf(c.__stderrp, "ds4: Metal quant kslice received unsupported type %u\n", format);
        return false;
    };
    const block: u64 = if (format == q4) 144 else 18;
    if (request.width % elements != 0 or first % elements != 0 or count % elements != 0)
        return report("ds4: Metal quant kslice dimensions are not block aligned");
    if (api.tensorHasBuffer(request.x) == 0 or api.tensorHasBuffer(request.out) == 0 or
        api.tensorBytes(request.x) < (input_offset +% count) *% 4 or api.tensorBytes(request.out) < request.outputs * 4)
        return report("ds4: Metal quant kslice matmul received undersized buffers");
    const row = (request.width / elements) * block;
    const bytes = request.outputs *% row;
    if (request.offset > request.size or bytes > request.size - request.offset)
        return report("ds4: Metal quant kslice weights are outside the mapped model");
    var weights = weight(request, bytes) orelse return false;
    defer weights.deinit();
    var command: Command = .{};
    defer command.drop();
    if (!command.acquire()) return false;
    if (format == q4 and count % 256 == 0 and !present("DS4_METAL_DISABLE_Q4_MV_CLASSIC")) {
        if (pipeline(c.DS4_DENSE_EXT, "kernel_mul_mv_q4_K_dense_f32", 8, false, false)) |pso| {
            weights.offset +%= (first / elements) * block;
            var p = launch(request, weights, c.DS4_DENSE_MV, pso);
            p.args.mv = classicArgs(count, request.outputs, 1, row);
            p.x_offset = input_offset *% 4;
            p.groups_x = (request.outputs + 3) / 4;
            p.threads_y = 2;
            p.shared_bytes = 32;
            return dispatch(&command, &p, "Q4_K kslice matvec");
        }
    }
    _ = c.fprintf(c.__stderrp, "ds4: Metal quant kslice has no kernel for type %u\n", format);
    return false;
}
