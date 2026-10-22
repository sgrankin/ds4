//! F32 matmul validation, packet construction and row/kernel policy.
//! Cached PSOs, ARC model buffers and command ownership remain in the bridge.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Tensor = api.Tensor;
const Mv = c.ds4_metal_mv_args;
const Ext = c.ds4_metal_mv_ext_args;

fn dimension(value: u64) i32 {
    // The C shader packet narrows admitted uint32 shapes to int32 bit patterns.
    return @bitCast(@as(u32, @truncate(value)));
}
fn mvArgs(width: u64, outputs: u64, rows: u64, rows_per_group: i32) Mv {
    const row = width * 4;
    const weights = row *% outputs;
    const activation = row * rows;
    return .{
        .ne00 = dimension(width),
        .ne01 = dimension(outputs),
        .ne02 = 1,
        .nb00 = 4,
        .nb01 = row,
        .nb02 = weights,
        .nb03 = weights,
        .ne10 = dimension(width),
        .ne11 = dimension(rows),
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = row,
        .nb12 = activation,
        .nb13 = activation,
        .ne0 = dimension(outputs),
        .ne1 = dimension(rows),
        .nr0 = rows_per_group,
        .r2 = 1,
        .r3 = 1,
    };
}
fn extArgs(width: u64, outputs: u64, rows: u64) Ext {
    const mv = mvArgs(width, outputs, rows, 2);
    return .{
        .ne00 = mv.ne00,
        .ne01 = mv.ne01,
        .ne02 = 1,
        .nb00 = 4,
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
fn mmArgs(width: u64, outputs: u64, rows: u64) api.MmArgs {
    const row = width * 4;
    const weights = row *% outputs;
    const activation = row * rows;
    return .{
        .ne00 = dimension(width),
        .ne02 = 1,
        .nb01 = row,
        .nb02 = weights,
        .nb03 = weights,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = row,
        .nb12 = activation,
        .nb13 = activation,
        .ne0 = dimension(outputs),
        .ne1 = dimension(rows),
        .r2 = 1,
        .r3 = 1,
    };
}
const Plain = struct { name: [*:0]const u8, groups: u64, rows: i32, shared: u32 };
fn plain(width: u64) Plain {
    if (width < 32) return .{ .name = "kernel_mul_mv_f32_f32_short", .groups = 1, .rows = 32, .shared = 0 };
    return .{ .name = if (width % 4 == 0) "kernel_mul_mv_f32_f32_4" else "kernel_mul_mv_f32_f32", .groups = @min((width + 127) / 128, 8), .rows = 2, .shared = 256 };
}
fn extRows(rows: u64) i16 {
    return switch (rows) {
        2 => 2,
        3, 6 => 3,
        4, 7, 8 => 4,
        5 => 5,
        else => if (rows > 8) 4 else 0,
    };
}
fn extName(rows: i16) ?[*:0]const u8 {
    return switch (rows) {
        2 => "kernel_mul_mv_ext_f32_f32_r1_2",
        3 => "kernel_mul_mv_ext_f32_f32_r1_3",
        4 => "kernel_mul_mv_ext_f32_f32_r1_4",
        5 => "kernel_mul_mv_ext_f32_f32_r1_5",
        else => null,
    };
}
const Program = struct { kind: u32, name: [*:0]const u8, nsg: i16 = 0, nxpsg: i16 = 0, bc_out: bool = false };
const Geometry = struct { x: u64, y: u64 = 1, threads_x: u64 = 32, threads_y: u64 = 1, shared: u32 = 0 };
const Buffers = struct { weights: *api.ModelBuffer, inner: u64, x: ?*const Tensor, out: ?*Tensor };
fn finish(command: *anyopaque, owned: c_int, program: Program, args: *const anyopaque, args_bytes: u64, buffers: Buffers, grid: Geometry, label: [*:0]const u8) c_int {
    const pipeline = c.ds4_zig_metal_f32_pipeline(program.kind, program.name, program.nsg, program.nxpsg, @intFromBool(program.bc_out)) orelse
        return api.release(command, owned, 0, label);
    const launch = c.ds4_metal_f32_launch{ .pipeline = pipeline, .args = args, .args_bytes = args_bytes, .kind = program.kind, .weights = buffers.weights, .weight_offset = buffers.inner, .x = buffers.x, .out = buffers.out, .groups_x = grid.x, .groups_y = grid.y, .threads_x = grid.threads_x, .threads_y = grid.threads_y, .shared_bytes = grid.shared };
    return api.release(command, owned, c.ds4_zig_metal_f32_encode(command, &launch), label);
}

pub export fn ds4_gpu_matmul_f32_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64) callconv(.c) c_int {
    if (api.init() == 0) return 0;
    if (width > std.math.maxInt(u32) or outputs > std.math.maxInt(u32) or rows == 0 or rows > std.math.maxInt(u32)) return 0;
    if (width > std.math.maxInt(u64) / rows / 4 or outputs > std.math.maxInt(u64) / rows / 4) return 0;
    if (api.tensorHasBuffer(x) == 0 or api.tensorHasBuffer(out) == 0 or
        api.tensorBytes(x) < rows * width * 4 or api.tensorBytes(out) < rows * outputs * 4)
        return api.reportError("ds4: Metal F32 tensor matmul received undersized activation buffers");
    const row_bytes = width * 4;
    // Preserve C's unchecked unsigned weight-range product, not a new policy.
    const weight_bytes = row_bytes *% outputs;
    if (offset > size or weight_bytes > size - offset)
        return api.reportError("ds4: Metal F32 tensor matmul range is outside the mapped model");
    var inner: u64 = 0;
    const weights = api.retainModelRange(map, size, offset, weight_bytes, &inner) orelse return 0;
    // The old local strong model-buffer scope survives encode and owned wait.
    defer api.releaseModelBuffer(weights);
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    const buffers = Buffers{ .weights = weights, .inner = inner, .x = x, .out = out };
    const facts = c.ds4_zig_metal_f32_facts();
    if (rows == 1) {
        var dispatch = plain(width);
        // Few-output M5 matvecs use one row per SIMD group without changing
        // the per-row walk/reduction tree. The explicit override works on all
        // devices and also applies to the short-width pipeline.
        const forced = c.ds4_zig_metal_env_u64("DS4_METAL_PLAIN_MV_NR0", 0, 0, 2);
        if ((forced != 0 and forced == 1) or (forced == 0 and outputs <= 1024 and facts & c.DS4_METAL_F32_M5 != 0)) {
            dispatch.rows = 1;
            dispatch.shared = 128;
        }
        const args = mvArgs(width, outputs, rows, dispatch.rows);
        const nr: u64 = @intCast(dispatch.rows);
        return finish(command, owned, .{ .kind = c.DS4_METAL_F32_MV, .name = dispatch.name, .nsg = @intCast(dispatch.groups) }, &args, @sizeOf(Mv), buffers, .{ .x = (outputs + nr - 1) / nr, .threads_y = dispatch.groups, .shared = dispatch.shared }, "F32 tensor matvec");
    }
    const router = width == 5120 and outputs == 384 and rows >= 32 and facts & c.DS4_METAL_F32_QUALITY == 0 and
        api.envPresent("DS4_METAL_DISABLE_V41_ROUTER_BATCH") == 0;
    if (router and rows >= 256 and api.envPresent("DS4_METAL_DISABLE_V41_ROUTER_MM") == 0) {
        const args = mmArgs(width, outputs, rows);
        return finish(command, owned, .{ .kind = c.DS4_METAL_F32_MM, .name = "kernel_mul_mm_f32_f32_full", .bc_out = rows % 32 != 0 }, &args, @sizeOf(api.MmArgs), buffers, .{ .x = (rows + 31) / 32, .y = outputs / 64, .threads_x = 128, .shared = 12288 }, "F32 router prefill matmul");
    }
    if ((rows <= 8 or router) and width % 128 == 0) {
        const nsg: i16 = if (router) 2 else @intCast(c.ds4_zig_metal_env_u64("DS4_METAL_MV_EXT_NSG", 2, 1, 8));
        const nxpsg: i16 = if (width % 256 == 0 and (rows < 3 or (rows == 3 and facts & c.DS4_METAL_F32_VERIFY_EXACT != 0))) 16 else if (width % 128 == 0) 8 else 4;
        const rows_per_group = extRows(rows);
        const name = extName(rows_per_group) orelse return api.release(command, owned, 0, "F32 tensor mul_mv_ext");
        const outputs_per_group = @as(u64, @intCast(@divTrunc(32, nxpsg))) * @as(u64, @intCast(nsg));
        const batch_per_group: u64 = @intCast(rows_per_group);
        const args = extArgs(width, outputs, rows);
        return finish(command, owned, .{ .kind = c.DS4_METAL_F32_MV_EXT, .name = name, .nsg = nsg, .nxpsg = nxpsg }, &args, @sizeOf(Ext), buffers, .{ .x = (outputs + outputs_per_group - 1) / outputs_per_group, .y = (rows + batch_per_group - 1) / batch_per_group, .threads_y = @intCast(nsg) }, "F32 tensor mul_mv_ext");
    }
    // Generic multi-row GLM prefill: one grid row per token through plain MV.
    const dispatch = plain(width);
    const args = mvArgs(width, outputs, rows, dispatch.rows);
    const nr: u64 = @intCast(dispatch.rows);
    return finish(command, owned, .{ .kind = c.DS4_METAL_F32_MV, .name = dispatch.name, .nsg = @intCast(dispatch.groups) }, &args, @sizeOf(Mv), buffers, .{ .x = (outputs + nr - 1) / nr, .y = rows, .threads_y = dispatch.groups, .shared = dispatch.shared }, "F32 tensor matmul");
}
