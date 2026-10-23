//! HC RMS-scale/project host control. Shaders and the ordinary RMS/F16 fallback
//! stay unchanged; opaque PSOs, mapped buffers and command ownership stay ObjC.
const std = @import("std");
const api = @import("bridge.zig");
const Tensor = api.Tensor;

fn fusedMode(width: u32, outputs: u32, rows: u32) bool {
    if (rows <= 8 or outputs != 24 or
        (width != 16384 and width != 20480 and width != 28672)) return false;
    // The V4.1 20480-wide path starts at 512 rows. The other known shapes need
    // an M3 device or the existing explicit GPU test flag.
    if (width == 20480 and rows < 512) return false;
    const facts = api.hcFacts();
    if ((width != 20480 and facts & (api.hcM3 | api.hcTest) == 0) or
        api.envPresent("DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ") != 0) return false;
    return facts & api.hcScalePresent != 0;
}

fn hasBytes(tensor: ?*const Tensor, bytes: u64) bool {
    return tensor != null and api.tensorHasBuffer(tensor) != 0 and api.tensorBytes(tensor) >= bytes;
}

fn fallback(out: ?*Tensor, scratch: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, x: ?*const Tensor, rows: u32, eps: f32) c_int {
    return @intFromBool(api.rmsPlain(scratch, x, width, rows, eps) != 0 and
        api.matmulF16(out, map, size, offset, width, outputs, scratch, rows) != 0);
}

fn rmsArgs(width: u32, rows: u32, eps: f32) api.RmsArgs {
    const row = @as(u64, width) * @sizeOf(f32);
    const bytes = row * rows;
    return .{
        .ne00 = @intCast(width),
        .ne00_t = @intCast(width / 4),
        .nb1 = row,
        .nb2 = bytes,
        .nb3 = bytes,
        .eps = eps,
        .nef1 = .{ @intCast(rows), 1, 1 },
        .nef2 = .{ 1, 1, 1 },
        .nef3 = .{ 1, 1, 1 },
        .nbf1 = .{ row, row, row },
        .nbf2 = .{ bytes, row, row },
        .nbf3 = .{ bytes, row, row },
    };
}

fn mmArgs(width: u32, outputs: u32, rows: u32, weight_row: u64) api.MmArgs {
    const row = @as(u64, width) * @sizeOf(f32);
    return .{
        .ne00 = @intCast(width),
        .ne02 = 1,
        .nb01 = weight_row,
        .nb02 = weight_row * outputs,
        .nb03 = weight_row * outputs,
        .ne12 = 1,
        .nb10 = @sizeOf(f32),
        .nb11 = row,
        .nb12 = row * rows,
        .nb13 = row * rows,
        .ne0 = @intCast(outputs),
        .ne1 = @intCast(rows),
        .r2 = 1,
        .r3 = 1,
    };
}

pub export fn ds4_gpu_hc_rms_scale_project_f16_tensor(out: ?*Tensor, scratch: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, x: ?*const Tensor, rows: u32, eps: f32) callconv(.c) c_int {
    if (api.init() == 0) return 0;
    if (out == null or scratch == null or map == null or x == null or
        width == 0 or outputs == 0 or rows == 0 or rows > std.math.maxInt(i32)) return 0;
    if (!fusedMode(width, outputs, rows))
        return fallback(out, scratch, map, size, offset, width, outputs, x, rows, eps);

    const x_elements = @as(u64, width) * rows;
    const out_elements = @as(u64, outputs) * rows;
    if (x_elements > std.math.maxInt(u64) / @sizeOf(f32) or
        out_elements > std.math.maxInt(u64) / @sizeOf(f32)) return 0;
    if (!hasBytes(x, x_elements * @sizeOf(f32)) or
        !hasBytes(out, out_elements * @sizeOf(f32)) or
        !hasBytes(scratch, @as(u64, rows) * @sizeOf(f32)))
        return api.reportError("ds4: Metal HC RMS scale projection received undersized activation buffers");

    const weight_row = @as(u64, width) * @sizeOf(u16);
    if (outputs > std.math.maxInt(u64) / weight_row) return 0;
    const weight_bytes = weight_row * outputs;
    if (offset > size or weight_bytes > size - offset)
        return api.reportError("ds4: Metal HC RMS scale projection weight range is outside the mapped model");

    const bc_in = width % 32 != 0;
    const bc_out = outputs % 64 != 0 or rows % 32 != 0;
    // Missing specialized MM retains the original norm-then-projection path;
    // a later missing scale PSO is an error, rather than another fallback.
    const mm_pipeline = api.hcMmPipeline(@intFromBool(bc_in), @intFromBool(bc_out)) orelse
        return fallback(out, scratch, map, size, offset, width, outputs, x, rows, eps);
    const scale_pipeline = api.hcScalePipeline() orelse return 0;

    var inner_offset: u64 = 0;
    const weights = api.retainModelRange(map, size, offset, weight_bytes, &inner_offset) orelse return 0;
    // Preserve the original local strong model-buffer scope through encode and
    // any owned command wait; caller-owned batches retain encoded resources.
    defer api.releaseModelBuffer(weights);
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;

    const norm = rmsArgs(width, rows, eps);
    const mm = mmArgs(width, outputs, rows, weight_row);
    var threads: u64 = 32;
    const width4: u64 = width / 4;
    while (threads < width4 and threads < 1024) threads *= 2;
    threads = @min(threads, width4);
    if (threads == 0) threads = 1;
    const scale_launch = api.HcLaunch{
        .stage = api.hcScaleStage,
        .pipeline = scale_pipeline,
        .args = &norm,
        .args_bytes = @sizeOf(api.RmsArgs),
        .x = x,
        .out = null,
        .scales = scratch,
        .weights = null,
        .weight_offset = 0,
        .groups_x = rows,
        .groups_y = 1,
        .threads = threads,
        .shared_bytes = 32 * @sizeOf(f32),
    };
    const project_launch = api.HcLaunch{
        .stage = api.hcProjectStage,
        .pipeline = mm_pipeline,
        .args = &mm,
        .args_bytes = @sizeOf(api.MmArgs),
        .x = x,
        .out = out,
        .scales = scratch,
        .weights = weights,
        .weight_offset = inner_offset,
        .groups_x = (@as(u64, rows) + 31) / 32,
        .groups_y = (@as(u64, outputs) + 63) / 64,
        .threads = 128,
        .shared_bytes = if (bc_out) 8192 else 6144,
    };
    const ok = api.hcEncode(command, &scale_launch) != 0 and api.hcEncode(command, &project_launch) != 0;
    return api.release(command, owned, @intFromBool(ok), "HC RMS scale F16 projection");
}
