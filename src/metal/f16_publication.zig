//! Native F32-to-F16 publication bounds, chunking and copy-launch policy.
//! The Objective-C bridge retains commands and binds resources; shader math
//! and caller-owned command batching are unchanged.
const std = @import("std");
const api = @import("bridge.zig");
const Tensor = api.Tensor;

fn copyThreads(work: u64, pipeline: u32) u64 {
    const maximum = api.copyMaxThreads(pipeline);
    var threads: u64 = 32;
    while (threads < work and threads < maximum) threads *= 2;
    threads = @min(threads, maximum, work);
    return if (threads == 0) 1 else threads;
}

fn copyArgs(count: u32) api.CopyArgs {
    const src_stride = @as(u64, count) * @sizeOf(f32);
    const dst_stride = @as(u64, count) * @sizeOf(u16);
    return .{
        .nk0 = count,
        .ne00 = count,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = @sizeOf(f32),
        .nb01 = src_stride,
        .nb02 = src_stride,
        .nb03 = src_stride,
        .ne0 = count,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = @sizeOf(u16),
        .nb1 = dst_stride,
        .nb2 = dst_stride,
        .nb3 = dst_stride,
    };
}

pub export fn ds4_gpu_tensor_copy_f32_to_f16(dst: ?*Tensor, dst_offset: u64, src: ?*const Tensor, src_offset: u64, count: u64) callconv(.c) c_int {
    if (dst == null or src == null) return 0;
    if (api.init() == 0) return 0;
    // Zero elements succeed before inspecting offsets, ranges or buffers.
    if (count == 0) return 1;
    if (count > std.math.maxInt(u64) / @sizeOf(f32) or
        count > std.math.maxInt(u64) / @sizeOf(u16)) return 0;
    const source_bytes = api.tensorBytes(src);
    const dest_bytes = api.tensorBytes(dst);
    if (src_offset > source_bytes or count * @sizeOf(f32) > source_bytes - src_offset or
        dst_offset > dest_bytes or count * @sizeOf(u16) > dest_bytes - dst_offset) return 0;

    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    var done: u64 = 0;
    var ok = true;
    while (done < count and ok) {
        const chunk: u32 = @intCast(@min(count - done, std.math.maxInt(u32)));
        const contiguous = api.envBool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0;
        const pipeline: u32 = if (contiguous) api.contiguous else api.generic;
        const work: u64 = if (contiguous) (@as(u64, chunk) + 3) / 4 else chunk;
        const threads = copyThreads(work, pipeline);
        const args = copyArgs(chunk);
        const launch = api.CopyLaunch{
            .pipeline = pipeline,
            .args = if (contiguous) @ptrCast(&chunk) else @ptrCast(&args),
            .args_bytes = if (contiguous) @sizeOf(u32) else @sizeOf(api.CopyArgs),
            .src = src,
            .dst = dst,
            .src_offset = src_offset + done * @sizeOf(f32),
            .dst_offset = dst_offset + done * @sizeOf(u16),
            .groups = (work + threads - 1) / threads,
            .threads = threads,
        };
        ok = api.encodeCopy(command, &launch) != 0;
        done += chunk;
    }
    // One retain covers every chunk; failed owned sequences are not committed.
    return api.release(command, owned, @intFromBool(ok), "tensor f32 to f16 copy");
}
