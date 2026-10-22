//! Complete borrowed-command copy packet/geometry policy. This module has no
//! exports or global state; callers keep the command and both buffers alive.
const c = @import("metal_c");
const api = @import("bridge.zig");
const View = c.ds4_metal_attention_view;
fn threads(work: u64, maximum: u64) u64 {
    var count: u64 = 32;
    while (count < work and count < maximum) count *= 2;
    return @max(1, @min(count, maximum, work));
}
fn args(count: u32, source: u64, target: u64) c.ds4_metal_copy_args {
    return .{
        .nk0 = count,
        .ne00 = count,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = source,
        .nb01 = @as(u64, count) * source,
        .nb02 = @as(u64, count) * source,
        .nb03 = @as(u64, count) * source,
        .ne0 = count,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = target,
        .nb1 = @as(u64, count) * target,
        .nb2 = @as(u64, count) * target,
        .nb3 = @as(u64, count) * target,
    };
}
pub fn one(command: ?*anyopaque, source: View, target: View, count: u32, kind: u32, zero_ok: bool) bool {
    if (command == null or source.buffer == null or target.buffer == null) return false;
    if (count == 0) return zero_ok;
    const half_copy = kind == c.DS4_PRIM_COPY_F16;
    const reverse = kind == c.DS4_PRIM_COPY_F16_F32;
    const contiguous = api.envBool(if (half_copy) "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY" else "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0;
    const selected: u32 = if (half_copy)
        (if (contiguous) c.DS4_PRIM_COPY_F16_CONTIG else c.DS4_PRIM_COPY_F16_BLIT)
    else if (contiguous)
        (if (reverse) c.DS4_PRIM_COPY_F16_F32_CONTIG else c.DS4_PRIM_COPY_F32_F16_CONTIG)
    else
        kind;
    const pipeline = c.ds4_metal_primitive_pipeline(selected);
    const work: u64 = if (contiguous) (@as(u64, count) + 3) / 4 else count;
    const nth = threads(work, pipeline.threads);
    const packet = args(count, if (half_copy or reverse) 2 else 4, if (reverse) 4 else 2);
    return c.ds4_metal_primitive_copy(command, selected, pipeline, &packet, count, source, target, (work + nth - 1) / nth, 1, nth) != 0;
}
pub const Region = struct {
    columns: u32,
    rows: u32,
    planes: u32,
    source_col: u64,
    source_row: u64,
    source_plane: u64,
    target_row: u64,
    target_plane: u64,
    kind: u32,
};
pub fn three(command: ?*anyopaque, source: View, target: View, region: Region) bool {
    if (command == null or source.buffer == null or target.buffer == null or region.columns == 0 or region.rows == 0 or region.planes == 0) return false;
    const pipeline = c.ds4_metal_primitive_pipeline(region.kind);
    const packet = c.ds4_metal_copy_args{
        .nk0 = region.columns,
        .ne00 = region.columns,
        .ne01 = region.rows,
        .ne02 = region.planes,
        .ne03 = 1,
        .nb00 = region.source_col,
        .nb01 = region.source_row,
        .nb02 = region.source_plane,
        .nb03 = @as(u64, region.planes) *% region.source_plane,
        .ne0 = region.columns,
        .ne1 = region.rows,
        .ne2 = region.planes,
        .ne3 = 1,
        .nb0 = if (region.kind == c.DS4_PRIM_COPY_F32) 4 else 2,
        .nb1 = region.target_row,
        .nb2 = region.target_plane,
        .nb3 = @as(u64, region.planes) *% region.target_plane,
    };
    const nth = threads(region.columns, pipeline.threads);
    return c.ds4_metal_primitive_copy(command, region.kind, pipeline, &packet, 0, source, target, ((@as(u64, region.columns) + nth - 1) / nth) * region.rows, region.planes, nth) != 0;
}
