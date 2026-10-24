//! Assemble the image rows intersecting one language-model chunk. The actual
//! overlay record is shared with the remaining legacy caller, not mirrored.
const c = @import("gguf_c");
const std = @import("std");
const Overlay = c.glm_vision_overlay;
const Segment = c.glm_vision_overlay_segment;
const Span = c.ds4_vision_span;
const allocator = std.heap.c_allocator;
const width = 4096;

pub fn deinit(overlay: *Overlay) void {
    c.ds4_gpu_tensor_free(overlay.tensor);
    c.free(overlay.segments);
    overlay.* = std.mem.zeroes(Overlay);
}

pub fn prepare(overlay: *Overlay, images: []const Span, position: u32, rows: u32) bool {
    overlay.* = std.mem.zeroes(Overlay);
    const chunk_end = @as(u64, position) + rows;
    var total: u64 = 0;
    var segments: usize = 0;
    for (images) |image| {
        if (image.embedding.data == null or image.embedding.token_count == 0) return false;
        const start = @max(image.token_start, position);
        const end = @min(@as(u64, image.token_start) + image.embedding.token_count, chunk_end);
        if (start < end) {
            total += end - start;
            segments += 1;
        }
    }
    if (segments == 0) return true;
    if (total > std.math.maxInt(u32) or total > std.math.maxInt(u64) / (width * @sizeOf(f32))) return false;
    const storage = allocator.alloc(Segment, segments) catch return false;
    overlay.segments = storage.ptr;
    overlay.tensor = c.ds4_gpu_tensor_alloc(total * width * @sizeOf(f32));
    if (overlay.tensor == null) {
        deinit(overlay);
        return false;
    }
    var upload_row: u32 = 0;
    for (images) |image| {
        const start = @max(image.token_start, position);
        const end = @min(@as(u64, image.token_start) + image.embedding.token_count, chunk_end);
        if (start >= end) continue;
        const count: u32 = @intCast(end - start);
        const image_row = start - image.token_start;
        if (c.ds4_gpu_tensor_write(overlay.tensor, @as(u64, upload_row) * width * @sizeOf(f32), image.embedding.data + @as(u64, image_row) * width, @as(u64, count) * width * @sizeOf(f32)) == 0) {
            deinit(overlay);
            return false;
        }
        overlay.segments[overlay.count] = .{ .dst_row = start - position, .src_row = upload_row, .rows = count };
        overlay.count += 1;
        upload_row += count;
    }
    return true;
}

pub fn syntheticCovered(token: i32, token_index: usize, images: []const Span, position: u32, vocabulary: u32) bool {
    if (token < vocabulary or token > vocabulary + c.DS4_DEEPSEEK4_IMAGE_END) return false;
    const absolute = @as(u64, position) + token_index;
    for (images) |image| {
        if (absolute >= image.token_start and absolute < @as(u64, image.token_start) + image.embedding.token_count) return true;
    }
    return false;
}
