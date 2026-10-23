//! Model-specific image resizing, patch order and synthetic-token layout.
//! Pixels and patches cross the public C ABI using the libc allocator. The
//! decoder owns RGB; each successful preprocessing/layout call transfers its
//! newly allocated arrays to the caller, released by the matching free API.
const std = @import("std");
const c = @import("image_c");
const Image = c.ds4_image;
const Patches = c.ds4_image_patches;
const DeepSeekPatches = c.ds4_deepseek4_image_patches;
const Layout = c.ds4_deepseek4_image_layout;
const allocator = std.heap.c_allocator;

const image_start: u8 = c.DS4_DEEPSEEK4_IMAGE_START;
const image_pad: u8 = c.DS4_DEEPSEEK4_IMAGE_PAD;
const image_pixel: u8 = c.DS4_DEEPSEEK4_IMAGE;
const image_newline: u8 = c.DS4_DEEPSEEK4_IMAGE_NEWLINE;
const image_end: u8 = c.DS4_DEEPSEEK4_IMAGE_END;

fn fail(message: [*:0]const u8, error_text: [*c]u8, capacity: usize) c_int {
    if (error_text != null and capacity != 0) _ = c.snprintf(error_text, capacity, "%s", message);
    return 0;
}

fn alignUp(value: u32, factor: u32) u32 {
    return (value +% factor -% 1) / factor *% factor;
}

const Size = struct { height: u32, width: u32 };

fn glmResize(height: u32, width: u32, minimum: u32, maximum: u32) ?Size {
    @setFloatMode(.optimized); // The original image TU is compiled with -ffast-math.
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const temporal = 2;
    const factor = 28;
    const pixels_per_token: u64 = temporal * factor * factor;
    const min_pixels = @as(u64, minimum) * pixels_per_token;
    const max_pixels = @as(u64, maximum) * pixels_per_token;
    var size = Size{ .height = alignUp(height, factor), .width = alignUp(width, factor) };
    var budget = @as(u64, temporal) * size.height * size.width;
    if (budget < min_pixels) {
        const min_area: f64 = @floatFromInt(min_pixels);
        const scale = c.sqrt(min_area / (2.0 * h * w));
        size.height = alignUp(@intFromFloat(c.ceil(h * scale)), factor);
        size.width = alignUp(@intFromFloat(c.ceil(w * scale)), factor);
        budget = @as(u64, temporal) * size.height * size.width;
    }
    if (budget > max_pixels) {
        if (max_pixels < temporal * factor * factor) return null;
        var low: u32 = 1;
        var high = height;
        size = .{ .height = factor, .width = factor };
        while (low <= high) {
            const content_height = low + (high - low) / 2;
            const content_h: f64 = @floatFromInt(content_height);
            const scaled_width: u32 = @intFromFloat(c.floor(w * content_h / h));
            const content_width = @max(@as(u32, 1), scaled_width);
            const candidate = Size{ .height = alignUp(content_height, factor), .width = alignUp(content_width, factor) };
            if (@as(u64, temporal) * candidate.height * candidate.width <= max_pixels) {
                size = candidate;
                low = content_height + 1;
            } else high = content_height - 1;
        }
    }
    return size;
}

fn cubic(input: f64) f64 {
    @setFloatMode(.optimized);
    const x = @abs(input);
    const a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

// Antialiased bicubic filtering keeps double accumulators and rounds each
// channel to an integer before storing float pixels, as the original C does.
// The support bounds deliberately truncate toward zero, rather than floor.
fn resizeRgb(source: []const u8, source_width: u32, source_height: u32, destination: []f32, width: u32, height: u32, stride: u32) void {
    @setFloatMode(.optimized);
    const scale_x = @as(f64, @floatFromInt(source_width)) / @as(f64, @floatFromInt(width));
    const scale_y = @as(f64, @floatFromInt(source_height)) / @as(f64, @floatFromInt(height));
    const filter_x = if (scale_x >= 1.0) 1.0 / scale_x else 1.0;
    const filter_y = if (scale_y >= 1.0) 1.0 / scale_y else 1.0;
    const support_x = if (scale_x >= 1.0) 2.0 * scale_x else 2.0;
    const support_y = if (scale_y >= 1.0) 2.0 * scale_y else 2.0;
    const source_rows: i32 = @intCast(source_height);
    const source_columns: i32 = @intCast(source_width);
    for (0..height) |dy| {
        const y: f64 = @floatFromInt(dy);
        const center_y = scale_y * (y + 0.5);
        const first_y: i32 = @intFromFloat(center_y - support_y + 0.5);
        const after_y: i32 = @intFromFloat(center_y + support_y + 0.5);
        const y0 = @max(@as(i32, 0), first_y);
        const y1 = @min(source_rows, after_y);
        for (0..width) |dx| {
            const x: f64 = @floatFromInt(dx);
            const center_x = scale_x * (x + 0.5);
            const first_x: i32 = @intFromFloat(center_x - support_x + 0.5);
            const after_x: i32 = @intFromFloat(center_x + support_x + 0.5);
            const x0 = @max(@as(i32, 0), first_x);
            const x1 = @min(source_columns, after_x);
            var sum = [_]f64{ 0, 0, 0 };
            var weight_sum: f64 = 0;
            var iy = y0;
            while (iy < y1) : (iy += 1) {
                const sample_y: f64 = @floatFromInt(iy);
                const source_row: usize = @intCast(iy);
                const wy = cubic((sample_y + 0.5 - center_y) * filter_y);
                var ix = x0;
                while (ix < x1) : (ix += 1) {
                    const sample_x: f64 = @floatFromInt(ix);
                    const source_column: u32 = @intCast(ix);
                    const wx = cubic((sample_x + 0.5 - center_x) * filter_x);
                    const weight = wx * wy;
                    const pixel = source[(source_row * source_width + source_column) * 3 ..][0..3];
                    for (&sum, pixel) |*channel, value| channel.* += @as(f64, @floatFromInt(value)) * weight;
                    weight_sum += weight;
                }
            }
            const pixel = destination[(dy * stride + dx) * 3 ..][0..3];
            for (pixel, sum) |*channel, value| {
                const rounded = c.round(value / weight_sum);
                channel.* = @floatCast(@min(255.0, @max(0.0, rounded)));
            }
        }
    }
}

// Both GLM and Qwen group neighbouring 2x2 patches before channel-major
// samples. GLM duplicates each still-image patch along the temporal axis.
fn mergeWindowPatches(canvas: []const f32, width: u32, height: u32, patch: u32, temporal: u32, output: []f32) usize {
    var index: usize = 0;
    for (0..height / patch / 2) |block_y| {
        for (0..width / patch / 2) |block_x| {
            for (0..2) |merge_y| {
                for (0..2) |merge_x| {
                    const patch_y = block_y * 2 + merge_y;
                    const patch_x = block_x * 2 + merge_x;
                    for (0..3) |channel| {
                        for (0..temporal) |_| {
                            for (0..patch) |y| {
                                for (0..patch) |x| {
                                    output[index] = canvas[((patch_y * patch + y) * width + patch_x * patch + x) * 3 + channel];
                                    index += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return index;
}

pub export fn ds4_image_preprocess_glm53(output: ?*Patches, input: ?*const Image, minimum: u32, maximum: u32, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    @setFloatMode(.optimized);
    const out = output orelse return 0;
    out.* = std.mem.zeroes(Patches);
    const image = input orelse return fail("invalid GLM 5.3 image preprocessing parameters", error_text, capacity);
    if (image.rgb == null or image.width == 0 or image.height == 0 or minimum == 0 or maximum < minimum or maximum > 8000)
        return fail("invalid GLM 5.3 image preprocessing parameters", error_text, capacity);
    const size = glmResize(image.height, image.width, minimum, maximum) orelse return fail("image token budget is too small", error_text, capacity);
    const h: f64 = @floatFromInt(image.height);
    const w: f64 = @floatFromInt(image.width);
    const target_h: f64 = @floatFromInt(size.height);
    const target_w: f64 = @floatFromInt(size.width);
    var scale = c.fmin(target_h / h, target_w / w);
    const min_pixels = @as(u64, 2 * 28 * 28) * minimum;
    if (@as(u64, 2) * image.height * image.width >= min_pixels and scale > 1.0) scale = 1.0;
    const scaled_height: u32 = @intFromFloat(c.floor(h * scale));
    const scaled_width: u32 = @intFromFloat(c.floor(w * scale));
    const content_height = @min(size.height, @max(@as(u32, 1), scaled_height));
    const content_width = @min(size.width, @max(@as(u32, 1), scaled_width));
    const canvas = allocator.alloc(f32, @as(usize, size.height) * size.width * 3) catch return fail("unable to allocate resized image", error_text, capacity);
    defer allocator.free(canvas);
    @memset(canvas, 0);
    const rgb = image.rgb[0 .. @as(usize, image.height) * image.width * 3];
    if (content_width == image.width and content_height == image.height) {
        for (0..content_height) |y| {
            for (0..content_width) |x| {
                const src = rgb[(y * image.width + x) * 3 ..][0..3];
                const dst = canvas[(y * size.width + x) * 3 ..][0..3];
                for (dst, src) |*value, byte| value.* = @floatFromInt(byte);
            }
        }
    } else resizeRgb(rgb, image.width, image.height, canvas, content_width, content_height, size.width);
    const mean = [_]f32{ 0.48145466, 0.4578275, 0.40821073 };
    const stddev = [_]f32{ 0.26862954, 0.26130258, 0.27577711 };
    for (0..size.height) |y| {
        for (0..size.width) |x| {
            const pixel = canvas[(y * size.width + x) * 3 ..][0..3];
            for (pixel, mean, stddev) |*channel, m, s| {
                const value: f32 = if (x < content_width and y < content_height) channel.* else 0.0;
                channel.* = (value / 255.0 - m) / s;
            }
        }
    }
    const grid_height = size.height / 14;
    const grid_width = size.width / 14;
    const count = grid_height * grid_width;
    const patches = allocator.alloc(f32, @as(usize, count) * 3 * 2 * 14 * 14) catch return fail("unable to allocate vision patches", error_text, capacity);
    if (mergeWindowPatches(canvas, size.width, size.height, 14, 2, patches) != patches.len) {
        allocator.free(patches);
        return fail("internal vision patch layout mismatch", error_text, capacity);
    }
    out.* = .{
        .content_width = content_width,
        .content_height = content_height,
        .padded_width = size.width,
        .padded_height = size.height,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .patch_count = count,
        .image_token_count = count / 4,
        .patches = patches.ptr,
    };
    return 1;
}

pub export fn ds4_image_preprocess_qwen4(output: ?*Patches, input: ?*const Image, minimum: u32, maximum: u32, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    @setFloatMode(.optimized);
    const out = output orelse return 0;
    out.* = std.mem.zeroes(Patches);
    const image = input orelse return fail("invalid Qwen3.8 image preprocessing parameters", error_text, capacity);
    if (image.rgb == null or image.width == 0 or image.height == 0 or minimum == 0 or maximum < minimum)
        return fail("invalid Qwen3.8 image preprocessing parameters", error_text, capacity);
    const h: f64 = @floatFromInt(image.height);
    const w: f64 = @floatFromInt(image.width);
    if (c.fmax(h, w) / c.fmin(h, w) > 200.0) return fail("image aspect ratio must be below 200", error_text, capacity);
    // Qwen2VL smart_resize: round to the factor, then scale into [min, max].
    const min_pixels = @as(f64, @floatFromInt(minimum)) * 32 * 32;
    const max_pixels = @as(f64, @floatFromInt(maximum)) * 32 * 32;
    var h_bar = c.fmax(32, c.round(h / 32) * 32);
    var w_bar = c.fmax(32, c.round(w / 32) * 32);
    if (h_bar * w_bar > max_pixels) {
        const beta = c.sqrt(h * w / max_pixels);
        h_bar = c.fmax(32, c.floor(h / beta / 32) * 32);
        w_bar = c.fmax(32, c.floor(w / beta / 32) * 32);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = c.sqrt(min_pixels / (h * w));
        h_bar = c.ceil(h * beta / 32) * 32;
        w_bar = c.ceil(w * beta / 32) * 32;
    }
    const height: u32 = @intFromFloat(h_bar);
    const width: u32 = @intFromFloat(w_bar);
    if (width > c.DS4_IMAGE_MAX_DIMENSION or height > c.DS4_IMAGE_MAX_DIMENSION or @as(u64, width) * height > c.DS4_IMAGE_MAX_PIXELS)
        return fail("resized image is too large", error_text, capacity);
    const canvas = allocator.alloc(f32, @as(usize, height) * width * 3) catch return fail("unable to allocate resized image", error_text, capacity);
    defer allocator.free(canvas);
    const rgb = image.rgb[0 .. @as(usize, image.height) * image.width * 3];
    if (width == image.width and height == image.height) {
        for (canvas, rgb) |*value, byte| value.* = @floatFromInt(byte);
    } else resizeRgb(rgb, image.width, image.height, canvas, width, height, width);
    for (canvas) |*value| value.* = (value.* / 255.0 - 0.5) / 0.5;
    const grid_height = height / 16;
    const grid_width = width / 16;
    const count = grid_height * grid_width;
    const patches = allocator.alloc(f32, @as(usize, count) * 3 * 16 * 16) catch return fail("unable to allocate vision patches", error_text, capacity);
    _ = mergeWindowPatches(canvas, width, height, 16, 1, patches);
    out.* = .{
        .content_width = width,
        .content_height = height,
        .padded_width = width,
        .padded_height = height,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .patch_count = count,
        .image_token_count = count / 4,
        .patches = patches.ptr,
    };
    return 1;
}

pub export fn ds4_image_patches_free(patches: ?*Patches) callconv(.c) void {
    const p = patches orelse return;
    c.free(p.patches);
    p.* = std.mem.zeroes(Patches);
}

fn gridTokens(height: u32, width: u32, llm: *Size, v41: bool) u32 {
    const h = (height / 14 + 2) / 3;
    const w = (width / 14 + 2) / 3;
    var count: u64 = @as(u64, h) * (w +% 1) + 2;
    if (!v41) {
        if (h & 1 != 0) count += w +% 1;
        if (((h + 1) / 2 *% (w +% 1)) & 1 != 0) count += 2;
    }
    llm.* = .{ .height = h, .width = w };
    return @intCast(@min(count, std.math.maxInt(u32)));
}

fn solveResize(height: u32, width: u32, budget: u32, v41: bool) ?Size {
    @setFloatMode(.optimized);
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const token_budget: f64 = @floatFromInt(budget);
    const ratio = h / w;
    const max_w_float = c.sqrt((token_budget - 2.0) / ratio + 0.25) - 0.5;
    const max_h_float = max_w_float * ratio;
    var max_h: u32 = undefined;
    var max_w: u32 = undefined;
    var size: Size = undefined;
    if (max_w_float < 1.0) {
        max_w = 1;
        max_h = (budget - 2) / (max_w + 1);
        if (!v41) max_h &= ~@as(u32, 1);
        if (max_h == 0) return null;
        size = .{ .width = max_w * 42, .height = max_h * 42 };
    } else if (max_h_float < (if (v41) @as(f64, 1.0) else 2.0)) {
        max_h = if (v41) 1 else 2;
        max_w = (budget - 2) / max_h - 1;
        if (max_w <= 1) return null;
        size = .{ .width = max_w * 42, .height = max_h * 42 };
    } else {
        max_w = @intFromFloat(c.floor(max_w_float));
        max_h = @intFromFloat(c.floor(max_h_float));
        if (!v41) max_h &= ~@as(u32, 1);
        if (max_w == 0 or max_h == 0) return null;
        const max_columns: f64 = @floatFromInt(max_w);
        const max_rows: f64 = @floatFromInt(max_h);
        const scale = c.fmin(max_columns * 42.0 / w, max_rows * 42.0 / h);
        const patch_columns: u32 = @intFromFloat(c.floor(w * scale / 14.0));
        const patch_rows: u32 = @intFromFloat(c.floor(h * scale / 14.0));
        size = .{
            .width = patch_columns * 14,
            .height = patch_rows * 14,
        };
    }
    return if (size.width != 0 and size.height != 0) size else null;
}

fn safeResize(height: u32, width: u32, size: *Size, llm: *Size, v41: bool) bool {
    const maximum: u32 = if (v41) 1024 else 381;
    var budget = maximum;
    while (gridTokens(size.height, size.width, llm, v41) > maximum) {
        if (budget <= 4) return false;
        size.* = solveResize(height, width, budget, v41) orelse return false;
        budget -= 1;
    }
    return true;
}

fn preprocessDeepSeek(output: ?*DeepSeekPatches, input: ?*const Image, error_text: [*c]u8, capacity: usize, v41: bool) c_int {
    @setFloatMode(.optimized);
    const out = output orelse return 0;
    out.* = std.mem.zeroes(DeepSeekPatches);
    const image = input orelse return fail("invalid DeepSeek V4 image preprocessing input", error_text, capacity);
    if (image.rgb == null or image.width == 0 or image.height == 0)
        return fail("invalid DeepSeek V4 image preprocessing input", error_text, capacity);
    var planned = Size{ .width = image.width, .height = image.height };
    if (!v41 and @as(u64, planned.width) > @as(u64, planned.height) * 8) planned.width = planned.height * 8;
    const pixels = @as(u64, planned.width) * planned.height;
    const minimum: u32 = if (v41) 295936 else 147456;
    if (pixels < minimum) {
        const min_area: f64 = @floatFromInt(minimum);
        const area: f64 = @floatFromInt(pixels);
        const w: f64 = @floatFromInt(planned.width);
        const h: f64 = @floatFromInt(planned.height);
        const scale = c.sqrt(min_area / area);
        const scaled_width: u32 = @intFromFloat(w * scale);
        const scaled_height: u32 = @intFromFloat(h * scale);
        planned.width = @max(@as(u32, 1), scaled_width);
        planned.height = @max(@as(u32, 1), scaled_height);
    }
    var size = Size{ .width = alignUp(planned.width, 14), .height = alignUp(planned.height, 14) };
    var llm: Size = undefined;
    if (!safeResize(planned.height, planned.width, &size, &llm, v41))
        return fail("unable to fit image within the vision token budget", error_text, capacity);
    if (size.width > c.DS4_IMAGE_MAX_DIMENSION or size.height > c.DS4_IMAGE_MAX_DIMENSION or @as(u64, size.width) * size.height > c.DS4_IMAGE_MAX_PIXELS)
        return fail("resized image exceeds the decoded image limits", error_text, capacity);
    const canvas = allocator.alloc(f32, @as(usize, size.width) * size.height * 3) catch return fail("unable to allocate resized image", error_text, capacity);
    defer allocator.free(canvas);
    @memset(canvas, 127.0);
    var content = size;
    var offset_x: u32 = 0;
    var offset_y: u32 = 0;
    const force_resize = !v41 and @as(u64, image.width) >= @as(u64, image.height) * 8;
    if (!force_resize) {
        const w: f64 = @floatFromInt(image.width);
        const h: f64 = @floatFromInt(image.height);
        const target_w: f64 = @floatFromInt(size.width);
        const target_h: f64 = @floatFromInt(size.height);
        const scale = c.fmin(target_w / w, target_h / h);
        // lrint obeys the C floating-point rounding mode (normally ties-even);
        // do not replace it with Zig @round or C round's ties-away behavior.
        const rounded_width: u32 = @intCast(c.lrint(w * scale));
        const rounded_height: u32 = @intCast(c.lrint(h * scale));
        content.width = @min(size.width, @max(@as(u32, 1), rounded_width));
        content.height = @min(size.height, @max(@as(u32, 1), rounded_height));
        const margin_x: f64 = @floatFromInt(size.width - content.width);
        const margin_y: f64 = @floatFromInt(size.height - content.height);
        offset_x = @intCast(c.lrint(margin_x * 0.5));
        offset_y = @intCast(c.lrint(margin_y * 0.5));
    }
    const rgb = image.rgb[0 .. @as(usize, image.width) * image.height * 3];
    resizeRgb(rgb, image.width, image.height, canvas[(@as(usize, offset_y) * size.width + offset_x) * 3 ..], content.width, content.height, size.width);
    for (canvas) |*value| value.* = value.* / 127.5 - 1.0;
    const grid_height = size.height / 14;
    const grid_width = size.width / 14;
    const count = grid_height * grid_width;
    const patches = allocator.alloc(f32, @as(usize, count) * 3 * 14 * 14) catch return fail("unable to allocate vision patches", error_text, capacity);
    var index: usize = 0;
    for (0..grid_height) |patch_y| {
        for (0..grid_width) |patch_x| {
            for (0..3) |channel| {
                for (0..14) |y| {
                    for (0..14) |x| {
                        patches[index] = canvas[((patch_y * 14 + y) * size.width + patch_x * 14 + x) * 3 + channel];
                        index += 1;
                    }
                }
            }
        }
    }
    if (index != patches.len) {
        allocator.free(patches);
        return fail("internal DeepSeek vision patch layout mismatch", error_text, capacity);
    }
    out.* = .{
        .content_width = content.width,
        .content_height = content.height,
        .padded_width = size.width,
        .padded_height = size.height,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .llm_grid_width = llm.width,
        .llm_grid_height = llm.height,
        .patch_count = count,
        .patches = patches.ptr,
    };
    return 1;
}

pub export fn ds4_image_preprocess_deepseek4(out: ?*DeepSeekPatches, image: ?*const Image, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    return preprocessDeepSeek(out, image, error_text, capacity, false);
}
pub export fn ds4_image_preprocess_deepseek41(out: ?*DeepSeekPatches, image: ?*const Image, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    return preprocessDeepSeek(out, image, error_text, capacity, true);
}

pub export fn ds4_deepseek41_image_layout_build(output: ?*Layout, height: u32, width: u32, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    const out = output orelse return 0;
    out.* = std.mem.zeroes(Layout);
    const count = @as(u64, height) * (@as(u64, width) + 1) + 2;
    if (height == 0 or width == 0 or count > 1024) return fail("invalid V4.1 image token grid", error_text, capacity);
    const types = allocator.alloc(u8, @intCast(count)) catch return fail("unable to allocate V4.1 image layout", error_text, capacity);
    const permutation = allocator.alloc(u32, @as(usize, height) * width) catch {
        allocator.free(types);
        return fail("unable to allocate V4.1 image layout", error_text, capacity);
    };
    var token: usize = 0;
    var image: usize = 0;
    types[token] = image_start;
    token += 1;
    for (0..height) |_| {
        for (0..width) |_| {
            types[token] = image_pixel;
            token += 1;
            permutation[image] = @intCast(image);
            image += 1;
        }
        types[token] = image_newline;
        token += 1;
    }
    types[token] = image_end;
    token += 1;
    out.* = .{
        .token_count = @intCast(token),
        .image_count = @intCast(image),
        .types = types.ptr,
        .perm = permutation.ptr,
    };
    return 1;
}

pub export fn ds4_deepseek4_image_layout_build(output: ?*Layout, height: u32, width: u32, start_pos: u32, error_text: [*c]u8, capacity: usize) callconv(.c) c_int {
    const out = output orelse return 0;
    out.* = std.mem.zeroes(Layout);
    if (height == 0 or width == 0) return fail("invalid DeepSeek image token grid", error_text, capacity);
    const rows = height +% (height & 1);
    const row_length = width +% 1;
    const compress_pad = 3 - start_pos % 4;
    const pad_last = ((rows / 2 *% row_length) & 1) * 2;
    const body_count = @as(u64, rows) * row_length;
    const token_count = compress_pad + 1 + body_count + pad_last + 1;
    const image_count = @as(u64, height) * width;
    if (token_count > 384 or token_count > std.math.maxInt(u32) or image_count > std.math.maxInt(u32))
        return fail("DeepSeek image token layout exceeds its model limit", error_text, capacity);
    const body_types = allocator.alloc(u8, @intCast(body_count)) catch return fail("unable to allocate DeepSeek image token layout", error_text, capacity);
    defer allocator.free(body_types);
    const body_images = allocator.alloc(i32, @intCast(body_count)) catch return fail("unable to allocate DeepSeek image token layout", error_text, capacity);
    defer allocator.free(body_images);
    const types = allocator.alloc(u8, @intCast(token_count)) catch return fail("unable to allocate DeepSeek image token layout", error_text, capacity);
    const permutation = allocator.alloc(u32, @intCast(image_count)) catch {
        allocator.free(types);
        return fail("unable to allocate DeepSeek image token layout", error_text, capacity);
    };
    for (0..rows) |row| {
        for (0..row_length) |column| {
            const index = row * row_length + column;
            if (row < height and column < width) {
                body_types[index] = image_pixel;
                body_images[index] = @bitCast(@as(u32, @intCast(row * width + column)));
            } else {
                body_types[index] = if (row < height) image_newline else image_pad;
                body_images[index] = -1;
            }
        }
    }
    var token: usize = 0;
    var image: usize = 0;
    @memset(types[0..compress_pad], image_pad);
    token += compress_pad;
    types[token] = image_start;
    token += 1;
    // V4 interleaves each pair of rows before compression; the permutation
    // maps only IMAGE entries back to their original row-major patch index.
    for (0..rows / 2) |pair| {
        for (0..row_length) |column| {
            for (0..2) |row_in_pair| {
                const source = (pair * 2 + row_in_pair) * row_length + column;
                types[token] = body_types[source];
                token += 1;
                if (body_images[source] >= 0) {
                    permutation[image] = @intCast(body_images[source]);
                    image += 1;
                }
            }
        }
    }
    @memset(types[token..][0..pad_last], image_pad);
    token += pad_last;
    types[token] = image_end;
    token += 1;
    if (token != token_count or image != image_count) {
        allocator.free(types);
        allocator.free(permutation);
        return fail("internal DeepSeek image token layout mismatch", error_text, capacity);
    }
    out.* = .{
        .token_count = @intCast(token),
        .image_count = @intCast(image),
        .types = types.ptr,
        .perm = permutation.ptr,
    };
    return 1;
}

pub export fn ds4_deepseek4_image_patches_free(patches: ?*DeepSeekPatches) callconv(.c) void {
    const p = patches orelse return;
    c.free(p.patches);
    p.* = std.mem.zeroes(DeepSeekPatches);
}
pub export fn ds4_deepseek4_image_layout_free(layout: ?*Layout) callconv(.c) void {
    const l = layout orelse return;
    c.free(l.types);
    c.free(l.perm);
    l.* = std.mem.zeroes(Layout);
}

// A block includes alignment PADs immediately before IMAGE_START. Return 1
// for a complete block, 0 at end and -1 for malformed synthetic tokens. On
// failure cursor/output positions remain untouched, matching the C contract.
pub export fn ds4_deepseek4_next_image_span(tokens: [*c]const c_int, count: u32, vocab: u32, cursor: ?*u32, block_start: ?*u32, start: ?*u32, end: ?*u32) callconv(.c) c_int {
    if (tokens == null or cursor == null or block_start == null or start == null or end == null or vocab == 0 or cursor.?.* > count) return -1;
    const ids = tokens[0..count];
    var index = cursor.?.*;
    while (index < count and ids[index] >= 0 and @as(u32, @intCast(ids[index])) < vocab) : (index += 1) {}
    if (index == count) {
        cursor.?.* = index;
        return 0;
    }
    if (ids[index] < 0) return -1;
    const block = index;
    const vocabulary: c_int = @bitCast(vocab);
    while (index < count and ids[index] == vocabulary +% image_pad) : (index += 1) {}
    if (index == count or ids[index] != vocabulary +% image_start) return -1;
    const first = index;
    index += 1;
    while (index < count) : (index += 1) {
        const token = ids[index];
        if (token < vocabulary or token > vocabulary +% image_end or token == vocabulary +% image_start) return -1;
        if (token == vocabulary +% image_end) {
            block_start.?.* = block;
            start.?.* = first;
            end.?.* = index;
            cursor.?.* = index + 1;
            return 1;
        }
    }
    return -1;
}

// Prefill must never split a DeepSeek image block across graph chunks.
pub export fn ds4_deepseek4_prefill_chunk(tokens: [*c]const c_int, count: u32, vocab: u32, position: u32, cap: u32, chunk: ?*u32) callconv(.c) c_int {
    if (tokens == null or chunk == null or vocab == 0 or cap == 0 or position >= count) return 0;
    var limit = position + @min(count - position, cap);
    var cursor = position;
    while (true) {
        var block: u32 = undefined;
        var start: u32 = undefined;
        var end: u32 = undefined;
        const parsed = ds4_deepseek4_next_image_span(tokens, count, vocab, &cursor, &block, &start, &end);
        if (parsed < 0) return 0;
        if (parsed == 0 or block >= limit) break;
        const after = end + 1;
        if (after > limit) {
            if (after - position > cap) {
                if (block == position) return 0;
                limit = block;
                break;
            }
            limit = after;
        }
    }
    chunk.?.* = limit - position;
    return @intFromBool(chunk.?.* != 0);
}

// Fill inclusive absolute raw-key bounds. Text is causal; queries inside an
// image block may see neighbouring image tokens within the model's 384 cap.
pub export fn ds4_deepseek4_attention_bounds(tokens: [*c]const c_int, count: u32, vocab: u32, position: u32, raw_rows: u32, window: u32, bounds: [*c]u32) callconv(.c) c_int {
    if (tokens == null or bounds == null or count == 0 or vocab == 0 or position > std.math.maxInt(u32) - count or raw_rows == 0 or raw_rows > position + count) return 0;
    const first_raw = position + count - raw_rows;
    const last_raw = first_raw + raw_rows - 1;
    for (0..count) |q| {
        const qpos = position + @as(u32, @intCast(q));
        var lo = first_raw;
        if (window != 0 and qpos +% 1 > window) lo = @max(lo, qpos +% 1 - window);
        bounds[2 * q] = lo;
        bounds[2 * q + 1] = @min(qpos, last_raw);
    }
    var cursor: u32 = 0;
    while (true) {
        var block: u32 = undefined;
        var start: u32 = undefined;
        var end: u32 = undefined;
        const parsed = ds4_deepseek4_next_image_span(tokens, count, vocab, &cursor, &block, &start, &end);
        if (parsed < 0) return 0;
        if (parsed == 0) break;
        var q = start;
        while (q <= end) : (q += 1) {
            const qpos = position + q;
            const left = @min(q - start, @as(u32, 383));
            const right = @min(end - q, @as(u32, 384));
            const left_add: u32 = if (window != 0 and left >= window) left - (window - 1) else 0;
            var lo: u32 = 0;
            if (window != 0) {
                const back = window - 1 + left_add;
                lo = if (qpos > back) qpos - back else 0;
            }
            const hi = qpos +% right;
            bounds[@as(usize, q) * 2] = @max(lo, first_raw);
            bounds[@as(usize, q) * 2 + 1] = @min(hi, last_raw);
        }
    }
    return 1;
}
