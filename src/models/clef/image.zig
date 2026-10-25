//! Clef image decoding and Qwen vision patch preparation.
//! The uint8 antialiased bicubic schedule follows MIT-licensed Cleffa's
//! clef_image.c (24fc91e2), itself matching PyTorch's separable CPU resize.
//! Existing Iris codecs provide PNG/JPEG decoding; no new image codec is added.
const std = @import("std");
const c = @import("image_c");
const media = @import("media_types.zig");
pub const Rgb = media.Rgb;
pub const Params = media.ImageParams;
pub const Patches = media.ImagePatches;
pub const Size = struct { height: u32, width: u32 };

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !Rgb {
    if (bytes.len == 0 or bytes.len > 64 * 1024 * 1024) return error.InvalidImage;
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) {
        const decoded = c.png_load_mem(bytes.ptr, bytes.len);
        if (decoded == null) return error.InvalidPng;
        defer c.png_free(decoded);
        return copyRgb(allocator, @intCast(decoded.*.width), @intCast(decoded.*.height), @intCast(decoded.*.channels), decoded.*.data);
    }
    if (std.mem.startsWith(u8, bytes, "\xff\xd8")) {
        const decoded = c.jpeg_load_mem(bytes.ptr, bytes.len);
        if (decoded == null) return error.InvalidJpeg;
        defer c.jpeg_free(decoded);
        return copyRgb(allocator, @intCast(decoded.*.width), @intCast(decoded.*.height), @intCast(decoded.*.channels), decoded.*.data);
    }
    return error.UnsupportedImage;
}

fn copyRgb(allocator: std.mem.Allocator, width: u32, height: u32, channels: usize, source: [*c]const u8) !Rgb {
    const pixels = try allocator.alloc(u8, @as(usize, width) * height * 3);
    for (0..@as(usize, width) * height) |i| {
        const src = source + i * channels;
        const dst = pixels[i * 3 ..][0..3];
        if (channels < 3) @memset(dst, src[0]) else @memcpy(dst, src[0..3]);
    }
    return .{ .width = width, .height = height, .pixels = pixels };
}

/// Python's round() uses ties-to-even; image dimensions are nonnegative.
pub fn roundEven(value: f64) f64 {
    const lower = @floor(value);
    const fraction = value - lower;
    return if (fraction < 0.5) lower else if (fraction > 0.5) lower + 1 else if (@mod(lower, 2) == 0) lower else lower + 1;
}

pub fn smartResize(height: u32, width: u32, params: Params) !Size {
    if (height == 0 or width == 0 or params.patch == 0 or params.merge == 0 or params.min_pixels == 0 or params.max_pixels < params.min_pixels) return error.InvalidImageGeometry;
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const factor: f64 = @floatFromInt(params.patch * params.merge);
    if (@max(h, w) / @min(h, w) > 200) return error.ImageAspectRatio;
    var rh = roundEven(h / factor) * factor;
    var rw = roundEven(w / factor) * factor;
    const min_pixels: f64 = @floatFromInt(params.min_pixels);
    const max_pixels: f64 = @floatFromInt(params.max_pixels);
    if (rh * rw > max_pixels) {
        const scale = @sqrt(h * w / max_pixels);
        rh = @max(factor, @floor(h / scale / factor) * factor);
        rw = @max(factor, @floor(w / scale / factor) * factor);
    } else if (rh * rw < min_pixels) {
        const scale = @sqrt(min_pixels / (h * w));
        rh = @ceil(h * scale / factor) * factor;
        rw = @ceil(w * scale / factor) * factor;
    }
    if (rh < factor or rw < factor or rh > 16384 or rw > 16384) return error.ImageTooLarge;
    return .{ .height = @intFromFloat(rh), .width = @intFromFloat(rw) };
}

fn cubic(input: f64) f64 {
    const x = @abs(input);
    if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}

const Range = struct { first: usize, count: usize };
const Filter = struct {
    ranges: []Range,
    weights: []i16,
    stride: usize,
    precision: u6,

    fn init(allocator: std.mem.Allocator, input: usize, output: usize) !Filter {
        const scale = @as(f64, @floatFromInt(input)) / @as(f64, @floatFromInt(output));
        const support = 2 * @max(1, scale);
        const stride: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const ranges = try allocator.alloc(Range, output);
        errdefer allocator.free(ranges);
        const weights = try allocator.alloc(i16, output * stride);
        errdefer allocator.free(weights);
        const temporary = try allocator.alloc(f64, output * stride);
        defer allocator.free(temporary);
        @memset(temporary, 0);
        var max_weight: f64 = 0;
        for (ranges, 0..) |*range, i| {
            const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
            const first: usize = @intFromFloat(@max(0, @trunc(center - support + 0.5)));
            const end: usize = @intFromFloat(@min(@as(f64, @floatFromInt(input)), @trunc(center + support + 0.5)));
            const count = @min(stride, end - first);
            range.* = .{ .first = first, .count = count };
            var sum: f64 = 0;
            for (0..count) |j| {
                const value = cubic((@as(f64, @floatFromInt(j + first)) - center + 0.5) / @max(1, scale));
                temporary[i * stride + j] = value;
                sum += value;
            }
            if (sum != 0) for (temporary[i * stride ..][0..count]) |*value| {
                value.* /= sum;
                max_weight = @max(max_weight, value.*);
            };
        }
        var precision: u6 = 0;
        while (precision < 22) : (precision += 1) {
            const next = 0.5 + max_weight * @as(f64, @floatFromInt(@as(u32, 1) << @as(u5, @intCast(precision + 1))));
            if (@as(u32, @intFromFloat(next)) >= 32768) break;
        }
        const unit: f64 = @floatFromInt(@as(u32, 1) << @as(u5, @intCast(precision)));
        for (weights, temporary) |*weight, value| {
            const scaled = value * unit;
            weight.* = @intFromFloat(@trunc(scaled + if (scaled < 0) @as(f64, -0.5) else 0.5));
        }
        return .{ .ranges = ranges, .weights = weights, .stride = stride, .precision = precision };
    }

    fn deinit(self: Filter, allocator: std.mem.Allocator) void {
        allocator.free(self.ranges);
        allocator.free(self.weights);
    }

    fn sample(self: Filter, source: []const u8, stride: usize, index: usize) u8 {
        const range = self.ranges[index];
        var sum: i64 = @as(i64, 1) << (self.precision - 1);
        for (0..range.count) |j| sum += @as(i64, source[(range.first + j) * stride]) * self.weights[index * self.stride + j];
        return @intCast(@min(255, @max(0, sum) >> self.precision));
    }
};

pub fn resize(allocator: std.mem.Allocator, source: Rgb, size: Size) !Rgb {
    if (source.width == 0 or source.height == 0 or size.width == 0 or size.height == 0 or source.pixels.len != @as(usize, source.width) * source.height * 3) return error.InvalidImageGeometry;
    const pixels = try allocator.alloc(u8, @as(usize, size.width) * size.height * 3);
    errdefer allocator.free(pixels);
    if (source.width == size.width and source.height == size.height) {
        @memcpy(pixels, source.pixels);
        return .{ .width = size.width, .height = size.height, .pixels = pixels };
    }
    var temporary: ?[]u8 = null;
    defer if (temporary) |buffer| allocator.free(buffer);
    var current = source.pixels;
    if (source.width != size.width) {
        const horizontal = try Filter.init(allocator, source.width, size.width);
        defer horizontal.deinit(allocator);
        const output = if (source.height == size.height) pixels else blk: {
            temporary = try allocator.alloc(u8, @as(usize, size.width) * source.height * 3);
            break :blk temporary.?;
        };
        for (0..source.height) |y| for (0..size.width) |x| for (0..3) |channel| {
            output[(y * size.width + x) * 3 + channel] = horizontal.sample(current[y * source.width * 3 + channel ..], 3, x);
        };
        current = output;
    }
    if (source.height != size.height) {
        const vertical = try Filter.init(allocator, source.height, size.height);
        defer vertical.deinit(allocator);
        const row = @as(usize, size.width) * 3;
        for (0..size.height) |y| for (0..row) |x| {
            pixels[y * row + x] = vertical.sample(current[x..], row, y);
        };
    }
    return .{ .width = size.width, .height = size.height, .pixels = pixels };
}

pub fn preprocess(allocator: std.mem.Allocator, input: Rgb, params: Params) !Patches {
    if (params.temporal == 0) return error.InvalidImageGeometry;
    var resized = try resize(allocator, input, try smartResize(input.height, input.width, params));
    defer resized.deinit(allocator);
    return patchesForPair(allocator, resized, resized, params.patch, params.merge, params.temporal);
}

pub fn patchesForPair(allocator: std.mem.Allocator, first: Rgb, second: Rgb, patch: u32, merge: u32, temporal: u32) !Patches {
    if (patch == 0 or merge == 0 or temporal != 2 or first.width != second.width or first.height != second.height or first.width % (patch * merge) != 0 or first.height % (patch * merge) != 0) return error.InvalidImageGeometry;
    const gh = first.height / patch;
    const gw = first.width / patch;
    const dim = 3 * temporal * patch * patch;
    const patches = try allocator.alloc(f32, @as(usize, gh) * gw * dim);
    var normalized: [256]f32 = undefined;
    for (&normalized, 0..) |*value, i| value.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
    var row: usize = 0;
    for (0..gh / merge) |by| for (0..gw / merge) |bx| for (0..merge) |my| for (0..merge) |mx| {
        const py = by * merge + my;
        const px = bx * merge + mx;
        for (0..3) |channel| for (0..temporal) |t| for (0..patch) |y| for (0..patch) |x| {
            const rgb = if (t == 0) first else second;
            const src = ((py * patch + y) * rgb.width + px * patch + x) * 3 + channel;
            patches[row * dim + ((channel * temporal + t) * patch + y) * patch + x] = normalized[rgb.pixels[src]];
        };
        row += 1;
    };
    return .{ .width = first.width, .height = first.height, .grid = .{ 1, gh, gw }, .patch_dim = dim, .token_count = @as(usize, gh) * gw / (merge * merge), .patches = patches };
}

pub fn positionInterpolation(grid_h: u32, grid_w: u32, side: u32, merge: u32, indices: []i32, weights: []f32) void {
    var row: usize = 0;
    for (0..grid_h / merge) |by| for (0..grid_w / merge) |bx| for (0..merge) |my| for (0..merge) |mx| {
        const y = by * merge + my;
        const x = bx * merge + mx;
        const h = linspace(grid_h, side, y);
        const w = linspace(grid_w, side, x);
        const h0: u32 = @intFromFloat(h);
        const w0: u32 = @intFromFloat(w);
        const h1 = @min(h0 + 1, side - 1);
        const w1 = @min(w0 + 1, side - 1);
        const hf = h - @as(f32, @floatFromInt(h0));
        const wf = w - @as(f32, @floatFromInt(w0));
        const neighbors: [4]i32 = .{ @intCast(h0 * side + w0), @intCast(h0 * side + w1), @intCast(h1 * side + w0), @intCast(h1 * side + w1) };
        @memcpy(indices[row * 4 ..][0..4], &neighbors);
        const coefficients: [4]f32 = .{ (1 - hf) * (1 - wf), (1 - hf) * wf, hf * (1 - wf), hf * wf };
        @memcpy(weights[row * 4 ..][0..4], &coefficients);
        row += 1;
    };
}

fn linspace(count: u32, side: u32, index: usize) f32 {
    if (count <= 1) return 0;
    const end: f32 = @floatFromInt(side - 1);
    const step = end / @as(f32, @floatFromInt(count - 1));
    return if (index < count / 2) step * @as(f32, @floatFromInt(index)) else end - step * @as(f32, @floatFromInt(count - 1 - index));
}
