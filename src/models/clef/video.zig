//! Frame selection, geometry and temporal patch preparation. Encoded video
//! decoding is a separate narrow codec boundary; this module accepts RGB frames.
const std = @import("std");
const image = @import("image.zig");
const media = @import("media_types.zig");
const Codec = @import("codec.zig");
pub const Decoded = Codec.Decoded;
pub const DecodeLimits = Codec.Limits;
pub const Rgb = media.Rgb;
pub const Patches = media.VideoPatches;

pub const Params = struct {
    min_pixels: usize = 4096,
    max_pixels: usize = 25165824,
    /// Dense Clef budgets all frames jointly. Omni uses a per-frame pixel budget.
    total_pixel_budget: bool = true,

    pub fn omni() Params {
        return .{ .min_pixels = 128 * 32 * 32, .max_pixels = 768 * 32 * 32, .total_pixel_budget = false };
    }
};

pub fn decode(allocator: std.mem.Allocator, encoded: []const u8, limits: DecodeLimits) !Decoded {
    return Codec.decode(allocator, encoded, true, limits);
}

pub fn sampleIndices(allocator: std.mem.Allocator, total: usize, source_fps: f64, target_fps: f64, min_frames: usize, max_frames: usize) ![]usize {
    if (total == 0 or !std.math.isFinite(source_fps) or source_fps <= 0 or !std.math.isFinite(target_fps) or target_fps <= 0 or min_frames == 0 or max_frames < min_frames) return error.InvalidVideoGeometry;
    const wanted: usize = @intFromFloat(@as(f64, @floatFromInt(total)) / source_fps * target_fps);
    const count = @min(total, @min(max_frames, @max(min_frames, wanted)));
    const indices = try allocator.alloc(usize, count);
    if (count == 1) {
        indices[0] = 0;
        return indices;
    }
    const step = @as(f64, @floatFromInt(total - 1)) / @as(f64, @floatFromInt(count - 1));
    for (indices, 0..) |*index, i| index.* = @intFromFloat(image.roundEven(step * @as(f64, @floatFromInt(i))));
    return indices;
}

pub fn resizeGeometry(frames: usize, height: u32, width: u32, params: Params) !image.Size {
    if (frames == 0 or frames > 768 or height < 32 or width < 32 or params.min_pixels == 0 or params.max_pixels < params.min_pixels) return error.InvalidVideoGeometry;
    if (!params.total_pixel_budget) return image.smartResize(height, width, .{ .min_pixels = params.min_pixels, .max_pixels = params.max_pixels });
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    if (@max(h, w) / @min(h, w) > 200) return error.ImageAspectRatio;
    var rh = image.roundEven(h / 32) * 32;
    var rw = image.roundEven(w / 32) * 32;
    const padded: f64 = @floatFromInt((frames + 1) / 2 * 2);
    const f: f64 = @floatFromInt(frames);
    const minimum: f64 = @floatFromInt(params.min_pixels);
    const maximum: f64 = @floatFromInt(params.max_pixels);
    if (padded * rh * rw > maximum) {
        const scale = @sqrt(f * h * w / maximum);
        rh = @max(32, @floor(h / scale / 32) * 32);
        rw = @max(32, @floor(w / scale / 32) * 32);
    } else if (padded * rh * rw < minimum) {
        const scale = @sqrt(minimum / (f * h * w));
        rh = @ceil(h * scale / 32) * 32;
        rw = @ceil(w * scale / 32) * 32;
    }
    if (rh > 16384 or rw > 16384 or rh * rw > 64 * 1024 * 1024) return error.ImageTooLarge;
    return .{ .height = @intFromFloat(rh), .width = @intFromFloat(rw) };
}

pub fn prepare(allocator: std.mem.Allocator, frames: []const Rgb, timestamps: []const f64, params: Params) !Patches {
    if (frames.len == 0 or frames.len != timestamps.len) return error.InvalidVideoGeometry;
    const size = try resizeGeometry(frames.len, frames[0].height, frames[0].width, params);
    const pairs = (frames.len + 1) / 2;
    const per_pair: usize = @as(usize, size.width / 16) * (size.height / 16) * 1536;
    const patches = try allocator.alloc(f32, pairs * per_pair);
    errdefer allocator.free(patches);
    const times = try allocator.alloc(f64, pairs);
    errdefer allocator.free(times);
    for (0..pairs) |i| {
        const first = i * 2;
        const second = @min(first + 1, frames.len - 1);
        if (frames[first].width != frames[0].width or frames[first].height != frames[0].height or frames[second].width != frames[0].width or frames[second].height != frames[0].height or !std.math.isFinite(timestamps[first]) or !std.math.isFinite(timestamps[second])) return error.InvalidVideoGeometry;
        var left = try image.resize(allocator, frames[first], size);
        defer left.deinit(allocator);
        var right = try image.resize(allocator, frames[second], size);
        defer right.deinit(allocator);
        var pair = try image.patchesForPair(allocator, left, right, 16, 2, 2);
        defer pair.deinit(allocator);
        @memcpy(patches[i * per_pair ..][0..per_pair], pair.patches);
        times[i] = (timestamps[first] + timestamps[second]) / 2;
    }
    const gh = size.height / 16;
    const gw = size.width / 16;
    return .{ .width = size.width, .height = size.height, .grid = .{ @intCast(pairs), gh, gw }, .patch_dim = 1536, .token_count = pairs * gh * gw / 4, .patches = patches, .timestamps = times };
}
