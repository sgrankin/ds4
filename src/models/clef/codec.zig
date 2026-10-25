//! Ownership around the narrow libav memory decoder. Source bytes have already
//! been fetched by the caller; codec demuxing cannot open secondary URLs/files.
const std = @import("std");
const c = @import("clef_codec_c");
const media = @import("media_types.zig");

pub const Limits = struct {
    max_frames: usize = 768,
    max_samples: usize = 16000 * 600,
    max_frame_pixels: usize = 256 * 32 * 32,
    fps: f64 = 2,
};

pub const Decoded = struct {
    frames: []media.Rgb,
    timestamps: []f64,
    samples: []f32,
    source_fps: f64,
    audio_start_seconds: f64,

    pub fn deinit(self: *Decoded, allocator: std.mem.Allocator) void {
        for (self.frames) |*frame| frame.deinit(allocator);
        allocator.free(self.frames);
        allocator.free(self.timestamps);
        allocator.free(self.samples);
        self.* = undefined;
    }
};

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8, video: bool, limits: Limits) !Decoded {
    var raw: c.ds4_clef_media = std.mem.zeroes(c.ds4_clef_media);
    var error_text: [256]u8 = @splat(0);
    if (c.ds4_clef_media_decode(bytes.ptr, bytes.len, @intFromBool(video), limits.fps, limits.max_frames, limits.max_samples, limits.max_frame_pixels, &raw, &error_text, error_text.len) == 0) {
        std.log.err("encoded media: {s}", .{std.mem.sliceTo(&error_text, 0)});
        return error.InvalidEncodedMedia;
    }
    defer c.ds4_clef_media_free(&raw);
    const frames = try allocator.alloc(media.Rgb, raw.frame_count);
    errdefer allocator.free(frames);
    var initialized: usize = 0;
    errdefer for (frames[0..initialized]) |*frame| frame.deinit(allocator);
    const timestamps = try allocator.alloc(f64, raw.frame_count);
    errdefer allocator.free(timestamps);
    const samples = try allocator.dupe(f32, if (raw.sample_count == 0) &.{} else raw.samples[0..raw.sample_count]);
    errdefer allocator.free(samples);
    for (frames, timestamps, 0..) |*frame, *time, i| {
        const source = raw.frames[i];
        const pixels = try allocator.dupe(u8, source.rgb[0 .. @as(usize, @intCast(source.width)) * @as(usize, @intCast(source.height)) * 3]);
        frame.* = .{ .width = @intCast(source.width), .height = @intCast(source.height), .pixels = pixels };
        time.* = source.seconds;
        initialized += 1;
    }
    return .{ .frames = frames, .timestamps = timestamps, .samples = samples, .source_fps = raw.source_fps, .audio_start_seconds = raw.audio_start_seconds };
}
