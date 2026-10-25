//! CPU-only actual preprocessing checks against independent numerical fixtures.
const std = @import("std");
const c = @import("image_c");
const Image = @import("models/clef/image.zig");
const Audio = @import("models/clef/audio.zig");
const Video = @import("models/clef/video.zig");
const Codec = @import("models/clef/codec.zig");
const a = std.heap.c_allocator;

fn read(directory: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ directory, name }, 0);
    defer a.free(path);
    const file = c.fopen(path, "rb") orelse return error.OpenFixture;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadFixture;
    const size = c.ftell(file);
    if (size < 0 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadFixture;
    const bytes = try a.alloc(u8, @intCast(size));
    errdefer a.free(bytes);
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadFixture;
    return bytes;
}

fn compare(directory: []const u8, name: []const u8, actual: []const f32, tolerance: f32) !void {
    const bytes = try read(directory, name);
    defer a.free(bytes);
    const expected = std.mem.bytesAsSlice(f32, bytes);
    if (expected.len != actual.len) return error.ShapeMismatch;
    var maximum: f32 = 0;
    var squared: f64 = 0;
    for (actual, expected) |x, y| {
        if (!std.math.isFinite(x)) return error.NonfiniteOutput;
        maximum = @max(maximum, @abs(x - y));
        squared += @as(f64, x - y) * (x - y);
    }
    std.debug.print("{s}: values={d} max_gap={d:.8} rms={d:.8}\n", .{ name, actual.len, maximum, @sqrt(squared / @as(f64, @floatFromInt(actual.len))) });
    if (maximum > tolerance) return error.NumericalMismatch;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var args = std.process.Args.Iterator.init(init.args);
    _ = args.next();
    const directory = args.next() orelse return error.MissingFixtureDirectory;
    const encoded = try read(directory, "image.png");
    defer a.free(encoded);
    var rgb = try Image.decode(a, encoded);
    defer rgb.deinit(a);
    const pixels = try read(directory, "image.rgb");
    defer a.free(pixels);
    if (!std.mem.eql(u8, pixels, rgb.pixels)) return error.DecodeMismatch;
    var patches = try Image.preprocess(a, rgb, .{});
    defer patches.deinit(a);
    try compare(directory, "image-patches.f32", patches.patches, 0);
    var indices: [64]i32 = undefined;
    var weights: [64]f32 = undefined;
    Image.positionInterpolation(4, 4, 48, 2, &indices, &weights);
    try compare(directory, "position-weights.f32", &weights, 0.00001);
    const expected_indices = try read(directory, "position-indices.i32");
    defer a.free(expected_indices);
    if (!std.mem.eql(u8, expected_indices, std.mem.sliceAsBytes(&indices))) return error.PositionMismatch;
    const wav = try read(directory, "audio.wav");
    defer a.free(wav);
    const samples = try Audio.decodeWav(a, wav, 16000 * 60);
    defer a.free(samples);
    try compare(directory, "audio-samples.f32", samples, 0);
    const features = try Audio.features(a, &.{samples});
    defer {
        for (features) |*feature| feature.deinit(a);
        a.free(features);
    }
    try compare(directory, "audio-mel.f32", features[0].mel, 0.0001);
    var layout = try Audio.layout(a, features);
    defer layout.deinit(a);
    if (layout.tokens != features[0].token_count) return error.AudioLayoutMismatch;
    const indices_sampled = try Video.sampleIndices(a, 96, 24, 2, 4, 768);
    defer a.free(indices_sampled);
    if (!std.mem.eql(usize, indices_sampled, &.{ 0, 14, 27, 41, 54, 68, 81, 95 })) return error.SamplingMismatch;
    // Three real RGB frames, including a distinct second temporal tap and an
    // odd final frame. Expected patches come from NumPy's HF reshape/transpose.
    const second_bytes = try read(directory, "second.png");
    defer a.free(second_bytes);
    var second = try Image.decode(a, second_bytes);
    defer second.deinit(a);
    var video = try Video.prepare(a, &.{ rgb, second, rgb }, &.{ 0, 0.5, 1 }, .{ .min_pixels = 4096, .max_pixels = 25165824 });
    defer video.deinit(a);
    try compare(directory, "video-patches.f32", video.patches, 0);
    if (!std.mem.eql(f64, video.timestamps, &.{ 0.25, 1 })) return error.TimestampMismatch;
    const movie = try read(directory, "video.mkv");
    defer a.free(movie);
    var decoded = try Codec.decode(a, movie, true, .{ .max_samples = 16000 * 60 });
    defer decoded.deinit(a);
    const expected_video = try read(directory, "decoded-video.rgb");
    defer a.free(expected_video);
    if (decoded.frames.len != 4 or !std.mem.eql(f64, decoded.timestamps, &.{ 0, 0.5, 1, 1.5 })) return error.VideoDecodeMismatch;
    for (decoded.frames, 0..) |frame, i| {
        if (!std.mem.eql(u8, frame.pixels, expected_video[i * frame.pixels.len ..][0..frame.pixels.len])) return error.VideoDecodeMismatch;
    }
    try compare(directory, "decoded-audio.f32", decoded.samples, 0);
    const offset_movie = try read(directory, "video-offset.mkv");
    defer a.free(offset_movie);
    var offset_decoded = try Codec.decode(a, offset_movie, true, .{ .max_samples = 16000 * 60 });
    defer offset_decoded.deinit(a);
    if (offset_decoded.frames.len != 4 or !std.mem.eql(f64, offset_decoded.timestamps, &.{ 2, 2.5, 3, 3.5 }) or offset_decoded.audio_start_seconds != 2) return error.VideoOffsetMismatch;
    for (offset_decoded.frames, decoded.frames) |offset_frame, frame| {
        if (!std.mem.eql(u8, offset_frame.pixels, frame.pixels)) return error.VideoOffsetMismatch;
    }
    try compare(directory, "decoded-audio.f32", offset_decoded.samples, 0);
    std.debug.print("Clef CPU media: native image/patches, audio mel/layout, video sampling and memory codec pass.\n", .{});
}
