//! Actual Omni audio-tower entry for numerical and latency qualification.
const std = @import("std");
const c = @import("clef_c");
const Backend = @import("models/clef/backend.zig");
const Audio = @import("models/clef/audio.zig");
const timer = @import("clef_runtime_c");
const a = std.heap.c_allocator;

fn read(path: [:0]const u8) ![]f32 {
    const file = c.fopen(path, "rb") orelse return error.OpenFeatures;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadFeatures;
    const length = c.ftell(file);
    if (length < 0 or @mod(length, 4) != 0 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadFeatures;
    const values = try a.alloc(f32, @intCast(@divTrunc(length, 4)));
    errdefer a.free(values);
    if (c.fread(values.ptr, 4, values.len, file) != values.len) return error.ReadFeatures;
    return values;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var args = std.process.Args.Iterator.init(init.args);
    _ = args.next();
    const model = args.next() orelse return error.MissingModel;
    const input = args.next() orelse return error.MissingFeatures;
    const count = args.next() orelse return error.MissingFrameCount;
    const output = args.next() orelse return error.MissingOutput;
    const frames = try std.fmt.parseInt(usize, count, 10);
    const values = try read(input);
    defer a.free(values);
    if (values.len != frames * 128) return error.InvalidFeatureShape;
    const engine = try Backend.Engine.open(model);
    defer engine.close();
    const feature = Audio.Features{ .mel = values, .frames = frames, .padded_frames = frames, .token_count = Audio.tokenCount(frames) };
    const hidden = try engine.audioFeatures(&.{feature}, null, null);
    var timings: [8]f64 = undefined;
    for (&timings) |*sample| {
        const start = now();
        _ = try engine.audioFeatures(&.{feature}, null, null);
        sample.* = now() - start;
    }
    std.mem.sort(f64, &timings, {}, std.sort.asc(f64));
    const file = c.fopen(output, "wb") orelse return error.OpenOutput;
    defer _ = c.fclose(file);
    if (c.fwrite(hidden.ptr, 4, hidden.len, file) != hidden.len) return error.WriteOutput;
    std.debug.print("Omni audio: frames={d} tokens={d} values={d} median_ms={d:.3}\n", .{ frames, feature.token_count, hidden.len, (timings[3] + timings[4]) * 500 });
}

fn now() f64 {
    var value: timer.struct_timespec = undefined;
    _ = timer.clock_gettime(timer.CLOCK_MONOTONIC, &value);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_nsec)) / 1e9;
}
