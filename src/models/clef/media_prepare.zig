//! Per-request CPU preparation and feature ownership. Decoded media belongs to
//! the request arena; only immutable model towers and scratch stay in Engine.
const std = @import("std");
const types = @import("types.zig");
const Record = @import("record.zig");
const Backend = @import("backend.zig");
const Source = @import("media_source.zig");
const Image = @import("image.zig");
const Video = @import("video.zig");
const Audio = @import("audio.zig");
const Codec = @import("codec.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const Input = union(enum) {
    image: Image.Patches,
    video: Video.Patches,
    samples: []const f32,
    audio: Audio.Features,
};
const Entry = struct { spec: Record.MediaSpec, input: Input };

pub const Plan = struct {
    entries: []Entry,
    specs: []Record.MediaSpec,

    /// No GPU lease or model mutex is needed to fetch/decode and prepare media.
    pub fn prepare(a: Allocator, sources: []const types.MediaSource, omni: bool, cancel: Backend.Cancel, context: ?*anyopaque) !Plan {
        var entries: std.ArrayList(Entry) = .empty;
        for (sources) |source| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            switch (source.kind) {
                .image => {
                    const bytes = try Source.loadCancelable(a, source.source, cancel, context);
                    defer a.free(bytes);
                    var rgb = try Image.decode(a, bytes);
                    defer rgb.deinit(a);
                    const patches = try Image.preprocess(a, rgb, if (omni) .omni() else .{});
                    try entries.append(a, .{ .spec = .{ .kind = .image, .grid = patches.grid, .token_count = patches.token_count, .label = source.label }, .input = .{ .image = patches } });
                },
                .audio => {
                    if (!omni) return error.UnsupportedAudio;
                    const samples = try decodeAudio(a, source.source, cancel, context);
                    try entries.append(a, .{ .spec = .{ .kind = .audio, .token_count = 0, .label = source.label }, .input = .{ .samples = samples } });
                },
                .video => {
                    var frames: []Image.Rgb = undefined;
                    var times: []const f64 = undefined;
                    var samples: []const f32 = &.{};
                    if (source.source == .object and source.source.object.get("frames") != null) {
                        const input_frames = source.source.object.get("frames").?;
                        if (input_frames != .array or input_frames.array.items.len == 0) return error.InvalidVideoFrames;
                        frames = try a.alloc(Image.Rgb, input_frames.array.items.len);
                        for (input_frames.array.items, frames) |frame, *out| {
                            const bytes = try Source.loadCancelable(a, frame, cancel, context);
                            defer a.free(bytes);
                            out.* = try Image.decode(a, bytes);
                        }
                        const timestamps = source.source.object.get("timestamps");
                        const owned_times = try a.alloc(f64, frames.len);
                        if (timestamps) |supplied| {
                            if (supplied != .array or supplied.array.items.len != frames.len) return error.InvalidVideoTimestamps;
                            for (supplied.array.items, owned_times) |value, *time| time.* = try real(value);
                        } else {
                            const fps = if (source.source.object.get("fps")) |value| try real(value) else 2;
                            if (!(fps > 0)) return error.InvalidVideoRate;
                            for (owned_times, 0..) |*time, i| time.* = @as(f64, @floatFromInt(i)) / fps;
                        }
                        times = owned_times;
                        if (source.source.object.get("audio")) |audio| {
                            if (!omni) return error.UnsupportedAudio;
                            samples = try decodeAudio(a, audio, cancel, context);
                        }
                    } else {
                        const bytes = try Source.loadCancelable(a, source.source, cancel, context);
                        defer a.free(bytes);
                        const decoded = try Codec.decode(a, bytes, true, .{ .max_frame_pixels = 16 * 1024 * 1024 });
                        frames = decoded.frames;
                        times = decoded.timestamps;
                        if (frames.len == 0) return error.EmptyVideo;
                        if (omni and decoded.samples.len != 0) {
                            samples = try alignAudio(a, decoded.samples, decoded.audio_start_seconds - times[0]);
                        }
                    }
                    const patches = try Video.prepare(a, frames, times, if (omni) .omni() else .{});
                    const index = entries.items.len;
                    try entries.append(a, .{ .spec = .{ .kind = .video, .grid = patches.grid, .token_count = patches.token_count, .timestamps = patches.timestamps, .label = source.label }, .input = .{ .video = patches } });
                    if (samples.len != 0) {
                        entries.items[index].spec.soundtrack = entries.items.len;
                        try entries.append(a, .{ .spec = .{ .kind = .audio, .token_count = 0 }, .input = .{ .samples = samples } });
                    }
                },
            }
        }
        var clips: std.ArrayList([]const f32) = .empty;
        for (entries.items) |entry| if (entry.input == .samples) try clips.append(a, entry.input.samples);
        if (clips.items.len != 0) {
            const features = try Audio.features(a, clips.items);
            var i: usize = 0;
            for (entries.items) |*entry| if (entry.input == .samples) {
                entry.input = .{ .audio = features[i] };
                entry.spec.token_count = features[i].token_count;
                i += 1;
            };
        }
        const specs = try a.alloc(Record.MediaSpec, entries.items.len);
        for (entries.items, specs) |entry, *spec| spec.* = entry.spec;
        return .{ .entries = try entries.toOwnedSlice(a), .specs = specs };
    }

    /// Towers reuse engine scratch, so retain each source's feature rows before
    /// preparing another source. The caller holds the engine mutex throughout.
    pub fn embeddings(self: Plan, a: Allocator, backend: *Backend.Engine, record: *const Record.Record, cancel: Backend.Cancel, context: ?*anyopaque) ![]Backend.Injection {
        if (record.media_rows.len != self.entries.len) return error.MediaPlacementMismatch;
        const inputs = try a.alloc(Backend.Injection, self.entries.len);
        var audio: std.ArrayList(Audio.Features) = .empty;
        for (self.entries) |entry| if (entry.input == .audio) try audio.append(a, entry.input.audio);
        const packed_audio = if (audio.items.len != 0) try backend.audioFeatures(audio.items, cancel, context) else &.{};
        var offset: usize = 0;
        for (self.entries, record.media_rows, inputs) |entry, rows, *out| {
            switch (entry.input) {
                .image, .video => {
                    const patches = switch (entry.input) {
                        .image => |image| image.patches,
                        .video => |video| video.patches,
                        else => unreachable,
                    };
                    const features = try backend.visionFeatures(patches, entry.spec.grid, cancel, context);
                    const stack = try a.alloc([]const f32, features.deepstack.len);
                    for (features.deepstack, stack) |values, *copy| copy.* = try a.dupe(f32, values);
                    out.* = .{ .rows = rows.rows, .features = try a.dupe(f32, features.features), .deepstack = stack };
                },
                .audio => |features| {
                    const n = features.token_count * backend.config.H;
                    if (n > packed_audio.len - offset) return error.InvalidAudioFeatures;
                    out.* = .{ .rows = rows.rows, .features = try a.dupe(f32, packed_audio[offset..][0..n]) };
                    offset += n;
                },
                .samples => return error.MediaNotPrepared,
            }
        }
        return inputs;
    }
};

fn real(value: Value) !f64 {
    const number: f64 = switch (value) {
        .float => value.float,
        .integer => @floatFromInt(value.integer),
        else => return error.ExpectedNumber,
    };
    if (!std.math.isFinite(number)) return error.NonFiniteMediaValue;
    return number;
}
fn decodeAudio(a: Allocator, source: Value, cancel: Backend.Cancel, context: ?*anyopaque) ![]const f32 {
    const values = if (source == .array) source else if (source == .object) source.object.get("samples") orelse .null else .null;
    if (values != .null) {
        if (values != .array or values.array.items.len == 0) return error.InvalidAudioSamples;
        if (source == .object) if (source.object.get("sample_rate")) |rate| {
            if (try real(rate) != 16000) return error.UnsupportedSampleRate;
        };
        const samples = try a.alloc(f32, values.array.items.len);
        for (values.array.items, samples) |value, *out| {
            out.* = @floatCast(try real(value));
            if (!std.math.isFinite(out.*)) return error.NonFiniteMediaValue;
        }
        return samples;
    }
    const bytes = try Source.loadCancelable(a, source, cancel, context);
    defer a.free(bytes);
    const decoded = try Codec.decode(a, bytes, false, .{});
    if (decoded.samples.len == 0) return error.EmptyAudio;
    return decoded.samples;
}
fn alignAudio(a: Allocator, samples: []const f32, seconds: f64) ![]const f32 {
    if (!std.math.isFinite(seconds) or @abs(seconds) > 600) return error.InvalidAudioOffset;
    const offset: isize = @intFromFloat(@round(seconds * 16000));
    if (offset <= 0) return samples[@min(samples.len, @as(usize, @intCast(-offset)))..];
    const padding: usize = @intCast(offset);
    const aligned = try a.alloc(f32, padding + samples.len);
    @memset(aligned[0..padding], 0);
    @memcpy(aligned[padding..], samples);
    return aligned;
}
