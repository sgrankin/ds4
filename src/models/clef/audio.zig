//! Omni's native CPU audio frontend: 16kHz mono WAV and Whisper log-mel.
//! The mixed-radix FFT/sparse filter schedule follows MIT Cleffa clef_audio.c;
//! learned audio-tower execution is owned by the GPU model component.
const std = @import("std");
const media = @import("media_types.zig");
const tables = @import("audio_tables.zig");
const Codec = @import("codec.zig");
pub const Features = media.AudioFeatures;

fn read16(bytes: []const u8) u16 {
    return std.mem.readInt(u16, bytes[0..2], .little);
}
fn read32(bytes: []const u8) u32 {
    return std.mem.readInt(u32, bytes[0..4], .little);
}

pub fn decodeWav(allocator: std.mem.Allocator, bytes: []const u8, max_samples: usize) ![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE") or @as(usize, read32(bytes[4..])) + 8 != bytes.len) return error.InvalidWav;
    var format: ?[]const u8 = null;
    var data: ?[]const u8 = null;
    var at: usize = 12;
    while (at < bytes.len) {
        if (bytes.len - at < 8) return error.InvalidWav;
        const length: usize = read32(bytes[at + 4 ..]);
        const tag = bytes[at..][0..4];
        at += 8;
        if (length + (length & 1) > bytes.len - at) return error.InvalidWav;
        if (std.mem.eql(u8, tag, "fmt ")) {
            if (format != null) return error.InvalidWav;
            format = bytes[at..][0..length];
        } else if (std.mem.eql(u8, tag, "data")) {
            if (data != null) return error.InvalidWav;
            data = bytes[at..][0..length];
        }
        at += length + (length & 1);
    }
    const fmt = format orelse return error.InvalidWav;
    const payload = data orelse return error.InvalidWav;
    if (fmt.len != 16 and (fmt.len != 18 or read16(fmt[16..]) != 0)) return error.UnsupportedWav;
    const kind = read16(fmt);
    const width: usize = switch (kind) {
        1 => 2,
        3 => 4,
        else => return error.UnsupportedWav,
    };
    if (read16(fmt[2..]) != 1 or read32(fmt[4..]) != 16000 or read32(fmt[8..]) != 16000 * width or read16(fmt[12..]) != width or read16(fmt[14..]) != width * 8 or payload.len % width != 0) return error.UnsupportedWav;
    const count = payload.len / width;
    if (count == 0 or (max_samples != 0 and count > max_samples)) return error.AudioTooLarge;
    const samples = try allocator.alloc(f32, count);
    errdefer allocator.free(samples);
    for (samples, 0..) |*value, i| {
        value.* = if (kind == 1) @as(f32, @floatFromInt(@as(i16, @bitCast(read16(payload[i * 2 ..]))))) / 32768 else @bitCast(read32(payload[i * 4 ..]));
        if (!std.math.isFinite(value.*)) return error.NonfiniteAudio;
    }
    return samples;
}

/// Common WAV input stays in the small native parser. Other encoded formats
/// use the platform codec boundary to downmix/resample into the same samples.
pub fn decodeEncoded(allocator: std.mem.Allocator, bytes: []const u8, max_samples: usize) ![]f32 {
    if (std.mem.startsWith(u8, bytes, "RIFF")) {
        return decodeWav(allocator, bytes, max_samples) catch |err| switch (err) {
            error.UnsupportedWav => decodeResampled(allocator, bytes, max_samples),
            else => err,
        };
    }
    return decodeResampled(allocator, bytes, max_samples);
}

fn decodeResampled(allocator: std.mem.Allocator, bytes: []const u8, max_samples: usize) ![]f32 {
    var decoded = try Codec.decode(allocator, bytes, false, .{ .max_samples = max_samples });
    defer decoded.deinit(allocator);
    const samples = decoded.samples;
    decoded.samples = &.{};
    return samples;
}

pub fn tokenCount(frames: usize) usize {
    return 13 * (frames / 100) + (frames % 100 + 7) / 8;
}

const Complex = struct { re: f64, im: f64 };

fn fft(output: []Complex, input: [*]const Complex, stride: usize, count: usize, roots: *const [400]Complex) void {
    if (count == 1) {
        output[0] = input[0];
        return;
    }
    const radix: usize = if (count % 2 == 0) 2 else 5;
    const width = count / radix;
    const root_stride = 400 / count;
    for (0..radix) |r| fft(output[r * width ..][0..width], input + r * stride, stride * radix, width, roots);
    var merged: [400]Complex = undefined;
    for (0..count) |k| {
        var sum = Complex{ .re = 0, .im = 0 };
        for (0..radix) |r| {
            const value = output[r * width + k % width];
            const root = roots[(r * k * root_stride) % 400];
            sum.re += value.re * root.re - value.im * root.im;
            sum.im += value.re * root.im + value.im * root.re;
        }
        merged[k] = sum;
    }
    @memcpy(output, merged[0..count]);
}

fn firstSample(frame: usize, longest: usize) usize {
    const low = @as(i64, @intCast(frame * 160)) - 200;
    if (low <= 0) return 0;
    const high = low + 399;
    if (high < longest) return @intCast(low);
    return @intCast(@min(low, 2 * @as(i64, @intCast(longest)) - 2 - high));
}

/// Every clip is right-padded to the longest before centered STFT, as in the
/// reference. Peak normalization includes padded frames; only each clip's own
/// frames are retained for its encoder. Zero-only padded windows skip the FFT.
pub fn features(allocator: std.mem.Allocator, clips: []const []const f32) ![]Features {
    if (clips.len == 0) return error.EmptyAudio;
    var longest: usize = 0;
    for (clips) |clip| {
        if (clip.len == 0) return error.EmptyAudio;
        for (clip) |value| if (!std.math.isFinite(value)) return error.NonfiniteAudio;
        longest = @max(longest, clip.len);
    }
    if (longest <= 200) return error.AudioTooShort;
    const padded_frames = longest / 160;
    const result = try allocator.alloc(Features, clips.len);
    errdefer allocator.free(result);
    var initialized: usize = 0;
    errdefer for (result[0..initialized]) |*item| item.deinit(allocator);
    var roots: [400]Complex = undefined;
    for (&roots, 0..) |*root, k| {
        const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / 400;
        root.* = .{ .re = @cos(angle), .im = @sin(angle) };
    }
    for (clips, result) |clip, *item| {
        const frames = @min((clip.len + 159) / 160, padded_frames);
        item.* = .{ .mel = try allocator.alloc(f32, 128 * frames), .frames = frames, .padded_frames = padded_frames, .token_count = tokenCount(frames) };
        initialized += 1;
        var peak = -std.math.inf(f32);
        for (0..padded_frames) |t| {
            if (firstSample(t, longest) >= clip.len) {
                peak = @max(peak, -10);
                continue;
            }
            var block: [400]Complex = undefined;
            var spectrum: [400]Complex = undefined;
            for (&block, 0..) |*value, j| {
                var index = @as(i64, @intCast(t * 160 + j)) - 200;
                if (index < 0) index = -index;
                if (index >= longest) index = 2 * @as(i64, @intCast(longest)) - 2 - index;
                const sample: f32 = if (index < clip.len) clip[@intCast(index)] else 0;
                const windowed: f32 = sample * tables.hann[j];
                value.* = .{ .re = windowed, .im = 0 };
            }
            fft(&spectrum, &block, 1, 400, &roots);
            var power: [201]f32 = undefined;
            for (&power, spectrum[0..201]) |*value, bin| {
                const re: f32 = @floatCast(bin.re);
                const im: f32 = @floatCast(bin.im);
                const magnitude = std.math.hypot(re, im);
                value.* = magnitude * magnitude;
                if (!std.math.isFinite(value.*)) return error.AudioPowerOverflow;
            }
            for (0..128) |mel| {
                var sum: f32 = 0;
                for (tables.filters[tables.offsets[mel]..tables.offsets[mel + 1]]) |filter| sum += power[filter.bin] * filter.weight;
                const log = @log10(@max(sum, 1e-10));
                if (!std.math.isFinite(log)) return error.AudioPowerOverflow;
                if (t < frames) item.mel[mel * frames + t] = log;
                peak = @max(peak, log);
            }
        }
        for (item.mel) |*value| value.* = (@max(value.*, peak - 8) + 4) / 4;
    }
    return result;
}

pub const Chunk = struct { clip: usize, frame: usize, frames: usize, first_token: usize };
pub const Layout = struct {
    chunks: []Chunk,
    positions: []i32,
    window_start: []i32,
    window_end: []i32,
    max_chunk_frames: usize,
    tokens: usize,

    pub fn deinit(self: *Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.chunks);
        allocator.free(self.positions);
        allocator.free(self.window_start);
        allocator.free(self.window_end);
        self.* = undefined;
    }
};

pub fn layout(allocator: std.mem.Allocator, inputs: []const Features) !Layout {
    if (inputs.len == 0) return error.EmptyAudio;
    var count: usize = 0;
    var tokens: usize = 0;
    var max_chunk_frames: usize = 0;
    for (inputs) |input| {
        if (input.frames == 0 or input.frames > input.padded_frames or input.token_count != tokenCount(input.frames)) return error.InvalidAudioFeatures;
        count += (input.frames + 99) / 100;
        tokens += input.token_count;
        max_chunk_frames = @max(max_chunk_frames, @min(input.frames, 100));
    }
    const chunks = try allocator.alloc(Chunk, count);
    errdefer allocator.free(chunks);
    const positions = try allocator.alloc(i32, tokens);
    errdefer allocator.free(positions);
    const start = try allocator.alloc(i32, tokens);
    errdefer allocator.free(start);
    const end = try allocator.alloc(i32, tokens);
    errdefer allocator.free(end);
    const window = (max_chunk_frames + 7) / 8 * 8;
    var chunk_index: usize = 0;
    var base: usize = 0;
    for (inputs, 0..) |input, clip| {
        var row = base;
        var frame: usize = 0;
        while (frame < input.frames) {
            const frames = @min(input.frames - frame, 100);
            chunks[chunk_index] = .{ .clip = clip, .frame = frame, .frames = frames, .first_token = row };
            chunk_index += 1;
            for (0..(frames + 7) / 8) |p| {
                positions[row] = @intCast(p);
                row += 1;
            }
            frame += frames;
        }
        var at = base;
        while (at < base + input.token_count) {
            const stop = @min(base + input.token_count, at + window);
            for (at..stop) |t| {
                start[t] = @intCast(at);
                end[t] = @intCast(stop);
            }
            at = stop;
        }
        base += input.token_count;
    }
    return .{ .chunks = chunks, .positions = positions, .window_start = start, .window_end = end, .max_chunk_frames = max_chunk_frames, .tokens = tokens };
}
