//! Native Omni audio encoder. A request's mel clips share CNN padding, then
//! packed bidirectional attention windows; learned weights and scratch belong
//! to this engine. Scheduling follows Cleffa's MIT audio implementation.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const Audio = @import("audio.zig");
const a = std.heap.c_allocator;
const View = b.View;
const width: u32 = 1280;
const ff_width: u32 = 5120;
const channels: u32 = 480;
const output_width: u32 = 2048;
const max_batch: usize = 8;
const Norm = struct { weight: View, bias: View };
const Linear = struct { weight: View, bias: ?View, input: u32, output: u32 };
const Layer = struct { norm1: Norm, norm2: Norm, query: Linear, key: Linear, value: Linear, projection: Linear, up: Linear, down: Linear };
const GemmArgs = extern struct { rows: u32, columns: u32, inner: u32 };
const ConvArgs = extern struct { chunks: u32, height: u32, width: u32, channels: u32 };
const BiasArgs = extern struct { width: u32, rows: u32, mode: u32 };
const NormArgs = extern struct { width: u32 = width, epsilon: f32 = 1e-5 };
const AttentionArgs = extern struct { tokens: u32, width: u32 = width, heads: u32 = 20, scale: f32 = 0.125 };
const GatherArgs = extern struct { rows: u32, width: u32 = width, padded_rows: u32 };

pub const State = struct {
    conv: [3]Linear,
    conv_out: Linear,
    layers: []Layer,
    post_norm: Norm,
    projection1: Linear,
    projection2: Linear,
    position: View,
    scratch: ?Scratch = null,

    pub fn open(engine: *b.Engine) !State {
        if (engine.config.family != .omni) return error.AudioUnsupported;
        try engine.compile(@embedFile("../../metal/clef/audio.metal"));
        const layers = try a.alloc(Layer, 32);
        errdefer a.free(layers);
        var conv: [3]Linear = undefined;
        var storage: [256]u8 = undefined;
        for (&conv, 0..) |*item, i| item.* = try loadConv(engine, try std.fmt.bufPrint(&storage, "thinker.audio_tower.conv2d{d}", .{i + 1}), if (i == 0) 1 else channels);
        for (layers, 0..) |*layer, i| {
            const prefix = try std.fmt.allocPrint(a, "thinker.audio_tower.layers.{d}", .{i});
            defer a.free(prefix);
            layer.* = .{
                .norm1 = try loadNorm(engine, try std.fmt.bufPrint(&storage, "{s}.self_attn_layer_norm", .{prefix})),
                .norm2 = try loadNorm(engine, try std.fmt.bufPrint(&storage, "{s}.final_layer_norm", .{prefix})),
                .query = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.self_attn.q_proj", .{prefix}), width, width, true),
                .key = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.self_attn.k_proj", .{prefix}), width, width, true),
                .value = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.self_attn.v_proj", .{prefix}), width, width, true),
                .projection = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.self_attn.out_proj", .{prefix}), width, width, true),
                .up = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.fc1", .{prefix}), ff_width, width, true),
                .down = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.fc2", .{prefix}), width, ff_width, true),
            };
        }
        const position = try engine.buffer(13 * width * 4);
        const values = b.contents(f32, position, 13 * width);
        const increment = @log(@as(f64, 10000)) / (width / 2 - 1);
        for (0..width / 2) |d| {
            const frequency = @exp(-increment * @as(f64, @floatFromInt(d)));
            for (0..13) |t| {
                const angle = @as(f64, @floatFromInt(t)) * frequency;
                values[t * width + d] = @floatCast(@sin(angle));
                values[t * width + d + width / 2] = @floatCast(@cos(angle));
            }
        }
        return .{
            .conv = conv,
            .conv_out = try loadLinear(engine, "thinker.audio_tower.conv_out", width, 16 * channels, false),
            .layers = layers,
            .post_norm = try loadNorm(engine, "thinker.audio_tower.ln_post"),
            .projection1 = try loadLinear(engine, "thinker.audio_tower.proj1", width, width, true),
            .projection2 = try loadLinear(engine, "thinker.audio_tower.proj2", output_width, width, true),
            .position = position,
        };
    }

    pub fn deinit(self: *State) void {
        if (self.scratch) |*scratch| scratch.deinit();
        a.free(self.layers);
    }

    pub fn run(self: *State, engine: *b.Engine, clips: []const Audio.Features, cancel: b.Cancel, context: ?*anyopaque) ![]const f32 {
        var layout = try Audio.layout(a, clips);
        defer layout.deinit(a);
        if (layout.tokens > 65536) return error.AudioTooLarge;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        const batch: usize = @min(max_batch, layout.chunks.len);
        if (self.scratch == null or self.scratch.?.capacity < layout.tokens or self.scratch.?.chunk_capacity < batch or self.scratch.?.chunk_width < layout.max_chunk_frames) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(engine.gpu, layout.tokens, batch, layout.max_chunk_frames);
        }
        const s = &self.scratch.?;
        const T: u32 = @intCast(layout.tokens);
        @memset(b.contents(u32, s.status, 1), 0);
        @memcpy(b.contents(i32, s.starts, layout.tokens), layout.window_start);
        @memcpy(b.contents(i32, s.ends, layout.tokens), layout.window_end);
        var first: usize = 0;
        while (first < layout.chunks.len) {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            const count: usize = @min(batch, layout.chunks.len - first);
            const input_width: u32 = @intCast(layout.max_chunk_frames);
            const padded_rows = (input_width + 7) / 8;
            const input = b.contents(f32, s.input, count * 128 * input_width);
            @memset(input, 0);
            const indices = b.contents(i32, s.indices, count * padded_rows);
            var valid: usize = 0;
            for (layout.chunks[first..][0..count], 0..) |chunk, ci| {
                const clip = clips[chunk.clip];
                for (0..128) |mel| for (0..chunk.frames) |t| {
                    const value = clip.mel[mel * clip.frames + chunk.frame + t];
                    if (!std.math.isFinite(value)) return error.NonfiniteAudio;
                    input[(ci * 128 + mel) * input_width + t] = value;
                };
                for (0..(chunk.frames + 7) / 8) |t| {
                    indices[valid] = @intCast(ci * padded_rows + t);
                    valid += 1;
                }
            }
            const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
            var pending = true;
            defer if (pending) {
                _ = c.flint_gpu_command_end(engine.gpu, command, true);
            };
            const e = b.Encoder{ .gpu = engine.gpu, .command = command };
            var source = s.input;
            var height: u32 = 128;
            var time = input_width;
            var input_channels: u32 = 1;
            for (self.conv, 0..) |linear, i| {
                const args = ConvArgs{ .chunks = @intCast(count), .height = height, .width = time, .channels = input_channels };
                const rows = @as(u32, @intCast(count)) * ((height + 1) / 2) * ((time + 1) / 2);
                try e.dispatch("clef_audio_im2col", args, &.{ source.bind(1), s.column.bind(2) }, .{ ceil(rows * input_channels * 9, 256), 1, 1 }, 256);
                const output = if (i == 1) s.conv_b else s.conv_a;
                try project(e, linear, s.column, output, rows, 1, null, s.status);
                source = output;
                height = (height + 1) / 2;
                time = (time + 1) / 2;
                input_channels = channels;
            }
            const pack = ConvArgs{ .chunks = @intCast(count), .height = height, .width = time, .channels = input_channels };
            try e.dispatch("clef_audio_conv_pack", pack, &.{ source.bind(1), s.column.bind(2) }, .{ ceil(@as(u32, @intCast(count)) * height * time * channels, 256), 1, 1 }, 256);
            try project(e, self.conv_out, s.column, s.projected, @as(u32, @intCast(count)) * time, 0, null, s.status);
            try e.dispatch("clef_audio_gather", GatherArgs{ .rows = @intCast(valid), .padded_rows = padded_rows }, &.{ s.projected.bind(1), s.indices.bind(2), self.position.bind(3), s.residual.at(layout.chunks[first].first_token * width * 4).bind(4), s.status.bind(5) }, .{ ceil(@as(u32, @intCast(valid)) * width, 256), 1, 1 }, 256);
            pending = false;
            try finish(engine, command, s.status);
            first += count;
        }
        for (self.layers) |layer| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
            var pending = true;
            defer if (pending) {
                _ = c.flint_gpu_command_end(engine.gpu, command, true);
            };
            const e = b.Encoder{ .gpu = engine.gpu, .command = command };
            try normalize(e, layer.norm1, s.residual, s.normalized, T, s.status);
            try project(e, layer.query, s.normalized, s.q, T, 0, null, s.status);
            try project(e, layer.key, s.normalized, s.k, T, 0, null, s.status);
            try project(e, layer.value, s.normalized, s.v, T, 0, null, s.status);
            try e.dispatch("clef_audio_attention", AttentionArgs{ .tokens = T }, &.{ s.q.bind(1), s.k.bind(2), s.v.bind(3), s.starts.bind(4), s.ends.bind(5), s.attention.bind(6), s.status.bind(7) }, .{ T, 20, 1 }, 32);
            try project(e, layer.projection, s.attention, s.normalized, T, 2, s.residual, s.status);
            try normalize(e, layer.norm2, s.residual, s.normalized, T, s.status);
            try project(e, layer.up, s.normalized, s.ff, T, 1, null, s.status);
            try project(e, layer.down, s.ff, s.normalized, T, 2, s.residual, s.status);
            pending = false;
            try finish(engine, command, s.status);
        }
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, command, true);
        };
        const e = b.Encoder{ .gpu = engine.gpu, .command = command };
        try normalize(e, self.post_norm, s.residual, s.normalized, T, s.status);
        try project(e, self.projection1, s.normalized, s.residual, T, 1, null, s.status);
        try project(e, self.projection2, s.residual, s.output, T, 0, null, s.status);
        pending = false;
        try finish(engine, command, s.status);
        return b.contents(f32, s.output, layout.tokens * output_width);
    }
};

fn finish(engine: *b.Engine, command: b.Buffer, status: View) !void {
    if (!c.flint_gpu_command_end(engine.gpu, command, true)) return error.GpuExecution;
    if (b.contents(u32, status, 1)[0] != 0) return error.NonfiniteAudioFeatures;
}
fn loadNorm(engine: *b.Engine, prefix: []const u8) !Norm {
    var name: [256]u8 = undefined;
    const weight_name = try std.fmt.bufPrint(&name, "{s}.weight", .{prefix});
    const tensor = try engine.tensor(weight_name);
    if (tensor.bytes.len != width * 2 and tensor.bytes.len != width * 4) return error.InvalidAudioNorm;
    const weight = try engine.vector(weight_name, 0, false);
    const bias = try engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false);
    return .{ .weight = weight, .bias = bias };
}
fn loadLinear(engine: *b.Engine, prefix: []const u8, output: u32, input: u32, bias: bool) !Linear {
    var name: [256]u8 = undefined;
    const tensor = try engine.tensor(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}));
    if (tensor.dtype != .bf16 or tensor.bytes.len != @as(usize, output) * input * 2) return error.InvalidAudioWeight;
    const weight = try engine.buffer(tensor.bytes.len);
    @memcpy(b.contents(u8, weight, tensor.bytes.len), tensor.bytes);
    return .{ .weight = weight, .bias = if (bias) try engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false) else null, .input = input, .output = output };
}
fn loadConv(engine: *b.Engine, prefix: []const u8, input: u32) !Linear {
    var name: [256]u8 = undefined;
    const tensor = try engine.tensor(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}));
    if (tensor.dtype != .bf16 or tensor.shape.len != 4 or tensor.shape[0] != channels or tensor.shape[1] != 3 or tensor.shape[2] != 3 or tensor.shape[3] != input) return error.InvalidAudioConvolution;
    const weight = try engine.buffer(tensor.bytes.len);
    const destination = b.contents(u16, weight, tensor.bytes.len / 2);
    for (0..channels) |out| for (0..input) |channel| for (0..3) |y| for (0..3) |x| {
        const src = ((out * 3 + y) * 3 + x) * input + channel;
        const dst = out * input * 9 + (channel * 3 + y) * 3 + x;
        destination[dst] = std.mem.readInt(u16, tensor.bytes[src * 2 ..][0..2], .little);
    };
    const bias = try engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false);
    return .{ .weight = weight, .bias = bias, .input = input * 9, .output = channels };
}
fn project(e: b.Encoder, linear: Linear, input: View, output: View, rows: u32, mode: u32, residual: ?View, status: View) !void {
    // Keep the row divisor equal to audio.metal's MPP tile. The measured
    // 16-row FP32 tile avoids padding work on short audio sequences.
    try e.dispatch("clef_audio_gemm", GemmArgs{ .rows = rows, .columns = linear.output, .inner = linear.input }, &.{ input.bind(1), linear.weight.bind(2), output.bind(3) }, .{ ceil(linear.output, 128), ceil(rows, 16), 1 }, 128);
    if (linear.bias) |bias| try e.dispatch("clef_audio_bias", BiasArgs{ .width = linear.output, .rows = rows, .mode = mode }, &.{ output.bind(1), bias.bind(2), (residual orelse output).bind(3), status.bind(4) }, .{ ceil(rows * linear.output, 256), 1, 1 }, 256);
}
fn normalize(e: b.Encoder, norm: Norm, input: View, output: View, rows: u32, status: View) !void {
    try e.dispatch("clef_audio_norm", NormArgs{}, &.{ input.bind(1), norm.weight.bind(2), norm.bias.bind(3), output.bind(4), status.bind(5) }, .{ rows, 1, 1 }, 32);
}
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}

const Scratch = struct {
    gpu: *c.flint_gpu,
    capacity: usize,
    chunk_capacity: usize,
    chunk_width: usize,
    buffers: std.ArrayList(b.Buffer) = .empty,
    input: View = undefined,
    column: View = undefined,
    conv_a: View = undefined,
    conv_b: View = undefined,
    projected: View = undefined,
    indices: View = undefined,
    residual: View = undefined,
    normalized: View = undefined,
    q: View = undefined,
    k: View = undefined,
    v: View = undefined,
    attention: View = undefined,
    ff: View = undefined,
    output: View = undefined,
    starts: View = undefined,
    ends: View = undefined,
    status: View = undefined,

    fn allocate(self: *Scratch, floats: usize) !View {
        const memory = c.flint_gpu_buffer(self.gpu, floats * 4) orelse return error.GpuAllocation;
        errdefer c.flint_gpu_buffer_free(memory);
        try self.buffers.append(a, memory);
        return .{ .buffer = memory };
    }
    fn init(gpu: *c.flint_gpu, tokens: usize, batch: usize, chunk_width: usize) !Scratch {
        var self = Scratch{ .gpu = gpu, .capacity = tokens, .chunk_capacity = batch, .chunk_width = chunk_width };
        errdefer self.deinit();
        const half = (chunk_width + 1) / 2;
        const quarter = (half + 1) / 2;
        const eighth = (quarter + 1) / 2;
        self.input = try self.allocate(batch * 128 * chunk_width);
        self.column = try self.allocate(batch * @max(@as(usize, 32) * quarter * channels * 9, eighth * 16 * channels));
        self.conv_a = try self.allocate(batch * 64 * half * channels);
        self.conv_b = try self.allocate(batch * 32 * quarter * channels);
        self.projected = try self.allocate(batch * eighth * width);
        self.indices = try self.allocate(batch * eighth);
        self.residual = try self.allocate(tokens * width);
        self.normalized = try self.allocate(tokens * width);
        self.q = try self.allocate(tokens * width);
        self.k = try self.allocate(tokens * width);
        self.v = try self.allocate(tokens * width);
        self.attention = try self.allocate(tokens * width);
        self.ff = try self.allocate(tokens * ff_width);
        self.output = try self.allocate(tokens * output_width);
        self.starts = try self.allocate(tokens);
        self.ends = try self.allocate(tokens);
        self.status = try self.allocate(1);
        return self;
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |memory| c.flint_gpu_buffer_free(memory);
        self.buffers.deinit(a);
    }
};
