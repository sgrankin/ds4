//! Native Qwen vision tower shared by dense Clef and Omni. CPU preprocessing
//! supplies HF patch rows; this owner produces merged embeddings and DeepStack.
//! Graph and precision follow Cloudflare; 72-wide attention follows Cleffa (MIT).
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const a = std.heap.c_allocator;
const View = b.View;
const width: u32 = 1152;
const heads: u32 = 16;
const head_dim: u32 = 72;
const ff_width: u32 = 4304;
const merged_width: u32 = 4 * width;
const patch_width: u32 = 3 * 2 * 16 * 16;
const layer_count = 27;

pub const Features = struct {
    features: []const f32,
    deepstack: []const []const f32,
    grid: [3]u32,
};
const Norm = struct { weight: View, bias: View };
const Linear = struct { weight: View, bias: View, input: u32, output: u32 };
const Layer = struct { norm1: Norm, norm2: Norm, qkv: Linear, projection: Linear, up: Linear, down: Linear };
const Merger = struct { norm: Norm, first: Linear, second: Linear, grouped_norm: bool };
const GemmArgs = extern struct { rows: u32, columns: u32, inner: u32 };
const NormArgs = extern struct { width: u32, epsilon: f32 = 1e-6 };
const RowsArgs = extern struct { width: u32, rows: u32, mode: u32 = 0 };
const AttentionArgs = extern struct {
    patches: u32,
    width: u32 = width,
    heads: u32 = heads,
    head_dim: u32 = head_dim,
    grid_width: u32,
    merge: u32 = 2,
    epsilon: f32 = 1e-6,
    scale: f32 = 1 / @sqrt(@as(f32, head_dim)),
};

pub const State = struct {
    patch: Linear,
    position: View,
    position_side: u32,
    rotary: View,
    layers: []Layer,
    merger: Merger,
    deepstack: []Merger,
    output_width: u32,
    scratch: ?Scratch = null,

    pub fn open(engine: *b.Engine) !State {
        try engine.compile(@embedFile("../../metal/clef/vision.metal"));
        const prefix = if (engine.config.family == .omni) "thinker.vision_tower" else "vision_tower";
        const layers = try a.alloc(Layer, layer_count);
        errdefer a.free(layers);
        const deepstack = try a.alloc(Merger, if (engine.config.family == .omni) 3 else 0);
        errdefer a.free(deepstack);
        var storage: [256]u8 = undefined;
        const patch_name = try std.fmt.bufPrint(&storage, "{s}.patch_embed.proj.weight", .{prefix});
        const tensor = try engine.tensor(patch_name);
        if (tensor.dtype != .bf16 or tensor.shape.len != 5 or
            tensor.shape[0] != width or tensor.shape[1] != 2 or tensor.shape[2] != 16 or tensor.shape[3] != 16 or tensor.shape[4] != 3)
            return error.InvalidVisionPatch;
        const patch = try engine.buffer(tensor.bytes.len);
        // MLX Conv3d is [out,t,y,x,channel]; HF patch rows are [channel,t,y,x].
        // This is a byte-exact weight permutation performed once at load.
        const destination = b.contents(u16, patch, tensor.bytes.len / 2);
        for (0..width) |out| for (0..3) |channel| for (0..2) |time| for (0..16) |y| for (0..16) |x| {
            const src = out * patch_width + ((time * 16 + y) * 16 + x) * 3 + channel;
            const dst = out * patch_width + ((channel * 2 + time) * 16 + y) * 16 + x;
            destination[dst] = std.mem.readInt(u16, tensor.bytes[src * 2 ..][0..2], .little);
        };
        const patch_bias = try engine.vector(try std.fmt.bufPrint(&storage, "{s}.patch_embed.proj.bias", .{prefix}), 0, false);
        const position_tensor = try engine.tensor(try std.fmt.bufPrint(&storage, "{s}.pos_embed.weight", .{prefix}));
        if (position_tensor.dtype != .bf16 or position_tensor.shape.len != 2 or position_tensor.shape[1] != width) return error.InvalidVisionPosition;
        const side: u32 = @intFromFloat(@sqrt(@as(f64, @floatFromInt(position_tensor.shape[0]))));
        if (@as(usize, side) * side != position_tensor.shape[0]) return error.InvalidVisionPosition;
        const position = try engine.buffer(position_tensor.bytes.len);
        @memcpy(b.contents(u8, position, position_tensor.bytes.len), position_tensor.bytes);
        const rotary = try engine.buffer((head_dim / 4) * 4);
        for (b.contents(f32, rotary, head_dim / 4), 0..) |*v, i|
            v.* = @floatCast(std.math.pow(f64, 10000, -@as(f64, @floatFromInt(i)) / @as(f64, head_dim / 4)));
        for (layers, 0..) |*layer, i| {
            const base = try std.fmt.allocPrint(a, "{s}.blocks.{d}", .{ prefix, i });
            defer a.free(base);
            layer.* = .{
                .norm1 = try loadNorm(engine, try std.fmt.bufPrint(&storage, "{s}.norm1", .{base}), width),
                .norm2 = try loadNorm(engine, try std.fmt.bufPrint(&storage, "{s}.norm2", .{base}), width),
                .qkv = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.attn.qkv", .{base}), 3 * width, width),
                .projection = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.attn.proj", .{base}), width, width),
                .up = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.mlp.linear_fc1", .{base}), ff_width, width),
                .down = try loadLinear(engine, try std.fmt.bufPrint(&storage, "{s}.mlp.linear_fc2", .{base}), width, ff_width),
            };
        }
        const merger = try loadMerger(engine, try std.fmt.bufPrint(&storage, "{s}.merger", .{prefix}), false);
        for (deepstack, 0..) |*item, i| item.* = try loadMerger(engine, try std.fmt.bufPrint(&storage, "{s}.deepstack_merger_list.{d}", .{ prefix, i }), true);
        return .{ .patch = .{ .weight = patch, .bias = patch_bias, .input = patch_width, .output = width }, .position = position, .position_side = side, .rotary = rotary, .layers = layers, .merger = merger, .deepstack = deepstack, .output_width = engine.config.H };
    }
    pub fn deinit(self: *State) void {
        if (self.scratch) |*scratch| scratch.deinit();
        a.free(self.layers);
        a.free(self.deepstack);
    }
    pub fn run(self: *State, engine: *b.Engine, patches: []const f32, grid: [3]u32, cancel: b.Cancel, context: ?*anyopaque) !Features {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0 or grid[1] % 2 != 0 or grid[2] % 2 != 0) return error.InvalidVisionGrid;
        const per_frame = grid[1] * grid[2];
        const count = grid[0] * per_frame;
        if (patches.len != @as(usize, count) * patch_width) return error.InvalidVisionPatches;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        if (self.scratch == null or self.scratch.?.capacity < count or self.scratch.?.frame_capacity < per_frame) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(engine.gpu, count, per_frame, self.output_width, self.deepstack.len);
        }
        const s = &self.scratch.?;
        @memcpy(b.contents(f32, s.input, patches.len), patches);
        positions(self.position_side, grid, b.contents(u32, s.position_ids, @as(usize, count) * 4), b.contents(f32, s.position_weights, @as(usize, count) * 4));
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, command, true);
        };
        const e = b.Encoder{ .gpu = engine.gpu, .command = command };
        try project(e, self.patch, s.input, s.residual, count);
        try e.dispatch("clef_vision_embed", RowsArgs{ .width = width, .rows = count }, &.{ s.residual.bind(1), self.patch.bias.bind(2), self.position.bind(3), s.position_ids.bind(4), s.position_weights.bind(5) }, .{ ceil(count * width, 256), 1, 1 }, 256);
        for (self.layers, 0..) |layer, i| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try normalize(e, layer.norm1, s.residual, s.normalized, count, width);
            try project(e, layer.qkv, s.normalized, s.branch, count);
            for (0..grid[0]) |frame| {
                const args = AttentionArgs{ .patches = per_frame, .grid_width = grid[2] };
                try e.dispatch("clef_vision_rope", args, &.{ s.branch.at(frame * per_frame * 3 * width * 4).bind(1), layer.qkv.bias.bind(2), self.rotary.bind(3), s.q.bind(4), s.k.bind(5), s.v.bind(6) }, .{ 1, heads, per_frame }, head_dim / 2);
                try e.dispatch("clef_vision_attention", args, &.{ s.q.bind(1), s.k.bind(2), s.v.bind(3), s.attention.at(frame * per_frame * width * 4).bind(4) }, .{ ceil(per_frame, 32), heads, 1 }, 128);
            }
            try project(e, layer.projection, s.attention, s.branch, count);
            try residual(e, layer.projection.bias, s.branch, s.residual, count, width);
            try normalize(e, layer.norm2, s.residual, s.normalized, count, width);
            try project(e, layer.up, s.normalized, s.ff, count);
            try activate(e, layer.up.bias, s.ff, count, ff_width, 0);
            try project(e, layer.down, s.ff, s.branch, count);
            try residual(e, layer.down.bias, s.branch, s.residual, count, width);
            for (self.deepstack, 0..) |merger, j| if (i == (j + 1) * 8) try self.merge(e, merger, s.deepstack[j], count);
        }
        try self.merge(e, self.merger, s.output, count);
        pending = false;
        if (!c.flint_gpu_command_end(engine.gpu, command, true)) return error.GpuExecution;
        for (s.deepstack, s.deepstack_views) |view, *out| out.* = b.contents(f32, view, @as(usize, count / 4) * self.output_width);
        return .{ .features = b.contents(f32, s.output, @as(usize, count / 4) * self.output_width), .deepstack = s.deepstack_views, .grid = grid };
    }
    fn merge(self: *State, e: b.Encoder, merger: Merger, output: View, count: u32) !void {
        const s = &self.scratch.?;
        try normalize(e, merger.norm, s.residual, s.normalized, if (merger.grouped_norm) count / 4 else count, if (merger.grouped_norm) merged_width else width);
        try project(e, merger.first, s.normalized, s.ff, count / 4);
        try activate(e, merger.first.bias, s.ff, count / 4, merged_width, 1);
        try project(e, merger.second, s.ff, output, count / 4);
        try activate(e, merger.second.bias, output, count / 4, self.output_width, 2);
    }
};

fn loadNorm(engine: *b.Engine, prefix: []const u8, n: usize) !Norm {
    var name: [256]u8 = undefined;
    const weight_name = try std.fmt.bufPrint(&name, "{s}.weight", .{prefix});
    const weight = try engine.tensor(weight_name);
    if (weight.bytes.len != n * 2 and weight.bytes.len != n * 4) return error.InvalidVisionNorm;
    const w = try engine.vector(weight_name, 0, false);
    const bias = try engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false);
    return .{ .weight = w, .bias = bias };
}
fn loadLinear(engine: *b.Engine, prefix: []const u8, out: u32, input: u32) !Linear {
    var name: [256]u8 = undefined;
    const tensor = try engine.tensor(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}));
    if (tensor.dtype != .bf16 or tensor.bytes.len != @as(usize, out) * input * 2) return error.InvalidVisionWeight;
    const weight = try engine.buffer(tensor.bytes.len);
    @memcpy(b.contents(u8, weight, tensor.bytes.len), tensor.bytes);
    const bias = try engine.vector(try std.fmt.bufPrint(&name, "{s}.bias", .{prefix}), 0, false);
    return .{ .weight = weight, .bias = bias, .input = input, .output = out };
}
fn loadMerger(engine: *b.Engine, prefix: []const u8, grouped: bool) !Merger {
    var name: [256]u8 = undefined;
    return .{
        .norm = try loadNorm(engine, try std.fmt.bufPrint(&name, "{s}.norm", .{prefix}), if (grouped) merged_width else width),
        .first = try loadLinear(engine, try std.fmt.bufPrint(&name, "{s}.linear_fc1", .{prefix}), merged_width, merged_width),
        .second = try loadLinear(engine, try std.fmt.bufPrint(&name, "{s}.linear_fc2", .{prefix}), engine.config.H, merged_width),
        .grouped_norm = grouped,
    };
}
fn project(e: b.Encoder, linear: Linear, input: View, output: View, rows: u32) !void {
    try e.dispatch("clef_vision_gemm", GemmArgs{ .rows = rows, .columns = linear.output, .inner = linear.input }, &.{ input.bind(1), linear.weight.bind(2), output.bind(3) }, .{ ceil(linear.output, 128), ceil(rows, 16), 1 }, 128);
}
fn normalize(e: b.Encoder, norm: Norm, input: View, output: View, rows: u32, n: u32) !void {
    try e.dispatch("clef_vision_norm", NormArgs{ .width = n }, &.{ input.bind(1), norm.weight.bind(2), norm.bias.bind(3), output.bind(4) }, .{ rows, 1, 1 }, 256);
}
fn residual(e: b.Encoder, bias: View, branch: View, output: View, rows: u32, n: u32) !void {
    try e.dispatch("clef_vision_residual", RowsArgs{ .width = n, .rows = rows }, &.{ branch.bind(1), bias.bind(2), output.bind(3) }, .{ ceil(rows * n, 256), 1, 1 }, 256);
}
fn activate(e: b.Encoder, bias: View, input: View, rows: u32, n: u32, mode: u32) !void {
    try e.dispatch("clef_vision_activate", RowsArgs{ .width = n, .rows = rows, .mode = mode }, &.{ input.bind(1), bias.bind(2) }, .{ ceil(rows * n, 256), 1, 1 }, 256);
}
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}

/// align_corners learned-position interpolation, in HF's merge-window order.
fn positions(side: u32, grid: [3]u32, indices: []u32, weights: []f32) void {
    var row: usize = 0;
    for (0..grid[0]) |_| for (0..grid[1] / 2) |by| for (0..grid[2] / 2) |bx| for (0..2) |dy| for (0..2) |dx| {
        const y = by * 2 + dy;
        const x = bx * 2 + dx;
        const fy = if (grid[1] > 1) @as(f64, @floatFromInt(y)) * (side - 1) / @as(f64, @floatFromInt(grid[1] - 1)) else 0;
        const fx = if (grid[2] > 1) @as(f64, @floatFromInt(x)) * (side - 1) / @as(f64, @floatFromInt(grid[2] - 1)) else 0;
        const y0: u32 = @intFromFloat(fy);
        const x0: u32 = @intFromFloat(fx);
        const y1 = @min(y0 + 1, side - 1);
        const x1 = @min(x0 + 1, side - 1);
        const wy = fy - @as(f64, @floatFromInt(y0));
        const wx = fx - @as(f64, @floatFromInt(x0));
        @memcpy(indices[row * 4 ..][0..4], &[_]u32{ y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1 });
        weights[row * 4] = @floatCast((1 - wy) * (1 - wx));
        weights[row * 4 + 1] = @floatCast((1 - wy) * wx);
        weights[row * 4 + 2] = @floatCast(wy * (1 - wx));
        weights[row * 4 + 3] = @floatCast(wy * wx);
        row += 1;
    };
}
const Scratch = struct {
    gpu: *c.flint_gpu,
    capacity: u32,
    frame_capacity: u32,
    buffers: std.ArrayList(b.Buffer) = .empty,
    input: View = undefined,
    residual: View = undefined,
    normalized: View = undefined,
    branch: View = undefined,
    attention: View = undefined,
    ff: View = undefined,
    q: View = undefined,
    k: View = undefined,
    v: View = undefined,
    position_ids: View = undefined,
    position_weights: View = undefined,
    output: View = undefined,
    deepstack: []View,
    deepstack_views: [][]const f32,
    fn alloc(self: *Scratch, bytes: usize) !View {
        const memory = c.flint_gpu_buffer(self.gpu, bytes) orelse return error.GpuAllocation;
        self.buffers.append(a, memory) catch |err| {
            c.flint_gpu_buffer_free(memory);
            return err;
        };
        return .{ .buffer = memory };
    }
    fn init(gpu: *c.flint_gpu, count: u32, frame_count: u32, out: u32, deep_count: usize) !Scratch {
        const deep = try a.alloc(View, deep_count);
        errdefer a.free(deep);
        const views = try a.alloc([]const f32, deep_count);
        errdefer a.free(views);
        var self = Scratch{ .gpu = gpu, .capacity = count, .frame_capacity = frame_count, .deepstack = deep, .deepstack_views = views };
        errdefer {
            for (self.buffers.items) |memory| c.flint_gpu_buffer_free(memory);
            self.buffers.deinit(a);
        }
        const n: usize = count;
        self.input = try self.alloc(n * patch_width * 4);
        self.residual = try self.alloc(n * width * 4);
        self.normalized = try self.alloc(n * width * 4);
        self.branch = try self.alloc(n * 3 * width * 4);
        self.attention = try self.alloc(n * width * 4);
        self.ff = try self.alloc(n * @max(ff_width, merged_width) * 4);
        const qkv_size = (@as(usize, frame_count) * width + 32 * head_dim) * 4;
        self.q = try self.alloc(qkv_size);
        self.k = try self.alloc(qkv_size);
        self.v = try self.alloc(qkv_size);
        self.position_ids = try self.alloc(n * 4 * 4);
        self.position_weights = try self.alloc(n * 4 * 4);
        self.output = try self.alloc(n / 4 * out * 4);
        for (deep) |*view| view.* = try self.alloc(n / 4 * out * 4);
        return self;
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |memory| c.flint_gpu_buffer_free(memory);
        self.buffers.deinit(a);
        a.free(self.deepstack);
        a.free(self.deepstack_views);
    }
};

test "vision positions preserve merged-window corners" {
    var indices: [16]u32 = undefined;
    var weights: [16]f32 = undefined;
    positions(48, .{ 1, 2, 2 }, &indices, &weights);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expectEqual(@as(u32, 47), indices[4]);
    try std.testing.expectEqual(@as(u32, 2256), indices[8]);
    try std.testing.expectEqual(@as(u32, 2303), indices[12]);
    try std.testing.expectEqual(@as(f32, 1), weights[0]);
}
