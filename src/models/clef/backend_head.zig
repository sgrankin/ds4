//! Wide schema-head projections on Metal, small question/option head on CPU.
const std = @import("std");
const b = @import("backend.zig");
const c = @import("clef_c");
const types = @import("types.zig");
const a = std.heap.c_allocator;
const View = b.View;
const Norm = struct { weight: View, bias: View };
const Module = struct { norm: ?Norm, weight: View };
const Gemm = extern struct { rows: u32, columns: u32, inner: u32 };
const NormArgs = extern struct { width: u32, epsilon: f32 = 1e-5 };

pub const State = struct {
    hidden_norm: Norm,
    memory_projection: View,
    modules: []Module,
    scratch: ?Scratch = null,

    pub fn open(engine: *b.Engine) !State {
        try engine.compile(@embedFile("../../metal/clef/head.metal"));
        for ([_][:0]const u8{ "clef_head_norm", "clef_head_gemm" }) |kernel| {
            if (!c.flint_gpu_prepare(engine.gpu, kernel)) return error.GpuPipeline;
        }
        const cfg = engine.config;
        const modules = try a.alloc(Module, cfg.routing_layers + cfg.head_layers);
        errdefer a.free(modules);
        const hidden_norm = try loadNorm(engine, "head.hidden_norm");
        const memory = try matrix(engine, "head.memory_projection.weight", 0, cfg.W, cfg.H);
        for (modules, 0..) |*module, i| {
            var storage: [192]u8 = undefined;
            if (i < cfg.routing_layers) {
                const prefix = try std.fmt.bufPrint(&storage, "head.evidence_layers.{d}.memory_norm", .{i});
                module.norm = try loadNorm(engine, prefix);
                const name = try std.fmt.bufPrint(&storage, "head.evidence_layers.{d}.attention.in_proj_weight", .{i});
                module.weight = try matrix(engine, name, cfg.W, 2 * cfg.W, cfg.W);
            } else {
                module.norm = null;
                const name = try std.fmt.bufPrint(&storage, "head.layers.{d}.multihead_attn.in_proj_weight", .{i - cfg.routing_layers});
                module.weight = try matrix(engine, name, cfg.W, 2 * cfg.W, cfg.W);
            }
        }
        return .{ .hidden_norm = hidden_norm, .memory_projection = memory, .modules = modules };
    }
    pub fn deinit(self: *State) void {
        if (self.scratch) |*scratch| scratch.deinit();
        a.free(self.modules);
    }
    pub fn run(self: *State, engine: *b.Engine, hidden: []const f32, cancel: b.Cancel, context: ?*anyopaque) !types.HeadFeatures {
        const cfg = engine.config;
        if (hidden.len == 0 or hidden.len % cfg.H != 0) return error.InvalidHeadInput;
        const T: u32 = @intCast(hidden.len / cfg.H);
        if (cancel) |check| if (check(context)) return error.Cancelled;
        if (self.scratch == null or self.scratch.?.capacity < T) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(engine.gpu, cfg, T, self.modules.len);
        }
        const s = &self.scratch.?;
        @memcpy(b.contents(f32, s.input, hidden.len), hidden);
        const command = c.flint_gpu_command_begin(engine.gpu) orelse return error.GpuCommand;
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(engine.gpu, command, true);
        };
        const e = b.Encoder{ .gpu = engine.gpu, .command = command };
        try normalize(e, self.hidden_norm, s.input, s.normalized_hidden, T, cfg.H);
        try project(e, self.memory_projection, s.normalized_hidden, s.memory, T, cfg.W, cfg.H);
        for (self.modules, s.kv) |module, output| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            const input = if (module.norm) |norm| blk: {
                try normalize(e, norm, s.memory, s.normalized_memory, T, cfg.W);
                break :blk s.normalized_memory;
            } else s.memory;
            try project(e, module.weight, input, output, T, 2 * cfg.W, cfg.W);
        }
        pending = false;
        if (!c.flint_gpu_command_end(engine.gpu, command, true)) return error.GpuExecution;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        for (s.kv, s.kv_views) |view, *out| out.* = b.contents(f32, view, @as(usize, T) * 2 * cfg.W);
        return .{
            .normalized_hidden = b.contents(f32, s.normalized_hidden, @as(usize, T) * cfg.H),
            .memory = b.contents(f32, s.memory, @as(usize, T) * cfg.W),
            .kv = s.kv_views,
        };
    }
};
fn matrix(engine: *b.Engine, name: []const u8, skip: usize, rows: usize, columns: usize) !View {
    const tensor = try engine.tensor(name);
    if (tensor.dtype != .bf16 or tensor.bytes.len < (skip + rows) * columns * 2) return error.InvalidHeadShape;
    const bytes = tensor.bytes[skip * columns * 2 ..][0 .. rows * columns * 2];
    const out = try engine.buffer(bytes.len);
    @memcpy(b.contents(u8, out, bytes.len), bytes);
    return out;
}
fn loadNorm(engine: *b.Engine, prefix: []const u8) !Norm {
    var storage: [192]u8 = undefined;
    const weight = try engine.vector(try std.fmt.bufPrint(&storage, "{s}.weight", .{prefix}), 0, false);
    const bias = try engine.vector(try std.fmt.bufPrint(&storage, "{s}.bias", .{prefix}), 0, false);
    return .{ .weight = weight, .bias = bias };
}
fn normalize(e: b.Encoder, norm: Norm, input: View, output: View, T: u32, H: u32) !void {
    try e.dispatch("clef_head_norm", NormArgs{ .width = H }, &.{ input.bind(1), norm.weight.bind(2), norm.bias.bind(3), output.bind(4) }, .{ T, 1, 1 }, 256);
}
fn project(e: b.Encoder, weight: View, input: View, output: View, T: u32, N: u32, K: u32) !void {
    try e.dispatch("clef_head_gemm", Gemm{ .rows = T, .columns = N, .inner = K }, &.{ input.bind(1), weight.bind(2), output.bind(3) }, .{ (N + 127) / 128, (T + 15) / 16, 1 }, 128);
}
const Scratch = struct {
    gpu: *c.flint_gpu,
    capacity: u32,
    buffers: std.ArrayList(b.Buffer) = .empty,
    input: View = undefined,
    normalized_hidden: View = undefined,
    memory: View = undefined,
    normalized_memory: View = undefined,
    kv: []View = undefined,
    kv_views: [][]const f32 = undefined,
    fn alloc(self: *Scratch, size: usize) !View {
        const buffer = c.flint_gpu_buffer(self.gpu, size) orelse return error.GpuAllocation;
        self.buffers.append(a, buffer) catch |err| {
            c.flint_gpu_buffer_free(buffer);
            return err;
        };
        return .{ .buffer = buffer };
    }
    fn init(gpu: *c.flint_gpu, cfg: b.Config, T: u32, count: usize) !Scratch {
        const kv = try a.alloc(View, count);
        errdefer a.free(kv);
        const views = try a.alloc([]const f32, count);
        errdefer a.free(views);
        var s = Scratch{ .gpu = gpu, .capacity = T, .kv = kv, .kv_views = views };
        errdefer {
            for (s.buffers.items) |buffer| c.flint_gpu_buffer_free(buffer);
            s.buffers.deinit(a);
        }
        const rows: usize = T;
        s.input = try s.alloc(rows * cfg.H * 4);
        s.normalized_hidden = try s.alloc(rows * cfg.H * 4);
        s.memory = try s.alloc(rows * cfg.W * 4);
        s.normalized_memory = try s.alloc(rows * cfg.W * 4);
        for (kv) |*view| view.* = try s.alloc(rows * 2 * cfg.W * 4);
        return s;
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |buffer| c.flint_gpu_buffer_free(buffer);
        self.buffers.deinit(a);
        a.free(self.kv);
        a.free(self.kv_views);
    }
};
