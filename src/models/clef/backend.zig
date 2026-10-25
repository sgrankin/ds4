//! Quantized Clef backbone. Zig owns model weights, graph and request scratch;
//! Metal/ARC stays behind the existing small shared-device bridge.
const std = @import("std");
const c = @import("clef_c");
const shaders = @import("../flint/prefill_shaders.zig");
const Omni = @import("backend_omni.zig");
const GpuHead = @import("backend_head.zig");
const FullHead = @import("backend_head_full.zig");
const records = @import("record.zig");
const types = @import("types.zig");
const Vision = @import("vision.zig");
const AudioEncoder = @import("audio_encoder.zig");
const Audio = @import("audio.zig");
const Media = @import("backend_media.zig");
const Prefix = @import("prefix.zig");
pub const Injection = Media.Injection;
const a = std.heap.c_allocator;
pub const Buffer = *anyopaque;

pub const Dtype = enum { bf16, f32, u32 };
pub const Cancel = ?*const fn (?*anyopaque) callconv(.c) bool;
pub const Tensor = struct { bytes: []const u8, dtype: Dtype, shape: []const usize };
pub const View = struct {
    buffer: Buffer,
    offset: usize = 0,
    pub fn bind(self: View, index: u32) c.flint_gpu_binding {
        return .{ .handle = self.buffer, .offset = self.offset, .index = index };
    }
    pub fn at(self: View, offset: usize) View {
        return .{ .buffer = self.buffer, .offset = self.offset + offset };
    }
};
const Mapping = struct {
    bytes: []align(16384) const u8,
    fn open(path: [:0]const u8) !Mapping {
        const fd = c.open(path, c.O_RDONLY);
        if (fd < 0) return error.OpenCheckpoint;
        defer _ = c.close(fd);
        const size = c.lseek(fd, 0, c.SEEK_END);
        if (size < 8) return error.TruncatedCheckpoint;
        const p = c.mmap(null, @intCast(size), c.PROT_READ, c.MAP_PRIVATE, fd, 0);
        if (p == c.MAP_FAILED) return error.MapCheckpoint;
        return .{ .bytes = @as([*]align(16384) const u8, @ptrCast(@alignCast(p)))[0..@intCast(size)] };
    }
    fn close(self: Mapping) void {
        _ = c.munmap(@constCast(self.bytes.ptr), self.bytes.len);
    }
};

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const p = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(p);
    const f = c.fopen(p, "rb") orelse return error.OpenFile;
    defer _ = c.fclose(f);
    if (c.fseek(f, 0, c.SEEK_END) != 0) return error.ReadFile;
    const n = c.ftell(f);
    if (n < 0 or n > 64 * 1024 * 1024) return error.FileTooLarge;
    c.rewind(f);
    const out = try allocator.alloc(u8, @intCast(n));
    errdefer allocator.free(out);
    if (c.fread(out.ptr, 1, out.len, f) != out.len) return error.ReadFile;
    return out;
}

const Catalog = struct {
    arena: std.heap.ArenaAllocator,
    maps: std.ArrayList(Mapping) = .empty,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    fn deinit(self: *Catalog) void {
        for (self.maps.items) |m| m.close();
        self.arena.deinit();
    }
    fn addFile(self: *Catalog, directory: []const u8, file: []const u8, prefix: []const u8) !void {
        const allocator = self.arena.allocator();
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ directory, file }, 0);
        const map = try Mapping.open(path);
        self.maps.append(allocator, map) catch |err| {
            map.close();
            return err;
        };
        const size = std.mem.readInt(u64, map.bytes[0..8], .little);
        if (size > map.bytes.len - 8) return error.TruncatedCheckpoint;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, map.bytes[8..][0..size], .{});
        var it = parsed.value.object.iterator();
        const data = map.bytes[8 + size ..];
        while (it.next()) |item| {
            if (std.mem.eql(u8, item.key_ptr.*, "__metadata__")) continue;
            const value = item.value_ptr.object;
            const offsets = value.get("data_offsets").?.array.items;
            const start: usize = @intCast(offsets[0].integer);
            const end: usize = @intCast(offsets[1].integer);
            if (start > end or end > data.len) return error.TruncatedCheckpoint;
            const raw_shape = value.get("shape").?.array.items;
            const shape = try allocator.alloc(usize, raw_shape.len);
            for (shape, raw_shape) |*out, v| out.* = @intCast(v.integer);
            const dtype = value.get("dtype").?.string;
            const kind: Dtype = if (std.mem.eql(u8, dtype, "BF16")) .bf16 else if (std.mem.eql(u8, dtype, "F32")) .f32 else if (std.mem.eql(u8, dtype, "U32")) .u32 else continue;
            const key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, item.key_ptr.* });
            const inserted = try self.tensors.getOrPut(allocator, key);
            if (inserted.found_existing) return error.DuplicateTensor;
            inserted.value_ptr.* = .{ .bytes = data[start..end], .dtype = kind, .shape = shape };
        }
    }
    fn open(directory: []const u8) !Catalog {
        var self: Catalog = .{ .arena = std.heap.ArenaAllocator.init(a) };
        errdefer self.deinit();
        const allocator = self.arena.allocator();
        const path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors.index.json", .{directory});
        const bytes = try readFile(allocator, path);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        var files: std.StringHashMapUnmanaged(void) = .empty;
        var it = parsed.value.object.get("weight_map").?.object.iterator();
        while (it.next()) |entry| try files.put(allocator, entry.value_ptr.string, {});
        var names = files.keyIterator();
        while (names.next()) |name| try self.addFile(directory, name.*, "");
        try self.addFile(directory, "joint_head.safetensors", "head.");
        return self;
    }
    fn get(self: *const Catalog, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.debug.print("clef: missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
    }
};

pub const Family = enum { dense, omni };
pub const Config = struct {
    family: Family = .dense,
    H: u32,
    layers: u32,
    ffn: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    Hk: u32,
    Hv: u32,
    dk: u32,
    dv: u32,
    conv: u32,
    rotary: u32,
    vocab: u32,
    eps: f32,
    theta: f32,
    W: u32,
    head_ff: u32,
    head_heads: u32,
    routing_layers: u32,
    head_layers: u32,
    full: []bool,
    fn parse(catalog: *Catalog, directory: []const u8) !Config {
        const allocator = catalog.arena.allocator();
        const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{directory});
        const root = (try std.json.parseFromSlice(std.json.Value, allocator, try readFile(allocator, path), .{})).value.object;
        const hp = try std.fmt.allocPrint(allocator, "{s}/joint_head_config.json", .{directory});
        const h = (try std.json.parseFromSlice(std.json.Value, allocator, try readFile(allocator, hp), .{})).value.object;
        const family: Family = if (root.get("thinker_config") != null) .omni else .dense;
        const t = if (family == .omni)
            root.get("thinker_config").?.object.get("text_config").?.object
        else
            (root.get("text_config") orelse return error.UnsupportedArchitecture).object;
        const layers = integer(t, "num_hidden_layers");
        const full = try allocator.alloc(bool, layers);
        if (family == .dense) {
            const kinds = t.get("layer_types").?.array.items;
            if (kinds.len != layers) return error.UnsupportedArchitecture;
            for (full, kinds) |*out, k| out.* = std.mem.eql(u8, k.string, "full_attention");
        } else @memset(full, true);
        const dim = integer(t, "head_dim");
        const rope = if (family == .dense) t.get("rope_parameters").?.object else t;
        const rotary = if (family == .omni) dim else blk: {
            const factor = number(rope.get("partial_rotary_factor") orelse t.get("partial_rotary_factor").?);
            break :blk @as(u32, @intFromFloat(@as(f32, @floatFromInt(dim)) * factor));
        };
        if (family == .omni and (integer(t, "num_experts") != 128 or integer(t, "num_experts_per_tok") != 8))
            return error.UnsupportedExperts;
        return .{
            .family = family,
            .H = integer(t, "hidden_size"),
            .layers = layers,
            .ffn = integer(t, "intermediate_size"),
            .heads = integer(t, "num_attention_heads"),
            .kv_heads = integer(t, "num_key_value_heads"),
            .head_dim = dim,
            .Hk = if (family == .dense) integer(t, "linear_num_key_heads") else 0,
            .Hv = if (family == .dense) integer(t, "linear_num_value_heads") else 0,
            .dk = if (family == .dense) integer(t, "linear_key_head_dim") else 128,
            .dv = if (family == .dense) integer(t, "linear_value_head_dim") else 128,
            .conv = if (family == .dense) integer(t, "linear_conv_kernel_dim") else 4,
            .rotary = rotary,
            .vocab = integer(t, "vocab_size"),
            .eps = number(t.get("rms_norm_eps").?),
            .theta = number(rope.get("rope_theta").?),
            .W = integer(h, "width"),
            .head_ff = integer(h, "feedforward"),
            .head_heads = integer(h, "heads"),
            .routing_layers = integer(h, "routing_layers"),
            .head_layers = integer(h, "layers"),
            .full = full,
        };
    }
    fn C(self: Config) u32 {
        return 2 * self.Hk * self.dk + self.Hv * self.dv;
    }
    fn attentionWidth(self: Config) u32 {
        return 2 * self.heads * self.head_dim + 2 * self.kv_heads * self.head_dim;
    }
    fn linearWidth(self: Config) u32 {
        return self.C() + self.Hv * self.dv + 2 * self.Hv;
    }
};
fn integer(object: std.json.ObjectMap, key: []const u8) u32 {
    return @intCast(object.get(key).?.integer);
}
fn number(value: std.json.Value) f32 {
    return switch (value) {
        .float => @floatCast(value.float),
        .integer => @floatFromInt(value.integer),
        else => unreachable,
    };
}

pub const Matrix = struct { w: View, scales: View, biases: View, rows: u32, columns: u32, bits: u32 = 4, experts: u32 = 1 };
const Layer = struct {
    norm: View,
    ffn_norm: View,
    in: Matrix,
    out: Matrix,
    gate_up: Matrix,
    down: Matrix,
    qnorm: View,
    knorm: View,
    conv: View,
    decay: View,
    dt: View,
    gated_norm: View,
};
const Rows = extern struct { rows: u32, width: u32 };
const Matmul = extern struct { m: i32, n: i32, k: i32, pad: i32 = 0 };
const Norm = extern struct { width: u32, epsilon: f32 };
const Act = extern struct { f16: u32 = 0, limit: f32 = 65504 };
const AttentionPrep = extern struct { heads: u32, kv: u32, dim: u32, rotary: u32, row: u32, epsilon: f32 };
const Attention = extern struct { heads: u32, kv: u32, dim: u32, rows: u32, scale: f32 };
const Conv = extern struct { channels: u32, row: u32, kernel: u32 };
const GdnPrep = extern struct { hk: u32, hv: u32, dk: u32, dv: u32, row: u32, channels: u32 };
const Gdn = extern struct { hk: u32, hv: u32, dk: u32, dv: u32, channels: u32 };
const GdnOut = extern struct { hv: u32, dv: u32, row: u32, offset: u32, epsilon: f32 };
const DenseTile = enum { automatic, ordinary, wide, tall, rows_first, grouped, expanded };
const dense_tiles = std.StaticStringMap(DenseTile).initComptime(.{
    .{ "auto", .automatic },
    .{ "64x64", .ordinary },
    .{ "64x128", .wide },
    .{ "128x64", .tall },
    .{ "rows-first", .rows_first },
    .{ "grouped", .grouped },
    .{ "expanded", .expanded },
});

pub const Engine = struct {
    gpu: *c.flint_gpu,
    catalog: Catalog,
    config: Config,
    buffers: std.ArrayList(Buffer) = .empty,
    layer: []Layer,
    embedding: Matrix,
    final_norm: View,
    inv_freq: View,
    scratch: ?Scratch = null,
    omni: ?Omni.State = null,
    head_gpu: ?GpuHead.State = null,
    head_full: ?FullHead.State = null,
    vision: ?Vision.State = null,
    audio: ?AudioEncoder.State = null,
    /// Experimental cooperative GPU groups; zero preserves one prefill command.
    layer_quantum: u32 = 0,
    /// Diagnostic stage synchronization; ordinary execution stays one command.
    profile: bool = false,
    dense_tile: DenseTile = .automatic,
    expanded_matrix: ?View = null,
    ffn_bf16: bool = true,
    prefix: ?Prefix.State = null,

    pub fn open(directory: []const u8) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        var catalog = try Catalog.open(directory);
        var catalog_owned = true;
        errdefer if (catalog_owned) catalog.deinit();
        const config = try Config.parse(&catalog, directory);
        if (config.dk != 128 or config.dv != 128 or config.H % 64 != 0 or
            config.head_dim != @as(u32, if (config.family == .omni) 128 else 256)) return error.UnsupportedArchitecture;
        const gpu = c.flint_gpu_create() orelse return error.MetalUnavailable;
        errdefer c.flint_gpu_free(gpu);
        self.* = .{ .gpu = gpu, .catalog = catalog, .config = config, .layer = undefined, .embedding = undefined, .final_norm = undefined, .inv_freq = undefined };
        catalog_owned = false;
        errdefer self.catalog.deinit();
        errdefer {
            for (self.buffers.items) |b| c.flint_gpu_buffer_free(b);
            self.buffers.deinit(a);
        }
        if (c.getenv("DS4_CLEF_LAYER_QUANTUM")) |value| {
            self.layer_quantum = std.fmt.parseInt(u32, std.mem.span(value), 10) catch return error.InvalidLayerQuantum;
        }
        if (c.getenv("DS4_CLEF_PROFILE")) |value| self.profile = std.mem.eql(u8, std.mem.span(value), "1");
        if (c.getenv("DS4_CLEF_FFN_BF16")) |value| self.ffn_bf16 = std.mem.eql(u8, std.mem.span(value), "1");
        if (c.getenv("DS4_CLEF_QMM_TILE")) |value| {
            const tile = std.mem.span(value);
            self.dense_tile = dense_tiles.get(tile) orelse return error.InvalidDenseTile;
        }
        const quant_source = try shaders.source(a);
        defer a.free(quant_source);
        if (config.family == .omni) {
            const source = try std.mem.concat(a, u8, &.{ quant_source, @embedFile("../../metal/clef/omni_attention.metal"), @embedFile("../../metal/clef/omni_experts.metal") });
            defer a.free(source);
            try self.compile(source);
            try self.compile(@embedFile("../../metal/clef/omni_ops.metal"));
        } else if (self.dense_tile != .ordinary) {
            const extra = if (self.dense_tile == .expanded or self.dense_tile == .automatic) @embedFile("../../metal/clef/dequant_gemm.metal") else @embedFile("../../metal/clef/dense_tiles.metal");
            const source = try std.mem.concat(a, u8, &.{ quant_source, extra });
            defer a.free(source);
            try self.compile(source);
        } else try self.compile(quant_source);
        try self.compile(@embedFile("../../metal/clef/backbone.metal"));
        try self.compile(@embedFile("../../metal/clef/ops.metal"));
        self.embedding = try self.matrix(&.{"language_model.model.embed_tokens"});
        // MLX checkpoints have already folded Qwen3.5 RMSNorm 1+w.
        self.final_norm = try self.vector("language_model.model.norm.weight", 0, false);
        self.inv_freq = try self.buffer(config.rotary / 2 * 4);
        const freq = contents(f32, self.inv_freq, config.rotary / 2);
        for (freq, 0..) |*out, i| out.* = std.math.pow(f32, config.theta, -2 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(config.rotary)));
        self.layer = try a.alloc(Layer, if (config.family == .omni) 0 else config.layers);
        errdefer a.free(self.layer);
        if (config.family == .omni) {
            self.omni = try Omni.State.open(self);
            std.debug.print("clef: native Omni INT4, {d} layers,128 experts/top8\n", .{config.layers});
            return self;
        }
        for (self.layer, 0..) |*layer, i| {
            var key: [256]u8 = undefined;
            const base = try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.", .{i});
            const allocator = self.catalog.arena.allocator();
            const input_norm = try self.vector(try std.fmt.allocPrint(allocator, "{s}input_layernorm.weight", .{base}), 0, false);
            const ffn_norm = try self.vector(try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm.weight", .{base}), 0, false);
            const gate = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{base});
            const up = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{base});
            const down = try self.matrix(&.{try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{base})});
            const gate_up = try self.matrix(&.{ gate, up });
            const dummy = input_norm;
            layer.* = .{ .norm = input_norm, .ffn_norm = ffn_norm, .in = undefined, .out = undefined, .gate_up = gate_up, .down = down, .qnorm = dummy, .knorm = dummy, .conv = dummy, .decay = dummy, .dt = dummy, .gated_norm = dummy };
            if (config.full[i]) {
                const names = .{ "q_proj", "k_proj", "v_proj" };
                var paths: [3][]const u8 = undefined;
                inline for (names, 0..) |name, j| paths[j] = try std.fmt.allocPrint(allocator, "{s}self_attn.{s}", .{ base, name });
                layer.in = try self.matrix(&paths);
                layer.out = try self.matrix(&.{try std.fmt.allocPrint(allocator, "{s}self_attn.o_proj", .{base})});
                layer.qnorm = try self.vector(try std.fmt.allocPrint(allocator, "{s}self_attn.q_norm.weight", .{base}), 0, false);
                layer.knorm = try self.vector(try std.fmt.allocPrint(allocator, "{s}self_attn.k_norm.weight", .{base}), 0, false);
            } else {
                const names = .{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" };
                var paths: [4][]const u8 = undefined;
                inline for (names, 0..) |name, j| paths[j] = try std.fmt.allocPrint(allocator, "{s}linear_attn.{s}", .{ base, name });
                layer.in = try self.matrix(&paths);
                layer.out = try self.matrix(&.{try std.fmt.allocPrint(allocator, "{s}linear_attn.out_proj", .{base})});
                layer.conv = try self.vector(try std.fmt.allocPrint(allocator, "{s}linear_attn.conv1d.weight", .{base}), 0, false);
                layer.decay = try self.vector(try std.fmt.allocPrint(allocator, "{s}linear_attn.A_log", .{base}), 0, true);
                layer.dt = try self.vector(try std.fmt.allocPrint(allocator, "{s}linear_attn.dt_bias", .{base}), 0, false);
                layer.gated_norm = try self.vector(try std.fmt.allocPrint(allocator, "{s}linear_attn.norm.weight", .{base}), 0, false);
            }
        }
        if (c.getenv("DS4_CLEF_PREFIX_CACHE")) |value| {
            if (std.mem.eql(u8, std.mem.span(value), "1")) self.prefix = try Prefix.State.open(self);
        }
        std.debug.print("clef: native affine INT4, {d} layers, hidden {d}\n", .{ config.layers, config.H });
        return self;
    }
    pub fn close(self: *Engine) void {
        if (self.prefix) |*prefix| prefix.deinit();
        if (self.audio) |*audio| audio.deinit();
        if (self.vision) |*vision| vision.deinit();
        if (self.head_full) |*head| head.deinit();
        if (self.head_gpu) |*head| head.deinit();
        if (self.omni) |*owner| owner.deinit();
        if (self.scratch) |*s| s.deinit();
        for (self.buffers.items) |b| c.flint_gpu_buffer_free(b);
        self.buffers.deinit(a);
        a.free(self.layer);
        self.catalog.deinit();
        c.flint_gpu_free(self.gpu);
        a.destroy(self);
    }
    pub fn tensor(self: *const Engine, name: []const u8) !Tensor {
        if (self.config.family == .omni and std.mem.startsWith(u8, name, "language_model.")) {
            var key: [512]u8 = undefined;
            return self.catalog.get(try std.fmt.bufPrint(&key, "thinker.{s}", .{name}));
        }
        return self.catalog.get(name);
    }
    pub fn compile(self: *Engine, source: []const u8) !void {
        var message: [4096]u8 = @splat(0);
        if (!c.flint_gpu_compile(self.gpu, source.ptr, source.len, &message, message.len)) {
            std.debug.print("clef: shader compilation {s}\n", .{std.mem.sliceTo(&message, 0)});
            return error.CompileMetal;
        }
    }
    pub fn buffer(self: *Engine, bytes: usize) !View {
        const b = c.flint_gpu_buffer(self.gpu, bytes) orelse return error.GpuAllocation;
        self.buffers.append(a, b) catch |err| {
            c.flint_gpu_buffer_free(b);
            return err;
        };
        return .{ .buffer = b };
    }
    pub fn vector(self: *Engine, name: []const u8, plus: f32, negative_exp: bool) !View {
        const t = try self.tensor(name);
        const n = t.bytes.len / (if (t.dtype == .bf16) @as(usize, 2) else 4);
        const out = try self.buffer(n * 4);
        const dst = contents(f32, out, n);
        for (dst, 0..) |*value, i| {
            const x = scalar(t, i);
            value.* = if (negative_exp) -@exp(x) else x + plus;
        }
        return out;
    }
    /// Concatenate output rows, independently for every routed expert.
    pub fn matrix(self: *Engine, prefixes: []const []const u8) !Matrix {
        const allocator = self.catalog.arena.allocator();
        var n: usize = 0;
        var k: usize = 0;
        var experts: usize = 0;
        var bits: usize = 0;
        for (prefixes) |prefix| {
            const t = try self.tensor(try std.fmt.allocPrint(allocator, "{s}.scales", .{prefix}));
            const w = try self.tensor(try std.fmt.allocPrint(allocator, "{s}.weight", .{prefix}));
            if (t.shape.len != 2 and t.shape.len != 3) return error.UnsupportedMatrix;
            const e = if (t.shape.len == 3) t.shape[0] else 1;
            const columns = t.shape[t.shape.len - 1] * 64;
            const rows = t.shape[t.shape.len - 2];
            const precision = w.bytes.len * 8 / (e * rows * columns);
            if ((k != 0 and k != columns) or (experts != 0 and experts != e) or
                (bits != 0 and bits != precision)) return error.MatrixShape;
            if (precision != 4 and precision != 8) return error.UnsupportedQuantization;
            k = columns;
            experts = e;
            bits = precision;
            n += rows;
        }
        const padded = std.mem.alignForward(usize, n, 64);
        const w = try self.buffer(experts * padded * k * bits / 8);
        const scales = try self.buffer(experts * padded * k / 64 * 2);
        const biases = try self.buffer(experts * padded * k / 64 * 2);
        @memset(contents(u8, w, experts * padded * k * bits / 8), 0);
        @memset(contents(u8, scales, experts * padded * k / 64 * 2), 0);
        @memset(contents(u8, biases, experts * padded * k / 64 * 2), 0);
        var row: usize = 0;
        for (prefixes) |prefix| {
            const tw = try self.tensor(try std.fmt.allocPrint(allocator, "{s}.weight", .{prefix}));
            const ts = try self.tensor(try std.fmt.allocPrint(allocator, "{s}.scales", .{prefix}));
            const tb = try self.tensor(try std.fmt.allocPrint(allocator, "{s}.biases", .{prefix}));
            const rows = ts.shape[ts.shape.len - 2];
            const weights_size = rows * k * bits / 8;
            const group_size = rows * k / 64 * 2;
            if (tw.dtype != .u32 or ts.dtype != .bf16 or tb.dtype != .bf16 or
                tw.bytes.len != experts * weights_size or ts.bytes.len != experts * group_size or
                tb.bytes.len != ts.bytes.len) return error.UnsupportedQuantization;
            for (0..experts) |expert| {
                const destination = expert * padded + row;
                @memcpy(contents(u8, w.at(destination * k * bits / 8), weights_size), tw.bytes[expert * weights_size ..][0..weights_size]);
                @memcpy(contents(u8, scales.at(destination * k / 64 * 2), group_size), ts.bytes[expert * group_size ..][0..group_size]);
                @memcpy(contents(u8, biases.at(destination * k / 64 * 2), group_size), tb.bytes[expert * group_size ..][0..group_size]);
            }
            row += rows;
        }
        return .{ .w = w, .scales = scales, .biases = biases, .rows = @intCast(padded), .columns = @intCast(k), .bits = @intCast(bits), .experts = @intCast(experts) };
    }
    /// Load GPU head weights and pipelines without running a request. Serving
    /// can call this at startup to keep head compilation out of first latency.
    pub fn prepareGpuHead(self: *Engine) !void {
        try self.prepareMemoryHead();
        if (self.head_full == null) {
            const first = self.buffers.items.len;
            self.head_full = FullHead.State.open(self) catch |err| {
                for (self.buffers.items[first..]) |handle| c.flint_gpu_buffer_free(handle);
                self.buffers.items.len = first;
                return err;
            };
        }
    }
    pub fn decideGpu(self: *Engine, allocator: std.mem.Allocator, record: *const records.Record, hidden: []const f32, lexical: []const f32, cancel: Cancel, context: ?*anyopaque) !types.Result {
        if (cancel) |check| if (check(context)) return error.Cancelled;
        try self.prepareGpuHead();
        _ = try self.headFeaturesCancelable(hidden, cancel, context);
        return self.head_full.?.run(self, allocator, record, lexical, cancel, context);
    }
    /// Packed native Omni audio features; caller splits by clip token counts.
    pub fn audioFeatures(self: *Engine, clips: []const Audio.Features, cancel: Cancel, context: ?*anyopaque) ![]const f32 {
        if (cancel) |check| if (check(context)) return error.Cancelled;
        if (self.audio == null) {
            const first = self.buffers.items.len;
            self.audio = AudioEncoder.State.open(self) catch |err| {
                for (self.buffers.items[first..]) |handle| c.flint_gpu_buffer_free(handle);
                self.buffers.items.len = first;
                return err;
            };
        }
        return self.audio.?.run(self, clips, cancel, context);
    }
    /// Native vision tower output; borrowed until the next vision call.
    pub fn visionFeatures(self: *Engine, patches: []const f32, grid: [3]u32, cancel: Cancel, context: ?*anyopaque) !Vision.Features {
        if (cancel) |check| if (check(context)) return error.Cancelled;
        if (self.vision == null) {
            const first = self.buffers.items.len;
            self.vision = Vision.State.open(self) catch |err| {
                for (self.buffers.items[first..]) |handle| c.flint_gpu_buffer_free(handle);
                self.buffers.items.len = first;
                return err;
            };
        }
        return self.vision.?.run(self, patches, grid, cancel, context);
    }
    /// Borrowed GPU-prepared memory side of the schema head. CPU query/option
    /// work can continue after the GPU transaction ends.
    pub fn headFeatures(self: *Engine, hidden: []const f32) !types.HeadFeatures {
        return self.headFeaturesCancelable(hidden, null, null);
    }
    pub fn headFeaturesCancelable(self: *Engine, hidden: []const f32, cancel: Cancel, context: ?*anyopaque) !types.HeadFeatures {
        if (cancel) |check| if (check(context)) return error.Cancelled;
        try self.prepareMemoryHead();
        return self.head_gpu.?.run(self, hidden, cancel, context);
    }
    fn prepareMemoryHead(self: *Engine) !void {
        if (self.head_gpu == null) {
            const first = self.buffers.items.len;
            self.head_gpu = GpuHead.State.open(self) catch |err| {
                for (self.buffers.items[first..]) |handle| c.flint_gpu_buffer_free(handle);
                self.buffers.items.len = first;
                return err;
            };
        }
    }
    pub fn embeddingRows(self: *Engine, ids: []const u32) ![]f32 {
        const out = try a.alloc(f32, ids.len * self.config.H);
        errdefer a.free(out);
        const weight = try self.tensor("language_model.lm_head.weight");
        const scale = try self.tensor("language_model.lm_head.scales");
        const bias = try self.tensor("language_model.lm_head.biases");
        const H = self.config.H;
        const bits = weight.shape[1] * 32 / H;
        if (bits != 4 and bits != 8) return error.UnsupportedQuantization;
        const pack = 32 / bits;
        const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
        for (ids, 0..) |id, row| {
            if (id >= self.config.vocab) return error.TokenOutOfRange;
            for (0..H) |col| {
                const word = std.mem.readInt(u32, weight.bytes[(@as(usize, id) * weight.shape[1] + col / pack) * 4 ..][0..4], .little);
                const code = (word >> @intCast((col % pack) * bits)) & mask;
                const group = @as(usize, id) * (H / 64) + col / 64;
                out[row * H + col] = bf16(@as(f32, @floatFromInt(code)) * scalar(scale, group) + scalar(bias, group));
            }
        }
        return out;
    }
    /// Borrowed final trunk-normalized hidden rows, valid until the next infer.
    pub fn infer(self: *Engine, ids: []const u32) ![]const f32 {
        return self.inferCancelable(ids, null, null);
    }
    pub fn inferCancelable(self: *Engine, ids: []const u32, cancel: Cancel, context: ?*anyopaque) ![]const f32 {
        return self.inferMedia(ids, &.{}, &.{}, cancel, context);
    }
    pub fn inferText(self: *Engine, ids: []const u32, schema_start: usize, cancel: Cancel, context: ?*anyopaque) ![]const f32 {
        if (self.prefix) |*prefix| {
            prefix.last_reused = 0;
            if (!self.profile and Prefix.boundary(schema_start) != 0) {
                const hidden = try prefix.infer(self, ids, schema_start, cancel, context);
                if (c.getenv("DS4_CLEF_TIMING")) |value| {
                    if (std.mem.eql(u8, std.mem.span(value), "1")) std.debug.print("clef-cache: tokens={d} reused={d} retained={d} bytes={d}\n", .{ ids.len, prefix.last_reused, prefix.length, prefix.bytes(self.config) });
                }
                return hidden;
            }
        }
        return self.inferCancelable(ids, cancel, context);
    }
    pub fn scratchFor(self: *Engine, rows: u32) !*Scratch {
        if (self.scratch == null or self.scratch.?.capacity < rows) {
            if (self.scratch) |*scratch| scratch.deinit();
            self.scratch = null;
            self.scratch = try Scratch.init(self.gpu, self.config, rows);
        }
        return &self.scratch.?;
    }
    /// Positions are flattened [tokens,3], and each injection lists the token
    /// rows occupied by its vision/audio features. Empty media is ordinary text.
    pub fn inferMedia(self: *Engine, ids: []const u32, positions: []const u32, injections: []const Injection, cancel: Cancel, context: ?*anyopaque) ![]const f32 {
        if (positions.len != 0 and positions.len != ids.len * 3) return error.InvalidPositions;
        if (self.omni) |*owner| return owner.infer(self, ids, positions, injections, cancel, context);
        if (cancel) |check| if (check(context)) return error.Cancelled;
        if (ids.len == 0 or ids.len > 16384) return error.InvalidContextLength;
        for (ids) |id| if (id >= self.config.vocab) return error.TokenOutOfRange;
        const T: u32 = @intCast(ids.len);
        const s = try self.scratchFor(T);
        @memcpy(contents(u32, s.ids, T), ids);
        if (positions.len != 0) {
            @memcpy(contents(u32, s.position, T * 3), positions);
        } else {
            const pos = contents(i32, s.position, T * 3);
            for (0..T) |i| for (0..3) |j| {
                pos[i * 3 + j] = @intCast(i);
            };
        }
        var media = try Media.Prepared.init(self.gpu, injections, T, self.config.H);
        defer media.deinit();
        @memset(contents(u32, s.starts, T), 0);
        const bounds = contents(u32, s.bounds, 2);
        bounds[0] = 0;
        bounds[1] = T;
        contents(u32, s.length, 1)[0] = T;
        const cmd = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        var e = Encoder{ .gpu = self.gpu, .command = cmd };
        var pending = true;
        defer if (pending) {
            _ = c.flint_gpu_command_end(self.gpu, e.command, true);
        };
        var profile = Profile{ .enabled = self.profile };
        const cfg = self.config;
        try e.dispatch("clef_embed", extern struct { rows: u32, width: u32, bits: u32, pad: u32 }{ .rows = T, .width = cfg.H, .bits = self.embedding.bits, .pad = 0 }, &.{ s.ids.bind(1), self.embedding.w.bind(2), self.embedding.scales.bind(3), self.embedding.biases.bind(4), s.hidden.bind(5) }, .{ ceil(cfg.H, 256), T, 1 }, 256);
        try media.inject(e, s.hidden, cfg.H, null);
        try profile.mark(&e, &pending, .embedding, true);
        for (self.layer, 0..) |l, i| {
            if (cancel) |check| if (check(context)) return error.Cancelled;
            try self.norm(e, l.norm, s.hidden, s.normal, T);
            try profile.mark(&e, &pending, .input_norm, true);
            try self.linear(e, l.in, s.normal, s.project16, T);
            try e.dispatch("clef_widen", Rows{ .rows = T, .width = l.in.rows }, &.{ s.project16.bind(1), s.project.bind(2) }, .{ ceil(T * l.in.rows, 256), 1, 1 }, 256);
            try profile.mark(&e, &pending, .input_projection, true);
            if (cfg.full[i]) {
                try e.dispatch("attn_prep", AttentionPrep{ .heads = cfg.heads, .kv = cfg.kv_heads, .dim = cfg.head_dim, .rotary = cfg.rotary, .row = l.in.rows, .epsilon = cfg.eps }, &.{ s.project.bind(1), l.qnorm.bind(2), l.knorm.bind(3), s.position.bind(4), self.inv_freq.bind(5), s.query.bind(6), s.key.bind(7), s.value.bind(8), s.gate.bind(9), s.length.bind(10) }, .{ T, cfg.heads + cfg.kv_heads, 1 }, 32);
                // Kernels mask ragged tiles but require finite speculative tail loads.
                @memset(contents(u8, s.query.at(@as(usize, cfg.heads) * T * cfg.head_dim * 4), 8 * cfg.head_dim * 4), 0);
                @memset(contents(u8, s.key.at(@as(usize, cfg.kv_heads) * T * cfg.head_dim * 4), 64 * cfg.head_dim * 4), 0);
                @memset(contents(u8, s.value.at(@as(usize, cfg.kv_heads) * T * cfg.head_dim * 4), 64 * cfg.head_dim * 4), 0);
                const long = T >= 1024;
                const kernel: [:0]const u8 = if (long) "attention_prefetch_64" else "attention_fa";
                try e.dispatch(kernel, Attention{ .heads = cfg.heads, .kv = cfg.kv_heads, .dim = cfg.head_dim, .rows = T, .scale = 1 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))) }, &.{ s.query.bind(1), s.key.bind(2), s.value.bind(3), s.gate.bind(4), s.starts.bind(5), s.activation.bind(6), s.act.bind(7), s.overflow.bind(8) }, if (long) .{ ceil(T, 32), cfg.heads, 1 } else .{ ceil(T, 8), cfg.kv_heads, 1 }, if (long) 128 else 32 * (cfg.heads / cfg.kv_heads));
                try profile.mark(&e, &pending, .attention, true);
            } else {
                try e.dispatch("ssm_conv", Conv{ .channels = cfg.C(), .row = l.in.rows, .kernel = cfg.conv }, &.{ s.project.bind(1), l.conv.bind(2), s.starts.bind(3), s.convolution.bind(4) }, .{ ceil(cfg.C(), 256), T, 1 }, 256);
                try e.dispatch("gdn_prep", GdnPrep{ .hk = cfg.Hk, .hv = cfg.Hv, .dk = cfg.dk, .dv = cfg.dv, .row = l.in.rows, .channels = cfg.C() }, &.{ s.convolution.bind(1), s.project.bind(2), l.decay.bind(3), l.dt.bind(4), s.beta.bind(5), s.decay.bind(6) }, .{ T, 2 * cfg.Hk + cfg.Hv, 1 }, 32);
                const gp = Gdn{ .hk = cfg.Hk, .hv = cfg.Hv, .dk = cfg.dk, .dv = cfg.dv, .channels = cfg.C() };
                if (T >= 256) {
                    const binds = &.{ s.convolution.bind(1), s.beta.bind(2), s.decay.bind(3), s.output.bind(5), s.gdn_w.bind(6), s.gdn_u.bind(7), s.gdn_ke.bind(8), s.gdn_a.bind(9), s.gdn_e.bind(10), s.length.bind(11), s.dummy.bind(12), s.dummy.bind(13), s.dummy.bind(14) };
                    try e.dispatch("gdn_chunk_prep_32", gp, binds, .{ ceil(T, 32), cfg.Hv, 1 }, 128);
                    try e.dispatch("gdn_chunk_scan_32_32", gp, binds, .{ cfg.Hv, 4, 1 }, 128);
                } else try e.dispatch("gdn_scan_8", gp, &.{ s.convolution.bind(1), s.beta.bind(2), s.decay.bind(3), s.bounds.bind(4), s.output.bind(5), s.dummy.bind(6), s.dummy.bind(7), s.dummy.bind(8) }, .{ cfg.Hv, 32, 1 }, 32);
                try e.dispatch("gdn_out", GdnOut{ .hv = cfg.Hv, .dv = cfg.dv, .row = l.in.rows, .offset = cfg.C(), .epsilon = cfg.eps }, &.{ s.output.bind(1), s.project.bind(2), l.gated_norm.bind(3), s.activation.bind(4), s.act.bind(5), s.starts.bind(6), s.overflow.bind(7) }, .{ T, cfg.Hv, 1 }, 32);
                try profile.mark(&e, &pending, .gdn, true);
            }
            try self.linear(e, l.out, s.activation, s.project16, T);
            try profile.mark(&e, &pending, .output_projection, true);
            try e.dispatch("clef_residual", Rows{ .rows = T, .width = cfg.H }, &.{ s.hidden.bind(1), s.project16.bind(2) }, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
            try profile.mark(&e, &pending, .residual, true);
            try self.norm(e, l.ffn_norm, s.hidden, s.normal, T);
            try profile.mark(&e, &pending, .ffn_norm, true);
            try self.linear(e, l.gate_up, s.normal, s.project16, T);
            if (!self.ffn_bf16) try e.dispatch("clef_widen", Rows{ .rows = T, .width = l.gate_up.rows }, &.{ s.project16.bind(1), s.project.bind(2) }, .{ ceil(T * l.gate_up.rows, 256), 1, 1 }, 256);
            try profile.mark(&e, &pending, .ffn_gate_up, true);
            try e.dispatch(if (self.ffn_bf16) "clef_swiglu_bf16" else "swiglu", extern struct { width: u32 }{ .width = cfg.ffn }, &.{ (if (self.ffn_bf16) s.project16 else s.project).bind(1), s.activation.bind(2), s.act.bind(3), s.starts.bind(4), s.overflow.bind(5) }, .{ ceil(cfg.ffn, 256), T, 1 }, 256);
            try profile.mark(&e, &pending, .ffn_activation, true);
            try self.linear(e, l.down, s.activation, s.project16, T);
            try profile.mark(&e, &pending, .ffn_down, true);
            try e.dispatch("clef_residual", Rows{ .rows = T, .width = cfg.H }, &.{ s.hidden.bind(1), s.project16.bind(2) }, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
            try profile.mark(&e, &pending, .residual, true);
            if (!profile.enabled) try self.yieldAfterLayer(&e, i, self.layer.len, &pending, cancel, context);
        }
        try e.dispatch("rmsnorm_f32", Norm{ .width = cfg.H, .epsilon = cfg.eps }, &.{ s.hidden.bind(1), self.final_norm.bind(2), s.final.bind(3) }, .{ T, 1, 1 }, 256);
        try e.dispatch("clef_round", Rows{ .rows = T, .width = cfg.H }, &.{s.final.bind(1)}, .{ ceil(T * cfg.H, 256), 1, 1 }, 256);
        if (profile.enabled) {
            try profile.mark(&e, &pending, .final_norm, false);
            profile.report(T);
        } else {
            pending = false;
            if (!c.flint_gpu_command_end(self.gpu, e.command, true)) {
                std.debug.print("clef: GPU execution {s}\n", .{std.mem.span(c.flint_gpu_error(self.gpu))});
                return error.GpuExecution;
            }
        }
        if (cancel) |check| if (check(context)) return error.Cancelled;
        return contents(f32, s.final, @as(usize, T) * cfg.H);
    }

    /// Arithmetic and buffers are unchanged. A completed group gives another
    /// model a scheduling/cancellation point before encoding the next layers.
    pub fn yieldAfterLayer(self: *Engine, encoder: *Encoder, layer_index: usize, total: usize, pending: *bool, cancel: Cancel, context: ?*anyopaque) !void {
        if (self.layer_quantum == 0 or layer_index + 1 == total or (layer_index + 1) % self.layer_quantum != 0) return;
        pending.* = false;
        if (!c.flint_gpu_command_end(self.gpu, encoder.command, true)) return error.GpuExecution;
        if (cancel) |check| if (check(context)) return error.Cancelled;
        encoder.command = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        pending.* = true;
    }
    pub fn norm(self: *Engine, e: Encoder, weight: View, input: View, output: View, T: u32) !void {
        const s = &self.scratch.?;
        try e.dispatch("rmsnorm_act", Norm{ .width = self.config.H, .epsilon = self.config.eps }, &.{ input.bind(1), weight.bind(2), output.bind(3), s.act.bind(4), s.starts.bind(5), s.overflow.bind(6) }, .{ T, 1, 1 }, 256);
    }
    pub fn linear(self: *Engine, e: Encoder, m: Matrix, input: View, output: View, T: u32) !void {
        const p = Matmul{ .m = @intCast(T), .n = @intCast(m.rows), .k = @intCast(m.columns) };
        const narrow = T <= 8 and m.bits == 4;
        // Expanding once pays off for long inputs; short requests avoid both
        // that fixed cost and the extra scratch allocation.
        const tile: DenseTile = if (self.dense_tile == .automatic) (if (T >= 2048) .expanded else .ordinary) else self.dense_tile;
        if (m.bits == 4 and T >= 256 and tile != .ordinary) {
            if (tile == .expanded) {
                if (self.expanded_matrix == null) self.expanded_matrix = try self.buffer(@as(usize, self.config.ffn) * self.config.H * 4);
                const weight = self.expanded_matrix.?;
                if (c.flint_gpu_buffer_size(weight.buffer) < @as(usize, m.rows) * m.columns * 2) return error.MatrixShape;
                try e.dispatch("clef_dequant_q4", p, &.{ m.w.bind(1), m.scales.bind(2), m.biases.bind(3), weight.bind(4) }, .{ ceil(m.rows * m.columns / 8, 256), 1, 1 }, 256);
                try e.dispatch("clef_dequant_gemm_64x128", p, &.{ input.bind(1), weight.bind(2), output.bind(3) }, .{ ceil(m.rows, 128), ceil(T, 64), 1 }, 128);
                return;
            }
            const wide = tile == .wide;
            const rows_first = tile == .rows_first;
            const kernel: [:0]const u8 = if (tile == .grouped) "clef_qmm4_grouped_rows" else if (rows_first) "clef_qmm4_rows_first" else if (wide) "clef_qmm4_64x128" else "clef_qmm4_128x64";
            const grid: [3]u32 = if (rows_first) .{ ceil(T, 64), ceil(m.rows, 64), 1 } else .{ ceil(m.rows, if (wide) 128 else 64), ceil(T, if (tile == .tall) 128 else 64), 1 };
            try e.dispatch(kernel, p, &.{ m.w.bind(1), m.scales.bind(2), m.biases.bind(3), input.bind(4), output.bind(5) }, grid, 128);
            return;
        }
        const kernel: [:0]const u8 = if (m.bits == 8) "flint_prefill_qmm8" else if (narrow) "flint_prefill_qmv4_packed" else "flint_prefill_qmm4";
        try e.dispatch(kernel, p, &.{ m.w.bind(1), m.scales.bind(2), m.biases.bind(3), input.bind(4), output.bind(5) }, if (narrow) .{ ceil(m.rows, 16), T, 1 } else .{ ceil(m.rows, 64), ceil(T, 64), 1 }, 128);
    }
};

pub fn scalar(t: Tensor, index: usize) f32 {
    return switch (t.dtype) {
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[index * 2 ..][0..2], .little)) << 16),
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[index * 4 ..][0..4], .little)),
        else => unreachable,
    };
}
fn bf16(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @bitCast(bits & 0xffff0000);
}
fn ceil(n: u32, d: u32) u32 {
    return (n + d - 1) / d;
}
pub fn contents(comptime T: type, view: View, n: usize) []T {
    const ptr = @as([*]u8, @ptrCast(c.flint_gpu_contents(view.buffer))) + view.offset;
    return @as([*]T, @ptrCast(@alignCast(ptr)))[0..n];
}
pub const Encoder = struct {
    gpu: *c.flint_gpu,
    command: Buffer,
    pub fn dispatch(self: Encoder, name: [:0]const u8, params: anytype, binds: []const c.flint_gpu_binding, grid: [3]u32, threads: u32) !void {
        const block = [3]u32{ threads, 1, 1 };
        if (!c.flint_gpu_dispatch(self.gpu, self.command, name, &params, @sizeOf(@TypeOf(params)), binds.ptr, binds.len, &grid, &block)) return error.GpuDispatch;
    }
};

/// This intentionally changes submission boundaries. GPU stage totals locate
/// expensive arithmetic; use an unprofiled run for whole-request latency.
const Profile = struct {
    const Stage = enum { embedding, input_norm, input_projection, attention, gdn, output_projection, residual, ffn_norm, ffn_gate_up, ffn_activation, ffn_down, final_norm };
    const count = @typeInfo(Stage).@"enum".field_names.len;
    enabled: bool,
    seconds: [count]f64 = @splat(0),
    calls: [count]usize = @splat(0),

    fn mark(self: *Profile, encoder: *Encoder, pending: *bool, stage: Stage, restart: bool) !void {
        if (!self.enabled) return;
        pending.* = false;
        var seconds: f64 = 0;
        if (!c.flint_gpu_command_end_profile(encoder.gpu, encoder.command, &seconds)) return error.GpuExecution;
        self.seconds[@backingInt(stage)] += seconds;
        self.calls[@backingInt(stage)] += 1;
        if (restart) {
            encoder.command = c.flint_gpu_command_begin(encoder.gpu) orelse return error.GpuCommand;
            pending.* = true;
        }
    }

    fn report(self: *const Profile, tokens: u32) void {
        inline for (@typeInfo(Stage).@"enum".field_names, 0..) |name, index| {
            if (self.calls[index] != 0) std.debug.print("clef-profile: tokens={d} stage={s} calls={d} gpu_ms={d:.3}\n", .{ tokens, name, self.calls[index], self.seconds[index] * 1000 });
        }
    }
};

pub const Scratch = struct {
    gpu: *c.flint_gpu,
    capacity: u32,
    buffers: std.ArrayList(Buffer) = .empty,
    ids: View = undefined,
    position: View = undefined,
    starts: View = undefined,
    bounds: View = undefined,
    length: View = undefined,
    hidden: View = undefined,
    normal: View = undefined,
    project: View = undefined,
    project16: View = undefined,
    query: View = undefined,
    key: View = undefined,
    value: View = undefined,
    gate: View = undefined,
    convolution: View = undefined,
    beta: View = undefined,
    decay: View = undefined,
    output: View = undefined,
    activation: View = undefined,
    final: View = undefined,
    act: View = undefined,
    overflow: View = undefined,
    dummy: View = undefined,
    gdn_w: View = undefined,
    gdn_u: View = undefined,
    gdn_ke: View = undefined,
    gdn_a: View = undefined,
    gdn_e: View = undefined,
    fn alloc(self: *Scratch, bytes: usize) !View {
        const b = c.flint_gpu_buffer(self.gpu, bytes) orelse return error.GpuAllocation;
        self.buffers.append(a, b) catch |err| {
            c.flint_gpu_buffer_free(b);
            return err;
        };
        return .{ .buffer = b };
    }
    fn deinit(self: *Scratch) void {
        for (self.buffers.items) |b| c.flint_gpu_buffer_free(b);
        self.buffers.deinit(a);
    }
    fn init(gpu: *c.flint_gpu, cfg: Config, T: u32) !Scratch {
        var s = Scratch{ .gpu = gpu, .capacity = T };
        errdefer s.deinit();
        const rows: usize = T;
        const projected = @max(2 * cfg.ffn, @max(cfg.linearWidth() + 64, cfg.attentionWidth()));
        s.ids = try s.alloc(rows * 4);
        s.position = try s.alloc(rows * 3 * 4);
        s.starts = try s.alloc(rows * 4);
        s.bounds = try s.alloc(8);
        s.length = try s.alloc(4);
        s.hidden = try s.alloc(rows * cfg.H * 4);
        s.normal = try s.alloc(rows * cfg.H * 2);
        s.project = try s.alloc(rows * projected * 4);
        s.project16 = try s.alloc(rows * projected * 2);
        s.query = try s.alloc((@as(usize, cfg.heads) * rows + 8) * cfg.head_dim * 4);
        s.key = try s.alloc((@as(usize, cfg.kv_heads) * rows + 64) * cfg.head_dim * 4);
        s.value = try s.alloc((@as(usize, cfg.kv_heads) * rows + 64) * cfg.head_dim * 4);
        s.gate = try s.alloc(rows * cfg.heads * cfg.head_dim * 4);
        s.convolution = try s.alloc(rows * cfg.C() * 4);
        s.beta = try s.alloc(rows * cfg.Hv * 4);
        s.decay = try s.alloc(rows * cfg.Hv * 4);
        s.output = try s.alloc(rows * cfg.Hv * cfg.dv * 4);
        s.activation = try s.alloc(rows * @max(cfg.ffn, @max(cfg.Hv * cfg.dv, cfg.heads * cfg.head_dim)) * 2);
        s.final = try s.alloc(rows * cfg.H * 4);
        s.act = try s.alloc(@sizeOf(Act));
        contents(Act, s.act, 1)[0] = .{};
        s.overflow = try s.alloc(rows * 4);
        @memset(contents(u32, s.overflow, rows), 0);
        s.dummy = try s.alloc(16);
        @memset(contents(u8, s.dummy, 16), 0);
        const chunks: usize = ceil(T, 32);
        const state_rows = @as(usize, cfg.Hv) * chunks * 32;
        s.gdn_w = try s.alloc(state_rows * 128 * 4);
        s.gdn_u = try s.alloc(state_rows * 128 * 4);
        s.gdn_ke = try s.alloc(state_rows * 128 * 4);
        s.gdn_a = try s.alloc(state_rows * 32 * 4);
        s.gdn_e = try s.alloc(state_rows * 4);
        return s;
    }
};
