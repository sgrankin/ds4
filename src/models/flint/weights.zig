//! The installed affine checkpoint is already packed. Map its safetensors,
//! validate the architecture, and copy into Flint's 13 aligned weight banks.
//! Bank metadata follows recursive-oss/rec e53663b262af (Apache-2.0).
const std = @import("std");
const c = @import("flint_c");
const a = std.heap.c_allocator;

const Tensor = struct { bytes: []const u8, dtype: []const u8, shape: []const std.json.Value };
const Mapping = struct {
    bytes: []align(16384) const u8,
    fn open(path: [:0]const u8) !Mapping {
        const fd = c.open(path, c.O_RDONLY);
        if (fd < 0) return error.OpenCheckpoint;
        defer _ = c.close(fd);
        const size = c.lseek(fd, 0, c.SEEK_END);
        if (size < 8) return error.TruncatedCheckpoint;
        const ptr = c.mmap(null, @intCast(size), c.PROT_READ, c.MAP_PRIVATE, fd, 0);
        if (ptr == c.MAP_FAILED) return error.MapCheckpoint;
        return .{ .bytes = @as([*]align(16384) const u8, @ptrCast(@alignCast(ptr)))[0..@intCast(size)] };
    }
    fn close(self: Mapping) void {
        _ = c.munmap(@constCast(self.bytes.ptr), self.bytes.len);
    }
};

pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const terminated = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(terminated);
    const file = c.fopen(terminated, "rb") orelse return error.OpenFile;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadFile;
    const length = c.ftell(file);
    if (length < 0 or length > 64 * 1024 * 1024) return error.FileTooLarge;
    c.rewind(file);
    const bytes = try allocator.alloc(u8, @intCast(length));
    errdefer allocator.free(bytes);
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadFile;
    return bytes;
}

const Catalog = struct {
    arena: std.heap.ArenaAllocator,
    maps: std.ArrayList(Mapping) = .empty,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,

    fn deinit(self: *Catalog) void {
        for (self.maps.items) |mapping| mapping.close();
        self.arena.deinit();
    }
    fn open(directory: []const u8) !Catalog {
        var self: Catalog = .{ .arena = std.heap.ArenaAllocator.init(a) };
        errdefer self.deinit();
        const allocator = self.arena.allocator();
        const manifest_path = try std.fmt.allocPrint(allocator, "{s}/flint-int4.json", .{directory});
        const manifest = try std.json.parseFromSlice(std.json.Value, allocator, try readFile(allocator, manifest_path), .{});
        const files = manifest.value.object.get("checkpoint_files") orelse return error.MissingManifest;
        var names = files.object.iterator();
        while (names.next()) |name| {
            if (!std.mem.endsWith(u8, name.key_ptr.*, ".safetensors")) continue;
            const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ directory, name.key_ptr.* }, 0);
            const mapping = try Mapping.open(path);
            self.maps.append(allocator, mapping) catch |err| {
                mapping.close();
                return err;
            };
            const header_size = std.mem.readInt(u64, mapping.bytes[0..8], .little);
            if (header_size > mapping.bytes.len - 8) return error.TruncatedCheckpoint;
            const header = try std.json.parseFromSlice(std.json.Value, allocator, mapping.bytes[8..][0..header_size], .{});
            var entries = header.value.object.iterator();
            while (entries.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
                const value = entry.value_ptr.object;
                const offsets = (value.get("data_offsets") orelse return error.InvalidTensor).array.items;
                const start: usize = @intCast(offsets[0].integer);
                const end: usize = @intCast(offsets[1].integer);
                const data = mapping.bytes[8 + header_size ..];
                if (start > end or end > data.len) return error.TruncatedCheckpoint;
                const inserted = try self.tensors.getOrPut(allocator, entry.key_ptr.*);
                if (inserted.found_existing) return error.DuplicateTensor;
                inserted.value_ptr.* = .{
                    .bytes = data[start..end],
                    .dtype = (value.get("dtype") orelse return error.InvalidTensor).string,
                    .shape = (value.get("shape") orelse return error.InvalidTensor).array.items,
                };
            }
        }
        return self;
    }
    fn get(self: *const Catalog, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.debug.print("flint: missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
    }
};

const Copy = struct { offset: usize, source: []const u8, widen: bool = false, zero: usize = 0 };
const Bank = struct {
    catalog: *Catalog,
    copies: std.ArrayList(Copy) = .empty,
    size: usize = 0,
    fn add(self: *Bank, name: []const u8, dtype: []const u8, widen: bool) !u32 {
        const tensor = try self.catalog.get(name);
        if (!std.mem.eql(u8, tensor.dtype, dtype)) return error.WrongTensorType;
        self.size = std.mem.alignForward(usize, self.size, 256);
        const offset = self.size;
        try self.copies.append(a, .{ .offset = offset, .source = tensor.bytes, .widen = widen });
        self.size += tensor.bytes.len * @as(usize, if (widen) 2 else 1);
        return @intCast(offset);
    }
    fn zeros(self: *Bank, bytes: usize) !u32 {
        self.size = std.mem.alignForward(usize, self.size, 256);
        const offset = self.size;
        try self.copies.append(a, .{ .offset = offset, .source = "", .zero = bytes });
        self.size += bytes;
        return @intCast(offset);
    }
    fn linear(self: *Bank, row: *[64]u32, base: usize, prefix: []const u8) !void {
        var key: [256]u8 = undefined;
        row[base] = try self.add(try std.fmt.bufPrint(&key, "{s}.weight", .{prefix}), "U8", false);
        row[base + 1] = try self.add(try std.fmt.bufPrint(&key, "{s}.scales", .{prefix}), "BF16", false);
        row[base + 2] = try self.add(try std.fmt.bufPrint(&key, "{s}.biases", .{prefix}), "BF16", false);
        const scale = try self.catalog.get(try std.fmt.bufPrint(&key, "{s}.scales", .{prefix}));
        if (scale.shape.len < 2) return error.WrongTensorShape;
        row[base + 3] = @intCast(scale.shape[scale.shape.len - 2].integer);
        row[base + 4] = @intCast(scale.shape[scale.shape.len - 1].integer * 64);
    }
    fn materialize(self: *Bank, gpu: *c.flint_gpu) !*anyopaque {
        const handle = c.flint_gpu_buffer(gpu, self.size) orelse return error.GpuAllocation;
        const bytes = @as([*]u8, @ptrCast(c.flint_gpu_contents(handle)))[0..self.size];
        @memset(bytes, 0);
        for (self.copies.items) |copy| {
            if (copy.widen) {
                const src: []align(1) const u16 = std.mem.bytesAsSlice(u16, copy.source);
                const dst: []align(1) u32 = std.mem.bytesAsSlice(u32, bytes[copy.offset..][0 .. src.len * 4]);
                for (src, dst) |bits, *out| out.* = @as(u32, bits) << 16;
            } else if (copy.source.len > 0) {
                @memcpy(bytes[copy.offset..][0..copy.source.len], copy.source);
            }
        }
        return handle;
    }
};

pub const Weights = struct {
    gpu: *c.flint_gpu,
    banks: [13]*anyopaque,
    metadata: [50][64]u32,
    pub fn deinit(self: *Weights) void {
        for (self.banks) |bank| c.flint_gpu_buffer_free(bank);
        c.flint_gpu_free(self.gpu);
        self.* = undefined;
    }
};

pub fn load(directory: []const u8) !Weights {
    var catalog = try Catalog.open(directory);
    defer catalog.deinit();
    const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{directory});
    defer a.free(config_path);
    const bytes = try readFile(a, config_path);
    defer a.free(bytes);
    const config = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer config.deinit();
    const expected = .{ .hidden_size = 2048, .num_hidden_layers = 49, .num_attention_heads = 32, .num_key_value_heads = 4, .head_dim = 128, .num_experts = 256, .num_experts_per_tok = 6, .sliding_window = 513, .vocab_size = 150144 };
    inline for (@typeInfo(@TypeOf(expected)).@"struct".field_names) |field| {
        const value = config.value.object.get(field) orelse return error.UnsupportedArchitecture;
        if (value != .integer or value.integer != @field(expected, field)) return error.UnsupportedArchitecture;
    }
    const export_config = config.value.object.get("mlx_export") orelse return error.UnsupportedQuantization;
    if (export_config.object.get("bits").?.integer != 4 or export_config.object.get("embedding_bits").?.integer != 8 or export_config.object.get("routed_expert_bits").?.integer != 4) return error.UnsupportedQuantization;
    const kinds = config.value.object.get("layer_types").?.array.items;
    if (kinds.len != 49) return error.UnsupportedArchitecture;
    const gpu = c.flint_gpu_create() orelse return error.MetalUnavailable;
    var result: Weights = .{ .gpu = gpu, .banks = undefined, .metadata = @splat(@splat(0)) };
    var completed: usize = 0;
    errdefer {
        for (result.banks[0..completed]) |bank| c.flint_gpu_buffer_free(bank);
        c.flint_gpu_free(gpu);
    }
    for (0..13) |bank_id| {
        var bank: Bank = .{ .catalog = &catalog };
        defer bank.copies.deinit(a);
        for (bank_id * 4..@min(bank_id * 4 + 4, 49)) |layer| {
            const row = &result.metadata[layer];
            row[0] = @intCast(bank_id);
            row[1] = @intFromBool(std.mem.eql(u8, kinds[layer].string, "full_attention"));
            row[2] = @intFromBool(layer >= 2);
            var key: [256]u8 = undefined;
            for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.softmax_offset" }, 3..) |name, slot| {
                row[slot] = try bank.add(try std.fmt.bufPrint(&key, "model.layers.{d}.{s}", .{ layer, name }), "BF16", false);
            }
            if (layer >= 2) {
                const name = try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.gate.weight", .{layer});
                row[8] = try bank.add(name, "BF16", true);
                row[9] = try bank.zeros(256 * 4);
                row[60] = try bank.add(name, "BF16", false);
            }
            for ([_][]const u8{ "q", "k", "v", "o" }, 0..) |name, index| {
                try bank.linear(row, 10 + index * 5, try std.fmt.bufPrint(&key, "model.layers.{d}.self_attn.{s}_proj", .{ layer, name }));
            }
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |name, index| {
                const family = if (layer < 2) "mlp" else "mlp.switch_mlp";
                try bank.linear(row, 30 + index * 5, try std.fmt.bufPrint(&key, "model.layers.{d}.{s}.{s}_proj", .{ layer, family, name }));
                if (layer >= 2) try bank.linear(row, 45 + index * 5, try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.shared_expert.{s}_proj", .{ layer, name }));
            }
        }
        if (bank_id == 12) {
            const row = &result.metadata[49];
            row[0] = 12;
            row[3] = try bank.add("model.norm.weight", "BF16", false);
            try bank.linear(row, 10, "model.embed_tokens");
            try bank.linear(row, 30, "lm_head");
        }
        result.banks[bank_id] = try bank.materialize(gpu);
        completed += 1;
        std.debug.print("flint: packed layers {d}..{d}, {d} bytes\n", .{ bank_id * 4, @min(bank_id * 4 + 4, 49), bank.size });
    }
    return result;
}
