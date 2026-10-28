//! Capture selected prefix frontiers inside an ordinary layer-major sweep.
//! Only rolling state needs private tensors. Compressed histories are append
//! only, so the existing canonical codec can read their prefix after the sweep.
//! The collector never installs an earlier frontier into the live session.
const std = @import("std");
const c = @import("gguf_c");
const Executor = @import("executor.zig").Executor;
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const Suffix = @import("suffix.zig").Suffix;
const Config = @import("../config.zig").ModelConfig;
const Graph = c.ds41_gpu_graph;
const Tensor = c.ds4_gpu_tensor;
const Archive = c.ds41_prerope_archive;
const allocator = std.heap.c_allocator;
const raw_rows = 128;
const tape_bytes = 128;

const Endpoint = struct {
    position: u32,
    window: [40]?*Tensor = @splat(null),
    archive: Archive = std.mem.zeroes(Archive),
    previous_kv: [3]?*Tensor = @splat(null),
    previous_score: [3]?*Tensor = @splat(null),
    residual: ?*Tensor = null,
    pre: ?*Tensor = null,
    layers: u64 = 0,
    pools: u8 = 0,
    final_row: bool = false,
    complete: bool = false,

    fn deinit(self: *Endpoint) void {
        for (self.window) |tensor| c.ds4_gpu_tensor_free(tensor);
        for (self.archive.raw) |tensor| c.ds4_gpu_tensor_free(tensor);
        for (self.previous_kv) |tensor| c.ds4_gpu_tensor_free(tensor);
        for (self.previous_score) |tensor| c.ds4_gpu_tensor_free(tensor);
        c.ds4_gpu_tensor_free(self.residual);
        c.ds4_gpu_tensor_free(self.pre);
    }
};

fn copy(destination: ?*Tensor, to: u64, source: ?*const Tensor, from: u64, bytes: u64) bool {
    return bytes == 0 or c.ds4_gpu_tensor_copy(destination, to, source, from, bytes) != 0;
}

/// Copy chronological source rows into their physical slots in a raw ring.
fn ringCopy(destination: ?*Tensor, source: ?*const Tensor, start: u32, count: u32, end: u32, row_bytes: u64) bool {
    if (end <= start or end > start + count) return true;
    const first = @max(start, end -| raw_rows);
    const rows = end - first;
    const slot = first % raw_rows;
    const part = @min(rows, raw_rows - slot);
    return copy(destination, @as(u64, slot) * row_bytes, source, @as(u64, first - start) * row_bytes, @as(u64, part) * row_bytes) and
        copy(destination, 0, source, @as(u64, first - start + part) * row_bytes, @as(u64, rows - part) * row_bytes);
}

pub const Collector = struct {
    endpoints: []Endpoint,
    outputs: []c.ds4_suffix_cache,
    tokens: []const c_int,
    config: *const Config,
    has_archive: bool,

    pub fn init(graph: *Graph, config: *const Config, tokens: []const c_int, positions: []const c_int, outputs: []c.ds4_suffix_cache) ?Collector {
        const endpoints = allocator.alloc(Endpoint, positions.len) catch return null;
        var self = Collector{ .endpoints = endpoints, .outputs = outputs, .tokens = tokens, .config = config, .has_archive = graph.prerope != null };
        for (endpoints, positions) |*endpoint, position| endpoint.* = .{ .position = @intCast(position) };
        var good = false;
        defer if (!good) self.deinit();
        const shape = &config.shape;
        for (endpoints) |*endpoint| {
            for (&endpoint.window) |*tensor| tensor.* = c.ds4_gpu_tensor_alloc(raw_rows * shape.n_head_dim * 4) orelse return null;
            if (self.has_archive) {
                endpoint.archive = graph.prerope.*;
                endpoint.archive.raw = @splat(null);
                for (&endpoint.archive.raw) |*tensor| tensor.* = c.ds4_gpu_tensor_alloc(raw_rows * tape_bytes) orelse return null;
            }
            if (endpoint.position & 1 != 0) {
                for (&endpoint.previous_kv) |*tensor| tensor.* = c.ds4_gpu_tensor_alloc(shape.n_head_dim * 4) orelse return null;
                for (&endpoint.previous_score) |*tensor| tensor.* = c.ds4_gpu_tensor_alloc(shape.n_head_dim * 4) orelse return null;
            }
            endpoint.residual = c.ds4_gpu_tensor_alloc(shape.n_hc * shape.n_embd * 4) orelse return null;
            endpoint.pre = c.ds4_gpu_tensor_alloc(shape.n_hc * 4) orelse return null;
        }
        // A short append may retain keys from the starting ring. Seed those
        // before any layer can overwrite the live graph's slots.
        if (c.ds4_gpu_begin_commands() == 0) return null;
        var ok = true;
        for (endpoints) |*endpoint| {
            for (endpoint.window, graph.window) |destination, source| ok = ok and copy(destination, 0, source, 0, raw_rows * shape.n_head_dim * 4);
            if (self.has_archive) for (endpoint.archive.raw, graph.prerope.*.raw) |destination, source| {
                ok = ok and copy(destination, 0, source, 0, raw_rows * tape_bytes);
            };
        }
        if (c.ds4_gpu_end_commands() == 0) ok = false;
        if (!ok) return null;
        good = true;
        return self;
    }

    pub fn deinit(self: *Collector) void {
        for (self.endpoints) |*endpoint| endpoint.deinit();
        allocator.free(self.endpoints);
    }

    /// Normalized BF16 KV is available before rotary/FP8 cache rounding.
    pub fn originals(self: *Collector, layer: u32, start: u32, count: u32, kv: ?*Tensor) bool {
        if (!self.has_archive) return true;
        const width = self.config.shape.n_head_dim;
        for (self.endpoints) |*endpoint| {
            const end = @min(endpoint.position, start + count);
            if (endpoint.complete or end <= start) continue;
            var position = @max(start, end -| raw_rows);
            while (position < end) {
                const slot = position % raw_rows;
                const rows = @min(end - position, raw_rows - slot);
                const view = c.ds4_gpu_tensor_view(kv, @as(u64, position - start) * width * 4, @as(u64, rows) * width * 4) orelse return false;
                const ok = c.ds4_gpu_dsv41_prerope_copy(endpoint.archive.raw[layer], slot, view, width, rows, true) != 0;
                c.ds4_gpu_tensor_free(view);
                if (!ok) return false;
                for (0..rows) |row| endpoint.archive.raw_epoch[layer][slot + row] = position + @as(u32, @intCast(row));
                position += rows;
            }
        }
        return true;
    }

    /// Actual post-RoPE keys and the unfinished paired-compression inputs.
    pub fn attention(self: *Collector, layer: u32, start: u32, count: u32, batch: *const c.ds41_prefill_row) bool {
        const width = self.config.shape.n_head_dim;
        const policy = self.config.layerPolicy(layer).?;
        const owner = @import("attention.zig").cacheOwner(layer);
        for (self.endpoints) |*endpoint| {
            const end = @min(endpoint.position, start + count);
            if (endpoint.complete or end <= start) continue;
            if (!ringCopy(endpoint.window[layer], batch.kv, start, count, end, @as(u64, width) * 4)) return false;
            if (end == endpoint.position) endpoint.layers |= @as(u64, 1) << @intCast(layer);
            if (end == endpoint.position and policy.publishes_kv and policy.compression_ratio == 2 and end & 1 != 0) {
                const offset = @as(u64, end - start - 1) * width * 4;
                if (!copy(endpoint.previous_kv[owner], 0, batch.pool_kv, offset, @as(u64, width) * 4) or
                    !copy(endpoint.previous_score[owner], 0, batch.pool_score, offset, @as(u64, width) * 4)) return false;
                endpoint.pools |= @as(u8, 1) << @intCast(owner);
            }
        }
        return true;
    }

    pub fn finalRows(self: *Collector, start: u32, count: u32, rows: [*c]const c.ds41_prefill_row, split_pre: bool) bool {
        const shape = &self.config.shape;
        for (self.endpoints) |*endpoint| {
            if (endpoint.complete or endpoint.position <= start or endpoint.position > start + count) continue;
            const row = &rows[endpoint.position - start - 1];
            if (!copy(endpoint.residual, 0, row.residual, 0, @as(u64, shape.n_hc) * shape.n_embd * 4) or
                !copy(endpoint.pre, 0, if (split_pre) row.ffn_split else row.pre, 0, @as(u64, shape.n_hc) * 4)) return false;
            endpoint.final_row = true;
        }
        return true;
    }

    /// Token-major remainders already finish all layers at each token.
    pub fn scalar(self: *Collector, graph: *Graph) bool {
        const shape = &self.config.shape;
        for (self.endpoints) |*endpoint| {
            if (endpoint.complete or endpoint.position != graph.pos) continue;
            for (endpoint.window, graph.window) |destination, source| if (!copy(destination, 0, source, 0, raw_rows * shape.n_head_dim * 4)) return false;
            if (self.has_archive) {
                endpoint.archive.raw_epoch = graph.prerope.*.raw_epoch;
                for (endpoint.archive.raw, graph.prerope.*.raw) |destination, source| if (!copy(destination, 0, source, 0, raw_rows * tape_bytes)) return false;
            }
            if (endpoint.position & 1 != 0) for (0..3) |owner| {
                if (!copy(endpoint.previous_kv[owner], 0, graph.previous_kv[owner], 0, shape.n_head_dim * 4) or
                    !copy(endpoint.previous_score[owner], 0, graph.previous_score[owner], 0, shape.n_head_dim * 4)) return false;
            };
            if (!copy(endpoint.residual, 0, graph.residual, 0, @as(u64, shape.n_hc) * shape.n_embd * 4) or
                !copy(endpoint.pre, 0, graph.pre, 0, @as(u64, shape.n_hc) * 4)) return false;
            endpoint.layers = (@as(u64, 1) << @intCast(shape.n_layer)) - 1;
            endpoint.pools = 7;
            endpoint.final_row = true;
        }
        return true;
    }

    /// All layers now agree. Serialize a borrowed view, leaving live residuals,
    /// windows, positions and frontier untouched. The caller computes the live
    /// output head afterwards, replacing the scratch logits used here.
    pub fn finish(self: *Collector, executor: Executor, errors: []u8) bool {
        const shape = &self.config.shape;
        const logits = allocator.alloc(f32, shape.n_vocab) catch return false;
        defer allocator.free(logits);
        for (self.endpoints, self.outputs) |*endpoint, *output| {
            if (endpoint.complete or endpoint.position > executor.graph.pos) continue;
            if (endpoint.layers != (@as(u64, 1) << @intCast(shape.n_layer)) - 1 or !endpoint.final_row or
                (endpoint.position & 1 != 0 and endpoint.pools != 7)) return false;
            var view = executor.graph.*;
            view.pos = endpoint.position;
            view.window = endpoint.window;
            view.residual = endpoint.residual;
            view.pre = endpoint.pre;
            for (0..3) |owner| if (endpoint.position & 1 != 0) {
                view.previous_kv[owner] = endpoint.previous_kv[owner];
                view.previous_score[owner] = endpoint.previous_score[owner];
            };
            if (self.has_archive) view.prerope = &endpoint.archive;
            var local = executor;
            local.graph = &view;
            if (!local.logits(logits.ptr)) return false;
            var state = std.mem.zeroes(c.ds4_session_frontier);
            state.checkpoint = .{ .v = @constCast(self.tokens.ptr), .len = @intCast(endpoint.position), .cap = @intCast(endpoint.position) };
            state.checkpoint_valid = true;
            state.logits = logits.ptr;
            const codec = Checkpoint{ .frontier = &state, .graph = &view, .config = self.config, .ready = true };
            if (codec.saveSnapshot(&output.state, errors) != 0) return false;
            if (self.has_archive and !(Suffix{ .graph = &view, .config = self.config }).captureArchive(output)) return false;
            output.ctx_size = @intCast(view.ctx);
            output.approximate = false;
            endpoint.complete = true;
        }
        return true;
    }
};
