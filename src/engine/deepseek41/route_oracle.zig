//! Diagnostic future-route tape. Native-endian headers and rows remain exact;
//! this adds no prediction to model arithmetic. One process/session owns the
//! files, borrowed graph identity and optional libc-backed replay words.
const std = @import("std");
const c = @import("gguf_c");
const Ops = @import("ops.zig").Ops;
const cache = @import("expert_cache.zig");
const Graph = c.ds41_gpu_graph;
const Layer = c.ds4_layer_weights;
const allocator = std.heap.c_allocator;
fn die(message: [*:0]const u8) noreturn {
    c.ds4_engine_model_die(message);
    unreachable;
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}

pub const Oracle = struct {
    initialized: bool = false,
    enabled: bool = false,
    active: bool = false,
    preattention: bool = false,
    defer_check: bool = false,
    probe: bool = false,
    record: ?*c.FILE = null,
    features: ?*c.FILE = null,
    cache_file: ?*c.FILE = null,
    owner: ?*const Graph = null,
    words: ?[]u32 = null,
    cursor: usize = 0,
    position: u32 = 0,
    token: u32 = 0,
    checked: u32 = 0,
    // The supported oracle contract admits exactly one graph/model. Retain
    // its immutable layer count for the process-exit report, not a Config borrow.
    layers: u32 = 0,
    predicted: [c.DS4_MAX_EXPERT_USED]i32 = @splat(0),
    probe_calls: [c.DS4_MAX_LAYER]u64 = @splat(0),
    probe_hits: [c.DS4_MAX_LAYER]u64 = @splat(0),
    probe_all: [c.DS4_MAX_LAYER]u64 = @splat(0),

    pub fn close(self: *Oracle) void {
        if (self.probe) for (0..self.layers) |index| {
            _ = c.fprintf(c.__stderrp, "ds4: routing probe layer=%u calls=%llu hits=%llu all=%llu\n", @as(u32, @intCast(index)), self.probe_calls[index], self.probe_hits[index], self.probe_all[index]);
        };
        if (self.cache_file != null and c.fclose(self.cache_file) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: routing cache recording close failed\n");
            c._Exit(1);
        }
        if (self.features != null and c.fclose(self.features) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: routing feature recording close failed\n");
            c._Exit(1);
        }
        if (self.record != null and c.fclose(self.record) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: oracle route recording close failed\n");
            c._Exit(1);
        }
        if (self.words) |words| {
            if (self.cursor != words.len) {
                _ = c.fprintf(c.__stderrp, "ds4: oracle route recording was not fully consumed\n");
                c._Exit(1);
            }
            allocator.free(words);
        }
    }
    pub fn begin(self: *Oracle, ops: Ops, graph: *Graph, token: u32, on_exit: *const fn () callconv(.c) void) void {
        const shape = &ops.config.shape;
        if (!self.initialized) {
            self.initialized = true;
            const record = c.getenv("DS4_V41_ROUTE_RECORD");
            const replay = c.getenv("DS4_V41_ROUTE_ORACLE");
            const features = c.getenv("DS4_V41_ROUTE_FEATURES");
            const cache_file = c.getenv("DS4_V41_ROUTE_CACHE");
            if (cache_file != null and record == null) die("routing cache capture requires route recording mode");
            self.probe = present("DS4_V41_ROUTE_PROBE") or features != null;
            if (features != null and record == null) die("routing features require route recording mode");
            if (self.probe and record == null) die("routing probe requires route recording mode");
            if ((present("DS4_V41_ORACLE_PREATTENTION") or present("DS4_V41_ORACLE_DEFER_CHECK")) and replay == null)
                die("oracle schedule requires a route recording");
            if (record == null and replay == null) return;
            if (record != null and replay != null) die("oracle recording and replay are mutually exclusive");
            const header = [_]u32{ 0x44535231, shape.n_layer, shape.n_expert, shape.n_expert_used };
            if (record != null) {
                self.record = c.fopen(record, "wbx");
                if (self.record == null or c.fwrite(&header, @sizeOf(@TypeOf(header)), 1, self.record) != 1)
                    die("cannot create oracle route recording");
            } else {
                const file = c.fopen(replay, "rb");
                var actual: [4]u32 = undefined;
                if (file == null or c.fread(&actual, @sizeOf(@TypeOf(actual)), 1, file) != 1 or
                    !std.mem.eql(u32, &actual, &header) or c.fseek(file, 0, c.SEEK_END) != 0) die("invalid oracle route header");
                const bytes = c.ftell(file);
                const row_bytes = (3 + shape.n_expert_used) * @sizeOf(u32);
                if (bytes <= @sizeOf(@TypeOf(header))) die("invalid oracle route length");
                const payload: usize = @as(usize, @intCast(bytes)) - @sizeOf(@TypeOf(header));
                if (payload % (row_bytes * shape.n_layer) != 0 or c.fseek(file, @sizeOf(@TypeOf(header)), c.SEEK_SET) != 0)
                    die("invalid oracle route length");
                const words = allocator.alloc(u32, payload / @sizeOf(u32)) catch die("cannot read oracle routes");
                self.words = words;
                if (c.fread(words.ptr, @sizeOf(u32), words.len, file) != words.len) die("cannot read oracle routes");
                _ = c.fclose(file);
                self.preattention = present("DS4_V41_ORACLE_PREATTENTION");
                self.defer_check = present("DS4_V41_ORACLE_DEFER_CHECK");
                if (self.defer_check and !self.preattention) die("deferred oracle check requires pre-attention loading");
            }
            if (features != null) {
                const header_ = [_]u32{ 0x44534631, 1, shape.n_layer, shape.n_embd, shape.n_expert, shape.n_expert_used, 2, 0 };
                self.features = c.fopen(features, "wbx");
                if (self.features == null or c.fwrite(&header_, @sizeOf(@TypeOf(header_)), 1, self.features) != 1)
                    die("cannot create routing feature recording");
            }
            if (cache_file != null) {
                const header_ = [_]u32{ 0x44534331, 1, shape.n_layer, shape.n_expert };
                self.cache_file = c.fopen(cache_file, "wbx");
                if (self.cache_file == null or c.fwrite(&header_, @sizeOf(@TypeOf(header_)), 1, self.cache_file) != 1)
                    die("cannot create routing cache recording");
            }
            self.enabled = true;
            self.owner = graph;
            self.layers = shape.n_layer;
            _ = c.atexit(on_exit);
        }
        if (!self.enabled) return;
        if (self.owner != graph) die("oracle supports one scalar session per process");
        if (!graph.streaming or graph.quality or graph.imatrix != null or graph.image_count != 0 or graph.tp_world != 1 or
            present("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") or present("DS4_METAL_V41_ASYNC_EXPERT_LOAD") or
            present("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD")) die("oracle requires ordinary single-rank text SSD early loading");
        if (self.defer_check and graph.oracle_selected == null) {
            graph.oracle_selected = c.ds4_gpu_tensor_alloc(@as(u64, shape.n_layer) * shape.n_expert_used * @sizeOf(i32)) orelse die("cannot allocate deferred oracle validation");
        }
        self.active = true;
        self.position = graph.pos;
        self.token = token;
        self.checked = 0;
    }
    fn ids(self: *const Oracle, ops: Ops, layer: u32) []const i32 {
        const shape = &ops.config.shape;
        const stride = 3 + shape.n_expert_used;
        const offset = self.cursor + @as(usize, layer) * stride;
        const words = self.words orelse die("oracle routes exhausted");
        if (offset + stride > words.len) die("oracle routes exhausted");
        const row = words[offset..][0..stride];
        if (row[0] != self.position or row[1] != self.token or row[2] != layer) die("oracle route position/token/layer mismatch");
        for (row[3..]) |id| if (id >= shape.n_expert) die("invalid oracle expert ID");
        return @as([*]const i32, @ptrCast(row.ptr + 3))[0..shape.n_expert_used];
    }
    fn check(self: *Oracle, ops: Ops, layer: u32, selected: []const i32) void {
        if (!self.active) return;
        const expected = self.checked;
        self.checked +%= 1;
        if (layer != expected) die("oracle layer order mismatch");
        if (self.probe) {
            var hits: u32 = 0;
            for (selected) |id| for (self.predicted[0..selected.len]) |predicted| if (id == predicted) {
                hits += 1;
                break;
            };
            self.probe_calls[layer] +%= 1;
            self.probe_hits[layer] +%= hits;
            self.probe_all[layer] +%= @intFromBool(hits == selected.len);
        }
        if (self.record) |file| {
            var row: [3 + c.DS4_MAX_EXPERT_USED]u32 = undefined;
            row[0..3].* = .{ self.position, self.token, layer };
            @memcpy(std.mem.sliceAsBytes(row[3..][0..selected.len]), std.mem.sliceAsBytes(selected));
            if (c.fwrite(&row, @sizeOf(u32), 3 + selected.len, file) != 3 + selected.len) die("oracle route write failed");
        } else if (!std.mem.eql(i32, selected, self.ids(ops, layer))) die("oracle prediction differs from exact router");
    }
    pub fn prefetch(self: *const Oracle, ops: Ops, layer: *const Layer, index: u32) bool {
        if (!self.active or !self.preattention) return true;
        const table = routeTable(ops, layer, index) orelse return false;
        // Submit preceding work before preparing buffers. Existing in-flight
        // tracking protects every cache buffer still referenced by that work.
        return c.ds4_gpu_flush_commands() != 0 and c.ds4_gpu_stream_expert_cache_begin_selected_load(&table, self.ids(ops, index).ptr, ops.config.shape.n_expert_used) != 0;
    }
    /// Intrusive offline accuracy probe, not the timed prefetch predictor.
    pub fn probeRoutes(self: *Oracle, ops: Ops, graph: *Graph, layer: *const Layer, token: u32) bool {
        if (!self.active or !self.probe) return true;
        const shape = &ops.config.shape;
        if (layer.ffn_exp_probs_b == null) return false;
        var ok = select(ops, graph, layer, graph.selected, token) and c.ds4_gpu_end_commands() != 0 and
            c.ds4_gpu_tensor_read(graph.selected, 0, &self.predicted, @as(u64, shape.n_expert_used) * @sizeOf(i32)) != 0;
        if (ok and self.features != null) {
            var values: [c.DS4_MAX_EMBD]f32 = undefined;
            var bf16: [c.DS4_MAX_EMBD]u16 = undefined;
            ok = c.ds4_gpu_tensor_read(graph.norm, 0, &values, @as(u64, shape.n_embd) * @sizeOf(f32)) != 0;
            if (ok) for (values[0..shape.n_embd], bf16[0..shape.n_embd]) |value, *half| {
                const bits: u32 = @bitCast(value);
                if (bits & 0xffff != 0) die("routing feature is not BF16-exact");
                half.* = @truncate(bits >> 16);
            };
            const meta = [_]u32{ self.position, token, self.checked };
            if (ok and (c.fwrite(&meta, @sizeOf(@TypeOf(meta)), 1, self.features) != 1 or
                c.fwrite(&bf16, @sizeOf(u16), shape.n_embd, self.features) != shape.n_embd or
                c.fwrite(&self.predicted, @sizeOf(i32), shape.n_expert_used, self.features) != shape.n_expert_used)) die("routing feature write failed");
        }
        return ok and c.ds4_gpu_begin_commands() != 0;
    }
    pub fn captureDeferred(self: *const Oracle, ops: Ops, graph: *Graph, index: u32, selected: []i32) bool {
        // Preserve actual GPU routes for validation after the token, while exact
        // recorded addresses can be loaded without waiting for this router.
        const bytes = @as(u64, selected.len) * @sizeOf(i32);
        if (c.ds4_gpu_tensor_copy(graph.oracle_selected, index * bytes, graph.selected, 0, bytes) == 0) return false;
        @memcpy(selected, self.ids(ops, index));
        return true;
    }
    pub fn observe(self: *Oracle, ops: Ops, table: *const c.ds4_gpu_stream_expert_table, index: u32, token: u32, selected: []const i32) void {
        self.check(ops, index, selected);
        if (self.cache_file) |file| {
            var resident: [c.DS4_MAX_EXPERT]u8 = undefined;
            const meta = [_]u32{ self.position, token, index };
            if (c.ds4_gpu_stream_expert_cache_snapshot(table, &resident) == 0 or
                c.fwrite(&meta, @sizeOf(@TypeOf(meta)), 1, file) != 1 or
                c.fwrite(&resident, 1, ops.config.shape.n_expert, file) != ops.config.shape.n_expert) die("routing cache snapshot failed");
        }
    }
    pub fn end(self: *Oracle, ops: Ops, graph: *Graph, success: bool) bool {
        var ok = success;
        if (self.active) {
            const shape = &ops.config.shape;
            if (ok and self.defer_check) {
                var actual: [c.DS4_MAX_LAYER * c.DS4_MAX_EXPERT_USED]i32 = undefined;
                ok = c.ds4_gpu_tensor_read(graph.oracle_selected, 0, &actual, @as(u64, shape.n_layer) * shape.n_expert_used * @sizeOf(i32)) != 0;
                if (ok) for (0..shape.n_layer) |index| self.check(ops, @intCast(index), actual[index * shape.n_expert_used ..][0..shape.n_expert_used]);
            }
            if (ok and self.checked != shape.n_layer) die("incomplete oracle step");
            if (self.words != null) self.cursor += @as(usize, shape.n_layer) * (3 + shape.n_expert_used);
            self.active = false;
        }
        return ok;
    }
};

pub fn routeTable(ops: Ops, layer: *const Layer, index: u32) ?c.ds4_gpu_stream_expert_table {
    var gate: u64 = 0;
    var down: u64 = 0;
    if (!c.ds4_gguf_tensor_nbytes(layer.ffn_gate_exps.*.type, ops.config.shape.n_embd, &gate) or
        !c.ds4_gguf_tensor_nbytes(layer.ffn_down_exps.*.type, ops.config.shape.n_ff_exp, &down)) return null;
    return cache.table(ops.config, ops.model, layer, index, gate *% ops.config.shape.n_ff_exp, down *% ops.config.shape.n_embd);
}
pub fn select(ops: Ops, graph: *Graph, layer: *const Layer, selected: ?*c.ds4_gpu_tensor, token: u32) bool {
    const shape = &ops.config.shape;
    return ops.matmul(graph.route_logits, @ptrCast(layer.ffn_gate_inp), graph.norm, false) and
        c.ds4_gpu_router_select_tensor(selected, graph.route_weights, graph.route_probs, ops.model.map, ops.model.size, layer.ffn_exp_probs_b.*.abs_offset, 0, 0, token, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale, 0, 0, true, false, graph.route_logits) != 0;
}
