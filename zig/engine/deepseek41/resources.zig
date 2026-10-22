//! Own V4.1 graph tensors, host hash buffers and uncached Engram descriptors.
//! C execution-row copies borrow this record. Only the owning graph may deinit
//! it; tensor aliases are separate retained view handles, never shared handles.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const layout = @import("layout.zig");
const Layout = layout.Layout;
const Record = c.ds41_gpu_graph;
const Row = c.ds41_prefill_row;
const Archive = c.ds41_prerope_archive;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Tensor = c.ds4_gpu_tensor;
const InitError = error{ UnsupportedContext, AllocationFailed, InvalidEngram, GpuTransferFailed };

fn emptyRecord() Record {
    var record = std.mem.zeroes(Record);
    record.table[0].fd = -1;
    record.table[1].fd = -1;
    return record;
}
fn tensorAlloc(bytes: u64, managed: bool) ?*Tensor {
    return if (managed) c.ds4_gpu_tensor_alloc_managed(bytes) else c.ds4_gpu_tensor_alloc(bytes);
}

pub const Graph = struct {
    record: Record = emptyRecord(),
    config: Config,
    weights: *const Weights, // Borrowed; its Model mapping also outlives Graph.

    pub fn init(config: Config, model: *const Model, weights: *const Weights, path: [*:0]const u8, context: u32, chunk: u32, streaming: bool) InitError!Graph {
        if (context == 0 or context > 1048576 or config.shape.family != c.DS4_MODEL_FAMILY_DEEPSEEK41)
            return error.UnsupportedContext;
        var self = Graph{ .config = config, .weights = weights };
        errdefer self.deinit();
        const plan = Layout.init(&self.config, context, chunk);
        const graph = &self.record;
        graph.ctx = context;
        graph.prefill_cap = plan.prefill;
        graph.prefill_max = plan.maximum;
        graph.carry_cap = plan.carry;
        graph.compact_carry = plan.compact;
        graph.prefill_alias = plan.alias;
        graph.compressed_kv_f16 = plan.compressed_f16;
        graph.allocation_bytes = plan.bytes();
        graph.streaming = streaming;
        graph.tp_world = 1;
        const token_map = c.malloc(@as(usize, config.shape.n_vocab) * @sizeOf(u32)) orelse return error.AllocationFailed;
        graph.token_map = @ptrCast(@alignCast(token_map));
        const id_rows = if (plan.carry != 0) plan.carry else plan.prefill;
        const ids = c.malloc(@as(usize, id_rows) * @sizeOf([2][c.DS4_ENGRAM_COLS]u32)) orelse return error.AllocationFailed;
        graph.prefill_ids = @ptrCast(@alignCast(ids));
        graph.engram.token_map = graph.token_map;
        graph.engram.vocab_size = config.shape.n_vocab;
        if (!c.ds4_gguf_get_u32(model, "deepseek41.engram.compressed_vocab_size", &graph.engram.compressed_vocab_size) or
            !c.ds4_gguf_get_u32(model, "deepseek41.engram.pad_id", &graph.engram.pad_id)) return error.InvalidEngram;
        if (!readArray(model, "deepseek41.engram.token_map", c.GGUF_VALUE_UINT32, config.shape.n_vocab, graph.token_map, @as(usize, config.shape.n_vocab) * 4) or
            !readArray(model, "deepseek41.engram.rows", c.GGUF_VALUE_UINT32, 2, &graph.engram.rows, @sizeOf(@TypeOf(graph.engram.rows))) or
            !readArray(model, "deepseek41.engram.primes", c.GGUF_VALUE_UINT32, 48, &graph.engram.primes, @sizeOf(@TypeOf(graph.engram.primes))) or
            !readArray(model, "deepseek41.engram.multipliers", c.GGUF_VALUE_UINT64, 8, &graph.engram.multipliers, @sizeOf(@TypeOf(graph.engram.multipliers))) or
            !c.ds4_engram_layout_valid(&graph.engram)) return error.InvalidEngram;
        for (0..2) |owner| {
            const layer: usize = if (owner == 0) 1 else 14;
            const table = findTensor(model, if (owner == 0) "blk.1.engram_embd.weight" else "blk.14.engram_embd.weight") orelse return error.InvalidEngram;
            if (!c.ds4_engram_table_open(&graph.table[owner], path, table.abs_offset, graph.engram.rows[owner])) return error.InvalidEngram;
            // A path replacement between model mapping and row-table open must
            // not silently combine weights and Engram from different files.
            var weights_stat: c.struct_stat = undefined;
            var rows_stat: c.struct_stat = undefined;
            if (c.fstat(model.fd, &weights_stat) != 0 or c.fstat(graph.table[owner].fd, &rows_stat) != 0 or
                weights_stat.st_dev != rows_stat.st_dev or weights_stat.st_ino != rows_stat.st_ino) return error.InvalidEngram;
            const bytes = @as(u64, config.shape.n_embd) * config.shape.n_hc * 4;
            graph.engram_q_norm[owner] = tensorAlloc(bytes, false) orelse return error.AllocationFailed;
            graph.engram_k_norm[owner] = tensorAlloc(bytes, false) orelse return error.AllocationFailed;
            const q_norm = tensorData(model, weights.layer[layer].engram_q_norm) orelse return error.GpuTransferFailed;
            const k_norm = tensorData(model, weights.layer[layer].engram_k_norm) orelse return error.GpuTransferFailed;
            if (c.ds4_gpu_tensor_write(graph.engram_q_norm[owner], 0, q_norm, bytes) == 0 or
                c.ds4_gpu_tensor_write(graph.engram_k_norm[owner], 0, k_norm, bytes) == 0) return error.GpuTransferFailed;
        }
        for (&graph.window) |*window| window.* = tensorAlloc(128 * 512 * 4, false) orelse return error.AllocationFailed;
        for (0..4) |owner| {
            const rows = @as(u64, context) / @as(u64, if (owner < 3) 2 else 1) + 1;
            graph.compressed[owner] = tensorAlloc(rows * 512 * @as(u64, if (plan.compressed_f16) 2 else 4), false) orelse return error.AllocationFailed;
            graph.index_cache[owner] = tensorAlloc(rows * 128 * 4, false) orelse return error.AllocationFailed;
            graph.previous_kv[owner] = tensorAlloc(512 * 4, false) orelse return error.AllocationFailed;
            graph.previous_score[owner] = tensorAlloc(512 * 4, false) orelse return error.AllocationFailed;
        }
        inline for (comptime std.meta.fieldNames(layout.Scratch)) |name| {
            const managed = comptime std.mem.eql(u8, name, "engram_prefetch") or std.mem.eql(u8, name, "image_text_mask");
            @field(graph.*, name) = tensorAlloc(@max(plan.scratchWidth(@field(layout.Scratch, name)), 1) * 4, managed) orelse return error.AllocationFailed;
        }
        if (!allocWorkspace(graph, &plan)) return error.AllocationFailed;
        inline for (layout.carry_fields) |field| {
            if (plan.carry != 0) @field(graph.carry, @tagName(field)) = tensorAlloc(plan.carryWidth(field) * plan.carry * 4, false) orelse return error.AllocationFailed;
        }
        // Apple uses the bounded batch reader directly. The reusable scalar
        // overlap reader belongs to the other backend path and stays NULL.
        resetRecord(graph);
        logAllocation(graph, &plan);
        return self;
    }

    pub fn deinit(self: *Graph) void {
        freeRecord(&self.record);
    }
    pub fn reset(self: *Graph) void {
        resetRecord(&self.record);
    }
    pub fn growWorkspace(self: *Graph, capacity: u32) bool {
        return growRecord(&self.record, &self.config, capacity);
    }
    // Transfer exactly one owner into the transitional C session. The remaining
    // Graph value is empty, so its deferred deinit is harmless.
    pub fn take(self: *Graph) Record {
        const record = self.record;
        self.record = emptyRecord();
        return record;
    }
};

fn readArray(model: *const Model, key: [*:0]const u8, kind: u32, count: u64, out: ?*anyopaque, bytes: usize) bool {
    var array: c.ds4_array_ref = undefined;
    if (!c.ds4_gguf_get_array(model, key, &array) or array.type != kind or array.len != count) return false;
    var cursor = c.ds4_cursor{ .base = model.map, .size = model.size, .pos = array.data_pos, .@"error" = @splat(0) };
    return c.ds4_gguf_read(&cursor, out, bytes);
}
fn findTensor(model: *const Model, name: []const u8) ?*const c.ds4_tensor {
    if (model.n_tensors == 0) return null;
    const tensors: [*]const c.ds4_tensor = model.tensors;
    for (tensors[0..@intCast(model.n_tensors)]) |*tensor| {
        if (tensor.name.len == name.len and std.mem.eql(u8, tensor.name.ptr[0..name.len], name)) return tensor;
    }
    return null;
}
fn tensorData(model: *const Model, maybe_tensor: [*c]const c.ds4_tensor) ?*const anyopaque {
    if (maybe_tensor == null) return null;
    const tensor: *const c.ds4_tensor = @ptrCast(maybe_tensor);
    if (tensor.abs_offset > model.size or tensor.bytes > model.size - tensor.abs_offset) return null;
    return model.map + @as(usize, @intCast(tensor.abs_offset));
}
fn freeRow(row: *Row) void {
    inline for (comptime std.meta.fieldNames(Row)) |name| c.ds4_gpu_tensor_free(@field(row.*, name));
    row.* = std.mem.zeroes(Row);
}
pub fn freeWorkspace(graph: *Record) void {
    if (graph.rows_view != null) {
        const rows: [*]Row = graph.rows_view;
        for (rows[0..graph.prefill_cap]) |*row| freeRow(row);
    }
    // Row views release before their batch parents. Alias views are retained
    // handles of their own; freeing both never double-frees a handle.
    freeRow(&graph.batch);
    c.free(graph.rows_view);
    graph.rows_view = null;
    c.ds4_gpu_tensor_free(graph.prefill_tokens);
    graph.prefill_tokens = null;
}
fn allocWorkspace(graph: *Record, plan: *const Layout) bool {
    const rows = c.calloc(graph.prefill_cap, @sizeOf(Row)) orelse return false;
    graph.rows_view = @ptrCast(@alignCast(rows));
    graph.prefill_tokens = tensorAlloc(@as(u64, graph.prefill_cap) * 4, false);
    if (graph.prefill_tokens == null) return false;
    inline for (layout.storage_fields) |field| {
        const managed = comptime field == .engram_rows or field == .selected_comp;
        @field(graph.batch, @tagName(field)) = tensorAlloc(plan.rowWidth(field) * graph.prefill_cap * 4, managed) orelse return false;
    }
    // Lifetime-based aliases: Engram ends before HC normalization; Q is dead
    // before routed outputs. Heads/low are dead before FFN, x before shared
    // down, and the routed sum may overwrite block. No arithmetic changes.
    if (graph.prefill_alias) {
        if (!alias(graph, plan, .engram_kv, .q, 0) or
            !alias(graph, plan, .flat_norm, .q, 0) or
            !alias(graph, plan, .experts, .q, 0) or
            !alias(graph, plan, .gate, .heads, 0) or
            !alias(graph, plan, .up, .heads, plan.rowWidth(.gate)) or
            !alias(graph, plan, .shared_gate, .low, 0) or
            !alias(graph, plan, .shared_up, .low, plan.rowWidth(.shared_gate)) or
            !alias(graph, plan, .shared_mid, .low, 2 * plan.rowWidth(.shared_gate)) or
            !alias(graph, plan, .shared, .x, 0) or
            !alias(graph, plan, .routed, .block, 0)) return false;
    } else {
        inline for (layout.alias_fields) |field| {
            @field(graph.batch, @tagName(field)) = tensorAlloc(plan.rowWidth(field) * graph.prefill_cap * 4, false) orelse return false;
        }
    }
    inline for (comptime std.meta.fieldNames(Row)) |name| {
        const width = plan.rowWidth(@field(layout.RowField, name));
        const view_rows: [*]Row = graph.rows_view;
        for (view_rows[0..graph.prefill_cap], 0..) |*row, index| {
            @field(row.*, name) = c.ds4_gpu_tensor_view(@field(graph.batch, name), @as(u64, index) * width * 4, width * 4) orelse return false;
        }
    }
    return c.ds4_gpu_tensor_fill_f32(graph.batch.block_mask, 0, @as(u64, graph.prefill_cap) * plan.rowWidth(.block_mask)) != 0;
}
fn alias(graph: *Record, plan: *const Layout, comptime target: layout.RowField, comptime parent: layout.RowField, offset: u64) bool {
    const tensor = c.ds4_gpu_tensor_view(@field(graph.batch, @tagName(parent)), offset * graph.prefill_cap * 4, plan.rowWidth(target) * graph.prefill_cap * 4) orelse return false;
    @field(graph.batch, @tagName(target)) = tensor;
    return true;
}

pub fn clearPrerope(maybe_archive: [*c]Archive, context: u32) void {
    if (maybe_archive == null) return;
    const archive: *Archive = @ptrCast(maybe_archive);
    @memset(std.mem.asBytes(&archive.raw_epoch), 0xff);
    for (0..4) |owner| {
        const count = context / @as(u32, if (owner < 3) 2 else 1) + 1;
        @memset(archive.published[owner][0..count], 0);
    }
}
pub fn freePrerope(maybe_archive: [*c]Archive) void {
    if (maybe_archive == null) return;
    const archive: *Archive = @ptrCast(maybe_archive);
    for (archive.raw) |tensor| c.ds4_gpu_tensor_free(tensor);
    for (0..4) |owner| {
        c.ds4_gpu_tensor_free(archive.compressed[owner]);
        c.ds4_gpu_tensor_free(archive.index[owner]);
        c.free(archive.published[owner]);
    }
    c.free(archive);
}
pub fn resetRecord(graph: *Record) void {
    graph.pos = 0;
    graph.decoder_replay_floor = 0;
    graph.valid = !graph.workspace_failed;
    c.ds4_engram_history_reset(&graph.history);
    c.ds4_native_graph_dspark_reset(graph.dspark);
    clearPrerope(graph.prerope, graph.ctx);
    // Visibility is controlled by lengths, not zeroed KV. Even-position tokens
    // overwrite unfinished pairs before those pairs can be consumed.
}
pub fn freeRecord(graph: *Record) void {
    c.ds4_native_graph_dspark_free(graph.dspark);
    c.ds4_engram_reader_free(graph.engram_reader);
    c.ds4_gpu_tensor_free(graph.tp_logits_half);
    if (graph.predict_load.active) {
        c.ds4_native_graph_selected_finish(&graph.predict_load);
        _ = c.ds4_gpu_routed_moe_set_selected_override(null, 0);
    }
    c.ds4_gpu_tensor_free(graph.predict_selected);
    c.ds4_gpu_tensor_free(graph.oracle_selected);
    freeWorkspace(graph);
    freePrerope(graph.prerope);
    inline for (layout.carry_fields) |field| c.ds4_gpu_tensor_free(@field(graph.carry, @tagName(field)));
    for (0..2) |owner| {
        c.ds4_engram_table_close(&graph.table[owner]);
        c.ds4_gpu_tensor_free(graph.engram_q_norm[owner]);
        c.ds4_gpu_tensor_free(graph.engram_k_norm[owner]);
    }
    for (graph.window) |tensor| c.ds4_gpu_tensor_free(tensor);
    for (0..4) |owner| {
        c.ds4_gpu_tensor_free(graph.compressed[owner]);
        c.ds4_gpu_tensor_free(graph.index_cache[owner]);
        c.ds4_gpu_tensor_free(graph.previous_kv[owner]);
        c.ds4_gpu_tensor_free(graph.previous_score[owner]);
    }
    inline for (comptime std.meta.fieldNames(layout.Scratch)) |name| c.ds4_gpu_tensor_free(@field(graph.*, name));
    // TP exchange views, image spans, imatrix and verify_previous_* are borrowed
    // from their respective owners. They must not be freed here.
    c.free(graph.token_map);
    c.free(graph.prefill_ids);
    graph.* = emptyRecord();
}
pub fn growRecord(graph: *Record, config: *const Config, capacity: u32) bool {
    if (capacity <= graph.prefill_cap) return true;
    if (capacity > graph.prefill_max or c.ds4_gpu_synchronize() == 0) return false;
    // Admission and weight retirement happen in the caller. Persistent KV,
    // history, original-key tape and carry storage survive workspace growth.
    freeWorkspace(graph);
    c.ds4_gpu_tensor_free(graph.raw_prefill);
    c.ds4_gpu_tensor_free(graph.image_text_mask);
    c.ds4_gpu_tensor_free(graph.index_packed);
    graph.prefill_cap = capacity;
    graph.raw_prefill = tensorAlloc((@as(u64, capacity) + 128) * config.shape.n_head_dim * 4, false);
    graph.image_text_mask = tensorAlloc((@as(u64, capacity) + 3) / 4 * 4, false);
    graph.index_packed = tensorAlloc(c.ds4_gpu_dsv41_indexer_packed_bytes(graph.ctx, capacity), false);
    const plan = Layout.fromRecord(config, graph);
    if (graph.raw_prefill == null or graph.image_text_mask == null or graph.index_packed == null or !allocWorkspace(graph, &plan)) {
        graph.workspace_failed = true;
        graph.valid = false;
        return false;
    }
    const estimate = Layout.init(config, graph.ctx, capacity);
    graph.allocation_bytes = estimate.bytes();
    return true;
}
fn logAllocation(graph: *const Record, plan: *const Layout) void {
    _ = c.fprintf(c.__stderrp, "ds4: V4.1 static context buffers %.2f MiB (ctx=%u), Engram disk-only\n", @as(f64, @floatFromInt(plan.bytes())) / 1048576.0, graph.ctx);
    if (!layout.present("DS4_V41_COMPRESSED_KV_F16")) return;
    var main: u64 = 0;
    var index: u64 = 0;
    var raw: u64 = 0;
    for (graph.compressed) |tensor| main += c.ds4_gpu_tensor_bytes(tensor);
    for (graph.index_cache) |tensor| index += c.ds4_gpu_tensor_bytes(tensor);
    for (graph.window) |tensor| raw += c.ds4_gpu_tensor_bytes(tensor);
    const storage: [*:0]const u8 = if (graph.compressed_kv_f16) "f16" else "f32";
    _ = c.fprintf(c.__stderrp, "ds4: V4.1 KV allocated main=%s main_bytes=%llu index_bytes=%llu raw_bytes=%llu\n", storage, @as(c_ulonglong, main), @as(c_ulonglong, index), @as(c_ulonglong, raw));
}
