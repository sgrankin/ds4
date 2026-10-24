//! One combined forward shares weight reads across independent sessions.
//! Row graphs borrow private recurrence/KV and retained views into one arena;
//! only those views are freed here. Session commits remain with the caller.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const moe = @import("moe.zig");
const embeddings = @import("embedding.zig");
const input = @import("input.zig");
const ngrams = @import("ngrams.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Member = c.ds4_qwen4_batch_member;
const Pool = c.ds4_qwen4_state_pool;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const maximum = 64;
const allocator = std.heap.c_allocator;

const row_fields = .{
    "R",   "ple_gated", "ple_normed", "qkv", "ga", "gb", "lin_o",
    "qg",  "kp",        "vp",         "iq",  "ik", "q",  "gate",
    "iqn", "attn_o",    "mixed",
};

const Row = struct {
    graph: Graph,
    views: [row_fields.len]?*Tensor = @splat(null),

    fn init(source: *const Graph, arena: *const Graph, first: u32, count: u32, mixed: bool) !Row {
        var row = Row{ .graph = source.* };
        errdefer row.deinit();
        const shape = &arena.config.shape;
        const embd: u64 = shape.n_embd;
        const hc = embd * shape.n_hc;
        const query = @as(u64, shape.n_head) * shape.n_head_dim;
        const kv = @as(u64, shape.n_head_kv) * shape.n_head_dim;
        const iq = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
        const widths = [_]u64{
            hc,        hc,    hc,   dimensions.convWidth(shape), shape.n_lin_v_head,       shape.n_lin_v_head, dimensions.valueWidth(shape),
            2 * query, kv,    kv,   iq,                          shape.n_indexer_head_dim, query,              query,
            iq,        query, embd,
        };
        inline for (row_fields, 0..) |name, index| {
            if (index != row_fields.len - 1 or mixed) {
                const view = ops.rowsView(@field(arena.*, name), first, count, widths[index]) orelse return error.AllocationFailed;
                row.views[index] = view;
                @field(row.graph, name) = view;
            }
        }
        return row;
    }

    fn deinit(self: *Row) void {
        for (self.views) |view| c.ds4_gpu_tensor_free(view);
        self.views = @splat(null);
    }
};

pub fn ensureScratch(graph: *Graph, requested_rows: u32) bool {
    const rows = @max(32, requested_rows);
    if (graph.batch_logits != null and graph.batch_logit_rows >= rows) return true;
    inline for (.{ "batch_logits", "batch_attn_rows", "batch_attn_part", "batch_head_x", "batch_gdn_rows" }) |name| c.ds4_gpu_tensor_free(@field(graph.*, name));
    graph.batch_logit_rows = 0;
    const shape = &graph.config.shape;
    graph.batch_gdn_rows = c.ds4_gpu_tensor_alloc(@as(u64, shape.n_layer) * maximum * c.DS4_GPU_QWEN4_GDN_ROW_BYTES);
    graph.batch_logits = c.ds4_gpu_tensor_alloc(@as(u64, rows) * shape.n_vocab * @sizeOf(f32));
    graph.batch_head_x = c.ds4_gpu_tensor_alloc(@as(u64, rows) * shape.n_embd * @sizeOf(f32));
    graph.batch_attn_rows = c.ds4_gpu_tensor_alloc(@as(u64, shape.n_layer) * maximum * c.DS4_GPU_QWEN4_ATTN_ROW_BYTES);
    graph.batch_attn_part = c.ds4_gpu_tensor_alloc(c.ds4_gpu_qwen4_attn_part_floats(rows, shape.n_head, shape.n_head_dim) * @sizeOf(f32));
    if (graph.batch_logits == null or graph.batch_head_x == null or graph.batch_attn_rows == null or
        graph.batch_attn_part == null or graph.batch_gdn_rows == null) return false;
    graph.batch_logit_rows = rows;
    return true;
}

pub fn supported(optional_arena: ?*const Graph, members: []const Member, speculative: bool) bool {
    const raw = c.getenv("DS4_QWEN4_SESSION_BATCH");
    if ((raw != null and raw[0] != 0 and c.strcmp(raw, "0") == 0) or members.len < 2 or members.len > maximum) return false;
    // Each speculative member may contribute two token rows. The existing
    // Metal row primitives admit64 rows, even though there may be64 sessions.
    if (speculative and members.len > maximum / 2) return false;
    const arena = optional_arena orelse return false;
    if (arena.cap_tokens < members.len) return false;
    if (speculative) {
        const rows = @as(u64, members.len) * 2;
        const projection = rows * (arena.config.shape.n_hc + 1) * arena.config.shape.n_embd * @sizeOf(f32);
        if (rows > arena.cap_tokens or c.ds4_gpu_tensor_bytes(arena.mid) < projection or c.ds4_gpu_tensor_bytes(arena.part) < 2 * projection) return false;
    }
    for (members) |member| {
        const graph: *const Graph = member.graph;
        const frontier: *const c.ds4_session_frontier = member.frontier;
        if (!frontier.checkpoint_valid or graph.owns_scratch or graph.R != arena.R or graph.n_block_cap > arena.n_block_cap or
            graph.pos != frontier.checkpoint.len or graph.vis_span_count != 0 or (speculative and graph.mtp_R == null) or
            graph.snap_after_first or graph.snap_after_second or graph.steer_attn_scale != 0 or graph.steer_ffn_scale != 0 or graph.dump_prompt_rows) return false;
    }
    return true;
}

fn stageEmbeddings(members: []const Member, model: *const Model, weights: *const Weights, arena: *Graph, ids: []u32) bool {
    const shape = &arena.config.shape;
    const embd: usize = shape.n_embd;
    const width = embd * shape.n_hc;
    for (members, 0..) |member, index| {
        const graph: *Graph = member.graph;
        const destination = arena.host_row[index * width ..][0..width];
        if (!embeddings.row(model, weights.token_embd, @intCast(member.tokens[0]), destination[0..embd])) return false;
        for (1..shape.n_hc) |stream| @memcpy(destination[stream * embd ..][0..embd], destination[0..embd]);
        const rope = input.ropePosition(&.{}, graph.pos, &graph.mrope_delta);
        const position = [4]u32{ rope[0], rope[1], rope[2], 0 };
        if (c.ds4_gpu_tensor_write(graph.pos3, @as(u64, graph.pos) * 16, &position, 16) == 0) return false;
        ngrams.step(&graph.config, member.tokens[0], &graph.ple_prev, ids[index * arena.config.ple.n_heads ..][0..arena.config.ple.n_heads]);
    }
    return c.ds4_gpu_tensor_write(arena.R, 0, arena.host_row, members.len * width * @sizeOf(f32)) != 0;
}

fn stageNgrams(rows: u32, model: *const Model, arena: *Graph, ids: []const u32) bool {
    const values = arena.host_row[0 .. @as(usize, rows) * arena.config.shape.n_embd];
    if (!ngrams.read(model, ids, values)) {
        _ = c.fprintf(c.__stderrp, "ds4: n-gram read failed: %s\n", c.strerror(c.__error().*));
        return false;
    }
    return c.ds4_gpu_tensor_write(arena.ple_emb, 0, values.ptr, values.len * @sizeOf(f32)) != 0;
}

fn linear(members: []const Member, rows: []const Row, arena: *Graph, pool: ?*const Pool, model: *const Model, layer: *const Layer, index: u32, count: u32) bool {
    const shape = &arena.config.shape;
    if (!(ops.dense(arena.qkv, model, layer.lin_qkv, arena.mixed, count) and ops.dense(arena.z, model, layer.lin_gate, arena.mixed, count) and
        ops.dense(arena.ga, model, layer.lin_alpha, arena.mixed, count) and ops.dense(arena.gb, model, layer.lin_beta, arena.mixed, count))) return false;
    var slots: [maximum]u32 = undefined;
    var pooled = pool != null and pool.?.state[index] != null and !ops.present("DS4_QWEN4_NO_BATCH_SCAN");
    if (pooled) for (members, 0..) |member, row| {
        if (member.slot < 0) {
            pooled = false;
            break;
        }
        slots[row] = @intCast(member.slot);
    };
    const conv: u32 = @intCast(dimensions.convWidth(shape));
    const values: u32 = @intCast(dimensions.valueWidth(shape));
    if (pooled) {
        if (c.ds4_gpu_qwen4_conv_stream_rows_tensor(arena.qkv, pool.?.history[index], model.map, model.size, layer.lin_conv.*.abs_offset, &slots, count, conv, shape.n_lin_conv, @intCast(dimensions.historyFloats(shape)), conv, 1) == 0) return false;
    } else {
        for (rows) |*row| {
            if (c.ds4_gpu_qwen4_conv_stream_tensor(row.graph.qkv, row.graph.layer_lin_hist[index], model.map, model.size, layer.lin_conv.*.abs_offset, 1, conv, shape.n_lin_conv, true) == 0) return false;
        }
    }
    if (c.ds4_gpu_qwen4_gdn_prep_tensor(arena.qkv, arena.ga, arena.gb, model.map, model.size, layer.lin_a.*.abs_offset, layer.lin_dt_bias.*.abs_offset, count, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim) == 0) return false;
    if (pooled) {
        if (c.ds4_gpu_qwen4_gdn_scan_rows_tensor(arena.lin_o, pool.?.state[index], arena.qkv, arena.ga, arena.gb, &slots, count, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim, @intCast(dimensions.stateFloats(shape)), conv, values) == 0) return false;
    } else {
        for (rows) |*row| {
            if (c.ds4_gpu_qwen4_gdn_scan_tensor(row.graph.lin_o, row.graph.layer_lin_state[index], row.graph.qkv, row.graph.ga, row.graph.gb, 1, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim, null, 0, null, 1) == 0) return false;
        }
    }
    return c.ds4_gpu_qwen4_gdn_out_tensor(arena.lin_o, arena.z, model.map, model.size, layer.lin_norm.*.abs_offset, count, shape.n_lin_v_head, shape.n_lin_head_dim, shape.rms_eps) != 0 and ops.dense(arena.blk, model, layer.lin_out, arena.lin_o, count);
}

fn attentionEntries(entries: []c.ds4_gpu_qwen4_attn_row, arena: *Graph, model: *const Model, layer: *const Layer, index: u32) bool {
    const shape = &arena.config.shape;
    const count: u32 = @intCast(entries.len);
    var block_stride: u32 = 0;
    var sparse = false;
    var block = false;
    for (entries) |entry| {
        block_stride = @max(block_stride, (entry.pos + 1) / 4);
        sparse = sparse or entry.use_sel != 0;
        block = block or (entry.pos + 1) % 4 == 0;
    }
    const first = @as(u64, index) * maximum;
    var ok = c.ds4_gpu_qwen4_attn_rows_stage(arena.batch_attn_rows, first, entries.ptr, count, 4) != 0 and
        c.ds4_gpu_qwen4_attn_prep_rows_tensor(arena.q, arena.gate, arena.iqn, arena.qg, arena.kp, arena.vp, arena.iq, arena.ik, arena.batch_attn_rows, first, entries.ptr, count, model.map, model.size, layer.attn_q_norm.*.abs_offset, layer.attn_k_norm.*.abs_offset, layer.indexer_q_norm.*.abs_offset, shape.n_head, shape.n_head_kv, shape.n_head_dim, shape.n_rot, shape.n_indexer_head, shape.n_indexer_head_dim, shape.rope_freq_base, shape.rms_eps) != 0;
    if (block) ok = ok and c.ds4_gpu_qwen4_idx_block_key_rows_tensor(arena.batch_attn_rows, first, entries.ptr, count, model.map, model.size, layer.indexer_k_norm.*.abs_offset, 4, shape.n_indexer_head_dim, shape.n_rot, shape.rope_freq_base, shape.rms_eps) != 0;
    if (sparse) ok = ok and c.ds4_gpu_qwen4_idx_score_rows_tensor(arena.score, arena.tile_max, arena.iqn, arena.batch_attn_rows, first, entries.ptr, count, block_stride, shape.n_indexer_head, shape.n_indexer_head_dim, 4) != 0 and
        c.ds4_gpu_qwen4_idx_select_rows_tensor(arena.sel_blocks, arena.score, arena.tile_max, arena.batch_attn_rows, first, entries.ptr, count, block_stride, arena.k_blocks) != 0 and
        c.ds4_gpu_qwen4_idx_expand_rows_tensor(arena.sel_tokens, arena.n_sel, arena.sel_blocks, arena.batch_attn_rows, first, count, arena.k_blocks, 4, arena.sel_stride) != 0;
    return ok and c.ds4_gpu_qwen4_attn_decode_rows_tensor(arena.attn_o, arena.q, arena.gate, arena.sel_tokens, arena.n_sel, arena.batch_attn_part, arena.batch_attn_rows, first, entries.ptr, count, shape.n_head, shape.n_head_kv, shape.n_head_dim, arena.sel_stride, 1 / @sqrt(@as(f32, @floatFromInt(shape.n_head_dim)))) != 0;
}

fn attend(rows: []Row, arena: *Graph, model: *const Model, layer: *const Layer, index: u32, count: u32) bool {
    const shape = &arena.config.shape;
    if (!(ops.dense(arena.qg, model, layer.attn_q, arena.mixed, count) and ops.dense(arena.kp, model, layer.attn_k, arena.mixed, count) and
        ops.dense(arena.vp, model, layer.attn_v, arena.mixed, count) and ops.dense(arena.iq, model, layer.indexer_q_proj, arena.mixed, count) and
        ops.dense(arena.ik, model, layer.indexer_k_proj, arena.mixed, count))) return false;
    const rows_path = !ops.present("DS4_QWEN4_NO_BATCH_ATTN") and !ops.present("DS4_QWEN4_NO_IDX_SELECT") and (shape.n_head_dim == 128 or shape.n_head_dim == 256);
    if (rows_path) {
        var entries: [maximum]c.ds4_gpu_qwen4_attn_row = undefined;
        const sparse_at = (arena.k_blocks + 1) * 4 - 1;
        for (rows, 0..) |*row, slot| {
            const graph = &row.graph;
            if (graph.pos >= graph.ctx_cap) return false;
            entries[slot] = .{
                .k_cache = graph.layer_k_cache[index],
                .v_cache = graph.layer_v_cache[index],
                .ik_cache = graph.layer_ik_cache[index],
                .block_key = graph.layer_block_key[index],
                .pos3 = graph.pos3,
                .pos = graph.pos,
                .use_sel = @intFromBool(graph.pos >= sparse_at),
            };
        }
        if (!attentionEntries(entries[0..rows.len], arena, model, layer, index)) return false;
    } else {
        for (rows) |*row| {
            if (!attention.tail(&row.graph, model, layer, index, row.graph.pos, 1)) return false;
        }
    }
    return ops.dense(arena.blk, model, layer.attn_output, arena.attn_o, count);
}

const Profile = struct {
    enabled: bool,
    last: f64,
    elapsed: [7]f64 = @splat(0),
    fn mark(self: *Profile, index: usize) void {
        if (!self.enabled) return;
        _ = c.ds4_gpu_end_commands();
        const time = ops.now();
        self.elapsed[index] += time - self.last;
        self.last = time;
        _ = ops.beginCommands();
    }
};
var timing: ?bool = null;

pub fn encode(members: []const Member, arena: *Graph, pool: ?*const Pool, model: *const Model, weights: *const Weights) bool {
    if (members.len == 0 or members.len > maximum) return false;
    const count: u32 = @intCast(members.len);
    const rows = allocator.alloc(Row, members.len) catch c.ds4_engine_model_die("out of memory");
    defer allocator.free(rows);
    var built: usize = 0;
    defer for (rows[0..built]) |*row| row.deinit();
    for (members, rows, 0..) |member, *row, index| {
        row.* = Row.init(member.graph, arena, @intCast(index), 1, false) catch return false;
        built += 1;
    }
    if (!ensureScratch(arena, count)) return false;
    if (timing == null) timing = ops.present("DS4_QWEN4_BATCH_TIMING");
    var profile = Profile{ .enabled = timing.?, .last = if (timing.?) ops.now() else 0 };
    var ids: [maximum * c.DS4_MAX_PLE_HEADS]u32 = undefined;
    var staged_ngrams = false;
    var ok = stageEmbeddings(members, model, weights, arena, &ids);
    profile.mark(0);
    const shape = &arena.config.shape;
    for (0..dimensions.trunkLayers(&arena.config)) |offset| {
        if (!ok) break;
        const index: u32 = @intCast(offset);
        const layer = &weights.layer[index];
        const ple = dimensions.isPle(&arena.config, index);
        if (!staged_ngrams and ple) {
            staged_ngrams = true;
            ok = (ops.present("DS4_QWEN4_NO_NGRAM_OVERLAP") or c.ds4_gpu_flush_commands() != 0) and
                stageNgrams(count, model, arena, ids[0 .. members.len * arena.config.ple.n_heads]);
            if (!ok) break;
        }
        if (ple) {
            ok = ops.dense(arena.ple_key, model, layer.ple_key, arena.ple_emb, count) and
                ops.dense(arena.ple_val, model, layer.ple_value, arena.ple_emb, count) and
                c.ds4_gpu_qwen4_ple_gate_tensor(arena.ple_gated, arena.ple_normed, arena.R, arena.ple_key, arena.ple_val, model.map, model.size, layer.ple_norm_key.*.abs_offset, layer.ple_norm_query.*.abs_offset, layer.ple_norm_conv.*.abs_offset, count, shape.n_embd, shape.n_hc, shape.rms_eps) != 0;
            for (rows) |*row| {
                ok = ok and c.ds4_gpu_qwen4_ple_conv_tensor(row.graph.R, row.graph.ple_gated, row.graph.ple_normed, row.graph.ple_hist, model.map, model.size, layer.ple_conv.*.abs_offset, layer.ple_conv.*.type, 1, shape.n_embd * shape.n_hc, shape.n_ple_conv, shape.n_ple_ngram, null, 0, null, 1) != 0;
            }
        }
        profile.mark(1);
        ok = ok and ops.hcMix(arena, model, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, count);
        profile.mark(2);
        const is_linear = dimensions.isLinear(&arena.config, index);
        if (ok) ok = if (is_linear) linear(members, rows, arena, pool, model, layer, index, count) else attend(rows, arena, model, layer, index, count);
        profile.mark(if (is_linear) 3 else 4);
        ok = ok and c.ds4_gpu_qwen4_hc_combine_tensor(arena.R, arena.blk, arena.inj, count, shape.n_embd, shape.n_hc) != 0 and
            ops.hcMix(arena, model, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, count);
        profile.mark(5);
        ok = ok and moe.run(arena, model, layer, count);
        profile.mark(6);
    }
    ok = ok and ops.hcMix(arena, model, weights.output_hc_norm, weights.output_hc_down, weights.output_hc_up, null, count) and
        ops.dense(arena.batch_logits, model, weights.output, arena.mixed, count);
    if (profile.enabled) _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 batch(T=%u) ms: stage %.1f ple %.1f hc_attn %.1f gdn %.1f attn %.1f hc_ffn %.1f moe %.1f\n", count, profile.elapsed[0] * 1e3, profile.elapsed[1] * 1e3, profile.elapsed[2] * 1e3, profile.elapsed[3] * 1e3, profile.elapsed[4] * 1e3, profile.elapsed[5] * 1e3, profile.elapsed[6] * 1e3);
    return ok;
}

fn stageRagged(members: []const Member, count: u32, model: *const Model, weights: *const Weights, arena: *Graph, ids: []u32) bool {
    const shape = &arena.config.shape;
    const embd: usize = shape.n_embd;
    const width = embd * shape.n_hc;
    const heads = arena.config.ple.n_heads;
    for (members) |member| {
        const graph: *Graph = member.graph;
        for (0..member.n) |token_row| {
            const row = @as(usize, member.row0) + token_row;
            const token = member.tokens[token_row];
            const destination = arena.host_row[row * width ..][0..width];
            if (!embeddings.row(model, weights.token_embd, @intCast(token), destination[0..embd])) return false;
            for (1..shape.n_hc) |stream| @memcpy(destination[stream * embd ..][0..embd], destination[0..embd]);
            const position = graph.pos + @as(u32, @intCast(token_row));
            const rope = input.ropePosition(&.{}, position, &graph.mrope_delta);
            const position3 = [4]u32{ rope[0], rope[1], rope[2], 0 };
            if (c.ds4_gpu_tensor_write(graph.pos3, @as(u64, position) * 16, &position3, 16) == 0) return false;
            ngrams.step(&graph.config, token, &graph.ple_prev, ids[row * heads ..][0..heads]);
            if (token_row == 0 and graph.snap_after_first) {
                graph.snap_ple_prev = graph.ple_prev;
                graph.snap_pos = graph.pos + 1;
                graph.snap_mrope_delta = graph.mrope_delta;
            }
        }
    }
    return c.ds4_gpu_tensor_write(arena.R, 0, arena.host_row, @as(u64, count) * width * @sizeOf(f32)) != 0;
}

fn raggedLinear(members: []const Member, rows: []const Row, arena: *Graph, model: *const Model, layer: *const Layer, index: u32, count: u32) bool {
    const shape = &arena.config.shape;
    if (!(ops.dense(arena.qkv, model, layer.lin_qkv, arena.mixed, count) and ops.dense(arena.z, model, layer.lin_gate, arena.mixed, count) and
        ops.dense(arena.ga, model, layer.lin_alpha, arena.mixed, count) and ops.dense(arena.gb, model, layer.lin_beta, arena.mixed, count))) return false;
    var entries: [maximum]c.ds4_gpu_qwen4_gdn_row = undefined;
    for (members, rows, 0..) |member, *row, slot| {
        const graph = &row.graph;
        const save = graph.snap_after_first and graph.snap_lin_state[index] != null and member.n == 2;
        entries[slot] = .{
            .state = graph.layer_lin_state[index],
            .hist = graph.layer_lin_hist[index],
            .snap_state = if (save) graph.snap_lin_state[index] else null,
            .snap_hist = if (save) graph.snap_lin_hist[index] else null,
            .row0 = member.row0,
            .n_tok = member.n,
        };
    }
    const first = @as(u64, index) * maximum;
    const sessions: u32 = @intCast(members.len);
    const conv: u32 = @intCast(dimensions.convWidth(shape));
    return c.ds4_gpu_qwen4_gdn_rows_stage(arena.batch_gdn_rows, first, &entries, sessions) != 0 and
        c.ds4_gpu_qwen4_conv_stream_rows2_tensor(arena.qkv, model.map, model.size, layer.lin_conv.*.abs_offset, arena.batch_gdn_rows, first, &entries, sessions, count, conv, shape.n_lin_conv, conv, 1) != 0 and
        c.ds4_gpu_qwen4_gdn_prep_tensor(arena.qkv, arena.ga, arena.gb, model.map, model.size, layer.lin_a.*.abs_offset, layer.lin_dt_bias.*.abs_offset, count, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim) != 0 and
        c.ds4_gpu_qwen4_gdn_scan_rows2_tensor(arena.lin_o, arena.qkv, arena.ga, arena.gb, arena.batch_gdn_rows, first, &entries, sessions, count, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim, conv, @intCast(dimensions.valueWidth(shape))) != 0 and
        c.ds4_gpu_qwen4_gdn_out_tensor(arena.lin_o, arena.z, model.map, model.size, layer.lin_norm.*.abs_offset, count, shape.n_lin_v_head, shape.n_lin_head_dim, shape.rms_eps) != 0 and ops.dense(arena.blk, model, layer.lin_out, arena.lin_o, count);
}

fn attentionProjections(arena: *Graph, model: *const Model, layer: *const Layer, count: u32) bool {
    return ops.dense(arena.qg, model, layer.attn_q, arena.mixed, count) and ops.dense(arena.kp, model, layer.attn_k, arena.mixed, count) and
        ops.dense(arena.vp, model, layer.attn_v, arena.mixed, count) and ops.dense(arena.iq, model, layer.indexer_q_proj, arena.mixed, count) and
        ops.dense(arena.ik, model, layer.indexer_k_proj, arena.mixed, count);
}

fn attentionEntry(graph: *const Graph, layer: u32, position: u32, sparse_at: u32) c.ds4_gpu_qwen4_attn_row {
    return .{
        .k_cache = graph.layer_k_cache[layer],
        .v_cache = graph.layer_v_cache[layer],
        .ik_cache = graph.layer_ik_cache[layer],
        .block_key = graph.layer_block_key[layer],
        .pos3 = graph.pos3,
        .pos = position,
        .use_sel = @intFromBool(position >= sparse_at),
    };
}

pub fn encodeRagged(members: []const Member, count: u32, arena: *Graph, model: *const Model, weights: *const Weights) bool {
    if (members.len == 0 or members.len > maximum or count == 0 or count > maximum) return false;
    const rows = allocator.alloc(Row, members.len) catch c.ds4_engine_model_die("out of memory");
    defer allocator.free(rows);
    var built: usize = 0;
    defer for (rows[0..built]) |*row| row.deinit();
    for (members, rows) |member, *row| {
        row.* = Row.init(member.graph, arena, member.row0, member.n, true) catch return false;
        built += 1;
    }
    const ids = allocator.alloc(u32, @as(usize, count) * c.DS4_MAX_PLE_HEADS) catch c.ds4_engine_model_die("out of memory");
    defer allocator.free(ids);
    if (!ensureScratch(arena, count) or !stageRagged(members, count, model, weights, arena, ids)) return false;
    const shape = &arena.config.shape;
    var staged_ngrams = false;
    var ok = true;
    for (0..dimensions.trunkLayers(&arena.config)) |offset| {
        if (!ok) break;
        const index: u32 = @intCast(offset);
        const layer = &weights.layer[index];
        const ple = dimensions.isPle(&arena.config, index);
        if (!staged_ngrams and ple) {
            staged_ngrams = true;
            ok = (ops.present("DS4_QWEN4_NO_NGRAM_OVERLAP") or c.ds4_gpu_flush_commands() != 0) and
                stageNgrams(count, model, arena, ids[0 .. @as(usize, count) * arena.config.ple.n_heads]);
            if (!ok) break;
        }
        if (ple) {
            ok = ops.dense(arena.ple_key, model, layer.ple_key, arena.ple_emb, count) and
                ops.dense(arena.ple_val, model, layer.ple_value, arena.ple_emb, count) and
                c.ds4_gpu_qwen4_ple_gate_tensor(arena.ple_gated, arena.ple_normed, arena.R, arena.ple_key, arena.ple_val, model.map, model.size, layer.ple_norm_key.*.abs_offset, layer.ple_norm_query.*.abs_offset, layer.ple_norm_conv.*.abs_offset, count, shape.n_embd, shape.n_hc, shape.rms_eps) != 0;
            for (members, rows) |member, *row| {
                const graph = &row.graph;
                ok = ok and c.ds4_gpu_qwen4_ple_conv_tensor(graph.R, graph.ple_gated, graph.ple_normed, graph.ple_hist, model.map, model.size, layer.ple_conv.*.abs_offset, layer.ple_conv.*.type, member.n, shape.n_embd * shape.n_hc, shape.n_ple_conv, shape.n_ple_ngram, if (graph.snap_after_first) graph.snap_ple_hist else null, 0, null, 1) != 0;
            }
        }
        ok = ok and ops.hcMix(arena, model, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, count);
        if (ok and dimensions.isLinear(&arena.config, index)) {
            ok = raggedLinear(members, rows, arena, model, layer, index, count);
        } else if (ok) {
            ok = attentionProjections(arena, model, layer, count);
            var entries: [maximum]c.ds4_gpu_qwen4_attn_row = undefined;
            const sparse_at = (arena.k_blocks + 1) * 4 - 1;
            for (members, rows) |member, *row| {
                for (0..member.n) |token_row| {
                    const position = row.graph.pos + @as(u32, @intCast(token_row));
                    if (position >= row.graph.ctx_cap) ok = false;
                    entries[member.row0 + token_row] = attentionEntry(&row.graph, index, position, sparse_at);
                }
            }
            ok = ok and attentionEntries(entries[0..count], arena, model, layer, index) and ops.dense(arena.blk, model, layer.attn_output, arena.attn_o, count);
        }
        ok = ok and c.ds4_gpu_qwen4_hc_combine_tensor(arena.R, arena.blk, arena.inj, count, shape.n_embd, shape.n_hc) != 0 and
            ops.hcMix(arena, model, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, count) and moe.run(arena, model, layer, count);
    }
    return ok and ops.hcMix(arena, model, weights.output_hc_norm, weights.output_hc_down, weights.output_hc_up, null, count) and
        ops.dense(arena.batch_logits, model, weights.output, arena.mixed, count);
}

/// Reuse idle trunk MoE scratch for the batched predictor. Rejected rows may
/// execute, but only each session's last committed row is scored for its draft.
pub fn drafts(members: []const Member, committed: []const u32, parents: []const c_int, positions: []const u32, count: u32, arena: *Graph, model: *const Model, weights: *const Weights, output: []c_int) bool {
    if (members.len == 0 or members.len > maximum or count == 0 or count > maximum) return false;
    const shape = &arena.config.shape;
    const index = shape.n_layer - 1;
    const layer = &weights.layer[index];
    const embd: u64 = shape.n_embd;
    const hc: u64 = shape.n_hc;
    const row_bytes = hc * embd * @sizeOf(f32);
    var ids: [maximum]c_int = undefined;
    var entries: [maximum]c.ds4_gpu_qwen4_attn_row = undefined;
    const sparse_at = (arena.k_blocks + 1) * 4 - 1;
    for (members, committed, parents, positions) |member, accepted, parent, position| {
        const graph: *const Graph = member.graph;
        for (0..member.n) |token_row| {
            const row = member.row0 + token_row;
            ids[row] = if (token_row + 1 < accepted) member.tokens[token_row + 1] else parent;
            const absolute = position + @as(u32, @intCast(token_row));
            if (absolute >= graph.ctx_cap) return false;
            entries[row] = attentionEntry(graph, index, absolute, sparse_at);
        }
    }
    for (ids[0..count], 0..) |token, row| {
        if (!embeddings.row(model, weights.token_embd, @intCast(token), arena.host_row[row * embd ..][0..@intCast(embd)])) return false;
    }
    if (c.ds4_gpu_tensor_write(arena.batch_head_x, 0, arena.host_row, @as(u64, count) * embd * @sizeOf(f32)) == 0 or !ops.beginCommands()) return false;
    var ok = true;
    for (0..count) |row| {
        if (!ok) break;
        const e = ops.rowsView(arena.batch_head_x, @intCast(row), 1, embd);
        defer c.ds4_gpu_tensor_free(e);
        const residual = ops.rowsView(arena.R, @intCast(row), 1, hc * embd);
        defer c.ds4_gpu_tensor_free(residual);
        const cat = ops.rowsView(arena.part, @intCast(row), 1, (hc + 1) * 2 * embd);
        defer c.ds4_gpu_tensor_free(cat);
        ok = e != null and residual != null and cat != null and c.ds4_gpu_qwen4_mtp_stage_tensor(cat, e, residual, model.map, model.size, layer.nextn_enorm.*.abs_offset, layer.nextn_hnorm.*.abs_offset, shape.n_embd, shape.n_hc, shape.rms_eps) != 0;
    }
    ok = ok and ops.dense(arena.mid, model, layer.nextn_eh_proj, arena.part, count * (shape.n_hc + 1));
    for (0..count) |row| {
        if (!ok) break;
        const projection = ops.rowsView(arena.mid, @intCast(row), 1, (hc + 1) * embd);
        defer c.ds4_gpu_tensor_free(projection);
        const residual = ops.rowsView(arena.R, @intCast(row), 1, hc * embd);
        defer c.ds4_gpu_tensor_free(residual);
        ok = projection != null and residual != null and c.ds4_gpu_qwen4_mtp_combine_tensor(residual, projection, shape.n_embd, shape.n_hc) != 0;
    }
    ok = ok and ops.hcMix(arena, model, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, count) and
        attentionProjections(arena, model, layer, count) and attentionEntries(entries[0..count], arena, model, layer, index) and
        ops.dense(arena.blk, model, layer.attn_output, arena.attn_o, count) and
        c.ds4_gpu_qwen4_hc_combine_tensor(arena.R, arena.blk, arena.inj, count, shape.n_embd, shape.n_hc) != 0 and
        ops.hcMix(arena, model, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, count) and moe.run(arena, model, layer, count);
    for (members, committed, 0..) |member, accepted, session| {
        if (!ok) break;
        ok = c.ds4_gpu_tensor_copy(arena.part, @as(u64, session) * row_bytes, arena.R, @as(u64, member.row0 + accepted - 1) * row_bytes, row_bytes) != 0;
    }
    const residual = arena.R;
    const mixed = arena.mixed;
    arena.R = arena.part;
    arena.mixed = arena.batch_head_x;
    ok = ok and ops.hcMix(arena, model, layer.nextn_hc_head_norm, layer.nextn_hc_head_down, layer.nextn_hc_head_up, null, @intCast(members.len));
    arena.R = residual;
    arena.mixed = mixed;
    ok = ok and ops.dense(arena.batch_logits, model, weights.output, arena.batch_head_x, @intCast(members.len));
    for (members, 0..) |member, session| {
        if (!ok) break;
        const graph: *Graph = member.graph;
        const logits = ops.rowsView(arena.batch_logits, @intCast(session), 1, shape.n_vocab);
        defer c.ds4_gpu_tensor_free(logits);
        ok = logits != null and c.ds4_gpu_qwen4_argmax_tensor(graph.mtp_argmax, graph.mtp_argmax_tmp, logits, shape.n_vocab) != 0;
    }
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    for (members, committed, positions, 0..) |member, accepted, position, session| {
        if (!ok) break;
        const graph: *Graph = member.graph;
        var token: i32 = -1;
        ok = c.ds4_gpu_tensor_read(graph.mtp_argmax, 0, &token, @sizeOf(i32)) != 0 and token >= 0 and token < shape.n_vocab;
        output[session] = token;
        graph.mtp_pos = position + accepted;
        graph.mtp_last_rows = 0;
    }
    return ok;
}
