//! Preserve the existing expert-cache warm-start policy. Router tapes are
//! captured on device; seeding reads only selected IDs and publishes through
//! the storage cache's existing GPU-copy/selected-load primitives.
const c = @import("gguf_c");
const std = @import("std");
const Spans = @import("../model_spans.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const allocator = std.heap.c_allocator;
const maximum_rows = 64;

pub fn reset(graph: *Graph) void {
    graph.prefill_seed_tokens = 0;
    @memset(&graph.prefill_seed_layer_captured, false);
}

fn seedRows(graph: *const Graph) u32 {
    if (!graph.ssd_streaming or c.getenv("DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED") == null) return 0;
    const env = c.getenv("DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K");
    if (env == null or env[0] == 0) return 1;
    var end: [*c]u8 = null;
    const value = c.strtoul(env, &end, 10);
    return if (end != env and end[0] == 0) @intCast(@min(value, maximum_rows)) else 1;
}

fn table(model: *const Model, layer: *const Layer, index: u32, experts: u32, gate_bytes: u64, down_bytes: u64) c.ds4_gpu_stream_expert_table {
    return .{ .model_map = model.map, .model_size = model.size, .layer = index, .n_total_expert = experts, .gate_offset = layer.ffn_gate_exps.*.abs_offset, .up_offset = layer.ffn_up_exps.*.abs_offset, .down_offset = layer.ffn_down_exps.*.abs_offset, .gate_expert_bytes = gate_bytes, .down_expert_bytes = down_bytes };
}

fn expertBytes(weight: *const c.ds4_tensor) ?u64 {
    var row: u64 = 0;
    if (!c.ds4_gguf_tensor_nbytes(weight.type, weight.dim[0], &row)) return null;
    return std.math.mul(u64, row, weight.dim[1]) catch null;
}

fn capture(graph: *Graph, layer: u32, rows: u32) bool {
    const count = @min(seedRows(graph), rows);
    if (count == 0) return true;
    const used = graph.config.shape.n_expert_used;
    if (graph.prefill_seed_router_selected == null or graph.batch_router_selected == null or layer >= graph.config.shape.n_layer or layer >= c.DS4_MAX_LAYER or used == 0 or used > c.DS4_MAX_EXPERT_USED) return false;
    const source = @as(u64, rows - count) * used * @sizeOf(i32);
    const destination = @as(u64, layer) * maximum_rows * used * @sizeOf(i32);
    if (c.ds4_gpu_tensor_copy(graph.prefill_seed_router_selected, destination, graph.batch_router_selected, source, @as(u64, count) * used * @sizeOf(i32)) == 0) return false;
    graph.prefill_seed_tokens = count;
    graph.prefill_seed_layer_captured[layer] = true;
    return true;
}

pub fn seedLayer(graph: *Graph, model: *const Model, weights: *const Weights, layer_weights: *const Layer, index: u32, rows: u32, gate_bytes: u64, down_bytes: u64, full_layer: bool) bool {
    if (!capture(graph, index, rows)) return false;
    const shape = &graph.config.shape;
    if (!full_layer or rows == 0 or graph.layer_count == 0 or graph.tp_world >= 2 or graph.layer_start != 0 or graph.layer_count != graph.normal_layers or
        c.getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED") != null or
        !Spans.uniform(weights, &graph.config.shape, index) or !c.ds4_native_glm_seed_layer_expected(graph, weights, layer_weights, index)) return true;
    const routed_layers = if (graph.layer_count > shape.n_leading_dense) graph.layer_count - shape.n_leading_dense else graph.layer_count;
    const target = @min(c.ds4_gpu_stream_expert_cache_configured_count() / routed_layers, 128);
    if (target == 0) return true;
    const count = @as(usize, rows) * shape.n_expert_used;
    const selected = allocator.alloc(i32, count) catch return false;
    defer allocator.free(selected);
    const had_commands = c.ds4_gpu_commands_active() != 0;
    var ok = !had_commands or c.ds4_gpu_end_commands() != 0;
    if (ok) ok = c.ds4_gpu_tensor_read(graph.batch_router_selected, 0, selected.ptr, selected.len * @sizeOf(i32)) != 0;
    var frequency = std.mem.zeroes([c.DS4_MAX_EXPERT]u32);
    const recent_start = @as(usize, rows - @min(rows, 32)) * shape.n_expert_used;
    if (ok) for (selected, 0..) |expert, offset| {
        if (expert < 0 or expert >= shape.n_expert) {
            ok = false;
            break;
        }
        const id: usize = @intCast(expert);
        frequency[id] += 1;
        if (offset >= recent_start) frequency[id] += @intCast(count);
    };
    var experts: [128]i32 = undefined;
    var priority: [128]u32 = undefined;
    var total: u32 = 0;
    while (ok and total < target) {
        var best: u32 = 0;
        for (1..shape.n_expert) |id| if (frequency[id] > frequency[best]) {
            best = @intCast(id);
        };
        if (frequency[best] == 0) break;
        experts[total] = @intCast(best);
        // Rank is a preload preference, not thousands of observed routes.
        priority[total] = 1 + @as(u32, @intCast(31 * @as(u64, target - total) / target));
        total += 1;
        frequency[best] = 0;
    }
    if (ok and total != 0) {
        ok = c.ds4_gpu_begin_commands() != 0;
        const mapping = table(model, layer_weights, index, shape.n_expert, gate_bytes, down_bytes);
        if (ok) ok = c.ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(&mapping, &experts, &priority, total) != 0;
    }
    if (had_commands and c.ds4_gpu_commands_active() == 0) {
        if (c.ds4_gpu_begin_commands() == 0) ok = false;
    } else if (!had_commands and c.ds4_gpu_commands_active() != 0) {
        if (c.ds4_gpu_end_commands() == 0) ok = false;
    }
    return ok;
}

pub fn finish(graph: *Graph, model: *const Model, weights: *const Weights) bool {
    const rows = graph.prefill_seed_tokens;
    if (seedRows(graph) == 0 or rows == 0) return true;
    const shape = &graph.config.shape;
    if (graph.prefill_seed_router_selected == null or rows > maximum_rows or shape.n_layer > c.DS4_MAX_LAYER or shape.n_expert_used == 0 or shape.n_expert_used > c.DS4_MAX_EXPERT_USED) return false;
    var any = false;
    for (graph.layer_start..graph.layer_end + 1) |index| any = any or graph.prefill_seed_layer_captured[index];
    if (!any) return true;
    const selected = allocator.alloc(i32, @as(usize, shape.n_layer) * maximum_rows * shape.n_expert_used) catch return false;
    defer allocator.free(selected);
    if (c.ds4_gpu_tensor_read(graph.prefill_seed_router_selected, 0, selected.ptr, selected.len * @sizeOf(i32)) == 0) return false;
    const profile = c.getenv("DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE") != null;
    const started = if (profile) seconds() else 0;
    var seeded_layers: u32 = 0;
    var seeded_rows: u32 = 0;
    for (graph.layer_start..graph.layer_end + 1) |index| {
        const layer_weights = &weights.layer[index];
        if (!graph.prefill_seed_layer_captured[index] or !c.ds4_native_glm_seed_layer_expected(graph, weights, layer_weights, @intCast(index))) continue;
        const gate_bytes = expertBytes(layer_weights.ffn_gate_exps) orelse return false;
        const down_bytes = expertBytes(layer_weights.ffn_down_exps) orelse return false;
        const mapping = table(model, layer_weights, @intCast(index), shape.n_expert, gate_bytes, down_bytes);
        for (0..rows) |row| {
            const offset = (index * maximum_rows + row) * shape.n_expert_used;
            if (c.ds4_gpu_stream_expert_cache_seed_selected(&mapping, selected.ptr + offset, shape.n_expert_used) == 0) return false;
            seeded_rows += 1;
        }
        seeded_layers += 1;
    }
    if (profile)
        _ = c.fprintf(c.__stderrp, "ds4: GLM streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n", rows, seeded_layers, seeded_rows, (seconds() - started) * 1000);
    return true;
}

fn seconds() f64 {
    var timestamp: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &timestamp);
    return @as(f64, @floatFromInt(timestamp.tv_sec)) + @as(f64, @floatFromInt(timestamp.tv_nsec)) / 1.0e9;
}
