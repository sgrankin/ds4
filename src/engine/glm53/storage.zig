//! GLM's bounded SSD mapping and expert-cache admission policy. Model spans
//! and their driver installation have one common native owner; this module
//! decides which view is needed and publishes the graph's current-map state.
const std = @import("std");
const c = @import("gguf_c");
const Spans = @import("../model_spans.zig");
const Cache = @import("../engine_cache_sizes.zig");
const Pagein = @import("../pagein.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Policy = c.ds4_streaming_policy;

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
const ExpertBytes = struct { gate: u64, down: u64 };
fn expertBytes(layer: *const Layer) ?ExpertBytes {
    if (layer.ffn_gate_exps == null or layer.ffn_down_exps == null) return null;
    const gate = Cache.rowBytes(layer.ffn_gate_exps);
    const down = Cache.rowBytes(layer.ffn_down_exps);
    if (gate == 0 or down == 0) return null;
    const gate_bytes = std.math.mul(u64, layer.ffn_gate_exps.*.dim[1], gate) catch return null;
    const down_bytes = std.math.mul(u64, layer.ffn_down_exps.*.dim[1], down) catch return null;
    return if (gate_bytes == 0 or down_bytes == 0) null else .{ .gate = gate_bytes, .down = down_bytes };
}
fn fitsCache(layer: *const Layer, count: u32) bool {
    const bytes = expertBytes(layer) orelse return false;
    return c.ds4_gpu_stream_expert_cache_budget_for_expert_size(bytes.gate, bytes.down) >= count;
}
fn addressCache(shape: *const c.ds4_shape, weights: *const Weights, layer: *const Layer, index: u32) bool {
    return Spans.glmAddrLayout(weights, shape, layer, index) and fitsCache(layer, shape.n_expert_used);
}
fn layerCache(graph: *const Graph, weights: ?*const Weights, layer: ?*const Layer, index: u32) bool {
    const all = weights orelse return false;
    const selected = layer orelse return false;
    const shape = &graph.config.shape;
    if (index < shape.n_leading_dense) return true;
    return addressCache(shape, all, selected, index) or
        (Spans.glmSelectedLayout(shape, selected, index) and fitsCache(selected, shape.n_expert_used));
}
pub fn selectedCache(optional_graph: ?*const Graph, layer: ?*const Layer, index: u32, policy: *const Policy) bool {
    const graph = optional_graph orelse return false;
    const selected = layer orelse return false;
    return graph.ssd_streaming and !Spans.residentLayer(&graph.config.shape, policy, selected, index) and
        layerCache(graph, graph.weights, selected, index);
}
pub fn mtpCache(optional_graph: ?*const Graph, layer: ?*const Layer, index: u32) bool {
    const graph = optional_graph orelse return false;
    // MTP preserves its existing cache choice even when ordinary layers use a
    // resident range. It has its own draft-layer mapping before execution.
    return graph.ssd_streaming and layerCache(graph, graph.weights, layer, index);
}
pub fn seedExpected(optional_graph: ?*const Graph, weights: ?*const Weights, layer: ?*const Layer, index: u32, policy: *const Policy) bool {
    const graph = optional_graph orelse return false;
    const all = weights orelse return false;
    const selected = layer orelse return false;
    if (!graph.ssd_streaming or graph.quality or selected.ffn_gate_exps == null or selected.ffn_up_exps == null or selected.ffn_down_exps == null) return false;
    const shape = &graph.config.shape;
    return !Spans.residentLayer(shape, policy, selected, index) and
        (Spans.glmSelectedLayout(shape, selected, index) or Spans.glmAddrLayout(all, shape, selected, index));
}

pub fn prefillCache(graph: *const Graph, weights: *const Weights, layer: *const Layer, index: u32, rows: u32) bool {
    const shape = &graph.config.shape;
    if (index < shape.n_leading_dense) return true;
    if (rows <= 1) return false;
    // Tiny eight-expert IQ2/IQ2 prefills use the selected-address pair path
    // only when the slab can contain every expert. Other prefills preserve the
    // existing uniform Q2/Q4 address-layout/budget requirement.
    if (graph.tp_world < 2 and layer.ffn_gate_exps != null and layer.ffn_up_exps != null and layer.ffn_down_exps != null and
        layer.ffn_gate_exps.*.type == c.DS4_TENSOR_IQ2_XXS and layer.ffn_up_exps.*.type == c.DS4_TENSOR_IQ2_XXS and layer.ffn_down_exps.*.type == c.DS4_TENSOR_IQ2_XXS and
        shape.n_expert_used == 8 and rows < 256 and
        !present("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") and !present("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION") and
        !present("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") and Spans.glmSelectedLayout(shape, layer, index) and fitsCache(layer, shape.n_expert)) return true;
    return addressCache(shape, weights, layer, index);
}

const Map = enum { token, output, layer, decode_layer, all_decode };
fn install(graph: *const Graph, model: *const Model, weights: *const Weights, policy: *const Policy, mode: Map, layer: u32) bool {
    const shape = &graph.config.shape;
    var spans = std.mem.zeroes(Spans.Vec);
    defer c.free(spans.v);
    const built = switch (mode) {
        .token => Spans.tokenOnly(weights, &spans),
        .output => Spans.outputOnly(weights, &spans),
        .layer => Spans.build(weights, shape, policy, layer, layer, layer == 0, false, .full, &spans),
        .decode_layer => Spans.build(weights, shape, policy, layer, layer, false, false, .runtime, &spans),
        .all_decode => Spans.build(weights, shape, policy, 0, std.math.maxInt(u32), true, true, .runtime, &spans),
    };
    const label: [*:0]const u8 = switch (mode) {
        .token => "token embedding",
        .output => "output head",
        .layer => "layer",
        .decode_layer => "decode layer",
        .all_decode => "static decode",
    };
    if (!built) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming could not build %s spans\n", label);
        return false;
    }
    return Pagein.install(model, &spans, label);
}
pub fn mapToken(optional_graph: ?*Graph, model: *const Model, weights: *const Weights) bool {
    const graph = optional_graph orelse return true;
    if (!graph.ssd_streaming) return true;
    graph.streaming_static_decode_map_current = false;
    const unused_policy = std.mem.zeroes(Policy);
    return install(graph, model, weights, &unused_policy, .token, 0);
}
pub fn mapOutput(optional_graph: ?*Graph, model: *const Model, weights: *const Weights) bool {
    const graph = optional_graph orelse return true;
    if (!graph.ssd_streaming) return true;
    graph.streaming_static_decode_map_current = false;
    const unused_policy = std.mem.zeroes(Policy);
    return install(graph, model, weights, &unused_policy, .output, 0);
}
pub fn mapBegin(graph: *Graph, model: *const Model, weights: *const Weights, from_hidden: bool, static_map: *bool, policy: *const Policy) bool {
    static_map.* = !from_hidden and graph.has_token_embd and graph.ssd_streaming and
        !present("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP") and Spans.staticSupported(weights, &graph.config.shape, policy);
    if (from_hidden) return true;
    if (static_map.*) {
        const cache_state = !present("DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE");
        if (!cache_state or !graph.streaming_static_decode_map_current) {
            if (!install(graph, model, weights, policy, .all_decode, 0)) return false;
            graph.streaming_static_decode_map_current = cache_state;
        }
        return true;
    }
    return mapToken(graph, model, weights);
}
pub fn mapLayer(optional_graph: ?*Graph, model: *const Model, weights: *const Weights, index: u32, policy: *const Policy) bool {
    const graph = optional_graph orelse return true;
    if (!graph.ssd_streaming) return true;
    graph.streaming_static_decode_map_current = false;
    var full = present("DS4_METAL_GLM_STREAMING_DECODE_FULL_LAYER_MAP") or present("DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP");
    if (!full and index < graph.config.shape.n_layer) {
        const layer = &weights.layer[index];
        full = Spans.residentLayer(&graph.config.shape, policy, layer, index) or !layerCache(graph, weights, layer, index);
    } else if (index >= graph.config.shape.n_layer) full = true;
    return install(graph, model, weights, policy, if (full) .layer else .decode_layer, index);
}
pub fn mapPrefill(optional_graph: ?*Graph, model: *const Model, weights: *const Weights, index: u32, rows: u32, full_layer: bool, policy: *const Policy) bool {
    const graph = optional_graph orelse return true;
    if (!graph.ssd_streaming) return true;
    graph.streaming_static_decode_map_current = false;
    const cached = !full_layer and index < graph.config.shape.n_layer and prefillCache(graph, weights, &weights.layer[index], index, rows);
    const trace = c.getenv("DS4_METAL_STREAMING_MAP_TRACE");
    if (trace != null and trace[0] != 0 and c.strcmp(trace, "0") != 0) {
        const mode: [*:0]const u8 = if (full_layer) "full-prefill" else if (cached) "decode-expert-cache" else "full-layer";
        _ = c.fprintf(c.__stderrp, "ds4: GLM SSD prefill map layer=%u tokens=%u mode=%s\n", index, rows, mode);
    }
    return install(graph, model, weights, policy, if (cached) .decode_layer else .layer, index);
}
