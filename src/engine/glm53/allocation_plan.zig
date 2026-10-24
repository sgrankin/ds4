//! Validate the trained model layout and derive graph geometry before any
//! tensor allocation. Platform working-set admission stays with the caller.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Planning = @import("planning.zig");
const Memory = @import("memory.zig");
const Weights = @import("weights.zig");
const Graph = c.ds4_glm_gpu_graph;

fn gateType(kind: u32) bool {
    return switch (kind) {
        c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_MXFP4 => true,
        else => false,
    };
}

fn downType(kind: u32) bool {
    return gateType(kind) or kind == c.DS4_TENSOR_Q6_K;
}

fn layout(graph: *Graph, model: *const c.ds4_model, weights: *const c.ds4_weights, first: u32, requested_last: u32, require_embedding: bool, require_output: bool) bool {
    const shape = &graph.config.shape;
    if (shape.variant != c.DS4_VARIANT_GLM53 or shape.n_layer <= shape.n_nextn_predict or shape.n_layer > c.DS4_MAX_LAYER or shape.n_rot >= shape.n_key_mla) {
        _ = c.fputs("ds4: GLM Metal graph found unsupported layer/key dimensions\n", c.__stderrp);
        return false;
    }
    const normal = Config.trunkLayers(&graph.config);
    const last = if (requested_last == std.math.maxInt(u32)) normal - 1 else requested_last;
    if (first > last or last >= normal) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM Metal graph found invalid layer slice %u:%u for %u normal layers\n", first, last, normal);
        return false;
    }
    // Reuse the model owner's typed weight validation. The graph adds the
    // narrower routed-kernel formats below; descriptor loading also serves CPU.
    var error_text: [512]u8 = undefined;
    const binding = Weights.Binding{ .table = @constCast(weights), .model = model, .config = &graph.config, .errors = &error_text };
    binding.validate(first, last, require_embedding, require_output) catch {
        _ = c.fprintf(c.__stderrp, "ds4: GLM-5.3 graph layout: %s\n", @as([*:0]const u8, @ptrCast(&error_text)));
        return false;
    };
    graph.has_token_embd = weights.token_embd != null;
    graph.has_output_head = weights.output_norm != null and weights.output != null;
    graph.glm53 = true;
    graph.normal_layers = normal;
    graph.layer_start = first;
    graph.layer_end = last;
    graph.layer_count = last - first + 1;
    graph.q_dim = @as(u64, shape.n_head) * shape.n_key_mla;
    graph.q_nope = shape.n_key_mla - shape.n_rot;
    graph.heads_dim = @as(u64, shape.n_head) * shape.n_value_mla;
    graph.dense_hidden_max = shape.n_ff_exp;
    graph.kv_raw_dim = 0;
    graph.generic_routed_moe = false;
    for (first..last + 1) |index| {
        const layer = &weights.layer[index];
        if (!Config.isKda(&graph.config, @intCast(index))) graph.kv_raw_dim = @max(graph.kv_raw_dim, layer.attn_kv_a_mqa.*.dim[1]);
        if (index < shape.n_leading_dense) {
            graph.dense_hidden_max = @max(graph.dense_hidden_max, layer.ffn_gate.*.dim[1]);
        } else {
            const gate = layer.ffn_gate_exps.*.type;
            if (gate != layer.ffn_up_exps.*.type or !gateType(gate) or !downType(layer.ffn_down_exps.*.type)) {
                _ = c.fprintf(c.__stderrp, "ds4: GLM Metal graph found unsupported routed quant layout in layer %u\n", @as(u32, @intCast(index)));
                return false;
            }
            if (gate == c.DS4_TENSOR_IQ2_XXS) graph.generic_routed_moe = true;
        }
    }
    graph.ffn_mid_elems = @max(graph.dense_hidden_max, @as(u64, shape.n_expert_used) * shape.n_ff_exp);
    // Preserve the original graph guard, including KDA-only slice rejection:
    // the shared graph currently requires at least one DSA projection layout.
    return graph.kv_raw_dim >= @as(u64, shape.n_kv_lora) + shape.n_rot;
}

pub fn initialize(graph: *Graph, model: *const c.ds4_model, weights: *const c.ds4_weights, first: u32, last: u32, require_embedding: bool, require_output: bool) bool {
    if (!layout(graph, model, weights, first, last, require_embedding, require_output)) return false;
    graph.ctx_cap = Planning.fullAttentionCap(graph.ctx_size, graph.ssd_streaming);
    // There is no production expanded-KV option. Both dense and sparse DSA use
    // the shared latent cache; full-KV records remain legacy/oracle state.
    graph.full_kv_cache = false;
    graph.compact_cache_cap = graph.ctx_size;
    graph.indexed_prefill_cap = if (graph.compact_cache_cap != 0) Planning.indexedChunk(graph.compact_cache_cap) else 0;
    graph.indexed_prefill_score_cap = Planning.scoreRows(graph.indexed_prefill_cap, graph.compact_cache_cap);
    graph.indexer_full_layers = Memory.indexerLayers(&graph.config, graph.layer_start, graph.layer_end);
    if (graph.ctx_size > graph.ctx_cap) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM session ctx=%u (model max=%u); prefill/work cap=%u; dense attention limit=%u\n", graph.ctx_size, @as(u32, @intCast(@min(graph.config.shape.rope_orig_ctx, std.math.maxInt(u32)))), graph.ctx_cap, Planning.denseLimit(graph));
    }
    return true;
}
