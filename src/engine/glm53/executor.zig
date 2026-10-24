//! Single-token Metal execution. Model arithmetic and layer traversal stay
//! here; C's temporary storage leaves retain only mapping/cache policy.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Ops = @import("ops.zig");
const Kda = @import("kda.zig");
const Attention = @import("attention.zig");
const Moe = @import("moe.zig");
const Head = @import("head.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;

fn begin() bool {
    return c.ds4_gpu_commands_active() != 0 or c.ds4_gpu_begin_commands() != 0;
}

fn end() bool {
    return c.ds4_gpu_commands_active() == 0 or c.ds4_gpu_end_commands() != 0;
}

fn embed(graph: *Graph, model: *const Model, weights: *const Weights) bool {
    const shape = &graph.config.shape;
    const weight = weights.token_embd;
    const ok = if (weight.*.type == c.DS4_TENSOR_BF16)
        c.ds4_gpu_glm53_embedding_bf16(graph.cur, model.map, model.size, weight.*.abs_offset, graph.prefill_tokens, 1, shape.n_embd, shape.n_vocab) != 0
    else
        c.ds4_gpu_embed_tokens_quant_tensor(graph.cur, graph.prefill_tokens, model.map, model.size, weight.*.abs_offset, weight.*.type, shape.n_vocab, 1, shape.n_embd) != 0;
    return ok and c.ds4_gpu_repeat_hc_tensor(graph.hc_cur, graph.cur, shape.n_embd, shape.n_hc) != 0;
}

fn prepareFfn(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32) bool {
    const shape = &graph.config.shape;
    c.ds4_native_graph_debug_tensor("attn_out", graph.attn_out, shape.n_embd, layer, position);
    if (graph.directional_steering_attn_scale != 0) {
        if (graph.directional_steering_dirs_by_tier[0] == null or
            c.ds4_gpu_directional_steering_project_tensor(graph.attn_out, graph.directional_steering_dirs_by_tier[0], layer, shape.n_embd, 1, graph.directional_steering_attn_scale) == 0) return false;
    }
    return c.ds4_gpu_hc_expand_tensor(graph.hc_after_attn, graph.attn_out, graph.hc_cur, graph.hc_post, graph.hc_comb, shape.n_embd, shape.n_hc) != 0 and
        Ops.hcPre(false, graph, model, weights.hc_ffn_fn, weights.hc_ffn_scale, weights.hc_ffn_base, weights.ffn_norm, graph.hc_after_attn, graph.after_attn, graph.ffn_norm, graph.hc_flat, graph.hc_mix, graph.hc_split, 1);
}

fn output(graph: *Graph, model: *const Model, weights: *const Weights, static_map: bool) bool {
    if (graph.ssd_streaming and !static_map and !c.ds4_native_glm_map_output(graph, model, weights)) return false;
    return begin() and Head.encode(graph, model, weights);
}

fn run(graph: *Graph, model: *const Model, weights: *const Weights, token: i32, input_hc: ?[*]const f32, position: u32, output_hc: ?[*]f32, logits: ?[*]f32, deferred: bool, ablate: u32) bool {
    const shape = &graph.config.shape;
    if (!graph.glm53 or graph.placement != null or graph.tp_world > 1 or graph.imatrix != null or
        token < 0 or token >= shape.n_vocab or graph.layer_count == 0 or position >= graph.ctx_size or
        (deferred and (graph.ssd_streaming or c.ds4_gpu_commands_active() == 0))) return false;
    if ((input_hc == null and !graph.has_token_embd) or (logits != null and !graph.has_output_head)) return false;
    const indexed = graph.compact_cache_cap != 0 and (!graph.full_kv_cache or position >= graph.ctx_cap);
    if ((!indexed and position >= graph.ctx_cap) or (graph.compact_cache_cap != 0 and position >= graph.compact_cache_cap)) return false;
    var flush_interval: u32 = 0;
    if (logits != null and !deferred) {
        flush_interval = if (graph.config.decode_flush_interval < 0)
            (if (indexed) @as(u32, 4) else 32)
        else
            @intCast(graph.config.decode_flush_interval);
        flush_interval = @min(flush_interval, graph.layer_count);
    }

    if (input_hc == null and c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, &token, @sizeOf(i32)) == 0) return false;
    if (input_hc) |hidden| {
        if (c.ds4_gpu_tensor_write(graph.hc_cur, 0, hidden, @as(u64, Config.hcWidth(&graph.config)) * @sizeOf(f32)) == 0) return false;
    }
    var static_map = false;
    if (!c.ds4_native_glm_map_decode_begin(graph, model, weights, input_hc != null, &static_map) or !begin()) return false;
    if (input_hc == null and !embed(graph, model, weights)) return false;
    const sync_each_layer = graph.ssd_streaming and !static_map;
    if (sync_each_layer and !end()) return false;

    var selection = std.mem.zeroes(c.ds4_glm_decode_selection);
    for (graph.layer_start..graph.layer_end + 1) |index| {
        const layer: u32 = @intCast(index);
        const layer_weights = &weights.layer[layer];
        if (graph.ssd_streaming) {
            if (!static_map and !c.ds4_native_glm_map_decode_layer(graph, model, weights, layer)) return false;
            if (!begin()) return false;
        }
        if (!Ops.hcPre(false, graph, model, layer_weights.hc_attn_fn, layer_weights.hc_attn_scale, layer_weights.hc_attn_base, layer_weights.attn_norm, graph.hc_cur, graph.cur, graph.attn_norm, graph.hc_flat, graph.hc_mix, graph.hc_split, 1)) return false;
        const attention_ok = if (Config.isKda(&graph.config, layer))
            Kda.attention(false, graph, model, layer_weights, layer, position, 1, graph.attn_out)
        else
            Attention.decode(graph, model, layer_weights, layer, position, indexed, &selection, ablate);
        if (!attention_ok or !prepareFfn(graph, model, layer_weights, layer, position) or
            !Moe.decode(graph, model, layer_weights, layer, position, ablate)) return false;
        std.mem.swap(?*c.ds4_gpu_tensor, &graph.hc_cur, &graph.hc_next);

        const completed = layer - graph.layer_start + 1;
        if (!graph.ssd_streaming and flush_interval != 0 and layer < graph.layer_end and completed % flush_interval == 0)
            if (c.ds4_gpu_flush_commands() == 0) return false;
        if (sync_each_layer and !end()) return false;
    }

    // Indexed output merges into the last batch when it can; the full-KV
    // fallback retains its separate output-head submission below.
    const merged_head = logits != null and (indexed or deferred);
    if (merged_head) {
        if (!output(graph, model, weights, static_map)) return false;
        if (graph.ssd_streaming and !end()) return false;
    }
    if (!graph.ssd_streaming and !deferred and !end()) return false;
    if (!deferred) {
        if (output_hc) |hidden| {
            if (c.ds4_gpu_tensor_read(graph.hc_cur, 0, hidden, @as(u64, Config.hcWidth(&graph.config)) * @sizeOf(f32)) == 0) return false;
        }
        if (logits) |values| {
            if (!merged_head and (!output(graph, model, weights, static_map) or !end())) return false;
            if (c.ds4_gpu_tensor_read(graph.logits, 0, values, @as(u64, shape.n_vocab) * @sizeOf(f32)) == 0) return false;
        }
    }
    return logits != null or !graph.ssd_streaming or end();
}

pub fn forward(graph: *Graph, model: *const Model, weights: *const Weights, token_id: i32, input_hc: ?[*]const f32, position: u32, output_hc: ?[*]f32, logits: ?[*]f32, deferred: bool, ablate: u32) bool {
    const ok = run(graph, model, weights, token_id, input_hc, position, output_hc, logits, deferred, ablate);
    if (!ok) _ = c.ds4_gpu_synchronize();
    return ok;
}
