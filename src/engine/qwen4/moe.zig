//! Routed and shared experts. Large prefills amortize weights in tiled GEMMs;
//! small decode batches retain per-row arithmetic and the measured Metal
//! crossover/grouped paths. The final reduction also applies the HC combine.
const std = @import("std");
const c = @import("gguf_c");
const ops = @import("ops.zig");
const resources = @import("resources.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;

const Profile = struct {
    enabled: bool,
    last: f64 = 0,
    elapsed: [7]f64 = @splat(0),

    fn mark(self: *Profile, stage: usize) bool {
        if (!self.enabled) return true;
        if (c.ds4_gpu_end_commands() == 0) return false;
        const time = ops.now();
        self.elapsed[stage] = time - self.last;
        self.last = time;
        return ops.beginCommands();
    }
};

fn sharedMid(graph: *Graph, model: *const Model, layer: *const Layer, tokens: u32) bool {
    return ops.dense(graph.sh_gate, model, layer.ffn_gate_shexp, graph.mixed, tokens) and
        ops.dense(graph.sh_up, model, layer.ffn_up_shexp, graph.mixed, tokens) and
        c.ds4_gpu_swiglu_tensor(graph.sh_mid, graph.sh_gate, graph.sh_up, tokens * graph.config.shape.n_ff_exp, 0, 1) != 0;
}

pub fn run(graph: *Graph, model: *const Model, layer: *const Layer, tokens: u32) bool {
    return runLayer(graph, model, layer, null, tokens);
}

pub fn runSteered(graph: *Graph, model: *const Model, layer: *const Layer, index: u32, tokens: u32) bool {
    return runLayer(graph, model, layer, index, tokens);
}

fn reduce(graph: *Graph, shared: ?*const c.ds4_gpu_tensor, index: ?u32, tokens: u32, stride: u32) bool {
    const shape = &graph.config.shape;
    if (index != null and graph.steer_ffn_scale != 0)
        return graph.steer_dirs != null and c.ds4_gpu_qwen4_moe_reduce_steering_tensor(graph.blk, graph.part, graph.weights, graph.sh_gate_logit, shared, graph.R, graph.inj, graph.steer_dirs, index.?, tokens, shape.n_expert_used, stride, shape.n_embd, shape.n_hc, @import("resources.zig").effectiveScale(graph, graph.steer_ffn_scale), graph.steer_mode) != 0;
    return c.ds4_gpu_qwen4_moe_reduce_tensor(graph.blk, graph.part, graph.weights, graph.sh_gate_logit, shared, graph.R, graph.inj, tokens, shape.n_expert_used, stride, shape.n_embd, shape.n_hc) != 0;
}

fn runLayer(graph: *Graph, model: *const Model, layer: *const Layer, index: ?u32, tokens: u32) bool {
    const shape = &graph.config.shape;
    var profile = Profile{ .enabled = tokens > 8 and ops.present("DS4_QWEN4_MOE_PROFILE") };
    if (!profile.mark(0)) return false;
    var ok = ops.dense(graph.router, model, layer.ffn_gate_inp, graph.mixed, tokens) and
        c.ds4_gpu_qwen4_router_topk_tensor(graph.selected, graph.weights, graph.router, graph.mixed, model.map, model.size, layer.ffn_gate_inp_shexp.*.abs_offset, layer.ffn_gate_inp_shexp.*.type, shape.n_embd, graph.sh_gate_logit, tokens, shape.n_expert, shape.n_expert_used) != 0 and profile.mark(0);
    // With512 experts a16-row batch leaves almost every expert tile empty.
    // Preserve Metal's64-row crossover rather than CUDA's lower threshold.
    const matrix = tokens > 64 and shape.n_embd % 64 == 0 and shape.n_ff_exp % 64 == 0 and
        resources.expertHasMatrixKernel(layer.ffn_gate_exps.*.type) and
        layer.ffn_up_exps.*.type == layer.ffn_gate_exps.*.type and resources.expertHasMatrixKernel(layer.ffn_down_exps.*.type) and
        resources.denseSupported(layer.ffn_gate_shexp) and resources.denseSupported(layer.ffn_up_shexp) and resources.denseSupported(layer.ffn_down_shexp);
    if (matrix) {
        ok = ok and c.ds4_gpu_qwen4_moe_build_lists_tensor(graph.moe_lists, graph.moe_counts, graph.selected, tokens, shape.n_expert_used, shape.n_expert, graph.cap_tokens) != 0 and
            profile.mark(1) and
            c.ds4_gpu_qwen4_moe_mm_mid_tensor(graph.mid, graph.mixed, graph.moe_lists, graph.moe_counts, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_gate_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_expert_used, shape.n_embd, shape.n_ff_exp, graph.cap_tokens) != 0 and
            profile.mark(2) and sharedMid(graph, model, layer, tokens) and profile.mark(3);
        ok = ok and c.ds4_gpu_qwen4_moe_mm_down_tensor(graph.part, graph.mid, graph.moe_lists, graph.moe_counts, model.map, model.size, layer.ffn_down_exps.*.abs_offset, layer.ffn_down_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_expert_used, shape.n_ff_exp, shape.n_embd, graph.cap_tokens) != 0 and
            profile.mark(4) and ops.dense(graph.sh_out, model, layer.ffn_down_shexp, graph.sh_mid, tokens) and profile.mark(5);
        ok = ok and reduce(graph, graph.sh_out, index, tokens, shape.n_expert_used) and profile.mark(6);
        if (profile.enabled) _ = c.fprintf(c.__stderrp, "ds4: Qwen MoE ms pos=%u T=%u weight=%.*s ok=%d router=%.3f lists=%.3f mid=%.3f shared_mid=%.3f down=%.3f shared_down=%.3f reduce=%.3f\n", graph.pos, tokens, @as(c_int, @intCast(layer.ffn_gate_exps.*.name.len)), layer.ffn_gate_exps.*.name.ptr, @as(c_int, @intFromBool(ok)), profile.elapsed[0] * 1e3, profile.elapsed[1] * 1e3, profile.elapsed[2] * 1e3, profile.elapsed[3] * 1e3, profile.elapsed[4] * 1e3, profile.elapsed[5] * 1e3, profile.elapsed[6] * 1e3);
        return ok;
    }
    // A shared dense product avoids reading the same shared expert once per
    // row. Single tokens and two/three-row verifies keep it as an extra slot.
    const shared_dense = tokens > 8 and resources.denseSupported(layer.ffn_gate_shexp) and
        resources.denseSupported(layer.ffn_up_shexp) and resources.denseSupported(layer.ffn_down_shexp);
    if (shared_dense) ok = ok and sharedMid(graph, model, layer, tokens) and ops.dense(graph.sh_out, model, layer.ffn_down_shexp, graph.sh_mid, tokens);
    const grouped = shared_dense and layer.ffn_gate_exps.*.type == c.DS4_TENSOR_Q4_K and
        layer.ffn_up_exps.*.type == c.DS4_TENSOR_Q4_K and layer.ffn_down_exps.*.type == c.DS4_TENSOR_MXFP4 and !ops.present("DS4_QWEN4_MOE_NO_GROUP");
    if (grouped) {
        ok = ok and c.ds4_gpu_qwen4_moe_build_lists_tensor(graph.moe_lists, graph.moe_counts, graph.selected, tokens, shape.n_expert_used, shape.n_expert, graph.cap_tokens) != 0 and
            c.ds4_gpu_qwen4_moe_mid_grouped_tensor(graph.mid, graph.mixed, graph.selected, graph.moe_lists, graph.moe_counts, graph.cap_tokens, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_gate_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_embd, shape.n_ff_exp) != 0 and
            c.ds4_gpu_qwen4_moe_down_grouped_tensor(graph.part, graph.mid, graph.selected, graph.moe_lists, graph.moe_counts, graph.cap_tokens, model.map, model.size, layer.ffn_down_exps.*.abs_offset, layer.ffn_down_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_ff_exp, shape.n_embd) != 0;
    } else {
        const shared_kind: u32 = if (shared_dense) std.math.maxInt(u32) else layer.ffn_gate_shexp.*.type;
        const down_kind: u32 = if (shared_dense) std.math.maxInt(u32) else layer.ffn_down_shexp.*.type;
        ok = ok and c.ds4_gpu_qwen4_moe_mid_tensor(graph.mid, graph.mixed, graph.selected, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_gate_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_embd, shape.n_ff_exp, if (shared_dense) 0 else layer.ffn_gate_shexp.*.abs_offset, if (shared_dense) 0 else layer.ffn_up_shexp.*.abs_offset, shared_kind) != 0 and
            c.ds4_gpu_qwen4_moe_down_tensor(graph.part, graph.mid, graph.selected, model.map, model.size, layer.ffn_down_exps.*.abs_offset, layer.ffn_down_exps.*.type, shape.n_expert, tokens, shape.n_expert_used, shape.n_ff_exp, shape.n_embd, if (shared_dense) 0 else layer.ffn_down_shexp.*.abs_offset, down_kind) != 0;
    }
    return ok and reduce(graph, if (shared_dense) graph.sh_out else null, index, tokens, shape.n_expert_used + @as(u32, if (shared_dense) 0 else 1));
}
