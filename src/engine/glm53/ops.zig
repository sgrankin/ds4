//! GLM dense projections and mHC preparation. Preserve the existing scalar and
//! row kernels: choosing a different reduction here can change later routing.
const c = @import("gguf_c");
const Config = @import("config.zig");
pub const Tensor = c.ds4_gpu_tensor;
pub const Weight = c.ds4_tensor;
pub const Model = c.ds4_model;
pub const Graph = c.ds4_glm_gpu_graph;

pub fn denseQuant(kind: u32) bool {
    return switch (kind) {
        c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q4_0, c.DS4_TENSOR_BF16 => true,
        else => false,
    };
}

pub fn matmul(output: ?*Tensor, model: *const Model, maybe_weight: ?*const Weight, inputs: u32, outputs: u32, input: ?*const Tensor, rows: u32) bool {
    const weight = maybe_weight orelse return false;
    if (output == null or input == null or rows == 0 or !denseQuant(weight.type)) return false;
    if (weight.type == c.DS4_TENSOR_BF16)
        return c.ds4_gpu_glm53_matmul_bf16(output, model.map, model.size, weight.abs_offset, inputs, outputs, input, rows) != 0;
    return c.ds4_gpu_matmul_quant_tensor(output, model.map, model.size, weight.abs_offset, weight.type, inputs, outputs, input, rows) != 0;
}

fn collapse(graph: *const Graph, model: *const Model, output: ?*Tensor, split: ?*Tensor, mix: ?*const Tensor, residual: ?*const Tensor, scale: *const Weight, base: *const Weight) bool {
    const shape = &graph.config.shape;
    if (graph.config.reference_hc)
        return c.ds4_gpu_hc_split_sinkhorn_tensor(split, mix, model.map, model.size, scale.abs_offset, base.abs_offset, shape.n_hc, shape.n_hc_sinkhorn_iter, shape.hc_eps) != 0 and
            c.ds4_gpu_hc_weighted_sum_tensor(output, residual, split, shape.n_embd, shape.n_hc) != 0;
    return c.ds4_gpu_hc_split_weighted_sum_tensor(output, split, mix, residual, model.map, model.size, scale.abs_offset, base.abs_offset, shape.n_embd, shape.n_hc, shape.n_hc_sinkhorn_iter, shape.hc_eps) != 0;
}

pub fn hcPre(comptime batch: bool, graph: *Graph, model: *const Model, maybe_mixer: ?*const Weight, maybe_scale: ?*const Weight, maybe_base: ?*const Weight, maybe_norm: ?*const Weight, residual: ?*const Tensor, collapsed: ?*Tensor, normalized: ?*Tensor, flat: ?*Tensor, mix: ?*Tensor, split: ?*Tensor, rows: u32) bool {
    const mixer = maybe_mixer orelse return false;
    const scale = maybe_scale orelse return false;
    const base = maybe_base orelse return false;
    const norm = maybe_norm orelse return false;
    if (residual == null or collapsed == null or normalized == null or flat == null or mix == null or split == null or rows == 0) return false;
    const shape = &graph.config.shape;
    const hc_width = Config.hcWidth(&graph.config);
    const norm_ok = if (batch)
        c.ds4_gpu_rms_norm_plain_rows_tensor(flat, residual, hc_width, rows, shape.rms_eps) != 0
    else
        c.ds4_gpu_rms_norm_plain_tensor(flat, residual, hc_width, shape.rms_eps) != 0;
    if (!norm_ok or !matmul(mix, model, mixer, hc_width, Config.hcMix(&graph.config), flat, rows) or
        !collapse(graph, model, collapsed, split, mix, residual, scale, base)) return false;
    return if (batch)
        c.ds4_gpu_rms_norm_weight_rows_tensor(normalized, collapsed, model.map, model.size, norm.abs_offset, shape.n_embd, rows, shape.rms_eps) != 0
    else
        c.ds4_gpu_rms_norm_weight_tensor(normalized, collapsed, model.map, model.size, norm.abs_offset, shape.n_embd, shape.rms_eps) != 0;
}
