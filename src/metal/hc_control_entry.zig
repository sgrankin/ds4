//! Sole public C facade; internal HC requests have named tensor/model fields.
const h = @import("hc_control.zig");
const p = @import("hc_producer.zig");
const Tensor = h.Tensor;

export fn ds4_gpu_hc_expand_split_steering_tensor(out: ?*Tensor, block: ?*Tensor, residual: ?*const Tensor, split: ?*const Tensor, directions: ?*const Tensor, layer: u32, width: u32, hc: u32, scale: f32, bf16: bool, mode: c_int) callconv(.c) c_int {
    return h.expandSteered(.{ .out = out, .block = block, .residual = residual, .split = split }, .{ .width = width, .hc = hc }, true, directions, layer, scale, bf16, mode);
}

export fn ds4_gpu_hc_expand_steering_tensor(out: ?*Tensor, block: ?*Tensor, residual: ?*const Tensor, post: ?*const Tensor, comb: ?*const Tensor, directions: ?*const Tensor, layer: u32, width: u32, hc: u32, scale: f32, mode: c_int) callconv(.c) c_int {
    return h.expandSteered(.{ .out = out, .block = block, .residual = residual, .post = post, .comb = comb }, .{ .width = width, .hc = hc }, false, directions, layer, scale, false, mode);
}

export fn ds4_gpu_qwen4_hc_combine_steering_tensor(residual: ?*Tensor, block: ?*Tensor, injection: ?*const Tensor, directions: ?*const Tensor, layer: u32, rows: u32, width: u32, hc: u32, scale: f32, mode: c_int) callconv(.c) c_int {
    return h.qwenCombineSteered(residual, block, injection, directions, layer, rows, .{ .width = width, .hc = hc }, scale, mode);
}

export fn ds4_gpu_qwen4_moe_reduce_steering_tensor(out: ?*Tensor, partials: ?*const Tensor, weights: ?*const Tensor, shared_gate: ?*const Tensor, shared: ?*const Tensor, residual: ?*Tensor, injection: ?*const Tensor, directions: ?*const Tensor, layer: u32, rows: u32, slots: u32, stride: u32, width: u32, hc: u32, scale: f32, mode: c_int) callconv(.c) c_int {
    return h.qwenReduceSteered(.{ .out = out, .partials = partials, .weights = weights, .shared_gate = shared_gate, .shared = shared, .residual = residual, .injection = injection, .directions = directions, .layer = layer, .rows = rows, .slots = slots, .stride = stride, .scale = scale, .mode = mode }, .{ .width = width, .hc = hc });
}

export fn ds4_gpu_residual_steering_tensor(out: ?*Tensor, block: ?*Tensor, block_add: ?*const Tensor, residual: ?*const Tensor, directions: ?*const Tensor, layer: u32, width: u32, rows: u32, scale: f32, mode: c_int) callconv(.c) c_int {
    return h.residualSteered(.{ .out = out, .block = block, .block_add = block_add, .residual = residual, .directions = directions, .layer = layer, .width = width, .rows = rows, .scale = scale, .mode = mode }, .{}, null);
}

export fn ds4_gpu_residual_norm_steering_tensor(norm_out: ?*Tensor, sum_out: ?*Tensor, block: ?*Tensor, residual: ?*const Tensor, directions: ?*const Tensor, model_map: ?*const anyopaque, model_size: u64, norm_offset: u64, layer: u32, width: u32, rows: u32, scale: f32, eps: f32, mode: c_int) callconv(.c) c_int {
    return h.residualSteered(.{ .out = sum_out, .norm_out = norm_out, .block = block, .residual = residual, .directions = directions, .layer = layer, .width = width, .rows = rows, .scale = scale, .mode = mode }, .{ .map = model_map, .size = model_size, .norm = norm_offset }, eps);
}

export fn ds4_gpu_directional_steering_tensor(values: ?*Tensor, directions: ?*const Tensor, layer: u32, width: u32, rows: u32, scale: f32, mode: c_int) callconv(.c) c_int {
    return h.transformResidual(values, directions, layer, width, rows, scale, mode, false);
}

export fn ds4_gpu_hc_split_sinkhorn_tensor(
    out: ?*Tensor,
    mix: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) callconv(.c) c_int {
    return h.sinkhorn(.{ .out = out, .mix = mix }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset }, .{ .hc = n_hc, .iterations = sinkhorn_iters, .eps = eps });
}

export fn ds4_gpu_hc_weighted_sum_tensor(
    out: ?*Tensor,
    residual_hc: ?*const Tensor,
    weights: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return h.weighted(.{ .out = out, .residual = residual_hc, .post = weights }, .{ .width = n_embd, .hc = n_hc }, @as(u64, n_hc) * 4, "HC weighted sum");
}

export fn ds4_gpu_hc_weighted_sum_split_tensor(
    out: ?*Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    const shape: h.Shape = .{ .width = n_embd, .hc = n_hc };
    return h.weighted(.{ .out = out, .residual = residual_hc, .post = split }, shape, shape.mix() *% 4, "HC weighted sum split");
}

export fn ds4_gpu_hc_split_weighted_sum_tensor(
    out: ?*Tensor,
    split: ?*Tensor,
    mix: ?*const Tensor,
    residual_hc: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) callconv(.c) c_int {
    return h.splitSum(.{ .out = out, .mix = mix, .split = split, .residual = residual_hc }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset }, .{ .width = n_embd, .hc = n_hc, .iterations = sinkhorn_iters, .eps = eps }, false);
}

export fn ds4_gpu_hc_split_weighted_sum_norm_tensor(
    out: ?*Tensor,
    norm_out: ?*Tensor,
    split: ?*Tensor,
    mix: ?*const Tensor,
    residual_hc: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
) callconv(.c) c_int {
    return h.splitSum(.{ .out = out, .norm_out = norm_out, .mix = mix, .split = split, .residual = residual_hc }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset, .norm = norm_weight_offset }, .{ .width = n_embd, .hc = n_hc, .iterations = sinkhorn_iters, .eps = eps, .norm_eps = norm_eps }, true);
}

export fn ds4_gpu_hc_rms_norm_mix_f16_available() callconv(.c) c_int {
    return p.available();
}

export fn ds4_gpu_hc_rms_norm_mix_f16_tensor(
    out: ?*Tensor,
    x: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    out_dim: u32,
    eps: f32,
) callconv(.c) c_int {
    return p.normMix(.{ .out = out, .input = x }, .{ .map = model_map, .size = model_size, .weight = weight_offset }, n, out_dim, eps);
}

export fn ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
    mix: ?*Tensor,
    out: ?*Tensor,
    norm_out: ?*Tensor,
    split: ?*Tensor,
    residual_hc: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    mix_weight_offset: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n: u32,
    mix_dim: u32,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    hc_eps: f32,
    norm_eps: f32,
) callconv(.c) c_int {
    return p.compound(.{ .out = out, .norm_out = norm_out, .mix = mix, .split = split, .residual = residual_hc }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset, .norm = norm_weight_offset, .weight = mix_weight_offset }, .{ .width = n_embd, .hc = n_hc, .iterations = sinkhorn_iters, .eps = hc_eps, .norm_eps = norm_eps }, n, mix_dim, eps, false);
}

export fn ds4_gpu_hc_expand_add_rms_norm_mix_split_norm_f16_tensor(
    mix: ?*Tensor,
    out: ?*Tensor,
    norm_out: ?*Tensor,
    split: ?*Tensor,
    residual_hc: ?*const Tensor,
    block_out: ?*const Tensor,
    block_add: ?*const Tensor,
    residual_prev: ?*const Tensor,
    post: ?*const Tensor,
    comb: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    mix_weight_offset: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n: u32,
    mix_dim: u32,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    hc_eps: f32,
    norm_eps: f32,
) callconv(.c) c_int {
    return p.compound(.{ .out = out, .norm_out = norm_out, .mix = mix, .split = split, .residual = residual_hc, .block = block_out, .block_add = block_add, .previous = residual_prev, .post = post, .comb = comb }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset, .norm = norm_weight_offset, .weight = mix_weight_offset }, .{ .width = n_embd, .hc = n_hc, .iterations = sinkhorn_iters, .eps = hc_eps, .norm_eps = norm_eps }, n, mix_dim, eps, true);
}

export fn ds4_gpu_output_hc_weights_tensor(
    out: ?*Tensor,
    pre: ?*const Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    eps: f32,
) callconv(.c) c_int {
    return p.outputWeights(.{ .out = out, .input = pre }, .{ .map = model_map, .size = model_size, .scale = scale_offset, .base = base_offset }, n_hc, eps);
}

export fn ds4_gpu_hc_expand_tensor(
    out_hc: ?*Tensor,
    block_out: ?*const Tensor,
    residual_hc: ?*const Tensor,
    post: ?*const Tensor,
    comb: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return h.expand(.{ .out = out_hc, .residual = residual_hc, .block = block_out, .post = post, .comb = comb }, .{ .width = n_embd, .hc = n_hc }, false, false);
}

export fn ds4_gpu_hc_expand_add_tensor(
    out_hc: ?*Tensor,
    block_out: ?*const Tensor,
    block_add: ?*const Tensor,
    residual_hc: ?*const Tensor,
    post: ?*const Tensor,
    comb: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return h.expand(.{ .out = out_hc, .residual = residual_hc, .block = block_out, .block_add = block_add, .post = post, .comb = comb }, .{ .width = n_embd, .hc = n_hc }, false, true);
}

export fn ds4_gpu_hc_expand_split_tensor(
    out_hc: ?*Tensor,
    block_out: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return h.expand(.{ .out = out_hc, .split = split, .residual = residual_hc, .block = block_out }, .{ .width = n_embd, .hc = n_hc }, true, false);
}

export fn ds4_gpu_hc_expand_split_half_tensor(
    out_hc: ?*Tensor,
    block_out_h: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    _ = out_hc;
    _ = block_out_h;
    _ = residual_hc;
    _ = split;
    _ = n_embd;
    _ = n_hc;
    return 0;
}

export fn ds4_gpu_hc_expand_add_split_tensor(
    out_hc: ?*Tensor,
    block_out: ?*const Tensor,
    block_add: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return h.expand(.{ .out = out_hc, .split = split, .residual = residual_hc, .block = block_out, .block_add = block_add }, .{ .width = n_embd, .hc = n_hc }, true, true);
}

export fn ds4_gpu_hc_expand_add_split_half_add_tensor(
    out_hc: ?*Tensor,
    block_out: ?*const Tensor,
    block_add_h: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    _ = out_hc;
    _ = block_out;
    _ = block_add_h;
    _ = residual_hc;
    _ = split;
    _ = n_embd;
    _ = n_hc;
    return 0;
}

export fn ds4_gpu_shared_down_hc_expand_q8_0_tensor(
    out_hc: ?*Tensor,
    shared_out: ?*Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    shared_mid: ?*const Tensor,
    routed_out: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return p.q8Expand(.{ .out = out_hc, .split = split, .residual = residual_hc, .block = shared_out, .block_add = routed_out, .input = shared_mid }, .{ .map = model_map, .size = model_size, .weight = weight_offset }, .{ .width = n_embd, .hc = n_hc }, in_dim, out_dim, true);
}

export fn ds4_gpu_matmul_q8_0_hc_expand_tensor(
    out_hc: ?*Tensor,
    block_out: ?*Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: ?*const Tensor,
    residual_hc: ?*const Tensor,
    split: ?*const Tensor,
    n_embd: u32,
    n_hc: u32,
) callconv(.c) c_int {
    return p.q8Expand(.{ .out = out_hc, .split = split, .residual = residual_hc, .block = block_out, .input = x }, .{ .map = model_map, .size = model_size, .weight = weight_offset }, .{ .width = n_embd, .hc = n_hc }, in_dim, out_dim, false);
}
