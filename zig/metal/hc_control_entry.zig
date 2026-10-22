//! Sole public C facade; internal HC requests have named tensor/model fields.
const h = @import("hc_control.zig");
const p = @import("hc_producer.zig");
const Tensor = h.Tensor;

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
