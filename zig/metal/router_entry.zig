//! Sole public ABI for the pure complete router selector.
const c = @import("metal_c");
const owner = @import("router.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_router_select_tensor(selected: ?*Tensor, weights: ?*Tensor, probs: ?*Tensor, map: ?*const anyopaque, size: u64, bias: u64, hash: u64, hash_rows: u32, token: u32, experts: u32, used: u32, scale: f32, groups: u32, used_groups: u32, has_bias: bool, hash_mode: bool, logits: ?*const Tensor) c_int {
    return owner.run(.{
        .selected = selected,
        .weights = weights,
        .probs = probs,
        .map = map,
        .size = size,
        .bias_offset = bias,
        .hash_offset = hash,
        .hash_rows = hash_rows,
        .token = token,
        .experts = experts,
        .used = used,
        .scale = scale,
        .groups = groups,
        .used_groups = used_groups,
        .has_bias = has_bias,
        .hash = hash_mode,
        .logits = logits,
    });
}
export fn ds4_gpu_router_select_batch_tensor(selected: ?*Tensor, weights: ?*Tensor, probs: ?*Tensor, map: ?*const anyopaque, size: u64, bias: u64, hash: u64, hash_rows: u32, groups: u32, used_groups: u32, has_bias: bool, hash_mode: bool, logits: ?*const Tensor, tokens: ?*const Tensor, experts: u32, used: u32, scale: f32, rows: u32) c_int {
    return owner.run(.{
        .selected = selected,
        .weights = weights,
        .probs = probs,
        .map = map,
        .size = size,
        .bias_offset = bias,
        .hash_offset = hash,
        .hash_rows = hash_rows,
        .experts = experts,
        .used = used,
        .scale = scale,
        .groups = groups,
        .used_groups = used_groups,
        .has_bias = has_bias,
        .hash = hash_mode,
        .logits = logits,
        .tokens = tokens,
        .rows = rows,
        .mode = .batch,
    });
}
export fn ds4_gpu_router_select_batch_visual_tensor(selected: ?*Tensor, weights: ?*Tensor, probs: ?*Tensor, map: ?*const anyopaque, size: u64, bias: u64, hash: u64, hash_rows: u32, has_bias: bool, hash_mode: bool, vision: ?*const anyopaque, vision_size: u64, visual_bias: u64, logits: ?*const Tensor, tokens: ?*const Tensor, vocab: u32, experts: u32, used: u32, scale: f32, rows: u32) c_int {
    return owner.run(.{
        .selected = selected,
        .weights = weights,
        .probs = probs,
        .map = map,
        .size = size,
        .bias_offset = bias,
        .hash_offset = hash,
        .hash_rows = hash_rows,
        .experts = experts,
        .used = used,
        .scale = scale,
        .has_bias = has_bias,
        .hash = hash_mode,
        .logits = logits,
        .tokens = tokens,
        .rows = rows,
        .mode = .visual,
        .vision = vision,
        .vision_size = vision_size,
        .visual_offset = visual_bias,
        .vocab = vocab,
    });
}
