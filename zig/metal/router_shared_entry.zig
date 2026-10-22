//! Sole public C facade. Mode selection below is argument adaptation; all
//! admission, ownership, shader packet and command policy is in Shared.
const c = @import("metal_c");
const owner = @import("router_shared.zig");
const Tensor = c.ds4_gpu_tensor;
fn request(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64, clamp: f32, mode: owner.Mode) owner.Shared {
    return .{ .gate = gate, .up = up, .mid = mid, .map = map, .size = size, .gate_offset = a, .up_offset = b, .width = width, .outputs = outputs, .x = x, .rows = rows, .clamp = clamp, .mode = mode };
}
export fn ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, clamp: f32) c_int {
    return owner.shared(request(gate, up, mid, map, size, a, b, width, outputs, x, 1, clamp, .store));
}
export fn ds4_gpu_shared_mid_swiglu_q8_0_tensor(mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, clamp: f32) c_int {
    return owner.shared(request(null, null, mid, map, size, a, b, width, outputs, x, 1, clamp, .mid_only));
}
export fn ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, clamp: f32) c_int {
    // The old model-view alias already selected the same model range policy.
    return owner.shared(request(gate, up, mid, map, size, a, b, width, outputs, x, 1, clamp, .store));
}
export fn ds4_gpu_dsv41_shared_gate_up_bf16_rows(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u32, clamp: f32) c_int {
    return owner.shared(request(gate, up, mid, map, size, a, b, width, outputs, x, rows, clamp, .bf16));
}
export fn ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64, clamp: f32) c_int {
    return owner.shared(request(gate, up, mid, map, size, a, b, width, outputs, x, rows, clamp, .ext));
}
export fn ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, outputs: u64, x: ?*const Tensor, rows: u64, clamp: f32) c_int {
    return owner.shared(request(gate, up, mid, map, size, a, b, width, outputs, x, rows, clamp, .scalar_rows));
}
export fn ds4_gpu_router_shared_gate_up_q8_0_tensor(logits: ?*Tensor, gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, map: ?*const anyopaque, size: u64, router_offset: u64, gate_offset: u64, up_offset: u64, width: u64, router_outputs: u64, outputs: u64, x: ?*const Tensor, clamp: f32, router_only: bool) c_int {
    return owner.routerShared(.{ .shared = request(gate, up, mid, map, size, gate_offset, up_offset, width, outputs, x, 1, clamp, .store), .logits = logits, .router_offset = router_offset, .router_outputs = router_outputs, .router_only = router_only });
}
export fn ds4_gpu_router_project_select_fused_tensor(logits: ?*Tensor, probs: ?*Tensor, selected: ?*Tensor, weights: ?*Tensor, map: ?*const anyopaque, size: u64, router_offset: u64, bias_offset: u64, has_bias: bool, x: ?*const Tensor) c_int {
    return owner.project(.{ .logits = logits, .probs = probs, .selected = selected, .weights = weights, .map = map, .size = size, .router_offset = router_offset, .bias_offset = bias_offset, .has_bias = has_bias, .x = x });
}
