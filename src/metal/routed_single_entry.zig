//! The sole public C facade. Policy, packets and transaction live in Routed.
const c = @import("metal_c");
const Routed = @import("routed_single.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_routed_moe_one_tensor(
    out: ?*Tensor,
    gate: ?*Tensor,
    up: ?*Tensor,
    mid: ?*Tensor,
    experts: ?*Tensor,
    model_map: ?*const anyopaque,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: ?*const Tensor,
    weights: ?*const Tensor,
    n_total_expert: u32,
    n_expert: u32,
    clamp: f32,
    x: ?*const Tensor,
    add_in: ?*const Tensor,
    layer_index: u32,
    force_resident: bool,
) c_int {
    const request = c.ds4_metal_routed_request{
        .out = out,
        .gate = gate,
        .up = up,
        .mid = mid,
        .experts = experts,
        .model = model_map,
        .model_bytes = model_size,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_type = gate_type,
        .down_type = down_type,
        .gate_expert_bytes = gate_expert_bytes,
        .gate_row_bytes = gate_row_bytes,
        .down_expert_bytes = down_expert_bytes,
        .down_row_bytes = down_row_bytes,
        .input_width = expert_in_dim,
        .middle_width = expert_mid_dim,
        .output_width = out_dim,
        .selected = selected,
        .weights = weights,
        .total_experts = n_total_expert,
        .used_experts = n_expert,
        .clamp = clamp,
        .x = x,
        .layer = layer_index,
        .rows = 1,
        .mid_is_f16 = null,
        .force_resident = force_resident,
    };
    return Routed.run(&request, add_in);
}

export fn ds4_gpu_routed_moe_set_selected_override(ids: [*c]const i32, count: u32) c_int {
    return @import("routed_single_selected.zig").setOverride(ids, count);
}

// One process owner. Shared pure imports do not instantiate another trace.
const Diagnostics = @import("routed_diagnostics.zig");
var diagnostics: Diagnostics.State = .{};
export fn ds4_metal_single_trace_replay(ids: [*c]i32, count: u32) c_int {
    return diagnostics.replay(ids, count);
}
export fn ds4_metal_single_trace_record(layer: u32, ids: [*c]const i32, count: u32, total: u32) bool {
    c.ds4_metal_single_note_selected_hotness(layer, ids, count);
    return diagnostics.recordTrace(ids, count) and diagnostics.recordHotlist(layer, ids, count, total);
}
export fn ds4_native_metal_selected_hotlist_record(layer: u32, ids: [*c]const i32, count: u32, total: u32) bool {
    return diagnostics.recordHotlist(layer, ids, count, total);
}
export fn ds4_native_metal_selected_hotlist_close() void {
    diagnostics.closeHotlist();
}
export fn ds4_native_metal_selected_trace_close() void {
    diagnostics.closeTrace();
}
