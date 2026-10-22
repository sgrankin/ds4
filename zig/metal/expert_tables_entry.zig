const c = @import("metal_c");
const owner = @import("expert_tables.zig");
const Cache = @import("expert_cache.zig");
const State = Cache.State;
const Entry = Cache.Entry;
const Buffer = Cache.Buffer;
fn tables(state: *State) owner.Tables {
    var facts: c.ds4_metal_table_facts = undefined;
    c.ds4_metal_tables_facts(&facts);
    return .{ .state = state, .facts = facts };
}
fn deviceFacts() c.ds4_metal_table_facts {
    var value: c.ds4_metal_table_facts = undefined;
    c.ds4_metal_tables_facts(&value);
    return value;
}
export fn ds4_native_table_stream_compact_addr_ensure_buffers(state: *State, layer: u32) c_int {
    return @intFromBool(tables(state).ensureCompact(layer));
}
export fn ds4_native_table_stream_compact_addr_prepare(state: *State, layer: u32, entries: [*c]const [*c]Entry, n_entries: u32) c_int {
    if (entries == null or n_entries == 0) return 0;
    return @intFromBool(tables(state).compact(layer, entries[0..n_entries]));
}
export fn ds4_native_table_stream_selected_ids_prepare(state: *State, layer: u32, selected_ids: [*c]const i32, n_selected: u32) c_int {
    if (selected_ids == null or n_selected == 0) return 0;
    return @intFromBool(tables(state).hostIds(layer, selected_ids[0..n_selected]));
}
export fn ds4_native_table_stream_expert_cache_ensure_addr_buffers(state: *State, layer: u32) c_int {
    return @intFromBool(tables(state).ensureAddresses(layer));
}
export fn ds4_native_table_stream_expert_cache_zero_addr_slot(state: *State, layer: u32, expert: u32) void {
    tables(state).zeroAddress(layer, expert);
}
export fn ds4_native_table_stream_expert_cache_set_addr_slot_raw(state: *State, layer: u32, expert: u32, gate_buffer: Buffer, gate_inner: usize, up_buffer: Buffer, up_inner: usize, down_buffer: Buffer, down_inner: usize) c_int {
    return @intFromBool(tables(state).setAddress(layer, expert, gate_buffer, gate_inner, up_buffer, up_inner, down_buffer, down_inner));
}
export fn ds4_native_table_stream_expert_cache_addr_buffers(state: *State, layer: u32) c_int {
    return @intFromBool(tables(state).ensureAddresses(layer));
}
export fn ds4_native_table_stream_full_expert_addr_clear_layer(state: *State, layer: u32) void {
    tables(state).clearFull(layer);
}
export fn ds4_native_table_stream_full_expert_addr_table_prepare(state: *State, model_map: ?*const anyopaque, model_size: u64, layer: u32, n_total_expert: u32, gate_abs_offset: u64, up_abs_offset: u64, down_abs_offset: u64, gate_expert_bytes: u64, down_expert_bytes: u64) ?*Entry {
    return tables(state).full(.{ .model = model_map, .model_bytes = model_size, .gate = gate_abs_offset, .up = up_abs_offset, .down = down_abs_offset, .gate_bytes = gate_expert_bytes, .down_bytes = down_expert_bytes }, layer, n_total_expert);
}
export fn ds4_native_table_q4_expert_table_group_size(state: *State, n_total_expert: u32) u32 {
    _ = state;
    return owner.groupSize(n_total_expert);
}
export fn ds4_native_table_q4_table_queue_residency_requested(state: *State) bool {
    _ = state;
    return owner.queueRequested();
}
export fn ds4_native_table_q4_non_streaming_opt_in_enabled(state: *State) bool {
    _ = state;
    return owner.nonStreamingOptIn();
}
export fn ds4_native_table_q4_selected_paths_allowed(state: *State) bool {
    _ = state;
    return owner.selectedAllowed(deviceFacts());
}
export fn ds4_native_table_pro_q4_expert_table_auto_available(state: *State) c_int {
    _ = state;
    return @intFromBool(owner.autoAvailable(deviceFacts()));
}
export fn ds4_native_table_q4_table_queue_residency_enabled(state: *State, auto_queue_residency: bool) bool {
    _ = state;
    return owner.queueEnabled(deviceFacts(), auto_queue_residency);
}
export fn ds4_native_table_q4_table_model_residency_enabled(state: *State) bool {
    _ = state;
    return owner.modelResidency();
}
export fn ds4_native_table_pro_q4_expert_indirect_shape_supported(state: *State, n_total_expert: u32, n_expert: u32, gate_tensor_bytes: u64, down_tensor_bytes: u64) bool {
    _ = state;
    return owner.indirectShape(n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}
export fn ds4_native_table_pro_q4_expert_table_auto_enabled(state: *State, n_total_expert: u32, n_expert: u32, gate_tensor_bytes: u64, down_tensor_bytes: u64) bool {
    _ = state;
    return owner.tableAuto(deviceFacts(), n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}
export fn ds4_native_table_pro_q4_expert_address_auto_enabled(state: *State, n_total_expert: u32, n_expert: u32, gate_tensor_bytes: u64, down_tensor_bytes: u64) bool {
    _ = state;
    return owner.addressAuto(deviceFacts(), n_total_expert, n_expert, gate_tensor_bytes, down_tensor_bytes);
}
export fn ds4_native_table_q4_table_bind_anchors_enabled(state: *State) bool {
    _ = state;
    return owner.bindAnchors();
}
export fn ds4_native_table_bind_q4_expert_table_anchors(state: *State, enc: c.ds4_metal_table_encoder_ref, table: c.ds4_metal_table_ref, first_index: usize, max_count: usize) c_int {
    _ = state;
    return @intFromBool(owner.anchors(enc, table, first_index, max_count));
}
export fn ds4_native_table_use_model_residency_set(state: *State, cb: c.ds4_metal_table_command_ref) c_int {
    _ = state;
    if (!owner.modelResidency()) return 1;
    if (c.ds4_metal_tables_use_model(cb)) return 1;
    _ = c.fputs("ds4: Metal Q4 table model residency set is not available\n", c.__stderrp);
    return 0;
}
export fn ds4_native_table_q4_expert_layer_residency_set(state: *State, gate_table: c.ds4_metal_table_ref, up_table: c.ds4_metal_table_ref, down_table: c.ds4_metal_table_ref, auto_queue_residency: bool) c.ds4_metal_residency_ref {
    return tables(state).layerResidency(gate_table, up_table, down_table, auto_queue_residency);
}
export fn ds4_native_table_q4_expert_table(state: *State, model_map: ?*const anyopaque, model_size: u64, tensor_offset: u64, expert_bytes: u64, n_total_expert: u32, encoder: c.ds4_metal_argument_ref) c.ds4_metal_table_ref {
    return tables(state).q4(model_map, model_size, tensor_offset, expert_bytes, n_total_expert, encoder, false);
}
export fn ds4_native_table_q4_expert_address_table(state: *State, model_map: ?*const anyopaque, model_size: u64, tensor_offset: u64, expert_bytes: u64, n_total_expert: u32) c.ds4_metal_table_ref {
    return tables(state).q4(model_map, model_size, tensor_offset, expert_bytes, n_total_expert, null, true);
}
export fn ds4_native_table_preload_q4_expert_tables(state: *State, model_map: ?*const anyopaque, model_size: u64, gate_offset: u64, up_offset: u64, down_offset: u64, gate_expert_bytes: u64, down_expert_bytes: u64, n_total_expert: u32) c_int {
    return @intFromBool(tables(state).preload(model_map, model_size, gate_offset, up_offset, down_offset, gate_expert_bytes, down_expert_bytes, n_total_expert));
}
export fn ds4_native_table_use_q4_expert_table_resources(state: *State, cb: c.ds4_metal_table_command_ref, enc: c.ds4_metal_table_encoder_ref, table: c.ds4_metal_table_ref, queue_residency: bool) void {
    _ = state;
    owner.useResources(cb, enc, table, queue_residency);
}
