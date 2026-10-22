const std = @import("std");
const c = @import("metal_c");
const owner = @import("expert_cache.zig");
export var ds4_native_expert_policy: c.ds4_metal_expert_policy = std.mem.zeroes(c.ds4_metal_expert_policy);
const State = owner.State;
const Entry = owner.Entry;
const Reuse = owner.Reuse;
const Buffer = owner.Buffer;
fn cache(state: *State, streaming: bool) owner.Cache {
    return .{ .state = state, .streaming = streaming };
}
fn ids(pointer: [*c]const i32, count: u32) []const i32 {
    return if (pointer == null or count == 0) &.{} else pointer[0..count];
}
fn identity(model: ?*const anyopaque, bytes: u64, gate: u64, up: u64, down: u64, gate_bytes: u64, down_bytes: u64) owner.Identity {
    return .{ .model = model, .model_bytes = bytes, .gate = gate, .up = up, .down = down, .gate_bytes = gate_bytes, .down_bytes = down_bytes };
}
export fn ds4_native_expert_cache_configured_count(state: *State, streaming: bool) u32 {
    return cache(state, streaming).configured();
}
export fn ds4_native_expert_cache_current_count(state: *State, streaming: bool) u32 {
    _ = streaming;
    return state.cache_entry_count;
}
export fn ds4_native_expert_cache_requested_budget(state: *State, streaming: bool) u32 {
    return cache(state, streaming).requested();
}
export fn ds4_native_expert_cache_configured_budget(state: *State, streaming: bool) u32 {
    return cache(state, streaming).budget();
}
export fn ds4_native_expert_split_ready(state: *State, streaming: bool) c_int {
    return @intFromBool(cache(state, streaming).splitReady());
}
export fn ds4_native_expert_cache_note_expert_size(state: *State, streaming: bool, gate_expert_bytes: u64, down_expert_bytes: u64) c_int {
    return @intFromBool(cache(state, streaming).noteSize(gate_expert_bytes, down_expert_bytes));
}
export fn ds4_native_expert_cache_budget_for_expert_size(state: *State, streaming: bool, gate_expert_bytes: u64, down_expert_bytes: u64) u32 {
    const value = cache(state, streaming);
    return if (value.noteSize(gate_expert_bytes, down_expert_bytes)) value.budget() else 0;
}
export fn ds4_native_expert_cache_effective_cap(state: *State, streaming: bool, layer: u32, n_total_expert: u32, n_selected: u32) u32 {
    _ = layer;
    return cache(state, streaming).effectiveCap(n_total_expert, n_selected);
}
export fn ds4_native_expert_timing_summary_enabled(state: *State, streaming: bool) c_int {
    _ = state;
    _ = streaming;
    return @intFromBool(owner.timingSummary());
}
export fn ds4_native_expert_cache_decay_route_hotness(state: *State, streaming: bool) void {
    cache(state, streaming).decay();
}
export fn ds4_native_expert_cache_reset_route_hotness(state: *State, streaming: bool) void {
    cache(state, streaming).resetHotness();
}
export fn ds4_native_expert_cache_maybe_decay_route_hotness(state: *State, streaming: bool) void {
    cache(state, streaming).maybeDecay();
}
export fn ds4_native_expert_cache_note_route_hotness(state: *State, streaming: bool, layer: u32, expert: u32, amount: u32) void {
    cache(state, streaming).addHotness(layer, expert, amount);
}
export fn ds4_native_expert_cache_note_selected_hotness(state: *State, streaming: bool, layer: u32, selected_ids: [*c]const i32, n_selected: u32) void {
    cache(state, streaming).noteSelected(layer, selected_ids, n_selected);
}
export fn ds4_native_expert_cache_note_frequency_hotness(state: *State, streaming: bool, layer: u32, frequency: [*c]const u32, n_total_expert: u32) void {
    cache(state, streaming).noteFrequency(layer, frequency, n_total_expert);
}
export fn ds4_native_expert_cache_note_token(state: *State, streaming: bool, layer_index: u32) void {
    cache(state, streaming).noteToken(layer_index);
}
export fn ds4_native_expert_cache_decay_tokens(state: *State, streaming: bool) u32 {
    _ = state;
    _ = streaming;
    return owner.decayTokens();
}
export fn ds4_native_expert_cache_next_cb_seq(state: *State, streaming: bool) u64 {
    return cache(state, streaming).nextSequence();
}
export fn ds4_native_expert_cache_note_batch_created(state: *State, streaming: bool) void {
    cache(state, streaming).batchCreated();
}
export fn ds4_native_expert_cache_note_batch_committed(state: *State, streaming: bool) void {
    cache(state, streaming).batchCommitted();
}
export fn ds4_native_expert_cache_note_owned_created(state: *State, streaming: bool) void {
    cache(state, streaming).ownedCreated();
}
export fn ds4_native_expert_cache_note_pending_completed(state: *State, streaming: bool) void {
    cache(state, streaming).pendingCompleted();
}
export fn ds4_native_expert_cache_note_owned_completed(state: *State, streaming: bool) void {
    cache(state, streaming).ownedCompleted();
}
export fn ds4_native_expert_cache_entry_inflight(state: *State, streaming: bool, e: ?*const Entry) c_int {
    return @intFromBool(cache(state, streaming).inflight(e));
}
export fn ds4_native_expert_cache_mark_inflight(state: *State, streaming: bool, e: ?*Entry) c_int {
    return @intFromBool(cache(state, streaming).mark(e));
}
export fn ds4_native_expert_cache_mark_entries_inflight(state: *State, streaming: bool, entries: [*c]const [*c]Entry, n_entries: u32, active_mask: u32) c_int {
    return @intFromBool(cache(state, streaming).markMany(entries, n_entries, active_mask));
}
export fn ds4_native_expert_slab_slot_range(state: *State, streaming: bool, slot: u32, slab_index: [*c]u32, slot_base: [*c]u64) c_int {
    const range = cache(state, streaming).slotRange(slot) orelse return 0;
    if (slab_index != null) slab_index[0] = range.slab;
    if (slot_base != null) slot_base[0] = range.base;
    return 1;
}
export fn ds4_native_expert_slab_slot_for_buffer(state: *State, streaming: bool, buffer: Buffer, gate_inner: usize, slot_out: [*c]u32) c_int {
    if (slot_out == null) return 0;
    slot_out[0] = cache(state, streaming).slotForBuffer(buffer, gate_inner) orelse return 0;
    return 1;
}
export fn ds4_native_expert_slab_push_free_slot(state: *State, streaming: bool, slot: u32) void {
    cache(state, streaming).recycle(slot);
}
export fn ds4_native_expert_slab_unlock_slot(state: *State, streaming: bool, slot: u32) c_int {
    return @intFromBool(cache(state, streaming).unlockSlot(slot));
}
export fn ds4_native_expert_unlock_explicit_buffer(state: *State, streaming: bool, buffer: Buffer) void {
    cache(state, streaming).unlockExplicit(buffer);
}
export fn ds4_native_expert_evict_dontneed_enabled(state: *State, streaming: bool) c_int {
    return @intFromBool(cache(state, streaming).evictEnabled());
}
export fn ds4_native_expert_evict_dontneed_range(state: *State, streaming: bool, model_map: ?*const anyopaque, model_size: u64, offset: u64, len: u64) void {
    cache(state, streaming).dontNeed(model_map, model_size, offset, len);
}
export fn ds4_native_expert_buffer_object_count(state: *State, streaming: bool, gate: Buffer, up: Buffer, down: Buffer) u64 {
    _ = state;
    _ = streaming;
    return owner.objectCount(gate, up, down);
}
export fn ds4_native_expert_cache_clear_entry_internal(state: *State, streaming: bool, layer: u32, expert: u32, count_eviction: c_int, recycle_slab_slot: c_int, reuse: ?*Reuse) void {
    cache(state, streaming).clear(layer, expert, count_eviction != 0, recycle_slab_slot != 0, reuse);
}
export fn ds4_native_expert_cache_trim(state: *State, streaming: bool, experts: u32) c_int {
    return @intFromBool(cache(state, streaming).trim(experts));
}
export fn ds4_native_expert_cache_clear_all(state: *State, streaming: bool, reset_stats: c_int) void {
    cache(state, streaming).clearAll(reset_stats != 0);
}
export fn ds4_native_expert_cache_is_protected(state: *State, streaming: bool, expert: u32, protect_ids: [*c]const i32, n_protect: u32) c_int {
    _ = state;
    _ = streaming;
    return @intFromBool(owner.protectedExpert(expert, ids(protect_ids, n_protect)));
}
export fn ds4_native_expert_cache_entry_protected(state: *State, streaming: bool, layer: u32, expert: u32, protect_layer: u32, protect_ids: [*c]const i32, n_protect: u32) c_int {
    return @intFromBool(cache(state, streaming).protected(layer, expert, protect_layer, ids(protect_ids, n_protect)));
}
export fn ds4_native_expert_cache_entry_reusable(state: *State, streaming: bool, e: ?*const Entry, gate_expert_bytes: u64, down_expert_bytes: u64) c_int {
    _ = state;
    _ = streaming;
    return @intFromBool(e != null and owner.reusable(e.?, gate_expert_bytes, down_expert_bytes));
}
export fn ds4_native_expert_cache_prune_layer(state: *State, streaming: bool, layer: u32, n_total_expert: u32, n_selected: u32, protect_ids: [*c]const i32, n_protect: u32) void {
    cache(state, streaming).pruneLayer(layer, n_total_expert, n_selected, ids(protect_ids, n_protect));
}
export fn ds4_native_expert_cache_prune_global(state: *State, streaming: bool, protect_layer: u32, protect_ids: [*c]const i32, n_protect: u32) void {
    cache(state, streaming).pruneGlobal(protect_layer, ids(protect_ids, n_protect));
}
export fn ds4_native_expert_cache_take_reusable(state: *State, streaming: bool, force_reuse: c_int, protect_layer: u32, protect_ids: [*c]const i32, n_protect: u32, gate_expert_bytes: u64, down_expert_bytes: u64, reuse: ?*Reuse) c_int {
    return @intFromBool(cache(state, streaming).takeReusable(force_reuse != 0, protect_layer, ids(protect_ids, n_protect), gate_expert_bytes, down_expert_bytes, reuse));
}
export fn ds4_native_expert_cache_take_reusable_batch(state: *State, streaming: bool, n_needed: u32, protect_layer: u32, protect_ids: [*c]const i32, n_protect: u32, gate_expert_bytes: u64, down_expert_bytes: u64, reuses: [*c]Reuse) u32 {
    return cache(state, streaming).takeBatch(n_needed, protect_layer, ids(protect_ids, n_protect), gate_expert_bytes, down_expert_bytes, reuses);
}
export fn ds4_native_expert_cache_release_mlock_margin(state: *State, streaming: bool, protect_layer: u32, protect_ids: [*c]const i32, n_protect: u32) u32 {
    return cache(state, streaming).releaseMargin(protect_layer, ids(protect_ids, n_protect));
}
export fn ds4_native_expert_cache_entry_matches(state: *State, streaming: bool, e: ?*const Entry, model_map: ?*const anyopaque, model_size: u64, gate_abs_offset: u64, up_abs_offset: u64, down_abs_offset: u64, gate_expert_bytes: u64, down_expert_bytes: u64) c_int {
    _ = state;
    _ = streaming;
    return @intFromBool(e != null and owner.matches(e.?, identity(model_map, model_size, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes)));
}
export fn ds4_native_expert_cache_snapshot(state: *State, streaming: bool, table: ?*const c.ds4_gpu_stream_expert_table, resident: [*c]u8) c_int {
    return @intFromBool(cache(state, streaming).snapshot(table, null, resident));
}
export fn ds4_native_expert_cache_snapshot_selected(state: *State, streaming: bool, table: ?*const c.ds4_gpu_stream_expert_table, selected: [*c]const i32, n_selected: u32, resident: [*c]u8) c_int {
    if (selected == null or n_selected == 0) return 0;
    return @intFromBool(cache(state, streaming).snapshot(table, selected[0..n_selected], resident));
}
export fn ds4_native_expert_cache_peek(state: *State, streaming: bool, model_map: ?*const anyopaque, model_size: u64, layer: u32, expert: u32, n_total_expert: u32, n_selected: u32, gate_abs_offset: u64, up_abs_offset: u64, down_abs_offset: u64, gate_expert_bytes: u64, down_expert_bytes: u64) ?*Entry {
    return cache(state, streaming).peek(identity(model_map, model_size, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes), layer, expert, n_total_expert, n_selected);
}
export fn ds4_native_expert_cache_install_loaded(state: *State, streaming: bool, model_map: ?*const anyopaque, model_size: u64, layer: u32, expert: u32, gate_abs_offset: u64, up_abs_offset: u64, down_abs_offset: u64, gate_expert_bytes: u64, down_expert_bytes: u64, gate_buf: Buffer, up_buf: Buffer, down_buf: Buffer, gate_inner: usize, up_inner: usize, down_inner: usize) ?*Entry {
    return cache(state, streaming).install(identity(model_map, model_size, gate_abs_offset, up_abs_offset, down_abs_offset, gate_expert_bytes, down_expert_bytes), layer, expert, gate_buf, up_buf, down_buf, gate_inner, up_inner, down_inner);
}
export fn ds4_native_expert_cache_clear_layer(state: *State, streaming: bool, layer: u32) void {
    cache(state, streaming).clearLayer(layer);
}
