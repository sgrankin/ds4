//! Sole C facade for the model-view policy owner.
const owner = @import("model_views.zig");
const State = owner.State;

export fn ds4_native_metal_model_views_clear(state: *State) void {
    owner.clear(state);
}
export fn ds4_native_metal_model_views_remove_map(state: *State, model_map: ?*const anyopaque) void {
    owner.removeMap(state, model_map);
}
export fn ds4_native_metal_model_views_add(state: *State, model_map: ?*const anyopaque, model_size: u64, map_offset: u64, map_size: u64, max_tensor_bytes: u64, use_default_view_cap: bool, mapped_model_size_out: [*c]u64) c_int {
    return @intFromBool(owner.add(state, model_map, model_size, map_offset, map_size, max_tensor_bytes, use_default_view_cap, mapped_model_size_out));
}
export fn ds4_native_metal_model_views_find(state: *const State, model_map: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64) u32 {
    return owner.find(state, model_map, model_size, offset, bytes);
}
export fn ds4_native_metal_model_views_cover_range(state: *const State, model_map: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64) c_int {
    return @intFromBool(owner.coversRange(state, model_map, model_size, offset, bytes));
}
export fn ds4_native_metal_model_views_cover_spans(state: *const State, model_map: ?*const anyopaque, model_size: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32) c_int {
    return @intFromBool(owner.coversSpans(state, model_map, model_size, offsets, sizes, count));
}

// The same binding Module and actual State identity compose both owners.
const Lifecycle = @import("model_lifecycle.zig");
export fn ds4_native_metal_model_map_range(state: *State, streaming: bool, map: ?*const anyopaque, size: u64, offset: u64, bytes: u64, maximum: u64) c_int {
    return @intFromBool(Lifecycle.range(state, streaming, map, size, offset, bytes, maximum));
}
export fn ds4_native_metal_model_map_spans(state: *State, streaming: bool, map: ?*const anyopaque, size: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32, maximum: u64) c_int {
    return @intFromBool(Lifecycle.spans(state, streaming, map, size, offsets, sizes, count, maximum));
}
