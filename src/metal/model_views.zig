//! Page-aligned no-copy model views. Native policy owns overlap, coverage and
//! registry order; the small ARC leaves create, move and retire Metal buffers.
const std = @import("std");
const c = @import("model_views_c");
pub const State = c.ds4_metal_model_views;
pub const missing = std.math.maxInt(u32);
const gib: u64 = 1024 * 1024 * 1024;

fn fail(message: [*:0]const u8) bool {
    _ = c.fprintf(c.__stderrp, "ds4: Metal %s\n", message);
    return false;
}
fn asGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

pub fn clear(state: *State) void {
    for (state.entries[0..state.count]) |*entry| c.ds4_metal_model_view_retire(entry);
    state.count = 0;
}

pub fn removeMap(state: *State, model_map: ?*const anyopaque) void {
    var kept: u32 = 0;
    for (state.entries[0..state.count], 0..) |*entry, index| {
        if (entry.model_map != model_map) {
            if (kept != index) c.ds4_metal_model_view_move(&state.entries[kept], entry);
            kept += 1;
        }
    }
    for (state.entries[kept..state.count]) |*entry| c.ds4_metal_model_view_retire(entry);
    state.count = kept;
}

pub fn find(state: *const State, model_map: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64) u32 {
    const end = offset +% bytes;
    for (state.entries[0..state.count], 0..) |entry, index| {
        if (entry.model_map == model_map and entry.model_size == model_size and
            offset >= entry.model_offset and end <= entry.model_offset +% entry.bytes)
            return @intCast(index);
    }
    return missing;
}

pub fn coversRange(state: *const State, model_map: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64) bool {
    if (model_map == null or model_size == 0 or bytes == 0 or
        offset > model_size or bytes > model_size - offset) return false;
    return find(state, model_map, model_size, offset, bytes) != missing;
}

pub fn coversSpans(state: *const State, model_map: ?*const anyopaque, model_size: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32) bool {
    if (offsets == null or sizes == null or count == 0) return false;
    for (offsets[0..count], sizes[0..count]) |offset, bytes| {
        if (!coversRange(state, model_map, model_size, offset, bytes)) return false;
    }
    return true;
}

pub fn add(state: *State, model_map: ?*const anyopaque, model_size: u64, map_offset: u64, map_size: u64, max_tensor_bytes: u64, use_default_view_cap: bool, mapped_model_size_out: [*c]u64) bool {
    const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
    const mask = page - 1;
    const model_address = @intFromPtr(model_map);
    if (model_address & mask != 0) return fail("model mmap base is not page aligned");
    if (map_offset > model_size or map_size > model_size - map_offset)
        return fail("model mapped range is outside the GGUF mapping");
    const page_offset = map_offset & ~mask;
    const leading = map_offset - page_offset;
    if (map_size > std.math.maxInt(u64) - leading or
        leading + map_size > std.math.maxInt(u64) - mask)
        return fail("model mapped range overflows page alignment");
    const mapped_bytes = (leading + map_size + mask) & ~mask;
    const max_buffer = c.ds4_metal_model_view_max_buffer_length() & ~mask;

    // Wrap tensor data only: metadata never reaches a kernel. The model can
    // exceed Metal's maximum buffer length. Stable overlapping views avoid
    // per-tensor VM/residency setup and guarantee that every admitted tensor
    // fits in ONE buffer; a hot encoder never has to split a weight tensor.
    if (max_tensor_bytes > map_size)
        return fail("model max tensor span is larger than a mapped tensor span");
    if (max_tensor_bytes > std.math.maxInt(u64) - mask)
        return fail("model max tensor span overflows page alignment");
    const tensor_bytes = (max_tensor_bytes + mask) & ~mask;
    if (tensor_bytes > std.math.maxInt(u64) - page)
        return fail("model view overlap overflows page slack");
    const overlap = tensor_bytes + page;
    if (max_buffer == 0 or max_buffer <= overlap) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal maxBufferLength is too small for DS4 model views (max tensor %.2f GiB, max buffer %.2f GiB)\n", asGiB(max_tensor_bytes), asGiB(max_buffer));
        return false;
    }

    var view_limit = max_buffer;
    const setting = c.getenv("DS4_METAL_MODEL_VIEW_MAX_GIB");
    if (setting != null and setting[0] != 0) {
        // Preserve the existing strtoull acceptance and unsigned multiplication
        // (including a trailing suffix); this is a port, not a new knob parser.
        var end: [*c]u8 = null;
        const value = c.strtoull(setting, &end, 10);
        if (end != setting and value > 0) {
            const limit = (value *% gib) & ~mask;
            if (limit > 0) view_limit = limit;
        }
    } else if (use_default_view_cap and mapped_bytes > max_buffer) {
        // Distributed span maps already need splitting. Cap their huge VM
        // ranges to bound first-command validation; contiguous maps stay as-is.
        view_limit = @min(view_limit, 128 * gib);
    }
    view_limit = @min(view_limit, max_buffer) & ~mask;
    if (view_limit == 0 or view_limit <= overlap) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal model view cap is too small for DS4 model views (cap %.2f GiB, max tensor %.2f GiB)\n", asGiB(view_limit), asGiB(max_tensor_bytes));
        return false;
    }

    const step = view_limit - overlap;
    var offset: u64 = 0;
    while (offset < mapped_bytes) {
        if (state.count == c.DS4_METAL_MAX_MODEL_VIEWS)
            return fail("model needs more mapped views than expected");
        const bytes = @min(mapped_bytes - offset, view_limit);
        const absolute = page_offset +% offset;
        const address: ?*const anyopaque = @ptrFromInt(model_address +% absolute);
        const entry = &state.entries[state.count];
        if (c.ds4_metal_model_view_install(entry, address, bytes, state.count) == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n", asGiB(absolute), asGiB(bytes));
            return false;
        }
        entry.model_map = model_map;
        entry.model_size = model_size;
        entry.model_offset = absolute;
        entry.bytes = bytes;
        state.count += 1;
        state.wrap_count +%= 1;
        state.wrap_bytes +%= bytes;
        state.wrap_max_bytes = @max(state.wrap_max_bytes, bytes);
        if (offset + bytes >= mapped_bytes) break;
        offset += step;
    }
    if (mapped_model_size_out != null) mapped_model_size_out[0] +%= mapped_bytes;
    return true;
}
