//! Publication and startup policy for the actual no-copy model-view owner.
//! The existing View registry owns buffers; fixed ARC leaves own residency
//! objects and warmup submission. No mapped-model or resource state is copied.
const std = @import("std");
const c = @import("model_views_c");
const Views = @import("model_views.zig");
const State = Views.State;
const max64 = std.math.maxInt(u64);
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn now() f64 {
    return c.ds4_metal_model_view_now_ms();
}
fn effectiveMaximum(bytes: u64, maximum: u64) u64 {
    return if (maximum != 0) maximum else @min(bytes, c.DS4_METAL_FALLBACK_MAX_TENSOR_BYTES);
}
fn logEnabled(streaming: bool) bool {
    if (!streaming) return true;
    const trace = c.getenv("DS4_METAL_STREAMING_MAP_TRACE");
    return trace != null and trace[0] != 0 and c.strcmp(trace, "0") != 0;
}
fn clearResidency(state: *State) void {
    c.ds4_metal_model_view_residency_clear(state);
}
fn residency(state: *State, streaming: bool) bool {
    // TP sharding faults only this rank's owned expert ranges. Whole-view
    // residency must stay off just as it does for SSD streaming.
    if (state.count == 0 or streaming or state.residency_skipped != 0 or present("DS4_METAL_NO_RESIDENCY")) return true;
    // Register the complete shared allocations before the first graph command
    // so driver VM validation does not appear as measured model compute. The
    // residency set is a budget hint, not a full-file prefetch request.
    return c.ds4_metal_model_view_residency_install(state, !present("DS4_METAL_DISABLE_QUEUE_RESIDENCY_SET")) != 0;
}
fn warm(state: *const State) bool {
    if (state.count == 0) return true;
    const pipeline = c.ds4_metal_model_view_warm_pipeline() orelse return false;
    var stride: u64 = 1024 * 1024;
    const megabytes = c.getenv("DS4_METAL_MODEL_WARMUP_STRIDE_MB");
    if (megabytes != null and megabytes[0] != 0) {
        var end: [*c]u8 = null;
        const value = c.strtoull(megabytes, &end, 10);
        if (end != megabytes and value > 0 and value <= 1024) stride = value * 1024 * 1024;
    }
    const kilobytes = c.getenv("DS4_METAL_MODEL_WARMUP_STRIDE_KB");
    if (kilobytes != null and kilobytes[0] != 0) {
        var end: [*c]u8 = null;
        const value = c.strtoull(kilobytes, &end, 10);
        if (end != kilobytes and value > 0 and value <= 1024 * 1024) {
            stride = @max(value * 1024, @as(u64, @intCast(c.getpagesize())));
        }
    }
    // Coarse validation touches avoid moving model-prefetch pressure into
    // startup. Preserve C's unsigned rounding and accumulation expressions.
    var ranges: [c.DS4_METAL_MAX_MODEL_VIEWS]c.ds4_metal_model_warm_range = undefined;
    var total: u64 = 0;
    for (state.entries[0..state.count], 0..) |entry, index| {
        const touches = (entry.bytes +% stride -% 1) / stride;
        ranges[index] = .{ .bytes = entry.bytes, .destination = total, .groups = (touches +% 255) / 256 };
        total +%= touches;
    }
    if (total == 0 or total > std.math.maxInt(usize)) return false;
    return c.ds4_metal_model_view_warm_encode(state, pipeline, &ranges, total, stride) != 0;
}
fn progress(stage: u32, text: ?[*:0]const u8) void {
    c.ds4_metal_model_view_progress(stage, text);
}
fn finish(state: *State, streaming: bool, started: f64, mapped: u64, display_offset: u64) bool {
    const mapped_at = now();
    const request = !streaming and !present("DS4_METAL_NO_RESIDENCY");
    if (request) progress(0, "requesting Metal residency (may take tens of seconds)");
    if (!residency(state, streaming)) {
        if (request) progress(2, null);
        return false;
    }
    if (request) progress(1, null);
    const resident_at = now();
    var warmed = true;
    const warm_started = now();
    if (!streaming and !present("DS4_METAL_NO_RESIDENCY") and !present("DS4_METAL_NO_MODEL_WARMUP")) {
        // First use of no-copy storage can pay queue setup, page-table and
        // shared-allocation validation costs. Sample each view before timed
        // graph execution; a dense prefetch would create startup pressure.
        // A TP command touching every view would demand residency of unowned
        // ranges, so its CPU-side sharded warmup handles those instead.
        if (state.residency_skipped == 0) {
            progress(0, "warming Metal model views");
            warmed = warm(state);
            progress(if (warmed) 1 else 2, null);
        }
    }
    c.ds4_metal_model_view_keepalive(); // also runs after a failed warmup in C
    const warmed_at = now();
    if (logEnabled(streaming)) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n", mapped_at - started, resident_at - mapped_at, warmed_at - warm_started, @as(f64, @floatFromInt(mapped)) / 1024.0 / 1024.0, @as(f64, @floatFromInt(display_offset)) / 1024.0 / 1024.0);
    }
    return warmed;
}
fn publish(state: *State, map: ?*const anyopaque, size: u64, offset: u64, bytes: u64, maximum: u64) void {
    state.published_map = map;
    state.published_size = size;
    state.published_offset = offset;
    state.published_bytes = bytes;
    state.published_max_tensor_bytes = maximum;
}
pub fn range(state: *State, streaming: bool, map: ?*const anyopaque, size: u64, offset: u64, bytes: u64, maximum: u64) bool {
    if (c.ds4_gpu_init() == 0 or map == null or size == 0) return false;
    if (offset > size or bytes == 0 or bytes > size - offset) return false;
    const effective = effectiveMaximum(bytes, maximum);
    if (state.published_map == map and state.published_size == size and state.published_offset == offset and state.published_bytes == bytes and state.published_max_tensor_bytes == effective) return true;
    if (Views.find(state, map, size, offset, bytes) != Views.missing) return true;
    clearResidency(state);
    const started = now();
    var mapped: u64 = 0;
    // A contiguous map appends views, including on partial add failure. The
    // caller's residency is cleared on failure, but existing views stay put.
    if (!Views.add(state, map, size, offset, bytes, effective, false, &mapped) or !finish(state, streaming, started, mapped, offset)) {
        clearResidency(state);
        return false;
    }
    publish(state, map, size, offset, bytes, effective);
    if (logEnabled(streaming)) _ = c.fprintf(c.__stderrp, "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n", state.count);
    return true;
}
pub fn spans(state: *State, streaming: bool, map: ?*const anyopaque, size: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32, maximum: u64) bool {
    if (c.ds4_gpu_init() == 0 or map == null or size == 0 or offsets == null or sizes == null or count == 0) return false;
    if (count == 1 and !streaming) return range(state, streaming, map, size, offsets[0], sizes[0], maximum);
    if (Views.coversSpans(state, map, size, offsets, sizes, count)) return true;
    const started = now();
    const effective = effectiveMaximum(size, maximum);
    clearResidency(state);
    // Streaming replaces only this target's active layer. Auxiliary vision
    // and drafter maps survive both single and disjoint target replacements.
    if (streaming) Views.removeMap(state, map) else Views.clear(state);
    var complete = false;
    defer if (!complete) {
        clearResidency(state);
        Views.clear(state);
    };
    var mapped: u64 = 0;
    var first: u64 = max64;
    for (offsets[0..count], sizes[0..count], 0..) |offset, bytes, index| {
        if (offset > size or bytes == 0 or bytes > size - offset) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal model span %u is outside the GGUF mapping\n", @as(c_uint, @intCast(index)));
            return false;
        }
        first = @min(first, offset);
        if (!Views.add(state, map, size, offset, bytes, @min(effective, bytes), true, &mapped)) return false;
    }
    if (!finish(state, streaming, started, mapped, first)) return false;
    publish(state, map, size, if (first == max64) 0 else first, mapped, effective);
    if (logEnabled(streaming)) _ = c.fprintf(c.__stderrp, "ds4: Metal mapped mmaped model as %u disjoint shared buffers across %u tensor spans\n", state.count, count);
    complete = true;
    return true;
}
