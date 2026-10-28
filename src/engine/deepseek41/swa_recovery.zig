//! Experimental block-only import + fixed 128-row SWA reconstruction.
//! SWB1 is deliberately not a canonical DSV4 checkpoint. No cache policy or
//! approximate persistence format is selected by this direct diagnostic API.
const std = @import("std");
const c = @import("gguf_c");
const Coordinator = @import("coordinator.zig").Coordinator;
const Payload = @import("state_payload.zig").Payload;
const Header = @import("state_payload.zig").Header;
const scheduling = @import("prefill_policy.zig");

fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const length = @min(errors.len - 1, message.len);
        @memcpy(errors[0..length], message[0..length]);
        errors[length] = 0;
    }
    return 1;
}

fn appendHash(hash: *u64, tensor: *c.ds4_gpu_tensor, bytes: u64, scratch: []u8) bool {
    var offset: u64 = 0;
    while (offset < bytes) {
        const count: usize = @intCast(@min(bytes - offset, scratch.len));
        if (c.ds4_gpu_tensor_read(tensor, offset, scratch.ptr, count) == 0) return false;
        for (scratch[0..count]) |byte| hash.* = (hash.* ^ byte) *% 1099511628211;
        offset += count;
    }
    return true;
}

// Debug-only byte check outside the timed campaign. Include pooling carries:
// returning before publish must leave scratch as well as cumulative owner KV untouched.
// At aligned E these carries are unused; the first fresh even token replaces
// them before publication. They are not imported by the even-end wire format.
fn blockHash(graph: *c.ds41_gpu_graph, end: u32) ?u64 {
    const scratch = std.heap.c_allocator.alloc(u8, 8 * 1024 * 1024) catch return null;
    defer std.heap.c_allocator.free(scratch);
    var hash: u64 = 14695981039346656037;
    for (0..4) |owner| {
        const rows = end / @as(u32, if (owner < 3) 2 else 1);
        if (!appendHash(&hash, @ptrCast(graph.compressed[owner]), @as(u64, rows) * @as(u64, if (graph.compressed_kv_fp4) 288 else if (graph.compressed_kv_f16) 1024 else 2048), scratch) or
            !appendHash(&hash, @ptrCast(graph.index_cache[owner]), @as(u64, rows) * @as(u64, if (graph.index_fp4) 68 else 512), scratch) or
            !appendHash(&hash, @ptrCast(graph.previous_kv[owner]), 512 * 4, scratch) or
            !appendHash(&hash, @ptrCast(graph.previous_score[owner]), 512 * 4, scratch)) return null;
    }
    return hash;
}

pub fn run(owner: *const Coordinator, snapshot: *const c.ds4_session_snapshot, verify: bool, errors: []u8) c_int {
    @setFloatMode(.strict);
    const runtime = &owner.runtime;
    const executor = &runtime.prefill.executor;
    const graph = executor.graph;
    const state = runtime.state;
    const config = executor.ops.config;
    if (!runtime.ready or owner.transport.active or graph.tp_world != 1 or graph.dspark != null or
        graph.quality or graph.imatrix != null or graph.image_count != 0 or
        state.checkpoint_image_count != 0 or state.sync_image_count != 0 or
        graph.prerope != null or graph.steering_dirs != null or scheduling.replaySetting() != 0)
        return fail(errors, "SWA prototype requires native DS41 text, no TP/drafter/images/steering/archive/decoder-replay");
    const header_bytes = @sizeOf(Header);
    if (snapshot.ptr == null or snapshot.len > snapshot.cap or snapshot.len < header_bytes or snapshot.len > std.math.maxInt(usize))
        return fail(errors, "invalid SWB1 block-only snapshot");
    const source: [*]const u8 = @ptrCast(snapshot.ptr);
    var header: Header = undefined;
    for (&header, 0..) |*word, index| word.* = std.mem.readInt(u32, source[index * 4 ..][0..4], .little);
    if (header[0] != 0x31425753 or header[1] != 1 or header[7] < 256 or header[7] % 128 != 0)
        return fail(errors, "SWB1 v1 requires a >=256-token endpoint aligned to128");
    const end = header[7];
    const start = end - 128;
    // The private tag admits only the block importer. Reuse established shape,
    // token and exact-byte-count checks after validating the distinct wire tag.
    header[0] = c.DS4_SESSION_PAYLOAD_MAGIC;
    header[1] = c.DS4_SESSION_PAYLOAD_VERSION;
    const stream = c.fmemopen(@constCast(source), @intCast(snapshot.len), "rb") orelse
        return fail(errors, "cannot open SWB1 snapshot");
    defer _ = c.fclose(stream);
    if (c.fseeko(stream, header_bytes, c.SEEK_SET) != 0) return fail(errors, "cannot seek SWB1 header");
    var success = false;
    defer if (!success) owner.invalidate();
    state.checkpoint_valid = false;
    state.approximate_suffix = true;
    state.approximate_swa_replay = true;
    state.approximate_decoder_replay = false;
    state.decoder_replay_rows = 0;
    state.decoder_replay_selected = false;
    var mutated = false;
    var tokens = (Payload{ .graph = graph, .config = config }).loadBlocks(@ptrCast(stream), &header, snapshot.len - header_bytes, state.logits[0..config.shape.n_vocab], &mutated, errors) catch return 1;
    defer tokens.deinit();
    const before = if (verify) blockHash(graph, end) orelse return fail(errors, "cannot hash imported block state") else 0;
    if (verify) {
        for (graph.window) |window| if (c.ds4_gpu_tensor_fill_f32(window, std.math.nan(f32), 128 * 512) == 0)
            return fail(errors, "cannot poison missing raw state");
    }
    graph.pos = start;
    c.ds4_engram_history_reset(&graph.history);
    const ids = tokens.items();
    for (0..@min(@as(u32, 3), start)) |index|
        graph.history.tail[index] = @intCast(graph.token_map[@intCast(ids[start - 1 - index])]);
    graph.decoder_replay_floor = start;
    graph.immutable_cache_end = end;
    defer {
        graph.decoder_replay_floor = 0;
        graph.immutable_cache_end = 0;
    }
    for (ids[start..end], 0..) |token, index| {
        if (owner.control.cancel) |cancel| if (cancel(owner.control.cancel_ud)) return fail(errors, "interrupted");
        const logits = if (index == 127) state.logits else null;
        if (!executor.step(token, logits)) return fail(errors, "SWA reconstruction step failed");
    }
    if (graph.pos != end) return fail(errors, "SWA reconstruction frontier mismatch");
    if (verify and before != (blockHash(graph, end) orelse return fail(errors, "cannot hash recovered block state")))
        return fail(errors, "SWA replay modified compressed/index/carry state");
    for (state.logits[0..config.shape.n_vocab]) |value| if (!std.math.isFinite(value))
        return fail(errors, "SWA reconstruction produced nonfinite logits");
    c.ds4_tokens_copy(&state.checkpoint, &tokens.record);
    state.checkpoint_valid = true;
    state.overthink_prompt_len = @intCast(end);
    state.mtp_draft_valid = false;
    owner.clearCapture();
    success = true;
    return 0;
}
