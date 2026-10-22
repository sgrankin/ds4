//! Routing and execution consume one live-state decision in the same tier
//! order. The probe never tokenizes/detokenizes: workers materialize its chosen
//! suffix and refresh cached text only after mutating the session.
const std = @import("std");
const c = @import("server_reuse_c");
const allocator = std.heap.c_allocator;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn equal(pointer: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, text(pointer), expected);
}
fn items(comptime T: type, pointer: [*c]T, length: usize) []T {
    return if (length == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..length];
}
fn marker(req: *const c.request, position: usize) [*:0]const u8 {
    return @ptrCast(&req.image_markers[position]);
}
fn contains(ids: *const c.stop_list, id: [*c]const u8) bool {
    if (text(id).len == 0) return false;
    for (items([*c]u8, ids.v, @intCast(ids.len))) |existing|
        if (equal(existing, text(id))) return true;
    return false;
}
fn clearIds(ids: *c.stop_list) void {
    for (items([*c]u8, ids.v, @intCast(ids.len))) |id| c.free(id);
    ids.len = 0;
    ids.max_len = 0;
}
fn addId(ids: *c.stop_list, id: [*c]const u8) void {
    if (text(id).len == 0 or contains(ids, id)) return;
    const owned = allocator.dupeSentinel(u8, text(id), 0) catch fatal();
    var list: std.ArrayList([*c]u8) = .empty;
    list.items = items([*c]u8, ids.v, @intCast(ids.len));
    list.capacity = @intCast(ids.cap);
    list.append(allocator, owned.ptr) catch fatal();
    ids.v = list.items.ptr;
    ids.len = @intCast(list.items.len);
    ids.cap = @intCast(list.capacity);
    ids.max_len = @max(ids.max_len, owned.len);
}

/// Normalize only actual request-local image nonce spans and preserve their
/// offsets. Literal client text cannot impersonate an image fingerprint.
pub fn visibleKey(optional_req: ?*const c.request, value: [*c]const u8, images: *c.visible_image_key) [*c]u8 {
    images.* = std.mem.zeroes(c.visible_image_key);
    const req = optional_req orelse return null;
    if (value == null or req.image_count > 16 or (req.image_count != 0 and req.image_markers == null)) return null;
    const raw = text(value);
    const owned = allocator.dupeSentinel(u8, raw, 0) catch fatal();
    var success = false;
    defer if (!success) allocator.free(owned);
    var cursor: usize = 0;
    for (0..req.image_count) |position| {
        const spelling = text(marker(req, position));
        const found = std.mem.findPos(u8, raw, cursor, spelling) orelse return null;
        if (spelling.len == 0) return null;
        images.offsets[images.count] = found;
        images.count += 1;
        @memset(owned[found .. found + spelling.len], 0x1e);
        cursor = found + spelling.len;
    }
    success = true;
    return owned.ptr;
}
pub fn imagesMatch(incoming: *const c.visible_image_key, old: *const c.visible_image_key, boundary: usize) bool {
    if (incoming.count < old.count) return false;
    for (0..old.count) |position|
        if (incoming.offsets[position] != old.offsets[position]) return false;
    return incoming.count == old.count or incoming.offsets[old.count] >= boundary;
}

/// Suffix sources and private decoder replay use distinct profile identities;
/// the live token/history checks still apply after this immutable key matches.
pub fn profile(optional_runtime: ?*const c.server, optional_req: ?*const c.request, out: []u8) void {
    @memset(out, 0);
    const runtime = optional_runtime orelse return;
    const req = optional_req orelse return;
    if (req.kind != c.REQ_CHAT or req.image_count != 0) return;
    var rows: [*c]const u8 = null;
    if (runtime.context_suffix_reuse) {
        if (text(req.prompt_cache_key).len == 0) return;
    } else {
        rows = c.getenv("DS4_V41_DECODER_REPLAY_ROWS");
        if (!equal(rows, "128") and !equal(rows, "256")) return;
    }
    var identity: std.ArrayList(u8) = .empty;
    defer identity.deinit(allocator);
    identity.appendSlice(allocator, text(&runtime.state_cache_profile)) catch fatal();
    if (runtime.context_suffix_reuse) {
        if (runtime.context_suffix_right) identity.appendSlice(allocator, ":right-v1") catch fatal();
        identity.appendSlice(allocator, ":suffix:") catch fatal();
        identity.appendSlice(allocator, text(req.prompt_cache_key)) catch fatal();
    } else {
        identity.appendSlice(allocator, ":decoder-replay-v1:") catch fatal();
        identity.appendSlice(allocator, text(rows)) catch fatal();
        const selected = c.getenv("DS4_V41_DECODER_REPLAY_SELECTED");
        identity.appendSlice(allocator, if (equal(selected, "1")) ":selected1" else ":selected0") catch fatal();
        if (text(req.prompt_cache_key).len != 0) {
            identity.appendSlice(allocator, ":hint:") catch fatal();
            identity.appendSlice(allocator, text(req.prompt_cache_key)) catch fatal();
        }
    }
    c.ds4_kvstore_sha1_bytes_hex(identity.items.ptr, identity.items.len, out.ptr);
}

pub fn clearTool(optional_state: ?*c.live_tool_state) void {
    const state = optional_state orelse return;
    clearIds(&state.call_ids);
    c.free(state.visible_text);
    state.visible_text = null;
    state.visible_len = 0;
    state.images = std.mem.zeroes(c.visible_image_key);
    state.valid = false;
    state.live_tokens = 0;
}
pub fn freeTool(optional_state: ?*c.live_tool_state) void {
    const state = optional_state orelse return;
    clearTool(state);
    c.free(@ptrCast(state.call_ids.v));
    state.* = std.mem.zeroes(c.live_tool_state);
}
pub fn clearVisible(optional_state: ?*c.visible_live_state) void {
    const state = optional_state orelse return;
    c.free(state.visible_text);
    state.visible_text = null;
    state.visible_len = 0;
    state.images = std.mem.zeroes(c.visible_image_key);
    state.live_tokens = 0;
    state.valid = false;
    state.token_text_disk_key = false;
}
pub fn freeVisible(optional_state: ?*c.visible_live_state) void {
    const state = optional_state orelse return;
    clearVisible(state);
    state.* = std.mem.zeroes(c.visible_live_state);
}
pub fn clear(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, api: c.api_style) void {
    const runtime = optional_runtime orelse return;
    const slot = optional_slot orelse return;
    _ = c.pthread_mutex_lock(&runtime.tool_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
    switch (api) {
        c.API_RESPONSES => clearTool(&slot.responses_live),
        c.API_ANTHROPIC => clearTool(&slot.anthropic_live),
        else => clearVisible(&slot.thinking_live),
    }
}
pub fn remember(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, api: c.api_style, visible: [*c]const u8, optional_calls: ?*const c.tool_calls, req: ?*const c.request) void {
    const runtime = optional_runtime orelse return;
    const slot = optional_slot orelse return;
    if (api == c.API_ANTHROPIC) {
        const calls = optional_calls orelse return;
        if (calls.len == 0) return;
        _ = c.pthread_mutex_lock(&runtime.tool_mu);
        defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
        clearTool(&slot.anthropic_live);
        for (items(c.tool_call, calls.v, @intCast(calls.len))) |call| addId(&slot.anthropic_live.call_ids, call.id);
        slot.anthropic_live.live_tokens = c.ds4_session_pos(slot.session);
        slot.anthropic_live.valid = slot.anthropic_live.call_ids.len > 0;
        return;
    }
    if (text(visible).len == 0) return;
    var images: c.visible_image_key = undefined;
    const key = visibleKey(req, visible, &images);
    _ = c.pthread_mutex_lock(&runtime.tool_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
    if (api == c.API_RESPONSES) {
        clearTool(&slot.responses_live);
        slot.responses_live.visible_text = key;
        slot.responses_live.visible_len = text(key).len;
        slot.responses_live.images = images;
        if (optional_calls) |calls| {
            for (items(c.tool_call, calls.v, @intCast(calls.len))) |call| addId(&slot.responses_live.call_ids, call.id);
        }
        slot.responses_live.live_tokens = c.ds4_session_pos(slot.session);
        slot.responses_live.valid = key != null;
    } else {
        clearVisible(&slot.thinking_live);
        slot.thinking_live.visible_text = key;
        slot.thinking_live.visible_len = text(key).len;
        slot.thinking_live.images = images;
        slot.thinking_live.live_tokens = c.ds4_session_pos(slot.session);
        slot.thinking_live.valid = key != null;
    }
}
pub fn matchesLocked(slot: *c.server_slot, api: c.api_style, ids: *const c.stop_list, live_tokens: c_int) bool {
    const state = if (api == c.API_RESPONSES) &slot.responses_live else &slot.anthropic_live;
    if (!state.valid or state.live_tokens != live_tokens or state.call_ids.len != ids.len) return false;
    for (items([*c]u8, ids.v, @intCast(ids.len))) |id|
        if (!contains(&state.call_ids, id)) return false;
    return true;
}
pub fn matches(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, api: c.api_style, optional_ids: ?*const c.stop_list, live_tokens: c_int) bool {
    const runtime = optional_runtime orelse return false;
    const slot = optional_slot orelse return false;
    const ids = optional_ids orelse return false;
    if (ids.len == 0) return false;
    _ = c.pthread_mutex_lock(&runtime.tool_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
    return matchesLocked(slot, api, ids, live_tokens);
}
pub fn canonical(optional_runtime: ?*const c.server, optional_req: ?*const c.request) bool {
    const runtime = optional_runtime orelse return false;
    const req = optional_req orelse return false;
    return runtime.context_suffix_reuse and runtime.context_suffix_canonical and
        req.kind == c.REQ_CHAT and req.api == c.API_RESPONSES and req.image_count == 0 and
        req.previous_response_id == null and req.conversation_id == null and
        !req.responses_requires_live_tool_state and text(req.prompt_cache_key).len != 0;
}
pub fn rewindTarget(can_rewind: bool, old_pos: c_int, prompt_len: c_int, common: c_int) c_int {
    if (!can_rewind or prompt_len <= 1 or prompt_len >= old_pos or common != prompt_len) return -1;
    return prompt_len - 1;
}
fn visibleMatches(req: *const c.request, value: [*c]const u8, old_text: [*c]const u8, old_length: usize, old_images: *const c.visible_image_key) bool {
    var images: c.visible_image_key = undefined;
    const key = visibleKey(req, value, &images);
    if (key == null) return false;
    defer c.free(key);
    const length = text(key).len;
    return old_text != null and old_length < length and imagesMatch(&images, old_images, old_length) and
        c.ds4_kvstore_byte_prefix_match(key, length, old_text, old_length);
}

/// Requires tool_mu held. Image fingerprints gate EVERY tier, so a nonce/text
/// coincidence cannot route another image's request to this sampled frontier.
/// Tier order is shared by dispatch and execution, never separately recreated.
pub fn probeLocked(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, optional_req: ?*const c.request) c.slot_reuse {
    const none = std.mem.zeroes(c.slot_reuse);
    const runtime = optional_runtime orelse return none;
    const slot = optional_slot orelse return none;
    const req = optional_req orelse return none;
    if (slot.session == null) return none;
    if (c.ds4_session_is_approximate(slot.session)) {
        if (req.previous_response_id != null and !equal(req.previous_response_id, text(&slot.last_response_id))) return none;
        var expected: [41]u8 = undefined;
        profile(runtime, req, &expected);
        if (expected[0] == 0 or !equal(&expected, text(&slot.context_cache_profile))) return none;
    }
    if (!c.ds4_session_checkpoint_valid(slot.session)) return none;
    const live_pointer = c.ds4_session_tokens(slot.session);
    if (live_pointer == null or live_pointer.*.len <= 0) return none;
    const live: *const c.ds4_tokens = @ptrCast(live_pointer);
    if (!c.ds4_session_vision_fingerprint_prefix_matches(slot.session, req.images, req.image_count)) return none;
    const is_canonical = canonical(runtime, req);

    // A scoped approximate frontier must match its full visible history.
    // Canonical full-input clients otherwise bypass ID-only/text splices.
    if (req.api == c.API_RESPONSES and req.prompt_text != null and
        (!is_canonical or c.ds4_session_is_approximate(slot.session)))
    {
        const old = &slot.responses_live;
        if (old.valid and old.live_tokens == live.len and
            visibleMatches(req, req.prompt_text, old.visible_text, old.visible_len, &old.images))
            return .{ .kind = c.REUSE_RESPONSES_VISIBLE, .reuse_tokens = live.len, .suffix_off = old.visible_len, .matched_ids = 0 };
    }
    if (!is_canonical and req.api == c.API_RESPONSES and req.previous_response_id == null and
        req.conversation_id == null and req.responses_live_suffix_text != null and
        req.responses_live_call_ids.len > 0 and matchesLocked(slot, c.API_RESPONSES, &req.responses_live_call_ids, live.len))
        return .{ .kind = c.REUSE_RESPONSES_TOOL_OUTPUT, .reuse_tokens = live.len, .suffix_off = 0, .matched_ids = req.responses_live_call_ids.len };
    if (req.api == c.API_ANTHROPIC and req.anthropic_live_suffix_text != null and
        req.anthropic_live_call_ids.len > 0 and matchesLocked(slot, c.API_ANTHROPIC, &req.anthropic_live_call_ids, live.len))
        return .{ .kind = c.REUSE_ANTHROPIC_TOOL_OUTPUT, .reuse_tokens = live.len, .suffix_off = 0, .matched_ids = req.anthropic_live_call_ids.len };

    const common = c.ds4_session_common_prefix(slot.session, &req.prompt);
    const token_images = c.ds4_session_vision_prefix_matches(slot.session, req.images, req.image_count);
    const rewind = rewindTarget(c.ds4_engine_is_glm_dsa(runtime.engine), live.len, req.prompt.len, common);
    if (rewind >= 0 and token_images)
        return .{ .kind = c.REUSE_MEMORY_REWIND, .reuse_tokens = rewind, .suffix_off = 0, .matched_ids = 0 };
    if (common == live.len and req.prompt.len >= live.len and token_images)
        return .{ .kind = c.REUSE_MEMORY_TOKEN, .reuse_tokens = common, .suffix_off = 0, .matched_ids = 0 };
    if (req.kind == c.REQ_CHAT and req.api != c.API_RESPONSES and req.prompt_text != null) {
        const old = &slot.thinking_live;
        if (old.valid and old.live_tokens == live.len and
            visibleMatches(req, req.prompt_text, old.visible_text, old.visible_len, &old.images))
            return .{ .kind = c.REUSE_THINKING_VISIBLE, .reuse_tokens = live.len, .suffix_off = old.visible_len, .matched_ids = 0 };
    }
    if (!is_canonical and req.prompt_text != null and slot.live_text != null and
        slot.live_text_pos == live.len and slot.live_text_len <= text(req.prompt_text).len and
        c.ds4_kvstore_byte_prefix_match(req.prompt_text, text(req.prompt_text).len, slot.live_text, slot.live_text_len))
        return .{ .kind = c.REUSE_MEMORY_TEXT, .reuse_tokens = live.len, .suffix_off = slot.live_text_len, .matched_ids = 0 };
    return none;
}
pub fn probe(runtime: *c.server, slot: ?*c.server_slot, req: ?*const c.request) c.slot_reuse {
    _ = c.pthread_mutex_lock(&runtime.tool_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
    return probeLocked(runtime, slot, req);
}

/// Retain sampled prefix token IDs and tokenize only the visible suffix. New
/// images reuse their exact wrapper/embedding tokens with rebased positions;
/// committing output waits for a complete image-prefix identity check.
pub fn suffix(runtime: *c.server, slot: *c.server_slot, req: *const c.request, suffix_text: [*c]const u8, out: *c.ds4_tokens) bool {
    const live = c.ds4_session_tokens(slot.session);
    if (live == null or suffix_text == null or req.image_count > 16) return false;
    var spans: [16]c.ds4_vision_span = undefined;
    var old_count = req.image_count;
    const raw = text(suffix_text);
    for (0..req.image_count) |position| {
        if (req.image_markers == null) return false;
        spans[position] = req.images[position];
        if (old_count == req.image_count and std.mem.find(u8, raw, text(marker(req, position))) != null) old_count = position;
    }
    if (!c.ds4_session_rebase_vision_state(slot.session, &spans, old_count)) return false;
    var prompt = std.mem.zeroes(c.ds4_tokens);
    var committed = false;
    defer if (!committed) c.ds4_tokens_free(&prompt);
    c.ds4_tokens_copy(&prompt, live);
    var cursor: usize = 0;
    const wrapper: c_int = if (c.ds4_engine_is_glm_dsa(runtime.engine)) 1 else 0;
    for (old_count..req.image_count) |position| {
        const spelling = text(marker(req, position));
        const found = std.mem.findPos(u8, raw, cursor, spelling) orelse return false;
        const start: i64 = @as(i64, req.images[position].token_start) - wrapper;
        const end = @as(u64, req.images[position].token_start) + req.images[position].embedding.token_count + @as(u64, @intCast(wrapper));
        if (start < 0 or end > @as(u64, @intCast(req.prompt.len))) return false;
        const prefix = allocator.dupeSentinel(u8, raw[cursor..found], 0) catch fatal();
        defer allocator.free(prefix);
        c.ds4_tokenize_rendered_chat(runtime.engine, prefix.ptr, &prompt);
        spans[position].token_start = @intCast(prompt.len + wrapper);
        var token = start;
        while (token < @as(i64, @intCast(end))) : (token += 1) c.ds4_tokens_push(&prompt, req.prompt.v[@intCast(token)]);
        cursor = found + spelling.len;
    }
    c.ds4_tokenize_rendered_chat(runtime.engine, suffix_text + cursor, &prompt);
    if (!c.ds4_session_vision_prefix_matches(slot.session, &spans, req.image_count)) return false;
    for (0..req.image_count) |position| req.images[position].token_start = spans[position].token_start;
    c.ds4_tokens_free(out);
    out.* = prompt;
    committed = true;
    return true;
}
pub fn unavailable(optional_req: ?*const c.request, materialized: bool) bool {
    const req = optional_req orelse return false;
    if (materialized) return false;
    return switch (req.api) {
        c.API_ANTHROPIC => req.anthropic_requires_live_tool_state,
        c.API_RESPONSES => req.responses_requires_live_tool_state,
        else => false,
    };
}
pub fn refresh(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot) void {
    const slot = optional_slot orelse return;
    c.free(slot.live_text);
    slot.live_text = null;
    slot.live_text_len = 0;
    slot.live_text_pos = 0;
    const runtime = optional_runtime orelse return;
    if (slot.session == null or !c.ds4_session_checkpoint_valid(slot.session)) return;
    const live = c.ds4_session_tokens(slot.session);
    if (live == null or live.*.len <= 0) return;
    slot.live_text = c.ds4_kvstore_render_tokens_text(runtime.engine, live, &slot.live_text_len);
    if (slot.live_text != null) slot.live_text_pos = live.*.len;
}
