//! Exact text checkpoints, private canonical/replay endpoints and bounded
//! suffix grafts. Each transaction retains the original lock/cancellation
//! boundary; public cache/payload owners keep all file and snapshot formats.
const std = @import("std");
const c = @import("server_persist_c");
const allocator = std.heap.c_allocator;
const match = @import("suffix_match");
const CachePolicy = @import("prompt_cache_policy.zig");

fn breakpoints(slot: *const c.server_slot) []const c_int {
    return items(c_int, slot.cache_breakpoints.v, @intCast(slot.cache_breakpoints.len));
}

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn duplicate(value: [*c]const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, text(value), 0) catch fatal()).ptr;
}
fn items(comptime T: type, pointer: [*c]T, length: usize) []T {
    return if (length == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..length];
}
fn log(kind: c.ds4_log_type, comptime pattern: [*:0]const u8, arguments: anytype) void {
    const length = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), pattern } ++ arguments);
    if (length < 0) {
        c.ds4_server_persist_log(kind, pattern);
        return;
    }
    const line = allocator.alloc(u8, @as(usize, @intCast(length)) + 1) catch fatal();
    defer allocator.free(line);
    _ = @call(.auto, c.snprintf, .{ line.ptr, line.len, pattern } ++ arguments);
    c.ds4_server_persist_log(kind, line.ptr);
}
fn nowSeconds() f64 {
    var value: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &value);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_nsec)) * 1e-9;
}
fn cancelled(slot: *const c.server_slot) bool {
    if (slot.running == null) return false;
    const running: *c.job = @ptrCast(slot.running);
    _ = c.pthread_mutex_lock(&running.mu);
    defer _ = c.pthread_mutex_unlock(&running.mu);
    return running.cancelled;
}

// These are the existing KVStore optional-trailer action contract, not field
// getters. KVStore borrows Server/wanted while this complete call is active.
fn trailerSize(context: ?*anyopaque, value: [*c]const u8, bytes: [*c]u64) callconv(.c) bool {
    if (bytes != null) bytes.* = 0;
    const runtime: *c.server = @ptrCast(@alignCast(context orelse return true));
    if (runtime.disable_exact_dsml_tool_replay or text(value).len == 0) return true;
    return c.ds4_server_tool_memory_size(&runtime.tool_mem, &runtime.tool_mu, value, bytes);
}
fn trailerWrite(context: ?*anyopaque, file: [*c]c.FILE, value: [*c]const u8, bytes: [*c]u64) callconv(.c) bool {
    if (bytes != null) bytes.* = 0;
    const runtime: *c.server = @ptrCast(@alignCast(context orelse return true));
    if (runtime.disable_exact_dsml_tool_replay or file == null or text(value).len == 0) return true;
    return c.ds4_server_tool_memory_write(&runtime.tool_mem, &runtime.tool_mu, file, value, bytes);
}
fn trailerLoad(context: ?*anyopaque, file: [*c]c.FILE, wanted: ?*const anyopaque) callconv(.c) c_int {
    const runtime: *c.server = @ptrCast(@alignCast(context orelse return 0));
    if (runtime.disable_exact_dsml_tool_replay or file == null) return 0;
    const ids: [*c]const c.stop_list = if (wanted) |pointer| @ptrCast(@alignCast(pointer)) else null;
    return c.ds4_server_tool_memory_load(&runtime.tool_mem, &runtime.tool_mu, file, ids);
}
fn hooks(runtime: *c.server) c.ds4_kvstore_trailer_hooks {
    return .{ .ud = runtime, .ext_flag = c.DS4_KVSTORE_EXT_TOOL_MAP, .serialized_size = trailerSize, .write = trailerWrite, .load = trailerLoad, .load_wanted = null };
}

pub fn store(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, tokens: [*c]const c.ds4_tokens, length: c_int, reason: [*c]const u8, visible: [*c]const u8, extension: u8, key: [*c]const u8) bool {
    const runtime = optional_runtime orelse return false;
    const slot = optional_slot orelse return false;
    if (!CachePolicy.allowsWrite(slot.prompt_cache_explicit, breakpoints(slot), length)) return false;
    var errors = std.mem.zeroes([160]u8);
    const trailer = hooks(runtime);
    _ = c.pthread_mutex_lock(&runtime.inference_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.inference_mu);
    // Text disk keys carry no image identity and cannot canonicalize a private
    // approximate graph. Both in-progress and completed vision state are gated.
    if (c.ds4_session_has_vision_state(slot.session) or c.ds4_session_is_approximate(slot.session)) return false;
    _ = c.pthread_mutex_lock(&runtime.kv_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.kv_mu);
    const published = c.ds4_kvstore_store_live_prefix_text(&runtime.kv, runtime.engine, slot.session, tokens, length, reason, visible, extension, key, &trailer, &errors, errors.len);
    if (published and slot.prompt_cache_explicit) slot.cache_written_tokens = @max(slot.cache_written_tokens, length);
    return published;
}
pub fn storeCurrent(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, reason: [*c]const u8) void {
    const runtime = optional_runtime orelse return;
    const slot = optional_slot orelse return;
    const tokens = c.ds4_session_tokens(slot.session);
    if (tokens == null) return;
    var visible: [*c]u8 = null;
    defer c.free(visible);
    var extension: u8 = 0;
    var key: [*c]const u8 = null;
    {
        _ = c.pthread_mutex_lock(&runtime.tool_mu);
        defer _ = c.pthread_mutex_unlock(&runtime.tool_mu);
        const responses = &slot.responses_live;
        const thinking = &slot.thinking_live;
        if (responses.valid and responses.live_tokens == tokens.*.len and text(responses.visible_text).len != 0) {
            visible = duplicate(responses.visible_text);
            extension = c.DS4_KVSTORE_EXT_RESPONSES_VISIBLE;
            key = "responses-visible";
        } else if (thinking.valid and !thinking.token_text_disk_key and thinking.live_tokens == tokens.*.len and text(thinking.visible_text).len != 0) {
            visible = duplicate(thinking.visible_text);
            extension = c.DS4_KVSTORE_EXT_THINKING_VISIBLE;
            key = "thinking-visible";
        }
    }
    // Keep hidden sampled KV but key its checkpoint by what clients replay.
    _ = store(runtime, slot, tokens, tokens.*.len, reason, visible, extension, key);
}
pub fn continuedTarget(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, live: c_int) c_int {
    const runtime = optional_runtime orelse return 0;
    const slot = optional_slot orelse return 0;
    if (slot.prompt_cache_explicit) return 0;
    var view = runtime.kv;
    view.continued_last_store_tokens = slot.continued_last_store_tokens;
    return c.ds4_kvstore_continued_store_target(&view, live);
}
pub fn noteStore(optional_slot: ?*c.server_slot, tokens: c_int) void {
    if (optional_slot) |slot| {
        if (tokens > slot.continued_last_store_tokens) slot.continued_last_store_tokens = tokens;
    }
}
pub fn suppress(runtime: ?*c.server, optional_slot: ?*c.server_slot, tokens: c_int) c_int {
    if (continuedTarget(runtime, optional_slot, tokens) != tokens) return -1;
    // Actual callers supply a slot, including when the continued target is 0.
    const slot = optional_slot.?;
    const old = slot.continued_last_store_tokens;
    noteStore(slot, tokens);
    return old;
}
pub fn restoreSuppressed(optional_slot: ?*c.server_slot, old: c_int, suppressed: c_int) void {
    if (optional_slot) |slot| {
        if (old >= 0 and slot.continued_last_store_tokens == suppressed) slot.continued_last_store_tokens = old;
    }
}
pub fn discard(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, path: [*c]const u8) void {
    const runtime = optional_runtime orelse return;
    const slot = optional_slot orelse return;
    if (path == null) return;
    {
        _ = c.pthread_mutex_lock(&runtime.kv_mu);
        defer _ = c.pthread_mutex_unlock(&runtime.kv_mu);
        if (c.unlink(path) == 0) {
            log(c.DS4_LOG_KVCACHE, "ds4-server: kv cache discarded reason=prefill-failed file=%s", .{path});
        } else if (c.__error().* != c.ENOENT) {
            log(c.DS4_LOG_WARNING, "ds4-server: kv cache failed to discard prefill-failed file=%s: %s", .{ path, c.strerror(c.__error().*) });
        }
    }
    slot.continued_last_store_tokens = 0;
    _ = c.pthread_mutex_lock(&runtime.inference_mu);
    defer _ = c.pthread_mutex_unlock(&runtime.inference_mu);
    c.ds4_session_invalidate(slot.session);
}
pub fn maybeStore(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot) void {
    const runtime = optional_runtime orelse return;
    const slot = optional_slot orelse return;
    const tokens = c.ds4_session_tokens(slot.session);
    if (tokens == null) return;
    const target = continuedTarget(runtime, slot, tokens.*.len);
    if (target != 0 and store(runtime, slot, tokens, target, "continued", null, 0, null)) noteStore(slot, target);
}
pub fn load(optional_runtime: ?*c.server, optional_slot: ?*c.server_slot, prompt: [*c]const u8, effective: *c.ds4_tokens, path: ?*[*c]u8, extension: ?*u8, responses: bool) c_int {
    const runtime = optional_runtime orelse return 0;
    const slot = optional_slot orelse return 0;
    if (path) |out| out.* = null;
    if (extension) |out| out.* = 0;
    const canonical: ?*const c.ds4_tokens = if (c.ds4_engine_is_deepseek41(runtime.engine)) mapped: {
        const running = slot.running;
        if (running == null or running.*.req.prompt_text != prompt) return 0;
        break :mapped &running.*.req.prompt;
    } else null;
    var result = std.mem.zeroes(c.ds4_kvstore_load_result);
    defer c.ds4_kvstore_load_result_free(&result);
    const trailer = hooks(runtime);
    const loaded = loaded: {
        _ = c.pthread_mutex_lock(&runtime.inference_mu);
        defer _ = c.pthread_mutex_unlock(&runtime.inference_mu);
        if (c.ds4_session_has_vision_state(slot.session)) c.ds4_session_invalidate(slot.session);
        _ = c.pthread_mutex_lock(&runtime.kv_mu);
        defer _ = c.pthread_mutex_unlock(&runtime.kv_mu);
        if (canonical) |tokens|
            break :loaded c.ds4_kvstore_try_load_text_mapped(&runtime.kv, runtime.engine, slot.session, prompt, tokens, if (slot.prompt_cache_explicit) &slot.cache_breakpoints else null, effective, &result, &trailer, responses);
        break :loaded c.ds4_kvstore_try_load_text(&runtime.kv, runtime.engine, slot.session, prompt, effective, &result, &trailer, responses);
    };
    if (loaded > 0) {
        if (path) |out| if (result.path != null) {
            out.* = duplicate(result.path);
        };
        if (extension) |out| out.* = result.ext_flags;
    }
    return loaded;
}

pub fn restoreTools(optional_runtime: ?*c.server, optional_messages: ?*const c.chat_msgs) void {
    const runtime = optional_runtime orelse return;
    const messages = optional_messages orelse return;
    if (runtime.disable_exact_dsml_tool_replay or !runtime.kv.enabled) return;
    // Borrow IDs from the actual messages for this directory scan only.
    var wanted: std.ArrayList([*c]u8) = .empty;
    defer wanted.deinit(allocator);
    for (items(c.chat_msg, messages.v, @intCast(messages.len))) |message| {
        addWanted(&wanted, message.tool_call_id);
        for (items([*c]u8, message.tool_call_ids, @intCast(message.tool_call_ids_len))) |id| addWanted(&wanted, id);
        for (items(c.tool_call, message.calls.v, @intCast(message.calls.len))) |call| addWanted(&wanted, call.id);
    }
    if (wanted.items.len == 0) return;
    const wanted_ids: c.stop_list = .{ .v = wanted.items.ptr, .len = @intCast(wanted.items.len), .cap = @intCast(wanted.capacity), .max_len = 0 };
    const model_id: u8 = if (runtime.engine != null) @truncate(@as(u32, @bitCast(c.ds4_engine_model_id(runtime.engine)))) else 0;
    const directory = c.opendir(runtime.kv.dir);
    if (directory == null) return;
    defer _ = c.closedir(directory);
    while (true) {
        const entry = c.readdir(directory);
        if (entry == null) break;
        var sha: [41]u8 = undefined;
        if (!c.ds4_kvstore_sha_hex_name(&entry.*.d_name, &sha)) continue;
        const path = c.ds4_kvstore_path_join(runtime.kv.dir, &entry.*.d_name);
        const file = c.fopen(path, "rb");
        c.free(path);
        if (file == null) continue;
        defer _ = c.fclose(file);
        var header = std.mem.zeroes(c.ds4_kvstore_entry);
        var text_bytes: u32 = 0;
        const valid = c.ds4_kvstore_read_header(file, &header, &text_bytes);
        const skip = @as(u64, text_bytes) +% header.payload_bytes;
        if (valid and header.model_id == model_id and header.ext_flags & c.DS4_KVSTORE_EXT_TOOL_MAP != 0 and
            skip <= std.math.maxInt(i64) and c.fseeko(file, @intCast(skip), c.SEEK_CUR) == 0)
            _ = c.ds4_server_tool_memory_load(&runtime.tool_mem, &runtime.tool_mu, file, &wanted_ids);
    }
}
fn addWanted(wanted: *std.ArrayList([*c]u8), id: [*c]u8) void {
    if (text(id).len == 0) return;
    for (wanted.items) |existing| if (std.mem.eql(u8, text(existing), text(id))) return;
    wanted.append(allocator, id) catch fatal();
}

/// Caller owns inference admission. Only restored rows beyond the usable live
/// frontier count as cache credit. A partially published failed load invalidates.
pub fn loadState(runtime: *c.server, slot: *c.server_slot, prompt: *const c.ds4_tokens) c_int {
    if (runtime.state_cache == null or prompt.len <= 0 or c.ds4_session_is_approximate(slot.session) or
        c.ds4_session_has_vision_state(slot.session) or cancelled(slot)) return 0;
    const live = c.ds4_session_pos(slot.session);
    const common = c.ds4_session_common_prefix(slot.session, prompt);
    const usable = if (common == live and prompt.len >= live) live else 0;
    var matched: u32 = 0;
    const reference = reference: {
        if (!slot.prompt_cache_explicit)
            break :reference c.ds4_state_cache_find_prefix(runtime.state_cache, &runtime.state_cache_profile, @intCast(runtime.ctx_size), prompt.v, @intCast(prompt.len), &matched) orelse return 0;
        const positions = breakpoints(slot);
        var index = positions.len;
        while (index > 0) {
            index -= 1;
            const position = positions[index];
            if (position <= 0 or position > prompt.len) continue;
            const candidate = c.ds4_state_cache_find_prefix(runtime.state_cache, &runtime.state_cache_profile, @intCast(runtime.ctx_size), prompt.v, @intCast(position), &matched) orelse continue;
            if (matched == @as(u32, @intCast(position))) break :reference candidate;
            c.ds4_state_cache_release(candidate);
        }
        return 0;
    };
    defer c.ds4_state_cache_release(reference);
    if (matched <= @as(u32, @intCast(usable))) return 0;
    var blob = std.mem.zeroes(c.ds4_state_cache_blob);
    defer c.ds4_state_cache_blob_free(&blob);
    var errors = std.mem.zeroes([160]u8);
    if (c.ds4_state_cache_materialize(reference, &blob, &errors, errors.len) != 0) {
        log(c.DS4_LOG_WARNING, "ds4-server: state cache read fallback slot=%d: %s", .{ slot.id, &errors });
        return 0;
    }
    if (cancelled(slot)) return 0;
    const snapshot: c.ds4_session_snapshot = .{ .ptr = blob.state, .len = blob.state_bytes, .cap = blob.state_bytes };
    var original = std.mem.zeroes(c.ds4_suffix_cache);
    original.state = snapshot;
    original.engine = runtime.engine;
    original.ctx_size = runtime.ctx_size;
    original.prerope = blob.prerope;
    original.prerope_bytes = blob.prerope_bytes;
    const loaded = if (blob.kind != c.DS4_STATE_CACHE_SUFFIX and blob.ctx_size == @as(u32, @intCast(runtime.ctx_size)) and blob.tokens == matched)
        (if (blob.kind == c.DS4_STATE_CACHE_REPLAY) c.ds4_session_load_replay_snapshot(slot.session, &snapshot, &errors, errors.len) else if (runtime.context_suffix_reuse and blob.prerope != null) c.ds4_session_load_suffix_prefix(slot.session, &original, &errors, errors.len) else c.ds4_session_load_snapshot(slot.session, &snapshot, &errors, errors.len))
    else
        @as(c_int, 1);
    if (loaded == 0 and c.ds4_session_pos(slot.session) == @as(c_int, @intCast(matched)) and
        c.ds4_tokens_starts_with(prompt, c.ds4_session_tokens(slot.session)))
    {
        log(c.DS4_LOG_KVCACHE, "ds4-server: state cache %s hit slot=%d tokens=%u", .{ @as([*:0]const u8, if (blob.kind == c.DS4_STATE_CACHE_REPLAY) "replay" else "exact"), slot.id, matched });
        return @as(c_int, @intCast(matched)) - usable;
    }
    c.ds4_session_invalidate(slot.session);
    log(c.DS4_LOG_WARNING, "ds4-server: state cache load fallback slot=%d: %s", .{ slot.id, @as([*c]const u8, if (errors[0] != 0) &errors else "checkpoint mismatch") });
    return 0;
}
fn enabled(name: [*:0]const u8) bool {
    const value = c.getenv(name);
    return value != null and std.mem.eql(u8, text(value), "1");
}
fn setting(name: [*:0]const u8, fallback: u32) u32 {
    const value = c.getenv(name);
    return if (value == null) fallback else std.fmt.parseInt(u32, text(value), 10) catch fallback;
}
pub fn configureContext(runtime: *c.server, slot: *c.server_slot, request: *const c.request) void {
    _ = runtime;
    slot.context_task_profile[0] = 0;
    slot.context_task_control_end = request.context_control_end;
    slot.context_task_user_begin = request.context_last_user_begin;
    if (!enabled("DS4_SERVER_CONTEXT_BOUNDARIES") or slot.context_cache_profile[0] == 0 or request.context_control_key[0] == 0 or
        request.context_last_user_begin < 0 or request.context_control_end <= request.context_last_user_begin) return;
    var hash = std.crypto.hash.Sha1.init(.{});
    hash.update("ongoing-controls-v3:");
    hash.update(text(&slot.context_cache_profile));
    hash.update(text(&request.context_control_key));
    var digest: [20]u8 = undefined;
    hash.final(&digest);
    const alphabet = "0123456789abcdef";
    for (digest, 0..) |byte, i| {
        slot.context_task_profile[2 * i] = alphabet[byte >> 4];
        slot.context_task_profile[2 * i + 1] = alphabet[byte & 15];
    }
    slot.context_task_profile[40] = 0;
}
pub fn boundaries(runtime: *c.server, slot: *c.server_slot, request: *const c.request, prompt: *const c.ds4_tokens, cached: c_int, out: [*c]c_int, capacity: usize) usize {
    if (!enabled("DS4_SERVER_CONTEXT_BOUNDARIES") or runtime.state_cache == null or !runtime.context_suffix_reuse or
        slot.context_cache_profile[0] == 0 or request.kind != c.REQ_CHAT or request.prompt_cache_explicit or
        request.previous_response_id != null or request.conversation_id != null or
        request.responses_requires_live_tool_state or request.anthropic_requires_live_tool_state or
        request.image_count != 0 or prompt.len != request.prompt.len or capacity == 0 or
        !std.mem.eql(c_int, items(c_int, prompt.v, @intCast(prompt.len)), items(c_int, request.prompt.v, @intCast(request.prompt.len)))) return 0;
    if (setting("DS4_STATE_CACHE_HOT_MB", 512) == 0) {
        log(c.DS4_LOG_WARNING, "ds4-server: automatic context boundaries require DS4_STATE_CACHE_HOT_MB > 0", .{});
        return 0;
    }
    const step: c_int = @intCast(@min(setting("DS4_SERVER_CONTEXT_BOUNDARY_MIN_TOKENS", 2048), 1_048_576));
    const limit = @min(capacity, @min(setting("DS4_SERVER_CONTEXT_BOUNDARY_LIMIT", 4), 16));
    _ = c.pthread_mutex_lock(&runtime.inference_mu);
    const live = c.ds4_session_pos(slot.session);
    const common = c.ds4_session_common_prefix(slot.session, prompt);
    _ = c.pthread_mutex_unlock(&runtime.inference_mu);
    const evaluated = CachePolicy.automaticFrontier(cached, live, common, prompt.len);
    const policy: CachePolicy.BoundaryPolicy = if (std.mem.eql(u8, text(c.getenv("DS4_SERVER_CONTEXT_BOUNDARY_POLICY")), "anchors")) .anchors else .recent;
    const kept = CachePolicy.semanticBoundaries(items(c_int, request.context_boundaries.v, @intCast(request.context_boundaries.len)), prompt.len, evaluated, step, limit, policy, items(c_int, out, capacity));
    if (kept != 0) log(c.DS4_LOG_KVCACHE, "ds4-server: context_boundaries slot=%d selected=%zu available=%d spacing=%d first=%d last=%d prompt=%d cached=%d evaluated=%d policy=%s", .{ slot.id, kept, request.context_boundaries.len, step, out[0], out[kept - 1], prompt.len, cached, evaluated, @as([*:0]const u8, if (policy == .anchors) "anchors" else "recent") });
    return kept;
}
pub fn endpoint(runtime: *c.server, slot: *c.server_slot) void {
    captureEndpoint(runtime, slot, false);
}
pub fn boundary(runtime: *c.server, slot: *c.server_slot) void {
    if (enabled("DS4_SERVER_CONTEXT_BOUNDARIES") and !c.ds4_session_is_decoder_replay(slot.session)) captureEndpoint(runtime, slot, true);
}
fn captureEndpoint(runtime: *c.server, slot: *c.server_slot, transient: bool) void {
    if (runtime.state_cache == null or cancelled(slot) or c.ds4_session_has_vision_state(slot.session)) return;
    if (!CachePolicy.allowsWrite(slot.prompt_cache_explicit, breakpoints(slot), c.ds4_session_pos(slot.session))) return;
    const replay = c.ds4_session_is_decoder_replay(slot.session);
    const approximate = c.ds4_session_is_approximate(slot.session);
    if (approximate and !replay and (!runtime.context_suffix_reuse or slot.context_cache_profile[0] == 0)) return;
    const tokens = c.ds4_session_tokens(slot.session);
    if (tokens == null or tokens.*.len <= 0 or c.ds4_session_pos(slot.session) != tokens.*.len) return;
    var captured = std.mem.zeroes(c.ds4_suffix_cache);
    defer c.ds4_suffix_cache_free(&captured);
    var errors = std.mem.zeroes([160]u8);
    var key: [41]u8 = undefined;
    if (replay) {
        var status = c.ds4_session_save_replay_snapshot(slot.session, &captured.state, &errors, errors.len);
        if (status == 0) status = c.ds4_state_cache_put_kind(runtime.state_cache, &runtime.state_cache_profile, captured.state.ptr, captured.state.len, null, 0, c.DS4_STATE_CACHE_REPLAY, &key, &errors, errors.len);
        if (status == 0 and slot.prompt_cache_explicit) slot.cache_written_tokens = @max(slot.cache_written_tokens, tokens.*.len);
        log(if (status != 0) c.DS4_LOG_WARNING else c.DS4_LOG_KVCACHE, "ds4-server: state cache replay store %s slot=%d tokens=%d%s%s", .{
            @as([*:0]const u8, if (status != 0) "skipped" else "ok"), slot.id, tokens.*.len,
            @as([*:0]const u8, if (errors[0] != 0) ": " else ""),     &errors,
        });
        return;
    }
    const started = nowSeconds();
    const status = if (runtime.context_suffix_reuse)
        c.ds4_session_capture_suffix_cache(slot.session, &captured, &errors, errors.len)
    else
        c.ds4_session_save_snapshot(slot.session, &captured.state, &errors, errors.len);
    const captured_at = nowSeconds();
    publishCapture(runtime, slot, tokens.*.len, &captured, transient, status, (captured_at - started) * 1000, &errors);
}

/// The engine collected this immutable endpoint while evaluating a larger
/// batch. Its token length belongs to the snapshot, not the later live graph.
pub fn collected(runtime: *c.server, slot: *c.server_slot, position: c_int, captured: *const c.ds4_suffix_cache) void {
    if (runtime.state_cache == null or cancelled(slot) or position <= 0 or captured.state.len == 0 or
        !CachePolicy.allowsWrite(slot.prompt_cache_explicit, breakpoints(slot), position)) return;
    var errors = std.mem.zeroes([160]u8);
    // Copy/capture happens inside prefill and is included in its timing. -1
    // distinguishes that from the separately timed ordinary snapshot action.
    publishCapture(runtime, slot, position, captured, true, 0, -1, &errors);
}

fn publishCapture(runtime: *c.server, slot: *c.server_slot, position: c_int, captured: *const c.ds4_suffix_cache, transient: bool, initial_status: c_int, capture_ms: f64, errors: []u8) void {
    const approximate = captured.approximate;
    const captured_at = nowSeconds();
    var status = initial_status;
    var key: [41]u8 = undefined;
    const kind: c.ds4_state_cache_kind = if (approximate) c.DS4_STATE_CACHE_SUFFIX else c.DS4_STATE_CACHE_NATIVE;
    const put = if (transient) &c.ds4_state_cache_put_transient else &c.ds4_state_cache_put_kind;
    if (status == 0 and !approximate) status = put(runtime.state_cache, &runtime.state_cache_profile, captured.state.ptr, captured.state.len, captured.prerope, captured.prerope_bytes, kind, &key, errors.ptr, errors.len);
    if (status == 0 and runtime.context_suffix_reuse and slot.context_cache_profile[0] != 0)
        status = put(runtime.state_cache, &slot.context_cache_profile, captured.state.ptr, captured.state.len, captured.prerope, captured.prerope_bytes, kind, &key, errors.ptr, errors.len);
    if (status == 0 and runtime.context_suffix_reuse and slot.context_task_profile[0] != 0 and slot.context_task_control_end > 0 and position >= slot.context_task_control_end)
        status = put(runtime.state_cache, &slot.context_task_profile, captured.state.ptr, captured.state.len, captured.prerope, captured.prerope_bytes, kind, &key, errors.ptr, errors.len);
    if (enabled("DS4_SERVER_CONTEXT_BOUNDARIES") or enabled("DS4_SERVER_CONTEXT_CAPTURE_PROFILE")) {
        const finished = nowSeconds();
        var cache_stats: c.ds4_state_cache_stats = undefined;
        c.ds4_state_cache_get_stats(runtime.state_cache, &cache_stats);
        log(c.DS4_LOG_KVCACHE, "ds4-server: context_capture slot=%d tokens=%d transient=%d state_bytes=%llu pre_bytes=%llu capture_ms=%.3f publish_ms=%.3f hot_bytes=%llu ssd_bytes=%llu endpoints=%llu status=%d", .{ slot.id, position, @as(c_int, @intFromBool(transient)), captured.state.len, captured.prerope_bytes, capture_ms, (finished - captured_at) * 1000, cache_stats.hot_bytes, cache_stats.ssd_bytes, cache_stats.endpoints, status });
    }
    if (status == 0 and slot.prompt_cache_explicit) slot.cache_written_tokens = @max(slot.cache_written_tokens, position);
    if (status != 0) log(c.DS4_LOG_WARNING, "ds4-server: state cache store skipped slot=%d: %s", .{ slot.id, errors.ptr });
}

/// A failed graft discards its mutated graph; ordinary prefill rebuilds it.
/// No undo snapshot or pointer to a mutable source survives this transaction.
pub fn suffix(runtime: *c.server, slot: *c.server_slot, original_prompt: *const c.ds4_tokens, padded_out: *c.ds4_tokens, stats: *c.ds4_suffix_relocation_stats, stop: *const volatile c.sig_atomic_t) c_int {
    if (slot.prompt_cache_explicit or !runtime.context_suffix_reuse or runtime.state_cache == null or slot.context_cache_profile[0] == 0 or
        original_prompt.len <= runtime.context_suffix_fresh) return 0;
    const user_id = c.ds4_token_user(runtime.engine);
    var latest_user: c_int = -1;
    var scan = original_prompt.len;
    while (scan > 0) {
        scan -= 1;
        if (original_prompt.v[@intCast(scan)] == user_id) {
            latest_user = scan;
            break;
        }
    }
    const last_role_marker = latest_user;
    const ongoing_requested = enabled("DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER") and enabled("DS4_SERVER_CONTEXT_BOUNDARIES");
    if (ongoing_requested and slot.context_task_profile[0] == 0) return 0;
    if (ongoing_requested) latest_user = slot.context_task_user_begin;
    const ongoing = ongoing_requested and slot.context_task_profile[0] != 0;
    if (latest_user < 64 and !ongoing) return 0;
    const cutoff: u32 = @intCast(@min(original_prompt.len - runtime.context_suffix_fresh, @max(latest_user, 0)));
    if (!c.ds4_server_coordinator_prefill_enter(runtime, slot, stop)) return -1;
    var admitted = true;
    defer if (admitted) c.ds4_server_coordinator_prefill_leave(runtime);
    var final_cleanup = false;
    var padded = std.mem.zeroes(c.ds4_tokens);
    defer if (!final_cleanup) c.ds4_tokens_free(&padded);
    var pad_at: c_int = -1;
    if (runtime.context_suffix_pad and original_prompt.len + 1 < runtime.ctx_size) {
        var offset: c_int = 0;
        while (offset + 1 < last_role_marker) : (offset += 1) {
            if (original_prompt.v[@intCast(offset)] != user_id) continue;
            pad_at = offset + 1;
            c.ds4_tokens_copy(&padded, original_prompt);
            c.ds4_tokens_push(&padded, runtime.context_suffix_pad_token);
            const values = items(c_int, padded.v, @intCast(padded.len));
            const insertion: usize = @intCast(pad_at);
            std.mem.copyBackwards(c_int, values[insertion + 1 ..], values[insertion .. values.len - 1]);
            values[insertion] = runtime.context_suffix_pad_token;
            break;
        }
    }
    var chosen: ?*c.ds4_state_cache_ref = null;
    defer if (!final_cleanup) c.ds4_state_cache_release(chosen);
    var best = std.mem.zeroes(c.ds4_suffix_match_plan);
    var chosen_padded = false;
    var chosen_ongoing = false;
    var chosen_durable = false;
    const profiles = [_][*c]const u8{ &slot.context_cache_profile, &slot.context_task_profile };
    for (profiles[0..@as(usize, if (ongoing) 2 else 1)], 0..) |profile, profile_index| {
        var cursor = std.mem.zeroes([41]u8);
        const source_cutoff: u32 = if (profile_index == 1) @intCast(original_prompt.len - runtime.context_suffix_fresh) else cutoff;
        while (true) {
            var ids: [*c]const c_int = null;
            var count: u32 = 0;
            var approximate = false;
            const candidate_ref = c.ds4_state_cache_next_candidate(runtime.state_cache, profile, @intCast(runtime.ctx_size), &cursor, &ids, &count, &approximate) orelse break;
            var state_bytes: u64 = 0;
            var pre_bytes: u64 = 0;
            c.ds4_state_cache_ref_sizes(candidate_ref, &state_bytes, &pre_bytes);
            var use_pad = false;
            var local = std.mem.zeroes(c.ds4_suffix_match_plan);
            if (pre_bytes != 0 and state_bytes +% pre_bytes +% (@as(u64, 8) << 20) < (@as(u64, setting("DS4_SERVER_CONTEXT_SUFFIX_STAGING_MB", 512)) << 20)) {
                _ = match.shift(c.ds4_suffix_match_plan, ids, count, original_prompt.v, @intCast(original_prompt.len), source_cutoff, runtime.context_suffix_right, &local);
                var candidate: c.ds4_suffix_match_plan = undefined;
                if (padded.len != 0 and match.shift(c.ds4_suffix_match_plan, ids, count, padded.v, @intCast(padded.len), source_cutoff + 1, runtime.context_suffix_right, &candidate) and
                    candidate.destination_start > @as(u32, @intCast(pad_at)) and candidate.reused_tokens > local.reused_tokens)
                {
                    local = candidate;
                    use_pad = true;
                }
            }
            if (local.reused_tokens > best.reused_tokens or
                (local.reused_tokens != 0 and local.reused_tokens == best.reused_tokens and chosen_padded and !use_pad))
            {
                c.ds4_state_cache_release(chosen);
                chosen = candidate_ref;
                best = local;
                chosen_padded = use_pad;
                chosen_ongoing = profile_index == 1;
                chosen_durable = c.ds4_state_cache_ref_durable(candidate_ref);
            } else c.ds4_state_cache_release(candidate_ref);
        }
    }
    if (chosen == null) {
        log(c.DS4_LOG_KVCACHE, "ds4-server: suffix exact fallback slot=%d reason=no-aligned-source", .{slot.id});
        return 0;
    }
    const prompt = if (chosen_padded) &padded else original_prompt;
    if (chosen_padded and pad_at <= latest_user) latest_user += 1;
    var blob = std.mem.zeroes(c.ds4_state_cache_blob);
    defer if (!final_cleanup) c.ds4_state_cache_blob_free(&blob);
    var prefix = prompt.*;
    prefix.len = @intCast(best.destination_start);
    var graft = prompt.*;
    const tail = prompt.len - runtime.context_suffix_fresh;
    if (tail > @as(c_int, @intCast(best.destination_end))) graft.len = tail;
    var errors = std.mem.zeroes([160]u8);
    const started = nowSeconds();
    var mutated = false;
    const result: c_int = grafting: {
        if (c.ds4_state_cache_materialize(chosen, &blob, &errors, errors.len) != 0 or cancelled(slot)) break :grafting 0;
        var source = std.mem.zeroes(c.ds4_suffix_cache);
        source.state = .{ .ptr = blob.state, .len = blob.state_bytes, .cap = blob.state_bytes };
        source.engine = runtime.engine;
        source.ctx_size = runtime.ctx_size;
        source.prerope = blob.prerope;
        source.prerope_bytes = blob.prerope_bytes;
        source.approximate = blob.approximate;
        mutated = true;
        c.ds4_session_invalidate(slot.session);
        _ = loadState(runtime, slot, &prefix);
        if (c.ds4_session_sync(slot.session, &prefix, &errors, errors.len) != 0 or cancelled(slot) or
            c.ds4_session_reuse_suffix_relocated(slot.session, &source, &graft, best.source_start, stats, &errors, errors.len) != 0 or cancelled(slot)) break :grafting 0;
        if (chosen_padded)
            slot.context_task_control_end = CachePolicy.controlEndAfterPadding(slot.context_task_control_end, pad_at);
        endpoint(runtime, slot);
        log(c.DS4_LOG_KVCACHE, "ds4-server: suffix APPROXIMATE slot=%d reused=%u fresh=%u graft_fresh=%u source=%u:%u destination=%u:%u latest_user_start=%d pre_rope=1 source_ssd=%d source_approximate=%d ongoing_user=%d prefix_pad=%d right_shift=%d %.3f ms", .{ slot.id, stats.reuse.reused_tokens, best.fresh_tokens, stats.reuse.fresh_tail_tokens, stats.source_start, stats.source_end, stats.destination_start, stats.destination_end, latest_user, @as(c_int, @intFromBool(chosen_durable)), @as(c_int, @intFromBool(blob.approximate)), @as(c_int, @intFromBool(chosen_ongoing)), @as(c_int, @intFromBool(chosen_padded)), @as(c_int, @intFromBool(stats.destination_start > stats.source_start)), (nowSeconds() - started) * 1000 });
        break :grafting 1;
    };
    if (result == 0) {
        if (mutated) c.ds4_session_invalidate(slot.session);
        log(c.DS4_LOG_KVCACHE, "ds4-server: suffix exact fallback slot=%d reason=%s", .{ slot.id, @as([*c]const u8, if (errors[0] != 0) &errors else "cancelled") });
    }
    if (result != 0 and chosen_padded) {
        padded_out.* = padded;
        padded = std.mem.zeroes(c.ds4_tokens);
    }
    // Cleanup/admission release precede the final cancellation query in C.
    c.ds4_state_cache_blob_free(&blob);
    c.ds4_state_cache_release(chosen);
    chosen = null;
    c.ds4_tokens_free(&padded);
    c.ds4_server_coordinator_prefill_leave(runtime);
    admitted = false;
    final_cleanup = true;
    return if (cancelled(slot)) -1 else result;
}
