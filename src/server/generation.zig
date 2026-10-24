//! One request's prefill, sampled decoding, recovery and durable publication.
//! Server/Slot/Job are the actual borrowed records. Engine and Session remain
//! opaque; only complete trace/log I/O and other-family rendering stay in C.
const std = @import("std");
const c = @import("server_generation_c");
const Canonical = @import("canonical.zig");
const QwenPrompt = @import("qwen_prompt");
const allocator = std.heap.c_allocator;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn items(comptime T: type, pointer: [*c]T, length: usize) []T {
    return if (length == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..length];
}
fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn duplicate(bytes: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, bytes, 0) catch fatal()).ptr;
}
fn same(pointer: [*c]const u8, value: []const u8) bool {
    return std.mem.eql(u8, text(pointer), value);
}
fn ptr(pointer: [*c]const u8) [*c]const u8 {
    return if (pointer == null) "" else pointer;
}
fn flag(value: bool) c_int {
    return @intFromBool(value);
}
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}
fn report(out: []u8, comptime pattern: [*:0]const u8, arguments: anytype) void {
    if (out.len != 0) _ = @call(.auto, c.snprintf, .{ out.ptr, out.len, pattern } ++ arguments);
}
fn formatted(comptime pattern: [*:0]const u8, arguments: anytype) [:0]u8 {
    const length = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), pattern } ++ arguments);
    if (length < 0) return allocator.dupeSentinel(u8, text(pattern), 0) catch fatal();
    const line = allocator.allocSentinel(u8, @intCast(length), 0) catch fatal();
    _ = @call(.auto, c.snprintf, .{ line.ptr, line.len + 1, pattern } ++ arguments);
    return line;
}
fn log(kind: c.ds4_log_type, comptime pattern: [*:0]const u8, arguments: anytype) void {
    const line = formatted(pattern, arguments);
    defer allocator.free(line);
    c.ds4_server_persist_log(kind, line.ptr);
}

// Buffers use libc storage, so the public render/output actions can grow the
// same owner. Preserve a nonnull allocated-empty base when adopting capacity.
fn append(out: *c.buf, value: []const u8) void {
    var list: std.ArrayList(u8) = .empty;
    list.items = items(u8, out.ptr, out.len);
    list.capacity = out.cap;
    list.appendSlice(allocator, value) catch fatal();
    list.append(allocator, 0) catch fatal();
    list.items.len -= 1;
    out.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
}
fn puts(out: *c.buf, value: [*c]const u8) void {
    append(out, text(value));
}
fn freeBuffer(out: *c.buf) void {
    c.free(out.ptr);
    out.* = zero(c.buf);
}
fn take(out: *c.buf) [*c]u8 {
    const result = if (out.ptr == null) duplicate("") else out.ptr;
    out.* = zero(c.buf);
    return result;
}
fn quote(out: *c.buf, value: [*c]const u8) void {
    append(out, "\"");
    for (text(value)) |byte| switch (byte) {
        '"' => append(out, "\\\""),
        '\\' => append(out, "\\\\"),
        '\n' => append(out, "\\n"),
        '\r' => append(out, "\\r"),
        '\t' => append(out, "\\t"),
        else => if (byte < 0x20) {
            const hex = "0123456789abcdef";
            append(out, &.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15] });
        } else append(out, &.{byte}),
    };
    append(out, "\"");
}
fn trim(out: *c.buf, value: [*c]const u8) void {
    const bytes = text(value);
    var start: usize = 0;
    var end = bytes.len;
    while (start < end and c.isspace(bytes[start]) != 0) start += 1;
    while (end > start and c.isspace(bytes[end - 1]) != 0) end -= 1;
    append(out, bytes[start..end]);
}
fn freeCalls(calls: *c.tool_calls) void {
    for (items(c.tool_call, calls.v, @intCast(calls.len))) |call| {
        c.free(call.id);
        c.free(call.name);
        c.free(call.arguments);
    }
    c.free(calls.v);
    c.free(calls.raw_tool_text);
    calls.* = zero(c.tool_calls);
}
fn clearLive(runtime: *c.server, slot: *c.server_slot) void {
    c.ds4_server_reuse_clear(runtime, slot, c.API_RESPONSES);
    c.ds4_server_reuse_clear(runtime, slot, c.API_ANTHROPIC);
    c.ds4_server_reuse_clear(runtime, slot, c.API_OPENAI);
}
pub fn cancelled(context: ?*anyopaque) callconv(.c) bool {
    const job: *c.job = @ptrCast(@alignCast(context orelse return false));
    _ = c.pthread_mutex_lock(&job.mu);
    defer _ = c.pthread_mutex_unlock(&job.mu);
    return job.cancelled;
}
fn markCancelled(job: [*c]c.job) void {
    if (job == null) return;
    _ = c.pthread_mutex_lock(&job.*.mu);
    job.*.cancelled = true;
    _ = c.pthread_mutex_unlock(&job.*.mu);
}
fn slotCancelled(slot: *c.server_slot) bool {
    return cancelled(if (slot.running == null) null else @ptrCast(slot.running));
}
fn stopped(stop: [*c]const volatile c.sig_atomic_t) bool {
    return stop.* != 0;
}
fn randomBytes(destination: []u8) bool {
    const descriptor = c.open("/dev/urandom", c.O_RDONLY);
    if (descriptor < 0) return false;
    defer _ = c.close(descriptor);
    var done: usize = 0;
    while (done < destination.len) {
        const count = c.read(descriptor, destination.ptr + done, destination.len - done);
        if (count < 0 and c.__error().* == c.EINTR) continue;
        if (count <= 0) return false;
        done += @intCast(count);
    }
    return true;
}
fn contextSpan(out: []u8, cached: c_int, prompt: c_int) void {
    const suffix: c_int = @max(0, prompt - cached);
    report(out, "%d..%d:%d", .{ cached, prompt, suffix });
}
fn flags(out: []u8, responses: bool, tools: bool, thinking: bool, start: bool, end: bool) void {
    var value = zero(c.buf);
    defer freeBuffer(&value);
    const enabled = [_]bool{ responses, tools, thinking, start, end };
    const names = [_][]const u8{ "RESPPROTO", "TOOLS", "THINKING", "DSML_START", "DSML_END" };
    for (enabled, names) |yes, name| if (yes) {
        if (value.len != 0) append(&value, " ");
        append(&value, name);
    };
    report(out, "%s", .{ptr(value.ptr)});
}
pub fn decodeBudget(requested: c_int, generated: c_int, room: c_int) c_int {
    return if (requested <= generated or room <= 0) 0 else @min(requested - generated, room);
}
pub fn reasoningDue(limit: c_int, count: c_int, forced: bool, inside: bool, output_left: c_int, context_left: c_int, safe: bool) bool {
    return limit > 0 and count >= limit and !forced and inside and safe and output_left >= 2 and context_left >= 2;
}
pub fn feedThinking(state: [*c]c.thinking_state, bytes: [*c]const u8, length: usize) void {
    if (state == null or bytes == null) return;
    for (bytes[0..length]) |byte| {
        if (state.*.tail_len == state.*.tail.len) {
            std.mem.copyForwards(u8, state.*.tail[0 .. state.*.tail.len - 1], state.*.tail[1..]);
            state.*.tail_len -= 1;
        }
        state.*.tail[@intCast(state.*.tail_len)] = byte;
        state.*.tail_len += 1;
        const tail = state.*.tail[0..@intCast(state.*.tail_len)];
        if (std.mem.endsWith(u8, tail, "<think>")) state.*.inside = true else if (std.mem.endsWith(u8, tail, "</think>")) state.*.inside = false;
    }
}
pub fn thinkingFromPrompt(request: [*c]const c.request) c.thinking_state {
    var state = zero(c.thinking_state);
    if (request != null and request.*.prompt_text != null) feedThinking(&state, request.*.prompt_text, text(request.*.prompt_text).len) else if (request != null and c.ds4_think_mode_enabled(request.*.think_mode)) state.inside = true;
    return state;
}
fn checkpointInside(engine: ?*c.ds4_engine, session: ?*c.ds4_session, close: c_int) bool {
    const tokens = c.ds4_session_tokens(session);
    if (tokens == null) return false;
    var position = tokens.*.len;
    while (position > 0) {
        position -= 1;
        const token = tokens.*.v[@intCast(position)];
        if (c.ds4_token_is_thinking_control(engine, token)) return token != close;
    }
    return false;
}
pub fn completeThinking(syntax: c.server_model_syntax, raw: [*c]const u8, length: usize, cursor: [*c]usize) bool {
    if (raw == null or cursor == null) return false;
    cursor.* = @min(cursor.*, length);
    const opening = c.ds4_server_reply_tool_start(raw + cursor.*);
    if (opening == null) {
        cursor.* = length - @min(length, 80);
        return false;
    }
    cursor.* = @intFromPtr(opening) - @intFromPtr(raw);
    if (c.ds4_server_reply_tool_end(opening) == null) return false;
    var content: [*c]u8 = null;
    var reasoning: [*c]u8 = null;
    var calls = zero(c.tool_calls);
    defer {
        c.free(content);
        c.free(reasoning);
        freeCalls(&calls);
    }
    return c.ds4_server_reply_parse(syntax, opening, false, &content, &reasoning, &calls) and calls.len > 0;
}
pub fn rememberThinking(request: [*c]const c.request, state: [*c]const c.thinking_state, finish: [*c]const u8) bool {
    if (request == null or request.*.kind != c.REQ_CHAT) return false;
    const qwen = request.*.model_syntax == c.SERVER_MODEL_SYNTAX_QWEN and request.*.api != c.API_RESPONSES and request.*.api != c.API_ANTHROPIC;
    if ((request.*.has_tools or request.*.prompt_preserves_reasoning) and !qwen) return false;
    return c.ds4_think_mode_enabled(request.*.think_mode) and !same(finish, "error") and !same(finish, "length") and (state == null or !state.*.inside);
}
pub fn canonicalizeNeeded(runtime: [*c]const c.server, calls: [*c]const c.tool_calls) bool {
    return calls != null and calls.*.len != 0 and !(runtime != null and !runtime.*.disable_exact_dsml_tool_replay and text(calls.*.raw_tool_text).len != 0);
}
pub fn multimodalFrontier(live: c_int, common: c_int, length: c_int, image_match: bool) c_int {
    return if (common == live and length >= live and image_match) live else 0;
}

pub fn assignIds(runtime: [*c]c.server, calls: [*c]c.tool_calls, api: c.api_style) void {
    if (calls == null) return;
    const entries = items(c.tool_call, calls.*.v, @intCast(calls.*.len));
    for (entries, 0..) |*call, index| {
        if (text(call.id).len != 0) continue;
        var id: [64]u8 = undefined;
        while (true) {
            c.ds4_server_protocol_random_tool_id(&id, id.len, api);
            var duplicate_id = false;
            for (entries[0..index]) |previous| if (previous.id != null and same(previous.id, text(&id))) {
                duplicate_id = true;
                break;
            };
            if (duplicate_id) continue;
            const known = runtime != null and !runtime.*.disable_exact_dsml_tool_replay and c.ds4_server_tool_memory_contains(&runtime.*.tool_mem, &runtime.*.tool_mu, &id);
            if (!known) break;
        }
        call.id = duplicate(text(&id));
    }
}

/// Synchronize one resident slot without monopolizing the model executor.
/// A restored endpoint is never truncated to the first scheduling quantum.
pub fn sync(runtime: ?*c.server, member: ?*c.server_slot, prompt: [*c]const c.ds4_tokens, credits: [*c]c_int, errors: []u8, stop: [*c]const volatile c.sig_atomic_t) c_int {
    const srv = runtime orelse return 1;
    const slot = member orelse return 1;
    if (prompt == null) return 1;
    if (!srv.batched_mode) {
        if (!c.ds4_server_coordinator_prefill_enter(srv, slot, stop)) return c.DS4_SESSION_SYNC_INTERRUPTED;
        defer c.ds4_server_coordinator_prefill_leave(srv);
        const restored = c.ds4_server_persist_load_state(srv, slot, prompt);
        if (credits != null) credits.* += restored;
        const anchor = prompt.*.len - srv.context_suffix_fresh;
        var result: c_int = 0;
        if (srv.context_suffix_reuse and slot.context_cache_profile[0] != 0 and anchor > 0 and c.ds4_session_pos(slot.session) < anchor) {
            var prefix = prompt.*;
            prefix.len = anchor;
            result = c.ds4_session_sync(slot.session, &prefix, errors.ptr, errors.len);
            if (result == 0) c.ds4_server_persist_endpoint(srv, slot);
        }
        if (result == 0) result = c.ds4_session_sync(slot.session, prompt, errors.ptr, errors.len);
        if (result == 0) c.ds4_server_persist_endpoint(srv, slot);
        return result;
    }
    _ = c.pthread_mutex_lock(&srv.inference_mu);
    const live = c.ds4_session_pos(slot.session);
    const common = c.ds4_session_common_prefix(slot.session, prompt);
    _ = c.pthread_mutex_unlock(&srv.inference_mu);
    var done: c_int = if (c.ds4_engine_is_flint(srv.engine)) common else if (common == live and prompt.*.len >= live) live else 0;
    var called = false;
    while (!stopped(stop) and !slotCancelled(slot) and (!called or done < prompt.*.len)) {
        const quantum = c.ds4_server_coordinator_quantum(srv);
        if (!c.ds4_server_coordinator_prefill_enter(srv, slot, stop)) return c.DS4_SESSION_SYNC_INTERRUPTED;
        if (!called and srv.state_cache != null) {
            const restored = c.ds4_server_persist_load_state(srv, slot, prompt);
            if (credits != null) credits.* += restored;
            const current = c.ds4_session_pos(slot.session);
            const shared = c.ds4_session_common_prefix(slot.session, prompt);
            done = if (shared == current and prompt.*.len >= current) current else 0;
        }
        var target = done +% quantum;
        if (target > prompt.*.len or target < done) target = prompt.*.len;
        const anchor = prompt.*.len - srv.context_suffix_fresh;
        if (srv.context_suffix_reuse and slot.context_cache_profile[0] != 0 and target == prompt.*.len and anchor > done) target = anchor;
        if (target <= 0) target = prompt.*.len;
        var prefix = prompt.*;
        prefix.len = target;
        const result = c.ds4_session_sync(slot.session, &prefix, errors.ptr, errors.len);
        if (result == 0) {
            done = c.ds4_session_pos(slot.session);
            c.ds4_server_persist_endpoint(srv, slot);
        }
        c.ds4_server_coordinator_prefill_leave(srv);
        called = true;
        if (result != 0) return result;
        if (done >= prompt.*.len) return 0;
        if (done < target) {
            report(errors, "prefill made no progress", .{});
            return 1;
        }
    }
    return if (stopped(stop) or slotCancelled(slot)) c.DS4_SESSION_SYNC_INTERRUPTED else 0;
}
pub fn syncMultimodal(runtime: ?*c.server, member: ?*c.server_slot, prompt: [*c]const c.ds4_tokens, images: [*c]const c.ds4_vision_span, count: usize, errors: []u8, stop: [*c]const volatile c.sig_atomic_t) c_int {
    if (count == 0) return sync(runtime, member, prompt, null, errors, stop);
    const srv = runtime orelse return 1;
    const slot = member orelse return 1;
    if (prompt == null or images == null) return 1;
    if (!srv.batched_mode) {
        if (!c.ds4_server_coordinator_prefill_enter(srv, slot, stop)) return c.DS4_SESSION_SYNC_INTERRUPTED;
        defer c.ds4_server_coordinator_prefill_leave(srv);
        return c.ds4_session_sync_multimodal(slot.session, prompt, images, count, errors.ptr, errors.len);
    }
    // Both tokens and image identities must describe the retained frontier.
    _ = c.pthread_mutex_lock(&srv.inference_mu);
    var done = multimodalFrontier(c.ds4_session_pos(slot.session), c.ds4_session_common_prefix(slot.session, prompt), prompt.*.len, c.ds4_session_vision_prefix_matches(slot.session, images, count));
    _ = c.pthread_mutex_unlock(&srv.inference_mu);
    var called = false;
    while (!stopped(stop) and !slotCancelled(slot) and (!called or done < prompt.*.len)) {
        var target = done +% c.ds4_server_coordinator_quantum(srv);
        if (target > prompt.*.len or target < done) target = prompt.*.len;
        if (target <= 0) target = prompt.*.len;
        for (images[0..count]) |image| {
            const end = @as(u64, image.token_start) + image.embedding.token_count;
            if (@as(u64, @intCast(target)) > image.token_start and @as(u64, @intCast(target)) < end) target = if (end > @as(u64, @intCast(prompt.*.len))) prompt.*.len else @intCast(end);
        }
        var prefix_images: usize = 0;
        while (prefix_images < count) : (prefix_images += 1) {
            const image = images[prefix_images];
            if (@as(u64, image.token_start) + image.embedding.token_count > @as(u64, @intCast(target))) break;
        }
        var prefix = prompt.*;
        prefix.len = target;
        if (!c.ds4_server_coordinator_prefill_enter(srv, slot, stop)) return c.DS4_SESSION_SYNC_INTERRUPTED;
        const result = c.ds4_session_sync_multimodal(slot.session, &prefix, images, prefix_images, errors.ptr, errors.len);
        if (result == 0) done = c.ds4_session_pos(slot.session);
        c.ds4_server_coordinator_prefill_leave(srv);
        called = true;
        if (result != 0) return result;
        if (done >= prompt.*.len) return 0;
        if (done < target) {
            report(errors, "prefill made no progress", .{});
            return 1;
        }
    }
    return if (stopped(stop) or slotCancelled(slot)) c.DS4_SESSION_SYNC_INTERRUPTED else 0;
}

pub fn progress(packet: [*c]c.server_prefill_progress, event: [*c]const u8, current: c_int, total: c_int, stop: [*c]const volatile c.sig_atomic_t) void {
    if (packet == null or event == null) return;
    const p: *c.server_prefill_progress = @ptrCast(packet);
    if (cancelled(if (p.request_job == null) null else @ptrCast(p.request_job))) return;
    const chunk = same(event, "prefill_chunk");
    const display = same(event, "prefill_display");
    if (!chunk and !display) return;
    const stamp = now();
    // Send headers once, then only comments. A failed write cancels the job;
    // the backend callback observes that at its existing safe boundary.
    const job: ?*c.job = if (p.request_job == null) null else @ptrCast(p.request_job);
    const websocket = if (job) |owner| owner.req.response_sink != null else false;
    if (websocket) p.headers_sent = true;
    if (p.stream and !websocket and p.fd >= 0 and !p.stream_failed) {
        if (!p.headers_sent) {
            p.headers_sent = true;
            if (c.ds4_server_http_sse_headers(p.fd, p.enable_cors, stop)) p.last_keepalive = stamp else {
                p.stream_failed = true;
                markCancelled(p.request_job);
                return;
            }
        } else if (stamp - p.last_keepalive >= 5.0) {
            const keepalive = ": prefill\n\n";
            if (c.ds4_server_http_send_all(p.fd, keepalive, keepalive.len, stop)) p.last_keepalive = stamp else {
                p.stream_failed = true;
                markCancelled(p.request_job);
                return;
            }
        }
    }
    if (display) return;
    if (p.seen and current == p.last_current) {
        if (p.srv != null and p.slot != null and current > p.cached_tokens) c.ds4_server_persist_maybe_store(p.srv, p.slot);
        return;
    }
    const elapsed = stamp - p.t0;
    var start = p.cached_tokens;
    if (start < 0 or start > p.prompt_tokens) start = 0;
    var length = p.prompt_tokens - start;
    if (length <= 0) {
        start = 0;
        length = @max(p.prompt_tokens, total);
    }
    const shown: c_int = @max(@as(c_int, 0), @min(current - start, length));
    const pct = if (length > 0) 100.0 * @as(f64, @floatFromInt(shown)) / @as(f64, @floatFromInt(length)) else 100.0;
    const average = if (elapsed > 0) @as(f64, @floatFromInt(shown)) / elapsed else 0.0;
    const interval: c_int = if (p.seen) @max(@as(c_int, 0), current - p.last_current) else 0;
    const seconds = if (p.seen) stamp - p.last_t else 0.0;
    const rate = if (seconds > 0) @as(f64, @floatFromInt(interval)) / seconds else 0.0;
    p.last_current = current;
    p.last_t = stamp;
    p.seen = true;
    var label: [64]u8 = undefined;
    flags(&label, p.responses_protocol, p.has_tools, false, false, false);
    log(c.DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs", .{ @as([*:0]const u8, if (p.kind == c.REQ_CHAT) "chat" else "completion"), &p.ctx, @as([*:0]const u8, if (label[0] != 0) " " else ""), &label, if (p.phase != null) p.phase else @as([*c]const u8, "prefill"), shown, length, pct, rate, average, elapsed });
    if (p.srv != null and p.slot != null and current > p.cached_tokens) c.ds4_server_persist_maybe_store(p.srv, p.slot);
}

fn appendCalls(out: *c.buf, request: ?*const c.request, calls: [*c]const c.tool_calls) void {
    const syntax = if (request) |req| req.model_syntax else c.SERVER_MODEL_SYNTAX_DEEPSEEK;
    const orders = if (request) |req| &req.tool_orders else null;
    if (syntax == c.SERVER_MODEL_SYNTAX_FLINT) c.ds4_server_render_flint_calls(out, calls, orders) else if (syntax == c.SERVER_MODEL_SYNTAX_GLM) c.ds4_server_render_glm_calls(out, calls, orders) else if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) c.ds4_server_render_qwen_calls(out, calls, out.len != 0, orders) else c.ds4_server_render_dsml_calls(out, calls, syntax == c.SERVER_MODEL_SYNTAX_DEEPSEEK41);
}
fn qwenSuffix(request: ?*const c.request, content: [*c]const u8, reasoning: [*c]const u8, with_reasoning: bool, calls: [*c]const c.tool_calls, include_stop: bool) [*c]u8 {
    var message = zero(c.chat_msg);
    message.role = @constCast("assistant");
    message.content = @constCast(content);
    message.reasoning = @constCast(reasoning);
    if (calls != null) message.calls = calls.*;
    const messages: c.chat_msgs = .{ .v = &message, .len = 1, .cap = 1 };
    var view = Canonical.View.init(allocator, c, &messages, if (request) |req| &req.tool_orders else null) catch fatal();
    defer view.deinit();
    return (QwenPrompt.assistantSuffix(allocator, view.messages[0], view.tools, .{
        .reasoning = if (request) |req| Canonical.reasoning(c, req.think_mode) else .none,
        .include_reasoning = with_reasoning,
        .include_stop = include_stop,
    }) catch fatal()).ptr;
}
pub fn assistantSuffix(request: ?*const c.request, content: [*c]const u8, reasoning: [*c]const u8, calls: [*c]const c.tool_calls, visible: bool) [*c]u8 {
    const syntax = if (request) |req| req.model_syntax else c.SERVER_MODEL_SYNTAX_DEEPSEEK;
    const include_reasoning = !visible or (request != null and request.?.reasoning_summary_emit and calls != null and calls.*.len > 0);
    if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) return qwenSuffix(request, content, reasoning, include_reasoning, calls, true);
    var out = zero(c.buf);
    if (syntax == c.SERVER_MODEL_SYNTAX_FLINT) {
        if (include_reasoning and text(reasoning).len > 0) {
            append(&out, "<think>");
            puts(&out, reasoning);
            append(&out, "</think>");
        }
        puts(&out, content);
        appendCalls(&out, request, calls);
        return take(&out);
    }
    // A visible Responses key contains reasoning only when the client can
    // replay a tool-call summary; hidden final-answer reasoning remains in KV.
    if (request != null and c.ds4_think_mode_enabled(request.?.think_mode)) {
        if (include_reasoning) puts(&out, reasoning);
        append(&out, "</think>");
    }
    puts(&out, content);
    appendCalls(&out, request, calls);
    if (syntax != c.SERVER_MODEL_SYNTAX_GLM) append(&out, "<｜end▁of▁sentence｜>");
    return take(&out);
}
pub fn thinkingVisible(request: ?*const c.request, content: [*c]const u8) [*c]u8 {
    const req = request orelse return null;
    if (req.prompt_text == null or !c.ds4_think_mode_enabled(req.think_mode)) return null;
    const prompt = text(req.prompt_text);
    var out = zero(c.buf);
    if (req.model_syntax == c.SERVER_MODEL_SYNTAX_QWEN) {
        if (!std.mem.endsWith(u8, prompt, "<|im_start|>assistant\n<think>\n")) return null;
        append(&out, prompt);
        append(&out, "\n</think>\n\n");
        trim(&out, content);
        append(&out, "<|im_end|>\n");
    } else {
        if (!std.mem.endsWith(u8, prompt, "<think>")) return null;
        append(&out, prompt[0 .. prompt.len - "<think>".len]);
        if (req.model_syntax == c.SERVER_MODEL_SYNTAX_GLM) {
            append(&out, "<think></think>");
            trim(&out, content);
        } else {
            append(&out, "</think>");
            puts(&out, content);
            append(&out, "<｜end▁of▁sentence｜>");
        }
    }
    return take(&out);
}
pub fn qwenToolVisible(request: ?*const c.request, finish: [*c]const u8, inside: bool, content: [*c]const u8, calls: [*c]const c.tool_calls) [*c]u8 {
    const req = request orelse return null;
    if (calls == null or calls.*.len == 0 or req.kind != c.REQ_CHAT or req.image_count != 0 or req.api == c.API_RESPONSES or req.api == c.API_ANTHROPIC or req.model_syntax != c.SERVER_MODEL_SYNTAX_QWEN or text(req.prompt_text).len == 0 or text(calls.*.raw_tool_text).len == 0 or !same(finish, "tool_calls") or inside) return null;
    const suffix = qwenSuffix(req, content, null, false, calls, false);
    defer c.free(suffix);
    var out = zero(c.buf);
    puts(&out, req.prompt_text);
    puts(&out, suffix);
    return take(&out);
}
fn systemRegion(prompt: [*c]const u8) []const u8 {
    var bytes = text(prompt);
    const bos = "<｜begin▁of▁sentence｜>";
    if (std.mem.startsWith(u8, bytes, bos)) bytes = bytes[bos.len..];
    const max_prefix = text(c.ds4_think_max_prefix());
    if (max_prefix.len != 0 and std.mem.startsWith(u8, bytes, max_prefix)) bytes = bytes[max_prefix.len..];
    var start: usize = 0;
    while (start < bytes.len and c.isspace(bytes[start]) != 0) start += 1;
    bytes = bytes[start..];
    const user = std.mem.find(u8, bytes, "<｜User｜>");
    const assistant = std.mem.find(u8, bytes, "<｜Assistant｜>");
    var end = @min(user orelse bytes.len, assistant orelse bytes.len);
    while (end > 0 and c.isspace(bytes[end - 1]) != 0) end -= 1;
    return bytes[0..end];
}
pub fn invalidSuffix(request: ?*const c.request, thinking: [*c]const c.thinking_state, detail: [*c]const u8) [*c]u8 {
    const syntax = if (request) |req| req.model_syntax else c.SERVER_MODEL_SYNTAX_DEEPSEEK;
    const enabled = request != null and c.ds4_think_mode_enabled(request.?.think_mode);
    var error_text = zero(c.buf);
    defer freeBuffer(&error_text);
    append(&error_text, if (syntax == c.SERVER_MODEL_SYNTAX_FLINT) "Tool error: invalid Flint tool call" else if (syntax == c.SERVER_MODEL_SYNTAX_GLM) "Tool error: invalid GLM tool call" else if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) "Tool error: invalid Qwen tool call" else "Tool error: invalid DSML tool call");
    if (text(detail).len != 0) {
        append(&error_text, ": ");
        puts(&error_text, detail);
    }
    if (syntax == c.SERVER_MODEL_SYNTAX_GLM or syntax == c.SERVER_MODEL_SYNTAX_FLINT) append(&error_text, "\nThe previous assistant output was not executed because the <tool_call> syntax was malformed. Emit a new valid <tool_call>, or answer normally if no tool is needed.") else if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) append(&error_text, "\nThe previous assistant output was not executed because the <tool_call><function=...> syntax was malformed. Emit a new valid <tool_call>, or answer normally if no tool is needed.") else {
        append(&error_text, "\nThe previous assistant output was not executed because the DSML syntax was malformed. Emit a new valid DSML tool call, or answer normally if no tool is needed.");
        const system = systemRegion(if (request) |req| req.prompt_text else null);
        if (system.len != 0) {
            append(&error_text, "\n\nSystem prompt reminder:\n");
            append(&error_text, system);
        }
    }
    var out = zero(c.buf);
    if (enabled and thinking != null and thinking.*.inside) append(&out, if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) "\n</think>\n\n" else "</think>");
    if (syntax == c.SERVER_MODEL_SYNTAX_FLINT) {
        append(&out, "<|observation|><tool_response>");
        puts(&out, error_text.ptr);
        append(&out, "</tool_response><|assistant|>");
    } else if (syntax == c.SERVER_MODEL_SYNTAX_GLM) {
        append(&out, "<|observation|><tool_response>");
        c.ds4_server_render_glm_result(&out, ptr(error_text.ptr));
        append(&out, "</tool_response><|assistant|>");
        append(&out, if (enabled) "<think>" else "<think></think>");
    } else if (syntax == c.SERVER_MODEL_SYNTAX_QWEN) {
        append(&out, "<|im_end|>\n<|im_start|>user\n<tool_response>\n");
        c.ds4_server_render_glm_body(&out, ptr(error_text.ptr), "</tool_response>");
        append(&out, "\n</tool_response><|im_end|>\n");
        append(&out, if (enabled) "<|im_start|>assistant\n<think>\n" else "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    } else {
        append(&out, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
        c.ds4_server_render_tool_result(&out, ptr(error_text.ptr));
        append(&out, "</tool_result><｜Assistant｜>");
        append(&out, if (enabled) "<think>" else "</think>");
    }
    return take(&out);
}

const ProgressContext = struct { packet: c.server_prefill_progress, stop: [*c]const volatile c.sig_atomic_t };
fn progressCallback(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    const value: *ProgressContext = @ptrCast(@alignCast(context orelse return));
    progress(&value.packet, event, current, total, value.stop);
}
fn setProgress(session: ?*c.ds4_session, context: *ProgressContext) void {
    c.ds4_session_set_progress(session, progressCallback, context);
    c.ds4_session_set_display_progress(session, progressCallback, context);
}
fn clearProgress(session: ?*c.ds4_session) void {
    c.ds4_session_set_progress(session, null, null);
    c.ds4_session_set_display_progress(session, null, null);
}
fn captureDiagnostic(live: [*c]const c.ds4_tokens, prompt: *const c.ds4_tokens, old: c_int, common: c_int) c.trace_cache_diag {
    var out = zero(c.trace_cache_diag);
    out.valid = true;
    out.rewind_to = -1;
    out.old_pos = old;
    out.prompt_len = prompt.len;
    out.common = common;
    const live_len = if (live != null) live.*.len else 0;
    out.start = @max(@as(c_int, 0), common - 8);
    const end = @max(out.start, @min(common + 9, @max(live_len, prompt.len)));
    out.count = @min(end - out.start, c.TRACE_CACHE_WINDOW);
    for (0..@intCast(out.count)) |index| {
        const position = out.start + @as(c_int, @intCast(index));
        out.live_id[index] = if (live != null and position < live_len) live.*.v[@intCast(position)] else -1;
        out.prompt_id[index] = if (position < prompt.len) prompt.v[@intCast(position)] else -1;
    }
    return out;
}
fn missReason(diagnostic: *const c.trace_cache_diag) [*:0]const u8 {
    if (!diagnostic.valid) return "unknown";
    if (diagnostic.old_pos == 0) return "no-live-checkpoint";
    if (diagnostic.rewind_to >= 0) return "live-prefix-rewind";
    if (diagnostic.common != diagnostic.old_pos) return "token-mismatch";
    if (diagnostic.prompt_len < diagnostic.old_pos) return "incoming-prompt-shorter-than-live-checkpoint";
    return "live-prefix-match";
}
pub fn findStop(stops: *const c.stop_list, raw: [*c]const u8, from: usize, position_out: *usize, length_out: *usize) bool {
    if (stops.len == 0 or raw == null) return false;
    var best: ?usize = null;
    var best_length: usize = 0;
    for (items([*c]u8, stops.v, @intCast(stops.len))) |stop| {
        const found = c.strstr(raw + from, stop);
        if (found == null) continue;
        const position = @intFromPtr(found) - @intFromPtr(raw);
        if (best == null or position < best.?) {
            best = position;
            best_length = text(stop).len;
        }
    }
    position_out.* = best orelse return false;
    length_out.* = best_length;
    return true;
}
fn stopPosition(stops: *const c.stop_list, raw: [*c]const u8, from: usize) ?usize {
    var position: usize = undefined;
    var length: usize = undefined;
    return if (findStop(stops, raw, from, &position, &length)) position else null;
}
pub fn streamSafe(stops: *const c.stop_list, length: usize) usize {
    if (stops.len == 0 or stops.max_len <= 1) return length;
    return length - @min(length, stops.max_len - 1);
}
pub fn structuredStream(req: *const c.request) bool {
    return req.stream and (req.api == c.API_ANTHROPIC or req.api == c.API_RESPONSES or (req.api == c.API_OPENAI and req.kind == c.REQ_CHAT));
}
fn appendReplay(req: *c.request, content: [*c]const u8, reasoning: [*c]const u8, calls: *const c.tool_calls, finish: [*c]const u8) void {
    if ((!req.responses_store and req.conversation_id == null and req.response_sink == null) or req.responses_replay == null or same(finish, "error") or same(finish, "length")) return;
    const history = text(req.responses_replay);
    if (history.len == 0 or history[history.len - 1] != ']') return;
    var out = zero(c.buf);
    append(&out, history[0 .. history.len - 1]);
    if (history.len > 2) append(&out, ",");
    var assistant = zero(c.chat_msg);
    assistant.role = @constCast("assistant");
    assistant.content = @constCast(ptr(content));
    assistant.reasoning = @constCast(reasoning);
    assistant.calls = calls.*;
    c.ds4_server_render_replay_message(&out, &assistant);
    append(&out, "]");
    c.free(req.responses_replay);
    req.responses_replay = take(&out);
}

pub const Generation = struct {
    runtime: *c.server,
    slot: *c.server_slot,
    job: *c.job,
    stop: [*c]const volatile c.sig_atomic_t,

    fn trace(self: Generation, id: u64, comptime pattern: [*:0]const u8, arguments: anytype) void {
        if (self.runtime.trace == null or id == 0) return;
        const line = formatted(pattern, arguments);
        defer allocator.free(line);
        c.ds4_server_generation_trace_event(self.runtime, id, line.ptr);
    }
    fn eval(self: Generation, token: c_int, errors: []u8) c_int {
        var accepted: [2]c_int = undefined;
        var count: c_int = undefined;
        return c.ds4_server_coordinator_eval(self.runtime, self.slot, token, false, &accepted, &count, errors.ptr, errors.len, self.stop);
    }
    fn httpError(self: Generation, status: c_int, message: [*c]const u8) void {
        var out = zero(c.buf);
        defer freeBuffer(&out);
        if (self.job.req.response_sink != null) {
            const sink = &self.job.req.response_sink[0];
            append(&out, "{\"type\":\"error\",\"status\":");
            var digits: [16]u8 = undefined;
            append(&out, std.mem.print(&digits, "{d}", .{status}) catch unreachable);
            append(&out, ",\"error\":{\"type\":\"server_error\",\"code\":\"server_error\",\"param\":null,\"message\":");
            quote(&out, message);
            append(&out, "}}");
            if (sink.emit) |emit| _ = emit(sink.context, out.ptr, out.len);
            return;
        }
        append(&out, "{\"error\":{\"message\":");
        quote(&out, message);
        append(&out, ",\"type\":\"invalid_request_error\"}}\n");
        _ = c.ds4_server_http_response(self.job.fd, self.runtime.enable_cors, status, "application/json", out.ptr, self.stop);
    }
    fn prefillFailure(self: Generation, p: *const c.server_prefill_progress, context: [*c]const u8, label: [*c]const u8, errors: [*c]const u8) void {
        if (self.job.req.response_sink != null) {
            self.httpError(500, errors);
            return;
        }
        if (self.job.req.stream and p.headers_sent) {
            const kind: [*:0]const u8 = if (self.job.req.kind == c.REQ_CHAT) "chat" else "completion";
            if (p.stream_failed) {
                log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s prefill failed after stream closed: %s", .{ kind, context, @as([*:0]const u8, if (text(label).len != 0) " " else ""), ptr(label), errors });
                return;
            }
            var out = zero(c.buf);
            defer freeBuffer(&out);
            if (self.job.req.api == c.API_ANTHROPIC) append(&out, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":") else append(&out, "event: error\ndata: {\"error\":{\"message\":");
            quote(&out, if (text(errors).len != 0) errors else @as([*c]const u8, "internal server error"));
            append(&out, if (self.job.req.api == c.API_ANTHROPIC) "}}\n\n" else ",\"type\":\"server_error\"}}\n\n");
            if (!c.ds4_server_http_send_all(self.job.fd, out.ptr, out.len, self.stop)) log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s prefill SSE error failed: %s", .{ kind, context, @as([*:0]const u8, if (text(label).len != 0) " " else ""), ptr(label), errors });
        } else self.httpError(500, errors);
    }
    fn rewind(self: Generation, position: c_int, resample: bool, last: c_int, errors: []u8) c_int {
        const srv = self.runtime;
        const slot = self.slot;
        _ = c.pthread_mutex_lock(&srv.inference_mu);
        // Exact mode already has boundary logits; replay could precede its
        // verifier snapshot. Ordinary recurrent rewind may require a rebuild.
        if (resample and c.ds4_session_rewind_speculative(slot.session, position)) {
            _ = c.pthread_mutex_unlock(&srv.inference_mu);
            return 0;
        }
        c.ds4_session_rewind(slot.session, position - @as(c_int, @intFromBool(resample)));
        var prefix = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&prefix);
        c.ds4_tokens_copy(&prefix, c.ds4_session_tokens(slot.session));
        const rebuild = c.ds4_session_common_prefix(slot.session, &prefix) != prefix.len;
        _ = c.pthread_mutex_unlock(&srv.inference_mu);
        var result = if (rebuild) syncMultimodal(srv, slot, &prefix, self.job.req.images, self.job.req.image_count, errors, self.stop) else @as(c_int, 0);
        if (result == 0 and resample) result = self.eval(last, errors);
        return result;
    }
    fn recover(self: Generation, thinking: *const c.thinking_state, detail: [*c]const u8, appended: *c_int, errors: []u8) bool {
        appended.* = 0;
        const suffix = invalidSuffix(&self.job.req, thinking, detail);
        defer c.free(suffix);
        if (text(suffix).len == 0) return true;
        const live = c.ds4_session_tokens(self.slot.session);
        if (live == null) {
            report(errors, "live session is unavailable", .{});
            return false;
        }
        var target = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&target);
        c.ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(self.runtime.engine, live, suffix, &target);
        const before = c.ds4_session_pos(self.slot.session);
        const ok = sync(self.runtime, self.slot, &target, null, errors, self.stop) == 0;
        if (ok) appended.* = @max(@as(c_int, 0), c.ds4_session_pos(self.slot.session) - before);
        return ok;
    }
    fn rememberVisible(self: Generation, context: [*c]const u8, trace_id: u64, content: [*c]const u8, continued: bool) void {
        const visible = thinkingVisible(&self.job.req, content);
        if (visible == null) return;
        defer c.free(visible);
        c.ds4_server_reuse_remember(self.runtime, self.slot, c.API_OPENAI, visible, null, &self.job.req);
        _ = c.pthread_mutex_lock(&self.runtime.tool_mu);
        self.slot.thinking_live.token_text_disk_key = self.job.req.prompt_preserves_reasoning and !continued;
        _ = c.pthread_mutex_unlock(&self.runtime.tool_mu);
        log(c.DS4_LOG_KVCACHE, "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu", .{ context, c.ds4_session_pos(self.slot.session), text(visible).len });
        self.trace(trace_id, "thinking live checkpoint remembered: live=%d visible=%zu", .{ c.ds4_session_pos(self.slot.session), text(visible).len });
    }
    fn rememberQwen(self: Generation, context: [*c]const u8, finish: [*c]const u8, inside: bool, content: [*c]const u8, calls: *const c.tool_calls) bool {
        const visible = qwenToolVisible(&self.job.req, finish, inside, content, calls);
        if (visible == null) return false;
        defer c.free(visible);
        c.ds4_server_reuse_remember(self.runtime, self.slot, c.API_OPENAI, visible, null, &self.job.req);
        log(c.DS4_LOG_KVCACHE, "ds4-server: qwen tool-turn visible checkpoint remembered ctx=%s live=%d visible=%zu", .{ context, c.ds4_session_pos(self.slot.session), text(visible).len });
        return true;
    }
    /// Prefer a suffix rewrite; when the raw ring cannot hold it, reload an
    /// older exact checkpoint and replay under the ordinary prefill scheduler.
    /// Never replace correctly conditioned image state with a text disk key.
    pub fn canonicalize(self: Generation, context: [*c]const u8, trace_id: u64, content: [*c]const u8, reasoning: [*c]const u8, calls: [*c]const c.tool_calls) void {
        const srv = self.runtime;
        const slot = self.slot;
        const req = &self.job.req;
        if (calls == null or calls.*.len == 0 or req.prompt_text == null) return;
        const suffix = assistantSuffix(req, content, reasoning, calls, false);
        defer c.free(suffix);
        var rendered = zero(c.buf);
        defer freeBuffer(&rendered);
        puts(&rendered, req.prompt_text);
        puts(&rendered, suffix);
        var canonical = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&canonical);
        c.ds4_tokenize_rendered_chat(srv.engine, ptr(rendered.ptr), &canonical);
        const live_length = c.ds4_session_pos(slot.session);
        const common = c.ds4_session_common_prefix(slot.session, &canonical);
        if (common == live_length and canonical.len == live_length) return;
        var length: usize = 0;
        const visible = c.ds4_kvstore_render_tokens_text(srv.engine, c.ds4_session_tokens(slot.session), &length);
        const equal = length == rendered.len and (length == 0 or std.mem.eql(u8, visible[0..length], rendered.ptr[0..length]));
        c.free(visible);
        if (equal) return; // Same rendered bytes may have a different BPE spelling.
        if (req.image_count != 0) {
            log(c.DS4_LOG_WARNING, "ds4-server: multimodal tool checkpoint canonicalization skipped ctx=%s common=%d live=%d canonical=%d reason=preserve-image-state", .{ context, common, live_length, canonical.len });
            self.trace(trace_id, "multimodal tool checkpoint canonicalization skipped: common=%d live=%d canonical=%d", .{ common, live_length, canonical.len });
            return;
        }
        if (common < req.prompt.len) {
            self.trace(trace_id, "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d", .{ common, req.prompt.len, live_length, canonical.len });
            return;
        }
        var errors: [160]u8 = @splat(0);
        _ = c.pthread_mutex_lock(&srv.inference_mu);
        const result = c.ds4_session_rewrite_from_common(slot.session, &canonical, common, &errors, errors.len);
        _ = c.pthread_mutex_unlock(&srv.inference_mu);
        if (result == c.DS4_SESSION_REWRITE_OK) {
            log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d", .{ context, common, live_length, canonical.len });
            self.trace(trace_id, "tool checkpoint canonicalized: common=%d live=%d canonical=%d", .{ common, live_length, canonical.len });
            return;
        }
        if (result != c.DS4_SESSION_REWRITE_REBUILD_NEEDED) {
            log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"", .{ context, common, live_length, canonical.len, &errors });
            self.trace(trace_id, "tool checkpoint canonicalization failed: %s", .{&errors});
            return;
        }
        var path: [*c]u8 = null;
        defer c.free(path);
        var effective = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&effective);
        const loaded = c.ds4_server_persist_load(srv, slot, ptr(rendered.ptr), &effective, &path, null, false);
        if (loaded == 0) {
            _ = c.pthread_mutex_lock(&srv.inference_mu);
            c.ds4_session_invalidate(slot.session);
            _ = c.pthread_mutex_unlock(&srv.inference_mu);
        }
        const target = if (loaded > 0) &effective else &canonical;
        var rebuild_context: [48]u8 = undefined;
        contextSpan(&rebuild_context, loaded, target.len);
        const replay: c_int = if (target.len - loaded < 0) target.len else target.len - loaded;
        const tail: c_int = if (canonical.len - common < 0) canonical.len else canonical.len - common;
        const discarded: c_int = @max(@as(c_int, 0), live_length - common);
        const source: [*:0]const u8 = if (loaded > 0) "disk" else "full";
        const started = now();
        log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s", .{ replay, &rebuild_context, context, tail, discarded, common, live_length, canonical.len, loaded, source, @as([*:0]const u8, if (path != null) " file=" else ""), ptr(path) });
        var rebuilding = ProgressContext{ .packet = zero(c.server_prefill_progress), .stop = self.stop };
        rebuilding.packet.srv = srv;
        rebuilding.packet.slot = slot;
        rebuilding.packet.kind = req.kind;
        rebuilding.packet.prompt_tokens = target.len;
        rebuilding.packet.cached_tokens = loaded;
        rebuilding.packet.phase = "tool checkpoint rebuild";
        rebuilding.packet.has_tools = req.has_tools;
        rebuilding.packet.t0 = started;
        rebuilding.packet.fd = self.job.fd;
        rebuilding.packet.stream = req.stream;
        rebuilding.packet.enable_cors = srv.enable_cors;
        rebuilding.packet.request_job = self.job;
        rebuilding.packet.headers_sent = true; // This response stream is already in flight.
        report(&rebuilding.packet.ctx, "%s", .{&rebuild_context});
        setProgress(slot.session, &rebuilding);
        var sync_errors: [160]u8 = @splat(0);
        const synchronized = sync(srv, slot, target, null, &sync_errors, self.stop) == 0;
        clearProgress(slot.session);
        if (synchronized) {
            if (loaded > 0) {
                log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs", .{ &rebuild_context, context, loaded, replay, canonical.len, now() - started });
                self.trace(trace_id, "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s", .{ common, live_length, canonical.len, loaded, ptr(path) });
            } else {
                log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs", .{ &rebuild_context, context, replay, canonical.len, now() - started });
                self.trace(trace_id, "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s", .{ common, live_length, canonical.len, &errors });
            }
        } else {
            log(c.DS4_LOG_KVCACHE, "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"", .{ &rebuild_context, context, source, loaded, replay, canonical.len, &sync_errors });
            self.trace(trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", .{&sync_errors});
        }
    }

    /// Keep cancellation installed until every early return and backend use is
    /// complete. Only then can the client retire its stack-owned Job/socket.
    pub fn run(self: Generation) void {
        const srv = self.runtime;
        const slot = self.slot;
        _ = c.pthread_mutex_lock(&srv.model_mu);
        slot.running = self.job;
        _ = c.pthread_mutex_unlock(&srv.model_mu);
        c.ds4_session_set_cancel(slot.session, cancelled, self.job);
        defer {
            c.ds4_session_set_cancel(slot.session, null, null);
            _ = c.pthread_mutex_lock(&srv.model_mu);
            if (slot.running == self.job) slot.running = null;
            _ = c.pthread_cond_broadcast(&srv.model_cv);
            _ = c.pthread_mutex_unlock(&srv.model_mu);
        }
        if (!cancelled(self.job)) self.inner();
    }

    fn decodeLog(self: Generation, prompt: c_int, completion: c_int, responses: bool, thinking: bool, saw_start: bool, saw_end: bool, started: f64, last_time: *f64, last_count: *c_int) void {
        const stamp = now();
        const elapsed = stamp - started;
        const interval = stamp - last_time.*;
        const rate = if (interval > 0) @as(f64, @floatFromInt(completion - last_count.*)) / interval else 0.0;
        const average = if (elapsed > 0) @as(f64, @floatFromInt(completion)) / elapsed else 0.0;
        var context: [48]u8 = undefined;
        contextSpan(&context, prompt + last_count.*, prompt + completion);
        var label: [80]u8 = undefined;
        flags(&label, responses, self.job.req.has_tools, thinking, saw_start, saw_end);
        log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs", .{ @as([*:0]const u8, if (self.job.req.kind == c.REQ_CHAT) "chat" else "completion"), &context, completion, @as([*:0]const u8, if (label[0] != 0) " " else ""), &label, rate, average, elapsed });
        last_time.* = stamp;
        last_count.* = completion;
    }
    fn decode(self: Generation, prompt_tokens: c_int, context: [*c]const u8, request_flags: [*c]const u8, progress_context: *ProgressContext, responses: bool, thinking_continued: bool, disk_flags: u8, started: f64, trace_id: u64, errors: []u8) void {
        const srv = self.runtime;
        const slot = self.slot;
        const job = self.job;
        const req = &job.req;
        const multimodal = req.image_count != 0;
        const space: [*:0]const u8 = if (text(request_flags).len != 0) " " else "";
        const kind: [*:0]const u8 = if (req.kind == c.REQ_CHAT) "chat" else "completion";
        var id: [96]u8 = undefined;
        c.ds4_server_output_random_id(&id, id.len, if (req.kind == c.REQ_CHAT) "chatcmpl-" else "cmpl-");
        const openai_chat = req.stream and req.api == c.API_OPENAI and req.kind == c.REQ_CHAT;
        const responses_chat = req.stream and req.api == c.API_RESPONSES and req.kind == c.REQ_CHAT;
        const structured = req.stream and (req.api == c.API_ANTHROPIC or req.api == c.API_RESPONSES or openai_chat);
        var anthropic = zero(c.anthropic_stream);
        var openai = zero(c.openai_stream);
        var output = zero(c.responses_stream);
        defer {
            c.ds4_server_anthropic_free(&anthropic);
            c.ds4_server_openai_free(&openai);
            c.ds4_server_output_free(&output);
        }
        const memory = if (srv.disable_exact_dsml_tool_replay) null else &srv.tool_mem;
        const mutex = if (srv.disable_exact_dsml_tool_replay) null else &srv.tool_mu;
        const created: c_long = @intCast(c.time(null));
        const p = &progress_context.packet;
        if (req.stream) {
            if (p.stream_failed) {
                log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s stream closed during prefill", .{ kind, context, space, request_flags });
                clearLive(srv, slot);
                return;
            }
            // Prefill may already have sent headers; never write them twice.
            if (!p.headers_sent and req.response_sink == null and !c.ds4_server_http_sse_headers(job.fd, srv.enable_cors, self.stop)) {
                markCancelled(job);
                log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s%s%s sse headers failed", .{ kind, context, space, request_flags });
                clearLive(srv, slot);
                return;
            }
            p.headers_sent = true;
            if (req.api == c.API_ANTHROPIC and !c.ds4_server_anthropic_start(job.fd, req, &id, prompt_tokens, &anthropic, self.stop)) {
                markCancelled(job);
                log(c.DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", .{context});
                clearLive(srv, slot);
                return;
            }
            if (req.api == c.API_OPENAI and req.kind == c.REQ_CHAT and !c.ds4_server_openai_chunk(job.fd, req, &id, null, null, self.stop)) {
                markCancelled(job);
                log(c.DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", .{context});
                clearLive(srv, slot);
                return;
            }
            if (openai_chat) c.ds4_server_openai_init(req, &openai);
            if (responses_chat) {
                c.ds4_server_output_init(req, &output);
                output.active = true;
                if (!c.ds4_server_output_created(job.fd, req, &output, created, self.stop)) {
                    markCancelled(job);
                    log(c.DS4_LOG_GENERATION, "ds4-server: chat ctx=%s%s%s responses created event failed", .{ context, space, request_flags });
                    clearLive(srv, slot);
                    return;
                }
            }
        }
        var recovered_once = false;
        var recovery_completion: c_int = 0;
        var rng = req.seed;
        if (rng == 0 and !randomBytes(std.mem.asBytes(&rng))) rng = (@as(u64, @bitCast(@as(i64, @intCast(c.time(null))))) << 32) ^ @intFromPtr(job);
        if (rng == 0) rng = 0x9e3779b97f4a7c15;
        var generated_thinking: c_int = 0;
        var deferred: c_int = 0;
        var forced = false;
        const close_token: c_int = if (req.model_syntax == c.SERVER_MODEL_SYNTAX_FLINT) 150007 else 128822;
        // A non-streaming invalid tool can inject one model-visible error and
        // retry inside the same total output budget. Each attempt owns its text
        // and parsed result, while RNG and reasoning counters stay continuous.
        while (true) {
            var sampled = zero(c.buf);
            defer freeBuffer(&sampled);
            var plain_position: usize = 0;
            var stop_cursor: usize = 0;
            var finish: [*c]const u8 = "length";
            var completion: c_int = 0;
            const room = c.ds4_session_ctx(slot.session) - c.ds4_session_pos(slot.session);
            const maximum = decodeBudget(req.max_tokens, recovery_completion, room);
            var saw_start = false;
            var saw_end = false;
            var saw_orphan = false;
            var client_stop = false;
            var tool_cursor: usize = 0;
            var next_trace: c_int = 128;
            var next_log: c_int = 50;
            var stop_detail: [*c]const u8 = if (maximum == room) "context limit" else "output limit";
            var stop_token: c_int = -1;
            self.trace(trace_id, "prefill done; decode_max=%d ctx_room=%d", .{ maximum, room });
            const decode_started = now();
            var last_log_time = decode_started;
            var last_log_completion: c_int = 0;
            var thinking = thinkingFromPrompt(req);
            const gate_markers = c.ds4_think_mode_enabled(req.think_mode);
            var wait_for_close = gate_markers and thinking.inside;
            var recovery_cursor: usize = 0;
            var tracker: c.dsml_decode_tracker = undefined;
            c.ds4_server_decode_init(&tracker);
            tracker.model_syntax = req.model_syntax;
            c.ds4_server_coordinator_generation_enter(srv);
            while (!stopped(self.stop) and !cancelled(job) and completion < maximum and c.ds4_session_pos(slot.session) < c.ds4_session_ctx(slot.session)) {
                const state = if (req.kind == c.REQ_CHAT and req.has_tools) tracker.decode else c.DSML_DECODE_OUTSIDE;
                const in_call = c.ds4_server_decode_is_tool(state);
                if (!(req.kind == c.REQ_CHAT and req.has_tools and (saw_start or in_call))) {
                    if (!multimodal) c.ds4_server_persist_maybe_store(srv, slot);
                }
                var temperature = req.temperature;
                var top_k = req.top_k;
                var top_p = req.top_p;
                var min_p = req.min_p;
                if (req.model_syntax != c.SERVER_MODEL_SYNTAX_FLINT and c.ds4_think_mode_enabled(req.think_mode)) {
                    // Explicit client sampling always wins, including greedy.
                    if (!req.temperature_set) temperature = c.DS4_DEFAULT_TEMPERATURE;
                    if (!req.top_k_set) top_k = 0;
                    if (!req.top_p_set) top_p = c.DS4_DEFAULT_TOP_P;
                    if (!req.min_p_set) min_p = c.DS4_DEFAULT_MIN_P;
                }
                const greedy_syntax = !thinking.inside and in_call and !c.ds4_server_decode_is_payload(state);
                const payload_temperature = temperature;
                if (greedy_syntax) temperature = 0;
                const eos = c.ds4_token_eos(srv.engine);
                const eligible = req.reasoning_budget > 0 and generated_thinking >= req.reasoning_budget and !forced and thinking.inside;
                const force_close = reasoningDue(req.reasoning_budget, generated_thinking, forced, thinking.inside, maximum - completion, c.ds4_session_ctx(slot.session) - c.ds4_session_pos(slot.session), !eligible or c.ds4_server_decode_boundary_safe(&sampled, &tracker));
                if (eligible and !force_close) deferred += 1;
                if (force_close) {
                    var length: usize = 0;
                    const piece = c.ds4_token_text(srv.engine, close_token, &length);
                    const valid = piece != null and length == 8 and std.mem.eql(u8, piece[0..8], "</think>") and c.ds4_token_is_thinking_control(srv.engine, close_token) and !c.ds4_token_is_stop_for_think_mode(srv.engine, close_token, req.think_mode) and checkpointInside(srv.engine, slot.session, close_token);
                    c.free(piece);
                    if (!valid) {
                        finish = "error";
                        report(errors, "invalid native reasoning-close token or checkpoint", .{});
                        break;
                    }
                }
                var token = if (force_close) close_token else if (req.ignore_eos) c.ds4_session_argmax_ignoring_eos(slot.session, req.think_mode) else c.ds4_session_sample(slot.session, temperature, top_k, top_p, min_p, &rng);
                if (token < 0) {
                    finish = "error";
                    report(errors, "failed to select a non-EOS token", .{});
                    break;
                }
                if (c.ds4_token_is_stop_for_think_mode(srv.engine, token, req.think_mode)) {
                    finish = "stop";
                    stop_detail = "stop token";
                    stop_token = token;
                    break;
                }
                var tokens: [17]c_int = undefined;
                var count: c_int = 0;
                const block_start = c.ds4_session_pos(slot.session);
                if (req.reasoning_budget == 0 and (!srv.batched_mode or c.ds4_engine_is_flint(srv.engine)) and c.ds4_engine_decode_packet_size(srv.engine) > 1 and c.getenv("DS4_MTP_SPEC_DISABLE") == null) {
                    count = if (req.ignore_eos) c.ds4_session_eval_speculative_argmax_ignoring_eos(slot.session, token, maximum - completion, eos, req.think_mode, &tokens, tokens.len, errors.ptr, errors.len) else c.ds4_session_eval_speculative(slot.session, token, maximum - completion, eos, temperature, top_k, top_p, min_p, &rng, &tokens, tokens.len, errors.ptr, errors.len);
                    if (count < 0) {
                        finish = "error";
                        break;
                    }
                } else if (req.reasoning_budget == 0 and srv.batched_mode and srv.qwen4_batch_mtp and maximum - completion >= 2 and !req.ignore_eos and (!c.ds4_engine_mtp_exact_sampling(srv.engine) or temperature == 0) and c.getenv("DS4_MTP_SPEC_DISABLE") == null) {
                    if (c.ds4_server_coordinator_eval(srv, slot, token, true, &tokens, &count, errors.ptr, errors.len, self.stop) != 0) {
                        finish = "error";
                        break;
                    }
                } else {
                    if (self.eval(token, errors) != 0) {
                        finish = "error";
                        break;
                    }
                    tokens[0] = token;
                    count = 1;
                }
                if (count == 0) {
                    finish = "error";
                    report(errors, "decode returned no tokens", .{});
                    break;
                }
                if (force_close) {
                    const checkpoint = c.ds4_session_tokens(slot.session);
                    if (count != 1 or tokens[0] != close_token or c.ds4_session_pos(slot.session) != block_start + 1 or !c.ds4_session_checkpoint_valid(slot.session) or checkpoint == null or checkpoint.*.len != block_start + 1 or checkpoint.*.v[@intCast(checkpoint.*.len - 1)] != close_token or checkpointInside(srv.engine, slot.session, close_token)) {
                        finish = "error";
                        report(errors, "native reasoning close did not commit outside thinking", .{});
                        break;
                    }
                    forced = true;
                    log(c.DS4_LOG_GENERATION, "ds4-server: reasoning budget=%d forced_close generated_thinking=%d pos=%d", .{ req.reasoning_budget, generated_thinking, block_start });
                    self.trace(trace_id, "reasoning forced_close budget=%d generated_thinking=%d", .{ req.reasoning_budget, generated_thinking });
                }
                var stop_decode = false;
                var resample = false;
                var text_stop = false;
                var kept: c_int = 0;
                var offset: c_int = 0;
                while (offset < count and completion < maximum) : (offset += 1) {
                    if (cancelled(job)) {
                        stop_decode = true;
                        break;
                    }
                    token = tokens[@intCast(offset)];
                    if (c.ds4_token_is_stop_for_think_mode(srv.engine, token, req.think_mode)) {
                        finish = "stop";
                        stop_detail = "stop token";
                        stop_token = token;
                        stop_decode = true;
                        break;
                    }
                    var length: usize = 0;
                    const piece = c.ds4_token_text(srv.engine, token, &length);
                    defer c.free(piece);
                    completion += 1;
                    kept += 1;
                    c.ds4_server_generation_trace_piece(srv, trace_id, piece, length);
                    append(&sampled, items(u8, piece, length));
                    const was_thinking = thinking.inside;
                    if (was_thinking and !c.ds4_token_is_thinking_control(srv.engine, token)) generated_thinking += 1;
                    req.generated_thinking_tokens = generated_thinking;
                    if (!c.ds4_server_decode_is_tool(tracker.decode)) feedThinking(&thinking, piece, length);
                    if (force_close and thinking.inside) {
                        finish = "error";
                        report(errors, "reasoning stream did not leave thinking", .{});
                        stop_decode = true;
                        break;
                    }
                    if (req.kind == c.REQ_CHAT and req.has_tools) {
                        if (thinking.inside) {
                            c.ds4_server_decode_init(&tracker);
                            tracker.model_syntax = req.model_syntax;
                            tracker.pos = sampled.len;
                        } else {
                            if (was_thinking) {
                                const end = c.ds4_server_reply_last_substring(sampled.ptr, "</think>");
                                tracker.pos = if (end != null) @intFromPtr(end) + 8 - @intFromPtr(sampled.ptr) else sampled.len;
                            }
                            c.ds4_server_decode_update(&tracker, sampled.ptr, sampled.len);
                        }
                    }
                    const stop_position = stopPosition(&req.stops, sampled.ptr, stop_cursor);
                    var stream_length = @min(stop_position orelse streamSafe(&req.stops, sampled.len), sampled.len);
                    stream_length = c.ds4_server_protocol_utf8_limit(sampled.ptr, plain_position, stream_length, stop_position != null);
                    if (stop_position == null and req.stops.max_len > 1) stop_cursor = sampled.len - @min(sampled.len, req.stops.max_len - 1);
                    if (req.stream and !structured and stream_length > plain_position) {
                        const delta = duplicate(sampled.ptr[plain_position..stream_length]);
                        defer c.free(delta);
                        if (!c.ds4_server_openai_chunk(job.fd, req, &id, delta, null, self.stop)) {
                            markCancelled(job);
                            finish = "error";
                            report(errors, "client stream write failed", .{});
                            stop_decode = true;
                            break;
                        }
                        plain_position = stream_length;
                    }
                    var sent = true;
                    if (req.stream and req.api == c.API_ANTHROPIC) sent = c.ds4_server_anthropic_update(job.fd, memory, mutex, req, &id, &anthropic, sampled.ptr, stream_length, false, self.stop);
                    if (sent and openai_chat) sent = c.ds4_server_openai_update(job.fd, memory, mutex, req, &id, &openai, sampled.ptr, stream_length, false, self.stop);
                    if (sent and responses_chat) sent = c.ds4_server_output_update(job.fd, req, &output, sampled.ptr, stream_length, false, self.stop);
                    if (!sent) {
                        markCancelled(job);
                        finish = "error";
                        report(errors, "client stream write failed", .{});
                        stop_decode = true;
                        break;
                    }
                    if (req.kind == c.REQ_CHAT and req.has_tools) {
                        if (gate_markers and thinking.inside) {
                            // A complete tool block inside unclosed reasoning is
                            // recoverable. A quoted opening marker alone is not.
                            if (completeThinking(req.model_syntax, sampled.ptr, sampled.len, &recovery_cursor)) {
                                saw_start = true;
                                saw_end = true;
                                finish = "tool_calls";
                                stop_decode = true;
                                log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s recovered a complete tool call from unclosed reasoning after %d generated tokens", .{ context, space, request_flags, completion });
                                self.trace(trace_id, "recovered complete tool call from unclosed reasoning after %d generated tokens", .{completion});
                                break;
                            }
                            wait_for_close = true;
                            tool_cursor = sampled.len;
                        } else {
                            if (wait_for_close) {
                                const end = c.ds4_server_reply_last_substring(sampled.ptr, "</think>");
                                tool_cursor = if (end != null) @intFromPtr(end) + 8 - @intFromPtr(sampled.ptr) else sampled.len;
                                tool_cursor = @min(tool_cursor, sampled.len);
                                wait_for_close = false;
                            }
                            tool_cursor = @min(tool_cursor, sampled.len);
                            const scan = if (sampled.ptr != null) sampled.ptr + tool_cursor else @as([*c]const u8, "");
                            var orphan = false;
                            const old_start = saw_start;
                            const old_end = saw_end;
                            c.ds4_server_decode_observe(&tracker, scan, &saw_start, &saw_end, &orphan);
                            if (orphan and !saw_orphan) {
                                saw_orphan = true;
                                log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens", .{ context, space, request_flags, completion });
                                self.trace(trace_id, "ignored orphan tool-call end marker after %d generated tokens", .{completion});
                            }
                            if (saw_start and !old_start) self.trace(trace_id, "entered tool-call block after %d generated tokens", .{completion});
                            if (saw_end and !old_end) self.trace(trace_id, "closed tool-call block after %d generated tokens", .{completion});
                            tool_cursor = @max(tool_cursor, sampled.len - @min(sampled.len, 80));
                            if (srv.trace != null and completion >= next_trace) {
                                self.trace(trace_id, "progress gen=%d dsml_start=%d dsml_end=%d", .{ completion, flag(saw_start), flag(saw_end) });
                                next_trace += 128;
                            }
                        }
                    }
                    if (completion >= next_log) {
                        self.decodeLog(prompt_tokens, completion, responses, thinking.inside, saw_start, saw_end, decode_started, &last_log_time, &last_log_completion);
                        next_log += 50;
                    }
                    if (stop_position) |position| {
                        finish = "stop";
                        stop_detail = "client stop sequence";
                        client_stop = true;
                        sampled.len = position;
                        sampled.ptr[position] = 0;
                        _ = c.pthread_mutex_lock(&srv.inference_mu);
                        c.ds4_session_invalidate(slot.session);
                        _ = c.pthread_mutex_unlock(&srv.inference_mu);
                        text_stop = true;
                        stop_decode = true;
                        break;
                    }
                    if (req.kind == c.REQ_CHAT and req.has_tools and saw_end) {
                        finish = "tool_calls";
                        stop_decode = true;
                        break;
                    }
                    const next_greedy = !thinking.inside and c.ds4_server_decode_is_tool(tracker.decode) and !c.ds4_server_decode_is_payload(tracker.decode);
                    // Exact sampled packets must restart when entering/leaving
                    // greedy tool syntax. Flint packets contain sampled tokens
                    // directly rather than opportunistic greedy drafts.
                    if (offset + 1 < count and (c.ds4_engine_mtp_exact_sampling(srv.engine) or c.ds4_engine_is_flint(srv.engine)) and payload_temperature > 0 and next_greedy != greedy_syntax) {
                        resample = true;
                        break;
                    }
                }
                if (kept < count and !text_stop and !cancelled(job) and !same(finish, "error")) {
                    const position = block_start + kept;
                    const rewind_started = now();
                    self.trace(trace_id, "speculative boundary rewind: from=%d to=%d", .{ c.ds4_session_pos(slot.session), position });
                    if (self.rewind(position, resample, if (kept != 0) tokens[@intCast(kept - 1)] else 0, errors) != 0) {
                        finish = "error";
                        stop_decode = true;
                    } else self.trace(trace_id, "speculative boundary: kept=%d discarded=%d resample=%d rewind_ms=%.3f", .{ kept, count - kept, flag(resample), (now() - rewind_started) * 1000 });
                }
                if (stop_decode) break;
            }
            c.ds4_server_coordinator_generation_leave(srv);
            if (cancelled(job)) {
                clearLive(srv, slot);
                self.trace(trace_id, "cancelled during generation after %d tokens", .{completion});
                return;
            }
            if (stopped(self.stop) and !same(finish, "error")) {
                finish = "error";
                report(errors, "shutdown requested", .{});
            }
            if (req.kind == c.REQ_CHAT and req.has_tools and saw_start and !saw_end and !same(finish, "error")) {
                log(c.DS4_LOG_WARNING, "ds4-server: incomplete %s tool call: stop=%s token=%d generated=%d limit=%d room=%d", .{ @as([*:0]const u8, if (req.model_syntax == c.SERVER_MODEL_SYNTAX_GLM) "GLM" else "DeepSeek"), stop_detail, stop_token, completion, maximum, room });
                self.trace(trace_id, "incomplete tool call: stop=%s token=%d generated=%d limit=%d room=%d", .{ stop_detail, stop_token, completion, maximum, room });
            }
            if (req.kind == c.REQ_CHAT and req.has_tools and saw_start and !saw_end and !same(finish, "error") and !same(finish, "length") and !client_stop) {
                var repaired = zero(c.buf);
                defer freeBuffer(&repaired);
                var complete = false;
                if (c.ds4_server_reply_repair(sampled.ptr, sampled.len, &repaired)) {
                    var content: [*c]u8 = null;
                    var reasoning: [*c]u8 = null;
                    var calls = zero(c.tool_calls);
                    const parsed = c.ds4_server_reply_parse(req.model_syntax, repaired.ptr, false, &content, &reasoning, &calls);
                    c.free(content);
                    c.free(reasoning);
                    if (parsed and calls.len > 0) {
                        freeBuffer(&sampled);
                        sampled = repaired;
                        repaired = zero(c.buf);
                        saw_end = true;
                        complete = true;
                        log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)", .{ context, space, request_flags, calls.len });
                        self.trace(trace_id, "repaired unterminated tool call (%d calls recovered)", .{calls.len});
                    }
                    freeCalls(&calls);
                }
                if (!complete) {
                    if (!req.stream and !recovered_once and completion < maximum) {
                        var appended: c_int = 0;
                        var recovery_error: [160]u8 = @splat(0);
                        log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error", .{ context, space, request_flags });
                        self.trace(trace_id, "unterminated tool call; continuing with model-visible tool error", .{});
                        if (self.recover(&thinking, "unterminated tool call", &appended, &recovery_error)) {
                            recovered_once = true;
                            recovery_completion += completion;
                            log(c.DS4_LOG_GENERATION, "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens", .{ context, space, request_flags, appended });
                            self.trace(trace_id, "tool-error continuation appended %d tokens", .{appended});
                            continue;
                        }
                        finish = "error";
                        report(errors, "invalid tool call recovery failed: %s", .{if (recovery_error[0] != 0) @as([*c]const u8, &recovery_error) else @as([*c]const u8, "unknown error")});
                    } else {
                        finish = "error";
                        report(errors, "unterminated tool call (%s; token=%d, generated=%d, limit=%d)", .{ stop_detail, stop_token, completion, maximum });
                    }
                }
            }
            if (completion > last_log_completion) self.decodeLog(prompt_tokens, completion, responses, thinking.inside, saw_start, saw_end, decode_started, &last_log_time, &last_log_completion);
            if (req.stream and !structured and sampled.len > plain_position) {
                const tail = duplicate(sampled.ptr[plain_position..sampled.len]);
                defer c.free(tail);
                if (!c.ds4_server_openai_chunk(job.fd, req, &id, tail, null, self.stop)) {
                    markCancelled(job);
                    finish = "error";
                }
            }
            if (cancelled(job)) {
                clearLive(srv, slot);
                self.trace(trace_id, "cancelled while flushing generation", .{});
                return;
            }
            var calls = zero(c.tool_calls);
            var content: [*c]u8 = null;
            var reasoning: [*c]u8 = null;
            defer {
                c.free(content);
                c.free(reasoning);
                freeCalls(&calls);
            }
            var final_finish = finish;
            var parse_recovered = false;
            if (req.kind == c.REQ_CHAT) {
                const parsed = c.ds4_server_reply_for_response(req.model_syntax, ptr(sampled.ptr), req.has_tools, saw_start, c.ds4_think_mode_enabled(req.think_mode), &final_finish, errors.ptr, errors.len, &content, &reasoning, &calls, &parse_recovered, &req.tool_orders);
                if (!parsed and parse_recovered and req.has_tools and saw_start) {
                    // Do not semantically repair rejected arguments. The model
                    // owns the next action after one explicit tool error.
                    if (!req.stream and !recovered_once and !same(final_finish, "length") and !client_stop and completion < maximum) {
                        var appended: c_int = 0;
                        var recovery_error: [160]u8 = @splat(0);
                        const detail = if (errors.len != 0 and errors[0] != 0) errors.ptr else @as([*c]const u8, "invalid tool call");
                        log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error", .{ context, space, request_flags });
                        self.trace(trace_id, "invalid tool call; continuing with model-visible tool error", .{});
                        if (self.recover(&thinking, detail, &appended, &recovery_error)) {
                            recovered_once = true;
                            recovery_completion += completion;
                            log(c.DS4_LOG_GENERATION, "ds4-server: chat ctx=%s%s%s tool-error continuation appended %d tokens", .{ context, space, request_flags, appended });
                            self.trace(trace_id, "tool-error continuation appended %d tokens", .{appended});
                            continue;
                        }
                        final_finish = "error";
                        report(errors, "invalid tool call recovery failed: %s", .{if (recovery_error[0] != 0) @as([*c]const u8, &recovery_error) else @as([*c]const u8, "unknown error")});
                    }
                    const opening = c.ds4_server_reply_tool_start(ptr(sampled.ptr));
                    const snippet: c_int = @intCast(if (opening != null) @min(sampled.len - (@intFromPtr(opening) - @intFromPtr(sampled.ptr)), 500) else 0);
                    const shown: c_int = @intCast(@min(sampled.len, 300));
                    log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]", .{ context, space, request_flags, final_finish, sampled.len, flag(saw_start), flag(saw_end), shown, if (sampled.ptr != null) @as([*c]const u8, sampled.ptr) else @as([*c]const u8, "(null)") });
                    log(c.DS4_LOG_WARNING, "ds4-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s", .{ context, space, request_flags, snippet, if (opening != null) opening else @as([*c]const u8, "(none)") });
                    self.trace(trace_id, "invalid tool call returned as assistant text finish=%s", .{final_finish});
                }
                if (cancelled(job)) {
                    clearLive(srv, slot);
                    self.trace(trace_id, "cancelled during response parsing", .{});
                    return;
                }
                if (calls.len != 0) {
                    if (openai_chat) c.ds4_server_openai_apply_ids(&calls, &openai);
                    if (req.api == c.API_ANTHROPIC and req.stream) c.ds4_server_anthropic_apply_ids(&calls, &anthropic);
                    assignIds(srv, &calls, req.api);
                    if (!srv.disable_exact_dsml_tool_replay) c.ds4_server_tool_memory_remember(&srv.tool_mem, &srv.tool_mu, &calls);
                    final_finish = "tool_calls";
                } else if (req.api == c.API_RESPONSES) c.ds4_server_reuse_clear(srv, slot, c.API_RESPONSES);
            }
            if (cancelled(job)) {
                clearLive(srv, slot);
                self.trace(trace_id, "cancelled before publishing response state", .{});
                return;
            }
            self.toolSummary(context, &calls, responses);
            completion += recovery_completion;
            log(c.DS4_LOG_GENERATION, "ds4-server: reasoning_summary id=%s ctx=%s budget=%d thinking_tokens=%d forced=%d deferred=%d", .{ &id, context, req.reasoning_budget, generated_thinking, flag(forced), deferred });
            c.ds4_server_generation_trace_finish(srv, trace_id, req, final_finish, completion, saw_start, saw_end, if (content != null) @as([*c]const u8, content) else ptr(sampled.ptr), reasoning, &calls, now() - started);
            if (req.api == c.API_RESPONSES) {
                if (!same(final_finish, "error") and !same(final_finish, "length")) {
                    const suffix = assistantSuffix(req, ptr(content), reasoning, &calls, true);
                    defer c.free(suffix);
                    var visible = zero(c.buf);
                    defer freeBuffer(&visible);
                    puts(&visible, req.prompt_text);
                    puts(&visible, suffix);
                    c.ds4_server_reuse_remember(srv, slot, c.API_RESPONSES, ptr(visible.ptr), if (calls.len != 0) &calls else null, req);
                } else c.ds4_server_reuse_clear(srv, slot, c.API_RESPONSES);
            }
            if (req.api == c.API_ANTHROPIC) {
                if (calls.len != 0 and !same(final_finish, "error") and !same(final_finish, "length")) c.ds4_server_reuse_remember(srv, slot, c.API_ANTHROPIC, null, &calls, null) else c.ds4_server_reuse_clear(srv, slot, c.API_ANTHROPIC);
            }
            if (req.kind == c.REQ_CHAT and calls.len != 0 and req.api != c.API_RESPONSES and canonicalizeNeeded(srv, &calls)) {
                self.canonicalize(context, trace_id, ptr(content), reasoning, &calls);
                c.ds4_server_reuse_clear(srv, slot, c.API_OPENAI);
            } else if (calls.len != 0) {
                if (!self.rememberQwen(context, finish, thinking.inside, ptr(content), &calls)) c.ds4_server_reuse_clear(srv, slot, c.API_OPENAI);
            } else if (rememberThinking(req, &thinking, final_finish)) self.rememberVisible(context, trace_id, ptr(content), thinking_continued or disk_flags & c.DS4_KVSTORE_EXT_THINKING_VISIBLE != 0) else c.ds4_server_reuse_clear(srv, slot, c.API_OPENAI);
            var response_ok = !cancelled(job);
            if (req.api == c.API_RESPONSES and response_ok) appendReplay(req, content, reasoning, &calls, final_finish);
            const final_content = if (content != null) @as([*c]const u8, content) else ptr(sampled.ptr);
            if (response_ok and req.stream) {
                if (req.api == c.API_ANTHROPIC) response_ok = c.ds4_server_anthropic_finish(job.fd, memory, mutex, req, &id, &anthropic, ptr(sampled.ptr), sampled.len, &calls, final_finish, completion, self.stop) else if (openai_chat) response_ok = c.ds4_server_openai_finish(job.fd, memory, mutex, req, &id, &openai, ptr(sampled.ptr), sampled.len, &calls, final_finish, prompt_tokens, completion, self.stop) else if (responses_chat) response_ok = c.ds4_server_output_finish(job.fd, req, &output, ptr(sampled.ptr), sampled.len, if (parse_recovered) content else null, &calls, final_finish, prompt_tokens, completion, created, self.stop) else if (structured) response_ok = c.ds4_server_openai_chat_finish(job.fd, req, &id, final_content, reasoning, &calls, final_finish, prompt_tokens, completion, self.stop) else response_ok = c.ds4_server_openai_chunk(job.fd, req, &id, null, final_finish, self.stop) and c.ds4_server_openai_done(job.fd, req, &id, prompt_tokens, completion, self.stop);
            } else if (response_ok and req.api == c.API_ANTHROPIC) response_ok = c.ds4_server_anthropic_final(job.fd, srv.enable_cors, req, &id, final_content, reasoning, &calls, final_finish, prompt_tokens, completion, self.stop) else if (response_ok and req.api == c.API_RESPONSES) response_ok = c.ds4_server_output_final(job.fd, srv.enable_cors, req, final_content, reasoning, &calls, final_finish, prompt_tokens, completion, self.stop) else if (response_ok) response_ok = c.ds4_server_openai_final(job.fd, srv.enable_cors, req, &id, final_content, reasoning, &calls, final_finish, prompt_tokens, completion, self.stop);
            if (cancelled(job)) response_ok = false;
            if (req.api == c.API_RESPONSES) {
                _ = c.pthread_mutex_lock(&srv.tool_mu);
                report(&slot.last_response_id, "%s", .{if (response_ok) @as([*c]const u8, &req.responses_published_id) else @as([*c]const u8, "")});
                _ = c.pthread_mutex_unlock(&srv.tool_mu);
            }
            if (!response_ok) {
                markCancelled(job);
                final_finish = "error";
                report(errors, "client disconnected", .{});
                clearLive(srv, slot);
                log(c.DS4_LOG_DEFAULT, "ds4-server: %s ctx=%s%s%s client disconnected", .{ kind, context, space, request_flags });
            }
            var final_flags: [80]u8 = undefined;
            flags(&final_flags, responses, req.has_tools, thinking.inside, req.kind == c.REQ_CHAT and req.has_tools and saw_start, req.kind == c.REQ_CHAT and req.has_tools and saw_end);
            if (same(final_finish, "error") and errors.len != 0 and errors[0] != 0) log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs", .{ kind, context, completion, @as([*:0]const u8, if (final_flags[0] != 0) " " else ""), &final_flags, final_finish, errors.ptr, now() - started }) else log(c.DS4_LOG_GENERATION, "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs", .{ kind, context, completion, @as([*:0]const u8, if (final_flags[0] != 0) " " else ""), &final_flags, final_finish, now() - started });
            return;
        }
    }
    fn toolSummary(self: Generation, context: [*c]const u8, calls: *const c.tool_calls, responses: bool) void {
        _ = self;
        if (calls.len == 0) return;
        var names = zero(c.buf);
        defer freeBuffer(&names);
        var ids = zero(c.buf);
        defer freeBuffer(&ids);
        for (items(c.tool_call, calls.v, @intCast(calls.len)), 0..) |call, offset| {
            if (offset != 0) {
                append(&names, ",");
                append(&ids, ",");
            }
            puts(&names, if (call.name != null) call.name else @as([*c]const u8, "?"));
            puts(&ids, if (call.id != null) call.id else @as([*c]const u8, "?"));
        }
        log(c.DS4_LOG_TOOL, "ds4-server: tool calls ctx=%s%s%s n=%d raw_tool_text=%d ids=[%s] names=[%s]", .{ context, @as([*:0]const u8, if (responses) " " else ""), @as([*:0]const u8, if (responses) "RESPPROTO" else ""), calls.len, flag(text(calls.raw_tool_text).len != 0), ptr(ids.ptr), ptr(names.ptr) });
    }

    fn inner(self: Generation) void {
        const srv = self.runtime;
        const slot = self.slot;
        const job = self.job;
        const req = &job.req;
        var errors: [160]u8 = @splat(0);
        const multimodal = req.image_count != 0;
        _ = c.pthread_mutex_lock(&srv.inference_mu);
        var profile: [41]u8 = undefined;
        c.ds4_server_reuse_profile(srv, req, &profile);
        _ = c.pthread_mutex_lock(&srv.tool_mu);
        const discard = c.ds4_session_is_approximate(slot.session) and (profile[0] == 0 or !same(&profile, text(&slot.context_cache_profile)) or (req.previous_response_id != null and !same(req.previous_response_id, text(&slot.last_response_id))));
        slot.context_cache_profile = profile;
        _ = c.pthread_mutex_unlock(&srv.tool_mu);
        if (discard) {
            c.ds4_session_invalidate(slot.session);
            clearLive(srv, slot);
            c.ds4_server_reuse_refresh(srv, slot);
        }
        const old_position = c.ds4_session_pos(slot.session);
        const common = c.ds4_session_common_prefix(slot.session, &req.prompt);
        const vision_match = c.ds4_session_vision_prefix_matches(slot.session, req.images, req.image_count);
        _ = c.pthread_mutex_unlock(&srv.inference_mu);
        var diagnostic = captureDiagnostic(c.ds4_session_tokens(slot.session), &req.prompt, old_position, common);
        var effective = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&effective);
        var prompt: *const c.ds4_tokens = &req.prompt;
        const responses = req.api == c.API_RESPONSES;
        var responses_continued = false;
        var anthropic_continued = false;
        var thinking_continued = false;
        var response_match: [*c]const u8 = null;
        var response_ids: c_int = 0;
        var anthropic_ids: c_int = 0;
        // Routing and execution share one probe and exactly the same tier order.
        const reuse = c.ds4_server_reuse_probe(srv, slot, req);
        _ = c.pthread_mutex_lock(&srv.tool_mu);
        slot.last_response_id[0] = 0;
        _ = c.pthread_mutex_unlock(&srv.tool_mu);
        const unavailable_responses = responses and req.responses_requires_live_tool_state and reuse.kind != c.REUSE_RESPONSES_VISIBLE and reuse.kind != c.REUSE_RESPONSES_TOOL_OUTPUT;
        const unavailable_anthropic = req.api == c.API_ANTHROPIC and req.anthropic_requires_live_tool_state and reuse.kind != c.REUSE_ANTHROPIC_TOOL_OUTPUT;
        if (unavailable_responses or unavailable_anthropic) {
            self.httpError(409, if (unavailable_anthropic) "Anthropic continuation state is not available; retry by replaying the full messages history" else "Responses continuation state is not available; retry by replaying the full input history");
            return;
        }
        var cached = reuse.reuse_tokens;
        var cache_source: [*:0]const u8 = "none";
        var materialized = false;
        switch (reuse.kind) {
            c.REUSE_RESPONSES_VISIBLE, c.REUSE_RESPONSES_TOOL_OUTPUT, c.REUSE_ANTHROPIC_TOOL_OUTPUT, c.REUSE_THINKING_VISIBLE, c.REUSE_MEMORY_TEXT => {
                const suffix = switch (reuse.kind) {
                    c.REUSE_RESPONSES_TOOL_OUTPUT => req.responses_live_suffix_text,
                    c.REUSE_ANTHROPIC_TOOL_OUTPUT => req.anthropic_live_suffix_text,
                    else => req.prompt_text + reuse.suffix_off,
                };
                if (c.ds4_server_reuse_suffix(srv, slot, req, suffix, &effective)) {
                    materialized = true;
                    prompt = &effective;
                    switch (reuse.kind) {
                        c.REUSE_RESPONSES_VISIBLE => {
                            cache_source = "responses-visible";
                            responses_continued = true;
                            response_match = "visible-prefix";
                            if (c.ds4_server_reuse_matches(srv, slot, c.API_RESPONSES, &req.responses_live_call_ids, old_position)) response_ids = req.responses_live_call_ids.len;
                        },
                        c.REUSE_RESPONSES_TOOL_OUTPUT => {
                            cache_source = "responses-tool-output";
                            responses_continued = true;
                            response_match = "tool-output-ids";
                            response_ids = reuse.matched_ids;
                        },
                        c.REUSE_ANTHROPIC_TOOL_OUTPUT => {
                            cache_source = "anthropic-tool-output";
                            anthropic_continued = true;
                            anthropic_ids = reuse.matched_ids;
                        },
                        c.REUSE_THINKING_VISIBLE => {
                            cache_source = "thinking-visible";
                            thinking_continued = true;
                        },
                        else => cache_source = "memory-text",
                    }
                } else cached = 0;
            },
            c.REUSE_MEMORY_REWIND => {
                _ = c.pthread_mutex_lock(&srv.inference_mu);
                c.ds4_session_rewind(slot.session, reuse.reuse_tokens);
                const valid = c.ds4_session_common_prefix(slot.session, &req.prompt) == reuse.reuse_tokens and (!multimodal or c.ds4_session_vision_state_matches(slot.session, req.images, req.image_count));
                _ = c.pthread_mutex_unlock(&srv.inference_mu);
                if (valid) {
                    materialized = true;
                    cache_source = "memory-rewind";
                    diagnostic.rewind_to = reuse.reuse_tokens;
                    log(c.DS4_LOG_KVCACHE, "ds4-server: rewound GLM live prefix from %d to %d; final prompt token will be reevaluated", .{ old_position, reuse.reuse_tokens });
                } else {
                    cached = 0;
                    log(c.DS4_LOG_KVCACHE, "ds4-server: GLM live prefix rewind from %d to %d requires rebuild", .{ old_position, reuse.reuse_tokens });
                }
            },
            c.REUSE_MEMORY_TOKEN => {
                materialized = true;
                cache_source = "memory-token";
            },
            else => cached = 0,
        }
        if (c.ds4_server_reuse_unavailable(req, materialized)) {
            self.httpError(409, if (req.api == c.API_ANTHROPIC) "Anthropic continuation state is not available; retry by replaying the full messages history" else "Responses continuation state is not available; retry by replaying the full input history");
            return;
        }
        var disk_cached: c_int = 0;
        var disk_path: [*c]u8 = null;
        defer c.free(disk_path);
        var disk_flags: u8 = 0;
        if (cached == 0 and old_position > 0) log(c.DS4_LOG_WARNING, "ds4-server: live kv cache miss%s live=%d prompt=%d common=%d vision=%s reason=%s", .{ @as([*:0]const u8, if (responses) " RESPPROTO" else ""), old_position, req.prompt.len, common, @as([*:0]const u8, if (vision_match) "match" else "mismatch"), missReason(&diagnostic) });
        if (multimodal and cached > 0) log(c.DS4_LOG_KVCACHE, "ds4-server: multimodal live kv hit images=%zu cached=%d prompt=%d identity=fingerprint-match", .{ req.image_count, cached, prompt.len });
        if (cached == 0) slot.continued_last_store_tokens = 0;
        // Preserve the newer live checkpoint before replacing it from disk.
        if (!multimodal and srv.kv.enabled and cached == 0 and old_position >= srv.kv.opt.min_tokens) c.ds4_server_persist_store_current(srv, slot, "evict");
        if (!multimodal and cached == 0 and !c.ds4_server_reuse_canonical(srv, req)) {
            disk_cached = c.ds4_server_persist_load(srv, slot, req.prompt_text, &effective, &disk_path, &disk_flags, responses);
            if (disk_cached > 0) {
                cached = disk_cached;
                cache_source = "disk-text";
                prompt = &effective;
            }
        }
        // An ID/text splice preserves logical history, but Flint recomputes
        // its generated tail through prefill. Credit only its saved prefix.
        if (c.ds4_engine_is_flint(srv.engine)) cached = c.ds4_session_common_prefix(slot.session, prompt);
        const reasoning_preserved = cached > 0 and (same(cache_source, "responses-visible") or same(cache_source, "responses-tool-output") or (same(cache_source, "disk-text") and disk_flags & c.DS4_KVSTORE_EXT_RESPONSES_VISIBLE != 0));
        const visible_without_reasoning = responses and req.responses_requires_live_reasoning and !reasoning_preserved;
        var prompt_tokens = prompt.len;
        req.cache_read_tokens = cached;
        req.cache_write_tokens = @max(@as(c_int, 0), prompt_tokens - cached);
        const started = now();
        const trace_id = c.ds4_server_generation_trace_begin(srv, job, cached, prompt_tokens, &diagnostic, cache_source, disk_cached, disk_path);
        var context: [48]u8 = undefined;
        contextSpan(&context, cached, prompt_tokens);
        var progress_context = ProgressContext{ .packet = zero(c.server_prefill_progress), .stop = self.stop };
        const p = &progress_context.packet;
        p.srv = srv;
        p.slot = slot;
        p.kind = req.kind;
        p.prompt_tokens = prompt_tokens;
        p.cached_tokens = cached;
        p.has_tools = req.has_tools;
        p.responses_protocol = responses;
        p.t0 = started;
        p.fd = job.fd;
        p.stream = req.stream;
        p.enable_cors = srv.enable_cors;
        p.request_job = job;
        report(&p.ctx, "%s", .{&context});
        var request_flags: [64]u8 = undefined;
        flags(&request_flags, responses, req.has_tools, false, false, false);
        if (responses_continued) log(c.DS4_LOG_PREFILL, "ds4-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d approximate=%d", .{ if (response_match != null) response_match else @as([*c]const u8, "unknown"), response_ids, cached, prompt_tokens, flag(c.ds4_session_is_approximate(slot.session)) }) else if (anthropic_continued) log(c.DS4_LOG_PREFILL, "ds4-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d", .{ anthropic_ids, cached, prompt_tokens }) else if (thinking_continued) log(c.DS4_LOG_PREFILL, "ds4-server: thinking live continuation match=visible-prefix cached=%d prompt=%d", .{ cached, prompt_tokens });
        if (visible_without_reasoning) {
            // The client omitted hidden reasoning. A cold recovery from that
            // visible surface is intentional lower fidelity, not a hard error.
            log(c.DS4_LOG_WARNING, "ds4-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d", .{ cache_source, cached, prompt_tokens });
            self.trace(trace_id, "responses replay missing reasoning state; continuing from visible history source=%s cached=%d", .{ cache_source, cached });
        }
        const kind: [*:0]const u8 = if (req.kind == c.REQ_CHAT) "chat" else "completion";
        log(c.DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s prompt start", .{ kind, &context, @as([*:0]const u8, if (request_flags[0] != 0) " " else ""), &request_flags });
        setProgress(slot.session, &progress_context);
        if (!multimodal and req.previous_response_id == null and !materialized and prompt == &req.prompt and !req.responses_requires_live_tool_state and !req.anthropic_requires_live_tool_state) {
            var stats = zero(c.ds4_suffix_relocation_stats);
            if (c.ds4_server_persist_suffix(srv, slot, prompt, &effective, &stats, self.stop) > 0) {
                if (effective.len != 0) {
                    prompt = &effective;
                    prompt_tokens = prompt.len;
                    p.prompt_tokens = prompt_tokens;
                }
                cached = @intCast(stats.reuse.reused_tokens);
                cache_source = "suffix-approximate";
                req.cache_read_tokens = cached;
                req.cache_write_tokens = prompt_tokens - cached;
                p.cached_tokens = cached;
                contextSpan(&context, cached, prompt_tokens);
                report(&p.ctx, "%s", .{&context});
                self.trace(trace_id, "suffix APPROXIMATE reused=%u fresh=%u source=%u:%u destination=%u:%u", .{ stats.reuse.reused_tokens, stats.reuse.fresh_tail_tokens, stats.source_start, stats.source_end, stats.destination_start, stats.destination_end });
            }
        }
        var credits: c_int = 0;
        var cold_length: c_int = 0;
        if (!multimodal and cached == 0 and srv.kv.enabled and prompt.len >= srv.kv.opt.min_tokens and srv.kv.opt.cold_max_tokens > 0 and prompt.len <= srv.kv.opt.cold_max_tokens) {
            const anchor = c.ds4_kvstore_chat_anchor_pos(&srv.kv, prompt, c.ds4_token_user(srv.engine), c.ds4_token_assistant(srv.engine));
            cold_length = if (anchor >= srv.kv.opt.min_tokens) anchor else c.ds4_kvstore_store_len(&srv.kv, prompt.len);
        }
        var suppressed: c_int = -1;
        // Suppress a duplicate continued write at the intentional cold stop;
        // failed cold publication restores the old continuation schedule.
        if (cold_length >= srv.kv.opt.min_tokens) suppressed = c.ds4_server_persist_suppress(srv, slot, cold_length);
        if (srv.kv.enabled and cold_length >= srv.kv.opt.min_tokens and cold_length < prompt.len) {
            var prefix = zero(c.ds4_tokens);
            c.ds4_kvstore_tokens_copy_prefix(&prefix, prompt, cold_length);
            const result = sync(srv, slot, &prefix, &credits, &errors, self.stop);
            c.ds4_tokens_free(&prefix);
            if (result != 0) {
                clearProgress(slot.session);
                c.ds4_server_persist_restore_suppressed(slot, suppressed, cold_length);
                c.ds4_server_persist_discard(srv, slot, disk_path);
                c.free(disk_path);
                disk_path = null;
                if (cancelled(job)) {
                    clearLive(srv, slot);
                    self.trace(trace_id, "cancelled during prefill", .{});
                    return;
                }
                self.trace(trace_id, "prefill failed: %s", .{&errors});
                self.prefillFailure(p, &context, &request_flags, &errors);
                return;
            }
            if (c.ds4_server_persist_store(srv, slot, prompt, cold_length, "cold", null, 0, null)) c.ds4_server_persist_note_store(slot, cold_length) else c.ds4_server_persist_restore_suppressed(slot, suppressed, cold_length);
            suppressed = -1;
        }
        const synchronized = if (multimodal) syncMultimodal(srv, slot, prompt, req.images, req.image_count, &errors, self.stop) else sync(srv, slot, prompt, &credits, &errors, self.stop);
        if (synchronized != 0) {
            clearProgress(slot.session);
            c.ds4_server_persist_restore_suppressed(slot, suppressed, cold_length);
            c.ds4_server_persist_discard(srv, slot, disk_path);
            c.free(disk_path);
            disk_path = null;
            if (cancelled(job)) {
                clearLive(srv, slot);
                self.trace(trace_id, "cancelled during prefill", .{});
                return;
            }
            self.trace(trace_id, "prefill failed: %s", .{&errors});
            self.prefillFailure(p, &context, &request_flags, &errors);
            return;
        }
        if (credits > 0) {
            cached = @max(@as(c_int, 0), @min(cached + credits, prompt_tokens));
            req.cache_read_tokens = cached;
            req.cache_write_tokens = prompt_tokens - cached;
            p.cached_tokens = cached;
            contextSpan(&context, cached, prompt_tokens);
            report(&p.ctx, "%s", .{&context});
            self.trace(trace_id, "state cache SSD credits=%d cached=%d prompt=%d", .{ credits, cached, prompt_tokens });
        }
        c.free(disk_path);
        disk_path = null;
        if (cancelled(job)) {
            clearProgress(slot.session);
            clearLive(srv, slot);
            self.trace(trace_id, "cancelled after prefill", .{});
            return;
        }
        if (!responses_continued) c.ds4_server_reuse_clear(srv, slot, c.API_RESPONSES);
        if (!anthropic_continued) c.ds4_server_reuse_clear(srv, slot, c.API_ANTHROPIC);
        if (!thinking_continued) c.ds4_server_reuse_clear(srv, slot, c.API_OPENAI);
        clearProgress(slot.session);
        if (!multimodal) c.ds4_server_persist_maybe_store(srv, slot);
        log(c.DS4_LOG_PREFILL, "ds4-server: %s ctx=%s%s%s prompt done %.3fs", .{ kind, &context, @as([*:0]const u8, if (request_flags[0] != 0) " " else ""), &request_flags, now() - started });
        if (cold_length == prompt.len) {
            if (!multimodal and c.ds4_server_persist_store(srv, slot, prompt, cold_length, "cold", null, 0, null)) c.ds4_server_persist_note_store(slot, cold_length) else c.ds4_server_persist_restore_suppressed(slot, suppressed, cold_length);
        }
        self.decode(prompt_tokens, &context, &request_flags, &progress_context, responses, thinking_continued, disk_flags, started, trace_id, &errors);
    }
};
