//! Whole resident routing, queue/worker lifetime and one-request HTTP serving.
//! All shared records are the actual C declarations; model handles stay opaque.
const std = @import("std");
const c = @import("server_serving_c");
const ModelBackend = @import("model.zig").Backend(c);
const ModelPolicy = @import("ds4_model");
const WebSocket = @import("websocket.zig");
const allocator = std.heap.c_allocator;
const stale_seconds: c.time_t = 60 * 60;
const idle_cap: c_int = 1 << 20;
const protected_base: c_int = -2 - idle_cap;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
fn same(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, text(value), expected);
}
fn items(comptime T: type, pointer: [*c]T, length: usize) []T {
    return if (length == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..length];
}
fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
fn append(out: *c.buf, value: []const u8) void {
    var list: std.ArrayList(u8) = .empty;
    list.items = items(u8, out.ptr, out.len);
    list.capacity = out.cap;
    list.appendSlice(allocator, value) catch fatal();
    list.ensureUnusedCapacity(allocator, 1) catch fatal();
    list.items.ptr[list.items.len] = 0;
    out.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
}
fn freeBuffer(out: *c.buf) void {
    c.free(out.ptr);
    out.* = zero(c.buf);
}
fn quote(out: *c.buf, value: [*c]const u8) void {
    append(out, "\"");
    const hex = "0123456789abcdef";
    for (text(value)) |byte| switch (byte) {
        '"', '\\' => append(out, &.{ '\\', byte }),
        '\n' => append(out, "\\n"),
        '\r' => append(out, "\\r"),
        '\t' => append(out, "\\t"),
        else => if (byte < 32) append(out, &.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15] }) else append(out, &.{byte}),
    };
    append(out, "\"");
}
fn number(out: *c.buf, value: c_int) void {
    var bytes: [32]u8 = undefined;
    const value_text = std.mem.print(&bytes, "{d}", .{value}) catch unreachable;
    append(out, value_text);
}
fn log(comptime pattern: [*:0]const u8, arguments: anytype) void {
    const size = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), pattern } ++ arguments);
    if (size < 0) return;
    const line = allocator.allocSentinel(u8, @intCast(size), 0) catch fatal();
    defer allocator.free(line);
    _ = @call(.auto, c.snprintf, .{ line.ptr, line.len + 1, pattern } ++ arguments);
    c.ds4_server_persist_log(c.DS4_LOG_KVCACHE, line.ptr);
}
pub fn modelId(engine: ?*c.ds4_engine) [*:0]const u8 {
    return (ModelBackend{ .engine = engine }).model().id().ptr;
}
pub fn knownModel(value: [*c]const u8) bool {
    return value != null and ModelPolicy.knownModel(text(value));
}
pub fn decodeInput(engine: ?*c.ds4_engine, runtime: ?*c.server, endpoint: c.ds4_server_input_endpoint, body: [*c]const u8, default_tokens: c_int, context: c_int, out: *c.request, errors: []u8) bool {
    return decodeWithSink(engine, runtime, endpoint, body, default_tokens, context, null, out, errors);
}
pub fn decodeWithSink(engine: ?*c.ds4_engine, runtime: ?*c.server, endpoint: c.ds4_server_input_endpoint, body: [*c]const u8, default_tokens: c_int, context: c_int, sink: ?*const c.server_response_sink, out: *c.request, errors: []u8) bool {
    const backend: ModelBackend = .{ .engine = engine };
    const syntax = backend.syntax();
    const incremental = backend.incrementalTools();
    const defaults = c.ds4_server_input_defaults{ .max_tokens = default_tokens, .ctx_size = context, .syntax = syntax, .responses_store_dir = if (runtime) |srv| srv.responses_store_dir else null, .incremental_tool_schemas = incremental };
    var input = zero(c.ds4_server_input);
    if (!c.ds4_server_input_decode(endpoint, body, &defaults, &input, errors.ptr, errors.len)) {
        out.* = zero(c.request);
        return false;
    }
    defer c.ds4_server_input_deinit(&input);
    input.req.response_sink = sink;
    return c.ds4_server_materialize(engine, runtime, endpoint, &input, context, incremental, out, errors.ptr, errors.len);
}
pub fn exceedsContext(req: [*c]const c.request, context: c_int) bool {
    return req != null and req.*.prompt.len >= context;
}

fn contains(state: *const c.live_tool_state, ids: [*c]const c.stop_list) bool {
    if (!state.valid or ids == null or ids.*.len == 0) return false;
    for (items([*c]u8, ids.*.v, @intCast(ids.*.len))) |wanted| {
        if (wanted == null or wanted[0] == 0) return false;
        var found = false;
        for (items([*c]u8, state.call_ids.v, @intCast(state.call_ids.len))) |have| if (have != null and same(have, text(wanted))) {
            found = true;
            break;
        };
        if (!found) return false;
    }
    return true;
}
pub fn requiredSlot(runtime: ?*c.server, pending: ?*const c.job) c_int {
    const srv = runtime orelse return -1;
    const job = pending orelse return -1;
    for (items(c.server_slot, srv.slots, @intCast(srv.slot_count)), 0..) |*slot, index| {
        if (job.req.responses_requires_live_tool_state and contains(&slot.responses_live, &job.req.responses_live_call_ids)) return @intCast(index);
        if (job.req.anthropic_requires_live_tool_state and contains(&slot.anthropic_live, &job.req.anthropic_live_call_ids)) return @intCast(index);
    }
    return -1;
}
/// Any reusable prefix beats an empty slot. With no reuse, stale idle
/// checkpoints are evicted before protected ones; within the protected tier,
/// a shorter checkpoint loses less work. Staleness never prevents its owner
/// from reusing a checkpoint, because the reuse tier is evaluated first.
pub fn score(runtime: ?*c.server, member: ?*c.server_slot, pending: ?*const c.job, required: c_int, stamp: c.time_t) c_int {
    const srv = runtime orelse return std.math.minInt(c_int);
    const slot = member orelse return std.math.minInt(c_int);
    const job = pending orelse return std.math.minInt(c_int);
    if (slot.busy or slot.assigned != null or (required >= 0 and slot.id != required)) return std.math.minInt(c_int);
    if (required == slot.id) return std.math.maxInt(c_int);
    const live = if (slot.session != null and c.ds4_session_checkpoint_valid(slot.session)) c.ds4_session_pos(slot.session) else 0;
    const reusable = c.ds4_server_reuse_probe_locked(srv, slot, &job.req).reuse_tokens;
    if (reusable > 0) return reusable;
    if (live == 0) return 0;
    const idle = if (stamp > slot.last_used) stamp - slot.last_used else 0;
    if (idle >= stale_seconds) return -1 - idle_cap + @as(c_int, @intCast(@min(idle, idle_cap)));
    return protected_base - live;
}
/// Caller holds mu. FIFO among runnable jobs; strict comparisons retain the
/// original first-slot tie order. tool_mu covers the complete shared probe.
pub fn dispatch(runtime: ?*c.server) void {
    const srv = runtime orelse return;
    if (!srv.batched_mode) return;
    const stamp = c.time(null);
    while (true) {
        var chosen: [*c]c.job = null;
        var predecessor: [*c]c.job = null;
        var member: [*c]c.server_slot = null;
        var chosen_score: c_int = std.math.minInt(c_int);
        var evicted: c_int = 0;
        var idle: c.time_t = 0;
        _ = c.pthread_mutex_lock(&srv.tool_mu);
        var previous: [*c]c.job = null;
        var pending = srv.head;
        while (pending != null) : ({
            previous = pending;
            pending = pending.*.next;
        }) {
            const required = requiredSlot(srv, @ptrCast(pending));
            var best: [*c]c.server_slot = null;
            var best_score: c_int = std.math.minInt(c_int);
            for (items(c.server_slot, srv.slots, @intCast(srv.slot_count))) |*slot| {
                const value = score(srv, slot, @ptrCast(pending), required, stamp);
                if (value > best_score) {
                    best_score = value;
                    best = slot;
                }
            }
            if (best != null) {
                chosen = pending;
                predecessor = previous;
                member = best;
                chosen_score = best_score;
                break;
            }
        }
        if (chosen != null and chosen_score < 0 and member.*.session != null and c.ds4_session_checkpoint_valid(member.*.session)) {
            evicted = c.ds4_session_pos(member.*.session);
            idle = if (stamp > member.*.last_used) stamp - member.*.last_used else 0;
        }
        _ = c.pthread_mutex_unlock(&srv.tool_mu);
        if (chosen == null or member == null) break;
        if (predecessor != null) predecessor.*.next = chosen.*.next else srv.head = chosen.*.next;
        if (srv.tail == chosen) srv.tail = predecessor;
        chosen.*.next = null;
        member.*.assigned = chosen;
        member.*.busy = true;
        if (evicted > 0) log("slot %d: request reuses nothing; evicting checkpoint (%d tokens, idle %lld s, %s)", .{ member.*.id, evicted, @as(c_longlong, idle), @as([*:0]const u8, if (idle >= stale_seconds) "stale" else "protected") });
        _ = c.pthread_cond_broadcast(&srv.cv);
    }
}
pub fn enqueue(srv: *c.server, job: *c.job) bool {
    _ = c.pthread_mutex_lock(&srv.mu);
    defer _ = c.pthread_mutex_unlock(&srv.mu);
    if (srv.stopping) return false;
    if (srv.tail != null) srv.tail.*.next = job else srv.head = job;
    srv.tail = job;
    if (srv.batched_mode) {
        dispatch(srv);
        _ = c.pthread_cond_broadcast(&srv.cv);
    } else _ = c.pthread_cond_signal(&srv.cv);
    return true;
}
fn dequeue(srv: *c.server) ?*c.job {
    _ = c.pthread_mutex_lock(&srv.mu);
    while (srv.head == null and !srv.stopping) _ = c.pthread_cond_wait(&srv.cv, &srv.mu);
    if (srv.head == null) {
        _ = c.pthread_mutex_unlock(&srv.mu);
        return null;
    }
    const job: *c.job = @ptrCast(srv.head);
    srv.head = job.next;
    if (srv.head == null) srv.tail = null;
    _ = c.pthread_mutex_unlock(&srv.mu);
    job.next = null;
    return job;
}
fn complete(job: *c.job) void {
    _ = c.pthread_mutex_lock(&job.mu);
    job.done = true;
    _ = c.pthread_cond_signal(&job.cv);
    _ = c.pthread_mutex_unlock(&job.mu);
}
pub fn worker(srv: *c.server, stop: [*c]const volatile c.sig_atomic_t) void {
    while (dequeue(srv)) |job| {
        c.ds4_server_generation_run(srv, &srv.slots[0], job, stop);
        complete(job);
        c.ds4_server_reuse_refresh(srv, &srv.slots[0]);
        srv.slots[0].last_used = c.time(null);
    }
}
pub fn slotWorker(slot: *c.server_slot, stop: [*c]const volatile c.sig_atomic_t) void {
    const srv: *c.server = @ptrCast(slot.srv);
    while (true) {
        _ = c.pthread_mutex_lock(&srv.mu);
        while (slot.assigned == null and (!srv.stopping or srv.head != null)) _ = c.pthread_cond_wait(&srv.cv, &srv.mu);
        if (slot.assigned == null and srv.stopping and srv.head == null) {
            _ = c.pthread_mutex_unlock(&srv.mu);
            break;
        }
        const job: *c.job = @ptrCast(slot.assigned);
        slot.assigned = null;
        _ = c.pthread_mutex_unlock(&srv.mu);
        c.ds4_server_generation_run(srv, slot, job, stop);
        complete(job);
        // Refresh the checkpoint before exposing an idle slot to the router.
        c.ds4_server_reuse_refresh(srv, slot);
        slot.last_used = c.time(null);
        _ = c.pthread_mutex_lock(&srv.mu);
        slot.busy = false;
        dispatch(srv);
        _ = c.pthread_mutex_unlock(&srv.mu);
    }
}
pub fn cancel(srv: *c.server, job: *c.job) void {
    _ = c.pthread_mutex_lock(&job.mu);
    job.cancelled = true;
    _ = c.pthread_mutex_unlock(&job.mu);
    // Never carry job.mu into either scheduler mutex. Only detach work that
    // has no backend owner; an in-flight request must reach its safe boundary.
    var detached = false;
    _ = c.pthread_mutex_lock(&srv.mu);
    var previous: [*c]c.job = null;
    var current = srv.head;
    while (current != null) : ({
        previous = current;
        current = current.*.next;
    }) {
        if (current != job) continue;
        if (previous != null) previous.*.next = current.*.next else srv.head = current.*.next;
        if (srv.tail == current) srv.tail = previous;
        current.*.next = null;
        detached = true;
        break;
    }
    if (!detached and srv.batched_mode) for (items(c.server_slot, srv.slots, @intCast(srv.slot_count))) |*slot| {
        if (slot.assigned != job) continue;
        slot.assigned = null;
        slot.busy = false;
        detached = true;
        dispatch(srv);
        break;
    };
    _ = c.pthread_cond_broadcast(&srv.cv);
    _ = c.pthread_mutex_unlock(&srv.mu);
    _ = c.pthread_mutex_lock(&srv.model_mu);
    for (items(c.server_slot, srv.slots, @intCast(srv.slot_count))) |*slot| if (slot.running == job) {
        _ = c.ds4_server_coordinator_cancel_pending(srv, slot);
        break;
    };
    _ = c.pthread_cond_broadcast(&srv.model_cv);
    _ = c.pthread_mutex_unlock(&srv.model_mu);
    if (detached) complete(job);
}
pub fn waitForJob(srv: *c.server, job: *c.job) void {
    _ = c.pthread_mutex_lock(&job.mu);
    while (!job.done) {
        var deadline: c.struct_timespec = undefined;
        _ = c.clock_gettime(c.CLOCK_REALTIME, &deadline);
        deadline.tv_nsec += 100000000;
        deadline.tv_sec += @divTrunc(deadline.tv_nsec, 1000000000);
        deadline.tv_nsec = @rem(deadline.tv_nsec, 1000000000);
        _ = c.pthread_cond_timedwait(&job.cv, &job.mu, &deadline);
        const done = job.done;
        _ = c.pthread_mutex_unlock(&job.mu);
        if (!done and c.ds4_server_http_disconnected(job.fd)) cancel(srv, job);
        _ = c.pthread_mutex_lock(&job.mu);
    }
    _ = c.pthread_mutex_unlock(&job.mu);
}
pub fn requestStop(srv: *c.server) void {
    _ = c.pthread_mutex_lock(&srv.mu);
    srv.stopping = true;
    dispatch(srv);
    _ = c.pthread_cond_broadcast(&srv.cv);
    _ = c.pthread_mutex_unlock(&srv.mu);
}

fn response(srv: *c.server, fd: c_int, status: c_int, body: [*c]const u8, stop: [*c]const volatile c.sig_atomic_t) void {
    _ = c.ds4_server_http_response(fd, srv.enable_cors, status, "application/json", body, stop);
}
fn httpError(srv: *c.server, fd: c_int, status: c_int, message: [*c]const u8, stop: [*c]const volatile c.sig_atomic_t) void {
    var out = zero(c.buf);
    defer freeBuffer(&out);
    append(&out, "{\"error\":{\"message\":");
    quote(&out, message);
    append(&out, ",\"type\":\"invalid_request_error\"}}\n");
    response(srv, fd, status, out.ptr, stop);
}
fn contextError(srv: *c.server, fd: c_int, req: *const c.request, stop: [*c]const volatile c.sig_atomic_t) void {
    var message: [160]u8 = undefined;
    _ = c.snprintf(&message, message.len, "Prompt has %d tokens, but the configured context size is %d tokens", req.prompt.len, srv.ctx_size);
    var out = zero(c.buf);
    defer freeBuffer(&out);
    if (req.api == c.API_ANTHROPIC) append(&out, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":") else append(&out, "{\"error\":{\"message\":");
    quote(&out, &message);
    if (req.api != c.API_ANTHROPIC) {
        append(&out, ",\"type\":\"invalid_request_error\",\"param\":");
        quote(&out, if (req.api == c.API_RESPONSES) "input" else if (req.kind == c.REQ_COMPLETION) "prompt" else "messages");
        append(&out, ",\"code\":\"context_length_exceeded\"");
    }
    append(&out, ",\"n_prompt_tokens\":");
    number(&out, req.prompt.len);
    append(&out, ",\"n_ctx\":");
    number(&out, srv.ctx_size);
    append(&out, "}}\n");
    response(srv, fd, 400, out.ptr, stop);
}
pub fn modelValues(out: *c.buf, id: [*c]const u8, name: [*c]const u8, context: c_int, default_tokens: c_int) void {
    append(out, "{\"id\":");
    quote(out, id);
    append(out, ",\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"ds4.c\",\"name\":");
    quote(out, name);
    append(out, ",\"context_length\":");
    number(out, context);
    append(out, ",\"top_provider\":{\"context_length\":");
    number(out, context);
    append(out, ",\"max_completion_tokens\":");
    number(out, @min(default_tokens, context));
    append(out, ",\"is_moderated\":false},\"supported_parameters\":[\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\",\"top_p\",\"top_k\",\"min_p\",\"ignore_eos\",\"stop\",\"seed\",\"stream\"");
    if (!same(id, "flint") and !same(id, "flint-int4")) append(out, ",\"reasoning_effort\"");
    append(out, "]}");
}
fn models(srv: *c.server, fd: c_int, id: [*c]const u8, stop: [*c]const volatile c.sig_atomic_t) void {
    var out = zero(c.buf);
    defer freeBuffer(&out);
    if (id != null) {
        modelValues(&out, id, c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens);
        append(&out, "\n");
    } else {
        append(&out, "{\"object\":\"list\",\"data\":[");
        if (c.ds4_engine_is_deepseek41(srv.engine) or c.ds4_engine_is_flint(srv.engine)) modelValues(&out, modelId(srv.engine), c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens) else if (c.ds4_engine_is_qwen4(srv.engine)) {
            for ([_][*:0]const u8{ "qwen3.8-flash-next", "qwen3.8-flash-next-chat", "qwen3.8-flash-next-reasoner" }, 0..) |alias, index| {
                if (index != 0) append(&out, ",");
                modelValues(&out, alias, c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens);
            }
        } else if (c.ds4_engine_is_glm_dsa(srv.engine)) {
            const base = modelId(srv.engine);
            var variant: [64]u8 = undefined;
            modelValues(&out, base, c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens);
            for ([_][*:0]const u8{ "-chat", "-reasoner" }) |suffix| {
                _ = c.snprintf(&variant, variant.len, "%s%s", base, suffix);
                append(&out, ",");
                modelValues(&out, &variant, c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens);
            }
        } else {
            for ([_][*:0]const u8{ "deepseek-v4-flash", "deepseek-v4-pro" }, 0..) |alias, index| {
                if (index != 0) append(&out, ",");
                modelValues(&out, alias, c.ds4_engine_model_name(srv.engine), srv.ctx_size, srv.default_tokens);
            }
        }
        append(&out, "]}\n");
    }
    response(srv, fd, 200, out.ptr, stop);
}

/// One connection owns its socket until every backend use of its stack Job is
/// finished. Worker completion follows callback retirement, never mere cancel.
pub fn client(srv: *c.server, fd: c_int, stop: [*c]const volatile c.sig_atomic_t) void {
    defer {
        _ = c.close(fd);
        _ = c.pthread_mutex_lock(&srv.mu);
        if (srv.clients > 0) srv.clients -= 1;
        _ = c.pthread_cond_broadcast(&srv.clients_cv);
        _ = c.pthread_mutex_unlock(&srv.mu);
    }
    var http = zero(c.ds4_server_http_request);
    defer c.ds4_server_http_request_free(&http);
    if (!c.ds4_server_http_read_request(fd, &http)) {
        httpError(srv, fd, 400, "bad HTTP request", stop);
        return;
    }
    if (same(&http.method, "OPTIONS")) {
        _ = c.ds4_server_http_response(fd, srv.enable_cors, 204, null, "", stop);
        return;
    }
    if (same(&http.method, "GET") and same(&http.path, "/v1/responses")) {
        WebSocket.serve(srv, fd, &http, @ptrCast(stop));
        return;
    }
    if (same(&http.method, "GET") and same(&http.path, "/v1/models")) {
        models(srv, fd, null, stop);
        return;
    }
    if (std.mem.startsWith(u8, text(&http.path), "/v1/conversations")) {
        var json: [*c]u8 = null;
        defer c.free(json);
        var errors: [160]u8 = @splat(0);
        const status = c.ds4_conversation_route(srv.responses_store_dir, &http.method, &http.path, http.body, &json, &errors, errors.len);
        if (status != 0) {
            if (status >= 400) httpError(srv, fd, status, &errors, stop) else response(srv, fd, status, json, stop);
            return;
        }
    }
    const response_prefix = "/v1/responses/";
    if ((same(&http.method, "GET") or same(&http.method, "DELETE")) and std.mem.startsWith(u8, text(&http.path), response_prefix)) {
        const id: [*c]const u8 = @as([*c]const u8, @ptrCast(&http.path)) + response_prefix.len;
        var errors: [160]u8 = @splat(0);
        if (srv.responses_store_dir == null) httpError(srv, fd, 404, "response storage is not configured", stop) else if (!c.ds4_response_store_valid_id(id)) httpError(srv, fd, 400, "invalid response id", stop) else if (same(&http.method, "DELETE")) {
            const removed = c.ds4_response_store_delete(srv.responses_store_dir, id, &errors, errors.len);
            if (removed != 0) httpError(srv, fd, if (removed == 1) 404 else 500, if (removed == 1) "response was not found" else @as([*c]const u8, &errors), stop) else {
                var out = zero(c.buf);
                defer freeBuffer(&out);
                append(&out, "{\"id\":\"");
                append(&out, text(id));
                append(&out, "\",\"object\":\"response.deleted\",\"deleted\":true}");
                response(srv, fd, 200, out.ptr, stop);
            }
        } else {
            var json: [*c]u8 = null;
            defer c.free(json);
            var replay: [*c]u8 = null;
            defer c.free(replay);
            const loaded = c.ds4_server_response_record_load(srv.responses_store_dir, id, &json, &replay, &errors, errors.len);
            if (loaded != 0) httpError(srv, fd, if (loaded == 1) 404 else 500, if (loaded == 1) "response was not found" else @as([*c]const u8, &errors), stop) else response(srv, fd, 200, json, stop);
        }
        return;
    }
    const model_prefix = "/v1/models/";
    if (same(&http.method, "GET") and std.mem.startsWith(u8, text(&http.path), model_prefix)) {
        const id: [*c]const u8 = @as([*c]const u8, @ptrCast(&http.path)) + model_prefix.len;
        if (knownModel(id)) {
            models(srv, fd, id, stop);
            return;
        }
    }
    const endpoint: c.ds4_server_input_endpoint = if (same(&http.method, "POST") and same(&http.path, "/v1/messages")) c.DS4_SERVER_INPUT_ANTHROPIC else if (same(&http.method, "POST") and same(&http.path, "/v1/chat/completions")) c.DS4_SERVER_INPUT_CHAT else if (same(&http.method, "POST") and same(&http.path, "/v1/responses")) c.DS4_SERVER_INPUT_RESPONSES else if (same(&http.method, "POST") and same(&http.path, "/v1/completions")) c.DS4_SERVER_INPUT_COMPLETION else {
        httpError(srv, fd, 404, "unknown endpoint", stop);
        return;
    };
    var req = zero(c.request);
    var errors: [160]u8 = undefined;
    const ok = decodeInput(srv.engine, if (endpoint == c.DS4_SERVER_INPUT_COMPLETION) null else srv, endpoint, http.body, srv.default_tokens, srv.ctx_size, &req, &errors);
    if (ok) req.raw_body = duplicate(items(u8, http.body, http.body_len));
    c.ds4_server_http_request_free(&http);
    if (!ok) {
        httpError(srv, fd, 400, &errors, stop);
        return;
    }
    if (!req.model_from_request) {
        c.free(req.model);
        req.model = duplicate(text(modelId(srv.engine)));
    }
    if (exceedsContext(&req, srv.ctx_size)) {
        contextError(srv, fd, &req, stop);
        c.ds4_server_input_request_free(&req);
        return;
    }
    c.ds4_server_http_nonblocking(fd);
    var job = zero(c.job);
    job.fd = fd;
    job.req = req;
    _ = c.pthread_mutex_init(&job.mu, null);
    _ = c.pthread_cond_init(&job.cv, null);
    if (!enqueue(srv, &job)) httpError(srv, fd, 503, "server shutting down", stop) else waitForJob(srv, &job);
    _ = c.pthread_cond_destroy(&job.cv);
    _ = c.pthread_mutex_destroy(&job.mu);
    c.ds4_server_input_request_free(&job.req);
}
