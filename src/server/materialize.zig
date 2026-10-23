//! One decoded Input becomes one owned Request. Durable history is resolved
//! before missing-input checks, sampled tool memory is attached before render,
//! and the Request commits only after prompt tokenization succeeds.
const std = @import("std");
const c = @import("server_materialize_c");
const ModelBackend = @import("model.zig").Backend(c);
const allocator = std.heap.c_allocator;
const Invalid = error{InvalidRequest};

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
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    // Allocated empty vectors retain their base when adopting C capacity.
    return if (count == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
fn errorPointer(errors: []u8) [*c]u8 {
    return if (errors.len == 0) null else errors.ptr;
}
fn report(errors: []u8, message: []const u8) void {
    if (errors.len != 0) {
        const length = @min(errors.len - 1, message.len);
        @memcpy(errors[0..length], message[0..length]);
        errors[length] = 0;
    }
}
fn fail(errors: []u8, message: []const u8) Invalid {
    report(errors, message);
    return error.InvalidRequest;
}
fn take(pointer: *[*c]u8) [*c]u8 {
    const result = pointer.*;
    pointer.* = null;
    return result;
}
fn messageItems(messages: *const c.chat_msgs) []const c.chat_msg {
    return items(c.chat_msg, messages.v, @intCast(messages.len));
}
fn appendMessage(messages: *c.chat_msgs, message: c.chat_msg) void {
    var list: std.ArrayList(c.chat_msg) = .empty;
    list.items = items(c.chat_msg, messages.v, @intCast(messages.len));
    list.capacity = @intCast(messages.cap);
    list.append(allocator, message) catch fatal();
    messages.* = .{ .v = list.items.ptr, .len = @intCast(list.items.len), .cap = @intCast(list.capacity) };
}
fn prependSystem(messages: *c.chat_msgs, content: [*c]u8, first: bool) void {
    var message = std.mem.zeroes(c.chat_msg);
    message.role = duplicate("system");
    message.content = content;
    appendMessage(messages, message);
    if (first) {
        const values = items(c.chat_msg, messages.v, @intCast(messages.len));
        std.mem.copyBackwards(c.chat_msg, values[1..], values[0 .. values.len - 1]);
        values[0] = message;
    }
}
fn prependHistory(messages: *c.chat_msgs, history: *c.chat_msgs) void {
    for (messageItems(messages)) |message| appendMessage(history, message);
    c.free(messages.v);
    messages.* = history.*;
    history.* = std.mem.zeroes(c.chat_msgs);
}
fn systemRole(role: [*c]const u8) bool {
    return equal(role, "system") or equal(role, "developer");
}
fn toolRole(role: [*c]const u8) bool {
    return equal(role, "tool") or equal(role, "function");
}
fn anthropicResult(message: *const c.chat_msg) bool {
    return equal(message.role, "user") and
        (text(message.tool_call_id).len != 0 or message.tool_call_ids_len > 0);
}
fn contains(ids: *const c.stop_list, id: [*c]const u8) bool {
    if (text(id).len == 0) return false;
    for (items([*c]u8, ids.v, @intCast(ids.len))) |value|
        if (equal(value, text(id))) return true;
    return false;
}
fn clearIds(ids: *c.stop_list) void {
    for (items([*c]u8, ids.v, @intCast(ids.len))) |id| c.free(id);
    ids.len = 0;
    ids.max_len = 0;
}
fn freeIds(ids: *c.stop_list) void {
    clearIds(ids);
    c.free(@ptrCast(ids.v));
    ids.* = std.mem.zeroes(c.stop_list);
}
fn addId(ids: *c.stop_list, id: [*c]const u8) void {
    if (text(id).len == 0 or contains(ids, id)) return;
    var list: std.ArrayList([*c]u8) = .empty;
    list.items = items([*c]u8, ids.v, @intCast(ids.len));
    list.capacity = @intCast(ids.cap);
    list.append(allocator, duplicate(text(id))) catch fatal();
    ids.v = list.items.ptr;
    ids.len = @intCast(list.items.len);
    ids.cap = @intCast(list.capacity);
    ids.max_len = @max(ids.max_len, text(id).len);
}
fn collectIds(message: *const c.chat_msg, ids: *c.stop_list) void {
    addId(ids, message.tool_call_id);
    for (items([*c]u8, message.tool_call_ids, @intCast(message.tool_call_ids_len))) |id| addId(ids, id);
}
fn historyUsesTools(messages: *const c.chat_msgs, schemas: [*c]const u8) bool {
    if (text(schemas).len != 0) return true;
    for (messageItems(messages)) |message| {
        if ((equal(message.role, "assistant") and message.calls.len > 0) or toolRole(message.role)) return true;
    }
    return false;
}
/// Read the real mutable live frontier under its existing mutex. Validation
/// still locks per ID, matching the old acceptance boundary rather than taking
/// an invented snapshot that could outlive another request's slot replacement.
fn liveContains(runtime: ?*c.server, api: c.api_style, id: [*c]const u8) bool {
    const server = runtime orelse return false;
    if (text(id).len == 0) return false;
    _ = c.pthread_mutex_lock(&server.tool_mu);
    defer _ = c.pthread_mutex_unlock(&server.tool_mu);
    for (items(c.server_slot, server.slots, @intCast(server.slot_count))) |*slot| {
        const live = if (api == c.API_RESPONSES) &slot.responses_live else &slot.anthropic_live;
        if (live.valid and contains(&live.call_ids, id)) return true;
    }
    return false;
}
fn liveOrder(runtime: ?*c.server, api: c.api_style, ids: *const c.stop_list) c.stop_list {
    var order = std.mem.zeroes(c.stop_list);
    const server = runtime orelse return order;
    if (ids.len == 0 or (api != c.API_RESPONSES and api != c.API_ANTHROPIC)) return order;
    _ = c.pthread_mutex_lock(&server.tool_mu);
    defer _ = c.pthread_mutex_unlock(&server.tool_mu);
    for (items(c.server_slot, server.slots, @intCast(server.slot_count))) |*slot| {
        const live = if (api == c.API_RESPONSES) &slot.responses_live else &slot.anthropic_live;
        if (!live.valid or live.call_ids.len != ids.len) continue;
        var matches = true;
        for (items([*c]u8, ids.v, @intCast(ids.len))) |id| {
            if (!contains(&live.call_ids, id)) {
                matches = false;
                break;
            }
        }
        if (!matches) continue;
        for (items([*c]u8, live.call_ids.v, @intCast(live.call_ids.len))) |id| addId(&order, id);
        break;
    }
    return order;
}

/// Nearest preceding assistant wins for a repeated call ID. Borrowed keys and
/// message pointers live for this pass; the map never owns client strings.
pub fn validate(runtime: ?*c.server, optional_messages: ?*const c.chat_msgs, api: c.api_style, mode: c.ds4_think_mode, needs_live: ?*bool, needs_reasoning: ?*bool, errors: []u8) bool {
    // Responses historically leaves both outputs untouched for NULL messages;
    // Anthropic initializes its live flag before the same early return.
    if (api == c.API_ANTHROPIC) if (needs_live) |flag| {
        flag.* = false;
    };
    const messages = optional_messages orelse return true;
    if (needs_live) |flag| flag.* = false;
    if (api == c.API_RESPONSES) if (needs_reasoning) |flag| {
        flag.* = false;
    };
    var by_id: std.StringHashMap(*const c.chat_msg) = .init(allocator);
    defer by_id.deinit();
    for (messageItems(messages)) |*message| {
        if (equal(message.role, "assistant")) {
            for (items(c.tool_call, message.calls.v, @intCast(message.calls.len))) |call| {
                if (text(call.id).len != 0) by_id.put(text(call.id), message) catch fatal();
            }
            continue;
        }
        if (if (api == c.API_RESPONSES) !toolRole(message.role) else !anthropicResult(message)) continue;
        var ids = std.mem.zeroes(c.stop_list);
        defer freeIds(&ids);
        collectIds(message, &ids);
        for (items([*c]u8, ids.v, @intCast(ids.len))) |id| {
            const known = liveContains(runtime, api, id);
            const prior = by_id.get(text(id));
            if (!known and prior == null) {
                const format: [*:0]const u8 = if (api == c.API_RESPONSES)
                    "Responses continuation state is not available for call_id %s; retry by replaying the full input history"
                else
                    "Anthropic continuation state is not available for tool_use_id %s; retry by replaying the full messages history";
                _ = c.snprintf(errorPointer(errors), errors.len, format, id);
                return false;
            }
            if (prior == null) {
                if (needs_live) |flag| flag.* = true;
            } else if (api == c.API_RESPONSES and c.ds4_think_mode_enabled(mode) and text(prior.?.reasoning).len == 0) {
                if (needs_reasoning) |flag| flag.* = true;
            }
        }
    }
    return true;
}

fn render(backend: ModelBackend, runtime: ?*c.server, req: *const c.request, messages: *const c.chat_msgs, start: c_int, schemas: [*c]const u8, live: bool, ids: ?*const c.stop_list, context: usize) [*c]u8 {
    if (live) {
        var order = liveOrder(runtime, req.api, ids.?);
        defer freeIds(&order);
        return backend.render(req, messages, @intCast(start), null, true, &order, context);
    }
    return backend.render(req, messages, 0, schemas, false, null, context);
}

/// This only prepares evidence. Generation still rechecks the actual sampled
/// frontier before choosing the live suffix instead of rendered-prefix replay.
pub fn prepareLive(runtime: ?*c.server, optional_request: ?*c.request, optional_messages: ?*const c.chat_msgs) void {
    const req = optional_request orelse return;
    const messages = optional_messages orelse return;
    if (messages.len == 0 or (req.api != c.API_RESPONSES and req.api != c.API_ANTHROPIC)) return;
    const values = messageItems(messages);
    var end = values.len;
    if (req.api == c.API_RESPONSES) {
        while (end > 0 and values[end - 1].discovered_tools != null) end -= 1;
    } else {
        while (end > 0 and systemRole(values[end - 1].role)) end -= 1;
    }
    var start = end;
    while (start > 0 and (if (req.api == c.API_RESPONSES) toolRole(values[start - 1].role) else anthropicResult(&values[start - 1]))) start -= 1;
    if (start == end) return;
    const ids = if (req.api == c.API_RESPONSES) &req.responses_live_call_ids else &req.anthropic_live_call_ids;
    clearIds(ids);
    if (req.api == c.API_RESPONSES and start > 0) {
        const assistant = &values[start - 1];
        if (!equal(assistant.role, "assistant") or assistant.calls.len == 0) return;
        for (items(c.tool_call, assistant.calls.v, @intCast(assistant.calls.len))) |call| addId(ids, call.id);
    } else {
        for (values[start..]) |*message| collectIds(message, ids);
    }
    if (ids.len == 0) return;
    const suffix = if (req.api == c.API_RESPONSES) &req.responses_live_suffix_text else &req.anthropic_live_suffix_text;
    c.free(suffix.*);
    const backend: ModelBackend = .{ .engine = if (runtime) |server| server.engine else null };
    suffix.* = render(backend, runtime, req, messages, @intCast(start), null, true, ids, 0);
}

/// The first matching STRING field is a probe, not a full-object validator.
/// Later duplicates and trailing bytes intentionally do not affect its result.
pub fn stringField(object: [*c]const u8, wanted: [*c]const u8) [*c]u8 {
    var cursor = object;
    c.ds4_server_json_ws(&cursor);
    if (cursor[0] != '{') return null;
    cursor += 1;
    while (cursor[0] != 0) {
        c.ds4_server_json_ws(&cursor);
        if (cursor[0] == '}') break;
        var key: [*c]u8 = null;
        if (!c.ds4_server_json_string(&cursor, &key)) return null;
        defer c.free(key);
        c.ds4_server_json_ws(&cursor);
        if (cursor[0] != ':') return null;
        cursor += 1;
        if (equal(key, text(wanted))) {
            var value: [*c]u8 = null;
            return if (c.ds4_server_json_string(&cursor, &value)) value else null;
        }
        if (!c.ds4_server_json_skip(&cursor)) return null;
        c.ds4_server_json_ws(&cursor);
        if (cursor[0] == ',') cursor += 1;
    }
    return null;
}

/// ResponseStore publishes atomic raw records; this layer owns their existing
/// tolerant envelope parser. LAST response/replay values win, both are required,
/// and the whole envelope must finish before either output is accepted.
pub fn loadResponse(directory: [*c]const u8, id: [*c]const u8, response: *[*c]u8, replay: *[*c]u8, errors: []u8) c_int {
    var record: [*c]u8 = null;
    var length: usize = 0;
    const status = c.ds4_response_store_read(directory, id, &record, &length, errorPointer(errors), errors.len);
    if (status != 0) return status;
    defer c.free(record);
    var cursor: [*c]const u8 = record;
    c.ds4_server_json_ws(&cursor);
    var valid = cursor[0] == '{';
    if (valid) cursor += 1;
    while (valid) {
        c.ds4_server_json_ws(&cursor);
        if (cursor[0] == '}') {
            cursor += 1;
            break;
        }
        var key: [*c]u8 = null;
        valid = c.ds4_server_json_string(&cursor, &key);
        defer c.free(key);
        c.ds4_server_json_ws(&cursor);
        if (!valid or cursor[0] != ':') {
            valid = false;
            break;
        }
        cursor += 1;
        if (equal(key, "response") or equal(key, "replay")) {
            var value: [*c]u8 = null;
            valid = c.ds4_server_json_raw(&cursor, &value);
            if (valid) {
                const target = if (equal(key, "response")) response else replay;
                c.free(target.*);
                target.* = value;
            }
        } else valid = c.ds4_server_json_skip(&cursor);
        c.ds4_server_json_ws(&cursor);
        if (cursor[0] == ',') cursor += 1;
    }
    c.ds4_server_json_ws(&cursor);
    if (valid and cursor[0] == 0 and response.* != null and replay.* != null) return 0;
    c.free(take(response));
    c.free(take(replay));
    report(errors, "invalid stored response record");
    return -1;
}

const Transaction = struct {
    backend: ModelBackend,
    runtime: ?*c.server,
    input: *c.ds4_server_input,
    req: c.request,
    messages: c.chat_msgs,
    schemas: [*c]u8,
    instructions: [*c]u8,
    loaded_schemas: [*c]u8,
    context: c_int,
    errors: []u8,

    fn begin(engine: ?*c.ds4_engine, runtime: ?*c.server, input: *c.ds4_server_input, context: c_int, errors: []u8) Transaction {
        const req = input.req;
        input.req = std.mem.zeroes(c.request);
        const messages = input.messages;
        input.messages = std.mem.zeroes(c.chat_msgs);
        return .{ .backend = .{ .engine = engine }, .runtime = runtime, .input = input, .req = req, .messages = messages, .schemas = take(&input.tool_schemas), .instructions = take(&input.instructions), .loaded_schemas = take(&input.loaded_tool_schemas), .context = context, .errors = errors };
    }
    fn deinit(self: *Transaction) void {
        c.ds4_server_input_request_free(&self.req);
        c.ds4_server_input_messages_free(&self.messages);
        c.free(self.schemas);
        c.free(self.instructions);
        c.free(self.loaded_schemas);
    }
    fn attachTools(self: *Transaction) void {
        c.ds4_server_materialize_restore_tools(self.runtime, &self.messages);
        if (self.runtime) |server| {
            if (!server.disable_exact_dsml_tool_replay) {
                c.ds4_server_tool_memory_attach(&server.tool_mem, &server.tool_mu, &self.messages, &self.req.tool_replay);
                return;
            }
        }
        for (messageItems(&self.messages)) |message| {
            if (message.calls.len == 0 or message.calls.raw_tool_text != null) continue;
            self.req.tool_replay.canonical += 1;
            self.req.tool_replay.missing_ids += message.calls.len;
        }
    }
    fn finishPrompt(self: *Transaction, schemas: [*c]const u8) Invalid!void {
        self.req.prompt_preserves_reasoning = historyUsesTools(&self.messages, schemas);
        self.req.prompt_text = render(self.backend, self.runtime, &self.req, &self.messages, 0, schemas, false, null, @intCast(@max(self.context, 0)));
        if (!self.backend.tokenize(self.runtime, &self.req, &self.messages, self.errors)) return error.InvalidRequest;
    }
    fn conversation(self: *Transaction, incremental: bool) Invalid!void {
        var record = std.mem.zeroes(c.ds4_conversation);
        if (c.ds4_conversation_load(self.req.responses_store_dir, self.req.conversation_id, &record, errorPointer(self.errors), self.errors.len) != 0) return error.InvalidRequest;
        defer c.ds4_conversation_free(&record);
        var normalized: [*c]u8 = null;
        defer c.free(normalized);
        const accepted = self.req.conversation_input != null and
            c.ds4_conversation_input_items(self.req.conversation_input, &normalized, errorPointer(self.errors), self.errors.len);
        var history = std.mem.zeroes(c.chat_msgs);
        defer c.ds4_server_input_messages_free(&history);
        var cursor: [*c]const u8 = if (record.replay != null) record.replay else record.items;
        const parsed = !record.deleted and accepted and (if (record.replay != null)
            c.ds4_server_json_messages(&cursor, &history, true)
        else
            c.ds4_server_json_responses_input(&cursor, &history, &self.loaded_schemas, &self.req.tool_orders, incremental));
        c.ds4_server_json_ws(&cursor);
        self.req.conversation_version = record.version;
        if (!parsed or cursor[0] != 0) {
            if (self.errors.len != 0 and self.errors[0] == 0) report(self.errors, "conversation has invalid or deleted history");
            return error.InvalidRequest;
        }
        c.free(self.req.conversation_input);
        self.req.conversation_input = take(&normalized);
        prependHistory(&self.messages, &history);
        if (self.req.prompt_cache_key == null) self.req.prompt_cache_key = duplicate(text(self.req.conversation_id));
    }
    fn previousResponse(self: *Transaction) Invalid!void {
        var response: [*c]u8 = null;
        var replay: [*c]u8 = null;
        defer c.free(response);
        defer c.free(replay);
        var status: c_int = 1;
        if (self.req.response_sink != null) {
            const sink = &self.req.response_sink[0];
            if (sink.load_parent) |load| status = load(sink.context, self.req.previous_response_id, &response, &replay);
        }
        if (status == 1 and self.req.responses_store_dir != null and (self.req.response_sink == null or self.req.responses_store))
            status = loadResponse(self.req.responses_store_dir, self.req.previous_response_id, &response, &replay, self.errors);
        if (status != 0) {
            if (status == 1) report(self.errors, "previous_response_id was not found");
            return error.InvalidRequest;
        }
        const status_text = stringField(response, "status");
        defer c.free(status_text);
        var parent = std.mem.zeroes(c.chat_msgs);
        defer c.ds4_server_input_messages_free(&parent);
        var cursor: [*c]const u8 = replay;
        const parsed = equal(status_text, "completed") and c.ds4_server_json_messages(&cursor, &parent, true);
        c.ds4_server_json_ws(&cursor);
        if (!parsed or cursor[0] != 0) return fail(self.errors, "previous response is incomplete or has invalid replay history");
        prependHistory(&self.messages, &parent);
    }
    fn chat(self: *Transaction) Invalid!void {
        if (!self.input.got_input) return fail(self.errors, "missing messages");
        if (self.req.ignore_eos and !(self.req.temperature_set and self.req.temperature == 0))
            return fail(self.errors, "ignore_eos requires an explicit temperature of 0");
        self.req.has_tools = text(self.schemas).len != 0 and !self.input.tool_choice_none;
        self.backend.thinking(self.input, &self.req, self.context);
        self.attachTools();
        try self.finishPrompt(if (self.req.has_tools) self.schemas else null);
    }
    fn anthropic(self: *Transaction) Invalid!void {
        if (!self.input.got_input) return fail(self.errors, "missing messages");
        if (text(self.instructions).len != 0)
            prependSystem(&self.messages, take(&self.instructions), self.req.model_syntax == c.SERVER_MODEL_SYNTAX_DEEPSEEK41);
        self.req.has_tools = text(self.schemas).len != 0 and !self.input.tool_choice_none;
        self.backend.thinking(self.input, &self.req, self.context);
        if (!validate(self.runtime, &self.messages, c.API_ANTHROPIC, self.req.think_mode, &self.req.anthropic_requires_live_tool_state, null, self.errors)) return error.InvalidRequest;
        self.attachTools();
        prepareLive(self.runtime, &self.req, &self.messages);
        try self.finishPrompt(if (self.req.has_tools) self.schemas else null);
    }
    fn responses(self: *Transaction, incremental: bool) Invalid!void {
        const disk_parent = self.req.previous_response_id != null and self.req.response_sink == null;
        if ((self.req.responses_store or disk_parent or self.req.conversation_id != null) and self.req.responses_store_dir == null)
            return fail(self.errors, "response storage requires DS4_SERVER_RESPONSE_STORE_DIR");
        if (self.req.conversation_id != null and self.req.previous_response_id != null)
            return fail(self.errors, "conversation and previous_response_id are mutually exclusive");
        if (self.req.conversation_id != null) try self.conversation(incremental);
        if (self.req.previous_response_id != null) try self.previousResponse();
        if (!self.input.got_input) return fail(self.errors, "missing input");

        // Top-level instructions are synthetic, precede durable history, and
        // are excluded again from the privately persisted logical replay.
        const synthetic = text(self.instructions).len != 0;
        if (synthetic) prependSystem(&self.messages, take(&self.instructions), true);
        const discovered = c.ds4_server_input_restore_discovered_tools(&self.messages, &self.req.tool_orders, self.input.tool_choice_none, self.input.explicit_tools);
        if (discovered < 0) return fail(self.errors, "invalid discovered tools in response history");
        var schemas: std.ArrayList(u8) = .empty;
        defer schemas.deinit(allocator);
        schemas.appendSlice(allocator, text(self.schemas)) catch fatal();
        if (text(self.loaded_schemas).len != 0) {
            if (schemas.items.len != 0) schemas.append(allocator, '\n') catch fatal();
            schemas.appendSlice(allocator, text(self.loaded_schemas)) catch fatal();
        }
        if (schemas.items.len != 0) {
            schemas.ensureUnusedCapacity(allocator, 1) catch fatal();
            schemas.items.ptr[schemas.items.len] = 0;
        }
        const active: [*c]const u8 = if (!self.input.tool_choice_none and schemas.items.len != 0) schemas.items.ptr else null;
        self.req.has_tools = !self.input.tool_choice_none and (text(active).len != 0 or discovered != 0);
        self.backend.thinking(self.input, &self.req, self.context);
        if (self.req.reasoning_budget != 0 and (self.backend.engine == null or !self.backend.model().capabilities().reasoning_budget))
            return fail(self.errors, "ds4_reasoning_budget currently supports native DS4.1 text only");
        if (!validate(self.runtime, &self.messages, c.API_RESPONSES, self.req.think_mode, &self.req.responses_requires_live_tool_state, &self.req.responses_requires_live_reasoning, self.errors)) return error.InvalidRequest;
        self.attachTools();
        if (self.req.responses_store or self.req.conversation_id != null or self.req.response_sink != null) {
            for (messageItems(&self.messages)) |message| {
                if (message.images.len != 0 and (self.req.responses_store or self.req.conversation_id != null)) return fail(self.errors, "stored responses currently support text and function tools");
            }
            self.req.responses_replay = c.ds4_server_render_replay_history(&self.messages, if (synthetic) 1 else 0);
        }
        // Durable parents supply complete replay. A call-ID-only append would
        // ignore current instructions/tools, so use rendered-prefix matching.
        if (self.req.previous_response_id == null and self.req.conversation_id == null)
            prepareLive(self.runtime, &self.req, &self.messages);
        try self.finishPrompt(active);
        if (self.req.reasoning_budget != 0 and self.req.image_count != 0)
            return fail(self.errors, "ds4_reasoning_budget currently supports native DS4.1 text only");
    }
    fn completion(self: *Transaction) Invalid!void {
        const prompt = take(&self.input.completion_prompt);
        if (prompt == null) return fail(self.errors, "missing prompt");
        self.backend.thinking(self.input, &self.req, self.context);
        prependSystem(&self.messages, duplicate("You are a helpful assistant"), false);
        var user = std.mem.zeroes(c.chat_msg);
        user.role = duplicate("user");
        user.content = prompt;
        appendMessage(&self.messages, user);
        self.req.prompt_text = render(self.backend, self.runtime, &self.req, &self.messages, 0, null, false, null, @intCast(@max(self.context, 0)));
        self.backend.tokenizeText(self.req.prompt_text, &self.req.prompt);
    }
};

pub fn materialize(engine: ?*c.ds4_engine, runtime: ?*c.server, endpoint: c.ds4_server_input_endpoint, input: *c.ds4_server_input, context: c_int, incremental: bool, out: *c.request, errors: []u8) bool {
    out.* = std.mem.zeroes(c.request);
    var transaction = Transaction.begin(engine, runtime, input, context, errors);
    defer transaction.deinit();
    const result = switch (endpoint) {
        c.DS4_SERVER_INPUT_CHAT => transaction.chat(),
        c.DS4_SERVER_INPUT_ANTHROPIC => transaction.anthropic(),
        c.DS4_SERVER_INPUT_RESPONSES => transaction.responses(incremental),
        c.DS4_SERVER_INPUT_COMPLETION => transaction.completion(),
        else => error.InvalidRequest,
    };
    result catch return false;
    out.* = transaction.req;
    transaction.req = std.mem.zeroes(c.request);
    return true;
}
