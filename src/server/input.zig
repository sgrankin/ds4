//! Wire-request normalization, shared by the four local API endpoints.
//! Native code owns endpoint fields, duplicate/key-order policy and lifetime;
//! the shared JSON/value codecs retain their protected acceptance rules.
//! Durable history, live tool bindings, rendering and tokenization follow once
//! in the C materializer. No engine/session/server-layout imports or callbacks.
const std = @import("std");
const c = @import("server_input_c");
const MessageValues = @import("message_values.zig");
const allocator = std.heap.c_allocator;
const Record = c.ds4_server_input;
const Request = c.request;
const Messages = c.chat_msgs;
const Endpoint = c.ds4_server_input_endpoint;
const Failure = error{ InvalidJson, Policy };

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn string(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn equal(pointer: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, string(pointer), expected);
}
fn duplicate(bytes: []const u8) [*c]u8 {
    const owned = allocator.dupeSentinel(u8, bytes, 0) catch fatal();
    return owned.ptr;
}
fn message(error_text: []u8, value: []const u8) void {
    if (error_text.len == 0) return;
    const length = @min(error_text.len - 1, value.len);
    @memcpy(error_text[0..length], value[0..length]);
    error_text[length] = 0;
}
fn require(success: bool) Failure!void {
    if (!success) return error.InvalidJson;
}

/// The cursor directly borrows the NUL-terminated body. The shared parser library
/// creates libc-owned strings; no DOM, second endpoint parse or SDK dump.
const Reader = struct {
    cursor: [*c]const u8,

    fn peek(self: *Reader) u8 {
        c.ds4_server_json_ws(&self.cursor);
        return self.cursor[0];
    }
    fn consume(self: *Reader, byte: u8) Failure!void {
        if (self.peek() != byte) return error.InvalidJson;
        self.cursor += 1;
    }
    fn literal(self: *Reader, value: [*:0]const u8) bool {
        c.ds4_server_json_ws(&self.cursor);
        return c.ds4_server_json_lit(&self.cursor, value);
    }
    fn skip(self: *Reader) Failure!void {
        try require(c.ds4_server_json_skip(&self.cursor));
    }
    fn readString(self: *Reader) Failure![*c]u8 {
        var result: [*c]u8 = null;
        try require(c.ds4_server_json_string(&self.cursor, &result));
        return result;
    }
    fn readRaw(self: *Reader) Failure![*c]u8 {
        var result: [*c]u8 = null;
        try require(c.ds4_server_json_raw(&self.cursor, &result));
        return result;
    }
    fn replaceString(self: *Reader, target: *[*c]u8) Failure!void {
        const replacement = try self.readString();
        c.free(target.*);
        target.* = replacement;
    }
    fn readBool(self: *Reader, target: *bool) Failure!void {
        try require(c.ds4_server_json_bool(&self.cursor, target));
    }
    fn readInt(self: *Reader, target: *c_int) Failure!void {
        try require(c.ds4_server_json_int(&self.cursor, target));
    }
    fn readNumber(self: *Reader) Failure!f64 {
        var result: f64 = 0;
        try require(c.ds4_server_json_number(&self.cursor, &result));
        return result;
    }
    fn nextField(self: *Reader) Failure!?[*c]u8 {
        const byte = self.peek();
        if (byte == '}') return null;
        if (byte == 0) return error.InvalidJson;
        const key = try self.readString();
        errdefer c.free(key);
        try self.consume(':');
        return key;
    }
    fn separator(self: *Reader) void {
        // Optional commas match the existing selective cursor parser.
        if (self.peek() == ',') self.cursor += 1;
        _ = self.peek();
    }
};

// Empty public C vectors may have a NULL base. Form a native slice only
// after the count proves that storage exists; partial decode cleanup is valid.
fn ownedItems(comptime T: type, pointer: [*c]T, count: usize) []T {
    // clearStops retains allocation/capacity even when its length becomes zero.
    return if (count == 0 and pointer == null) &.{} else pointer[0..count];
}

fn freeStrings(values: *c.stop_list) void {
    for (ownedItems([*c]u8, values.v, @intCast(values.len))) |value| c.free(value);
    c.free(@ptrCast(values.v));
    values.* = std.mem.zeroes(c.stop_list);
}
fn freeOrders(orders: *c.tool_schema_orders) void {
    for (ownedItems(c.tool_schema_order, orders.v, @intCast(orders.len))) |*order| {
        c.free(order.name);
        c.free(order.wire_name);
        c.free(order.namespace);
        for (ownedItems([*c]u8, order.prop, @intCast(order.len))) |property| c.free(property);
        for (ownedItems([*c]u8, order.prop_schema, @intCast(order.len))) |schema| c.free(schema);
        c.free(@ptrCast(order.prop));
        c.free(@ptrCast(order.prop_schema));
    }
    c.free(orders.v);
    orders.* = std.mem.zeroes(c.tool_schema_orders);
}
pub fn freeMessages(messages: *Messages) void {
    for (ownedItems(c.chat_msg, messages.v, @intCast(messages.len))) |*item|
        MessageValues.freeMessage(item);
    c.free(messages.v);
    messages.* = std.mem.zeroes(Messages);
}

pub fn freeRequest(request: *Request) void {
    c.ds4_tokens_free(&request.prompt);
    for (ownedItems(c.ds4_vision_span, request.images, request.image_count)) |*image| c.ds4_vision_embedding_free(&image.embedding);
    c.free(request.images);
    c.free(request.image_markers);
    inline for (.{ "model", "raw_body", "prompt_text", "prompt_cache_key", "previous_response_id", "responses_replay", "conversation_id", "conversation_input", "responses_live_suffix_text", "anthropic_live_suffix_text" }) |field|
        c.free(@field(request, field));
    freeStrings(&request.stops);
    freeStrings(&request.responses_live_call_ids);
    freeStrings(&request.anthropic_live_call_ids);
    freeOrders(&request.tool_orders);
    request.* = std.mem.zeroes(Request);
}
fn pushMessage(messages: *Messages, item: c.chat_msg) void {
    var list: std.ArrayList(c.chat_msg) = .empty;
    if (messages.v != null) {
        list.items = ownedItems(c.chat_msg, messages.v, @intCast(messages.len));
        list.capacity = @intCast(messages.cap);
    }
    list.append(allocator, item) catch fatal();
    messages.v = list.items.ptr;
    messages.len = @intCast(list.items.len);
    messages.cap = @intCast(list.capacity);
}
fn clearStops(stops: *c.stop_list) void {
    for (ownedItems([*c]u8, stops.v, @intCast(stops.len))) |value| c.free(value);
    stops.len = 0;
    stops.max_len = 0;
}
fn pushStop(stops: *c.stop_list, value: [*c]u8) void {
    const bytes = string(value);
    if (bytes.len == 0) {
        c.free(value);
        return;
    }
    var list: std.ArrayList([*c]u8) = .empty;
    if (stops.v != null) {
        list.items = ownedItems([*c]u8, stops.v, @intCast(stops.len));
        list.capacity = @intCast(stops.cap);
    }
    list.append(allocator, value) catch fatal();
    stops.v = list.items.ptr;
    stops.len = @intCast(list.items.len);
    stops.cap = @intCast(list.capacity);
    stops.max_len = @max(stops.max_len, bytes.len);
}
fn stopsValue(reader: *Reader, stops: *c.stop_list) Failure!void {
    clearStops(stops);
    if (reader.peek() == '"') {
        pushStop(stops, try reader.readString());
        return;
    }
    if (reader.peek() != '[') return reader.skip();
    try reader.consume('[');
    while (reader.peek() != 0 and reader.peek() != ']') {
        if (reader.peek() == '"') pushStop(stops, try reader.readString()) else try reader.skip();
        reader.separator();
    }
    try reader.consume(']');
}
fn effortValue(reader: *Reader, effort: *c.ds4_think_mode) Failure!void {
    if (reader.literal("null")) return;
    const name = try reader.readString();
    defer c.free(name);
    const named: ?c.ds4_think_mode = if (equal(name, "max")) c.DS4_THINK_MAX else if (equal(name, "xhigh") or equal(name, "high")) c.DS4_THINK_HIGH else if (equal(name, "medium")) c.DS4_THINK_MEDIUM else if (equal(name, "low") or equal(name, "minimal")) c.DS4_THINK_LOW else if (equal(name, "none")) c.DS4_THINK_NONE else null;
    // minimal is the smallest nonzero level; none disables reasoning.
    if (named) |value| effort.* = value else try require(c.ds4_think_mode_parse_level(name, effort));
}
fn thinkingValue(reader: *Reader, enabled: *bool) Failure!void {
    if (reader.literal("null")) return;
    if (reader.peek() == 't' or reader.peek() == 'f') return reader.readBool(enabled);
    if (reader.peek() != '{') return reader.skip();
    try reader.consume('{');
    while (try reader.nextField()) |key| {
        defer c.free(key);
        if (equal(key, "type")) {
            const kind = try reader.readString();
            defer c.free(kind);
            if (equal(kind, "enabled")) enabled.* = true else if (equal(kind, "disabled")) enabled.* = false;
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
}
fn nestedControls(reader: *Reader, input: *Record, template_kwargs: bool) Failure!void {
    if (reader.literal("null")) return;
    if (reader.peek() != '{') return reader.skip();
    try reader.consume('{');
    while (try reader.nextField()) |key| {
        defer c.free(key);
        if (template_kwargs and equal(key, "enable_thinking")) {
            try reader.readBool(&input.thinking_enabled);
            input.got_thinking = true;
        } else if (equal(key, if (template_kwargs) "reasoning_effort" else "effort")) {
            try effortValue(reader, &input.reasoning_effort);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
}
fn streamOptions(reader: *Reader, request: *Request) Failure!void {
    if (reader.peek() != '{') return reader.skip();
    try reader.consume('{');
    while (try reader.nextField()) |key| {
        defer c.free(key);
        if (equal(key, "include_usage")) try reader.readBool(&request.stream_include_usage) else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
}
fn responsesReasoning(reader: *Reader, input: *Record) Failure!void {
    if (reader.literal("null")) return;
    if (reader.peek() != '{') return reader.skip();
    try reader.consume('{');
    var effort_seen = false;
    while (try reader.nextField()) |key| {
        defer c.free(key);
        if (equal(key, "effort")) {
            // Explicit null has the same effect as an omitted effort. Summary
            // alone must not defeat model-alias thinking defaults.
            if (!reader.literal("null")) {
                try effortValue(reader, &input.reasoning_effort);
                effort_seen = true;
            }
        } else if (equal(key, "summary")) {
            if (reader.literal("null")) {
                input.req.reasoning_summary_emit = false;
            } else {
                if (reader.peek() == '"') {
                    const mode = try reader.readString();
                    defer c.free(mode);
                    if (equal(mode, "auto") or equal(mode, "concise") or equal(mode, "detailed")) input.req.reasoning_summary_emit = true;
                } else try reader.skip();
            }
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
    if (effort_seen) {
        input.got_thinking = true;
        if (input.reasoning_effort == c.DS4_THINK_NONE) input.thinking_enabled = false;
    }
}
fn toolChoice(reader: *Reader, endpoint: Endpoint, input: *Record, error_text: []u8) Failure!void {
    if (endpoint == c.DS4_SERVER_INPUT_ANTHROPIC) {
        if (reader.peek() != '{') return reader.skip();
        try reader.consume('{');
        while (try reader.nextField()) |key| {
            defer c.free(key);
            if (equal(key, "type")) {
                const choice = try reader.readString();
                defer c.free(choice);
                input.tool_choice_none = equal(choice, "none");
            } else try reader.skip();
            reader.separator();
        }
        return reader.consume('}');
    }
    if (reader.peek() == '"') {
        const choice = try reader.readString();
        defer c.free(choice);
        if (endpoint != c.DS4_SERVER_INPUT_RESPONSES) {
            input.tool_choice_none = equal(choice, "none");
        } else if (equal(choice, "none")) {
            input.tool_choice_none = true;
        } else if (!equal(choice, "auto")) {
            if (error_text.len != 0) _ = c.snprintf(error_text.ptr, error_text.len, "tool_choice=%s not supported", choice);
            return error.Policy;
        }
    } else if (endpoint == c.DS4_SERVER_INPUT_RESPONSES and reader.peek() == '{') {
        // Forced/required targets require constrained decoding. Report this
        // supported-subset limit instead of silently downgrading to auto.
        message(error_text, "forced tool_choice not supported");
        return error.Policy;
    } else try reader.skip();
}
fn objectString(raw: [*c]const u8, wanted: []const u8) [*c]u8 {
    var reader = Reader{ .cursor = raw };
    reader.consume('{') catch return null;
    while (reader.nextField() catch return null) |key| {
        defer c.free(key);
        if (equal(key, wanted)) return reader.readString() catch null;
        reader.skip() catch return null;
        reader.separator();
    }
    return null;
}

pub const Input = struct {
    value: Record,

    pub fn deinit(self: *Input) void {
        freeMessages(&self.value.messages);
        freeRequest(&self.value.req);
        inline for (.{ "tool_schemas", "loaded_tool_schemas", "explicit_tools", "instructions", "completion_prompt" }) |field|
            c.free(@field(self.value, field));
        self.value = std.mem.zeroes(Record);
    }

    pub fn decode(endpoint: Endpoint, body: [:0]const u8, defaults: *const c.ds4_server_input_defaults, error_text: []u8) Failure!Input {
        var result = Input{ .value = std.mem.zeroes(Record) };
        errdefer result.deinit();
        const input = &result.value;
        const request = &input.req;
        request.kind = if (endpoint == c.DS4_SERVER_INPUT_COMPLETION) c.REQ_COMPLETION else c.REQ_CHAT;
        request.api = switch (endpoint) {
            c.DS4_SERVER_INPUT_ANTHROPIC => c.API_ANTHROPIC,
            c.DS4_SERVER_INPUT_RESPONSES => c.API_RESPONSES,
            else => c.API_OPENAI,
        };
        request.model_syntax = defaults.syntax;
        request.model = duplicate("deepseek-v4-flash");
        request.max_tokens = defaults.max_tokens;
        request.temperature = c.DS4_DEFAULT_TEMPERATURE;
        request.top_p = c.DS4_DEFAULT_TOP_P;
        request.min_p = c.DS4_DEFAULT_MIN_P;
        if (defaults.syntax == c.SERVER_MODEL_SYNTAX_FLINT) {
            // Match the model's reference harness; explicit fields decoded
            // below retain their own flags, including zero-valued controls.
            request.temperature = 1;
            request.top_k = 20;
            request.top_p = 0.95;
            request.min_p = 0;
            // Flint chooses when to think; expose that output without making
            // clients opt into a separate summary or effort control.
            request.reasoning_summary_emit = endpoint == c.DS4_SERVER_INPUT_RESPONSES;
        }
        request.think_mode = c.DS4_THINK_HIGH;
        input.thinking_enabled = true;
        input.reasoning_effort = c.DS4_THINK_HIGH;
        if (endpoint == c.DS4_SERVER_INPUT_RESPONSES) {
            if (error_text.len != 0) error_text[0] = 0;
            request.responses_store_dir = defaults.responses_store_dir;
            request.responses_store = defaults.responses_store_dir != null;
        }
        var reader = Reader{ .cursor = body.ptr };
        try reader.consume('{');
        while (try reader.nextField()) |key| {
            defer c.free(key);
            try result.decodeField(&reader, endpoint, string(key), defaults, error_text);
            reader.separator();
        }
        // The old API decoder stops at the closing brace. It does not impose
        // end-of-document validation or change partial malformed-input cursors.
        if (reader.peek() != '}') return error.InvalidJson;
        if (defaults.syntax == c.SERVER_MODEL_SYNTAX_FLINT) {
            // Match Flint's reference server: accept unsupported reasoning
            // hints without letting them change its model-controlled policy.
            input.got_thinking = false;
            input.thinking_enabled = true;
            input.reasoning_effort = c.DS4_THINK_HIGH;
        }
        return result;
    }

    fn decodeField(self: *Input, reader: *Reader, endpoint: Endpoint, key: []const u8, defaults: *const c.ds4_server_input_defaults, error_text: []u8) Failure!void {
        const input = &self.value;
        const request = &input.req;
        const responses = endpoint == c.DS4_SERVER_INPUT_RESPONSES;
        const anthropic = endpoint == c.DS4_SERVER_INPUT_ANTHROPIC;
        const chat = endpoint == c.DS4_SERVER_INPUT_CHAT;
        const completion = endpoint == c.DS4_SERVER_INPUT_COMPLETION;
        if ((chat or anthropic) and std.mem.eql(u8, key, "messages")) {
            freeMessages(&input.messages);
            try require(if (anthropic) c.ds4_server_json_anthropic_messages(&reader.cursor, &input.messages) else c.ds4_server_json_messages(&reader.cursor, &input.messages, false));
            input.got_input = true;
        } else if (responses and std.mem.eql(u8, key, "input")) {
            freeMessages(&input.messages);
            _ = reader.peek();
            const start = reader.cursor;
            if (reader.peek() == '"') {
                const plain = try reader.readString();
                var item = std.mem.zeroes(c.chat_msg);
                item.role = duplicate("user");
                item.content = plain;
                pushMessage(&input.messages, item);
            } else try require(c.ds4_server_json_responses_input(&reader.cursor, &input.messages, &input.loaded_tool_schemas, &request.tool_orders, defaults.incremental_tool_schemas));
            c.free(request.conversation_input);
            request.conversation_input = duplicate(start[0 .. @intFromPtr(reader.cursor) - @intFromPtr(start)]);
            input.got_input = true;
        } else if (completion and std.mem.eql(u8, key, "prompt")) {
            var prompt: [*c]u8 = null;
            errdefer c.free(prompt);
            try require(c.ds4_server_json_prompt(&reader.cursor, &prompt));
            c.free(input.completion_prompt);
            input.completion_prompt = prompt;
        } else if (anthropic and std.mem.eql(u8, key, "system")) {
            var system: [*c]u8 = null;
            try require(c.ds4_server_json_anthropic_system(&reader.cursor, &system));
            c.free(input.instructions);
            input.instructions = system;
        } else if (responses and std.mem.eql(u8, key, "instructions")) {
            c.free(input.instructions);
            input.instructions = null;
            input.instructions = if (reader.literal("null")) duplicate("") else try reader.readString();
        } else if (!completion and std.mem.eql(u8, key, "tools")) {
            c.free(input.tool_schemas);
            input.tool_schemas = null;
            const start = reader.cursor;
            try require(c.ds4_server_json_tools(&reader.cursor, &input.tool_schemas, &request.tool_orders));
            if (responses) {
                c.free(input.explicit_tools);
                input.explicit_tools = duplicate(start[0 .. @intFromPtr(reader.cursor) - @intFromPtr(start)]);
            }
        } else if (!completion and std.mem.eql(u8, key, "tool_choice")) {
            try toolChoice(reader, endpoint, input, error_text);
        } else if (std.mem.eql(u8, key, "model")) {
            try reader.replaceString(&request.model);
            request.model_from_request = true;
        } else if (std.mem.eql(u8, key, "max_tokens") or (chat and std.mem.eql(u8, key, "max_completion_tokens")) or (responses and std.mem.eql(u8, key, "max_output_tokens"))) {
            try reader.readInt(&request.max_tokens);
        } else if (std.mem.eql(u8, key, "temperature")) {
            request.temperature = @floatCast(try reader.readNumber());
            request.temperature_set = true;
        } else if (std.mem.eql(u8, key, "top_p")) {
            request.top_p = @floatCast(try reader.readNumber());
            request.top_p_set = true;
        } else if ((chat or completion or defaults.syntax == c.SERVER_MODEL_SYNTAX_FLINT) and std.mem.eql(u8, key, "min_p")) {
            request.min_p = @floatCast(try reader.readNumber());
            request.min_p_set = true;
        } else if ((!responses or defaults.syntax == c.SERVER_MODEL_SYNTAX_FLINT) and std.mem.eql(u8, key, "top_k")) {
            try reader.readInt(&request.top_k);
            request.top_k_set = true;
        } else if ((chat or completion) and std.mem.eql(u8, key, "seed")) {
            const value = try reader.readNumber();
            request.seed = if (value > 0) @intFromFloat(value) else 0;
        } else if (std.mem.eql(u8, key, "stream")) {
            try reader.readBool(&request.stream);
        } else if ((chat or completion) and std.mem.eql(u8, key, "stream_options")) {
            try streamOptions(reader, request);
        } else if (chat and std.mem.eql(u8, key, "ignore_eos")) {
            try reader.readBool(&request.ignore_eos);
        } else if (!responses and std.mem.eql(u8, key, "thinking")) {
            try thinkingValue(reader, &input.thinking_enabled);
            input.got_thinking = true;
        } else if (!responses and std.mem.eql(u8, key, "reasoning_effort")) {
            try effortValue(reader, &input.reasoning_effort);
            if (defaults.syntax == c.SERVER_MODEL_SYNTAX_FLINT) input.got_thinking = true;
        } else if (chat and std.mem.eql(u8, key, "chat_template_kwargs")) {
            // Qwen's documented enable_thinking/reasoning_effort controls;
            // unknown template keys remain ignored extensions.
            try nestedControls(reader, input, true);
        } else if ((chat or completion) and std.mem.eql(u8, key, "think")) {
            try reader.readBool(&input.thinking_enabled);
            input.got_thinking = true;
        } else if (anthropic and std.mem.eql(u8, key, "output_config")) {
            try nestedControls(reader, input, false);
        } else if (((chat or completion) and std.mem.eql(u8, key, "stop")) or (anthropic and std.mem.eql(u8, key, "stop_sequences"))) {
            try stopsValue(reader, &request.stops);
        } else if (responses and std.mem.eql(u8, key, "reasoning")) {
            try responsesReasoning(reader, input);
        } else if ((chat or responses) and std.mem.eql(u8, key, "prompt_cache_key")) {
            try reader.replaceString(&request.prompt_cache_key);
        } else if (responses and std.mem.eql(u8, key, "store")) {
            try reader.readBool(&request.responses_store);
        } else if (responses and std.mem.eql(u8, key, "previous_response_id")) {
            if (!reader.literal("null")) try reader.replaceString(&request.previous_response_id);
        } else if (responses and std.mem.eql(u8, key, "conversation")) {
            if (!reader.literal("null")) {
                if (reader.peek() == '"') {
                    try reader.replaceString(&request.conversation_id);
                } else {
                    const raw = try reader.readRaw();
                    defer c.free(raw);
                    c.free(request.conversation_id);
                    request.conversation_id = objectString(raw, "id");
                    if (request.conversation_id == null) return error.InvalidJson;
                }
            }
        } else if (responses and std.mem.eql(u8, key, "ds4_reasoning_budget")) {
            const value = reader.readNumber() catch {
                message(error_text, "ds4_reasoning_budget must be an integer from 0 to 65536");
                return error.Policy;
            };
            if (!std.math.isFinite(value) or value < 0 or value > 65536 or value != @trunc(value)) {
                message(error_text, "ds4_reasoning_budget must be an integer from 0 to 65536");
                return error.Policy;
            }
            request.reasoning_budget = @intFromFloat(value);
        } else try reader.skip();
    }
};

pub fn ds4_server_input_decode(endpoint: Endpoint, body: [*c]const u8, defaults: *const c.ds4_server_input_defaults, out: *Record, error_text: [*c]u8, error_length: usize) bool {
    out.* = std.mem.zeroes(Record);
    const errors: []u8 = if (error_length == 0) &.{} else error_text[0..error_length];
    const input = Input.decode(endpoint, std.mem.span(body), defaults, errors) catch |failure| {
        if (failure == error.InvalidJson) message(errors, "invalid JSON request");
        return false;
    };
    out.* = input.value;
    return true;
}
pub fn ds4_server_input_deinit(record: *Record) void {
    var input = Input{ .value = record.* };
    input.deinit();
    record.* = input.value;
}
pub fn ds4_server_input_messages_free(messages: *Messages) void {
    freeMessages(messages);
}
pub fn ds4_server_input_request_free(request: *Request) void {
    freeRequest(request);
}
