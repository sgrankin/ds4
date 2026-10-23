//! Responses stream/nonstream lifecycle over the actual shared request/state.
//! The generation worker supplies parsed content/tool calls and owns sockets.
//! Native output owns framing, item identity/order/status and text visibility;
//! the existing durable publisher and tolerant argument codec remain libraries.
const std = @import("std");
const c = @import("server_output_c");
const allocator = std.heap.c_allocator;
pub const Stream = c.responses_stream;
pub const Item = c.responses_tool_item;
const Request = c.request;
const Stop = *const volatile c.sig_atomic_t;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn callSlice(calls: ?*const c.tool_calls) []const c.tool_call {
    const value = calls orelse return &.{};
    return if (value.len == 0) &.{} else @as([*]const c.tool_call, @ptrCast(value.v))[0..@intCast(value.len)];
}

const Text = struct {
    list: std.ArrayList(u8) = .empty,
    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, bytes: []const u8) void {
        self.list.appendSlice(allocator, bytes) catch fatal();
    }
    fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    fn number(self: *Text, value: anytype) void {
        self.list.print(allocator, "{d}", .{value}) catch fatal();
    }
    fn quoted(self: *Text, bytes: []const u8) void {
        // C json_escape_n first makes a terminated copy: an embedded NUL ends
        // the JSON string even if the raw stream buffer contains later bytes.
        const string = bytes[0..(std.mem.findScalar(u8, bytes, 0) orelse bytes.len)];
        self.byte('"');
        const hex = "0123456789abcdef";
        for (string) |value| switch (value) {
            '"', '\\' => {
                self.byte('\\');
                self.byte(value);
            },
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            else => if (value < 0x20) {
                self.append("\\u00");
                self.byte(hex[value >> 4]);
                self.byte(hex[value & 15]);
            } else self.byte(value),
        };
        self.byte('"');
    }
    fn terminated(self: *Text) [*:0]const u8 {
        self.list.ensureUnusedCapacity(allocator, 1) catch fatal();
        self.list.items.ptr[self.list.items.len] = 0;
        return @ptrCast(self.list.items.ptr);
    }
};

fn appendBuffer(buffer: *c.buf, bytes: []const u8) void {
    // Temporarily adopt the real C allocation; transfer it back, including its
    // terminator capacity, rather than keeping a second permanent text mirror.
    var list: std.ArrayList(u8) = .empty;
    if (buffer.ptr != null) list.items = buffer.ptr[0..buffer.len];
    list.capacity = buffer.cap;
    list.appendSlice(allocator, bytes) catch fatal();
    list.ensureUnusedCapacity(allocator, 1) catch fatal();
    list.items.ptr[list.items.len] = 0;
    buffer.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
}
fn bufferText(buffer: *const c.buf) []const u8 {
    return if (buffer.len == 0) &.{} else buffer.ptr[0..buffer.len];
}

var fallback_counter: u64 = 0;
fn randomBytes(bytes: []u8) bool {
    const descriptor = c.open("/dev/urandom", c.O_RDONLY);
    if (descriptor < 0) return false;
    defer _ = c.close(descriptor);
    var done: usize = 0;
    while (done < bytes.len) {
        const count = c.read(descriptor, bytes.ptr + done, bytes.len - done);
        if (count < 0 and c.__error().* == c.EINTR) continue;
        if (count <= 0) return false;
        done += @intCast(count);
    }
    return true;
}
/// The 24-hex suffix and prefix identify one item throughout its lifecycle.
/// Retain the existing bounded-prefix and entropy-failure fallback behavior.
pub fn randomId(destination: []u8, prefix: []const u8) void {
    if (destination.len == 0) return;
    const copied = @min(prefix.len, destination.len - 1);
    @memcpy(destination[0..copied], prefix[0..copied]);
    destination[copied] = 0;
    if (prefix.len >= destination.len) return;
    var bytes: [12]u8 = undefined;
    if (!randomBytes(&bytes)) {
        const a = (@as(u64, @bitCast(@as(i64, c.time(null)))) << 32) ^ @as(u64, @intCast(c.getpid()));
        fallback_counter +%= 1;
        const b = fallback_counter ^ @as(u64, @intFromPtr(destination.ptr));
        @memcpy(bytes[0..8], std.mem.asBytes(&a));
        @memcpy(bytes[8..12], std.mem.asBytes(&b)[0..4]);
    }
    const hex = "0123456789abcdef";
    var position = prefix.len;
    for (bytes) |value| {
        if (position + 2 >= destination.len) break;
        destination[position] = hex[value >> 4];
        destination[position + 1] = hex[value & 15];
        position += 2;
    }
    destination[position] = 0;
}
pub fn init(request: *const Request, stream: *Stream) void {
    stream.* = std.mem.zeroes(Stream);
    stream.sink = request.response_sink;
    stream.mode = if (c.ds4_think_mode_enabled(request.think_mode)) c.RESP_STREAM_THINKING else c.RESP_STREAM_TEXT;
    randomId(&stream.response_id, "resp_");
    randomId(&stream.reasoning_id, "rs_");
    randomId(&stream.message_id, "msg_");
    stream.reasoning_index = -1;
    stream.message_index = -1;
}
pub fn deinit(stream: *Stream) void {
    c.free(stream.reasoning_text.ptr);
    c.free(stream.message_text.ptr);
    stream.reasoning_text = std.mem.zeroes(c.buf);
    stream.message_text = std.mem.zeroes(c.buf);
}
fn status(finish: []const u8) []const u8 {
    if (std.mem.eql(u8, finish, "length")) return "incomplete";
    if (std.mem.eql(u8, finish, "error")) return "failed";
    return "completed";
}
fn itemStatus(finish: []const u8) []const u8 {
    return if (std.mem.eql(u8, finish, "length") or std.mem.eql(u8, finish, "error")) "incomplete" else "completed";
}
fn clampUsage(value: c_int, maximum: c_int) c_int {
    return if (value < 0) 0 else if (maximum >= 0 and value > maximum) maximum else value;
}
fn usage(out: *Text, request: ?*const Request, input: c_int, output: c_int) void {
    const cached = clampUsage(if (request) |r| r.cache_read_tokens else 0, input);
    const written = clampUsage(if (request) |r| r.cache_write_tokens else 0, input - cached);
    out.append("{\"input_tokens\":");
    out.number(input);
    out.append(",\"input_tokens_details\":{\"cached_tokens\":");
    out.number(cached);
    out.append(",\"cache_write_tokens\":");
    out.number(written);
    out.append("},\"output_tokens\":");
    out.number(output);
    out.append(",\"output_tokens_details\":{\"reasoning_tokens\":");
    out.number(clampUsage(if (request) |r| r.generated_thinking_tokens else 0, output));
    out.append("},\"total_tokens\":");
    out.number(input + output);
    out.byte('}');
}
pub fn appendUsage(out: *c.buf, request: ?*const Request, input: c_int, output: c_int) void {
    var json: Text = .{};
    defer json.deinit();
    usage(&json, request, input, output);
    appendBuffer(out, json.list.items);
}
fn schema(orders: ?*const c.tool_schema_orders, name: [*c]const u8) ?*const c.tool_schema_order {
    if (name == null) return null;
    const values = orders orelse return null;
    const items: []const c.tool_schema_order = if (values.len == 0) &.{} else @as([*]const c.tool_schema_order, @ptrCast(values.v))[0..@intCast(values.len)];
    for (items) |*order| if (order.name != null and std.mem.eql(u8, text(order.name), text(name))) return order;
    return null;
}
fn toolSearch(call: *const c.tool_call, order: ?*const c.tool_schema_order) bool {
    return call.name != null and std.mem.eql(u8, text(call.name), "tool_search") and (order == null or order.?.responses_tool_search);
}
fn canonicalArguments(arguments: [*c]const u8) [*c]u8 {
    return c.ds4_server_output_object_json(arguments);
}
fn appendItem(out: *Text, call: *const c.tool_call, item: *const Item, item_status: []const u8, with_args: bool, orders: ?*const c.tool_schema_orders) void {
    const order = schema(orders, call.name);
    out.append("{\"id\":");
    out.quoted(text(&item.fc_id));
    if (toolSearch(call, order)) {
        out.append(",\"type\":\"tool_search_call\",\"status\":");
        out.quoted(item_status);
        // Retain legacy hosted-call interpolation. Generated IDs are safe;
        // ordinary function_call IDs below already use JSON escaping in C.
        out.append(",\"call_id\":\"");
        out.append(text(&item.call_id));
        out.byte('"');
        out.append(",\"execution\":\"client\",\"arguments\":");
        if (with_args) {
            const arguments = canonicalArguments(call.arguments);
            defer c.free(arguments);
            out.append(text(arguments));
        } else out.append("{}");
        out.byte('}');
        return;
    }
    out.append(",\"type\":");
    out.quoted(if (item.is_custom) "custom_tool_call" else "function_call");
    out.append(",\"status\":");
    out.quoted(item_status);
    out.append(",\"name\":");
    out.quoted(if (order != null and order.?.wire_name != null) text(order.?.wire_name) else text(call.name));
    if (order != null and order.?.namespace != null) {
        out.append(",\"namespace\":");
        out.quoted(text(order.?.namespace));
    }
    out.append(",\"call_id\":");
    out.quoted(text(&item.call_id));
    out.append(if (item.is_custom) ",\"input\":" else ",\"arguments\":");
    if (!with_args) out.quoted("") else if (item.is_custom) out.quoted(text(call.arguments)) else {
        const arguments = canonicalArguments(call.arguments);
        defer c.free(arguments);
        out.quoted(text(arguments));
    }
    out.byte('}');
}
pub fn appendFunctionCall(out: *c.buf, call: *const c.tool_call, item: *const Item, item_status: []const u8, with_args: bool, orders: ?*const c.tool_schema_orders) void {
    var json: Text = .{};
    defer json.deinit();
    appendItem(&json, call, item, item_status, with_args, orders);
    appendBuffer(out, json.list.items);
}
fn buildItems(calls: ?*const c.tool_calls, starting_index: c_int) []Item {
    const values = callSlice(calls);
    const items = allocator.alloc(Item, values.len) catch fatal();
    for (values, items, 0..) |call, *item, index| {
        item.* = std.mem.zeroes(Item);
        randomId(&item.fc_id, "fc_");
        const id = text(call.id);
        if (id.len != 0) {
            const length = @min(id.len, item.call_id.len - 1);
            @memcpy(item.call_id[0..length], id[0..length]);
        } else randomId(&item.call_id, "call_");
        // The internal call has no custom/function discriminator. Preserve the
        // original default while retaining client IDs and stable fc_ identity.
        item.is_custom = false;
        item.output_index = starting_index + @as(c_int, @intCast(index));
    }
    return items;
}
fn storageFields(out: *Text, request: *const Request) void {
    out.append(if (request.responses_store) ",\"store\":true,\"previous_response_id\":" else ",\"store\":false,\"previous_response_id\":");
    if (request.previous_response_id != null) out.quoted(text(request.previous_response_id)) else out.append("null");
    out.append(",\"conversation\":");
    if (request.conversation_id != null) {
        out.append("{\"id\":");
        out.quoted(text(request.conversation_id));
        out.byte('}');
    } else out.append("null");
}
fn reasonItem(out: *Text, id: []const u8, item_status: []const u8, value: []const u8) void {
    out.append("{\"id\":");
    out.quoted(id);
    out.append(",\"type\":\"reasoning\",\"status\":");
    out.quoted(item_status);
    out.append(",\"summary\":[");
    if (value.len != 0) {
        out.append("{\"type\":\"summary_text\",\"text\":");
        out.quoted(value);
        out.byte('}');
    }
    out.append("]}");
}
fn messageItem(out: *Text, id: []const u8, item_status: []const u8, value: []const u8) void {
    out.append("{\"id\":");
    out.quoted(id);
    out.append(",\"type\":\"message\",\"status\":");
    out.quoted(item_status);
    out.append(",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
    out.quoted(value);
    out.append(",\"annotations\":[]}]}");
}
// Optional request metadata uses the standard parser. The actual sampling and
// storage values below come from Request, never an unchecked wire override.
const Options = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,

    fn init(request: *const Request) Options {
        if (request.raw_body == null) return .{};
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, text(request.raw_body), .{ .duplicate_field_behavior = .use_last }) catch return .{};
        return .{ .parsed = parsed };
    }
    fn deinit(self: *Options) void {
        if (self.parsed) |parsed| parsed.deinit();
    }
    fn field(self: *const Options, name: []const u8) ?std.json.Value {
        const parsed = self.parsed orelse return null;
        return if (parsed.value == .object) parsed.value.object.get(name) else null;
    }
    fn append(self: *const Options, out: *Text, name: []const u8, fallback: []const u8) void {
        out.append(",\"");
        out.append(name);
        out.append("\":");
        if (self.field(name)) |value| jsonValue(out, value) else out.append(fallback);
    }
};
fn jsonValue(out: *Text, value: std.json.Value) void {
    const encoded = std.json.Stringify.valueAlloc(allocator, value, .{}) catch fatal();
    defer allocator.free(encoded);
    out.append(encoded);
}
fn objectField(out: *Text, object: std.json.ObjectMap, name: []const u8, fallback: []const u8) void {
    out.append(",\"");
    out.append(name);
    out.append("\":");
    if (object.get(name)) |value| jsonValue(out, value) else out.append(fallback);
}
fn toolsField(out: *Text, options: *const Options) void {
    out.append(",\"tools\":[");
    if (options.field("tools")) |tools| if (tools == .array) {
        for (tools.array.items, 0..) |tool, index| {
            if (index != 0) out.byte(',');
            // Function output requires nullable fields that may be absent from
            // the input declaration. Preserve provider extensions as-is.
            if (tool == .object) {
                const kind = tool.object.get("type");
                if (kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "function")) {
                    out.append("{\"type\":\"function\",\"name\":");
                    if (tool.object.get("name")) |name| jsonValue(out, name) else out.quoted("");
                    objectField(out, tool.object, "description", "null");
                    objectField(out, tool.object, "parameters", "null");
                    objectField(out, tool.object, "strict", "null");
                    out.byte('}');
                    continue;
                }
            }
            jsonValue(out, tool);
        }
    };
    out.byte(']');
}
fn reasoningField(out: *Text, request: *const Request, options: *const Options) void {
    const effort: []const u8 = switch (request.think_mode) {
        c.DS4_THINK_NONE => "none",
        c.DS4_THINK_LOW => "low",
        c.DS4_THINK_MEDIUM => "medium",
        c.DS4_THINK_MAX => "xhigh",
        c.DS4_THINK_HIGH => "high",
        else => block: {
            const level = c.ds4_think_mode_level(request.think_mode);
            break :block if (level <= 0) "none" else if (level <= 25) "low" else if (level <= 50) "medium" else if (level <= 75) "high" else "xhigh";
        },
    };
    out.append(",\"reasoning\":{\"effort\":");
    out.quoted(effort);
    out.append(",\"summary\":");
    if (request.reasoning_summary_emit) {
        const raw = options.field("reasoning");
        if (raw != null and raw.? == .object) {
            if (raw.?.object.get("summary")) |summary| jsonValue(out, summary) else out.quoted("auto");
        } else out.quoted("auto");
    } else out.append("null");
    out.byte('}');
}
fn terminalDetails(out: *Text, finish: ?[]const u8) void {
    if (finish != null and std.mem.eql(u8, finish.?, "error")) out.append(",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}") else out.append(",\"error\":null");
    if (finish != null and std.mem.eql(u8, finish.?, "length")) out.append(",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}") else out.append(",\"incomplete_details\":null");
}
/// All ResponseResource envelopes share this shape across both transports.
/// Output and usage follow after these common fields.
fn responsePrefix(out: *Text, request: *const Request, id: []const u8, created_at: c_long, response_status: []const u8, finish: ?[]const u8) void {
    var options = Options.init(request);
    defer options.deinit();
    out.append("{\"id\":");
    out.quoted(id);
    out.append(",\"object\":\"response\",\"created_at\":");
    out.number(created_at);
    out.append(",\"status\":");
    out.quoted(response_status);
    out.append(",\"model\":");
    out.quoted(text(request.model));
    storageFields(out, request);
    out.append(",\"completed_at\":");
    if (std.mem.eql(u8, response_status, "completed")) out.number(c.time(null)) else out.append("null");
    terminalDetails(out, finish);
    options.append(out, "instructions", "null");
    toolsField(out, &options);
    options.append(out, "tool_choice", "\"auto\"");
    // Report fixed policies honestly rather than echoing ignored options.
    out.append(",\"truncation\":\"disabled\",\"parallel_tool_calls\":true,\"text\":{\"format\":{\"type\":\"text\"}},\"top_p\":");
    out.number(request.top_p);
    out.append(",\"presence_penalty\":0,\"frequency_penalty\":0,\"top_logprobs\":0,\"temperature\":");
    out.number(request.temperature);
    reasoningField(out, request, &options);
    out.append(",\"max_output_tokens\":");
    if (request.max_tokens >= 0) out.number(request.max_tokens) else out.append("null");
    out.append(",\"max_tool_calls\":null,\"background\":false,\"service_tier\":\"default\"");
    options.append(out, "metadata", "{}");
    options.append(out, "safety_identifier", "null");
    out.append(",\"prompt_cache_key\":");
    if (request.prompt_cache_key != null) out.quoted(text(request.prompt_cache_key)) else out.append("null");
}
fn event(kind: []const u8) Text {
    var out: Text = .{};
    out.append("{\"type\":");
    out.quoted(kind);
    return out;
}
fn itemIdentity(out: *Text, id: []const u8, index: c_int) void {
    out.append(",\"item_id\":");
    out.quoted(id);
    out.append(",\"output_index\":");
    out.number(index);
}

pub const Output = struct {
    descriptor: c_int,
    stop_requested: Stop,

    fn emit(self: Output, stream: *Stream, body: *Text) bool {
        // Inject sequence_number immediately after the type string, before any
        // other fields. Advance before sending, including a failed send.
        const bytes = body.list.items;
        var frame: Text = .{};
        defer frame.deinit();
        var closing: ?usize = null;
        if (std.mem.startsWith(u8, bytes, "{\"type\":\"")) {
            var index: usize = 9;
            while (index < bytes.len and bytes[index] != '"') {
                index += if (bytes[index] == '\\' and index + 1 < bytes.len) @as(usize, 2) else 1;
            }
            if (index < bytes.len) closing = index + 1;
        }
        if (closing) |end| {
            frame.append(bytes[0..end]);
            frame.append(",\"sequence_number\":");
            frame.number(stream.sequence);
            stream.sequence += 1;
            frame.append(bytes[end..]);
        } else frame.append(bytes);
        if (stream.sink != null) {
            const sink = stream.sink.*;
            if (sink.emit) |send| return send(sink.context, frame.list.items.ptr, frame.list.items.len);
            return false;
        }
        var wire: Text = .{};
        defer wire.deinit();
        if (closing) |end| {
            wire.append("event: ");
            wire.append(bytes[9 .. end - 1]);
            wire.byte('\n');
        }
        wire.append("data: ");
        wire.append(frame.list.items);
        wire.append("\n\n");
        return c.ds4_server_http_send_all(self.descriptor, wire.list.items.ptr, wire.list.items.len, self.stop_requested);
    }
    pub fn created(self: Output, request: *const Request, stream: *Stream, created_at: c_long) bool {
        var out = event("response.created");
        defer out.deinit();
        out.append(",\"response\":");
        responsePrefix(&out, request, text(&stream.response_id), created_at, "in_progress", null);
        out.append(",\"output\":[],\"usage\":null}}");
        if (!self.emit(stream, &out)) return false;
        out.list.clearRetainingCapacity();
        out.append("{\"type\":\"response.in_progress\",\"response\":");
        responsePrefix(&out, request, text(&stream.response_id), created_at, "in_progress", null);
        out.append(",\"output\":[],\"usage\":null}}");
        return self.emit(stream, &out);
    }
    fn added(self: Output, stream: *Stream, reasoning: bool) bool {
        var out = event("response.output_item.added");
        defer out.deinit();
        out.append(",\"output_index\":");
        out.number(if (reasoning) stream.reasoning_index else stream.message_index);
        out.append(",\"item\":{\"id\":");
        out.quoted(if (reasoning) text(&stream.reasoning_id) else text(&stream.message_id));
        out.append(if (reasoning) ",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}}" else ",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}");
        return self.emit(stream, &out);
    }
    fn partAdded(self: Output, stream: *Stream, reasoning: bool) bool {
        var out = event(if (reasoning) "response.reasoning_summary_part.added" else "response.content_part.added");
        defer out.deinit();
        itemIdentity(&out, if (reasoning) text(&stream.reasoning_id) else text(&stream.message_id), if (reasoning) stream.reasoning_index else stream.message_index);
        out.append(if (reasoning) ",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}" else ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}");
        return self.emit(stream, &out);
    }
    fn delta(self: Output, stream: *Stream, reasoning: bool, value: []const u8) bool {
        if (value.len == 0) return true;
        var out = event(if (reasoning) "response.reasoning_summary_text.delta" else "response.output_text.delta");
        defer out.deinit();
        itemIdentity(&out, if (reasoning) text(&stream.reasoning_id) else text(&stream.message_id), if (reasoning) stream.reasoning_index else stream.message_index);
        out.append(if (reasoning) ",\"summary_index\":0,\"delta\":" else ",\"content_index\":0,\"delta\":");
        out.quoted(value);
        out.byte('}');
        return self.emit(stream, &out);
    }
    fn done(self: Output, stream: *Stream, reasoning: bool, finish: []const u8) bool {
        const id = if (reasoning) text(&stream.reasoning_id) else text(&stream.message_id);
        const index = if (reasoning) stream.reasoning_index else stream.message_index;
        const value = if (reasoning) bufferText(&stream.reasoning_text) else bufferText(&stream.message_text);
        // A missing </think> always makes a reasoning item incomplete, even
        // when the response itself stopped normally. Never publish it as a
        // replayable completed hidden-state turn.
        const item_status = if (reasoning) (if (stream.reasoning_closed_naturally) "completed" else "incomplete") else itemStatus(finish);
        var out = event(if (reasoning) "response.reasoning_summary_text.done" else "response.output_text.done");
        defer out.deinit();
        itemIdentity(&out, id, index);
        out.append(if (reasoning) ",\"summary_index\":0,\"text\":" else ",\"content_index\":0,\"text\":");
        out.quoted(value);
        out.byte('}');
        if (!self.emit(stream, &out)) return false;
        if (!reasoning or stream.reasoning_summary_started) {
            out.list.clearRetainingCapacity();
            out.append("{\"type\":");
            out.quoted(if (reasoning) "response.reasoning_summary_part.done" else "response.content_part.done");
            itemIdentity(&out, id, index);
            out.append(if (reasoning) ",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":" else ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":");
            out.quoted(value);
            out.append(if (reasoning) "}}" else ",\"annotations\":[]}}");
            if (!self.emit(stream, &out)) return false;
        }
        out.list.clearRetainingCapacity();
        out.append("{\"type\":\"response.output_item.done\",\"output_index\":");
        out.number(index);
        out.append(",\"item\":");
        if (reasoning) reasonItem(&out, id, item_status, value) else messageItem(&out, id, item_status, value);
        out.byte('}');
        return self.emit(stream, &out);
    }
    fn toolEvent(self: Output, stream: *Stream, call: *const c.tool_call, item: *const Item, orders: *const c.tool_schema_orders, finish: []const u8, is_done: bool) bool {
        // Added carries empty arguments: delta+done consumers must not append
        // the same complete JSON twice. Final status appears only in done.
        var out = event(if (is_done) "response.output_item.done" else "response.output_item.added");
        defer out.deinit();
        out.append(",\"output_index\":");
        out.number(item.output_index);
        out.append(",\"item\":");
        appendItem(&out, call, item, if (is_done) itemStatus(finish) else "in_progress", is_done, orders);
        out.byte('}');
        return self.emit(stream, &out);
    }
    fn argumentsDone(self: Output, stream: *Stream, call: *const c.tool_call, item: *const Item, orders: *const c.tool_schema_orders) bool {
        // The model emits a complete invoke before the worker identifies the
        // tool. Send its arguments as one delta + done between item added/done.
        const order = schema(orders, call.name);
        if (item.is_custom or toolSearch(call, order)) return true;
        const arguments = canonicalArguments(call.arguments);
        defer c.free(arguments);
        var out = event("response.function_call_arguments.delta");
        defer out.deinit();
        itemIdentity(&out, text(&item.fc_id), item.output_index);
        out.append(",\"delta\":");
        out.quoted(text(arguments));
        out.byte('}');
        if (!self.emit(stream, &out)) return false;
        out.list.clearRetainingCapacity();
        out.append("{\"type\":\"response.function_call_arguments.done\"");
        itemIdentity(&out, text(&item.fc_id), item.output_index);
        out.append(",\"name\":");
        out.quoted(if (order != null and order.?.wire_name != null) text(order.?.wire_name) else text(call.name));
        if (order != null and order.?.namespace != null) {
            out.append(",\"namespace\":");
            out.quoted(text(order.?.namespace));
        }
        out.append(",\"arguments\":");
        out.quoted(text(arguments));
        out.byte('}');
        return self.emit(stream, &out);
    }
    pub fn completed(self: Output, request: *Request, stream: *Stream, calls: ?*const c.tool_calls, items: []const Item, finish: []const u8, input: c_int, output: c_int, created_at: c_long) bool {
        // Clients route terminal behavior by event type, not response.status.
        const kind: []const u8 = if (std.mem.eql(u8, finish, "error")) "response.failed" else if (std.mem.eql(u8, finish, "length")) "response.incomplete" else "response.completed";
        var body: Text = .{};
        defer body.deinit();
        responsePrefix(&body, request, text(&stream.response_id), created_at, status(finish), finish);
        body.append(",\"output\":[");
        var wrote = false;
        if (stream.reasoning_emitted_any) {
            reasonItem(&body, text(&stream.reasoning_id), if (stream.reasoning_closed_naturally) "completed" else "incomplete", bufferText(&stream.reasoning_text));
            wrote = true;
        }
        if (stream.message_emitted_any) {
            if (wrote) body.byte(',');
            messageItem(&body, text(&stream.message_id), itemStatus(finish), bufferText(&stream.message_text));
            wrote = true;
        }
        for (callSlice(calls), 0..) |*call, index| {
            if (items.len == 0) break;
            if (wrote) body.byte(',');
            appendItem(&body, call, &items[index], itemStatus(finish), true, &request.tool_orders);
            wrote = true;
        }
        body.append("],\"usage\":");
        usage(&body, request, input, output);
        body.byte('}');
        var error_text: [160]u8 = @splat(0);
        const stored = c.ds4_server_output_publish(request, &stream.response_id, body.terminated(), &error_text, error_text.len);
        var framed = event(if (stored) kind else "error");
        defer framed.deinit();
        if (stored) {
            framed.append(",\"response\":");
            framed.append(body.list.items);
            framed.byte('}');
        } else {
            if (stream.sink != null) framed.append(",\"status\":500");
            framed.append(",\"error\":{\"type\":\"server_error\",\"code\":\"server_error\",\"message\":");
            framed.quoted(text(&error_text));
            framed.append(",\"param\":null}}");
        }
        return self.emit(stream, &framed);
    }
    pub fn update(self: Output, request: *const Request, stream: *Stream, raw: []const u8, final: bool) bool {
        if (!stream.active) return true;
        // Walk reasoning even when summary output was not requested, but never
        // expose it on the visible message surface as a shortcut.
        // Generation starts inside the prompt's think tag. Missing an opening
        // tag is not proof that reasoning has ended: only </think> switches it.
        if (stream.mode == c.RESP_STREAM_THINKING) {
            if (!stream.checked_think_prefix) {
                if (!final and raw.len < 7 and std.mem.startsWith(u8, "<think>", raw)) return true;
                if (std.mem.startsWith(u8, raw, "<think>")) stream.emit_pos = 7;
                stream.checked_think_prefix = true;
            }
            const close = findCString(raw, stream.emit_pos, "</think>");
            const tool = if (request.has_tools) findTool(raw, stream.emit_pos, false) else null;
            const before_close = tool != null and (close == null or tool.? < close.?);
            const complete_tool = before_close and findTool(raw, tool.?, true) != null;
            const limit = if (before_close) trimSeparator(raw, stream.emit_pos, tool.?) else if (close) |position| position else if (final) raw.len else safeUtf8(raw, stream.emit_pos, if (raw.len > 7) raw.len - 7 else stream.emit_pos, false);
            if (limit > stream.emit_pos) {
                if (request.reasoning_summary_emit) {
                    if (!stream.reasoning_item_opened) {
                        stream.reasoning_index = stream.next_output_index;
                        stream.next_output_index += 1;
                        if (!self.added(stream, true)) return false;
                        stream.reasoning_item_opened = true;
                    }
                    if (!stream.reasoning_summary_started) {
                        if (!self.partAdded(stream, true)) return false;
                        stream.reasoning_summary_started = true;
                    }
                    if (!self.delta(stream, true, raw[stream.emit_pos..limit])) return false;
                    appendBuffer(&stream.reasoning_text, raw[stream.emit_pos..limit]);
                    stream.reasoning_emitted_any = true;
                }
                stream.emit_pos = limit;
            }
            if (before_close) {
                if (complete_tool) {
                    stream.emit_pos = tool.?;
                    stream.mode = c.RESP_STREAM_SUPPRESS;
                }
                return true;
            }
            if (close) |position| {
                stream.emit_pos = position + 8;
                stream.mode = c.RESP_STREAM_TEXT;
                stream.reasoning_closed_naturally = true;
            } else if (final) {
                stream.mode = c.RESP_STREAM_SUPPRESS;
                return true;
            } else return true;
        }
        if (stream.mode == c.RESP_STREAM_TEXT) {
            const tool = if (request.has_tools) findTool(raw, stream.emit_pos, false) else null;
            const limit = safeTextLimit(raw, stream.emit_pos, request.has_tools, final);
            if (limit > stream.emit_pos) {
                if (!stream.message_item_opened) {
                    stream.message_index = stream.next_output_index;
                    stream.next_output_index += 1;
                    if (!self.added(stream, false)) return false;
                    stream.message_item_opened = true;
                }
                if (!stream.message_text_part_open) {
                    if (!self.partAdded(stream, false)) return false;
                    stream.message_text_part_open = true;
                }
                if (!self.delta(stream, false, raw[stream.emit_pos..limit])) return false;
                appendBuffer(&stream.message_text, raw[stream.emit_pos..limit]);
                stream.message_emitted_any = true;
                stream.emit_pos = limit;
            }
            if (tool) |position| {
                stream.emit_pos = position;
                stream.mode = c.RESP_STREAM_SUPPRESS;
            } else if (final) stream.mode = c.RESP_STREAM_SUPPRESS;
        }
        return true;
    }
    pub fn finishLive(self: Output, request: *Request, stream: *Stream, raw: ?[]const u8, recovered: bool, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int, created_at: c_long) bool {
        if (raw) |value| if (!self.update(request, stream, value, true)) return false;
        // Close summary parts before later message/tool items and completion.
        if (stream.reasoning_item_opened and !stream.reasoning_item_closed) {
            if (!self.done(stream, true, finish)) return false;
            stream.reasoning_item_closed = true;
        }
        // Parsing recovery may promote a suppressed raw tool tail to text. Use
        // the stream cursor, not recovered_content (which may include thinking).
        if (recovered and raw != null and stream.emit_pos < raw.?.len) {
            if (!stream.message_item_opened) {
                stream.message_index = stream.next_output_index;
                stream.next_output_index += 1;
                if (!self.added(stream, false)) return false;
                stream.message_item_opened = true;
            }
            if (!stream.message_text_part_open) {
                if (!self.partAdded(stream, false)) return false;
                stream.message_text_part_open = true;
            }
            const tail = raw.?[stream.emit_pos..];
            if (!self.delta(stream, false, tail)) return false;
            appendBuffer(&stream.message_text, tail);
            stream.message_emitted_any = true;
            stream.emit_pos = raw.?.len;
        }
        if (stream.message_item_opened and !stream.message_item_closed) {
            if (!self.done(stream, false, finish)) return false;
            stream.message_item_closed = true;
        }
        const items = buildItems(calls, stream.next_output_index);
        defer allocator.free(items);
        if (items.len != 0) stream.next_output_index += @intCast(items.len);
        for (callSlice(calls), items) |*call, *item| {
            if (!self.toolEvent(stream, call, item, &request.tool_orders, finish, false) or
                !self.argumentsDone(stream, call, item, &request.tool_orders) or
                !self.toolEvent(stream, call, item, &request.tool_orders, finish, true)) return false;
        }
        return self.completed(request, stream, calls, items, finish, input, output, created_at);
    }
    pub fn respond(self: Output, cors: bool, request: *Request, content: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int) bool {
        var response_id: [40]u8 = undefined;
        var reasoning_id: [40]u8 = undefined;
        var message_id: [40]u8 = undefined;
        randomId(&response_id, "resp_");
        randomId(&reasoning_id, "rs_");
        randomId(&message_id, "msg_");
        const items = buildItems(calls, 0);
        defer allocator.free(items);
        var body: Text = .{};
        defer body.deinit();
        responsePrefix(&body, request, text(&response_id), c.time(null), status(finish), finish);
        body.append(",\"output\":[");
        var wrote = false;
        // Worker post-processing supplies parsed reasoning; retain response
        // finish-status rules here, distinct from live natural-close tracking.
        if (reasoning.len != 0 and request.reasoning_summary_emit) {
            reasonItem(&body, text(&reasoning_id), itemStatus(finish), reasoning);
            wrote = true;
        }
        if (content.len != 0) {
            if (wrote) body.byte(',');
            messageItem(&body, text(&message_id), itemStatus(finish), content);
            wrote = true;
        }
        for (callSlice(calls), items) |*call, *item| {
            if (wrote) body.byte(',');
            appendItem(&body, call, item, itemStatus(finish), true, &request.tool_orders);
            wrote = true;
        }
        body.append("],\"usage\":");
        usage(&body, request, input, output);
        body.byte('}');
        var error_text: [160]u8 = @splat(0);
        if (c.ds4_server_output_publish(request, &response_id, body.terminated(), &error_text, error_text.len))
            return c.ds4_server_http_response(self.descriptor, cors, 200, "application/json", body.terminated(), self.stop_requested);
        var failure: Text = .{};
        defer failure.deinit();
        failure.append("{\"error\":{\"message\":");
        failure.quoted(text(&error_text));
        failure.append(",\"type\":\"invalid_request_error\"}}\n");
        return c.ds4_server_http_response(self.descriptor, cors, if (request.responses_store_error_status != 0) request.responses_store_error_status else 500, "application/json", failure.terminated(), self.stop_requested);
    }
};

const starts = [_][]const u8{ "<｜DSML｜ calls>", "<｜DSML｜tool_calls>", "<DSML｜tool_calls>", "<tool_calls>", "<tool_call>" };
const ends = [_][]const u8{ "</｜DSML｜ calls>", "</｜DSML｜tool_calls>", "</DSML｜tool_calls>", "</tool_calls>", "</tool_call>" };
fn findCString(raw: []const u8, offset: usize, marker: []const u8) ?usize {
    if (offset > raw.len) return null;
    const end = if (std.mem.findScalar(u8, raw[offset..], 0)) |position| offset + position else raw.len;
    return std.mem.findPos(u8, raw[0..end], offset, marker);
}
fn findTool(raw: []const u8, offset: usize, closing: bool) ?usize {
    var best: ?usize = null;
    for (if (closing) ends else starts) |marker| if (findCString(raw, offset, marker)) |position| {
        if (best == null or position < best.?) best = position;
    };
    return best;
}
fn trimSeparator(raw: []const u8, start: usize, end: usize) usize {
    var limit = end;
    while (limit > start and c.isspace(raw[limit - 1]) != 0) limit -= 1;
    return limit;
}
fn expectedUtf8(byte: u8) usize {
    return if (byte >= 0xc2 and byte <= 0xdf) 2 else if (byte >= 0xe0 and byte <= 0xef) 3 else if (byte >= 0xf0 and byte <= 0xf4) 4 else 1;
}
fn safeUtf8(raw: []const u8, start: usize, limit: usize, final: bool) usize {
    if (final or limit <= start) return limit;
    var position = limit;
    var continuations: usize = 0;
    while (position > start and continuations < 4 and raw[position - 1] & 0xc0 == 0x80) {
        position -= 1;
        continuations += 1;
    }
    if (position == limit) return if (expectedUtf8(raw[limit - 1]) > 1) limit - 1 else limit;
    if (position == start and raw[position] & 0xc0 == 0x80) return start;
    const lead = position - 1;
    return if (limit - lead < expectedUtf8(raw[lead])) lead else limit;
}
fn safeTextLimit(raw: []const u8, start: usize, has_tools: bool, final: bool) usize {
    if (raw.len <= start) return raw.len;
    var limit = raw.len;
    if (has_tools) {
        if (findTool(raw, start, false)) |position| return trimSeparator(raw, start, position);
        if (!final) {
            // Hold syntax separator whitespace and a split '<...tool_calls...'
            // marker, so the client's replay cannot duplicate canonical '\n\n'.
            limit = trimSeparator(raw, start, limit);
            const scan = if (raw.len - start > 80) raw.len - 80 else start;
            var position = raw.len;
            while (position > scan) {
                position -= 1;
                if (raw[position] == '<') {
                    limit = @min(limit, position);
                    break;
                }
            }
            limit = trimSeparator(raw, start, limit);
        }
    }
    // Avoid emitting an incomplete UTF-8 scalar that clients would replace and
    // round-trip as U+FFFD, breaking their next exact KV-prefix match.
    return safeUtf8(raw, start, limit, final);
}
