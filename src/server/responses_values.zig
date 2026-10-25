//! Responses items become the same logical chat records as OpenAI/Anthropic.
//! This owner preserves item order, pending reasoning, tool-result groups and
//! private discovery metadata. Durable history/live-call resolution stays with
//! the existing materializer; primitive JSON and image resources remain C.
const std = @import("std");
const c = @import("server_input_c");
const messages = @import("message_values.zig");
const tools = @import("input_values.zig");
const Reader = messages.Reader;
const Text = messages.Text;
const allocator = std.heap.c_allocator;
const Invalid = error{InvalidJson};

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn bytes(value: [*c]const u8) []const u8 {
    return if (value == null) "" else std.mem.span(value);
}
fn equal(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, bytes(value), expected);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
fn records(list: *c.chat_msgs) []c.chat_msg {
    return if (list.len == 0) &.{} else @as([*]c.chat_msg, @ptrCast(list.v))[0..@intCast(list.len)];
}

const ContentBlock = struct {
    kind: [*c]u8 = null,
    text: [*c]u8 = null,
    url: [*c]u8 = null,

    fn deinit(self: *ContentBlock) void {
        c.free(self.kind);
        c.free(self.text);
        c.free(self.url);
    }
    fn read(self: *ContentBlock, reader: *Reader) Invalid!void {
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "type")) {
                try reader.replace(&self.kind, false);
            } else if (equal(key, "text")) {
                // Typed text fields are strings; null denotes an empty block.
                if (reader.literal("null")) {
                    c.free(self.text);
                    self.text = duplicate("");
                } else try reader.replace(&self.text, false);
            } else if (equal(key, "image_url")) {
                c.free(self.url);
                self.url = null;
                if (!messages.ds4_server_json_image_url(&reader.cursor, &self.url)) return error.InvalidJson;
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
    }
};

/// Fail closed on unknown/missing block types or text, rather than silently
/// discard replay context. A bare string/null or an array of strings/recognized
/// typed blocks is admitted; primitive/array elements and other shapes reject.
fn contentValue(reader: *Reader, images: [*c]c.server_image_inputs) Invalid![*c]u8 {
    if (reader.peek() == '"') return reader.string();
    if (reader.literal("null")) return duplicate("");
    try reader.consume('[');
    var text = Text{};
    defer text.deinit();
    while (reader.more(']')) {
        if (reader.peek() == '"') {
            const part = try reader.string();
            defer c.free(part);
            text.append(bytes(part));
        } else if (reader.peek() == '{') {
            var block = ContentBlock{};
            defer block.deinit();
            try block.read(reader);
            const text_block = equal(block.kind, "input_text") or equal(block.kind, "output_text") or
                equal(block.kind, "text") or equal(block.kind, "summary_text") or equal(block.kind, "reasoning_text");
            if (equal(block.kind, "input_image")) {
                var marker: [c.SERVER_IMAGE_MARKER_BYTES]u8 = undefined;
                if (images == null or !c.ds4_server_image_data_uri(images, block.url, &marker)) return error.InvalidJson;
                text.append(std.mem.sliceTo(&marker, 0));
            } else if (!text_block or block.text == null) {
                return error.InvalidJson;
            } else text.append(bytes(block.text));
        } else return error.InvalidJson;
        reader.separator();
    }
    try reader.consume(']');
    return text.take();
}
fn replaceContent(reader: *Reader, target: *[*c]u8) Invalid!void {
    const replacement = try contentValue(reader, null);
    c.free(target.*);
    target.* = replacement;
}
fn replaceMultimodal(reader: *Reader, target: *[*c]u8, images: *c.server_image_inputs) Invalid!void {
    messages.freeImages(images);
    errdefer messages.freeImages(images);
    const replacement = try contentValue(reader, images);
    c.free(target.*);
    target.* = replacement;
}

const ProgramMetadata = struct {
    arguments: [*c]u8 = null,
    call_ids: ?std.json.Parsed([]const []const u8) = null,
    has_bindings: bool = false,

    fn deinit(self: *ProgramMetadata) void {
        c.free(self.arguments);
        if (self.call_ids) |group| group.deinit();
    }
    fn read(self: *ProgramMetadata, reader: *Reader) Invalid!void {
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "arguments")) {
                try reader.replace(&self.arguments, false);
            } else if (equal(key, "call_ids")) {
                const raw = try reader.raw();
                defer c.free(raw);
                if (self.call_ids) |group| group.deinit();
                self.call_ids = null;
                self.call_ids = std.json.parseFromSlice([]const []const u8, allocator, bytes(raw), .{ .allocate = .alloc_always }) catch return error.InvalidJson;
            } else if (equal(key, "bindings")) {
                self.has_bindings = reader.peek() == '{';
                // The public item retains the binding snapshot. Native logical
                // replay needs only the original invocation and compact result.
                try reader.skip();
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
    }
};

const Item = struct {
    kind: [*c]u8 = null,
    role: [*c]u8 = null,
    content: [*c]u8 = null,
    name: [*c]u8 = null,
    namespace: [*c]u8 = null,
    call_id: [*c]u8 = null,
    call_ids: ?std.json.Parsed([]const []const u8) = null,
    id: [*c]u8 = null,
    arguments: [*c]u8 = null,
    output: [*c]u8 = null,
    input: [*c]u8 = null,
    summary: [*c]u8 = null,
    action: [*c]u8 = null,
    result: [*c]u8 = null,
    tools_json: [*c]u8 = null,
    status: [*c]u8 = null,
    model: [*c]u8 = null,
    code: [*c]u8 = null,
    fingerprint: [*c]u8 = null,
    program: ProgramMetadata = .{},
    code_string: bool = false,
    arguments_json: bool = false,
    input_json: bool = false,
    result_json: bool = false,
    content_images: c.server_image_inputs = std.mem.zeroes(c.server_image_inputs),
    output_images: c.server_image_inputs = std.mem.zeroes(c.server_image_inputs),

    fn deinit(self: *Item) void {
        inline for (.{ "kind", "role", "content", "name", "namespace", "call_id", "id", "arguments", "output", "input", "summary", "action", "result", "tools_json", "status", "model", "code", "fingerprint" }) |field|
            c.free(@field(self, field));
        self.program.deinit();
        messages.freeImages(&self.content_images);
        messages.freeImages(&self.output_images);
        if (self.call_ids) |group| group.deinit();
    }
    fn read(self: *Item, reader: *Reader) Invalid!void {
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "type")) {
                try reader.replace(&self.kind, false);
            } else if (equal(key, "role")) {
                // The SDK may type an unknown provider receipt as a message
                // and include its absent optional role as null when dumping.
                if (reader.literal("null")) {
                    c.free(self.role);
                    self.role = null;
                } else try reader.replace(&self.role, false);
            } else if (equal(key, "content")) {
                try replaceMultimodal(reader, &self.content, &self.content_images);
            } else if (equal(key, "name")) {
                try reader.replace(&self.name, false);
            } else if (equal(key, "namespace")) {
                try reader.replace(&self.namespace, false);
            } else if (equal(key, "call_id")) {
                try reader.replace(&self.call_id, false);
            } else if (equal(key, "call_ids")) {
                const raw = try reader.raw();
                defer c.free(raw);
                if (self.call_ids) |group| group.deinit();
                self.call_ids = null;
                self.call_ids = std.json.parseFromSlice([]const []const u8, allocator, bytes(raw), .{ .allocate = .alloc_always }) catch return error.InvalidJson;
            } else if (equal(key, "id")) {
                try reader.replace(&self.id, false);
            } else if (equal(key, "arguments")) {
                self.arguments_json = reader.peek() != '"';
                try reader.replace(&self.arguments, self.arguments_json);
            } else if (equal(key, "output")) {
                messages.freeImages(&self.output_images);
                if (reader.peek() == '[') {
                    try replaceMultimodal(reader, &self.output, &self.output_images);
                } else try reader.replace(&self.output, reader.peek() != '"');
            } else if (equal(key, "input")) {
                self.input_json = reader.peek() != '"';
                try reader.replace(&self.input, self.input_json);
            } else if (equal(key, "summary")) {
                try replaceContent(reader, &self.summary);
            } else if (equal(key, "action")) {
                try reader.replace(&self.action, true);
            } else if (equal(key, "result")) {
                self.result_json = reader.peek() != '"';
                try reader.replace(&self.result, self.result_json);
            } else if (equal(key, "status")) {
                try reader.replace(&self.status, false);
            } else if (equal(key, "model")) {
                try reader.replace(&self.model, false);
            } else if (equal(key, "code")) {
                self.code_string = reader.peek() == '"';
                try reader.replace(&self.code, true);
            } else if (equal(key, "fingerprint")) {
                try reader.replace(&self.fingerprint, false);
            } else if (equal(key, "ds4")) {
                try self.program.read(reader);
            } else if (equal(key, "tools")) {
                // Declarations stay separate from the visible result body;
                // malformed lists must never become ordinary tool output.
                try reader.replace(&self.tools_json, true);
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
    }
    fn typeName(self: *const Item) [*c]const u8 {
        return if (self.kind != null) self.kind else "message";
    }
    fn validate(self: *const Item) Invalid!void {
        // Provider failures are terminal receipts with model-visible results;
        // ordinary partial/failed items remain unconfirmed history.
        const kind = self.typeName();
        if (equal(kind, "program")) {
            if (bytes(self.id).len == 0 or bytes(self.call_id).len == 0 or !self.code_string or
                bytes(self.fingerprint).len == 0 or self.program.arguments == null or
                !self.program.has_bindings or
                (bytes(self.status).len != 0 and !equal(self.status, "completed"))) return error.InvalidJson;
            const group = self.program.call_ids orelse return error.InvalidJson;
            var contains_call = false;
            for (group.value) |id| {
                if (id.len == 0) return error.InvalidJson;
                if (std.mem.eql(u8, id, bytes(self.call_id))) contains_call = true;
            }
            if (!contains_call) return error.InvalidJson;
        } else if (equal(kind, "program_output")) {
            if (bytes(self.id).len == 0 or bytes(self.call_id).len == 0 or self.result == null or self.result_json or
                (!equal(self.status, "completed") and !equal(self.status, "incomplete"))) return error.InvalidJson;
        } else if (equal(kind, "ds4:decision_call")) {
            if (!equal(self.status, "completed") and !equal(self.status, "failed")) return error.InvalidJson;
            if (bytes(self.id).len == 0 or bytes(self.call_id).len == 0 or bytes(self.model).len == 0 or
                self.arguments == null or self.arguments_json or self.input == null or
                !self.result_json or !std.mem.startsWith(u8, bytes(self.result), "{")) return error.InvalidJson;
            const group = self.call_ids orelse return error.InvalidJson;
            var contains_call = false;
            for (group.value) |id| {
                if (id.len == 0) return error.InvalidJson;
                if (std.mem.eql(u8, id, bytes(self.call_id))) contains_call = true;
            }
            if (!contains_call) return error.InvalidJson;
            if (equal(self.status, "completed") and self.input_json and !std.mem.startsWith(u8, bytes(self.input), "[")) return error.InvalidJson;
        } else if (bytes(self.status).len != 0 and !equal(self.status, "completed")) return error.InvalidJson;
        if (self.content_images.len != 0 and (!equal(kind, "message") or (self.role != null and !equal(self.role, "user")))) return error.InvalidJson;
        if (self.output_images.len != 0 and !equal(kind, "function_call_output") and !equal(kind, "custom_tool_call_output")) return error.InvalidJson;
    }
    fn callId(self: *const Item) [*c]const u8 {
        return if (self.call_id != null) self.call_id else if (self.id != null) self.id else "";
    }
};

fn pushResult(list: *c.chat_msgs, message: c.chat_msg) void {
    // A later parallel result belongs before the trailing schema announcements
    // for its complete result group, rather than after a system update.
    var at: usize = @intCast(list.len);
    const before = records(list);
    while (at > 0 and before[at - 1].discovered_tools != null) at -= 1;
    messages.pushMessage(list, message);
    const after = records(list);
    if (at < after.len - 1) {
        std.mem.copyBackwards(c.chat_msg, after[at + 1 ..], after[at .. after.len - 1]);
        after[at] = message;
    }
}

const Decoder = struct {
    list: *c.chat_msgs,
    loaded: ?*Text,
    orders: [*c]c.tool_schema_orders,
    incremental_schemas: bool,
    pending_reasoning: Text = .{},
    // A provider receipt combines invocation and result in one public item.
    // Hold its result until adjacent call items finish, preserving the same
    // parallel assistant-call group as the private live transcript.
    pending_results: std.ArrayList(c.chat_msg) = .empty,
    pending_call_ids: std.ArrayList([]u8) = .empty,
    pending_program: [*c]u8 = null,

    fn deinit(self: *Decoder) void {
        self.pending_reasoning.deinit();
        for (self.pending_results.items) |*message| messages.freeMessage(message);
        self.pending_results.deinit(allocator);
        self.clearCallIds();
        self.pending_call_ids.deinit(allocator);
        c.free(self.pending_program);
    }
    fn clearCallIds(self: *Decoder) void {
        for (self.pending_call_ids.items) |id| allocator.free(id);
        self.pending_call_ids.clearRetainingCapacity();
    }
    fn sameCallGroup(self: *const Decoder, call_id: [*c]const u8) bool {
        for (self.pending_call_ids.items) |id| if (std.mem.eql(u8, id, bytes(call_id))) return true;
        return false;
    }
    fn flushResults(self: *Decoder) void {
        for (self.pending_results.items) |message| pushResult(self.list, message);
        self.pending_results.clearRetainingCapacity();
        self.clearCallIds();
    }
    fn flushReasoning(self: *Decoder) void {
        if (self.pending_reasoning.list.items.len == 0) return;
        var message = std.mem.zeroes(c.chat_msg);
        message.role = duplicate("assistant");
        message.content = duplicate("");
        message.reasoning = self.pending_reasoning.take();
        messages.pushMessage(self.list, message);
    }
    fn appendReasoning(self: *Decoder, value: [*c]const u8) void {
        if (bytes(value).len == 0) return;
        if (self.pending_reasoning.list.items.len != 0) self.pending_reasoning.byte('\n');
        self.pending_reasoning.append(bytes(value));
    }
    fn appendCall(self: *Decoder, call: c.tool_call) void {
        // Text and tool calls can arrive as separate assistant output items.
        // Merge adjacent assistant records so a call keeps its assistant prefix.
        const current = records(self.list);
        if (current.len != 0 and equal(current[current.len - 1].role, "assistant")) {
            const last = &current[current.len - 1];
            if (self.pending_reasoning.list.items.len != 0 and bytes(last.reasoning).len == 0) {
                c.free(last.reasoning);
                last.reasoning = self.pending_reasoning.take();
            }
            messages.pushCall(&last.calls, call);
        } else {
            var message = std.mem.zeroes(c.chat_msg);
            message.role = duplicate("assistant");
            message.content = duplicate("");
            if (self.pending_reasoning.list.items.len != 0) message.reasoning = self.pending_reasoning.take();
            messages.pushCall(&message.calls, call);
            messages.pushMessage(self.list, message);
        }
    }
    fn ordinaryCall(self: *Decoder, item: *const Item) void {
        var call = std.mem.zeroes(c.tool_call);
        call.id = duplicate(bytes(item.callId()));
        // custom input is free text; the renderer falls back to a text param
        // when this argument blob is not a JSON object.
        call.arguments = duplicate(if (item.arguments != null) bytes(item.arguments) else if (item.input != null) bytes(item.input) else "{}");
        if (!equal(item.typeName(), "custom_tool_call") and bytes(item.namespace).len != 0 and bytes(item.name).len != 0) {
            var qualified = Text{};
            defer qualified.deinit();
            qualified.append(bytes(item.namespace));
            qualified.append(bytes(item.name));
            call.name = qualified.take();
        } else call.name = duplicate(bytes(item.name));
        self.appendCall(call);
    }
    fn hostedCall(self: *Decoder, item: *const Item) void {
        const kind = item.typeName();
        var call = std.mem.zeroes(c.tool_call);
        call.id = duplicate(bytes(item.callId()));
        call.name = duplicate(if (equal(kind, "tool_search_call")) "tool_search" else if (equal(kind, "local_shell_call")) "local_shell" else bytes(kind));
        call.arguments = duplicate(if (item.action != null) bytes(item.action) else if (item.arguments != null) bytes(item.arguments) else if (item.input != null) bytes(item.input) else "{}");
        // Hosted tools need not be registered now. Preserve prior actions as
        // function-call-shaped history when clients replay their output items.
        self.appendCall(call);
    }
    fn selectCallGroup(self: *Decoder, group: []const []const u8) Invalid!void {
        if (self.pending_call_ids.items.len == 0) {
            for (group) |id| self.pending_call_ids.append(allocator, allocator.dupe(u8, id) catch fatal()) catch fatal();
        } else {
            if (group.len != self.pending_call_ids.items.len) return error.InvalidJson;
            for (group, self.pending_call_ids.items) |id, pending| if (!std.mem.eql(u8, id, pending)) return error.InvalidJson;
        }
    }
    fn decisionReceipt(self: *Decoder, item: *const Item) Invalid!void {
        try self.selectCallGroup(item.call_ids.?.value);
        var call = std.mem.zeroes(c.tool_call);
        call.id = duplicate(bytes(item.call_id));
        call.name = duplicate("ds4_decide");
        call.arguments = duplicate(bytes(item.arguments));
        self.appendCall(call);
        var message = std.mem.zeroes(c.chat_msg);
        message.role = duplicate("tool");
        message.content = duplicate(bytes(item.result));
        messages.addCallId(&message, item.call_id);
        self.pending_results.append(allocator, message) catch fatal();
    }
    fn programCall(self: *Decoder, item: *const Item) Invalid!void {
        try self.selectCallGroup(item.program.call_ids.?.value);
        var call = std.mem.zeroes(c.tool_call);
        call.id = duplicate(bytes(item.call_id));
        call.name = duplicate("ds4_exec");
        call.arguments = duplicate(bytes(item.program.arguments));
        self.appendCall(call);
        self.pending_program = duplicate(bytes(item.call_id));
    }
    fn programOutput(self: *Decoder, item: *const Item) Invalid!void {
        if (self.pending_program == null or !equal(self.pending_program, bytes(item.call_id))) return error.InvalidJson;
        c.free(self.pending_program);
        self.pending_program = null;
        var message = std.mem.zeroes(c.chat_msg);
        message.role = duplicate("tool");
        message.content = duplicate(bytes(item.result));
        messages.addCallId(&message, item.call_id);
        self.pending_results.append(allocator, message) catch fatal();
    }
    fn toolOutput(self: *Decoder, item: *Item) void {
        var message = std.mem.zeroes(c.chat_msg);
        message.role = duplicate("tool");
        message.content = if (item.output != null) item.output else duplicate("");
        item.output = null;
        message.images = item.output_images;
        item.output_images = std.mem.zeroes(c.server_image_inputs);
        if (item.call_id != null or item.id != null) messages.addCallId(&message, item.callId());
        pushResult(self.list, message);
    }
    fn hostedOutput(self: *Decoder, item: *const Item) Invalid!void {
        var announcement: [*c]u8 = null;
        defer c.free(announcement);
        if (equal(item.typeName(), "tool_search_output") and item.tools_json != null and self.loaded != null and self.orders != null) {
            var cursor: [*c]const u8 = item.tools_json;
            var schemas: [*c]u8 = null;
            defer c.free(schemas);
            if (!tools.parseTools(&cursor, &schemas, self.orders)) return error.InvalidJson;
            if (bytes(schemas).len != 0) {
                if (self.incremental_schemas) {
                    var update = Text{};
                    defer update.deinit();
                    update.append("\n\n### Updated Tool Schemas\n\n");
                    update.append(bytes(schemas));
                    announcement = update.take();
                } else {
                    const loaded = self.loaded.?;
                    if (loaded.list.items.len != 0) loaded.byte('\n');
                    loaded.append(bytes(schemas));
                }
            }
        }
        var message = std.mem.zeroes(c.chat_msg);
        message.role = duplicate("tool");
        // The late system update already carries definitions. Preserve any
        // caller output/result; otherwise render a short discovery receipt.
        message.content = duplicate(if (item.output != null) bytes(item.output) else if (item.result != null) bytes(item.result) else if (announcement != null) "Tool discovery complete." else bytes(item.tools_json));
        if (item.call_id != null or item.id != null) messages.addCallId(&message, item.callId());
        pushResult(self.list, message);
        if (announcement != null) {
            var update = std.mem.zeroes(c.chat_msg);
            update.role = duplicate("system");
            update.content = announcement;
            announcement = null;
            update.discovered_tools = duplicate(bytes(item.tools_json));
            messages.pushMessage(self.list, update);
        }
    }
    fn accept(self: *Decoder, item: *Item) Invalid!void {
        try item.validate();
        const kind = item.typeName();
        const program_call = equal(kind, "program");
        const program_output = equal(kind, "program_output");
        // Native programs complete before yielding. A missing output or nested
        // client suspension cannot be silently rehydrated as completed work.
        if (self.pending_program != null and !program_output) return error.InvalidJson;
        const decision_call = equal(kind, "ds4:decision_call");
        const hosted_call = equal(kind, "local_shell_call") or equal(kind, "web_search_call") or equal(kind, "tool_search_call") or equal(kind, "image_generation_call");
        const ordinary_call = equal(kind, "function_call") or equal(kind, "custom_tool_call");
        if ((!decision_call and !hosted_call and !ordinary_call and !program_call and !program_output) or
            (self.pending_results.items.len != 0 and !self.sameCallGroup(item.callId()))) self.flushResults();
        const bookkeeping = equal(kind, "compaction") or equal(kind, "context_compaction");
        const consumes_reasoning = (equal(kind, "message") and equal(item.role, "assistant")) or ordinary_call or hosted_call or decision_call or program_call or program_output;
        // User/tool/other items flush in-position. Bookkeeping is transparent.
        // This also preserves the old boundary between consecutive reasoning
        // items rather than coalescing them into a different history shape.
        if (!consumes_reasoning and !bookkeeping) self.flushReasoning();
        if (equal(kind, "message")) {
            var message = std.mem.zeroes(c.chat_msg);
            message.role = duplicate(if (item.role != null) bytes(item.role) else "user");
            message.content = if (item.content != null) item.content else duplicate("");
            item.content = null;
            message.images = item.content_images;
            item.content_images = std.mem.zeroes(c.server_image_inputs);
            if (equal(message.role, "assistant") and self.pending_reasoning.list.items.len != 0) message.reasoning = self.pending_reasoning.take();
            messages.pushMessage(self.list, message);
        } else if (ordinary_call) {
            self.ordinaryCall(item);
        } else if (decision_call) {
            try self.decisionReceipt(item);
        } else if (program_call) {
            try self.programCall(item);
        } else if (program_output) {
            try self.programOutput(item);
        } else if (equal(kind, "function_call_output") or equal(kind, "custom_tool_call_output")) {
            self.toolOutput(item);
        } else if (equal(kind, "reasoning")) {
            // Visible summary/content is replayable. Encrypted reasoning is an
            // ignored extension here; the later C validator records fidelity.
            self.appendReasoning(item.summary);
            self.appendReasoning(item.content);
        } else if (hosted_call) {
            self.hostedCall(item);
        } else if (equal(kind, "local_shell_call_output") or equal(kind, "web_search_call_output") or equal(kind, "tool_search_output") or equal(kind, "tool_search_call_output") or equal(kind, "image_generation_call_output")) {
            try self.hostedOutput(item);
        } else if (!bookkeeping) {
            // Unknown items could silently discard history. Only the two true
            // compaction bookkeeping types pass without a logical record.
            return error.InvalidJson;
        }
    }
    fn read(self: *Decoder, reader: *Reader) Invalid!void {
        try reader.consume('[');
        while (reader.more(']')) {
            var item = Item{};
            defer item.deinit();
            try item.read(reader);
            try self.accept(&item);
            reader.separator();
        }
        try reader.consume(']');
        if (self.pending_program != null) return error.InvalidJson;
        self.flushResults();
        // Reasoning-only tails still need an assistant <think> block.
        self.flushReasoning();
    }
};

pub fn parseInput(cursor: *[*c]const u8, list: *c.chat_msgs, loaded_schemas: [*c][*c]u8, orders: [*c]c.tool_schema_orders, incremental: bool) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    var loaded = Text{};
    if (loaded_schemas != null and loaded_schemas[0] != null) {
        // The old C adapter borrowed this sole owner with capacity strlen+1.
        // Its non-NULL base remains valid even when the string is empty.
        const count = bytes(loaded_schemas[0]).len;
        loaded.list.items = @as([*]u8, @ptrCast(loaded_schemas[0]))[0..count];
        loaded.list.capacity = count + 1;
    }
    defer {
        // Publish partial appended schemas on error too, preserving one owner.
        if (loaded_schemas != null) loaded_schemas[0] = loaded.take();
        loaded.deinit();
    }
    var decoder = Decoder{
        .list = list,
        .loaded = if (loaded_schemas != null) &loaded else null,
        .orders = orders,
        .incremental_schemas = incremental,
    };
    defer decoder.deinit();
    decoder.read(&reader) catch return false;
    return true;
}

/// Rehydrate private declarations without relocating their visible schemas.
/// Current explicit tools override historical metadata regardless of key order.
pub fn restoreDiscovered(list: *c.chat_msgs, orders: [*c]c.tool_schema_orders, tool_choice_none: bool, explicit_tools: [*c]const u8) c_int {
    var found: c_int = 0;
    var index: usize = 0;
    while (index < @as(usize, @intCast(list.len))) {
        const current = records(list);
        if (current[index].discovered_tools == null) {
            index += 1;
            continue;
        }
        if (tool_choice_none) {
            messages.freeMessage(&current[index]);
            std.mem.copyForwards(c.chat_msg, current[index .. current.len - 1], current[index + 1 ..]);
            list.len -= 1;
            continue;
        }
        var cursor: [*c]const u8 = current[index].discovered_tools;
        var schemas: [*c]u8 = null;
        const valid = tools.parseTools(&cursor, &schemas, orders);
        if (valid and bytes(schemas).len != 0) found = 1;
        c.free(schemas);
        if (!valid) return -1;
        index += 1;
    }
    if (explicit_tools != null) {
        if (!tools.overlayRequestTools(explicit_tools, orders)) return -1;
    }
    return found;
}

pub fn ds4_server_json_responses_input(cursor: *[*c]const u8, list: *c.chat_msgs, loaded_schemas: [*c][*c]u8, orders: [*c]c.tool_schema_orders, incremental: bool) bool {
    return parseInput(cursor, list, loaded_schemas, orders, incremental);
}
pub fn ds4_server_json_responses_content(cursor: *[*c]const u8, output: *[*c]u8, images: [*c]c.server_image_inputs) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    output.* = contentValue(&reader, images) catch return false;
    return true;
}
pub fn ds4_server_input_restore_discovered_tools(list: *c.chat_msgs, orders: [*c]c.tool_schema_orders, tool_choice_none: bool, explicit_tools: [*c]const u8) c_int {
    return restoreDiscovered(list, orders, tool_choice_none, explicit_tools);
}
