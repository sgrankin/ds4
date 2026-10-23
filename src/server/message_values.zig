//! OpenAI/Anthropic message normalization over the actual server records.
//! The shared JSON cursor retains its grammar; image byte/marker creation
//! remains a C service. Partial messages, tool boundaries and cleanup live here.
const std = @import("std");
const c = @import("server_input_c");
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
fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    // Keep allocated empty storage when adopting its retained C capacity.
    return if (count == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
fn adopted(comptime T: type, pointer: [*c]T, count: c_int, capacity: c_int) std.ArrayList(T) {
    var list: std.ArrayList(T) = .empty;
    list.items = items(T, pointer, @intCast(count));
    list.capacity = @intCast(capacity);
    return list;
}

pub const Reader = struct {
    cursor: [*c]const u8,

    pub fn peek(self: *Reader) u8 {
        c.ds4_server_json_ws(&self.cursor);
        return self.cursor[0];
    }
    pub fn consume(self: *Reader, character: u8) Invalid!void {
        if (self.peek() != character) return error.InvalidJson;
        self.cursor += 1;
    }
    pub fn more(self: *Reader, closing: u8) bool {
        const character = self.peek();
        return character != 0 and character != closing;
    }
    pub fn literal(self: *Reader, value: [*:0]const u8) bool {
        _ = self.peek();
        return c.ds4_server_json_lit(&self.cursor, value);
    }
    pub fn separator(self: *Reader) void {
        // Match the selective parser's optional commas and cursor position.
        if (self.peek() == ',') self.cursor += 1;
        _ = self.peek();
    }
    pub fn string(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_string(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
    pub fn raw(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_raw(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
    pub fn skip(self: *Reader) Invalid!void {
        if (!c.ds4_server_json_skip(&self.cursor)) return error.InvalidJson;
    }
    pub fn key(self: *Reader) Invalid![*c]u8 {
        const name = try self.string();
        errdefer c.free(name);
        try self.consume(':');
        return name;
    }
    pub fn replace(self: *Reader, target: *[*c]u8, raw_value: bool) Invalid!void {
        const replacement = if (raw_value) try self.raw() else try self.string();
        c.free(target.*);
        target.* = replacement;
    }
};

pub const Text = struct {
    list: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    pub fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal();
    }
    pub fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    pub fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    }
    fn systemPart(self: *Text, value: []const u8) void {
        if (value.len == 0 or std.mem.startsWith(u8, value, "x-anthropic-")) return;
        if (self.list.items.len != 0 and self.list.items[self.list.items.len - 1] != '\n') self.byte('\n');
        self.append(value);
    }
    fn toolResult(self: *Text, value: []const u8) void {
        // Tool output is data, not HTML. Protect only the wrapper's closing
        // delimiter and spellings that would collide with that one escape.
        const closing = "</tool_result>";
        var position: usize = 0;
        while (position < value.len) : (position += 1) {
            const remaining = value[position..];
            var escaped_close = false;
            if (remaining[0] == '&') {
                var tail = remaining[1..];
                while (std.mem.startsWith(u8, tail, "amp;")) tail = tail[4..];
                escaped_close = std.mem.startsWith(u8, tail, "lt;/tool_result>");
            }
            if (std.mem.startsWith(u8, remaining, closing) or escaped_close) {
                self.append(if (remaining[0] == '<') "&lt;" else "&amp;");
            } else self.byte(remaining[0]);
        }
    }
};
fn appendOwned(target: *[*c]u8, value: []const u8) void {
    var text = Text{};
    defer text.deinit();
    text.append(bytes(target.*));
    text.append(value);
    c.free(target.*);
    target.* = text.take();
}
fn freeCall(call: *c.tool_call) void {
    c.free(call.id);
    c.free(call.name);
    c.free(call.arguments);
    call.* = std.mem.zeroes(c.tool_call);
}
fn freeCalls(calls: *c.tool_calls) void {
    for (items(c.tool_call, calls.v, @intCast(calls.len))) |*call| freeCall(call);
    c.free(calls.raw_tool_text);
    c.free(calls.v);
    calls.* = std.mem.zeroes(c.tool_calls);
}
pub fn freeImages(images: *c.server_image_inputs) void {
    for (items(c.server_image_input, images.v, images.len)) |image| c.free(image.encoded);
    c.free(images.v);
    images.* = std.mem.zeroes(c.server_image_inputs);
}
pub fn freeMessage(message: *c.chat_msg) void {
    inline for (.{ "role", "content", "discovered_tools", "reasoning", "tool_call_id" }) |field| c.free(@field(message, field));
    freeImages(&message.images);
    for (items([*c]u8, message.tool_call_ids, @intCast(message.tool_call_ids_len))) |id| c.free(id);
    c.free(@ptrCast(message.tool_call_ids));
    freeCalls(&message.calls);
    for (items(c.tool_result_span, message.tool_results, @intCast(message.tool_results_len))) |result| {
        c.free(result.id);
        c.free(result.content);
    }
    c.free(message.tool_results);
    message.* = std.mem.zeroes(c.chat_msg);
}
pub fn pushCall(calls: *c.tool_calls, call: c.tool_call) void {
    var list = adopted(c.tool_call, calls.v, calls.len, calls.cap);
    list.append(allocator, call) catch fatal();
    calls.v = list.items.ptr;
    calls.len = @intCast(list.items.len);
    calls.cap = @intCast(list.capacity);
}
pub fn pushMessage(messages: *c.chat_msgs, message: c.chat_msg) void {
    var list = adopted(c.chat_msg, messages.v, messages.len, messages.cap);
    list.append(allocator, message) catch fatal();
    messages.v = list.items.ptr;
    messages.len = @intCast(list.items.len);
    messages.cap = @intCast(list.capacity);
}
pub fn addCallId(message: *c.chat_msg, id: [*c]const u8) void {
    if (bytes(id).len == 0) return;
    if (message.tool_call_id == null) message.tool_call_id = duplicate(bytes(id));
    for (items([*c]u8, message.tool_call_ids, @intCast(message.tool_call_ids_len))) |existing| {
        if (existing != null and std.mem.eql(u8, bytes(existing), bytes(id))) return;
    }
    var list = adopted([*c]u8, message.tool_call_ids, message.tool_call_ids_len, message.tool_call_ids_cap);
    list.append(allocator, duplicate(bytes(id))) catch fatal();
    message.tool_call_ids = list.items.ptr;
    message.tool_call_ids_len = @intCast(list.items.len);
    message.tool_call_ids_cap = @intCast(list.capacity);
}

fn plainContent(reader: *Reader) Invalid![*c]u8 {
    if (reader.peek() == '"') return reader.string();
    if (reader.literal("null")) return duplicate("");
    if (reader.peek() != '[') {
        try reader.skip();
        return duplicate("");
    }
    try reader.consume('[');
    var text = Text{};
    defer text.deinit();
    while (reader.more(']')) {
        if (reader.peek() == '"') {
            const part = try reader.string();
            defer c.free(part);
            text.append(bytes(part));
        } else if (reader.peek() == '{') {
            try reader.consume('{');
            while (reader.more('}')) {
                const key = try reader.key();
                defer c.free(key);
                if (equal(key, "text")) {
                    const part = try reader.string();
                    defer c.free(part);
                    text.append(bytes(part));
                } else try reader.skip();
                reader.separator();
            }
            try reader.consume('}');
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume(']');
    return text.take();
}
fn replaceContent(reader: *Reader, target: *[*c]u8) Invalid!void {
    const replacement = try plainContent(reader);
    c.free(target.*);
    target.* = replacement;
}
fn functionCall(reader: *Reader, call: *c.tool_call) Invalid!void {
    try reader.consume('{');
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "name")) {
            try reader.replace(&call.name, false);
        } else if (equal(key, "arguments")) {
            try reader.replace(&call.arguments, reader.peek() != '"');
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
}
fn toolCalls(reader: *Reader, calls: *c.tool_calls) Invalid!void {
    if (reader.literal("null")) return;
    try reader.consume('[');
    while (reader.more(']')) {
        try reader.consume('{');
        var call = std.mem.zeroes(c.tool_call);
        defer freeCall(&call);
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "id")) {
                try reader.replace(&call.id, false);
            } else if (equal(key, "function")) {
                try functionCall(reader, &call);
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
        if (call.name != null and call.arguments != null) {
            pushCall(calls, call);
            call = std.mem.zeroes(c.tool_call);
        }
        reader.separator();
    }
    try reader.consume(']');
}
fn imageUrl(reader: *Reader, url: *[*c]u8) Invalid!void {
    if (reader.peek() == '"') {
        // The direct-string variant writes the output directly, as before.
        url.* = try reader.string();
        return;
    }
    try reader.consume('{');
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "url")) try reader.replace(url, false) else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
    if (url.* == null) return error.InvalidJson;
}
fn openaiObject(reader: *Reader, message: *c.chat_msg) Invalid!void {
    try reader.consume('{');
    var kind: [*c]u8 = null;
    var text: [*c]u8 = null;
    var url: [*c]u8 = null;
    defer c.free(kind);
    defer c.free(text);
    defer c.free(url);
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "type")) {
            try reader.replace(&kind, false);
        } else if (equal(key, "text")) {
            try replaceContent(reader, &text);
        } else if (equal(key, "image_url")) {
            c.free(url);
            url = null;
            try imageUrl(reader, &url);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
    if (equal(kind, "image_url") or equal(kind, "input_image")) {
        var marker: [c.SERVER_IMAGE_MARKER_BYTES]u8 = undefined;
        if (!c.ds4_server_image_data_uri(&message.images, url, &marker)) return error.InvalidJson;
        appendOwned(&message.content, std.mem.sliceTo(&marker, 0));
    } else if (text != null) appendOwned(&message.content, bytes(text));
}
fn openaiContent(reader: *Reader, message: *c.chat_msg) Invalid!void {
    if (reader.peek() == '"') {
        const text = try reader.string();
        defer c.free(text);
        appendOwned(&message.content, bytes(text));
        return;
    }
    if (reader.literal("null")) return;
    try reader.consume('[');
    while (reader.more(']')) {
        if (reader.peek() == '"') {
            const text = try reader.string();
            defer c.free(text);
            appendOwned(&message.content, bytes(text));
        } else if (reader.peek() == '{') {
            try openaiObject(reader, message);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume(']');
}

const Block = struct {
    kind: [*c]u8 = null,
    text: [*c]u8 = null,
    thinking: [*c]u8 = null,
    id: [*c]u8 = null,
    name: [*c]u8 = null,
    input: [*c]u8 = null,
    result: [*c]u8 = null,
    source_kind: [*c]u8 = null,
    media_type: [*c]u8 = null,
    image_data: [*c]u8 = null,

    fn deinit(self: *Block) void {
        inline for (comptime std.meta.fieldNames(Block)) |field| c.free(@field(self, field));
    }
    fn source(self: *Block, reader: *Reader) Invalid!void {
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "type")) {
                try reader.replace(&self.source_kind, false);
            } else if (equal(key, "media_type")) {
                try reader.replace(&self.media_type, false);
            } else if (equal(key, "data")) {
                try reader.replace(&self.image_data, false);
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
    }
    fn read(self: *Block, reader: *Reader) Invalid!void {
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "type")) {
                try reader.replace(&self.kind, false);
            } else if (equal(key, "text")) {
                try replaceContent(reader, &self.text);
            } else if (equal(key, "thinking")) {
                try replaceContent(reader, &self.thinking);
            } else if (equal(key, "id") or equal(key, "tool_use_id")) {
                try reader.replace(&self.id, false);
            } else if (equal(key, "name")) {
                try reader.replace(&self.name, false);
            } else if (equal(key, "input")) {
                try reader.replace(&self.input, true);
            } else if (equal(key, "content")) {
                try reader.replace(&self.result, true);
            } else if (equal(key, "source")) {
                try self.source(reader);
            } else try reader.skip();
            reader.separator();
        }
        try reader.consume('}');
    }
};

fn anthropicBlock(reader: *Reader, message: *c.chat_msg, allow_tools: bool) Invalid!void {
    var block = Block{};
    defer block.deinit();
    try block.read(reader);
    // Member order is not meaningful: content can precede the enclosing role.
    // Classify blocks by their own type, then validate the final message role.
    if (!allow_tools and !equal(block.kind, "text") and !equal(block.kind, "image")) return error.InvalidJson;
    if (equal(block.kind, "tool_use")) {
        pushCall(&message.calls, .{
            .id = if (block.id != null) duplicate(bytes(block.id)) else null,
            .name = duplicate(bytes(block.name)),
            .arguments = duplicate(if (block.input != null) bytes(block.input) else "{}"),
        });
    } else if (equal(block.kind, "tool_result")) {
        var nested = std.mem.zeroes(c.chat_msg);
        defer freeMessage(&nested);
        var content = Reader{ .cursor = if (block.result != null) block.result else "\"\"" };
        try anthropicContent(&content, &nested, false);
        addCallId(message, block.id);
        var text = Text{};
        defer text.deinit();
        text.append(bytes(message.content));
        const begin = text.list.items.len;
        text.append("<tool_result>");
        text.toolResult(bytes(nested.content));
        text.append("</tool_result>");
        // Boundaries come from parsed blocks, not markup ordinary text can
        // also contain. V4.1 uses these spans to order and separate results.
        const count: usize = @intCast(message.tool_results_len);
        const spans = allocator.realloc(items(c.tool_result_span, message.tool_results, count), count + 1) catch fatal();
        spans[count] = .{
            .begin = begin,
            .end = text.list.items.len,
            .id = if (block.id != null) duplicate(bytes(block.id)) else null,
            .content = nested.content,
        };
        message.tool_results = spans.ptr;
        message.tool_results_len += 1;
        nested.content = null;
        c.free(message.content);
        message.content = text.take();
        if (nested.images.len != 0) {
            const total = message.images.len + nested.images.len;
            const images = allocator.realloc(items(c.server_image_input, message.images.v, message.images.len), total) catch fatal();
            @memcpy(images[message.images.len..], items(c.server_image_input, nested.images.v, nested.images.len));
            message.images.v = images.ptr;
            message.images.len = total;
            message.images.cap = total;
            nested.images.len = 0;
        }
    } else if (equal(block.kind, "image")) {
        var marker: [c.SERVER_IMAGE_MARKER_BYTES]u8 = undefined;
        if (!equal(block.source_kind, "base64") or !c.ds4_server_image_base64(&message.images, block.media_type, block.image_data, &marker)) return error.InvalidJson;
        appendOwned(&message.content, std.mem.sliceTo(&marker, 0));
    } else {
        if (block.text != null) appendOwned(&message.content, bytes(block.text));
        if (block.thinking != null) appendOwned(&message.reasoning, bytes(block.thinking));
    }
}

/// Anthropic blocks collapse text/thinking into strings and assistant tool_use
/// into calls. Tool results retain both escaped text and their parsed spans.
fn anthropicContent(reader: *Reader, message: *c.chat_msg, allow_tools: bool) Invalid!void {
    if (reader.peek() == '"') {
        message.content = try reader.string();
        return;
    }
    if (reader.literal("null")) {
        message.content = duplicate("");
        return;
    }
    if (reader.peek() != '[') {
        if (allow_tools) return reader.skip();
        return error.InvalidJson;
    }
    try reader.consume('[');
    while (reader.more(']')) {
        if (reader.peek() == '"') {
            const text = try reader.string();
            defer c.free(text);
            appendOwned(&message.content, bytes(text));
        } else if (reader.peek() == '{') {
            try anthropicBlock(reader, message, allow_tools);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume(']');
    if (message.content == null) message.content = duplicate("");
}

fn readMessage(reader: *Reader, anthropic: bool, trusted_store: bool) Invalid!c.chat_msg {
    try reader.consume('{');
    var message = std.mem.zeroes(c.chat_msg);
    errdefer freeMessage(&message);
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "role")) {
            try reader.replace(&message.role, false);
        } else if (equal(key, "content")) {
            if (anthropic) {
                // A duplicate would leave stale block offsets or tool IDs.
                if (message.content != null or message.calls.len != 0 or message.tool_results_len != 0) return error.InvalidJson;
                try anthropicContent(reader, &message, true);
            } else {
                c.free(message.content);
                message.content = null;
                freeImages(&message.images);
                try openaiContent(reader, &message);
            }
        } else if (!anthropic and equal(key, "reasoning_content")) {
            try replaceContent(reader, &message.reasoning);
        } else if (!anthropic and equal(key, "tool_call_id")) {
            const id = try reader.string();
            defer c.free(id);
            addCallId(&message, id);
        } else if (!anthropic and equal(key, "tool_calls")) {
            freeCalls(&message.calls);
            try toolCalls(reader, &message.calls);
        } else if (!anthropic and trusted_store and equal(key, "raw_tool_text")) {
            try reader.replace(&message.calls.raw_tool_text, false);
        } else if (!anthropic and trusted_store and equal(key, "ds4_discovered_tools")) {
            try reader.replace(&message.discovered_tools, true);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
    if (message.role == null) message.role = duplicate("user");
    if (message.content == null) message.content = duplicate("");
    if (message.images.len != 0 and !equal(message.role, "user")) return error.InvalidJson;
    return message;
}
fn messagesValue(reader: *Reader, messages: *c.chat_msgs, anthropic: bool, trusted_store: bool) Invalid!void {
    try reader.consume('[');
    while (reader.more(']')) {
        pushMessage(messages, try readMessage(reader, anthropic, trusted_store));
        reader.separator();
    }
    try reader.consume(']');
}
fn systemObject(reader: *Reader, text: *Text) Invalid!void {
    try reader.consume('{');
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "text")) {
            const part = try reader.string();
            defer c.free(part);
            text.systemPart(bytes(part));
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume('}');
}
fn systemValue(reader: *Reader) Invalid![*c]u8 {
    var text = Text{};
    defer text.deinit();
    if (reader.peek() == '"') {
        const part = try reader.string();
        defer c.free(part);
        text.systemPart(bytes(part));
        return text.take();
    }
    if (reader.literal("null")) return duplicate("");
    if (reader.peek() != '[') {
        try reader.skip();
        return duplicate("");
    }
    try reader.consume('[');
    while (reader.more(']')) {
        if (reader.peek() == '"') {
            const part = try reader.string();
            defer c.free(part);
            text.systemPart(bytes(part));
        } else if (reader.peek() == '{') {
            try systemObject(reader, &text);
        } else try reader.skip();
        reader.separator();
    }
    try reader.consume(']');
    return text.take();
}

pub fn ds4_server_json_messages(cursor: *[*c]const u8, messages: *c.chat_msgs, trusted_store: bool) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    messagesValue(&reader, messages, false, trusted_store) catch return false;
    return true;
}
pub fn ds4_server_json_anthropic_messages(cursor: *[*c]const u8, messages: *c.chat_msgs) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    messagesValue(&reader, messages, true, false) catch return false;
    return true;
}
pub fn ds4_server_json_anthropic_system(cursor: *[*c]const u8, output: *[*c]u8) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    output.* = systemValue(&reader) catch return false;
    return true;
}
pub fn ds4_server_json_content(cursor: *[*c]const u8, output: *[*c]u8) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    output.* = plainContent(&reader) catch return false;
    return true;
}
pub fn ds4_server_json_function_call(cursor: *[*c]const u8, call: *c.tool_call) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    functionCall(&reader, call) catch return false;
    return true;
}
pub fn ds4_server_json_tool_calls(cursor: *[*c]const u8, calls: *c.tool_calls) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    toolCalls(&reader, calls) catch return false;
    return true;
}
pub fn ds4_server_json_image_url(cursor: *[*c]const u8, output: *[*c]u8) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    imageUrl(&reader, output) catch return false;
    return true;
}
