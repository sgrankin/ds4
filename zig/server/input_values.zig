//! Tool declaration normalization over the server's actual logical records.
//! The shared JSON cursor retains the accepted grammar. This owner keeps
//! schema unwrapping, namespace names, ordered properties and libc lifetimes.
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

const Reader = struct {
    cursor: [*c]const u8,

    fn peek(self: *Reader) u8 {
        c.ds4_server_json_ws(&self.cursor);
        return self.cursor[0];
    }
    fn consume(self: *Reader, byte: u8) Invalid!void {
        if (self.peek() != byte) return error.InvalidJson;
        self.cursor += 1;
    }
    fn more(self: *Reader, closing: u8) bool {
        const byte = self.peek();
        return byte != 0 and byte != closing;
    }
    fn separator(self: *Reader) void {
        // Optional commas preserve the selective parser's existing grammar.
        if (self.peek() == ',') self.cursor += 1;
        _ = self.peek();
    }
    fn string(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_string(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
    fn raw(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_raw(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
    fn skip(self: *Reader) Invalid!void {
        if (!c.ds4_server_json_skip(&self.cursor)) return error.InvalidJson;
    }
    fn key(self: *Reader) Invalid![*c]u8 {
        const name = try self.string();
        errdefer c.free(name);
        try self.consume(':');
        return name;
    }
    fn replace(self: *Reader, target: *[*c]u8, raw_value: bool) Invalid!void {
        const replacement = if (raw_value) try self.raw() else try self.string();
        c.free(target.*);
        target.* = replacement;
    }
};

const Text = struct {
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal();
    }
    fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    fn line(self: *Text, value: [*c]const u8) void {
        const text = bytes(value);
        if (text.len == 0) return;
        if (self.list.items.len != 0) self.byte('\n');
        self.append(text);
    }
    fn quoted(self: *Text, value: []const u8) void {
        // Match json_escape's byte policy, including non-UTF8 strings and the
        // named newline/CR/tab forms. Raw parameter schemas stay unchanged.
        const hex = "0123456789abcdef";
        self.byte('"');
        for (value) |character| switch (character) {
            '"', '\\' => self.append(&.{ '\\', character }),
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            0...8, 11, 12, 14...31 => self.append(&.{ '\\', 'u', '0', '0', hex[character >> 4], hex[character & 15] }),
            else => self.byte(character),
        };
        self.byte('"');
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    }
};

fn freeOrder(order: *c.tool_schema_order) void {
    c.free(order.name);
    c.free(order.wire_name);
    c.free(order.namespace);
    for (items([*c]u8, order.prop, @intCast(order.len))) |property| c.free(property);
    for (items([*c]u8, order.prop_schema, @intCast(order.len))) |schema| c.free(schema);
    c.free(@ptrCast(order.prop));
    c.free(@ptrCast(order.prop_schema));
    order.* = std.mem.zeroes(c.tool_schema_order);
}
fn pushProperty(order: *c.tool_schema_order, property: [*c]u8, schema: [*c]u8) void {
    var names = adopted([*c]u8, order.prop, order.len, order.cap);
    var schemas = adopted([*c]u8, order.prop_schema, order.len, order.cap);
    names.append(allocator, property) catch fatal();
    schemas.append(allocator, schema) catch fatal();
    order.prop = names.items.ptr;
    order.prop_schema = schemas.items.ptr;
    order.len = @intCast(names.items.len);
    order.cap = @intCast(@min(names.capacity, schemas.capacity));
}
fn pushOrder(orders: *c.tool_schema_orders, order: c.tool_schema_order) void {
    // A later declaration replaces the complete metadata at the original slot.
    // This preserves prompt order while letting current explicit tools win.
    for (items(c.tool_schema_order, orders.v, @intCast(orders.len))) |*existing| {
        if (existing.name != null and std.mem.eql(u8, bytes(existing.name), bytes(order.name))) {
            freeOrder(existing);
            existing.* = order;
            return;
        }
    }
    var list = adopted(c.tool_schema_order, orders.v, orders.len, orders.cap);
    list.append(allocator, order) catch fatal();
    orders.v = list.items.ptr;
    orders.len = @intCast(list.items.len);
    orders.cap = @intCast(list.capacity);
}

fn properties(raw: [*c]const u8, order: *c.tool_schema_order) Invalid!void {
    var reader = Reader{ .cursor = raw };
    try reader.consume('{');
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "properties")) {
            try reader.consume('{');
            while (reader.more('}')) {
                const property = try reader.key();
                errdefer c.free(property);
                const schema = try reader.raw();
                pushProperty(order, property, schema);
                reader.separator();
            }
            try reader.consume('}');
        } else try reader.skip();
        reader.separator();
    }
    // A partial property list is retained by addOrder, exactly as the C codec.
    if (reader.peek() != '}') return error.InvalidJson;
}
fn readOrder(reader: *Reader, order: *c.tool_schema_order) Invalid!void {
    try reader.consume('{');
    while (reader.more('}')) {
        const key = try reader.key();
        defer c.free(key);
        if (equal(key, "name")) {
            try reader.replace(&order.name, false);
        } else if (equal(key, "input_schema") or equal(key, "parameters")) {
            const schema = try reader.raw();
            defer c.free(schema);
            properties(schema, order) catch {};
        } else try reader.skip();
        reader.separator();
    }
    // The old metadata helper admits a decoded name at EOF; do not add a new
    // closing-brace requirement while moving this tolerant normalization.
}
pub fn addOrder(orders: [*c]c.tool_schema_orders, raw: [*c]const u8, namespace: [*c]const u8, wire_name: [*c]const u8, hosted_search: bool) void {
    if (orders == null or raw == null) return;
    var order = std.mem.zeroes(c.tool_schema_order);
    defer freeOrder(&order);
    var reader = Reader{ .cursor = raw };
    readOrder(&reader, &order) catch return;
    if (order.name == null) return;
    if (bytes(namespace).len != 0) order.namespace = duplicate(bytes(namespace));
    if (bytes(wire_name).len != 0) order.wire_name = duplicate(bytes(wire_name));
    order.responses_tool_search = hosted_search;
    pushOrder(@ptrCast(orders), order);
    order = std.mem.zeroes(c.tool_schema_order);
}

fn unwrappedFunction(raw: [*c]const u8) [*c]u8 {
    var reader = Reader{ .cursor = raw };
    reader.consume('{') catch return null;
    while (reader.more('}')) {
        const key = reader.key() catch return null;
        defer c.free(key);
        // The first function field wins, even if later wrapper fields differ.
        if (equal(key, "function")) return reader.raw() catch null;
        reader.skip() catch return null;
        reader.separator();
    }
    return null;
}

const Definition = struct {
    kind: [*c]u8 = null,
    name: [*c]u8 = null,
    description: [*c]u8 = null,
    parameters: [*c]u8 = null,
    tools: [*c]u8 = null,

    const Form = enum { hosted, function, namespace };

    fn deinit(self: *Definition) void {
        inline for (.{ "kind", "name", "description", "parameters", "tools" }) |field| c.free(@field(self, field));
    }
    fn read(self: *Definition, raw: [*c]const u8, form: Form) Invalid!void {
        var reader = Reader{ .cursor = raw };
        try reader.consume('{');
        while (reader.more('}')) {
            const key = try reader.key();
            defer c.free(key);
            if (equal(key, "type")) {
                try reader.replace(&self.kind, false);
            } else if (form != .hosted and equal(key, "name")) {
                try reader.replace(&self.name, false);
            } else if (form != .namespace and equal(key, "description")) {
                try reader.replace(&self.description, false);
            } else if (form != .namespace and (equal(key, "parameters") or (form == .function and equal(key, "input_schema")))) {
                try reader.replace(&self.parameters, true);
            } else if (form == .namespace and equal(key, "tools")) {
                try reader.replace(&self.tools, true);
            } else try reader.skip();
            reader.separator();
        }
    }
    fn schema(self: *const Definition, name: []const u8, default_description: []const u8) [*c]u8 {
        var text = Text{};
        defer text.deinit();
        text.append("{\"name\":");
        text.quoted(name);
        text.append(",\"description\":");
        text.quoted(if (self.description != null) bytes(self.description) else default_description);
        text.append(",\"parameters\":");
        text.append(if (self.parameters != null) bytes(self.parameters) else "{\"type\":\"object\",\"properties\":{}}");
        text.byte('}');
        return text.take();
    }
};

fn hostedSchema(raw: [*c]const u8) [*c]u8 {
    var definition = Definition{};
    defer definition.deinit();
    definition.read(raw, .hosted) catch return null;
    if (!equal(definition.kind, "tool_search")) return null;
    return definition.schema("tool_search", "Search available tools.");
}
fn namespaceFunction(raw: [*c]const u8, namespace: [*c]const u8, wire_name: *[*c]u8) [*c]u8 {
    var definition = Definition{};
    defer definition.deinit();
    definition.read(raw, .function) catch return null;
    if (definition.kind != null and !equal(definition.kind, "function")) return null;
    if (namespace == null or bytes(definition.name).len == 0) return null;
    var prompt_name = Text{};
    defer prompt_name.deinit();
    prompt_name.append(bytes(namespace));
    prompt_name.append(bytes(definition.name));
    const schema = definition.schema(prompt_name.list.items, "");
    wire_name.* = duplicate(bytes(definition.name));
    return schema;
}
fn namespaceSchemas(text: *Text, orders: [*c]c.tool_schema_orders, raw: [*c]const u8) bool {
    var definition = Definition{};
    defer definition.deinit();
    definition.read(raw, .namespace) catch return false;
    if (!equal(definition.kind, "namespace") or definition.name == null or definition.tools == null) return false;
    var reader = Reader{ .cursor = definition.tools };
    reader.consume('[') catch return false;
    var appended = false;
    while (reader.more(']')) {
        const tool = reader.raw() catch return appended;
        defer c.free(tool);
        var wire_name: [*c]u8 = null;
        defer c.free(wire_name);
        const schema = namespaceFunction(tool, definition.name, &wire_name);
        defer c.free(schema);
        if (schema != null) {
            text.line(schema);
            addOrder(orders, schema, definition.name, wire_name, false);
            appended = true;
        }
        reader.separator();
    }
    // Already appended children survive a truncated namespace tool list.
    return appended;
}

/// OpenAI wraps {type:function,function:{...}}; Anthropic sends the schema
/// directly. Render one raw function schema per line. Responses namespaces are
/// flattened for DSML while their client-facing name/namespace remain in orders.
pub fn parseTools(cursor: *[*c]const u8, output: *[*c]u8, orders: [*c]c.tool_schema_orders) bool {
    var reader = Reader{ .cursor = cursor.* };
    defer cursor.* = reader.cursor;
    _ = reader.peek();
    if (c.ds4_server_json_lit(&reader.cursor, "null")) {
        output.* = duplicate("");
        return true;
    }
    reader.consume('[') catch return false;
    var schemas = Text{};
    defer schemas.deinit();
    while (reader.more(']')) {
        const raw = reader.raw() catch return false;
        defer c.free(raw);
        const function = unwrappedFunction(raw);
        defer c.free(function);
        if (function != null) {
            schemas.line(function);
            addOrder(orders, function, null, null, false);
        } else if (!namespaceSchemas(&schemas, orders, raw)) {
            const special = hostedSchema(raw);
            defer c.free(special);
            if (special != null) {
                schemas.line(special);
                // Hosted tool_search differs from a function with that name.
                addOrder(orders, special, null, null, true);
            } else {
                schemas.line(raw);
                addOrder(orders, raw, null, null, false);
            }
        }
        reader.separator();
    }
    reader.consume(']') catch return false;
    output.* = schemas.take();
    return true;
}

pub fn ds4_server_json_tools(cursor: *[*c]const u8, output: *[*c]u8, orders: [*c]c.tool_schema_orders) bool {
    return parseTools(cursor, output, orders);
}
pub fn ds4_server_input_schema_order_add(orders: [*c]c.tool_schema_orders, raw: [*c]const u8, namespace: [*c]const u8, wire_name: [*c]const u8, hosted_search: bool) void {
    addOrder(orders, raw, namespace, wire_name, hosted_search);
}
