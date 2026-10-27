//! One request contract for prompt schemas and native constrained decoding.
//! JSON ownership is an arena per preparation; the returned strings own their
//! bytes. The compiler remains the authority for schema/regex compilation.
const std = @import("std");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

pub const Failure = error{
    InvalidSchema,
    UnsupportedSchema,
    InvalidToolChoice,
    UnknownTool,
    ThinkingToolChoice,
    InvalidResponseFormat,
    IncompatibleConstraints,
    DuplicateToolName,
    OutOfMemory,
};
pub const Choice = enum { auto, none, required, named };
pub const Prepared = struct {
    prompt: [:0]u8,
    constraint: ?[:0]u8,
    required: bool,
    pub fn deinit(self: Prepared, allocator: Allocator) void {
        allocator.free(self.prompt);
        if (self.constraint) |value| allocator.free(value);
    }
};
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn str(value: ?Value) ?[]const u8 {
    return if (value) |v| (if (v == .string) v.string else null) else null;
}
fn object(allocator: Allocator) Value {
    _ = allocator;
    return .{ .object = .empty };
}
fn array(allocator: Allocator) Value {
    return .{ .array = std.array_list.Managed(Value).init(allocator) };
}
fn put(allocator: Allocator, value: *Value, key: []const u8, item: Value) Failure!void {
    try value.object.put(allocator, try allocator.dupe(u8, key), item);
}
fn parse(allocator: Allocator, text: []const u8) Failure!Value {
    return std.json.parseFromSliceLeaky(Value, allocator, text, .{ .allocate = .alloc_always }) catch return error.InvalidSchema;
}

/// Reject guarantees that this pinned compiler explicitly ignores. This is a
/// supported-subset check, not another schema validator. Annotations are fine.
fn supported(node: Value) Failure!void {
    if (node == .bool) return;
    if (node != .object) return error.InvalidSchema;
    const allowed = [_][]const u8{
        "$schema",          "$id",              "$ref",        "$defs",         "definitions",          "$comment",
        "type",             "title",            "description", "default",       "examples",             "deprecated",
        "readOnly",         "writeOnly",        "properties",  "required",      "additionalProperties", "items",
        "prefixItems",      "minItems",         "maxItems",    "minProperties", "maxProperties",        "anyOf",
        "oneOf",            "allOf",            "enum",        "const",         "minimum",              "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "multipleOf",  "minLength",     "maxLength",            "pattern",
        "format",
    };
    var it = node.object.iterator();
    while (it.next()) |entry| {
        var known = false;
        for (allowed) |keyword| if (eq(keyword, entry.key_ptr.*)) {
            known = true;
            break;
        };
        if (!known) return error.UnsupportedSchema;
    }
    // The converter selects these branches before their siblings. Permit
    // annotations/definitions, but do not promise ignored sibling constraints.
    for ([_][]const u8{ "$ref", "anyOf", "oneOf", "allOf", "const", "enum" }) |selector| {
        if (!node.object.contains(selector)) continue;
        for (node.object.keys()) |key| {
            if (eq(key, selector)) continue;
            if (std.mem.startsWith(u8, key, "$")) continue;
            var annotation = false;
            for ([_][]const u8{ "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly", "definitions" }) |name| if (eq(name, key)) {
                annotation = true;
                break;
            };
            if (annotation) continue;
            if (eq(key, "type") and (eq(selector, "const") or eq(selector, "enum"))) continue;
            return error.UnsupportedSchema;
        }
    }
    // These cases compile with a warning and a weaker grammar in v0.2.8.
    // Reject the promise instead of duplicating the upstream converter.
    for ([_][]const u8{ "allOf", "oneOf" }) |keyword| {
        if (node.object.get(keyword)) |branches| {
            if (branches != .array) return error.InvalidSchema;
            if (branches.array.items.len != 1) return error.UnsupportedSchema;
        }
    }
    if (node.object.contains("pattern") and node.object.contains("format")) return error.UnsupportedSchema;
    if (node.object.get("multipleOf")) |multiple| {
        const divisor: f64 = switch (multiple) {
            .integer => @floatFromInt(multiple.integer),
            .float => multiple.float,
            .number_string => std.fmt.parseFloat(f64, multiple.number_string) catch return error.InvalidSchema,
            else => return error.InvalidSchema,
        };
        if (divisor < 1 or divisor > 1024 or @trunc(divisor) != divisor) return error.UnsupportedSchema;
    }
    if ((node.object.contains("pattern") or node.object.contains("format")) and (node.object.contains("minLength") or node.object.contains("maxLength"))) return error.UnsupportedSchema;
    if (node.object.contains("multipleOf") and !eq(str(node.object.get("type")) orelse "", "integer")) return error.UnsupportedSchema;
    for ([_][]const u8{ "properties", "$defs", "definitions" }) |key| {
        if (node.object.get(key)) |children| {
            if (children != .object) return error.InvalidSchema;
            for (children.object.values()) |child| try supported(child);
        }
    }
    for ([_][]const u8{ "items", "additionalProperties" }) |key| {
        if (node.object.get(key)) |child| try supported(child);
    }
    for ([_][]const u8{ "anyOf", "oneOf", "allOf", "prefixItems" }) |key| {
        if (node.object.get(key)) |children| {
            if (children != .array) return error.InvalidSchema;
            for (children.array.items) |child| try supported(child);
        }
    }
}

/// Recipe strictness: every object lists all properties and closes additions.
/// Preserve standard definitions/references and learned descriptions as written.
fn normalize(allocator: Allocator, node: *Value) Failure!void {
    if (node.* == .bool) return;
    if (node.* != .object) return error.InvalidSchema;
    const kind = str(node.object.get("type"));
    if ((kind != null and eq(kind.?, "object")) or node.object.contains("properties")) {
        const properties = node.object.get("properties") orelse object(allocator);
        if (properties != .object) return error.InvalidSchema;
        if (node.object.get("additionalProperties")) |extra| {
            if (extra != .bool or extra.bool) return error.InvalidSchema;
        }
        if (node.object.get("required")) |required| {
            if (required != .array or required.array.items.len != properties.object.count()) return error.InvalidSchema;
            var seen: std.StringHashMap(void) = .init(allocator);
            for (required.array.items) |name| {
                if (name != .string or !properties.object.contains(name.string) or seen.contains(name.string)) return error.InvalidSchema;
                try seen.put(name.string, {});
            }
        }
        var required = array(allocator);
        for (properties.object.keys()) |name| try required.array.append(.{ .string = name });
        try put(allocator, node, "properties", properties);
        try put(allocator, node, "required", required);
        try put(allocator, node, "additionalProperties", .{ .bool = false });
    }
    for ([_][]const u8{ "properties", "$defs", "definitions" }) |key| {
        if (node.object.getPtr(key)) |children| {
            if (children.* != .object) return error.InvalidSchema;
            for (children.object.values()) |*child| try normalize(allocator, child);
        }
    }
    for ([_][]const u8{ "items", "additionalProperties" }) |key| if (node.object.getPtr(key)) |child| try normalize(allocator, child);
    for ([_][]const u8{ "anyOf", "oneOf", "allOf", "prefixItems" }) |key| {
        if (node.object.getPtr(key)) |children| {
            if (children.* != .array) return error.InvalidSchema;
            for (children.array.items) |*child| try normalize(allocator, child);
        }
    }
}

fn strictFlag(value: Value) Failure!bool {
    const flag = value.object.get("strict") orelse return false;
    if (flag != .bool) return error.InvalidSchema;
    return flag.bool;
}

pub fn prepare(backing: Allocator, schemas: []const u8, choice_json: ?[]const u8, format_json: ?[]const u8, thinking: bool) Failure!Prepared {
    var arena: std.heap.ArenaAllocator = .init(backing);
    defer arena.deinit();
    const allocator = arena.allocator();
    var choice: Choice = .auto;
    var selected: ?[]const u8 = null;
    if (choice_json) |raw| {
        const value = try parse(allocator, raw);
        if (value == .string) {
            choice = if (eq(value.string, "auto")) .auto else if (eq(value.string, "none")) .none else if (eq(value.string, "required")) .required else return error.InvalidToolChoice;
        } else if (value == .object) {
            const kind = str(value.object.get("type")) orelse return error.InvalidToolChoice;
            if (eq(kind, "none")) choice = .none else if (eq(kind, "auto")) choice = .auto else if (eq(kind, "any") or eq(kind, "required")) choice = .required else if (eq(kind, "function") or eq(kind, "tool")) {
                choice = .named;
                selected = str(value.object.get("name"));
                if (value.object.get("function")) |function| {
                    if (function != .object) return error.InvalidToolChoice;
                    selected = str(function.object.get("name"));
                }
                if (selected == null or selected.?.len == 0) return error.InvalidToolChoice;
                if (str(value.object.get("namespace"))) |namespace| selected = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ namespace, selected.? });
            } else return error.InvalidToolChoice;
        } else return error.InvalidToolChoice;
    }
    const required = choice == .required or choice == .named;
    if (required and thinking) return error.ThinkingToolChoice;

    var tools = array(allocator);
    var lines: std.ArrayList(u8) = .empty;
    var names: std.StringHashMap(void) = .init(allocator);
    var strict_count: usize = 0;
    var selected_found = false;
    var iterator = std.mem.tokenizeScalar(u8, schemas, '\n');
    while (iterator.next()) |line| {
        var tool = try parse(allocator, line);
        if (tool != .object) return error.InvalidSchema;
        const name = str(tool.object.get("name")) orelse return error.InvalidSchema;
        if (name.len == 0 or std.mem.findAny(u8, name, "\"<>\r\n") != null) return error.InvalidSchema;
        if (names.contains(name)) return error.DuplicateToolName;
        try names.put(name, {});
        if (selected) |wanted| {
            if (!eq(name, wanted)) continue;
            selected_found = true;
        }
        if (choice == .none) continue;
        const strict = try strictFlag(tool);
        var parameters = tool.object.get("parameters") orelse tool.object.get("input_schema") orelse try parse(allocator, "{\"type\":\"object\",\"properties\":{}}");
        if (parameters != .object or !eq(str(parameters.object.get("type")) orelse "", "object")) return error.InvalidSchema;
        if (parameters.object.get("properties")) |properties| {
            if (properties != .object) return error.InvalidSchema;
            for (properties.object.keys()) |key| {
                if (key.len == 0 or std.mem.findAny(u8, key, "\"<>\r\n") != null or std.mem.find(u8, key, "｜DSML｜") != null) return error.UnsupportedSchema;
            }
        }
        if (strict) {
            try supported(parameters);
            try normalize(allocator, &parameters);
            strict_count += 1;
        }
        try put(allocator, &tool, "parameters", parameters);
        _ = tool.object.swapRemove("input_schema");
        // Prompt and compiler consume the same normalized schema, while the
        // explicit flag is retained for the compiler's non-strict policy.
        try put(allocator, &tool, "strict", .{ .bool = strict });
        if (lines.items.len != 0) try lines.append(allocator, '\n');
        var prompt_tool = object(allocator);
        for ([_][]const u8{ "name", "description", "parameters" }) |key| {
            if (tool.object.get(key)) |value| try put(allocator, &prompt_tool, key, value);
        }
        try lines.appendSlice(allocator, try std.json.Stringify.valueAlloc(allocator, prompt_tool, .{}));
        var wrapper = object(allocator);
        try put(allocator, &wrapper, "type", .{ .string = "function" });
        try put(allocator, &wrapper, "function", tool);
        try tools.array.append(wrapper);
    }
    if (selected != null and !selected_found) return error.UnknownTool;
    if (required and tools.array.items.len == 0) return error.InvalidToolChoice;
    if (strict_count != 0 and strict_count != tools.array.items.len) return error.IncompatibleConstraints;

    var format: Value = .null;
    if (format_json) |raw| {
        format = try parse(allocator, raw);
        if (format != .object) return error.InvalidResponseFormat;
        const kind = str(format.object.get("type")) orelse return error.InvalidResponseFormat;
        if (eq(kind, "text")) format = .null else if (eq(kind, "json_object")) {} else if (eq(kind, "regex")) {
            if (str(format.object.get("pattern")) == null) return error.InvalidResponseFormat;
        } else if (eq(kind, "json_schema")) {
            if (format.object.get("json_schema")) |inner| {
                if (inner != .object) return error.InvalidResponseFormat;
                format = inner;
                try put(allocator, &format, "type", .{ .string = "json_schema" });
            }
            const schema = format.object.getPtr("schema") orelse return error.InvalidResponseFormat;
            try supported(schema.*);
            if (try strictFlag(format)) {
                if (schema.* != .object or !eq(str(schema.object.get("type")) orelse "", "object")) return error.InvalidResponseFormat;
                try normalize(allocator, schema);
            }
        } else return error.InvalidResponseFormat;
    }
    if (format != .null and strict_count != 0) return error.IncompatibleConstraints;
    var descriptor = object(allocator);
    try put(allocator, &descriptor, "tools", tools);
    try put(allocator, &descriptor, "required", .{ .bool = required });
    try put(allocator, &descriptor, "thinking", .{ .bool = thinking });
    // Like recipe, active tools take precedence over a non-strict answer format.
    try put(allocator, &descriptor, "format", if (tools.array.items.len == 0) format else .null);
    const prompt = try backing.dupeSentinel(u8, lines.items, 0);
    errdefer backing.free(prompt);
    const constraint = if (tools.array.items.len == 0 and format == .null and choice != .none) null else try backing.dupeSentinel(u8, try std.json.Stringify.valueAlloc(allocator, descriptor, .{}), 0);
    return .{ .prompt = prompt, .constraint = constraint, .required = required };
}

pub fn message(failure: Failure) []const u8 {
    return switch (failure) {
        error.ThinkingToolChoice => "Thinking mode does not support required or named tool_choice",
        error.UnknownTool => "Named tool_choice does not name a declared function",
        error.InvalidToolChoice => "Invalid tool_choice or required choice without available tools",
        error.IncompatibleConstraints => "Active strict tools require all tools to be strict and cannot combine with a response format",
        error.UnsupportedSchema => "Schema uses constraints unsupported by the native grammar compiler",
        error.DuplicateToolName => "Tool declarations contain a duplicate or colliding model function name",
        error.InvalidResponseFormat => "Invalid or unsupported structured response format",
        error.InvalidSchema => "Invalid tool/schema declaration: function parameters must be an object schema",
        error.OutOfMemory => "Out of memory while preparing output constraints",
    };
}

test "strict normalization is shared by prompt and constraint and retains definitions" {
    const a = std.testing.allocator;
    const result = try prepare(a, "{\"name\":\"inspect\",\"strict\":true,\"parameters\":{\"type\":\"object\",\"properties\":{\"value\":{\"$ref\":\"#/$defs/item\"}},\"$defs\":{\"item\":{\"type\":\"string\"}}}}", null, null, true);
    defer result.deinit(a);
    const prompt = try std.json.parseFromSlice(Value, a, result.prompt, .{});
    defer prompt.deinit();
    const contract = try std.json.parseFromSlice(Value, a, result.constraint.?, .{});
    defer contract.deinit();
    const schema = prompt.value.object.get("parameters").?;
    try std.testing.expect(schema.object.contains("$defs"));
    try std.testing.expect(!schema.object.get("additionalProperties").?.bool);
    try std.testing.expectEqualStrings("value", schema.object.get("required").?.array.items[0].string);
    const compiler_schema = contract.value.object.get("tools").?.array.items[0].object.get("function").?.object.get("parameters").?;
    const left = try std.json.Stringify.valueAlloc(a, schema, .{});
    defer a.free(left);
    const right = try std.json.Stringify.valueAlloc(a, compiler_schema, .{});
    defer a.free(right);
    try std.testing.expectEqualStrings(left, right);
}

test "omitted strict stays non-strict and named choices narrow declarations" {
    const a = std.testing.allocator;
    const tools = "{\"name\":\"a\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}\n{\"name\":\"files::read\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}";
    const result = try prepare(a, tools, "{\"type\":\"function\",\"namespace\":\"files\",\"name\":\"read\"}", null, false);
    defer result.deinit(a);
    try std.testing.expect(result.required);
    try std.testing.expect(std.mem.find(u8, result.prompt, "\"name\":\"a\"") == null);
    const descriptor = try std.json.parseFromSlice(Value, a, result.constraint.?, .{});
    defer descriptor.deinit();
    try std.testing.expect(!descriptor.value.object.get("tools").?.array.items[0].object.get("function").?.object.get("strict").?.bool);
    try std.testing.expectError(error.ThinkingToolChoice, prepare(a, tools, "\"required\"", null, true));
    const none = try prepare(a, tools, "\"none\"", null, true);
    defer none.deinit(a);
    try std.testing.expect(none.prompt.len == 0 and none.constraint != null);
}

test "ignored schema guarantees and incompatible promises fail clearly" {
    const a = std.testing.allocator;
    const tool = "{\"name\":\"a\",\"strict\":true,\"parameters\":{\"type\":\"object\",\"properties\":{}}}";
    try std.testing.expectError(error.IncompatibleConstraints, prepare(a, tool, null, "{\"type\":\"json_object\"}", false));
    try std.testing.expectError(error.UnsupportedSchema, prepare(a, "", null, "{\"type\":\"json_schema\",\"schema\":{\"type\":\"array\",\"uniqueItems\":true,\"items\":{\"type\":\"string\"}}}", false));
}
