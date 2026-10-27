//! Model framing around XGrammar's schema converter. Controls are token IDs;
//! ordinary tokens spelling the same bytes remain payload. The native bridge
//! supplies the two token-aware XML styles without another schema compiler.
const std = @import("std");
const Value = std.json.Value;
const A = std.mem.Allocator;
pub const markers = [_][]const u8{
    "<think>",           "</think>",
    "｜DSML｜",
    "<tool_call>",       "</tool_call>",
    "<arg_key>",         "</arg_key>",
    "<arg_value>",       "</arg_value>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜System｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|begin_of_text|>", "<|end_of_text|>",
    "<|system|>",        "<|developer|>",
    "<|user|>",          "<|assistant|>",
    "<|observation|>",
};
pub const Ids = [markers.len]i32;
fn s(value: []const u8) Value {
    return .{ .string = value };
}
fn obj() Value {
    return .{ .object = .empty };
}
fn arr(a: A, values: []const Value) !Value {
    var result: Value = .{ .array = std.json.Array.init(a) };
    try result.array.appendSlice(values);
    return result;
}
fn field(a: A, value: *Value, key: []const u8, data: Value) !void {
    try value.object.put(a, key, data);
}
fn node(a: A, kind: []const u8) !Value {
    var value = obj();
    try field(a, &value, "type", s(kind));
    return value;
}
fn constant(a: A, bytes: []const u8) !Value {
    var value = try node(a, "const_string");
    try field(a, &value, "value", s(bytes));
    return value;
}
fn token(a: A, id: i32) !Value {
    if (id < 0) return error.MissingControlToken;
    var value = try node(a, "token");
    try field(a, &value, "token", .{ .integer = id });
    return value;
}
fn group(a: A, kind: []const u8, values: []const Value) !Value {
    var value = try node(a, kind);
    try field(a, &value, "elements", try arr(a, values));
    return value;
}
fn unary(a: A, kind: []const u8, content: Value) !Value {
    var value = try node(a, kind);
    try field(a, &value, "content", content);
    return value;
}
fn free(a: A, ids: Ids) !Value {
    var value = try node(a, "any_tokens");
    var excluded = try arr(a, &.{});
    for (ids) |id| if (id >= 0) try excluded.array.append(.{ .integer = id });
    try field(a, &value, "exclude_tokens", excluded);
    return value;
}
fn syntax(a: A, bytes: []const u8, ids: Ids) !Value {
    var parts = try arr(a, &.{});
    var position: usize = 0;
    var plain: usize = 0;
    while (position < bytes.len) {
        var found = false;
        for (markers, ids) |spelling, id| {
            if (id < 0 or !std.mem.startsWith(u8, bytes[position..], spelling)) continue;
            if (plain != position) try parts.array.append(try constant(a, bytes[plain..position]));
            try parts.array.append(try token(a, id));
            position += spelling.len;
            plain = position;
            found = true;
            break;
        }
        if (!found) position += 1;
    }
    if (plain != bytes.len) try parts.array.append(try constant(a, bytes[plain..]));
    return group(a, "sequence", parts.array.items);
}
fn schema(a: A, value: Value, style: []const u8) !Value {
    var result = try node(a, "json_schema");
    try field(a, &result, "json_schema", value);
    try field(a, &result, "style", s(style));
    return result;
}
fn reasoning(a: A, ids: Ids, flint: bool) !Value {
    const body = try group(a, "sequence", &.{ try free(a, ids), try token(a, ids[1]) });
    return if (flint) unary(a, "optional", try group(a, "sequence", &.{ try token(a, ids[0]), body })) else body;
}

pub fn build(a: A, descriptor: Value, flint: bool, ids: Ids) ![]u8 {
    const tools = descriptor.object.get("tools").?.array.items;
    const required = descriptor.object.get("required").?.bool;
    const answer = descriptor.object.get("format").?;
    var elements = try arr(a, &.{});
    if (flint or descriptor.object.get("thinking").?.bool) try elements.array.append(try reasoning(a, ids, flint));
    if (tools.len != 0) {
        var variants = try arr(a, &.{});
        for (tools) |wrapper| {
            const function = wrapper.object.get("function").?;
            const name = function.object.get("name").?.string;
            const strict = function.object.get("strict").?.bool;
            const parameters: Value = if (strict) function.object.get("parameters").? else .{ .bool = true };
            const prefix = if (flint)
                try std.fmt.allocPrint(a, "<tool_call>{s}", .{name})
            else
                try std.fmt.allocPrint(a, "<｜DSML｜ invoke name=\"{s}\">\n", .{name});
            try variants.array.append(try group(a, "sequence", &.{
                try syntax(a, prefix, ids),
                try schema(a, parameters, if (flint) "ds4_glm_xml" else "ds4_deepseek41_xml"),
                try syntax(a, if (flint) "</tool_call>" else "</｜DSML｜ invoke>\n", ids),
            }));
        }
        const parallel = if (descriptor.object.get("parallel")) |v| v.bool else true;
        const choices = try group(a, "or", variants.array.items);
        const calls = if (parallel) try unary(a, "plus", choices) else choices;
        const block = if (flint)
            try group(a, "sequence", &.{ calls, try token(a, ids[20]) })
        else
            try group(a, "sequence", &.{ try syntax(a, "\n\n<｜DSML｜ calls>\n", ids), calls, try syntax(a, "</｜DSML｜ calls>", ids) });
        if (!required) try elements.array.append(try free(a, ids));
        try elements.array.append(if (required) block else try unary(a, "optional", block));
    } else if (answer != .null) {
        const kind = answer.object.get("type").?.string;
        if (std.mem.eql(u8, kind, "json_object")) {
            const value = try std.json.parseFromSliceLeaky(Value, a, "{\"type\":\"object\"}", .{});
            try elements.array.append(try schema(a, value, "json"));
        } else if (std.mem.eql(u8, kind, "json_schema")) {
            try elements.array.append(try schema(a, answer.object.get("schema").?, "json"));
        } else {
            var regex = try node(a, "regex");
            try field(a, &regex, "pattern", answer.object.get("pattern").?);
            try elements.array.append(regex);
        }
    } else try elements.array.append(try free(a, ids));
    var result = try node(a, "structural_tag");
    try field(a, &result, "format", try group(a, "sequence", elements.array.items));
    return std.json.Stringify.valueAlloc(a, result, .{});
}
