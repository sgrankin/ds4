//! Select the representation of one XML value. XGrammar validates schemas and
//! output; this helper disambiguates raw strings from JSON scalars/containers.
const std = @import("std");
const Value = std.json.Value;
pub const Kind = enum { string, json, ambiguous };

pub fn references(schema: Value) bool {
    if (schema != .object) return false;
    if (schema.object.contains("$ref")) return true;
    for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |key| if (schema.object.get(key)) |branches| {
        if (branches == .array) for (branches.array.items) |branch| if (references(branch)) return true;
    };
    return false;
}

pub fn direct(schema: Value) Kind {
    if (schema != .object) return .ambiguous;
    for ([_][]const u8{ "$ref", "const", "enum", "anyOf", "oneOf", "allOf" }) |key| {
        if (schema.object.contains(key)) return .ambiguous;
    }
    const kind = schema.object.get("type") orelse return .ambiguous;
    if (kind != .string) return .ambiguous;
    if (std.mem.eql(u8, kind.string, "string")) return .string;
    for ([_][]const u8{ "null", "boolean", "number", "integer", "object", "array" }) |name| {
        if (std.mem.eql(u8, kind.string, name)) return .json;
    }
    return .ambiguous;
}
fn number(value: Value) ?f64 {
    return switch (value) {
        .integer => |integer| @floatFromInt(integer),
        .float => |float| float,
        else => null,
    };
}
fn eql(left: Value, right: Value) bool {
    if (left == .integer and right == .integer) return left.integer == right.integer;
    if (number(left)) |a| if (number(right)) |b| return a == b;
    if (std.meta.activeTag(left) != std.meta.activeTag(right)) return false;
    return switch (left) {
        .null => true,
        .bool => |a| a == right.bool,
        .string => |a| std.mem.eql(u8, a, right.string),
        .number_string => |a| std.mem.eql(u8, a, right.number_string),
        .integer, .float => unreachable,
        .array => |a| blk: {
            if (a.items.len != right.array.items.len) break :blk false;
            for (a.items, right.array.items) |item, other| if (!eql(item, other)) break :blk false;
            break :blk true;
        },
        .object => |a| blk: {
            if (a.count() != right.object.count()) break :blk false;
            var fields = a.iterator();
            while (fields.next()) |field| {
                const other = right.object.get(field.key_ptr.*) orelse break :blk false;
                if (!eql(field.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}
fn local(a: std.mem.Allocator, root: Value, reference: []const u8) !?Value {
    if (reference.len == 0 or reference[0] != '#') return null;
    const uri: std.Uri.Component = .{ .percent_encoded = reference[1..] };
    const pointer = try uri.toRawMaybeAlloc(a);
    if (pointer.len == 0) return root;
    if (pointer[0] != '/') return null;
    var current = root;
    var parts = std.mem.splitScalar(u8, pointer[1..], '/');
    while (parts.next()) |part| {
        const slash = try std.mem.replaceOwned(u8, a, part, "~1", "/");
        const key = try std.mem.replaceOwned(u8, a, slash, "~0", "~");
        current = switch (current) {
            .object => |object| object.get(key) orelse return null,
            .array => |array| blk: {
                const index = std.fmt.parseInt(usize, key, 10) catch return null;
                if (index >= array.items.len) return null;
                break :blk array.items[index];
            },
            else => return null,
        };
    }
    return current;
}
fn typeMatches(kind: []const u8, value: Value) bool {
    if (std.mem.eql(u8, kind, "string")) return value == .string;
    if (std.mem.eql(u8, kind, "null")) return value == .null;
    if (std.mem.eql(u8, kind, "boolean")) return value == .bool;
    if (std.mem.eql(u8, kind, "array")) return value == .array;
    if (std.mem.eql(u8, kind, "object")) return value == .object;
    if (std.mem.eql(u8, kind, "number")) return number(value) != null;
    if (std.mem.eql(u8, kind, "integer")) return value == .integer or (value == .float and value.float == @trunc(value.float));
    return false;
}
/// Only representation-bearing keywords belong here, not a parallel schema
/// validator. Bounds, patterns and nested object guarantees stay with XGrammar.
fn eligible(a: std.mem.Allocator, root: Value, schema: Value, value: Value, depth: usize) !bool {
    if (depth > 64) return false;
    if (schema == .bool) return schema.bool;
    if (schema != .object) return false;
    if (schema.object.get("$ref")) |reference| {
        if (reference != .string) return false;
        const target = (try local(a, root, reference.string)) orelse return false;
        if (!try eligible(a, root, target, value, depth + 1)) return false;
    }
    if (schema.object.get("type")) |kind| {
        var matches = false;
        if (kind == .string) matches = typeMatches(kind.string, value) else if (kind == .array) {
            for (kind.array.items) |item| if (item == .string and typeMatches(item.string, value)) {
                matches = true;
                break;
            };
        }
        if (!matches) return false;
    }
    if (schema.object.get("const")) |constant| if (!eql(constant, value)) return false;
    if (schema.object.get("enum")) |choices| {
        if (choices != .array) return false;
        var found = false;
        for (choices.array.items) |choice| if (eql(choice, value)) {
            found = true;
            break;
        };
        if (!found) return false;
    }
    for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |key| if (schema.object.get(key)) |branches| {
        if (branches != .array) return false;
        var matches: usize = 0;
        for (branches.array.items) |branch| if (try eligible(a, root, branch, value, depth + 1)) {
            matches += 1;
        };
        if (std.mem.eql(u8, key, "anyOf") and matches == 0) return false;
        if (std.mem.eql(u8, key, "oneOf") and matches != 1) return false;
        if (std.mem.eql(u8, key, "allOf") and matches != branches.array.items.len) return false;
    };
    return true;
}
pub fn stringValue(a: std.mem.Allocator, root: Value, schema: Value, raw: []const u8) !bool {
    const parsed = std.json.parseFromSliceLeaky(Value, a, raw, .{}) catch return true;
    const string = try eligible(a, root, schema, .{ .string = raw }, 0);
    const json = try eligible(a, root, schema, parsed, 0);
    if (string != json) return string;
    // Preserve nullable-string convention. Other ambiguous forms prefer raw
    // string spelling; either is an eligible representation when both match.
    if (string and json) return parsed != .null;
    return false;
}

const strings: u3 = 1;
const nulls: u3 = 2;
const other_json: u3 = 4;
const any_kind: u3 = strings | nulls | other_json;
const Summary = struct { kinds: u3 = any_kind, string_limits: bool = false };
fn kindBits(kind: []const u8) u3 {
    if (std.mem.eql(u8, kind, "string")) return strings;
    if (std.mem.eql(u8, kind, "null")) return nulls;
    return other_json;
}
fn valueBits(value: Value) u3 {
    return if (value == .string) strings else if (value == .null) nulls else other_json;
}
/// Conservative structural preflight, not validation of concrete values.
/// Cross-type string constraints could require validating both candidate forms;
/// reject that subset instead of guessing or building another schema validator.
fn summarize(a: std.mem.Allocator, root: Value, schema: Value, depth: usize) !Summary {
    if (depth > 64 or schema == .bool) return .{ .kinds = if (schema == .bool and !schema.bool) 0 else any_kind };
    if (schema != .object) return .{};
    var out = Summary{};
    for ([_][]const u8{ "pattern", "format", "minLength", "maxLength" }) |key| out.string_limits = out.string_limits or schema.object.contains(key);
    if (schema.object.get("type")) |kind| {
        var bits: u3 = 0;
        if (kind == .string) bits = kindBits(kind.string) else if (kind == .array) {
            for (kind.array.items) |item| if (item == .string) {
                bits |= kindBits(item.string);
            };
        }
        out.kinds &= bits;
    }
    if (schema.object.get("const")) |value| out.kinds &= valueBits(value);
    if (schema.object.get("enum")) |values| if (values == .array) {
        var bits: u3 = 0;
        for (values.array.items) |value| bits |= valueBits(value);
        out.kinds &= bits;
    };
    if (schema.object.get("$ref")) |reference| if (reference == .string) {
        if (try local(a, root, reference.string)) |target| {
            const child = try summarize(a, root, target, depth + 1);
            out.kinds &= child.kinds;
            out.string_limits = out.string_limits or child.string_limits;
        }
    };
    for ([_][]const u8{ "anyOf", "oneOf", "allOf" }) |key| if (schema.object.get(key)) |branches| {
        if (branches != .array) continue;
        const intersection = std.mem.eql(u8, key, "allOf");
        var kinds: u3 = if (intersection) any_kind else 0;
        for (branches.array.items) |branch| {
            const child = try summarize(a, root, branch, depth + 1);
            if (intersection) kinds &= child.kinds else kinds |= child.kinds;
            out.string_limits = out.string_limits or child.string_limits;
        }
        out.kinds &= kinds;
    };
    return out;
}
pub fn unsupportedStrictTools(backing: std.mem.Allocator, descriptor: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(backing);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, a, descriptor, .{});
    for (parsed.object.get("tools").?.array.items) |wrapper| {
        const function = wrapper.object.get("function").?;
        if (!function.object.get("strict").?.bool) continue;
        const parameters = function.object.get("parameters").?;
        const properties = parameters.object.get("properties") orelse continue;
        for (properties.object.values()) |property| {
            const summary = try summarize(a, parameters, property, 0);
            if (summary.string_limits and (summary.kinds & strings) != 0 and (summary.kinds & other_json) != 0) return true;
        }
    }
    return false;
}

test "constant reference, mixed enum and nullable spelling select schema value types" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try std.json.parseFromSliceLeaky(Value, a,
        \\{"$defs":{"truth":{"const":"true"}},"properties":{
        \\ "ref":{"$ref":"#/$defs/truth"},"enum":{"enum":[1,"x"]},
        \\ "nullable":{"type":["string","null"]}}}
    , .{});
    const properties = root.object.get("properties").?.object;
    try std.testing.expect(try stringValue(a, root, properties.get("ref").?, "true"));
    try std.testing.expect(!try stringValue(a, root, properties.get("enum").?, "1"));
    try std.testing.expect(try stringValue(a, root, properties.get("enum").?, "x"));
    try std.testing.expect(!try stringValue(a, root, properties.get("nullable").?, "null"));
}

test "strict Flint preflight rejects cross-type string limits through references" {
    const rejected =
        \\{"tools":[{"function":{"strict":true,"parameters":{"type":"object",
        \\ "$defs":{"limited":{"anyOf":[{"type":"string","pattern":"^x$"},{"type":"integer"}]}},
        \\ "properties":{"value":{"$ref":"#/$defs/limited"}}}}}]}
    ;
    const supported =
        \\{"tools":[{"function":{"strict":true,"parameters":{"type":"object",
        \\ "$defs":{"truth":{"const":"true"}},"properties":{
        \\ "ref":{"$ref":"#/$defs/truth"},"mixed":{"enum":[1,"x"]},
        \\ "nullable":{"type":["string","null"],"pattern":"^x$"}}}}}]}
    ;
    try std.testing.expect(try unsupportedStrictTools(std.testing.allocator, rejected));
    try std.testing.expect(!try unsupportedStrictTools(std.testing.allocator, supported));
}
