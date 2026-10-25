//! Server-owned PTC over fresh QuickJS contexts and the existing decision tool.
//! Bound data/templates and native call audit never enter the generator's
//! logical result; only explicit text() output does. No client-tool suspension.
const std = @import("std");
const Runtime = @import("program_runtime.zig");
const Decisions = @import("hosted_decisions.zig");
const a = std.heap.c_allocator;
const Value = std.json.Value;

pub const tool_type = "programmatic_tool_calling";
pub const function_name = "ds4_exec";

fn owned(value: anytype) ![:0]u8 {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(raw);
    return a.dupeSentinel(u8, raw, 0);
}
fn object() Value {
    return .{ .object = .empty };
}
fn put(allocator: std.mem.Allocator, target: *Value, key: []const u8, value: Value) !void {
    try target.object.put(allocator, key, value);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn section(config: Value, name: []const u8) !Value {
    const value = config.object.get(name) orelse return object();
    if (value != .object) return error.InvalidProgramBindings;
    return value;
}
fn configValue(value: Value) !Value {
    if (value != .object) return error.InvalidProgramBindings;
    for (value.object.keys()) |key| {
        if (!std.mem.eql(u8, key, "inputs") and !std.mem.eql(u8, key, "schemas") and !std.mem.eql(u8, key, "description"))
            return error.UnsupportedProgramBinding;
    }
    _ = try section(value, "inputs");
    _ = try section(value, "schemas");
    if (value.object.get("description")) |description|
        if (description != .string) return error.InvalidProgramBindings;
    return value;
}

fn manifest(allocator: std.mem.Allocator, value: Value) !Value {
    var out = object();
    try put(allocator, &out, "type", .{ .string = @tagName(value) });
    const sample = if (value == .array and value.array.items.len != 0) value.array.items[0] else value;
    if (value == .array) try put(allocator, &out, "length", .{ .integer = @intCast(value.array.items.len) });
    if (sample == .object) {
        var fields = Value{ .array = std.json.Array.init(allocator) };
        for (sample.object.keys()) |key| try fields.array.append(.{ .string = key });
        try put(allocator, &out, "fields", fields);
    }
    return out;
}

/// Only the manifest enters the tool schema. The values remain request-bound.
pub const Binding = struct {
    config: [:0]u8,
    globals: [:0]u8,
    schema: [:0]u8,

    pub fn parse(raw: []const u8) !Binding {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const allocator = arena.allocator();
        const root = try std.json.parseFromSliceLeaky(Value, allocator, raw, .{ .allocate = .alloc_always });
        if (root != .object) return error.InvalidProgramTool;
        const kind = root.object.get("type") orelse return error.InvalidProgramTool;
        if (kind != .string or !std.mem.eql(u8, kind.string, tool_type) or root.object.contains("enabled")) return error.InvalidProgramTool;
        return fromConfig(try configValue(root.object.get("ds4") orelse object()));
    }
    pub fn parseConfig(raw: []const u8) !Binding {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        return fromConfig(try configValue(try std.json.parseFromSliceLeaky(Value, arena.allocator(), raw, .{ .allocate = .alloc_always })));
    }
    fn fromConfig(config: Value) !Binding {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const allocator = arena.allocator();
        var globals = object();
        const inputs = try section(config, "inputs");
        const schemas = try section(config, "schemas");
        try put(allocator, &globals, "inputs", inputs);
        try put(allocator, &globals, "schemas", schemas);
        var summary = object();
        var input_manifest = object();
        var schema_manifest = object();
        for (inputs.object.keys(), inputs.object.values()) |name, value|
            try put(allocator, &input_manifest, name, try manifest(allocator, value));
        for (schemas.object.keys(), schemas.object.values()) |name, value| {
            var details = try manifest(allocator, value);
            if (value == .object) {
                if (value.object.get("type")) |kind| if (kind == .string and kind.string.len <= 64) {
                    try put(allocator, &details, "question_type", kind);
                };
                if (value.object.get("choices")) |choices| if (choices == .array) {
                    var labels = Value{ .array = std.json.Array.init(allocator) };
                    for (choices.array.items[0..@min(16, choices.array.items.len)]) |choice| if (choice == .object) {
                        if (choice.object.get("value")) |label| if (label == .bool or (label == .string and label.string.len <= 128)) {
                            try labels.array.append(label);
                        };
                    };
                    try put(allocator, &details, "choices", labels);
                    try put(allocator, &details, "choice_count", .{ .integer = @intCast(choices.array.items.len) });
                };
            }
            try put(allocator, &schema_manifest, name, details);
        }
        try put(allocator, &summary, "inputs", input_manifest);
        try put(allocator, &summary, "schemas", schema_manifest);
        const rendered = try std.json.Stringify.valueAlloc(allocator, summary, .{});
        var only_choices = schemas.object.count() != 0;
        for (schemas.object.values()) |value| {
            const kind = if (value == .object) value.object.get("type") else null;
            only_choices = only_choices and kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "choice");
        }
        const answer_description = if (only_choices)
            "Its result has answers:[{name,type,choice,confidence,probabilities}] and usage."
        else
            "Its result has answers with name/type: choices return choice/confidence/probabilities, predicates return probability, and scores return score/confidence/probabilities. It also has usage.";
        const emit_example = if (only_choices) "text(r.answers.map(a=>({id:a.name,choice:a.choice})));" else "text(r.answers);";
        var example: []const u8 = "";
        if (inputs.object.count() != 0 and schemas.object.count() != 0) {
            const input_name = try std.json.Stringify.valueAlloc(allocator, inputs.object.keys()[0], .{});
            const schema_name = try std.json.Stringify.valueAlloc(allocator, schemas.object.keys()[0], .{});
            const sample = inputs.object.values()[0];
            if (sample == .object and sample.object.contains("records")) {
                example = try std.fmt.allocPrint(allocator, "For a records collection, this pattern uses the existing objects: " ++
                    "const data=inputs[{s}], t=schemas[{s}]; " ++
                    "const questions=data.records.map(row=>({{...t,name:String(row.id),instructions:t.instructions+' Evaluate only record '+row.id}})); " ++
                    "const r=await tools.ds4_decide({{input:JSON.stringify(data),questions}}); " ++
                    "{s} ", .{ input_name, schema_name, emit_example });
            } else {
                example = try std.fmt.allocPrint(allocator, "Example using existing bindings: const r=await tools.ds4_decide({{input:JSON.stringify(inputs[{s}]),questions:[schemas[{s}]]}}); text(r.answers); ", .{ input_name, schema_name });
            }
        }
        const description = try std.fmt.allocPrint(allocator, "Run bounded JavaScript on the server with top-level await. Read request-bound inputs.NAME and schemas.NAME without copying their data into code. " ++
            "These globals are ALREADY POPULATED with full evidence and complete question objects; do not recreate them from the visible manifest. " ++
            "Copy question templates with object spread, preserving their type/choices. A question name is only an output label; identify its target record in instructions. " ++
            "Use await tools.ds4_decide({{questions:[...],input:stringOrInputParts}}) when that helper allows programmatic calls. " ++
            "{s} Batch questions in one call where practical. " ++
            "Serialize structured evidence with JSON.stringify. Call text(value) to emit compact JSON with required evidence IDs; return values are ignored. " ++
            "Prefer direct answers for small fully visible tasks. {s}" ++
            "No filesystem/network/process/console APIs. Native calls are synchronous; Promise.all does not parallelize them. " ++
            "Limits: 10 seconds including native work, 64 MiB JS heap, 64 KiB final output. Binding manifest: {s}. {s}", .{ answer_description, example, rendered, if (config.object.get("description")) |value| value.string else "" });
        const schema = try owned(.{ .name = function_name, .description = description, .parameters = .{ .type = "object", .properties = .{ .code = .{ .type = "string" } }, .required = .{"code"}, .additionalProperties = false } });
        errdefer a.free(schema);
        const encoded_config = try owned(config);
        errdefer a.free(encoded_config);
        return .{ .config = encoded_config, .globals = try owned(globals), .schema = schema };
    }
    pub fn deinit(self: *Binding) void {
        a.free(self.config);
        a.free(self.globals);
        a.free(self.schema);
    }
};

pub const Invocation = struct {
    status: Runtime.Status,
    output: [:0]u8,
    audit: [:0]u8,
    limit_exceeded: bool = false,
    pub fn deinit(self: *Invocation) void {
        a.free(self.output);
        a.free(self.audit);
    }
};

fn failure(code: []const u8, message: []const u8) ![:0]u8 {
    return owned(.{ .output = [0]Value{}, .@"error" = .{ .code = code, .message = message } });
}

pub fn invoke(comptime c: type, registry: ?*c.ds4_decision_registry, config_json: []const u8, model: [*c]const u8, evidence: [*c]const u8, permitted: bool, arguments: []const u8, used: *c_int, maximum: c_int, cancel: c.ds4_decision_cancel_fn, cancel_context: ?*anyopaque) !Invocation {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const allocator = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(Value, allocator, arguments, .{ .allocate = .alloc_always }) catch
        return .{ .status = .failed, .output = try failure("invalid_arguments", "Expected an object containing JavaScript code"), .audit = try a.dupeSentinel(u8, "[]", 0) };
    const code = if (parsed == .object) parsed.object.get("code") else null;
    if (code == null or code.? != .string or parsed.object.count() != 1 or code.?.string.len > 65536)
        return .{ .status = .failed, .output = try failure("invalid_arguments", "Supply only code, as a JavaScript string up to 64 KiB"), .audit = try a.dupeSentinel(u8, "[]", 0) };
    var binding = try Binding.parseConfig(config_json);
    defer binding.deinit();
    const Bridge = struct {
        registry: ?*c.ds4_decision_registry,
        model: [*c]const u8,
        evidence: [*c]const u8,
        permitted: bool,
        used: *c_int,
        maximum: c_int,
        cancel: c.ds4_decision_cancel_fn,
        cancel_context: ?*anyopaque,
        deadline: u64,
        allocator: std.mem.Allocator,
        audit: Value,
        audit_bytes: usize = 0,
        budget_exceeded: bool = false,

        fn external(self: *@This()) bool {
            return if (self.cancel) |check| check(self.cancel_context) else false;
        }
        fn stopped(context: ?*anyopaque) bool {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            return self.external() or Runtime.nowMillis() >= self.deadline;
        }
        fn nativeCancel(context: ?*anyopaque) callconv(.c) bool {
            return stopped(context);
        }
        fn call(context: ?*anyopaque, raw: []const u8) anyerror![:0]u8 {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            if (stopped(context)) return error.Cancelled;
            if (!self.permitted or self.model == null) return error.ToolNotAllowed;
            if (raw.len > 8 * 1024 * 1024) return error.ToolArgumentsTooLarge;
            if (self.used.* >= self.maximum) {
                self.budget_exceeded = true;
                return error.ToolCallLimit;
            }
            self.used.* += 1;
            const started = Runtime.nowMillis();
            var native = try Decisions.invoke(c, self.registry, text(self.model), if (self.evidence == null) null else text(self.evidence), raw, nativeCancel, self);
            defer native.deinit();
            const elapsed = Runtime.nowMillis() - started;
            // The JS heap limit cannot bound allocations made by a native tool.
            const bytes = raw.len + native.output.len + (if (native.input) |value| value.len else @as(usize, 0));
            if (self.audit_bytes + bytes > 16 * 1024 * 1024) return error.ToolAuditLimit;
            self.audit_bytes += bytes;
            var entry = object();
            try put(self.allocator, &entry, "name", .{ .string = "ds4_decide" });
            try put(self.allocator, &entry, "model", .{ .string = text(self.model) });
            try put(self.allocator, &entry, "arguments", try std.json.parseFromSliceLeaky(Value, self.allocator, raw, .{ .allocate = .alloc_always }));
            try put(self.allocator, &entry, "input", if (native.input) |value| try std.json.parseFromSliceLeaky(Value, self.allocator, value, .{ .allocate = .alloc_always }) else .null);
            try put(self.allocator, &entry, "result", try std.json.parseFromSliceLeaky(Value, self.allocator, native.output, .{ .allocate = .alloc_always }));
            try put(self.allocator, &entry, "status", .{ .integer = native.status });
            try put(self.allocator, &entry, "elapsed_ms", .{ .integer = @intCast(elapsed) });
            try self.audit.array.append(entry);
            if (stopped(context) or native.status == 499) return error.Cancelled;
            return a.dupeSentinel(u8, native.output, 0);
        }
    };
    const limits = Runtime.Limits{};
    var bridge = Bridge{ .registry = registry, .model = model, .evidence = evidence, .permitted = permitted, .used = used, .maximum = maximum, .cancel = cancel, .cancel_context = cancel_context, .deadline = Runtime.nowMillis() + limits.timeout_ms, .allocator = allocator, .audit = .{ .array = std.json.Array.init(allocator) } };
    var result = try Runtime.run(code.?.string, binding.globals, .{ .context = &bridge, .call = Bridge.call, .cancelled = Bridge.stopped }, limits);
    errdefer result.deinit();
    if (bridge.external()) result.status = .cancelled else if (bridge.budget_exceeded or Runtime.nowMillis() >= bridge.deadline) {
        const replacement = try failure(if (bridge.budget_exceeded) "max_tool_calls_exceeded" else "timeout", if (bridge.budget_exceeded) "Program exhausted the shared hosted tool-call budget" else "Program deadline exceeded");
        a.free(result.output);
        result.output = replacement;
        result.status = .failed;
    }
    return .{ .status = result.status, .output = result.output, .audit = try owned(bridge.audit), .limit_exceeded = bridge.budget_exceeded };
}
