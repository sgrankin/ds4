//! Compatibility boundary from the existing wire records to native serving
//! input. The arena owns descriptors and normalized parameter text only;
//! message, image and schema bytes continue to belong to the decoded request.
//! Deinitialize this view before releasing or mutating those request records.
const std = @import("std");
const types = @import("serving_types");

pub fn text(pointer: [*c]const u8) ?[]const u8 {
    return if (pointer == null) null else std.mem.span(pointer);
}

pub fn reasoning(comptime c: type, mode: c.ds4_think_mode) types.Reasoning {
    return switch (mode) {
        c.DS4_THINK_NONE => .none,
        c.DS4_THINK_LOW => .low,
        c.DS4_THINK_MEDIUM => .medium,
        c.DS4_THINK_HIGH => .high,
        c.DS4_THINK_MAX => .max,
        else => if (mode >= c.DS4_THINK_LEVEL_BASE and mode <= c.DS4_THINK_LEVEL_BASE + 100)
            .{ .level = @intCast(mode - c.DS4_THINK_LEVEL_BASE) }
        else
            .none,
    };
}

pub fn reasoningMode(comptime c: type, value: types.Reasoning) c.ds4_think_mode {
    return switch (value) {
        .none => c.DS4_THINK_NONE,
        .low => c.DS4_THINK_LOW,
        .medium => c.DS4_THINK_MEDIUM,
        .high => c.DS4_THINK_HIGH,
        .max => c.DS4_THINK_MAX,
        .level => |level| @intCast(c.DS4_THINK_LEVEL_BASE + @as(c_int, level)),
    };
}

pub fn controls(comptime c: type, input: *const c.ds4_server_input) types.ReasoningControls {
    return .{
        .enabled = if (input.got_thinking) input.thinking_enabled else null,
        .effort = reasoning(c, input.reasoning_effort),
    };
}

fn items(comptime T: type, pointer: [*c]const T, length: usize) []const T {
    return if (length == 0) &.{} else pointer[0..length];
}

pub const View = struct {
    arena: std.heap.ArenaAllocator,
    messages: []const types.Message = &.{},
    tools: []const types.ToolSchema = &.{},

    /// c is the caller's generated binding module, preserving C type identity.
    pub fn init(backing: std.mem.Allocator, comptime c: type, messages: ?*const c.chat_msgs, orders: ?*const c.tool_schema_orders) std.mem.Allocator.Error!View {
        var view: View = .{ .arena = .init(backing) };
        errdefer view.deinit();
        if (messages) |source| view.messages = try view.messageViews(c, source);
        if (orders) |source| view.tools = try view.toolViews(c, source);
        return view;
    }

    pub fn deinit(self: *View) void {
        self.arena.deinit();
        self.messages = &.{};
        self.tools = &.{};
    }

    /// Initial tool-call order is another borrowed vector. Nullable C and Zig
    /// pointers are accepted so render and materialization owners share this.
    pub fn strings(self: *View, source: anytype) std.mem.Allocator.Error![]const []const u8 {
        return switch (@typeInfo(@TypeOf(source))) {
            .optional => if (source) |value| self.strings(value) else &.{},
            .null => &.{},
            else => block: {
                if (comptime @typeInfo(@TypeOf(source)).pointer.size == .c) {
                    if (source == null) break :block &.{};
                }
                break :block try self.stringViews(source.*.v, @intCast(source.*.len));
            },
        };
    }

    fn stringViews(self: *View, pointers: [*c][*c]u8, length: usize) std.mem.Allocator.Error![]const []const u8 {
        const result = try self.arena.allocator().alloc([]const u8, length);
        for (items([*c]u8, pointers, length), result) |value, *out| out.* = text(value) orelse "";
        return result;
    }

    fn parameters(self: *View, comptime c: type, raw: [*c]const u8) std.mem.Allocator.Error!?[]const types.Parameter {
        var parsed = std.mem.zeroes(c.json_args);
        // Keep the established tolerant parser: a DOM round-trip would change
        // duplicate keys, raw number spelling and accepted trailing text.
        const accepted = c.ds4_server_render_arguments_parse(raw, &parsed);
        defer c.ds4_server_render_arguments_free(&parsed);
        if (!accepted) return null;
        const arena = self.arena.allocator();
        const result = try arena.alloc(types.Parameter, @intCast(parsed.len));
        for (items(c.json_arg, parsed.v, result.len), result) |value, *out| out.* = .{
            .name = try arena.dupe(u8, text(value.key) orelse ""),
            .value = try arena.dupe(u8, text(value.value) orelse ""),
            .is_string = value.is_string,
        };
        return result;
    }

    fn messageViews(self: *View, comptime c: type, source: *const c.chat_msgs) std.mem.Allocator.Error![]const types.Message {
        const arena = self.arena.allocator();
        const result = try arena.alloc(types.Message, @intCast(source.len));
        for (items(c.chat_msg, source.v, result.len), result) |message, *out| {
            const calls = try arena.alloc(types.ToolCall, @intCast(message.calls.len));
            const raw_tool_text = text(message.calls.raw_tool_text);
            const raw_replay = if (raw_tool_text) |raw| raw.len != 0 else false;
            for (items(c.tool_call, message.calls.v, calls.len), calls) |call, *converted| converted.* = .{
                .id = text(call.id),
                .name = text(call.name),
                .arguments = text(call.arguments),
                .parameters = if (raw_replay) null else try self.parameters(c, call.arguments),
            };
            const images = try arena.alloc(types.Image, message.images.len);
            for (items(c.server_image_input, message.images.v, images.len), images) |*image, *converted| converted.* = .{
                .marker = std.mem.sliceTo(&image.marker, 0),
                .encoded = items(u8, image.encoded, image.encoded_len),
            };
            const results = try arena.alloc(types.ToolResultSpan, @intCast(message.tool_results_len));
            for (items(c.tool_result_span, message.tool_results, results.len), results) |value, *converted| converted.* = .{
                .begin = value.begin,
                .end = value.end,
                .id = text(value.id),
                .content = text(value.content),
            };
            out.* = .{
                .role = text(message.role),
                .content = text(message.content),
                .reasoning = text(message.reasoning),
                .discovered_tools = text(message.discovered_tools),
                .images = images,
                .tool_call_id = text(message.tool_call_id),
                .tool_call_ids = try self.stringViews(message.tool_call_ids, @intCast(message.tool_call_ids_len)),
                .calls = calls,
                .raw_tool_text = raw_tool_text,
                .tool_results = results,
            };
        }
        return result;
    }

    fn toolViews(self: *View, comptime c: type, source: *const c.tool_schema_orders) std.mem.Allocator.Error![]const types.ToolSchema {
        const arena = self.arena.allocator();
        const result = try arena.alloc(types.ToolSchema, @intCast(source.len));
        for (items(c.tool_schema_order, source.v, result.len), result) |order, *out| {
            const properties = try arena.alloc(types.Property, @intCast(order.len));
            for (properties, 0..) |*property, index| property.* = .{
                .name = text(order.prop[index]),
                .schema = if (order.prop_schema == null) null else text(order.prop_schema[index]),
            };
            out.* = .{
                .name = text(order.name),
                .wire_name = text(order.wire_name),
                .namespace = text(order.namespace),
                .responses_tool_search = order.responses_tool_search,
                .properties = properties,
            };
        }
        return result;
    }
};
