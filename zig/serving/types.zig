//! Logical serving input, independent of a wire protocol, model or C ABI.
//! Slices borrow input storage. Their owner must keep that storage alive until
//! model preparation returns; model code must not retain these views.

pub const Endpoint = enum { chat, responses, anthropic, completion };

pub const Reasoning = union(enum) {
    none,
    low,
    medium,
    high,
    max,
    level: u8,

    pub fn enabled(self: Reasoning) bool {
        return switch (self) {
            .none => false,
            .level => |value| value != 0,
            else => true,
        };
    }
};

pub const ReasoningControls = struct {
    /// Null means the client left the model's default enabled state unchanged.
    enabled: ?bool = null,
    effort: Reasoning = .high,
};

pub const Parameter = struct {
    name: []const u8,
    value: []const u8,
    is_string: bool,
};

pub const ToolCall = struct {
    id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    arguments: ?[]const u8 = null,
    /// Null retains the raw argument fallback. An empty slice is a parsed {}.
    /// Descriptors preserve parameter order, duplicate keys and value spelling.
    parameters: ?[]const Parameter = null,
};

pub const ToolResultSpan = struct {
    begin: usize,
    end: usize,
    id: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

pub const Image = struct {
    marker: []const u8,
    encoded: []const u8,
};

pub const Message = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
    reasoning: ?[]const u8 = null,
    discovered_tools: ?[]const u8 = null,
    images: []const Image = &.{},
    tool_call_id: ?[]const u8 = null,
    tool_call_ids: []const []const u8 = &.{},
    calls: []const ToolCall = &.{},
    raw_tool_text: ?[]const u8 = null,
    tool_results: []const ToolResultSpan = &.{},
};

pub const Property = struct {
    name: ?[]const u8 = null,
    schema: ?[]const u8 = null,
};

pub const ToolSchema = struct {
    name: ?[]const u8 = null,
    wire_name: ?[]const u8 = null,
    namespace: ?[]const u8 = null,
    responses_tool_search: bool = false,
    properties: []const Property = &.{},
};

pub const Request = struct {
    model: ?[]const u8 = null,
    messages: []const Message = &.{},
    tools: []const ToolSchema = &.{},
    tool_schemas: ?[]const u8 = null,
    reasoning: ReasoningControls = .{},
};
