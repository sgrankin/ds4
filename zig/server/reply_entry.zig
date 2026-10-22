//! The sole C export facade for completed generated-reply decoding.
const c = @import("server_reply_c");
const Reply = @import("reply.zig");

export fn ds4_server_reply_tool_start(text: [*c]const u8) [*c]const u8 {
    return Reply.toolStart(text);
}
export fn ds4_server_reply_tool_end(text: [*c]const u8) [*c]const u8 {
    return Reply.toolEnd(text);
}
export fn ds4_server_reply_trim_separator(text: [*c]const u8, start: usize, limit: usize) usize {
    return Reply.trimSeparator(text, start, limit);
}
export fn ds4_server_reply_last_substring(text: [*c]const u8, needle: [*c]const u8) [*c]const u8 {
    return Reply.lastSubstring(text, needle);
}
export fn ds4_server_reply_attribute(tag: [*c]const u8, name: [*c]const u8) [*c]u8 {
    return Reply.attr(tag, name);
}
export fn ds4_server_reply_parse(syntax: c.server_model_syntax, text: [*c]const u8, require_closed: bool, content: *[*c]u8, reasoning: *[*c]u8, calls: *c.tool_calls) bool {
    return Reply.parse(syntax, text, require_closed, content, reasoning, calls);
}
export fn ds4_server_reply_repair(text: [*c]const u8, length: usize, out: *c.buf) bool {
    return Reply.repair(text, length, out);
}
export fn ds4_server_reply_for_response(syntax: c.server_model_syntax, text: [*c]const u8, has_tools: bool, saw_tool_start: bool, require_closed: bool, finish: [*c][*c]const u8, err: [*c]u8, errlen: usize, content: *[*c]u8, reasoning: *[*c]u8, calls: *c.tool_calls, recovered: [*c]bool, orders: [*c]const c.tool_schema_orders) bool {
    return Reply.parseForResponse(syntax, text, has_tools, saw_tool_start, require_closed, finish, err, errlen, content, reasoning, calls, recovered, orders);
}
