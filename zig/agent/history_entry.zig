//! Sole C facade. The imported owners have no duplicate C exports.
const c = @import("agent_presentation_c");
const owner = @import("history.zig");

export fn ds4_agent_history_render(engine: ?*c.ds4_engine, syntax: c.agent_tool_syntax, context: ?*anyopaque, emit: c.ds4_agent_presentation_emit_fn, use_color: bool, bytes: [*c]const u8, length: usize, user_turns: c_int) void {
    const span: []const u8 = if (length == 0) &.{} else bytes[0..length];
    (owner.History{ .engine = engine, .syntax = syntax, .context = context, .emit = emit, .use_color = use_color }).render(span, user_turns);
}
