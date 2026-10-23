//! Sole transitional C facade; edit.zig is the reusable transaction owner.
const c = @import("agent_tool_actions_c");
const Edit = @import("edit.zig");
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
export fn ds4_agent_edit_write(call: *const c.agent_tool_call) [*c]u8 {
    return Edit.write(call);
}
export fn ds4_agent_edit_apply(allow_upto: bool, call: *const c.agent_tool_call) [*c]u8 {
    return Edit.apply(allow_upto, call);
}
export fn ds4_agent_edit_preflight(allow_upto: bool, call: *const c.agent_tool_call, error_text: [*c]u8, error_size: usize) bool {
    return Edit.preflight(allow_upto, call, errors(error_text, error_size));
}
export fn ds4_agent_edit_force_upto(forcer: ?*c.agent_edit_upto_forcer, parser: ?*c.agent_dsml_parser, next_text: [*c]const u8, next_length: usize) bool {
    return Edit.forceUpto(forcer, parser, if (next_length == 0) &.{} else next_text[0..next_length]);
}
