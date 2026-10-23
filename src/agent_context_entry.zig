// The single C facade. The owner itself contains only ordinary pub Zig APIs.
const owner = @import("agent_context.zig");
const c = @import("agent_context_c");

fn errors(pointer: [*c]u8, count: usize) []u8 {
    return if (count == 0) &.{} else pointer[0..count];
}
export fn ds4_agent_context_clear(context: *c.ds4_agent_context) void {
    owner.clear(context);
}
export fn ds4_agent_context_body_text(text: [*c]const u8, len: usize, quote: bool) [*c]u8 {
    return owner.bodyText(if (len == 0) &.{} else text[0..len], quote);
}
export fn ds4_agent_context_render(engine: *c.ds4_engine, transcript: *c.ds4_tokens, system: *const c.ds4_tokens, context: *c.ds4_agent_context, length: [*c]usize) [*c]u8 {
    const view = owner.Context{ .engine = engine, .transcript = transcript, .state = context };
    return view.render(system, if (length == null) null else @ptrCast(length)) orelse null;
}
export fn ds4_agent_context_text_safe(engine: *c.ds4_engine, tokens: *const c.ds4_tokens, text: [*c]const u8, len: usize, err: [*c]u8, errlen: usize) bool {
    return owner.textSafe(engine, tokens, if (len == 0) &.{} else text[0..len], errors(err, errlen));
}
export fn ds4_agent_context_prepare(engine: *c.ds4_engine, transcript: *c.ds4_tokens, system: *const c.ds4_tokens, context: *c.ds4_agent_context, text: [*c]const u8, len: usize, images: usize, ctx_size: c_int, edit: *c.ds4_agent_context_edit, err: [*c]u8, errlen: usize) c_int {
    const view = owner.Context{ .engine = engine, .transcript = transcript, .state = context };
    const changed = view.prepare(system, if (len == 0) &.{} else text[0..len], images, ctx_size, edit, errors(err, errlen)) catch return -1;
    return @intFromBool(changed);
}
export fn ds4_agent_context_commit(transcript: *c.ds4_tokens, context: *c.ds4_agent_context, edit: *c.ds4_agent_context_edit) void {
    owner.commit(transcript, context, edit);
}
export fn ds4_agent_context_edit_free(edit: *c.ds4_agent_context_edit) void {
    owner.editFree(edit);
}
