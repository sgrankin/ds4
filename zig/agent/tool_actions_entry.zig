//! Sole C export facade for the pure tool-action and Observation owners.
const c = @import("agent_tool_actions_c");
const owner = @import("tool_actions.zig");
fn errors(pointer: [*c]u8, count: usize) []u8 {
    return if (count == 0) &.{} else pointer[0..count];
}
export fn ds4_agent_read_default_lines(context_size: c_int) c_int {
    return owner.defaultLines(context_size);
}
export fn ds4_agent_read_range(pager: *owner.Pager, context_size: c_int, path: [*c]const u8, start: c_int, count: c_int, whole: bool, bare: bool, set_more: bool, offset: c.off_t, mid_line: bool) [*c]u8 {
    return owner.readRange(pager, context_size, path, start, count, whole, bare, set_more, offset, mid_line);
}
export fn ds4_agent_tool_read(pager: *owner.Pager, context_size: c_int, call: *const owner.Call) [*c]u8 {
    return owner.read(pager, context_size, call);
}
export fn ds4_agent_tool_more(pager: *owner.Pager, context_size: c_int, call: *const owner.Call) [*c]u8 {
    return owner.more(pager, context_size, call);
}
export fn ds4_agent_tool_list(call: *const owner.Call) [*c]u8 {
    return owner.list(call);
}
export fn ds4_agent_tool_search(call: *const owner.Call) [*c]u8 {
    return owner.search(call);
}
export fn ds4_agent_observation_init(observation: *owner.Observation) void {
    owner.initObservation(observation);
}
export fn ds4_agent_observation_puts(observation: *owner.Observation, value: [*c]const u8) void {
    owner.puts(observation, value);
}
export fn ds4_agent_observation_add_image(observation: *owner.Observation, embedding: *c.ds4_vision_embedding) void {
    owner.addImage(observation, embedding);
}
export fn ds4_agent_observation_free(observation: ?*owner.Observation) void {
    owner.freeObservation(observation);
}
export fn ds4_agent_observation_execute(observation: *owner.Observation, worker: ?*anyopaque, engine: ?*c.ds4_engine, calls: *const c.agent_tool_calls) void {
    owner.execute(observation, worker, engine, calls);
}
export fn ds4_agent_observation_build(engine: ?*c.ds4_engine, transcript: *const c.ds4_tokens, observation: *const owner.Observation, tokens: *c.ds4_tokens, spans: [*c][*c]c.ds4_vision_span, error_buffer: [*c]u8, error_size: usize) bool {
    return owner.build(engine, transcript, observation, tokens, spans, errors(error_buffer, error_size));
}
export fn ds4_agent_observation_fits(engine: ?*c.ds4_engine, transcript: *const c.ds4_tokens, context_size: c_int, observation: *const owner.Observation, reserve: c_int, tokens_out: [*c]c_int, error_buffer: [*c]u8, error_size: usize) c_int {
    return owner.fits(engine, transcript, context_size, observation, reserve, tokens_out, errors(error_buffer, error_size));
}
