//! Sole tool-result C facade; owner is pure and shares the actual Worker ABI.
const c = @import("agent_runtime_c");
const Transaction = @import("result.zig").Transaction;
export fn ds4_agent_result_after_tools(worker: *c.agent_worker, observation: *c.agent_tool_observation) c_int {
    return @backingInt((Transaction{ .worker = worker }).afterTools(observation));
}
export fn ds4_agent_result_append(worker: *c.agent_worker, observation: *c.agent_tool_observation, error_text: [*c]u8, capacity: usize) c_int {
    const errors: []u8 = if (error_text == null or capacity == 0) &.{} else error_text[0..capacity];
    return @backingInt((Transaction{ .worker = worker }).append(observation, errors));
}

export fn ds4_agent_result_commit(worker: *c.agent_worker, observation: *const c.agent_tool_observation, error_text: [*c]u8, capacity: usize) bool {
    const errors: []u8 = if (error_text == null or capacity == 0) &.{} else error_text[0..capacity];
    return (Transaction{ .worker = worker }).commit(observation, errors);
}
