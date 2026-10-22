//! Sole C facade. The pure Generation owner contains no exported copies.
const c = @import("server_generation_c");
const owner = @import("generation.zig");

export fn ds4_server_generation_run(runtime: *c.server, slot: *c.server_slot, job: *c.job, stop: [*c]const volatile c.sig_atomic_t) void {
    (owner.Generation{ .runtime = runtime, .slot = slot, .job = job, .stop = stop }).run();
}
export fn ds4_server_generation_progress(packet: [*c]c.server_prefill_progress, event: [*c]const u8, current: c_int, total: c_int, stop: [*c]const volatile c.sig_atomic_t) void {
    owner.progress(packet, event, current, total, stop);
}
export fn ds4_server_generation_thinking_feed(state: [*c]c.thinking_state, value: [*c]const u8, length: usize) void {
    owner.feedThinking(state, value, length);
}
export fn ds4_server_generation_thinking_prompt(req: [*c]const c.request) c.thinking_state {
    return owner.thinkingFromPrompt(req);
}
export fn ds4_server_generation_reasoning_due(limit: c_int, count: c_int, forced: bool, inside: bool, output_left: c_int, context_left: c_int, safe: bool) bool {
    return owner.reasoningDue(limit, count, forced, inside, output_left, context_left, safe);
}
export fn ds4_server_generation_complete_thinking(syntax: c.server_model_syntax, value: [*c]const u8, length: usize, cursor: [*c]usize) bool {
    return owner.completeThinking(syntax, value, length, cursor);
}
export fn ds4_server_generation_remember_thinking(req: [*c]const c.request, state: [*c]const c.thinking_state, finish: [*c]const u8) bool {
    return owner.rememberThinking(req, state, finish);
}
export fn ds4_server_generation_canonical_needed(runtime: [*c]const c.server, calls: [*c]const c.tool_calls) bool {
    return owner.canonicalizeNeeded(runtime, calls);
}
export fn ds4_server_generation_multimodal_frontier(live: c_int, common: c_int, length: c_int, matches: bool) c_int {
    return owner.multimodalFrontier(live, common, length, matches);
}
export fn ds4_server_generation_decode_budget(requested: c_int, generated: c_int, room: c_int) c_int {
    return owner.decodeBudget(requested, generated, room);
}
export fn ds4_server_generation_assign_ids(runtime: [*c]c.server, calls: [*c]c.tool_calls, api: c.api_style) void {
    owner.assignIds(runtime, calls, api);
}
export fn ds4_server_generation_assistant_suffix(req: ?*const c.request, content: [*c]const u8, reasoning: [*c]const u8, calls: [*c]const c.tool_calls, visible: bool) [*c]u8 {
    return owner.assistantSuffix(req, content, reasoning, calls, visible);
}
export fn ds4_server_generation_thinking_visible(req: ?*const c.request, content: [*c]const u8) [*c]u8 {
    return owner.thinkingVisible(req, content);
}
export fn ds4_server_generation_qwen_tool_visible(req: ?*const c.request, finish: [*c]const u8, inside: bool, content: [*c]const u8, calls: [*c]const c.tool_calls) [*c]u8 {
    return owner.qwenToolVisible(req, finish, inside, content, calls);
}
export fn ds4_server_generation_invalid_suffix(req: ?*const c.request, state: [*c]const c.thinking_state, detail: [*c]const u8) [*c]u8 {
    return owner.invalidSuffix(req, state, detail);
}
export fn ds4_server_generation_find_stop(stops: *const c.stop_list, value: [*c]const u8, from: usize, position: *usize, length: *usize) bool {
    return owner.findStop(stops, value, from, position, length);
}
export fn ds4_server_generation_stream_safe(stops: *const c.stop_list, length: usize) usize {
    return owner.streamSafe(stops, length);
}
export fn ds4_server_generation_structured_stream(req: *const c.request) bool {
    return owner.structuredStream(req);
}
