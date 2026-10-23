//! Sole C facade; importing the pure planner does not duplicate exports.
const std = @import("std");
const c = @import("agent_compaction_c");
const owner = @import("agent_compaction.zig");

fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
fn errors(value: [*c]u8, len: usize) []u8 {
    return if (len == 0) &.{} else value[0..len];
}
fn images(value: [*c]const c.ds4_vision_span, count: usize) []const c.ds4_vision_span {
    return if (count == 0) &.{} else @as([*]const c.ds4_vision_span, @ptrCast(value))[0..count];
}
export fn ds4_agent_compaction_should_compact(engine: *c.ds4_engine, ctx: c_int, used: c_int) bool {
    return owner.shouldCompact(engine, ctx, used);
}
export fn ds4_agent_compaction_tail_boundary(tokens: *const c.ds4_tokens, bottom: c_int, system_len: c_int, budget: c_int, user: c_int) c_int {
    return owner.tailBoundary(tokens, bottom, system_len, budget, user);
}
export fn ds4_agent_compaction_summary_budget(ctx: c_int) c_int {
    return owner.summaryBudget(ctx);
}
export fn ds4_agent_compaction_reserve(engine: *c.ds4_engine, ctx: c_int) c_int {
    return owner.reserve(engine, ctx);
}
export fn ds4_agent_compaction_image_boundary(value: [*c]const c.ds4_vision_span, count: usize, glm: bool, position: c_int) c_int {
    return owner.imageBoundary(images(value, count), glm, position);
}
export fn ds4_agent_compaction_prepare(plan: *c.ds4_agent_compaction_plan, engine: *c.ds4_engine, transcript: *const c.ds4_tokens, system: *const c.ds4_tokens, value: [*c]const c.ds4_vision_span, count: usize, ctx: c_int, reason: [*c]const u8, err: [*c]u8, errlen: usize) bool {
    return (owner.Plan{ .data = plan, .engine = engine, .transcript = transcript, .system = system, .images = images(value, count), .ctx = ctx }).prepare(text(reason), errors(err, errlen));
}
export fn ds4_agent_compaction_rebuild(plan: *c.ds4_agent_compaction_plan, engine: *c.ds4_engine, transcript: *const c.ds4_tokens, system: *const c.ds4_tokens, value: [*c]const c.ds4_vision_span, count: usize, ctx: c_int, prerope: bool, summary: [*c]const u8, open: [*c]const c_int, err: [*c]u8, errlen: usize) bool {
    return (owner.Plan{ .data = plan, .engine = engine, .transcript = transcript, .system = system, .images = images(value, count), .ctx = ctx }).rebuild(text(summary), prerope, if (open == null) null else open[0], errors(err, errlen));
}
export fn ds4_agent_compaction_free(plan: *c.ds4_agent_compaction_plan) void {
    owner.free(plan);
}
