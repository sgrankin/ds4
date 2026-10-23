//! Construct the complete protected system/tool/prefix token surface. Built-in
//! control examples are trusted; user -sys and the live-file instructions are
//! appended as ordinary system message content.
const std = @import("std");
const c = @import("agent_runtime_c");
const Prompts = @import("prompts.zig");
const Prefix = @import("prompt_prefix_append");
const allocator = std.heap.c_allocator;

pub fn build(worker: *c.agent_worker, output: *c.ds4_tokens) void {
    const config: *const c.agent_config = @ptrCast(worker.cfg);
    c.ds4_chat_begin(worker.engine, output);
    const thinking = c.ds4_think_mode_for_context(config.gen.think_mode, config.gen.ctx_size);
    if (c.ds4_engine_is_qwen4(worker.engine)) {
        const effort = c.ds4_qwen4_reasoning_effort_text(thinking);
        if (effort != null) c.ds4_chat_append_message(worker.engine, output, "system", effort);
    } else c.ds4_chat_append_think_prefix(worker.engine, output, thinking);
    Prompts.appendSystem(worker.engine, output, config.gen.system, config.edit_upto);
    Prefix.append(c, worker.engine, output, &config.gen.prefix);
    if (config.gen.live_context_path == null) return;

    var instructions: std.ArrayList(u8) = .empty;
    defer instructions.deinit(allocator);
    instructions.appendSlice(allocator, "Live context file: ") catch fatal();
    instructions.appendSlice(allocator, std.mem.span(config.gen.live_context_path)) catch fatal();
    const rules: [*c]const u8 = c.agent_live_context_instructions;
    instructions.appendSlice(allocator, std.mem.span(rules)) catch fatal();
    instructions.append(allocator, 0) catch fatal();
    c.ds4_chat_append_message(worker.engine, output, "system", instructions.items.ptr);
}

fn fatal() noreturn {
    c.perror("ds4-agent: malloc");
    c.exit(1);
}
