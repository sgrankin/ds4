//! Two-phase compaction text planning. C owns the private model exchange and
//! live-session transaction; this owner only constructs temporary token vectors.
const std = @import("std");
const c = @import("agent_compaction_c");
const allocator = std.heap.c_allocator;
pub const Record = c.ds4_agent_compaction_plan;

fn fatal(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn tokenSlice(tokens: *const c.ds4_tokens) []const c_int {
    return if (tokens.len == 0) &.{} else @as([*]const c_int, @ptrCast(tokens.v))[0..@intCast(tokens.len)];
}
fn appendRange(out: *c.ds4_tokens, tokens: *const c.ds4_tokens, first: c_int, last: c_int) void {
    const start: usize = @intCast(@max(0, first));
    const end: usize = @intCast(@min(last, tokens.len));
    if (start >= end) return;
    for (tokenSlice(tokens)[start..end]) |token| c.ds4_tokens_push(out, token);
}
fn specialId(engine: *c.ds4_engine, text: [*:0]const u8) c_int {
    var tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&tokens);
    c.ds4_tokenize_rendered_chat(engine, text, &tokens);
    return if (tokens.len == 1) tokens.v[0] else -1;
}

// The private request forbids tools: its result is consumed internally, never
// delivered as an assistant turn. Preserve requirements and observed facts,
// leaving unfinished work pending rather than asking the summary to solve it.
const request =
    "Internal ds4-agent context compaction request. This is not a user request.\n" ++
    "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n" ++
    "- user goals, constraints, and preferences\n" ++
    "- files inspected or edited\n" ++
    "- commands run and important results\n" ++
    "- decisions, rejected approaches, known bugs, and pending next steps\n" ++
    "- reloadable bulky data with exact paths/ranges/commands when available\n\n" ++
    "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n" ++
    "Do not solve unfinished tasks, calculate new results, or finish incomplete code in the summary. " ++
    "Record the user's requirements and what actually happened, leaving unfinished work pending.\n" ++
    "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n" ++
    "Output only the compact summary.\n";

const Text = struct {
    list: std.ArrayList(u8) = .empty,
    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, text: []const u8) void {
        self.list.appendSlice(allocator, text) catch fatal("ds4-agent: realloc");
    }
    fn take(self: *Text) [:0]u8 {
        return self.list.toOwnedSliceSentinel(allocator, 0) catch fatal("ds4-agent: malloc");
    }
};
fn makePrompt(reason: []const u8) [:0]u8 {
    var text = Text{};
    defer text.deinit();
    text.append(request);
    if (reason.len != 0) {
        text.append("\nCompaction reason: ");
        text.append(reason);
        text.append("\n");
    }
    return text.take();
}

pub fn summaryBudget(ctx: c_int) c_int {
    return @min(c.DS4_AGENT_COMPACT_SUMMARY_MAX_TOKENS, @max(256, @divTrunc(ctx, 8)));
}
pub fn reserve(engine: *c.ds4_engine, ctx: c_int) c_int {
    const text = makePrompt("context pressure before continuing the current task");
    defer allocator.free(text);
    var tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&tokens);
    c.ds4_chat_append_message(engine, &tokens, "user", text.ptr);
    c.ds4_chat_append_assistant_prefix(engine, &tokens, c.DS4_THINK_NONE);
    return tokens.len + 64 + summaryBudget(ctx);
}

/// The proportional cap makes small contexts compact before exhausting their
/// generation reserve instead of applying the fixed large-context threshold.
pub fn shouldCompact(engine: *c.ds4_engine, ctx: c_int, used: c_int) bool {
    if (ctx <= 0 or used <= 0) return false;
    if (used >= @divTrunc(ctx * c.DS4_AGENT_COMPACT_SOFT_PERCENT, 100)) return true;
    const free_threshold = @max(@min(c.DS4_AGENT_COMPACT_MIN_FREE_TOKENS, @divTrunc(ctx, 8)), reserve(engine, ctx) + 128);
    return ctx - used <= free_threshold;
}

/// A slightly longer complete user turn is preferable to an answer fragment
/// that has lost the user's constraints. Keep the original backward scan/ties.
pub fn tailBoundary(tokens: *const c.ds4_tokens, bottom: c_int, system_len: c_int, tail_budget: c_int, user_id: c_int) c_int {
    const target = @max(system_len, bottom - tail_budget);
    if (user_id < 0) return target;
    const earliest = @max(system_len, bottom - 2 * tail_budget);
    var index = bottom;
    while (index > earliest) {
        index -= 1;
        if (tokens.v[@intCast(index)] == user_id) return index;
    }
    return target;
}
fn tailStart(engine: *c.ds4_engine, tokens: *const c.ds4_tokens, ctx: c_int, system_len: c_int, prerope: bool) c_int {
    const budget = @max(1, @min(c.DS4_AGENT_COMPACT_TAIL_CAP_TOKENS, @divTrunc(ctx, c.DS4_AGENT_COMPACT_TAIL_DIVISOR)));
    const user = specialId(engine, if (c.ds4_engine_is_glm_dsa(engine)) "<|user|>" else "<｜User｜>");
    if (prerope and user >= 0) {
        // Retain an earlier complete turn near the ordinary budget if possible:
        // cached work can precede the latest user, whose history stays fresh.
        const earliest = @max(system_len, tokens.len - 2 * budget);
        var index = @max(system_len, tokens.len - budget);
        while (index >= earliest) : (index -= 1) {
            if (tokens.v[@intCast(index)] == user) return index;
            if (index == earliest) break;
        }
    }
    return tailBoundary(tokens, tokens.len, system_len, budget, user);
}

/// A summary prefix or retained tail must never split an image block. GLM has
/// one wrapper token on each side; embeddings and their owners stay untouched.
pub fn imageBoundary(images: []const c.ds4_vision_span, glm: bool, position: c_int) c_int {
    const wrapper: u32 = @intFromBool(glm);
    for (images) |image| {
        const start = @as(i64, image.token_start) - wrapper;
        const end = @as(u64, image.token_start) + image.embedding.token_count + wrapper;
        // The optional leading wrapper can put start at -1; crossing it
        // still implies a nonnegative position before the unsigned comparison.
        if (position > start and @as(u64, @intCast(position)) < end) return @intCast(start);
    }
    return position;
}
fn fail(errors: []u8, text: [*:0]const u8) bool {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", text);
    return false;
}

pub const Plan = struct {
    data: *Record,
    engine: *c.ds4_engine,
    transcript: *const c.ds4_tokens,
    system: *const c.ds4_tokens,
    images: []const c.ds4_vision_span,
    ctx: c_int,

    pub fn prepare(self: Plan, reason: []const u8, errors: []u8) bool {
        const text = makePrompt(reason);
        defer allocator.free(text);
        var suffix = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&suffix);
        c.ds4_chat_append_message(self.engine, &suffix, "user", text.ptr);
        c.ds4_chat_append_assistant_prefix(self.engine, &suffix, c.DS4_THINK_NONE);
        const budget = summaryBudget(self.ctx);
        const bottom = @min(self.transcript.len, self.ctx - suffix.len - budget - 1);
        self.data.summary_bottom = imageBoundary(self.images, c.ds4_engine_is_glm_dsa(self.engine), bottom);
        if (self.data.summary_bottom <= self.system.len)
            return fail(errors, "context too small to summarize this conversation; use a larger --ctx");
        // A full restored session may leave no summary room. Summarize a
        // bounded prefix and preserve EVERY unsummarized token in the new tail.
        appendRange(&self.data.prompt, self.transcript, 0, self.data.summary_bottom);
        appendRange(&self.data.prompt, &suffix, 0, suffix.len);
        self.data.summary_images = 0;
        while (self.data.summary_images < self.images.len) {
            const image = self.images[self.data.summary_images];
            if (@as(u64, image.token_start) + image.embedding.token_count > @as(u64, @intCast(self.data.summary_bottom))) break;
            self.data.summary_images += 1;
        }
        self.data.summary_room = self.ctx - self.data.prompt.len - 1;
        self.data.summary_max = @min(self.data.summary_room, budget);
        return true;
    }

    pub fn rebuild(self: Plan, summary: []const u8, prerope: bool, open_assistant: ?c_int, errors: []u8) bool {
        const data = self.data;
        data.tail_start = @min(tailStart(self.engine, self.transcript, self.ctx, self.system.len, prerope), data.summary_bottom);
        data.tail_start = imageBoundary(self.images, c.ds4_engine_is_glm_dsa(self.engine), data.tail_start);
        c.ds4_tokens_copy(&data.compacted, self.system);
        var message = Text{};
        defer message.deinit();
        message.append("\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n");
        message.append(summary);
        if (message.list.items.len != 0 and message.list.items[message.list.items.len - 1] != '\n') message.append("\n");
        message.append("[End compacted summary. Recent conversation continues verbatim below.]\n\n");
        const text = message.take();
        defer allocator.free(text);
        c.ds4_chat_append_message(self.engine, &data.compacted, "user", text.ptr);
        data.resumed_start = -1;
        if (open_assistant != null and data.tail_start > open_assistant.?) {
            // Resume the SAME assistant message, including partial words/code.
            // A synthetic user 'continue' request can skip unfinished fragments.
            const think_start = specialId(self.engine, "<think>");
            const think_end = specialId(self.engine, "</think>");
            var thinking = false;
            for (tokenSlice(self.transcript)[@intCast(open_assistant.?)..@intCast(data.tail_start)]) |token| {
                if (token == think_start) thinking = true;
                if (token == think_end) thinking = false;
            }
            data.resumed_start = data.compacted.len;
            c.ds4_chat_append_assistant_prefix(self.engine, &data.compacted, if (thinking) c.DS4_THINK_HIGH else c.DS4_THINK_NONE);
        } else if (data.tail_start < self.transcript.len and
            self.transcript.v[@intCast(data.tail_start)] != c.ds4_token_user(self.engine) and
            self.transcript.v[@intCast(data.tail_start)] != c.ds4_token_assistant(self.engine))
        {
            c.ds4_chat_append_message(self.engine, &data.compacted, "user", "[Verbatim tail of the previous conversation; it may begin mid-message.]\n");
        }
        data.tail_dst_start = data.compacted.len;
        if (open_assistant != null and data.resumed_start < 0)
            data.resumed_start = data.tail_dst_start + open_assistant.? - data.tail_start;
        appendRange(&data.compacted, self.transcript, data.tail_start, self.transcript.len);
        if (data.compacted.len >= self.ctx - reserve(self.engine, self.ctx) - 128)
            return fail(errors, "compacted conversation leaves no working room; use a larger --ctx");
        return true;
    }
};

pub fn free(plan: *Record) void {
    c.ds4_tokens_free(&plan.prompt);
    c.ds4_tokens_free(&plan.compacted);
    plan.* = std.mem.zeroes(Record);
}
