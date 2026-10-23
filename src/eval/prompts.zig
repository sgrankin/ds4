//! Prompt spelling and budget policy are shared by sizing and actual runs.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Grading = @import("grading.zig");
const allocator = std.heap.c_allocator;
pub const system = "You are solving a hard benchmark question. Reason carefully. " ++
    "The final answer must follow the requested format exactly.";
fn append(bytes: *std.ArrayList(u8), value: []const u8) void {
    bytes.appendSlice(allocator, value) catch @import("ui.zig").fatal();
}
pub fn question(case: *const c.eval_case) [:0]u8 {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    append(&bytes, Common.text(case.question));
    append(&bytes, "\n");
    const kind = Grading.answerKind(case);
    if (kind == c.EVAL_ANSWER_CHOICE) {
        append(&bytes, "\nChoices:\n");
        for (0..@intCast(Grading.choices(case))) |index| {
            bytes.append(allocator, @intCast('A' + index)) catch @import("ui.zig").fatal();
            append(&bytes, ". ");
            append(&bytes, Common.text(case.choice[index]));
            append(&bytes, "\n");
        }
        append(&bytes, "\nSolve the question. At the end, write exactly one final line in this " ++
            "format and do not write anything after it:\nAnswer: <letter>");
    } else if (kind == c.EVAL_ANSWER_LINE_SET) {
        append(&bytes, "\nAt the end, write exactly one final line in this format and do not " ++
            "write anything after it:\nAnswer: <line number or comma-separated line numbers>");
    } else if (kind == c.EVAL_ANSWER_INTEGER) {
        append(&bytes, "\nSolve the problem. At the end, write exactly one final line in this " ++
            "format and do not write anything after it:\nAnswer: <integer>");
    } else if (kind == c.EVAL_ANSWER_RATIONAL) {
        append(&bytes, "\nSolve the problem. Reduce the result. At the end, write exactly one final " ++
            "line in this format and do not write anything after it:\nAnswer: <integer or reduced fraction>");
    } else if (kind == c.EVAL_ANSWER_ORDERED_SEQUENCE) {
        append(&bytes, "\nSolve the problem. At the end, write exactly one final line containing the " ++
            "answers in the requested order, separated by commas, and do not write anything " ++
            "after it:\nAnswer: <ordered answers>");
    } else {
        append(&bytes, "\nSolve the problem. At the end, write exactly one final line in this format " ++
            "and do not write anything after it:\nAnswer: <exact answer>");
    }
    return bytes.toOwnedSliceSentinel(allocator, 0) catch @import("ui.zig").fatal();
}
pub fn budget(config: *const c.eval_config, case: *const c.eval_case, retry: bool) c_int {
    var tokens: u64 = if (!config.max_tokens_set and case.max_tokens != 0) case.max_tokens else @as(u32, @intCast(config.max_tokens));
    if (retry) tokens *= 2;
    return @intCast(@min(tokens, c.EVAL_MAX_CONTEXT));
}
pub fn maxBudget(config: *const c.eval_config, cases: []const c.eval_case) c_int {
    var maximum: c_int = 0;
    for (cases) |*case| maximum = @max(maximum, budget(config, case, config.retry_incomplete));
    return maximum;
}
pub const Largest = struct { tokens: c_int = 0, index: c_int = -1 };
pub fn maxPrompt(engine: *c.ds4_engine, config: *const c.eval_config, cases: []const c.eval_case, context: c_int) Largest {
    var largest: Largest = .{};
    const think = c.ds4_think_mode_for_context(config.think_mode, context);
    for (cases, 0..) |*case, index| {
        const text = question(case);
        defer allocator.free(text);
        var tokens = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&tokens);
        c.ds4_encode_chat_prompt(engine, system, text.ptr, think, &tokens);
        if (tokens.len > largest.tokens) largest = .{ .tokens = tokens.len, .index = @intCast(index) };
    }
    return largest;
}
pub const Sized = struct { context: c_int, largest: Largest };
pub fn autoContext(engine: *c.ds4_engine, config: *const c.eval_config, cases: []const c.eval_case, max_generation: c_int) Sized {
    var context: c_int = c.EVAL_MAX_CONTEXT;
    var largest: Largest = .{};
    const minimum: c_int = if (config.think_mode == c.DS4_THINK_MAX) @intCast(c.ds4_think_max_min_context()) else 1;
    // Think Max downgrades below its context minimum. Iterate so sizing uses
    // the same effective thinking mode as the actual prompt encoder.
    for (0..3) |_| {
        largest = maxPrompt(engine, config, cases, context);
        const required = @max(@as(i64, largest.tokens) + max_generation, minimum);
        if (required > c.EVAL_MAX_CONTEXT) {
            _ = c.fprintf(
                c.__stderrp,
                "ds4-eval: largest prompt (%d tokens, case %d) + generation budget (%d) exceeds the %d token context cap\n",
                largest.tokens,
                largest.index + 1,
                max_generation,
                @as(c_int, c.EVAL_MAX_CONTEXT),
            );
            c.exit(2);
        }
        if (required == context) break;
        context = @intCast(required);
    }
    return .{ .context = context, .largest = largest };
}
pub fn warn(config: *const c.eval_config, largest: Largest, max_generation: c_int) void {
    if (largest.tokens >= config.ctx_size) {
        _ = c.fprintf(
            c.__stderrp,
            "ds4-eval: warning: largest prompt (%d tokens, case=%d) does not fit ctx=%d\n",
            largest.tokens,
            largest.index + 1,
            config.ctx_size,
        );
    } else if (config.ctx_size - largest.tokens < max_generation) {
        _ = c.fprintf(
            c.__stderrp,
            "ds4-eval: warning: largest prompt (%d tokens, case=%d) leaves %d generation tokens in ctx=%d; requested %d\n",
            largest.tokens,
            largest.index + 1,
            config.ctx_size - largest.tokens,
            config.ctx_size,
            max_generation,
        );
    }
}
pub fn warnThink(config: *const c.eval_config) void {
    if (config.think_mode == c.DS4_THINK_MAX and c.ds4_think_mode_for_context(config.think_mode, config.ctx_size) != c.DS4_THINK_MAX)
        _ = c.fprintf(
            c.__stderrp,
            "ds4-eval: warning: --think-max needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
            c.ds4_think_max_min_context(),
            config.ctx_size,
        );
}
