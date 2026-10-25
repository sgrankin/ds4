//! Public Engine/Session evaluation loop. Sampling order, answer reserves and
//! pause accounting are benchmark policy; there is no private engine layout.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Grade = @import("grading.zig");
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig");
const Terminal = @import("ui.zig");
const Ui = Terminal.Ui;
const allocator = std.heap.c_allocator;
pub const Result = enum { ok, failure, switched, quit };
fn puts(bytes: [*:0]const u8) void {
    _ = c.fputs(bytes, c.__stdoutp);
}
fn color(enabled: bool, ansi: [*:0]const u8) [*:0]const u8 {
    return if (enabled) ansi else "";
}
fn plainReset(enabled: bool) void {
    if (enabled) puts(Terminal.reset);
}
pub fn hasSwitch(ui: *const Ui, running_index: c_int) bool {
    return ui.enabled and ui.requested_case >= 0 and ui.requested_case < @as(c_int, @intCast(ui.cases.len)) and ui.requested_case != running_index;
}
fn hasQuit(ui: *const Ui) bool {
    return ui.enabled and ui.quit_requested;
}
fn pending(ui: *Ui, index: usize) void {
    ui.status[index] = c.EVAL_PENDING;
    ui.guess[index][0] = 0;
}
pub fn waitPaused(ui: *Ui, phase: []const u8) f64 {
    if (!ui.enabled or !ui.paused) return 0;
    const was_running = ui.run_clock_active;
    if (was_running) ui.clockStop();
    const start = Terminal.now();
    ui.refresh(phase);
    while (ui.paused) {
        _ = c.usleep(50000);
        ui.consume();
        ui.refresh(phase);
    }
    if (was_running) ui.clockStart();
    ui.refresh(phase);
    return Terminal.now() - start;
}
fn progress(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    if (context == null or event == null) return;
    const name = Common.text(event);
    if (!std.mem.eql(u8, name, "prefill_chunk") and !std.mem.eql(u8, name, "prefill_display")) return;
    const ui: *Ui = @ptrCast(@alignCast(context.?));
    ui.consume();
    ui.clockTick();
    ui.prefill_current = current;
    ui.prefill_total = total;
    const elapsed = Terminal.now() - ui.phase_start_sec;
    ui.speed_tps = if (elapsed > 0.001) @as(f64, @floatFromInt(current)) / elapsed else 0;
    ui.refresh("prefill");
    const paused = waitPaused(ui, "prefill");
    if (paused > 0) ui.phase_start_sec += paused;
}
const Run = struct {
    config: *const c.eval_config,
    ui: *Ui,
    trace: ?*c.FILE,
    index: usize,
    think: c.ds4_think_mode,
    question: [:0]const u8,
    prompt_tokens: c_int,
    generation_limit: c_int,
    raw: std.ArrayList(u8) = .empty,
    close: Trace.CloseInfo = std.mem.zeroes(Trace.CloseInfo),

    fn case(self: *const Run) *const c.eval_case {
        return &self.ui.cases[self.index];
    }
    fn output(self: *const Run) []const u8 {
        // libc string consumers in the original stop at a decoded NUL even
        // though byte output and generation accounting retain the full token.
        return Common.written(self.raw.items);
    }
    fn record(self: *const Run, status: [*:0]const u8, errors: [*c]const u8, generated: c_int, elapsed: f64, picked: [*c]const u8, close: ?*const Trace.CloseInfo) void {
        Trace.writeCase(self.trace, .{
            .config = self.config,
            .case = self.case(),
            .index = @intCast(self.index),
            .count = @intCast(self.ui.cases.len),
            .status = status,
            .errors = errors,
            .system = Prompts.system,
            .question = self.question.ptr,
            .generated = self.output(),
            .think = self.think,
            .prompt_tokens = self.prompt_tokens,
            .generated_tokens = generated,
            .generation_budget = self.generation_limit,
            .elapsed = elapsed,
            .picked = picked,
            .close = close,
        });
    }
    fn interrupt(self: *Run, start: f64) ?Result {
        const ui = self.ui;
        const quit = hasQuit(ui);
        if (!quit and !hasSwitch(ui, @intCast(self.index))) return null;
        if (quit) ui.status[self.index] = c.EVAL_STOPPED else pending(ui, self.index);
        ui.generated_tokens[self.index] = ui.generated;
        ui.clockStop();
        ui.refresh("idle");
        self.record(if (quit) "STOPPED" else "SWITCHED", null, ui.generated, Terminal.now() - start, "?", &self.close);
        return if (quit) .quit else .switched;
    }
};
pub fn runOne(engine: *c.ds4_engine, session: *c.ds4_session, config: *const c.eval_config, ui: *Ui, trace: ?*c.FILE, index: usize, generation_budget: c_int, rng: *u64) Result {
    const case = &ui.cases[index];
    const tty = ui.enabled;
    const use_color = !tty and c.isatty(c.STDOUT_FILENO) != 0;
    const think = c.ds4_engine_think_mode_for_context(engine, config.think_mode, config.ctx_size);
    const question = Prompts.question(case);
    defer allocator.free(question);
    var prompt = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&prompt);
    c.ds4_encode_chat_prompt(engine, Prompts.system, question.ptr, think, &prompt);
    ui.prompt_tokens[index] = prompt.len;
    ui.generated_tokens[index] = 0;
    var run = Run{ .config = config, .ui = ui, .trace = trace, .index = index, .think = think, .question = question, .prompt_tokens = prompt.len, .generation_limit = generation_budget };
    defer run.raw.deinit(allocator);
    if (prompt.len >= config.ctx_size) {
        ui.active_case = @intCast(index);
        if (!ui.selection_active) ui.selected_case = @intCast(index);
        ui.guess[index][0] = 0;
        ui.status[index] = c.EVAL_SKIPPED;
        ui.refresh("idle");
        run.record("SKIPPED", "prompt does not fit context", 0, 0, "?", null);
        if (!tty) _ = c.printf(
            "\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
            @as(c_int, @intCast(index + 1)),
            @as(c_int, @intCast(ui.cases.len)),
            case.source,
            case.id,
            prompt.len,
            config.ctx_size,
        );
        return .ok;
    }
    run.generation_limit = @min(generation_budget, config.ctx_size - prompt.len);
    if (run.generation_limit < 1) {
        ui.active_case = @intCast(index);
        if (!ui.selection_active) ui.selected_case = @intCast(index);
        ui.guess[index][0] = 0;
        ui.status[index] = c.EVAL_SKIPPED;
        ui.refresh("idle");
        run.record("SKIPPED", "prompt leaves no generation room", 0, 0, "?", null);
        if (!tty) _ = c.printf(
            "\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
            @as(c_int, @intCast(index + 1)),
            @as(c_int, @intCast(ui.cases.len)),
            case.source,
            case.id,
            prompt.len,
            config.ctx_size,
        );
        return .ok;
    }
    ui.active_case = @intCast(index);
    if (!ui.selection_active) ui.selected_case = @intCast(index);
    ui.requested_case = -1;
    ui.guess[index][0] = 0;
    ui.status[index] = c.EVAL_PREFILL;
    ui.max_tokens = run.generation_limit;
    ui.think_max_tokens = @max(run.generation_limit - config.hard_limit_reply_budget, 0);
    ui.clockStart();
    if (tty) {
        ui.resetStream(case, c.ds4_think_mode_enabled(think));
        ui.refresh("prefill");
    } else {
        _ = c.printf(
            "\n%s[%d/%d] %s%s%s/%s%s%s %s%s\n",
            color(use_color, Terminal.bold),
            @as(c_int, @intCast(index + 1)),
            @as(c_int, @intCast(ui.cases.len)),
            color(use_color, Terminal.cyan),
            case.source,
            color(use_color, Terminal.reset),
            color(use_color, Terminal.cyan),
            case.domain,
            color(use_color, Terminal.reset),
            case.title,
            color(use_color, Terminal.reset),
        );
        _ = c.fflush(c.__stdoutp);
    }
    var errors: [256]u8 = undefined;
    c.ds4_session_set_progress(session, progress, ui);
    c.ds4_session_set_display_progress(session, progress, ui);
    const synced = c.ds4_session_sync(session, &prompt, &errors, errors.len);
    c.ds4_session_set_progress(session, null, null);
    c.ds4_session_set_display_progress(session, null, null);
    if (synced != 0) {
        ui.clockStop();
        _ = c.fprintf(c.__stderrp, "ds4-eval: prefill failed for %s: %s\n", case.id, &errors);
        run.record("ERROR", &errors, 0, 0, "?", null);
        return .failure;
    }
    run.prompt_tokens = prompt.len;
    ui.prompt_tokens[index] = prompt.len;
    c.ds4_tokens_free(&prompt);
    ui.consume();
    _ = waitPaused(ui, "prefill");
    if (hasQuit(ui)) {
        ui.status[index] = c.EVAL_STOPPED;
        ui.generated_tokens[index] = 0;
        ui.clockStop();
        ui.refresh("idle");
        run.record("STOPPED", null, 0, 0, "?", null);
        return .quit;
    }
    if (hasSwitch(ui, @intCast(index))) {
        pending(ui, index);
        ui.clockStop();
        ui.refresh("idle");
        run.record("SWITCHED", null, 0, 0, "?", null);
        return .switched;
    }
    ui.status[index] = c.EVAL_THINKING;
    ui.generated = 0;
    ui.phase_start_sec = Terminal.now();
    ui.speed_tps = 0;
    var plain_in_think = c.ds4_think_mode_enabled(think);
    var generation_in_think = c.ds4_think_mode_enabled(think);
    var close_tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&close_tokens);
    if (generation_in_think) c.ds4_tokenize_text(engine, "</think>", &close_tokens);
    if (!tty and plain_in_think and use_color) puts(Terminal.dim);
    ui.refresh("thinking");
    var start = ui.phase_start_sec;
    var forced_close: c_int = -1;
    for (0..@intCast(run.generation_limit)) |_| {
        if (tty) {
            ui.consume();
            if (run.interrupt(start)) |result| return result;
            const paused = waitPaused(ui, if (ui.in_think) "thinking" else "answer");
            if (paused > 0) {
                ui.phase_start_sec += paused;
                start += paused;
            }
            if (run.interrupt(start)) |result| return result;
        }
        const remaining = run.generation_limit - ui.generated;
        var close_rank: c_int = 0;
        var token: c_int = -1;
        var close_kind: c.eval_think_close_kind = c.EVAL_THINK_CLOSE_NONE;
        // The soft limit accepts the model's own high-ranked desire to close
        // thinking. The hard limit reserves answer tokens uniformly, rather
        // than letting the full benchmark budget disappear inside <think>.
        if (generation_in_think and close_tokens.len > 0) {
            if (forced_close >= 0) {
                token = close_tokens.v[@intCast(forced_close)];
                forced_close += 1;
                if (forced_close >= close_tokens.len) forced_close = -1;
            } else if (remaining <= config.hard_limit_reply_budget) {
                close_kind = c.EVAL_THINK_CLOSE_HARD;
                token = close_tokens.v[0];
                forced_close = if (close_tokens.len > 1) 1 else -1;
            } else if (remaining <= config.soft_limit_reply_budget and close_tokens.len == 1) {
                close_rank = Trace.tokenRank(session, close_tokens.v[0], config.soft_limit_think_close_rank);
                if (close_rank > 0) {
                    close_kind = c.EVAL_THINK_CLOSE_SOFT;
                    token = close_tokens.v[0];
                }
            }
        }
        if (token < 0) token = c.ds4_session_sample(session, config.temperature, 0, config.top_p, config.min_p, rng);
        if (c.ds4_token_is_stop(engine, token)) break;
        if (close_kind != c.EVAL_THINK_CLOSE_NONE and run.close.kind == c.EVAL_THINK_CLOSE_NONE) {
            run.close = .{ .kind = close_kind, .token_index = ui.generated + 1, .remaining_budget = remaining, .rank = close_rank };
        }
        if (c.ds4_session_eval(session, token, &errors, errors.len) != 0) {
            plainReset(use_color);
            ui.generated_tokens[index] = ui.generated;
            ui.clockStop();
            _ = c.fprintf(c.__stderrp, "ds4-eval: decode failed for %s: %s\n", case.id, &errors);
            run.record("ERROR", &errors, ui.generated, Terminal.now() - start, "?", &run.close);
            return .failure;
        }
        var length: usize = 0;
        const allocation = c.ds4_token_text(engine, token, &length);
        defer c.free(allocation);
        const text: []const u8 = if (length == 0) &.{} else allocation[0..length];
        run.raw.appendSlice(allocator, text) catch Terminal.fatal();
        ui.generated += 1;
        ui.generated_tokens[index] = ui.generated;
        ui.clockTick();
        if (generation_in_think and std.mem.find(u8, run.output(), "</think>") != null) {
            generation_in_think = false;
            if (run.close.kind == c.EVAL_THINK_CLOSE_NONE) run.close = .{ .kind = c.EVAL_THINK_CLOSE_NATURAL, .token_index = ui.generated, .remaining_budget = remaining, .rank = 0 };
        }
        const elapsed = Terminal.now() - ui.phase_start_sec;
        ui.speed_tps = if (elapsed > 0.001) @as(f64, @floatFromInt(ui.generated)) / elapsed else 0;
        if (tty) {
            ui.append(text, false);
            ui.refresh(if (ui.in_think) "thinking" else "answer");
        } else {
            if (plain_in_think and std.mem.find(u8, run.output(), "</think>") != null) {
                plain_in_think = false;
                plainReset(use_color);
            }
            _ = c.fwrite(text.ptr, 1, text.len, c.__stdoutp);
            _ = c.fflush(c.__stdoutp);
        }
    }
    if (tty) {
        ui.append(&.{}, true);
        ui.drawStream();
    } else {
        plainReset(use_color);
        if (run.raw.items.len == 0 or run.raw.items[run.raw.items.len - 1] != '\n') _ = c.fputc('\n', c.__stdoutp);
    }
    var got: [c.EVAL_ANSWER_MAX]u8 = undefined;
    const complete = Grade.strict(case, run.output(), &got);
    Common.copy(&ui.guess[index], Common.written(&got));
    const passed = complete and Grade.matches(case, &got);
    ui.status[index] = if (!complete) c.EVAL_INCOMPLETE else if (passed) c.EVAL_PASSED else c.EVAL_FAILED;
    ui.generated_tokens[index] = ui.generated;
    ui.clockStop();
    const elapsed = Terminal.now() - start;
    const result_name: [*:0]const u8 = if (!complete) "INCOMPLETE" else if (passed) "PASSED" else "FAILED";
    ui.refresh(if (!complete) "incomplete" else if (passed) "passed" else "failed");
    run.record(result_name, null, ui.generated, elapsed, &got, &run.close);
    if (!tty) _ = c.printf(
        "%s%s%s got %s expected %s (%.1fs, %d tokens)\n",
        if (use_color) @as([*:0]const u8, if (!complete) Terminal.yellow else if (passed) Terminal.green else Terminal.red) else @as([*:0]const u8, ""),
        if (!complete) @as([*:0]const u8, "INCOMPLETE") else if (passed) @as([*:0]const u8, "PASSED") else @as([*:0]const u8, "FAIL"),
        color(use_color, Terminal.reset),
        &got,
        case.answer,
        elapsed,
        ui.generated,
    );
    if (tty and config.pause_ms > 0) _ = c.usleep(@as(c.useconds_t, @intCast(config.pause_ms)) *% 1000);
    return .ok;
}
pub fn nextPending(ui: *const Ui, start: c_int) c_int {
    const count: c_int = @intCast(ui.cases.len);
    if (count <= 0) return -1;
    var offset: c_int = 0;
    while (offset < count) : (offset += 1) {
        const index = @mod(start + offset, count);
        if (ui.status[@intCast(index)] == c.EVAL_PENDING) return index;
    }
    return -1;
}
pub fn sequence(argument: [*c]const u8, count: c_int) ?[]c_int {
    var values: std.ArrayList(c_int) = .empty;
    defer values.deinit(allocator);
    var cursor = argument;
    while (cursor != null and cursor[0] != 0) {
        while (c.isspace(cursor[0]) != 0) cursor += 1;
        if (cursor[0] == 0) break;
        c.__error().* = 0;
        var end: [*c]u8 = null;
        const value = c.strtol(cursor, &end, 10);
        if (end == cursor or c.__error().* != 0 or value < 1 or value > count) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: invalid --case-sequence entry near '%s' (valid range: 1..%d)\n", cursor, count);
            return null;
        }
        values.append(allocator, @intCast(value - 1)) catch {
            _ = c.fprintf(c.__stderrp, "ds4-eval: out of memory while parsing --case-sequence\n");
            return null;
        };
        cursor = end;
        while (c.isspace(cursor[0]) != 0) cursor += 1;
        if (cursor[0] == ',') {
            cursor += 1;
            continue;
        }
        if (cursor[0] != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: expected comma in --case-sequence near '%s'\n", cursor);
            return null;
        }
    }
    if (values.items.len == 0) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: --case-sequence cannot be empty\n");
        return null;
    }
    return values.toOwnedSlice(allocator) catch {
        _ = c.fprintf(c.__stderrp, "ds4-eval: out of memory while parsing --case-sequence\n");
        return null;
    };
}
pub fn waitRoute(session: *c.ds4_session) bool {
    var errors: [256]u8 = @splat(0);
    var last: [256]u8 = @splat(0);
    var ticks: c_uint = 0;
    const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 250000000 };
    while (true) {
        const ready = c.ds4_session_distributed_route_ready(session, &errors, errors.len);
        if (ready > 0) {
            if (ticks != 0) _ = c.fprintf(c.__stderrp, "ds4-eval: distributed route ready\n");
            return true;
        }
        if (ready < 0) {
            _ = c.fprintf(
                c.__stderrp,
                "ds4-eval: distributed route readiness failed: %s\n",
                if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "unknown error"),
            );
            return false;
        }
        const why: [*c]const u8 = if (errors[0] != 0) &errors else "route incomplete";
        if (c.strcmp(&last, why) != 0 or ticks % 20 == 0) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: waiting for distributed route: %s\n", why);
            Common.copy(&last, Common.text(why));
        }
        _ = c.nanosleep(&delay, null);
        ticks +%= 1;
    }
}
fn reportName(status: c.eval_status) [*:0]const u8 {
    return switch (status) {
        c.EVAL_PASSED => "PASSED",
        c.EVAL_FAILED => "FAILED",
        c.EVAL_INCOMPLETE => "INCOMPLETE",
        c.EVAL_SKIPPED => "SKIPPED",
        c.EVAL_STOPPED => "STOPPED",
        c.EVAL_PREFILL => "PREFILL",
        c.EVAL_THINKING => "RUNNING",
        else => "PENDING",
    };
}
pub fn report(ui: *const Ui, passed: c_int, failed: c_int, incomplete: c_int) void {
    const minutes: c_ulonglong = @intFromFloat(@max(ui.elapsed(), 0) / 60.0);
    _ = c.printf("ds4-eval: %d/%d passed", passed, @as(c_int, @intCast(ui.cases.len)));
    if (failed != 0) _ = c.printf(", %d failed", failed);
    if (incomplete != 0) _ = c.printf(", %d incomplete", incomplete);
    _ = c.printf(", runtime %02lluh:%02llum\n", minutes / 60, minutes % 60);
    _ = c.printf(
        "%-3s %-10s %8s %8s %8s %-20s %-20s %s\n",
        @as([*:0]const u8, "#"),
        @as([*:0]const u8, "state"),
        @as([*:0]const u8, "prompt"),
        @as([*:0]const u8, "gen"),
        @as([*:0]const u8, "total"),
        @as([*:0]const u8, "given"),
        @as([*:0]const u8, "correct"),
        @as([*:0]const u8, "test"),
    );
    for (ui.cases, 0..) |case, index| {
        const prompt = ui.prompt_tokens[index];
        const generated = ui.generated_tokens[index];
        _ = c.printf(
            "%3d %-10s %8d %8d %8d %-20.20s %-20.20s %s/%s\n",
            @as(c_int, @intCast(index + 1)),
            reportName(ui.status[index]),
            prompt,
            generated,
            prompt + generated,
            if (ui.guess[index][0] != 0) @as([*c]const u8, &ui.guess[index]) else @as([*c]const u8, "-"),
            case.answer,
            case.source,
            case.id,
        );
    }
}
