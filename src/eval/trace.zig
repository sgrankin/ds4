//! The trace is a counted-block text format. Regrading skips model payloads by
//! byte count so an answer containing a CASE header cannot become metadata.
//! Delimiter-only MODEL_OUTPUT blocks remain accepted for old traces.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Cases = @import("cases.zig");
const Grading = @import("grading.zig");
pub const Outcome = enum(c_int) { not_graded, passed, failed };
pub const CloseInfo = c.eval_think_close_info;

fn say(comptime pattern: [*:0]const u8, arguments: anytype) void {
    _ = @call(.auto, c.fprintf, .{ c.__stderrp, pattern } ++ arguments);
}
fn lineEnd(bytes: []const u8, start: usize) usize {
    return std.mem.findScalarPos(u8, bytes, start, '\n') orelse bytes.len;
}
fn afterLine(bytes: []const u8, end: usize) usize {
    return if (end < bytes.len) end + 1 else end;
}
const Declared = struct { length: usize, end: usize };
fn declaredLength(bytes: []const u8, start: usize, end: usize) ?Declared {
    var parsed_end: [*c]u8 = null;
    const parsed = c.strtoull(bytes.ptr + start, &parsed_end, 10);
    const consumed = @intFromPtr(parsed_end) - @intFromPtr(bytes.ptr);
    if (consumed == start or consumed > end) return null;
    return .{ .length = @intCast(parsed), .end = consumed };
}
fn skipCounted(bytes: []const u8, start: usize, end: usize) ?usize {
    const suffix = "_BEGIN bytes=";
    const relative = std.mem.find(u8, bytes[start..end], suffix) orelse return null;
    if (relative == 0 or relative > 96) return null;
    const declared = declaredLength(bytes, start + relative + suffix.len, end) orelse return null;
    const content = afterLine(bytes, end);
    if (declared.length > bytes.len - content) return null;
    var marker = content + declared.length;
    if (marker < bytes.len and bytes[marker] == '\n') marker += 1;
    const label = bytes[start..][0..relative];
    if (!std.mem.startsWith(u8, bytes[marker..], label)) return null;
    marker += label.len;
    if (!std.mem.startsWith(u8, bytes[marker..], "_END")) return null;
    return afterLine(bytes, lineEnd(bytes, marker + 4));
}
pub fn nextCase(bytes: []const u8) ?usize {
    var position: usize = 0;
    while (position < bytes.len) {
        const end = lineEnd(bytes, position);
        if (skipCounted(bytes, position, end)) |after| {
            if (after > position) {
                position = after;
                continue;
            }
        }
        if (std.mem.startsWith(u8, bytes[position..end], "===== CASE ")) return position;
        position = afterLine(bytes, end);
    }
    return null;
}
fn blockBegin(bytes: []const u8, label: []const u8) ?usize {
    var position: usize = 0;
    while (position < bytes.len) {
        const end = lineEnd(bytes, position);
        if (std.mem.startsWith(u8, bytes[position..end], label)) return position;
        if (skipCounted(bytes, position, end)) |after| {
            if (after > position) {
                position = after;
                continue;
            }
        }
        position = afterLine(bytes, end);
    }
    return null;
}
fn field(bytes: []const u8, key: []const u8, output: []u8) bool {
    if (output.len != 0) output[0] = 0;
    var position: usize = 0;
    while (position < bytes.len) {
        const end = lineEnd(bytes, position);
        if (std.mem.startsWith(u8, bytes[position..end], key)) {
            Common.copy(output, std.mem.trimEnd(u8, bytes[position + key.len .. end], "\r\n"));
            return true;
        }
        position = afterLine(bytes, end);
    }
    return false;
}
pub fn copyOutput(bytes: []const u8) [*c]u8 {
    const begin = blockBegin(bytes, "MODEL_OUTPUT_BEGIN") orelse return null;
    const end = lineEnd(bytes, begin);
    if (end == bytes.len) return null;
    const content = end + 1;
    var length: usize = undefined;
    if (std.mem.find(u8, bytes[begin..end], "bytes=")) |relative| {
        const declared = declaredLength(bytes, begin + relative + 6, end) orelse return null;
        if (declared.length > bytes.len - content) return null;
        length = declared.length;
        var marker = content + length;
        if (marker < bytes.len and bytes[marker] == '\n') marker += 1;
        if (!std.mem.startsWith(u8, bytes[marker..], "MODEL_OUTPUT_END")) return null;
    } else {
        length = std.mem.find(u8, bytes[content..], "\nMODEL_OUTPUT_END") orelse return null;
    }
    const allocation = c.malloc(length + 1) orelse {
        say("ds4-eval: out of memory\n", .{});
        return null;
    };
    const output: [*]u8 = @ptrCast(allocation);
    @memcpy(output[0..length], bytes[content..][0..length]);
    output[length] = 0;
    return output;
}

/// Only completed graded runs contribute to drift. Interrupted output may look
/// correct by coincidence; empty status retains compatibility with old traces.
pub fn outcome(case: *const c.eval_case, status: []const u8, generated: []const u8, output: []u8) Outcome {
    if (status.len != 0 and !std.mem.eql(u8, status, "PASSED") and !std.mem.eql(u8, status, "FAILED")) {
        if (output.len != 0) output[0] = 0;
        return .not_graded;
    }
    Grading.extract(case, generated, output);
    return if (Grading.matches(case, output.ptr)) .passed else .failed;
}
fn read(path: [*c]const u8, length: *usize) [*c]u8 {
    const file = c.fopen(path, "rb") orelse {
        say("ds4-eval: cannot open trace '%s': %s\n", .{ path, c.strerror(c.__error().*) });
        return null;
    };
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) {
        say("ds4-eval: cannot seek trace '%s': %s\n", .{ path, c.strerror(c.__error().*) });
        return null;
    }
    const size = c.ftell(file);
    if (size < 0) {
        say("ds4-eval: cannot tell trace size '%s': %s\n", .{ path, c.strerror(c.__error().*) });
        return null;
    }
    c.rewind(file);
    const count: usize = @intCast(size);
    const allocation = c.malloc(count + 1) orelse {
        say("ds4-eval: out of memory\n", .{});
        return null;
    };
    const bytes: [*]u8 = @ptrCast(allocation);
    if (count != 0 and c.fread(bytes, 1, count, file) != count) {
        say("ds4-eval: cannot read trace '%s': %s\n", .{ path, if (c.ferror(file) != 0) c.strerror(c.__error().*) else @as([*c]const u8, "short read") });
        c.free(bytes);
        return null;
    }
    bytes[count] = 0;
    length.* = count;
    return bytes;
}
pub fn regrade(path: [*c]const u8) c_int {
    var length: usize = 0;
    const allocation = read(path, &length);
    if (allocation == null) return 2;
    defer c.free(allocation);
    const bytes = allocation[0..length];
    var position: usize = 0;
    var total: c_int = 0;
    var passed: c_int = 0;
    var failed: c_int = 0;
    var changed: c_int = 0;
    var unknown: c_int = 0;
    var parse_errors: c_int = 0;
    var not_graded: c_int = 0;
    while (nextCase(bytes[position..])) |relative| {
        const begin = position + relative;
        const after = afterLine(bytes, lineEnd(bytes, begin));
        const end = if (nextCase(bytes[after..])) |next| after + next else bytes.len;
        position = end;
        total += 1;
        const record = bytes[begin..end];
        var source: [64]u8 = undefined;
        var id: [128]u8 = undefined;
        var status: [32]u8 = undefined;
        var picked: [c.EVAL_ANSWER_MAX]u8 = undefined;
        if (!field(record, "source: ", &source) or !field(record, "id: ", &id)) {
            say("ds4-eval: trace case %d is missing source/id\n", .{total});
            parse_errors += 1;
            continue;
        }
        _ = field(record, "status: ", &status);
        _ = field(record, "picked: ", &picked);
        const case = Cases.find(&source, &id) orelse {
            say("ds4-eval: trace case %d not found in embedded cases: %s/%s\n", .{ total, &source, &id });
            unknown += 1;
            continue;
        };
        const generated = copyOutput(record);
        if (generated == null) {
            say("ds4-eval: trace case %d is missing MODEL_OUTPUT block: %s/%s\n", .{ total, &source, &id });
            parse_errors += 1;
            continue;
        }
        defer c.free(generated);
        var extracted: [c.EVAL_ANSWER_MAX]u8 = undefined;
        const result = outcome(case, Common.written(&status), Common.text(generated), &extracted);
        if (result == .not_graded) {
            not_graded += 1;
            continue;
        }
        const ok = result == .passed;
        if (ok) passed += 1 else failed += 1;
        const was_ok = std.mem.eql(u8, Common.written(&status), "PASSED");
        if ((status[0] != 0 and ok != was_ok) or (picked[0] != 0 and !std.mem.eql(u8, Common.written(&extracted), Common.written(&picked)))) {
            changed += 1;
            _ = c.printf(
                "case %d %s/%s: trace %s picked=%s -> regrade %s picked=%s expected=%s\n",
                total,
                &source,
                &id,
                if (status[0] != 0) @as([*c]const u8, &status) else @as([*c]const u8, "?"),
                if (picked[0] != 0) @as([*c]const u8, &picked) else @as([*c]const u8, "?"),
                if (ok) @as([*c]const u8, "PASSED") else @as([*c]const u8, "FAILED"),
                &extracted,
                if (case.answer != null) case.answer else @as([*c]const u8, "?"),
            );
        }
    }
    _ = c.printf(
        "ds4-eval: regraded %d cases from %s: passed=%d failed=%d changed=%d not_graded=%d unknown=%d parse_errors=%d\n",
        total,
        path,
        passed,
        failed,
        changed,
        not_graded,
        unknown,
        parse_errors,
    );
    return if (unknown != 0 or parse_errors != 0 or total == 0) 1 else 0;
}

pub fn block(file: ?*c.FILE, label: [*:0]const u8, bytes: []const u8) void {
    const trace = file orelse return;
    _ = c.fprintf(trace, "%s_BEGIN bytes=%zu\n", label, bytes.len);
    if (bytes.len != 0) {
        _ = c.fwrite(bytes.ptr, 1, bytes.len, trace);
        if (bytes[bytes.len - 1] != '\n') _ = c.fputc('\n', trace);
    }
    _ = c.fprintf(trace, "%s_END\n", label);
}
fn closeKind(kind: c.eval_think_close_kind) [*:0]const u8 {
    return switch (kind) {
        c.EVAL_THINK_CLOSE_NATURAL => "natural",
        c.EVAL_THINK_CLOSE_SOFT => "soft_forced",
        c.EVAL_THINK_CLOSE_HARD => "hard_forced",
        else => "none",
    };
}
pub fn tokenRank(session: *c.ds4_session, token: c_int, max_rank: c_int) c_int {
    if (token < 0 or max_rank <= 0) return 0;
    const allocation = c.malloc(@as(usize, @intCast(max_rank)) * @sizeOf(c.ds4_token_score)) orelse return 0;
    defer c.free(allocation);
    const top: [*]c.ds4_token_score = @ptrCast(@alignCast(allocation));
    const count = c.ds4_session_top_logprobs(session, top, max_rank);
    for (top[0..@intCast(count)], 0..) |entry, index| {
        if (entry.id == token) return @intCast(index + 1);
    }
    return 0;
}
pub fn header(file: ?*c.FILE, config: *const c.eval_config, model_name: [*c]const u8, count: c_int, max_prompt: c_int, max_generation: c_int) void {
    const trace = file orelse return;
    _ = c.fprintf(
        trace,
        "# ds4-eval trace\n" ++
            "started_unix: %lld\n" ++
            "model: %s\n" ++
            "model_shape: %s\n" ++
            "backend: %s\n" ++
            "suite: %s\n" ++
            "source_filter: %s\n" ++
            "domain_filter: %s\n" ++
            "case_id_filter: %s\n" ++
            "ctx: %d\n" ++
            "max_tokens: %d\n" ++
            "max_generation_tokens: %d\n" ++
            "max_prompt_tokens: %d\n" ++
            "questions: %d\n" ++
            "temperature: %.6g\n" ++
            "top_p: %.6g\n" ++
            "min_p: %.6g\n" ++
            "seed: %llu\n" ++
            "think_mode_requested: %s\n" ++
            "soft_limit_reply_budget: %d\n" ++
            "hard_limit_reply_budget: %d\n" ++
            "soft_limit_think_close_rank: %d\n" ++
            "\n",
        @as(c_longlong, @intCast(c.time(null))),
        config.model_path,
        if (model_name != null) model_name else @as([*c]const u8, "unknown"),
        c.ds4_backend_name(config.backend),
        Cases.suiteName(config.suite_mask),
        if (config.source_filter != null) config.source_filter else @as([*c]const u8, "*"),
        if (config.domain_filter != null) config.domain_filter else @as([*c]const u8, "*"),
        if (config.case_id_filter != null) config.case_id_filter else @as([*c]const u8, "*"),
        config.ctx_size,
        config.max_tokens,
        max_generation,
        max_prompt,
        count,
        @as(f64, config.temperature),
        @as(f64, config.top_p),
        @as(f64, config.min_p),
        @as(c_ulonglong, config.seed),
        c.ds4_think_mode_name(config.think_mode),
        config.soft_limit_reply_budget,
        config.hard_limit_reply_budget,
        config.soft_limit_think_close_rank,
    );
    _ = c.fflush(trace);
}
/// Named native event fields keep prompt/generated counts and timing domains
/// explicit at the formatted C I/O boundary.
pub const CaseEvent = struct {
    config: *const c.eval_config,
    case: *const c.eval_case,
    index: c_int,
    count: c_int,
    status: [*:0]const u8,
    errors: [*c]const u8,
    system: [*c]const u8,
    question: [*c]const u8,
    generated: []const u8,
    think: c.ds4_think_mode,
    prompt_tokens: c_int,
    generated_tokens: c_int,
    generation_budget: c_int,
    elapsed: f64,
    picked: [*c]const u8,
    close: ?*const CloseInfo,
};
pub fn writeCase(file: ?*c.FILE, event: CaseEvent) void {
    const config = event.config;
    const case = event.case;
    const trace = file orelse return;
    _ = c.fprintf(
        trace,
        "===== CASE %d/%d %s/%s =====\n" ++
            "timestamp_unix: %lld\n" ++
            "source: %s\n" ++
            "id: %s\n" ++
            "domain: %s\n" ++
            "title: %s\n" ++
            "status: %s\n" ++
            "picked: %s\n" ++
            "expected: %s\n" ++
            "prompt_tokens: %d\n" ++
            "generated_tokens: %d\n" ++
            "generation_budget: %d\n" ++
            "elapsed_sec: %.3f\n" ++
            "temperature: %.6g\n" ++
            "top_p: %.6g\n" ++
            "min_p: %.6g\n" ++
            "think_mode_effective: %s\n",
        event.index + 1,
        event.count,
        case.source,
        case.id,
        @as(c_longlong, @intCast(c.time(null))),
        case.source,
        case.id,
        case.domain,
        case.title,
        event.status,
        if (event.picked != null and event.picked[0] != 0) event.picked else @as([*c]const u8, "?"),
        case.answer,
        event.prompt_tokens,
        event.generated_tokens,
        event.generation_budget,
        event.elapsed,
        @as(f64, config.temperature),
        @as(f64, config.top_p),
        @as(f64, config.min_p),
        c.ds4_think_mode_name(event.think),
    );
    if (event.close != null and event.close.?.kind != c.EVAL_THINK_CLOSE_NONE) {
        const info = event.close.?;
        _ = c.fprintf(
            trace,
            "think_close: %s\nthink_close_token_index: %d\nthink_close_remaining_budget: %d\nthink_close_rank: %d\n",
            closeKind(info.kind),
            info.token_index,
            info.remaining_budget,
            info.rank,
        );
    } else _ = c.fprintf(trace, "think_close: none\n");
    if (event.errors != null and event.errors[0] != 0) _ = c.fprintf(trace, "error: %s\n", event.errors);
    const choice_count = Grading.choices(case);
    if (choice_count > 0) {
        _ = c.fprintf(trace, "choices:\n");
        for (0..@intCast(choice_count)) |choice| _ = c.fprintf(trace, "  %c. %s\n", @as(c_int, @intCast('A' + choice)), case.choice[choice]);
    }
    _ = c.fprintf(trace, "answer_kind: %s\n", Grading.kindName(Grading.answerKind(case)));
    block(trace, "SYSTEM_PROMPT", Common.text(event.system));
    block(trace, "QUESTION_PROMPT", Common.text(event.question));
    block(trace, "MODEL_OUTPUT", event.generated);
    _ = c.fputc('\n', trace);
    _ = c.fflush(trace);
}
