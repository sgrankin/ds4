//! Borrow the original core/hard literal data. Selection owns only its copied
//! record vector; question strings and aliases live for the program lifetime.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Grade = @import("grading.zig");
const Case = c.eval_case;
fn records(pointer: [*c]const Case, length: usize) []const Case {
    return if (length == 0) &.{} else @as([*]const Case, @ptrCast(pointer))[0..length];
}
pub fn core() []const Case {
    return records(c.ds4_eval_core_data, c.ds4_eval_core_case_count);
}
pub fn hard() []const Case {
    return records(c.ds4_eval_hard_data, c.eval_hard_case_count);
}
pub fn at(index: usize) ?*const Case {
    const initial = core();
    if (index < initial.len) return &initial[index];
    const extra = hard();
    const offset = index - initial.len;
    return if (offset < extra.len) &extra[offset] else null;
}
pub fn find(source: [*c]const u8, id: [*c]const u8) ?*const Case {
    for ([_][]const Case{ core(), hard() }) |suite| for (suite) |*case| {
        if (case.source != null and case.id != null and c.strcmp(case.source, source) == 0 and c.strcmp(case.id, id) == 0) return case;
    };
    return null;
}
pub fn suiteName(mask: u32) [*:0]const u8 {
    return switch (mask) {
        c.EVAL_SUITE_CORE => "core",
        c.EVAL_SUITE_HARD => "hard",
        c.EVAL_SUITE_CORE | c.EVAL_SUITE_HARD => "all",
        c.EVAL_SUITE_HARD_SMOKE => "hard-smoke",
        else => "custom",
    };
}
fn matches(case: *const Case, options: *const c.eval_config) bool {
    return !(options.source_filter != null and c.strcasecmp(case.source, options.source_filter) != 0) and
        !(options.domain_filter != null and c.strcasecmp(case.domain, options.domain_filter) != 0) and
        !(options.case_id_filter != null and c.strcmp(case.id, options.case_id_filter) != 0);
}
/// Return a libc-freeable vector, with count published only on success.
pub fn select(options: *const c.eval_config, count_out: *c_int) [*c]Case {
    const initial = core();
    const extra = hard();
    const capacity = initial.len + extra.len;
    const storage = c.calloc(@max(capacity, 1), @sizeOf(Case)) orelse {
        _ = c.fprintf(c.__stderrp, "ds4-eval: out of memory while selecting cases\n");
        return null;
    };
    const selected: [*]Case = @ptrCast(@alignCast(storage));
    var count: usize = 0;
    if (options.suite_mask & c.EVAL_SUITE_CORE != 0) for (initial) |*case| {
        if (matches(case, options)) {
            selected[count] = case.*;
            count += 1;
        }
    };
    if (options.suite_mask & (c.EVAL_SUITE_HARD | c.EVAL_SUITE_HARD_SMOKE) != 0) for (extra) |*case| {
        if (options.suite_mask & c.EVAL_SUITE_HARD_SMOKE != 0 and options.suite_mask & c.EVAL_SUITE_HARD == 0 and case.suites & c.EVAL_SUITE_HARD_SMOKE == 0) continue;
        if (matches(case, options)) {
            selected[count] = case.*;
            count += 1;
        }
    };
    if (options.question_limit > @as(c_int, @intCast(count))) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: only %d questions match the selected suite and filters\n", @as(c_int, @intCast(count)));
        c.free(storage);
        return null;
    }
    if (options.question_limit > 0 and options.question_limit < @as(c_int, @intCast(count))) count = @intCast(options.question_limit);
    if (count == 0) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: no questions match the selected suite and filters\n");
        c.free(storage);
        return null;
    }
    count_out.* = @intCast(count);
    return selected;
}
fn sourceDeclared(source: [*c]const u8) bool {
    if (c.eval_hard_source_count == 0) return false;
    const sources: [*]const c.eval_source = @ptrCast(c.ds4_eval_source_data);
    for (sources[0..c.eval_hard_source_count]) |item| if (c.strcmp(source, item.name) == 0) {
        return true;
    };
    return false;
}
pub fn validate(verbose: bool) c_int {
    const initial = core();
    const extra = hard();
    const total = initial.len + extra.len;
    var errors: c_int = 0;
    var smoke: usize = 0;
    for (0..total) |index| {
        const case = at(index).?;
        const is_hard = index >= initial.len;
        const where: [*:0]const u8 = if (is_hard) "hard" else "core";
        inline for (.{ "source", "id", "domain", "title", "question", "answer" }) |name| {
            const value = @field(case.*, name);
            if (value == null or value[0] == 0) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: %s case %zu has no %s\n", where, index + 1, @as([*:0]const u8, name));
                errors += 1;
            }
        }
        if (is_hard and case.suites & c.EVAL_SUITE_HARD_SMOKE != 0) smoke += 1;
        if (case.source == null or case.id == null or case.answer == null) continue;
        if (case.answer_kind < c.EVAL_ANSWER_AUTO or case.answer_kind > c.EVAL_ANSWER_LINE_SET) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has an invalid answer kind\n", case.source, case.id);
            errors += 1;
            continue;
        }
        const count = Grade.choices(case);
        var gap = false;
        for (case.choice) |choice| {
            if (choice == null) gap = true else if (gap) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has a gap in its choices\n", case.source, case.id);
                errors += 1;
                break;
            }
        }
        const kind = Grade.answerKind(case);
        if (kind == c.EVAL_ANSWER_CHOICE) {
            if (count < 2 or c.strlen(case.answer) != 1 or case.answer[0] < 'A' or case.answer[0] >= 'A' + count) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has an invalid choice key\n", case.source, case.id);
                errors += 1;
            }
        } else if (count != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has choices but is not choice-graded\n", case.source, case.id);
            errors += 1;
        } else if (kind == c.EVAL_ANSWER_LINE_SET) {
            var lines: [256]bool = @splat(false);
            if (!Grade.parseLineSet(case.answer, &lines)) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has an invalid line-set key\n", case.source, case.id);
                errors += 1;
            }
        } else {
            var normalized: [c.EVAL_ANSWER_MAX]u8 = undefined;
            Grade.normalizeExpected(kind, Common.text(case.answer), &normalized);
            if (normalized[0] == 0 or std.mem.eql(u8, Common.written(&normalized), "?")) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: %s/%s has an invalid answer key\n", case.source, case.id);
                errors += 1;
            }
        }
        if (is_hard and (case.answer_kind == c.EVAL_ANSWER_AUTO or case.suites & c.EVAL_SUITE_HARD == 0 or case.max_tokens == 0 or !sourceDeclared(case.source))) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: hard case %s/%s lacks type, suite, budget, or source metadata\n", case.source, case.id);
            errors += 1;
        }
        for (index + 1..total) |other_index| {
            const other = at(other_index).?;
            if (other.source != null and other.id != null and c.strcmp(case.source, other.source) == 0 and c.strcmp(case.id, other.id) == 0) {
                _ = c.fprintf(c.__stderrp, "ds4-eval: duplicate case ID %s/%s\n", case.source, case.id);
                errors += 1;
            }
        }
    }
    if (extra.len != 50 or smoke != 12) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: hard suite shape changed: cases=%zu smoke=%zu (expected 50/12)\n", extra.len, smoke);
        errors += 1;
    }
    if (verbose or errors != 0) _ = c.printf(
        "ds4-eval: validated %zu core and %zu hard cases (%zu hard-smoke): %s\n",
        initial.len,
        extra.len,
        smoke,
        @as([*:0]const u8, if (errors != 0) "FAILED" else "ok"),
    );
    return @intFromBool(errors != 0);
}
pub fn list(selected: []const Case) void {
    _ = c.printf("#\tsource\tdomain\tid\tanswer-kind\ttitle\n");
    for (selected, 0..) |*case, index| _ = c.printf(
        "%d\t%s\t%s\t%s\t%s\t%s\n",
        @as(c_int, @intCast(index + 1)),
        case.source,
        case.domain,
        case.id,
        Grade.kindName(Grade.answerKind(case)),
        case.title,
    );
}
