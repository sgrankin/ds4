//! One C ABI facade: native program entry plus the original C extractor tests.
//! Imported owners remain pure, so no implementation exports are duplicated.
const c = @import("eval_c");
const Frontend = @import("frontend.zig");
const Common = @import("common.zig");
const Grading = @import("grading.zig");
const Trace = @import("trace.zig");
const Cases = @import("cases.zig");
export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    return Frontend.main(argc, argv);
}
export fn ds4_eval_find_case_answer(case: *const c.eval_case, generated: [*c]const u8, output: [*c]u8, capacity: usize) void {
    Grading.extract(case, Common.text(generated), Common.output(output, capacity));
}
export fn ds4_eval_find_case_answer_strict(case: *const c.eval_case, generated: [*c]const u8, output: [*c]u8, capacity: usize) bool {
    return Grading.strict(case, Common.text(generated), Common.output(output, capacity));
}
export fn ds4_eval_answer_matches(case: *const c.eval_case, actual: [*c]const u8) bool {
    return Grading.matches(case, actual);
}
fn range(start: [*c]const u8, end: [*c]const u8) []const u8 {
    const length = @intFromPtr(end) - @intFromPtr(start);
    return if (length == 0) &.{} else start[0..length];
}
export fn ds4_eval_trace_find_next_case(start: [*c]const u8, end: [*c]const u8) [*c]const u8 {
    const position = Trace.nextCase(range(start, end)) orelse return null;
    return start + position;
}
export fn ds4_eval_trace_copy_model_output(start: [*c]const u8, end: [*c]const u8) [*c]u8 {
    return Trace.copyOutput(range(start, end));
}
export fn ds4_eval_regrade_case_outcome(case: *const c.eval_case, status: [*c]const u8, generated: [*c]const u8, output: [*c]u8, capacity: usize) c.regrade_outcome {
    return @intCast(@backingInt(Trace.outcome(case, Common.text(status), Common.text(generated), Common.output(output, capacity))));
}
export fn ds4_eval_validate_cases(verbose: bool) c_int {
    return Cases.validate(verbose);
}
