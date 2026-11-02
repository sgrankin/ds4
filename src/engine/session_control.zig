//! Common request controls over the one record embedded in a session. Callback
//! user data is borrowed for the bounded request, never retained by a model.
const std = @import("std");
const c = @import("gguf_c");
const Control = c.ds4_session_control;

pub fn beginRequest(control: *Control) void {
    control.dspark_rewind_end = 0;
    control.dspark_sched_cycles = 0;
    control.dspark_sched_accepted = 0;
    control.dspark_sched_no_draft = 0;
    control.dspark_sched_extra_ms = 0;
    control.dspark_sched_saved_ms = 0;
    control.dspark_sched_skip = 0;
    control.dspark_sched_lifetime_accepted = 0;
    control.dspark_sched_life_extra_ms = 0;
    control.dspark_sched_life_saved_ms = 0;
    control.dspark_sched_skipped_cycle = false;
    control.dspark_sched_long_accept_seen = false;
    control.dspark_sched_bypass = false;
}

pub fn clearCapture(control: *Control) void {
    control.dspark_rewind_end = 0;
    control.dspark_draft_valid = false;
    control.dspark_draft_len = 0;
    control.dspark_tp_proposal = false;
}

pub fn cancelled(control: *const Control) bool {
    return if (control.cancel) |callback| callback(control.cancel_ud) else false;
}

pub fn report(control: *const Control, event: [*c]const u8, current: c_int, total: c_int) void {
    if (event != null) if (control.progress) |callback| callback(control.progress_ud, event, current, total);
}

fn schedulerEnabled() bool {
    const value = c.getenv("DS4_DSPARK_SCHEDULER");
    return value == null or value[0] == 0 or !std.mem.eql(u8, std.mem.span(value), "0");
}
fn schedulerSetting(name: [*:0]const u8, fallback: u32) u32 {
    const value = c.getenv(name);
    if (value == null or value[0] == 0) return fallback;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const number = c.strtoul(value, &end, 10);
    return if (end == value or c.__error().* != 0 or number > std.math.maxInt(u32)) fallback else @intCast(number);
}

/// DS41 uses the shared scheduler's count-only path (extra_ms=0 upstream).
/// Its performance decision never depends on wall-clock timing or GPU order.
pub fn ds41ShouldSkip(control: *Control, statistics: *c.ds4_dspark_spec_stats) bool {
    if (!schedulerEnabled()) return false;
    control.dspark_sched_skipped_cycle = false;
    if (!control.dspark_sched_bypass and control.dspark_sched_skip == 0) return false;
    if (!control.dspark_sched_bypass) control.dspark_sched_skip -= 1;
    control.dspark_sched_skipped_cycle = true;
    statistics.scheduler_skips +%= 1;
    if (c.getenv("DS4_DSPARK_SPEC_LOG") != null)
        _ = c.fprintf(c.__stderrp, "ds4: DSpark scheduler skip remaining=%u\n", control.dspark_sched_skip);
    return true;
}

pub fn ds41Note(control: *Control, accepted: u32, no_draft: bool, default_skip: u32) void {
    if (!schedulerEnabled()) return;
    if (control.dspark_sched_skipped_cycle) {
        control.dspark_sched_skipped_cycle = false;
        return;
    }
    control.dspark_sched_cycles +%= 1;
    control.dspark_sched_accepted +%= accepted;
    control.dspark_sched_lifetime_accepted +|= accepted;
    if (accepted > 2) control.dspark_sched_long_accept_seen = true;
    if (no_draft) control.dspark_sched_no_draft +%= 1;
    const no_draft_skip = schedulerSetting("DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", 3);
    if (no_draft and no_draft_skip != 0) {
        var pause = no_draft_skip;
        if (control.dspark_sched_lifetime_accepted != 0 and !control.dspark_sched_long_accept_seen)
            pause = @max(pause, schedulerSetting("DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", 4));
        control.dspark_sched_skip = @max(control.dspark_sched_skip, pause);
    }
    const configured_window = schedulerSetting("DS4_DSPARK_SCHEDULER_WINDOW", 4);
    const window = if (configured_window == 0) 4 else configured_window;
    if (control.dspark_sched_cycles < window) return;
    const average = @as(u64, control.dspark_sched_accepted) * 1000 / control.dspark_sched_cycles;
    const low_accept = average < schedulerSetting("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", 1500);
    const many_no_draft = @as(u64, control.dspark_sched_no_draft) * 2 >= control.dspark_sched_cycles;
    if (low_accept or many_no_draft) {
        control.dspark_sched_skip = schedulerSetting("DS4_DSPARK_SCHEDULER_SKIP", default_skip);
        if (many_no_draft)
            control.dspark_sched_skip = @max(control.dspark_sched_skip, schedulerSetting("DS4_DSPARK_SCHEDULER_SLOW_SKIP", 4));
        if (c.getenv("DS4_DSPARK_SPEC_LOG") != null)
            _ = c.fprintf(c.__stderrp, "ds4: DSpark scheduler pause cycles=%u accepted=%u skip=%u\n", control.dspark_sched_cycles, control.dspark_sched_accepted, control.dspark_sched_skip);
    }
    control.dspark_sched_cycles = 0;
    control.dspark_sched_accepted = 0;
    control.dspark_sched_no_draft = 0;
    control.dspark_sched_extra_ms = 0;
    control.dspark_sched_saved_ms = 0;
}

test "DS41 count scheduler pauses low acceptance and retries after bounded skips" {
    const names = [_][*:0]const u8{
        "DS4_DSPARK_SCHEDULER",               "DS4_DSPARK_SCHEDULER_WINDOW",
        "DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", "DS4_DSPARK_SCHEDULER_SKIP",
        "DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", "DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP",
        "DS4_DSPARK_SCHEDULER_SLOW_SKIP",
    };
    const settings = [_][*:0]const u8{ "1", "4", "1500", "2", "3", "4", "4" };
    var saved: [names.len]?[:0]u8 = @splat(null);
    for (names, 0..) |name, index| {
        const value = c.getenv(name);
        if (value != null) saved[index] = try std.testing.allocator.dupeSentinel(u8, std.mem.span(value), 0);
    }
    defer for (names, saved) |name, old| {
        if (old) |value| {
            _ = c.setenv(name, value.ptr, 1);
            std.testing.allocator.free(value);
        } else _ = c.unsetenv(name);
    };
    for (names, settings) |name, value| try std.testing.expectEqual(@as(c_int, 0), c.setenv(name, value, 1));
    var control = std.mem.zeroes(Control);
    var stats = std.mem.zeroes(c.ds4_dspark_spec_stats);
    for (0..4) |_| ds41Note(&control, 1, false, 2);
    try std.testing.expectEqual(@as(u32, 2), control.dspark_sched_skip);
    try std.testing.expect(ds41ShouldSkip(&control, &stats));
    try std.testing.expect(ds41ShouldSkip(&control, &stats));
    try std.testing.expect(!ds41ShouldSkip(&control, &stats));
    try std.testing.expectEqual(@as(u64, 2), stats.scheduler_skips);
    ds41Note(&control, 0, true, 2);
    try std.testing.expectEqual(@as(u32, 4), control.dspark_sched_skip);
    beginRequest(&control);
    for (0..4) |_| ds41Note(&control, 3, false, 2);
    try std.testing.expectEqual(@as(u32, 0), control.dspark_sched_skip);
    try std.testing.expectEqual(@as(u32, 12), control.dspark_sched_lifetime_accepted);
    try std.testing.expect(control.dspark_sched_long_accept_seen);
    try std.testing.expectEqual(@as(c_int, 0), c.setenv("DS4_DSPARK_SCHEDULER", "0", 1));
    control.dspark_sched_skip = 8;
    try std.testing.expect(!ds41ShouldSkip(&control, &stats));
    ds41Note(&control, 0, true, 2);
    try std.testing.expectEqual(@as(u32, 8), control.dspark_sched_skip);
}
