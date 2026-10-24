//! Common request controls over the one record embedded in a session. Callback
//! user data is borrowed for the bounded request, never retained by a model.
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
