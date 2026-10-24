//! Reserve the next graph plus every not-yet-created server slot. Existing
//! live graphs remain charged, and arithmetic saturates before admission.
pub fn graphBudget(live_bytes: u64, live_count: u32, session_hint: c_int, next: u64) u64 {
    const uncreated = if (session_hint > 0 and @as(u32, @intCast(session_hint)) > live_count)
        @as(u32, @intCast(session_hint)) - live_count
    else
        1;
    return live_bytes +| (next *| uncreated);
}
