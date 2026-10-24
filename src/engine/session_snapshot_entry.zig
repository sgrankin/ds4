//! Common CPU-only snapshot/staged-file entries. No graph/GPU owner is pulled
//! into model-free fixtures; the opaque session is served by its public API.
const c = @import("session_snapshot_c");
const snapshots = @import("session_snapshot.zig");
fn errors(pointer: [*c]u8, capacity: usize) snapshots.Errors {
    return .{ .pointer = pointer, .capacity = capacity };
}
export fn ds4_native_session_save_snapshot(
    session: ?*c.ds4_session,
    snapshot: ?*c.ds4_session_snapshot,
    message: [*c]u8,
    capacity: usize,
) c_int {
    return snapshots.saveSnapshot(session, snapshot, errors(message, capacity));
}
export fn ds4_native_session_load_snapshot(
    session: ?*c.ds4_session,
    snapshot: ?*const c.ds4_session_snapshot,
    message: [*c]u8,
    capacity: usize,
) c_int {
    return snapshots.loadSnapshot(session, snapshot, errors(message, capacity));
}
export fn ds4_native_session_stage_payload(
    session: ?*c.ds4_session,
    output: ?*c.ds4_session_payload_file,
    message: [*c]u8,
    capacity: usize,
) c_int {
    return snapshots.stagePayload(session, output, errors(message, capacity));
}
export fn ds4_session_snapshot_free(snapshot: ?*c.ds4_session_snapshot) void {
    snapshots.freeSnapshot(snapshot);
}
export fn ds4_session_payload_file_free(staged: ?*c.ds4_session_payload_file) void {
    snapshots.freeStaged(staged);
}
export fn ds4_session_write_staged_payload(
    staged: ?*const c.ds4_session_payload_file,
    stream: ?*c.FILE,
    message: [*c]u8,
    capacity: usize,
) c_int {
    return snapshots.writeStaged(staged, stream, errors(message, capacity));
}
