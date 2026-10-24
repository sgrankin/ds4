//! Common snapshot and staged-file envelopes. Model payload APIs own their
//! wire layouts and frontier invalidation; this module owns stdio lifetime,
//! error precedence and publishing a completed host snapshot/staged file.
const std = @import("std");
const c = @import("session_snapshot_c");
const Snapshot = c.ds4_session_snapshot;
const Staged = c.ds4_session_payload_file;
const Session = c.ds4_session;

pub const Errors = struct {
    pointer: [*c]u8,
    capacity: usize,

    fn fail(self: Errors, message: [:0]const u8) c_int {
        if (self.capacity != 0) _ = c.snprintf(self.pointer, self.capacity, "%s", message.ptr);
        return 1;
    }
};

const buffers = @import("snapshot_buffer.zig");
const Codec = struct {
    session: *Session,
    pub fn canonicalBytes(self: Codec) u64 {
        return c.ds4_session_payload_bytes(self.session);
    }
    pub fn saveCanonical(self: Codec, stream: *c.FILE, errors: []u8) c_int {
        return c.ds4_session_save_payload(self.session, stream, errors.ptr, errors.len);
    }
    pub fn loadCanonical(self: Codec, stream: *c.FILE, bytes: u64, errors: []u8) c_int {
        return c.ds4_session_load_payload(self.session, stream, bytes, errors.ptr, errors.len);
    }
};
pub fn saveSnapshot(session: ?*Session, snapshot: ?*Snapshot, errors: Errors) c_int {
    const target = session orelse return errors.fail("invalid session snapshot save");
    if (snapshot == null) return errors.fail("invalid session snapshot save");
    if (c.ds4_session_is_distributed(target)) return errors.fail("distributed session snapshots are not supported yet");
    return buffers.save(c, Codec{ .session = target }, snapshot, errors.pointer[0..errors.capacity]);
}
pub fn loadSnapshot(session: ?*Session, snapshot: ?*const Snapshot, errors: Errors) c_int {
    const target = session orelse return errors.fail("invalid session snapshot load");
    if (snapshot == null or snapshot.?.ptr == null or snapshot.?.len == 0) return errors.fail("invalid session snapshot load");
    if (c.ds4_session_is_distributed(target)) return errors.fail("distributed session snapshots are not supported yet");
    return buffers.load(c, Codec{ .session = target }, snapshot, errors.pointer[0..errors.capacity]);
}

pub fn stagePayload(session: ?*Session, maybe_output: ?*Staged, errors: Errors) c_int {
    const output = maybe_output orelse return errors.fail("invalid session payload staging request");
    output.* = std.mem.zeroes(Staged);
    const target = session orelse return errors.fail("session has no valid checkpoint to stage");
    if (!c.ds4_session_checkpoint_valid(target)) return errors.fail("session has no valid checkpoint to stage");

    // Preserve the existing transient file name/lifetime; cache persistence
    // policy and useful experiment workspaces remain callers' responsibility.
    var template = "/tmp/ds4-session-payload.XXXXXX".*;
    const fd = c.mkstemp(&template);
    if (fd < 0) return errors.fail("failed to create staged session payload");
    var keep = false;
    defer if (!keep) {
        _ = c.unlink(&template);
    };
    const stream = c.fdopen(fd, "wb") orelse {
        const saved = c.__error().*;
        _ = c.close(fd);
        if (errors.capacity != 0)
            _ = c.snprintf(errors.pointer, errors.capacity, "failed to open staged session payload: %s", c.strerror(saved));
        return 1;
    };

    var result = c.ds4_session_save_payload(target, stream, errors.pointer, errors.capacity);
    if (result == 0 and c.fflush(stream) != 0) result = errors.fail("failed to flush staged session payload");
    var bytes: c.off_t = -1;
    if (result == 0) {
        bytes = c.ftello(stream);
        if (bytes < 0) result = errors.fail("failed to measure staged session payload");
    }
    if (c.fclose(stream) != 0 and result == 0) result = errors.fail("failed to close staged session payload");
    if (result != 0) return 1;

    output.path = c.strdup(&template);
    if (output.path == null) {
        _ = c.fprintf(c.__stderrp, "ds4: out of memory\n");
        c.exit(1);
    }
    output.bytes = @intCast(bytes);
    keep = true;
    return 0;
}

pub fn freeSnapshot(snapshot: ?*Snapshot) void {
    buffers.free(c, snapshot);
}
pub fn freeStaged(maybe_staged: ?*Staged) void {
    const staged = maybe_staged orelse return;
    if (staged.path != null) {
        _ = c.unlink(staged.path);
        c.free(staged.path);
    }
    staged.* = std.mem.zeroes(Staged);
}
pub fn writeStaged(maybe_staged: ?*const Staged, maybe_stream: ?*c.FILE, errors: Errors) c_int {
    const staged = maybe_staged orelse return errors.fail("invalid staged session payload");
    const stream = maybe_stream orelse return errors.fail("invalid staged session payload");
    if (staged.path == null) return errors.fail("invalid staged session payload");
    const source = c.fopen(staged.path, "rb") orelse return errors.fail("failed to open staged session payload");
    const result = c.ds4_native_payload_copy_file(source, stream, staged.bytes, errors.pointer, errors.capacity);
    if (c.fclose(source) != 0 and result == 0) return errors.fail("failed to close staged session payload");
    return result;
}
