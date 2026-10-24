//! Shared canonical memory envelope. The extra fmemopen terminator byte and
//! close/error precedence apply to public snapshots and private suffix captures.
const std = @import("std");
fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return 1;
}
pub fn save(comptime c: type, codec: anytype, maybe_snapshot: ?*c.ds4_session_snapshot, errors: []u8) c_int {
    const snapshot = maybe_snapshot orelse return fail(errors, "invalid session snapshot save");
    const bytes = codec.canonicalBytes();
    if (bytes == 0) return fail(errors, "session has no valid checkpoint to snapshot");
    if (bytes >= std.math.maxInt(usize)) return fail(errors, "session snapshot is too large for this platform");

    // fmemopen appends NUL even in binary mode. Keep its terminator outside
    // the payload so fclose cannot overwrite the final cache-tensor byte.
    const capacity = bytes + 1;
    if (snapshot.cap < capacity) {
        const memory = c.realloc(snapshot.ptr, @intCast(capacity)) orelse
            return fail(errors, "out of memory while allocating session snapshot");
        snapshot.ptr = @ptrCast(memory);
        snapshot.cap = capacity;
    }
    const stream = c.fmemopen(snapshot.ptr, @intCast(capacity), "wb") orelse
        return fail(errors, "failed to open memory stream for session snapshot");
    const result = codec.saveCanonical(stream, errors);
    if (c.fclose(stream) != 0 and result == 0) return fail(errors, "failed to finalize memory session snapshot");
    if (result != 0) return 1;
    snapshot.len = bytes;
    return 0;
}

pub fn load(comptime c: type, codec: anytype, maybe_snapshot: ?*const c.ds4_session_snapshot, errors: []u8) c_int {
    const snapshot = maybe_snapshot orelse return fail(errors, "invalid session snapshot load");
    if (snapshot.ptr == null or snapshot.len == 0) return fail(errors, "invalid session snapshot load");
    if (snapshot.len > std.math.maxInt(usize)) return fail(errors, "session snapshot is too large for this platform");
    const stream = c.fmemopen(snapshot.ptr, @intCast(snapshot.len), "rb") orelse
        return fail(errors, "failed to open memory stream for session snapshot restore");
    const result = codec.loadCanonical(stream, snapshot.len, errors);
    // Closing can fail after a completed model restore. Preserve that model's
    // checkpoint publication, reporting only this envelope's stdio error.
    if (c.fclose(stream) != 0 and result == 0) return fail(errors, "failed to close memory session snapshot");
    return result;
}

pub fn free(comptime c: type, snapshot: ?*c.ds4_session_snapshot) void {
    const target = snapshot orelse return;
    c.free(target.ptr);
    target.* = std.mem.zeroes(c.ds4_session_snapshot);
}
