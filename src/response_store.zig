//! Raw Responses JSON files, independent of model/cache lifetime.
//! Same-directory rename publishes a whole old/new inode; callers own JSON
//! parsing and libc read buffers. No engine or request records cross this API.
const std = @import("std");
const c = @import("response_store_c");

fn errnoValue() c_int {
    return c.__error().*;
}
export fn ds4_response_store_valid_id(id: [*c]const u8) bool {
    if (id == null) return false;
    const text = std.mem.span(id);
    if (text.len <= 5 or text.len > c.DS4_RESPONSE_STORE_ID_MAX or !std.mem.startsWith(u8, text, "resp_")) return false;
    for (text[5..]) |byte| {
        if (!(std.ascii.isAlphanumeric(byte) or byte == '_')) return false;
    }
    return true;
}
export fn ds4_response_store_error(err: [*c]u8, errlen: usize, operation: [*c]const u8, code: c_int) c_int {
    if (err != null and errlen != 0) _ = c.snprintf(err, errlen, "%s: %s", operation, c.strerror(code));
    return -1;
}
fn clearError(err: [*c]u8, length: usize) void {
    if (err != null and length != 0) err[0] = 0;
}

export fn ds4_response_store_path(path: [*c]u8, directory: [*c]const u8, id: [*c]const u8, temporary: bool, err: [*c]u8, errlen: usize) c_int {
    if (directory == null or directory[0] == 0 or !ds4_response_store_valid_id(id))
        return ds4_response_store_error(err, errlen, "response store path", c.EINVAL);
    const count = c.snprintf(path, c.PATH_MAX, if (temporary) "%s/.%s.XXXXXX" else "%s/%s.json", directory, id);
    if (count < 0 or count >= c.PATH_MAX)
        return ds4_response_store_error(err, errlen, "response store path", c.ENAMETOOLONG);
    return 0;
}

export fn ds4_response_store_init(directory: [*c]const u8, err: [*c]u8, errlen: usize) c_int {
    clearError(err, errlen);
    if (directory == null or directory[0] == 0)
        return ds4_response_store_error(err, errlen, "response store directory", c.EINVAL);
    if (c.mkdir(directory, 0o700) != 0 and errnoValue() != c.EEXIST)
        return ds4_response_store_error(err, errlen, "mkdir response store", errnoValue());
    var status: c.struct_stat = undefined;
    if (c.stat(directory, &status) != 0)
        return ds4_response_store_error(err, errlen, "stat response store", errnoValue());
    if (status.st_mode & c.S_IFMT != c.S_IFDIR)
        return ds4_response_store_error(err, errlen, "response store directory", c.ENOTDIR);
    if (c.chmod(directory, 0o700) != 0)
        return ds4_response_store_error(err, errlen, "chmod response store", errnoValue());
    return 0;
}

export fn ds4_response_store_write(directory: [*c]const u8, id: [*c]const u8, json: [*c]const u8, length: usize, err: [*c]u8, errlen: usize) c_int {
    clearError(err, errlen);
    if ((json == null and length != 0) or length > c.DS4_RESPONSE_STORE_MAX_BYTES)
        return ds4_response_store_error(err, errlen, "response store payload", c.EINVAL);
    var path: [c.PATH_MAX]u8 = undefined;
    var temporary: [c.PATH_MAX]u8 = undefined;
    if (ds4_response_store_path(&path, directory, id, false, err, errlen) != 0 or
        ds4_response_store_path(&temporary, directory, id, true, err, errlen) != 0) return -1;
    var descriptor = c.mkstemp(&temporary);
    if (descriptor < 0)
        return ds4_response_store_error(err, errlen, "create response temporary", errnoValue());
    var published = false;
    defer {
        if (descriptor >= 0) _ = c.close(descriptor);
        if (!published) _ = c.unlink(&temporary);
    }
    if (c.fchmod(descriptor, 0o600) != 0)
        return ds4_response_store_error(err, errlen, "write response store", errnoValue());
    var done: usize = 0;
    while (done < length) {
        const written = c.write(descriptor, json + done, length - done);
        if (written < 0) {
            if (errnoValue() == c.EINTR) continue;
            return ds4_response_store_error(err, errlen, "write response store", errnoValue());
        }
        if (written == 0) return ds4_response_store_error(err, errlen, "write response store", c.EIO);
        done += @intCast(written);
    }
    while (c.fsync(descriptor) != 0) {
        if (errnoValue() != c.EINTR)
            return ds4_response_store_error(err, errlen, "write response store", errnoValue());
    }
    const closed = c.close(descriptor);
    descriptor = -1; // Do not retry close, including its failure path.
    if (closed != 0) return ds4_response_store_error(err, errlen, "write response store", errnoValue());
    if (c.rename(&temporary, &path) != 0)
        return ds4_response_store_error(err, errlen, "write response store", errnoValue());
    published = true;
    return 0;
}

export fn ds4_response_store_read(directory: [*c]const u8, id: [*c]const u8, maybe_json: ?*[*c]u8, maybe_length: ?*usize, err: [*c]u8, errlen: usize) c_int {
    clearError(err, errlen);
    if (maybe_json) |json| json.* = null;
    if (maybe_length) |length| length.* = 0;
    if (maybe_json == null or maybe_length == null)
        return ds4_response_store_error(err, errlen, "response read outputs", c.EINVAL);
    var path: [c.PATH_MAX]u8 = undefined;
    if (ds4_response_store_path(&path, directory, id, false, err, errlen) != 0) return -1;
    var descriptor = c.open(&path, c.O_RDONLY);
    if (descriptor < 0) return if (errnoValue() == c.ENOENT) 1 else ds4_response_store_error(err, errlen, "open response store", errnoValue());
    var memory: ?*anyopaque = null;
    defer {
        if (descriptor >= 0) _ = c.close(descriptor);
        c.free(memory);
    }
    var status: c.struct_stat = undefined;
    if (c.fstat(descriptor, &status) != 0)
        return ds4_response_store_error(err, errlen, "read response store", errnoValue());
    if (status.st_mode & c.S_IFMT != c.S_IFREG or status.st_size < 0 or
        @as(u64, @intCast(status.st_size)) > c.DS4_RESPONSE_STORE_MAX_BYTES)
        return ds4_response_store_error(err, errlen, "read response store", c.EFBIG);
    const size: usize = @intCast(status.st_size);
    memory = c.malloc(size + 1) orelse
        return ds4_response_store_error(err, errlen, "read response store", c.ENOMEM);
    const buffer: [*]u8 = @ptrCast(memory.?);
    var done: usize = 0;
    while (done < size) {
        const received = c.read(descriptor, buffer + done, size - done);
        if (received < 0) {
            if (errnoValue() == c.EINTR) continue;
            return ds4_response_store_error(err, errlen, "read response store", errnoValue());
        }
        if (received == 0) return ds4_response_store_error(err, errlen, "read response store", c.EIO);
        done += @intCast(received);
    }
    const closed = c.close(descriptor);
    descriptor = -1;
    if (closed != 0) return ds4_response_store_error(err, errlen, "read response store", errnoValue());
    buffer[size] = 0;
    maybe_json.?.* = buffer;
    maybe_length.?.* = size;
    memory = null; // Transfer libc ownership only after a complete successful read.
    return 0;
}

export fn ds4_response_store_delete(directory: [*c]const u8, id: [*c]const u8, err: [*c]u8, errlen: usize) c_int {
    clearError(err, errlen);
    var path: [c.PATH_MAX]u8 = undefined;
    if (ds4_response_store_path(&path, directory, id, false, err, errlen) != 0) return -1;
    if (c.unlink(&path) != 0) return if (errnoValue() == c.ENOENT) 1 else ds4_response_store_error(err, errlen, "delete response store", errnoValue());
    return 0;
}
