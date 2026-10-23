//! Agent file transactions and exact edit anchors; no worker or model state.
//! C strings remain the path/error ABI, while file contents are bounded bytes.
const std = @import("std");
const c = @import("agent_file_c");
const file_limit: usize = c.DS4_AGENT_FILE_MAX_BYTES;

fn errnoValue() c_int {
    return c.__error().*;
}
fn setErrno(value: c_int) void {
    c.__error().* = value;
}
fn text(pointer: [*c]const u8) []const u8 {
    return std.mem.span(pointer);
}
fn bytes(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
fn grow(pointer: [*c]u8, capacity: usize) [*c]u8 {
    return @ptrCast(c.realloc(pointer, @max(capacity, 1)) orelse {
        c.perror("ds4-agent: realloc");
        c.exit(1);
    });
}

export fn ds4_agent_file_open_regular(path: [*c]const u8) ?*c.FILE {
    const descriptor = c.open(path, c.O_RDONLY | c.O_NONBLOCK | c.O_CLOEXEC);
    if (descriptor < 0) return null;
    var status: c.struct_stat = undefined;
    const result = c.fstat(descriptor, &status);
    if (result != 0 or status.st_mode & c.S_IFMT != c.S_IFREG) {
        const saved = if (result != 0) errnoValue() else c.EINVAL;
        _ = c.close(descriptor);
        setErrno(saved);
        return null;
    }
    const stream = c.fdopen(descriptor, "rb");
    if (stream == null) {
        const saved = errnoValue();
        _ = c.close(descriptor);
        setErrno(saved);
    }
    return stream;
}

export fn ds4_agent_file_read_bytes(path: [*c]const u8, output: *[*c]u8, length: *usize, err: [*c]u8, errlen: usize) c_int {
    const stream = ds4_agent_file_open_regular(path) orelse {
        _ = c.snprintf(err, errlen, "open %s: %s", path, c.strerror(errnoValue()));
        return -1;
    };
    var buffer: [*c]u8 = null;
    defer {
        _ = c.fclose(stream);
        c.free(buffer);
    }
    var used: usize = 0;
    var capacity: usize = 0;
    var chunk: [8192]u8 = undefined;
    while (true) {
        const count = c.fread(&chunk, 1, chunk.len, stream);
        if (count != 0) {
            if (used + count > file_limit) {
                _ = c.snprintf(err, errlen, "file too large: %s exceeds %d bytes", path, @as(c_int, @intCast(file_limit)));
                return -1;
            }
            if (used + count + 1 > capacity) {
                capacity = if (capacity == 0) 8192 else capacity * 2;
                while (capacity < used + count + 1) capacity *= 2;
                buffer = grow(buffer, capacity);
            }
            @memcpy(buffer[used..][0..count], chunk[0..count]);
            used += count;
            buffer[used] = 0;
        }
        if (count < chunk.len) {
            if (c.ferror(stream) != 0) {
                _ = c.snprintf(err, errlen, "read %s: %s", path, c.strerror(errnoValue()));
                return -1;
            }
            break;
        }
    }
    if (buffer == null) {
        // Even an empty result transfers a libc-owned trailing-NUL string.
        buffer = @ptrCast(c.malloc(1) orelse {
            c.perror("ds4-agent: malloc");
            c.exit(1);
        });
        buffer[0] = 0;
    }
    output.* = buffer;
    length.* = used;
    buffer = null; // Transfer only on success; failures leave outputs untouched.
    return 0;
}

fn sameVersion(a: *const c.struct_stat, b: *const c.struct_stat) bool {
    return a.st_dev == b.st_dev and a.st_ino == b.st_ino and
        a.st_size == b.st_size and a.st_nlink == b.st_nlink and
        a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec and a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec and
        a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec and a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
}
fn failed(path: [*c]const u8, err: [*c]u8, errlen: usize) c_int {
    _ = c.snprintf(err, errlen, "replace %s: %s", path, c.strerror(errnoValue()));
    return -1;
}
fn changed(path: [*c]const u8, err: [*c]u8, errlen: usize) c_int {
    _ = c.snprintf(err, errlen, "file changed while editing; read it again: %s", path);
    return -1;
}

// Commit only complete files. Follow existing symlinks, but refuse hard links:
// atomic rename cannot preserve their shared-inode semantics. Version checks
// catch concurrent edits, although unrelated writers do not take a lock.
export fn ds4_agent_file_replace(path: [*c]const u8, data: [*c]const u8, length: usize, expected: [*c]const u8, expected_length: usize, err: [*c]u8, errlen: usize) c_int {
    var target: [c.PATH_MAX]u8 = undefined;
    var temporary: [c.PATH_MAX]u8 = @splat(0);
    var before: c.struct_stat = undefined;
    var source: c_int = -1;
    var destination: c_int = -1;
    defer {
        if (destination >= 0) _ = c.close(destination);
        if (source >= 0) _ = c.close(source);
        if (temporary[0] != 0) _ = c.unlink(&temporary);
    }
    const exists = c.lstat(path, &before) == 0;
    if (!exists and errnoValue() != c.ENOENT) return failed(path, err, errlen);
    if (exists) {
        if (c.realpath(path, &target) == null) return failed(path, err, errlen);
        source = c.open(&target, c.O_RDWR | c.O_NONBLOCK | c.O_NOFOLLOW);
        if (source < 0 or c.fstat(source, &before) != 0) return failed(path, err, errlen);
        if (before.st_mode & c.S_IFMT != c.S_IFREG or before.st_nlink != 1) {
            _ = c.snprintf(err, errlen, "refusing to replace non-regular or hard-linked file: %s", path);
            return -1;
        }
        if (expected != null) {
            if (before.st_size < 0 or @as(u64, @intCast(before.st_size)) != expected_length) return changed(path, err, errlen);
            var chunk: [8192]u8 = undefined;
            var position: usize = 0;
            while (position < expected_length) {
                const count = @min(expected_length - position, chunk.len);
                const read = c.read(source, &chunk, count);
                if (read < 0 and errnoValue() == c.EINTR) continue;
                if (read < 0) return failed(path, err, errlen);
                const received: usize = @intCast(read);
                if (received == 0 or !std.mem.eql(u8, chunk[0..received], expected[position..][0..received])) return changed(path, err, errlen);
                position += received;
            }
        }
    } else {
        if (expected != null) return changed(path, err, errlen);
        if (c.snprintf(&target, target.len, "%s", path) >= @as(c_int, @intCast(target.len))) {
            setErrno(c.ENAMETOOLONG);
            return failed(path, err, errlen);
        }
    }
    if (c.snprintf(&temporary, temporary.len, "%s.ds4-XXXXXX", &target) >= @as(c_int, @intCast(temporary.len))) {
        temporary[0] = 0;
        setErrno(c.ENAMETOOLONG);
        return failed(path, err, errlen);
    }
    destination = c.mkstemp(&temporary);
    if (destination < 0) {
        temporary[0] = 0;
        return failed(path, err, errlen);
    }
    if (!exists) {
        // Let open apply the process umask without changing it in this thread.
        if (c.unlink(&temporary) != 0) return failed(path, err, errlen);
        _ = c.close(destination);
        destination = c.open(&temporary, c.O_WRONLY | c.O_CREAT | c.O_EXCL, @as(c_int, 0o666));
        if (destination < 0) {
            temporary[0] = 0;
            return failed(path, err, errlen);
        }
    }
    var position: usize = 0;
    while (position < length) {
        const written = c.write(destination, data + position, length - position);
        if (written < 0 and errnoValue() == c.EINTR) continue;
        if (written <= 0) {
            if (written == 0) setErrno(c.EIO);
            return failed(path, err, errlen);
        }
        position += @intCast(written);
    }
    if (exists) {
        if (c.fchown(destination, before.st_uid, before.st_gid) != 0) return failed(path, err, errlen);
        if (c.fchmod(destination, before.st_mode & 0o7777) != 0) return failed(path, err, errlen);
        // A metadata-copy failure aborts publication rather than losing ACLs or
        // extended attributes. The original Linux fallback stays in C.
        if (c.fcopyfile(source, destination, null, c.COPYFILE_ACL | c.COPYFILE_XATTR) != 0) return failed(path, err, errlen);
    }
    if (c.fsync(destination) != 0) return failed(path, err, errlen);
    if (c.close(destination) != 0) {
        destination = -1;
        return failed(path, err, errlen);
    }
    destination = -1;
    if (exists) {
        var resolved: [c.PATH_MAX]u8 = undefined;
        var after: c.struct_stat = undefined;
        if (c.realpath(path, &resolved) == null or !std.mem.eql(u8, text(&resolved), text(&target)) or
            c.stat(&target, &after) != 0 or !sameVersion(&before, &after)) return changed(path, err, errlen);
        if (c.rename(&temporary, &target) != 0) return failed(path, err, errlen);
    } else {
        // Unlike rename, link will not overwrite a concurrently created file.
        if (c.link(&temporary, &target) != 0) return failed(path, err, errlen);
        _ = c.unlink(&temporary);
    }
    temporary[0] = 0;
    return 0;
}

fn findUnique(data: []const u8, needle: []const u8, start: usize, after_head: bool, label: [*c]const u8, err: [*c]u8, errlen: usize) ?usize {
    if (needle.len == 0) {
        _ = c.snprintf(err, errlen, "%s anchor is empty", label);
        return null;
    }
    if (start > data.len) {
        _ = c.snprintf(err, errlen, "%s search starts outside file", label);
        return null;
    }
    const first = std.mem.indexOfPos(u8, data, start, needle) orelse {
        _ = c.snprintf(err, errlen, if (after_head) "%s anchor not found after old head" else "%s anchor not found", label);
        return null;
    };
    // Restart one BYTE after the first start, so overlapping duplicates count.
    if (std.mem.indexOfPos(u8, data, first + 1, needle) != null) {
        _ = c.snprintf(err, errlen, if (after_head) "%s anchor is not unique after old head" else "%s anchor is not unique", label);
        return null;
    }
    return first;
}
export fn ds4_agent_file_find_unique(data: [*c]const u8, length: usize, needle: [*c]const u8, needle_length: usize, match: *[*c]const u8, label: [*c]const u8, err: [*c]u8, errlen: usize) bool {
    // Reject an empty anchor before examining the file span, as in C.
    if (needle == null or needle_length == 0) {
        _ = c.snprintf(err, errlen, "%s anchor is empty", label);
        return false;
    }
    const anchor = bytes(needle, needle_length);
    const start = findUnique(bytes(data, length), anchor, 0, false, label, err, errlen) orelse return false;
    match.* = data + start;
    return true;
}

export fn ds4_agent_file_find_old_span(data: [*c]const u8, length: usize, old: [*c]const u8, allow_upto: bool, match: *[*c]const u8, match_length: *usize, anchored: *bool, err: [*c]u8, errlen: usize) bool {
    const contents = bytes(data, length);
    const requested = text(old);
    const marker = "[upto]";
    const split = std.mem.indexOf(u8, requested, marker);
    if (!allow_upto or split == null) {
        anchored.* = false;
        if (!ds4_agent_file_find_unique(data, length, old, requested.len, match, "old text", err, errlen)) return false;
        match_length.* = requested.len;
        return true;
    }
    const head_end = split.?;
    var tail_start = head_end + marker.len;
    if (std.mem.indexOf(u8, requested[tail_start..], marker) != null) {
        _ = c.snprintf(err, errlen, "old text contains more than one [upto] marker");
        return false;
    }
    // Head already contains its newline. Strip only extra leading CR/LF from
    // the tail, whether injected by the forcer or written by the model.
    while (tail_start < requested.len and (requested[tail_start] == '\n' or requested[tail_start] == '\r')) tail_start += 1;
    const tail = requested[tail_start..];
    var nonspace = false;
    for (tail) |byte| {
        if (c.isspace(byte) == 0) {
            nonspace = true;
            break;
        }
    }
    if (!nonspace) {
        _ = c.snprintf(err, errlen, "old text after [upto] must include a unique tail anchor");
        return false;
    }
    const head = findUnique(contents, requested[0..head_end], 0, false, "old head", err, errlen) orelse return false;
    // Tail uniqueness is suffix-only: an earlier occurrence before the head
    // is legitimate and must not reject this anchored edit.
    const end = findUnique(contents, tail, head + head_end, true, "old tail", err, errlen) orelse return false;
    anchored.* = true;
    match.* = data + head;
    match_length.* = end - head + tail.len;
    return true;
}
