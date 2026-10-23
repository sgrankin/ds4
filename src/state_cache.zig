//! Host-only DS4.1 checkpoint storage. One caller serializes operations (the
//! server inference lock is sufficient); a directory has one process owner.
//! Sessions keep their own mutable GPU state. All stored chunks are immutable.
//! Limits count resident chunk bytes and physical disk bytes respectively;
//! metadata and caller-owned materializations are reported/accounted separately.
//!
//! The caller's profile identifies model weights/tokenizer and the fixed forward
//! policy. Rendered text is never an exact-prefix key. DS4SC01 manifests and the
//! DSV4/DSSD/DRP1 checkpoint bytes retain their existing C formats.
//!
//! The exported functions and Blob/Stats records are the C ABI. Cache and Ref
//! are opaque handles to C callers. Allocations use libc so ownership can cross
//! that boundary; returned materializations belong to the caller, not the cache.
const std = @import("std");
const c = std.c;
const api = @import("state_cache_c");
const FILE = opaque {};
extern "c" fn malloc(usize) ?*anyopaque;
extern "c" fn calloc(usize, usize) ?*anyopaque;
extern "c" fn realloc(?*anyopaque, usize) ?*anyopaque;
extern "c" fn free(?*anyopaque) void;
extern "c" fn strdup([*:0]const u8) ?[*:0]u8;
extern "c" fn snprintf([*c]u8, usize, [*:0]const u8, ...) c_int;
extern "c" fn fopen([*:0]const u8, [*:0]const u8) ?*FILE;
extern "c" fn fdopen(c_int, [*:0]const u8) ?*FILE;
extern "c" fn fclose(*FILE) c_int;
extern "c" fn fflush(*FILE) c_int;
extern "c" fn fread([*c]u8, usize, usize, *FILE) usize;
extern "c" fn fwrite([*c]const u8, usize, usize, *FILE) usize;
extern "c" fn fgetc(*FILE) c_int;
extern "c" fn mkstemp([*c]u8) c_int;
extern "c" fn utime([*:0]const u8, ?*const anyopaque) c_int;
extern "c" fn ds4_kvstore_sha1_bytes_hex(?*const anyopaque, usize, [*c]u8) void;
extern "c" fn ds4_kvstore_le_get32([*c]const u8) u32;
extern "c" fn ds4_kvstore_le_put32([*c]u8, u32) void;

const header = 96;
const record = 64;
const page = 2048;
const pre_page = 65536;
const vocab = 129280;
const max_manifest = 8 * 1024 * 1024;
// On-disk DS4SC01 kind at header offset 16. Values 0/1 retain the original
// exact/approximate encoding; older readers reject the new replay kind.
const native: c_int = 0;
const suffix: c_int = 1;
const replay: c_int = 2;
// Internal layouts stay C-compatible because metadata accounting includes
// their sizeof values. No internal pointer is serialized into a manifest.
const Chunk = extern struct {
    key: [41]u8,
    bytes: u64,
    used: u64,
    refs: u32, // Segment records that refer to this immutable byte-hashed chunk.
    pins: u32, // Segment records held by acquired references; blocks eviction.
    present: bool,
    hot: [*c]u8,
    next: ?*Chunk,
};
// blob_index selects the checkpoint (0) or original pre-RoPE blob (1).
const Segment = extern struct { key: [41]u8, blob_index: u32, offset: u64, bytes: u64 };
const Endpoint = extern struct {
    key: [41]u8,
    prefix: [41]u8,
    profile: [*c]u8,
    ctx: u32,
    tokens: u32,
    nseg: u32,
    pins: u32,
    kind: c_int,
    state_bytes: u64,
    pre_bytes: u64,
    manifest_bytes: u64,
    used: u64,
    seg: [*c]Segment,
    next: ?*Endpoint,
};
const Cache = extern struct {
    dir: [*c]u8,
    lock_fd: c_int,
    hot_limit: u64,
    disk_limit: u64,
    hot_bytes: u64,
    disk_bytes: u64,
    clock: u64,
    lease_bytes: u64,
    hot_reads: u64,
    disk_reads: u64,
    evictions: u64,
    chunks: ?*Chunk,
    endpoints: ?*Endpoint,
};
// ids is allocated lazily for candidate enumeration and lives until release.
const Ref = extern struct { cache: *Cache, endpoint: *Endpoint, ids: [*c]c_int };
// Public records are generated from the actual C header, not repeated here.
pub const Blob = api.ds4_state_cache_blob;
pub const Stats = api.ds4_state_cache_stats;

// Internal callers validate nullable C arguments before reaching string helpers.
fn cStringBytes(p: [*c]const u8) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}
fn sameCString(a: [*c]const u8, b: [*c]const u8) bool {
    return std.mem.eql(u8, cStringBytes(a), cStringBytes(b));
}
fn freeOwned(p: anytype) void {
    free(@ptrCast(p));
}
fn create(comptime T: type) ?*T {
    return @ptrCast(@alignCast(calloc(1, @sizeOf(T))));
}
fn allocBytes(n: u64) [*c]u8 {
    return @ptrCast(malloc(@intCast(n)));
}
fn copy(dst: [*c]u8, src: [*c]const u8, n: u64) void {
    @memcpy(dst[0..@intCast(n)], src[0..@intCast(n)]);
}
fn scError(err: [*c]u8, len: usize, message: [*:0]const u8) c_int {
    if (err != null and len != 0) _ = snprintf(err, len, "%s", message);
    return -1;
}
fn u32at(p: [*c]const u8) u32 {
    return ds4_kvstore_le_get32(p);
}
fn u64at(p: [*c]const u8) u64 {
    return u32at(p) | (@as(u64, u32at(p + 4)) << 32);
}
fn put64(p: [*c]u8, n: u64) void {
    ds4_kvstore_le_put32(p, @truncate(n));
    ds4_kvstore_le_put32(p + 4, @truncate(n >> 32));
}
fn keyValid(s: [*c]const u8) bool {
    if (s == null or cStringBytes(s).len != 40) return false;
    for (s[0..40]) |ch| if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return false;
    return true;
}
fn pathFor(cache: *const Cache, key: [*c]const u8, ext: [*:0]const u8) [*c]u8 {
    const n = cStringBytes(cache.dir).len + cStringBytes(key).len + std.mem.len(ext) + 10;
    const p = allocBytes(n);
    if (p != null) _ = snprintf(p, n, "%s/ds4sc-%s.%s", cache.dir, key, ext);
    return p;
}
fn atomicWrite(dir: [*c]const u8, path: [*c]const u8, p: [*c]const u8, n: usize) bool {
    // Publish only after the complete file is flushed and synced. A failed
    // write leaves no visible partial chunk or manifest at its final path.
    const size = cStringBytes(dir).len + 30;
    const temp = allocBytes(size);
    if (temp == null) return false;
    defer freeOwned(temp);
    _ = snprintf(temp, size, "%s/.ds4sc-tmp-XXXXXX", dir);
    const fd = mkstemp(temp);
    const fp = if (fd < 0) null else fdopen(fd, "wb");
    var ok = if (fp) |f| fwrite(p, 1, n, f) == n and fflush(f) == 0 and c.fsync(fd) == 0 else false;
    if (fp) |f| {
        if (fclose(f) != 0) ok = false;
    } else if (fd >= 0) {
        _ = c.close(fd);
    }
    if (ok) ok = c.rename(@ptrCast(temp), @ptrCast(path)) == 0;
    if (!ok) _ = c.unlink(@ptrCast(temp));
    return ok;
}
fn findChunk(cache: *Cache, key: [*c]const u8) ?*Chunk {
    var p = cache.chunks;
    while (p) |q| : (p = q.next) {
        if (sameCString(&q.key, key)) return q;
    }
    return null;
}
fn findEndpoint(cache: *Cache, key: [*c]const u8) ?*Endpoint {
    var p = cache.endpoints;
    while (p) |q| : (p = q.next) {
        if (sameCString(&q.key, key)) return q;
    }
    return null;
}
fn stateBytes(n: u32) u64 {
    const raw: u64 = @min(n, 128);
    var size: u64 = 52 + (@as(u64, n) + vocab) * 4 + 40 * raw * 512 * 4;
    for (0..4) |i| {
        size += @as(u64, n / @as(u32, if (i < 3) 2 else 1)) * (512 + 128) * 4;
        if (i < 3 and n & 1 != 0) size += 2 * 512 * 4;
    }
    return size;
}
fn snapshotHeaderBytes(kind: c_int) u64 {
    // Replay adds a 32-byte DRP1 envelope before the unchanged 52-byte DSV4 header.
    return if (kind == replay) 84 else 52;
}
fn stateValid(initial: [*c]const u8, initial_bytes: u64, kind: c_int, ctx: *u32, tokens: *u32) bool {
    if (initial == null or initial_bytes < snapshotHeaderBytes(kind) or initial_bytes > std.math.maxInt(usize) or kind < 0 or kind > replay) return false;
    var p = initial;
    var bytes = initial_bytes;
    if (kind == replay) {
        // DRP1 v1: kind, K, selected flag, inner bytes, reserved. The inner
        // canonical body is storage only; it must not be loaded as exact.
        if (u32at(p) != 0x31505244 or u32at(p + 4) != 1 or u32at(p + 8) != 1 or
            (u32at(p + 12) != 128 and u32at(p + 12) != 256) or u32at(p + 16) > 1 or
            u64at(p + 20) != bytes - 32 or u32at(p + 28) != 0) return false;
        p += 32;
        bytes -= 32;
    }
    if (u32at(p) != @as(u32, if (kind == suffix) 0x44535344 else 0x34565344) or u32at(p + 4) != 2) return false;
    var h: [13]u32 = undefined;
    for (&h, 0..) |*word, i| word.* = u32at(p + i * 4);
    if (h[2] == 0 or h[2] > 1048576 or h[7] == 0 or h[7] >= h[2] or h[3] != 1 or
        h[4] != 128 or h[5] != 128 or h[6] != h[2] + 1 or h[8] != 40 or h[9] != 512 or
        h[10] != 128 or h[11] != vocab or h[12] != 0x413431 or stateBytes(h[7]) != bytes) return false;
    for (0..h[7]) |i| if (u32at(p + 52 + i * 4) >= vocab) return false;
    ctx.* = h[2];
    tokens.* = h[7];
    return true;
}
// Hash the profile, context size and exact token IDs. Inputs are either host
// integers from a request or already-little-endian IDs from a snapshot.
fn prefixKey(profile: [*c]const u8, ctx: u32, tokens: ?*const anyopaque, n: u32, host_ids: bool, key: *[41]u8) bool {
    const plen = cStringBytes(profile).len;
    const bytes = plen + 9 + @as(u64, n) * 4;
    if (bytes > std.math.maxInt(usize)) return false;
    const p = allocBytes(bytes);
    if (p == null) return false;
    defer freeOwned(p);
    copy(p, profile, plen + 1);
    ds4_kvstore_le_put32(p + plen + 1, ctx);
    ds4_kvstore_le_put32(p + plen + 5, n);
    const dst = p + plen + 9;
    if (host_ids) {
        const ids: [*]const c_int = @ptrCast(@alignCast(tokens.?));
        for (0..n) |i| {
            if (ids[i] < 0 or ids[i] >= vocab) return false;
            ds4_kvstore_le_put32(dst + i * 4, @intCast(ids[i]));
        }
    } else copy(dst, @ptrCast(tokens), @as(u64, n) * 4);
    ds4_kvstore_sha1_bytes_hex(p, @intCast(bytes), key);
    return true;
}
fn addSegment(e: *Endpoint, data: [*c]const u8, blob_index: u32, offset: u64, bytes: u64) bool {
    if (bytes == 0) return true;
    const resized = realloc(@ptrCast(e.seg), @as(usize, e.nseg + 1) * @sizeOf(Segment)) orelse return false;
    e.seg = @ptrCast(@alignCast(resized));
    const s = &e.seg[e.nseg];
    e.nseg += 1;
    s.blob_index = blob_index;
    s.offset = offset;
    s.bytes = bytes;
    ds4_kvstore_sha1_bytes_hex(data + @as(usize, @intCast(offset)), @intCast(bytes), &s.key);
    return true;
}
fn pages(e: *Endpoint, p: [*c]const u8, blob_index: u32, offset: *u64, initial_bytes: u64, page_bytes: u64) bool {
    var bytes = initial_bytes;
    while (bytes != 0) {
        const n: u64 = @min(bytes, page_bytes);
        if (!addSegment(e, p, blob_index, offset.*, n)) return false;
        offset.* += n;
        bytes -= n;
    }
    return true;
}
fn split(e: *Endpoint, p: [*c]const u8, pre: [*c]const u8) bool {
    // Keep the header, token pages, logits, layer rings and compressed/index
    // regions separate so immutable pages can be shared across checkpoints.
    var off: u64 = 0;
    const h = snapshotHeaderBytes(e.kind);
    if (!pages(e, p, 0, &off, h, h) or !pages(e, p, 0, &off, @as(u64, e.tokens) * 4, page * 4) or
        !pages(e, p, 0, &off, vocab * 4, vocab * 4)) return false;
    const raw: u64 = @min(e.tokens, 128);
    if (!pages(e, p, 0, &off, 40 * raw * 512 * 4, raw * 512 * 4)) return false;
    for (0..4) |i| {
        const ratio: u32 = if (i < 3) 2 else 1;
        if (!pages(e, p, 0, &off, @as(u64, e.tokens / ratio) * 512 * 4, (page / ratio) * 512 * 4) or
            !pages(e, p, 0, &off, @as(u64, e.tokens / ratio) * 128 * 4, (page / ratio) * 128 * 4)) return false;
        if (i < 3 and e.tokens & 1 != 0 and !pages(e, p, 0, &off, 2 * 512 * 4, 512 * 4)) return false;
    }
    if (off != e.state_bytes) return false;
    off = 0;
    return e.pre_bytes == 0 or pages(e, pre, 1, &off, e.pre_bytes, pre_page);
}
fn manifest(e: *Endpoint) [*c]u8 {
    const plen = cStringBytes(e.profile).len;
    e.manifest_bytes = header + plen + @as(u64, e.nseg) * record;
    if (e.manifest_bytes > max_manifest) return null;
    const p: [*c]u8 = @ptrCast(calloc(1, @intCast(e.manifest_bytes)));
    if (p == null) return null;
    copy(p, "DS4SC01", 7);
    ds4_kvstore_le_put32(p + 8, e.ctx);
    ds4_kvstore_le_put32(p + 12, e.tokens);
    ds4_kvstore_le_put32(p + 16, @intCast(e.kind));
    ds4_kvstore_le_put32(p + 20, @intCast(plen));
    put64(p + 24, e.state_bytes);
    put64(p + 32, e.pre_bytes);
    ds4_kvstore_le_put32(p + 40, e.nseg);
    copy(p + 48, &e.prefix, 41);
    copy(p + header, e.profile, plen);
    for (e.seg[0..e.nseg], 0..) |s, i| {
        const r = p + header + plen + i * record;
        ds4_kvstore_le_put32(r, s.blob_index);
        ds4_kvstore_le_put32(r + 4, @intCast(s.bytes));
        put64(r + 8, s.offset);
        copy(r + 16, &s.key, 41);
    }
    ds4_kvstore_sha1_bytes_hex(p, @intCast(e.manifest_bytes), &e.key);
    return p;
}
fn freeEndpoint(maybe_e: ?*Endpoint) void {
    const e = maybe_e orelse return;
    freeOwned(e.profile);
    freeOwned(e.seg);
    freeOwned(e);
}
fn parseManifest(p: [*c]const u8, bytes: u64, key: [*c]const u8) ?*Endpoint {
    if (bytes < header or bytes > max_manifest or !std.mem.eql(u8, p[0..8], "DS4SC01\x00")) return null;
    const plen = u32at(p + 20);
    const nseg = u32at(p + 40);
    if (plen == 0 or plen > 255 or nseg == 0 or bytes != header + plen + @as(u64, nseg) * record or
        p[88] != 0 or u32at(p + 16) > replay) return null;
    var digest: [41]u8 = undefined;
    ds4_kvstore_sha1_bytes_hex(p, @intCast(bytes), &digest);
    if (!sameCString(&digest, key)) return null;
    const e = create(Endpoint) orelse return null;
    var ok = false;
    defer if (!ok) freeEndpoint(e);
    copy(&e.key, key, 41);
    copy(&e.prefix, p + 48, 41);
    e.ctx = u32at(p + 8);
    e.tokens = u32at(p + 12);
    e.kind = @intCast(u32at(p + 16));
    e.state_bytes = u64at(p + 24);
    e.pre_bytes = u64at(p + 32);
    e.manifest_bytes = bytes;
    e.nseg = nseg;
    e.profile = @ptrCast(calloc(1, @as(usize, plen) + 1));
    e.seg = @ptrCast(@alignCast(calloc(nseg, @sizeOf(Segment))));
    if (e.profile == null or e.seg == null or !keyValid(&e.prefix) or e.ctx == 0 or e.ctx > 1048576 or
        e.tokens == 0 or e.tokens >= e.ctx or e.state_bytes != stateBytes(e.tokens) + snapshotHeaderBytes(e.kind) - 52 or
        e.pre_bytes > std.math.maxInt(usize) or (e.kind == suffix and e.pre_bytes == 0) or
        (e.kind == replay and e.pre_bytes != 0)) return null;
    copy(e.profile, p + header, plen);
    if (cStringBytes(e.profile).len != plen) return null;
    var cursor = [2]u64{ 0, 0 };
    for (e.seg[0..nseg], 0..) |*s, i| {
        const r = p + header + plen + i * record;
        s.blob_index = u32at(r);
        s.bytes = u32at(r + 4);
        s.offset = u64at(r + 8);
        copy(&s.key, r + 16, 41);
        if (s.blob_index > 1 or s.bytes == 0 or s.bytes > 4 * 1024 * 1024 or !keyValid(&s.key) or s.offset != cursor[s.blob_index]) return null;
        const total = if (s.blob_index != 0) e.pre_bytes else e.state_bytes;
        if (s.bytes > total -% cursor[s.blob_index]) return null;
        cursor[s.blob_index] += s.bytes;
    }
    if (cursor[0] != e.state_bytes or cursor[1] != e.pre_bytes) return null;
    ok = true;
    return e;
}
fn removeChunk(cache: *Cache, chunk: *Chunk, erase: bool) void {
    var link = &cache.chunks;
    while (link.* != null and link.*.? != chunk) link = &link.*.?.next;
    if (link.* != null) link.* = chunk.next;
    if (chunk.hot != null) {
        cache.hot_bytes -= chunk.bytes;
        freeOwned(chunk.hot);
    }
    if (chunk.present) {
        cache.disk_bytes -= chunk.bytes;
        if (erase) {
            const path = pathFor(cache, &chunk.key, "chunk");
            if (path != null) {
                _ = c.unlink(@ptrCast(path));
                freeOwned(path);
            }
        }
    }
    freeOwned(chunk);
}
fn dropSegments(cache: *Cache, e: *Endpoint, count: u32, erase: bool) void {
    for (e.seg[0..count]) |s| if (findChunk(cache, &s.key)) |p| {
        p.refs -= 1;
        if (p.refs == 0 and p.pins == 0) removeChunk(cache, p, erase);
    };
}
fn register(cache: *Cache, e: *Endpoint, from_disk: bool) bool {
    var done: u32 = 0;
    while (done < e.nseg) : (done += 1) {
        const s = &e.seg[done];
        var maybe_p = findChunk(cache, &s.key);
        if (maybe_p != null and maybe_p.?.bytes != s.bytes) break;
        if (maybe_p == null) {
            if (from_disk) {
                const path = pathFor(cache, &s.key, "chunk");
                var st: c.Stat = undefined;
                const valid = path != null and c.stat(@ptrCast(path), &st) == 0 and c.S.ISREG(st.mode) and st.size >= 0 and @as(u64, @intCast(st.size)) == s.bytes;
                freeOwned(path);
                if (!valid) break;
            }
            const p = create(Chunk) orelse break;
            p.key = s.key;
            p.bytes = s.bytes;
            p.present = from_disk;
            p.next = cache.chunks;
            cache.chunks = p;
            if (from_disk) cache.disk_bytes += p.bytes;
            maybe_p = p;
        }
        maybe_p.?.refs += 1;
    }
    if (done == e.nseg) return true;
    dropSegments(cache, e, done, false);
    return false;
}
fn evictEndpoint(cache: *Cache, e: *Endpoint) void {
    var link = &cache.endpoints;
    while (link.* != null and link.*.? != e) link = &link.*.?.next;
    if (link.* != null) link.* = e.next;
    const path = pathFor(cache, &e.key, "manifest");
    if (path != null) {
        _ = c.unlink(@ptrCast(path));
        freeOwned(path);
    }
    cache.disk_bytes -= e.manifest_bytes;
    dropSegments(cache, e, e.nseg, true);
    freeEndpoint(e);
    cache.evictions += 1;
}
fn diskRoom(cache: *Cache, extra: u64) bool {
    if (extra > cache.disk_limit) return false;
    while (cache.disk_bytes > cache.disk_limit - extra) {
        var old: ?*Endpoint = null;
        var p = cache.endpoints;
        while (p) |e| : (p = e.next) {
            if (e.pins == 0 and (old == null or e.used < old.?.used)) old = e;
        }
        if (old == null) return false;
        evictEndpoint(cache, old.?);
    }
    return true;
}
fn tick(cache: *Cache) u64 {
    cache.clock +%= 1;
    return cache.clock;
}
fn makeHot(cache: *Cache, p: *Chunk, data: [*c]const u8) void {
    if (p.hot != null or p.bytes > cache.hot_limit) return;
    while (cache.hot_bytes > cache.hot_limit - p.bytes) {
        var old: ?*Chunk = null;
        var it = cache.chunks;
        while (it) |q| : (it = q.next) {
            if (q.hot != null and q.pins == 0 and (old == null or q.used < old.?.used)) old = q;
        }
        const victim = old orelse return;
        freeOwned(victim.hot);
        victim.hot = null;
        cache.hot_bytes -= victim.bytes;
    }
    p.hot = allocBytes(p.bytes);
    if (p.hot != null) {
        copy(p.hot, data, p.bytes);
        cache.hot_bytes += p.bytes;
        p.used = tick(cache);
    }
}
fn readChunk(cache: *Cache, maybe_p: ?*Chunk, out: [*c]u8) bool {
    const p = maybe_p orelse return false;
    if (!p.present) return false;
    p.used = tick(cache);
    if (p.hot != null) {
        copy(out, p.hot, p.bytes);
        cache.hot_reads += 1;
        return true;
    }
    const path = pathFor(cache, &p.key, "chunk");
    const fp = if (path != null) fopen(@ptrCast(path), "rb") else null;
    const ok = if (fp) |f| fread(out, 1, @intCast(p.bytes), f) == p.bytes and fgetc(f) == -1 else false;
    if (fp) |f| _ = fclose(f);
    freeOwned(path);
    if (!ok) return false;
    var key: [41]u8 = undefined;
    ds4_kvstore_sha1_bytes_hex(out, @intCast(p.bytes), &key);
    if (!sameCString(&key, &p.key)) return false;
    cache.disk_reads += 1;
    makeHot(cache, p, out);
    return true;
}
fn acquire(cache: *Cache, e: *Endpoint) ?*Ref {
    // Pin the endpoint and every referenced segment before exposing a handle.
    const ref: *Ref = @ptrCast(@alignCast(malloc(@sizeOf(Ref)) orelse return null));
    ref.* = .{ .cache = cache, .endpoint = e, .ids = null };
    e.pins += 1;
    e.used = tick(cache);
    cache.lease_bytes += @sizeOf(Ref);
    for (e.seg[0..e.nseg]) |s| findChunk(cache, &s.key).?.pins += 1;
    const path = pathFor(cache, &e.key, "manifest");
    if (path != null) {
        _ = utime(@ptrCast(path), null);
        freeOwned(path);
    }
    return ref;
}
fn nameKey(name: [*c]const u8, ext: []const u8, key: *[41]u8) bool {
    const text = cStringBytes(name);
    if (text.len != 47 + ext.len or !std.mem.startsWith(u8, text, "ds4sc-") or !std.mem.eql(u8, text[47..], ext) or text[46] != '.') return false;
    @memcpy(key[0..40], text[6..46]);
    key[40] = 0;
    return keyValid(key);
}

/// Zero hot bytes means disk-only. The directory must have an existing parent;
/// the process holds its exclusive lock until a successful close.
export fn ds4_state_cache_open(dir: [*c]const u8, hot_bytes: u64, ssd_bytes: u64, err: [*c]u8, errlen: usize) ?*Cache {
    if (err != null and errlen != 0) err[0] = 0;
    if (dir == null or dir[0] == 0) {
        _ = scError(err, errlen, "missing state-cache directory");
        return null;
    }
    if (c.mkdir(@ptrCast(dir), 0o700) != 0 and c._errno().* != @backingInt(c.E.EXIST)) {
        _ = scError(err, errlen, "cannot create state-cache directory");
        return null;
    }
    const cache = create(Cache) orelse return null;
    cache.lock_fd = -1;
    cache.dir = @ptrCast(strdup(@ptrCast(dir)));
    cache.hot_limit = hot_bytes;
    cache.disk_limit = ssd_bytes;
    var ok = false;
    defer if (!ok) {
        if (err != null and errlen != 0 and err[0] == 0) _ = scError(err, errlen, "cannot open state cache");
        _ = ds4_state_cache_close(cache, null, 0);
    };
    if (cache.dir == null) return null;
    const size = cStringBytes(dir).len + 20;
    const lock = allocBytes(size);
    if (lock == null) return null;
    _ = snprintf(lock, size, "%s/.ds4sc.lock", dir);
    cache.lock_fd = c.open(@ptrCast(lock), .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true }, @as(c_uint, 0o600));
    freeOwned(lock);
    if (cache.lock_fd < 0 or c.flock(cache.lock_fd, c.LOCK.EX | c.LOCK.NB) != 0) {
        _ = scError(err, errlen, "state-cache directory already owned or unavailable");
        return null;
    }
    const dp = c.opendir(@ptrCast(dir)) orelse return null;
    defer _ = c.closedir(dp);
    while (c.readdir(dp)) |entry| {
        var key: [41]u8 = undefined;
        if (!nameKey(&entry.name, "manifest", &key)) continue;
        const path = pathFor(cache, &key, "manifest");
        var st: c.Stat = undefined;
        var e: ?*Endpoint = null;
        if (path != null and c.stat(@ptrCast(path), &st) == 0 and c.S.ISREG(st.mode) and st.size >= header and st.size <= max_manifest) {
            const fp = fopen(@ptrCast(path), "rb");
            const bytes = allocBytes(@intCast(st.size));
            if (fp != null and bytes != null and fread(bytes, 1, @intCast(st.size), fp.?) == @as(usize, @intCast(st.size))) e = parseManifest(bytes, @intCast(st.size), &key);
            if (fp) |f| _ = fclose(f);
            freeOwned(bytes);
        }
        if (e != null and register(cache, e.?, true)) {
            e.?.used = if (st.mtime().sec > 0) @intCast(st.mtime().sec) else 0;
            cache.clock = @max(cache.clock, e.?.used);
            e.?.next = cache.endpoints;
            cache.endpoints = e;
            cache.disk_bytes += e.?.manifest_bytes;
        } else {
            freeEndpoint(e);
            // Incomplete cache publication is a miss.
            if (path != null) _ = c.unlink(@ptrCast(path));
        }
        freeOwned(path);
    }
    c.rewinddir(dp);
    while (c.readdir(dp)) |entry| {
        var key: [41]u8 = undefined;
        if (nameKey(&entry.name, "chunk", &key) and findChunk(cache, &key) == null) {
            const path = pathFor(cache, &key, "chunk");
            if (path != null) {
                _ = c.unlink(@ptrCast(path));
                freeOwned(path);
            }
        } else if (std.mem.startsWith(u8, cStringBytes(&entry.name), ".ds4sc-tmp-")) {
            const n = cStringBytes(dir).len + cStringBytes(&entry.name).len + 2;
            const path = allocBytes(n);
            if (path != null) {
                _ = snprintf(path, n, "%s/%s", dir, &entry.name);
                _ = c.unlink(@ptrCast(path));
                freeOwned(path);
            }
        }
    }
    if (!diskRoom(cache, 0)) return null;
    ok = true;
    return cache;
}

/// Borrow inputs only until return. Store DSV4 as native, DSSD as suffix state,
/// or DRP1 as resumable decoder replay. Suffix states require their original
/// pre-RoPE blob; replay states cannot carry one or serve as suffix sources.
/// A successful put is committed to SSD. Pages are shared by byte hash, not by
/// mutable pointers. key is a stable 40-character identifier for this source.
export fn ds4_state_cache_put_kind(maybe_cache: ?*Cache, profile: [*c]const u8, state: ?*const anyopaque, state_bytes: u64, pre: ?*const anyopaque, pre_bytes: u64, kind: c_int, key: [*c]u8, err: [*c]u8, errlen: usize) c_int {
    if (maybe_cache == null or profile == null or profile[0] == 0 or cStringBytes(profile).len > 255 or key == null or kind < 0 or kind > replay or
        pre_bytes > std.math.maxInt(usize) or (pre_bytes != 0 and pre == null) or (kind == suffix and pre_bytes == 0) or
        (kind == replay and pre_bytes != 0)) return scError(err, errlen, "invalid state-cache put");
    const cache = maybe_cache.?;
    const endpoint = create(Endpoint) orelse return scError(err, errlen, "state-cache allocation failed");
    var manifest_data: [*c]u8 = null;
    var registered = false;
    var committed = false;
    defer if (!committed) {
        if (registered) dropSegments(cache, endpoint, endpoint.nseg, true);
        freeOwned(manifest_data);
        freeEndpoint(endpoint);
    };
    if (!stateValid(@ptrCast(state), state_bytes, kind, &endpoint.ctx, &endpoint.tokens)) return scError(err, errlen, "not a complete DS4.1 payload of the requested kind");
    endpoint.profile = @ptrCast(strdup(@ptrCast(profile)));
    endpoint.state_bytes = state_bytes;
    endpoint.pre_bytes = pre_bytes;
    endpoint.kind = kind;
    const state_data: [*c]const u8 = @ptrCast(state);
    const prerope_data: [*c]const u8 = @ptrCast(pre);
    if (endpoint.profile == null or !prefixKey(profile, endpoint.ctx, state_data + @as(usize, @intCast(snapshotHeaderBytes(kind))), endpoint.tokens, false, &endpoint.prefix) or !split(endpoint, state_data, prerope_data))
        return scError(err, errlen, "state-cache allocation or atomic publication failed");
    manifest_data = manifest(endpoint);
    if (manifest_data == null) return scError(err, errlen, "state-cache allocation or atomic publication failed");
    if (findEndpoint(cache, &endpoint.key)) |existing| {
        existing.used = tick(cache);
        copy(key, &endpoint.key, 41);
        return 0;
    }
    // An endpoint must itself fit, even if all other cache entries disappear.
    var own = endpoint.manifest_bytes;
    for (endpoint.seg[0..endpoint.nseg], 0..) |s, i| {
        var first = true;
        for (endpoint.seg[0..i]) |earlier| {
            if (sameCString(&earlier.key, &s.key)) {
                first = false;
                break;
            }
        }
        if (first) own += s.bytes;
    }
    if (own > cache.disk_limit) return scError(err, errlen, "state-cache SSD budget is occupied or endpoint exceeds it");
    if (!register(cache, endpoint, false)) return scError(err, errlen, "state-cache allocation or atomic publication failed");
    registered = true;
    var missing = endpoint.manifest_bytes;
    var it = cache.chunks;
    while (it) |chunk| : (it = chunk.next) {
        if (!chunk.present) missing += chunk.bytes;
    }
    // Registered references protect shared pages while old endpoints evict.
    if (!diskRoom(cache, missing)) return scError(err, errlen, "state-cache SSD budget is occupied or endpoint exceeds it");
    for (endpoint.seg[0..endpoint.nseg]) |s| {
        const chunk = findChunk(cache, &s.key).?;
        const data = if (s.blob_index != 0) prerope_data else state_data;
        if (!chunk.present) {
            const path = pathFor(cache, &chunk.key, "chunk");
            const ok = path != null and atomicWrite(cache.dir, path, data + @as(usize, @intCast(s.offset)), @intCast(s.bytes));
            freeOwned(path);
            if (!ok) return scError(err, errlen, "state-cache allocation or atomic publication failed");
            chunk.present = true;
            cache.disk_bytes += chunk.bytes;
        }
    }
    const path = pathFor(cache, &endpoint.key, "manifest");
    const ok = path != null and atomicWrite(cache.dir, path, manifest_data, @intCast(endpoint.manifest_bytes));
    freeOwned(path);
    if (!ok) return scError(err, errlen, "state-cache allocation or atomic publication failed");
    endpoint.used = tick(cache);
    endpoint.next = cache.endpoints;
    cache.endpoints = endpoint;
    cache.disk_bytes += endpoint.manifest_bytes;
    for (endpoint.seg[0..endpoint.nseg]) |s| makeHot(cache, findChunk(cache, &s.key).?, (if (s.blob_index != 0) prerope_data else state_data) + @as(usize, @intCast(s.offset)));
    copy(key, &endpoint.key, 41);
    freeOwned(manifest_data);
    committed = true;
    return 0;
}
/// Compatibility entry point: false selects native, true selects suffix state.
export fn ds4_state_cache_put(cache: ?*Cache, profile: [*c]const u8, state: ?*const anyopaque, state_bytes: u64, pre: ?*const anyopaque, pre_bytes: u64, approximate: bool, key: [*c]u8, err: [*c]u8, errlen: usize) c_int {
    return ds4_state_cache_put_kind(cache, profile, state, state_bytes, pre, pre_bytes, if (approximate) suffix else native, key, err, errlen);
}
/// Returned references pin the endpoint and its chunks against eviction.
export fn ds4_state_cache_get(maybe_cache: ?*Cache, key: [*c]const u8) ?*Ref {
    const cache = maybe_cache orelse return null;
    if (!keyValid(key)) return null;
    return acquire(cache, findEndpoint(cache, key) orelse return null);
}
/// Return the longest captured native or replay prefix of ids[0..n]. The
/// profile must include replay K/selected policy; the core loader also checks
/// that policy. Restoring replay state does not make it exact.
export fn ds4_state_cache_find_prefix(maybe_cache: ?*Cache, profile: [*c]const u8, ctx: u32, ids: [*c]const c_int, n: u32, matched: [*c]u32) ?*Ref {
    if (matched != null) matched[0] = 0;
    const cache = maybe_cache orelse return null;
    if (profile == null or ids == null) return null;
    var best: ?*Endpoint = null;
    var it = cache.endpoints;
    while (it) |e| : (it = e.next) {
        if (e.kind == suffix or e.ctx != ctx or e.tokens > n or !sameCString(e.profile, profile) or (best != null and best.?.tokens >= e.tokens)) continue;
        var key: [41]u8 = undefined;
        if (prefixKey(profile, ctx, @ptrCast(ids), e.tokens, true, &key) and sameCString(&key, &e.prefix)) best = e;
    }
    const e = best orelse return null;
    const ref = acquire(cache, e);
    if (ref != null and matched != null) matched[0] = e.tokens;
    return ref;
}
fn tokenTail(cache: *Cache, e: *Endpoint, ids: [*c]const c_int, n: u32) bool {
    const start = snapshotHeaderBytes(e.kind) + @as(u64, e.tokens - n) * 4;
    const stop = snapshotHeaderBytes(e.kind) + @as(u64, e.tokens) * 4;
    for (e.seg[0..e.nseg]) |s| {
        if (s.blob_index != 0 or s.offset >= stop or s.offset + s.bytes <= start) continue;
        const buf = allocBytes(s.bytes);
        if (buf == null) return false;
        var ok = readChunk(cache, findChunk(cache, &s.key), buf);
        var off: u64 = @max(s.offset, start);
        const end: u64 = @min(s.offset + s.bytes, stop);
        while (ok and off < end) : (off += 4) {
            const id = ids[@intCast((off - start) / 4)];
            ok = id >= 0 and id < vocab and u32at(buf + @as(usize, @intCast(off - s.offset))) == @as(u32, @intCast(id));
        }
        freeOwned(buf);
        if (!ok) return false;
    }
    return true;
}
/// Find a non-replay source whose endpoint ends in all supplied ids;
/// source_start locates them in the stored original token history. The caller
/// still checks relocation, fresh-tail and latest-user eligibility.
export fn ds4_state_cache_find_suffix(maybe_cache: ?*Cache, profile: [*c]const u8, ctx: u32, ids: [*c]const c_int, n: u32, source_start: [*c]u32) ?*Ref {
    if (source_start != null) source_start[0] = 0;
    const cache = maybe_cache orelse return null;
    if (profile == null or ids == null or n == 0) return null;
    var best: ?*Endpoint = null;
    var it = cache.endpoints;
    while (it) |e| : (it = e.next) {
        if (e.kind != replay and e.ctx == ctx and e.tokens >= n and sameCString(e.profile, profile) and
            (best == null or e.used > best.?.used) and tokenTail(cache, e, ids, n)) best = e;
    }
    const e = best orelse return null;
    const ref = acquire(cache, e);
    if (ref != null and source_start != null) source_start[0] = e.tokens - n;
    return ref;
}
fn refTokens(ref: *Ref) bool {
    const e = ref.endpoint;
    ref.ids = @ptrCast(@alignCast(malloc(@as(usize, e.tokens) * @sizeOf(c_int))));
    if (ref.ids == null) return false;
    ref.cache.lease_bytes += @as(u64, e.tokens) * @sizeOf(c_int);
    const start = snapshotHeaderBytes(e.kind);
    const stop = start + @as(u64, e.tokens) * 4;
    for (e.seg[0..e.nseg]) |s| {
        if (s.blob_index != 0 or s.offset >= stop or s.offset + s.bytes <= start) continue;
        const buf = allocBytes(s.bytes);
        if (buf == null) return false;
        var ok = readChunk(ref.cache, findChunk(ref.cache, &s.key), buf);
        var off: u64 = @max(s.offset, start);
        const end: u64 = @min(s.offset + s.bytes, stop);
        while (ok and off < end) : (off += 4) {
            const id = u32at(buf + @as(usize, @intCast(off - s.offset)));
            if (id >= vocab) ok = false else ref.ids[@intCast((off - start) / 4)] = @intCast(id);
        }
        freeOwned(buf);
        if (!ok) return false;
    }
    return true;
}
/// Restart discovery: initialize cursor[0]=0, iterate key order, and release
/// every returned reference. ids are borrowed immutable bytes owned by that
/// reference, valid until release. Native and suffix sources appear; resumable
/// replay states are excluded because they cannot be relocated. The caller
/// matches its anchor and checks full suffix geometry. No graph or large state
/// blob is materialized merely to enumerate token histories.
export fn ds4_state_cache_next_candidate(maybe_cache: ?*Cache, profile: [*c]const u8, ctx: u32, cursor: [*c]u8, ids: [*c][*c]const c_int, n: [*c]u32, approximate: [*c]bool) ?*Ref {
    if (ids != null) ids[0] = null;
    if (n != null) n[0] = 0;
    const cache = maybe_cache orelse return null;
    if (profile == null or cursor == null or (cursor[0] != 0 and !keyValid(cursor))) return null;
    var next: ?*Endpoint = null;
    var it = cache.endpoints;
    while (it) |e| : (it = e.next) {
        if (e.kind != replay and e.ctx == ctx and sameCString(e.profile, profile) and std.mem.order(u8, cStringBytes(&e.key), cStringBytes(cursor)) == .gt and
            (next == null or std.mem.order(u8, cStringBytes(&e.key), cStringBytes(&next.?.key)) == .lt)) next = e;
    }
    const e = next orelse return null;
    const ref = acquire(cache, e) orelse return null;
    copy(cursor, &e.key, 41);
    if (!refTokens(ref)) {
        ds4_state_cache_release(ref);
        return null;
    }
    if (ids != null) ids[0] = ref.ids;
    if (n != null) n[0] = e.tokens;
    if (approximate != null) approximate[0] = e.kind != native;
    return ref;
}
/// Admit caller-owned allocation before materializing a source.
export fn ds4_state_cache_ref_sizes(ref: ?*const Ref, state_bytes: [*c]u64, pre_bytes: [*c]u64) void {
    if (state_bytes != null) state_bytes[0] = if (ref) |r| r.endpoint.state_bytes else 0;
    if (pre_bytes != null) pre_bytes[0] = if (ref) |r| r.endpoint.pre_bytes else 0;
}
/// Return fresh owned bytes that remain valid after release/reset/overwrite of
/// any session. Native state uses the existing payload loader; replay needs its
/// dedicated loader. A suffix caller rebinds its engine after checking profile
/// and context: no runtime pointer is persisted.
export fn ds4_state_cache_materialize(maybe_ref: ?*Ref, maybe_out: ?*Blob, err: [*c]u8, errlen: usize) c_int {
    if (maybe_ref == null or maybe_out == null) return scError(err, errlen, "missing state-cache reference/output");
    const ref = maybe_ref.?;
    const e = ref.endpoint;
    var blob = std.mem.zeroes(Blob);
    var ok = false;
    defer if (!ok) ds4_state_cache_blob_free(&blob);
    blob.state = allocBytes(e.state_bytes);
    if (e.pre_bytes != 0) blob.prerope = allocBytes(e.pre_bytes);
    if (blob.state == null or (e.pre_bytes != 0 and blob.prerope == null)) return scError(err, errlen, "state-cache materialization failed or chunk is corrupt");
    blob.state_bytes = e.state_bytes;
    blob.prerope_bytes = e.pre_bytes;
    blob.ctx_size = e.ctx;
    blob.tokens = e.tokens;
    blob.kind = @intCast(e.kind);
    blob.approximate = e.kind != native;
    for (e.seg[0..e.nseg]) |s| {
        const p = if (s.blob_index != 0) blob.prerope else blob.state;
        if (!readChunk(ref.cache, findChunk(ref.cache, &s.key), p + @as(usize, @intCast(s.offset)))) return scError(err, errlen, "state-cache materialization failed or chunk is corrupt");
    }
    var ctx: u32 = undefined;
    var n: u32 = undefined;
    if (!stateValid(blob.state, blob.state_bytes, @intCast(blob.kind), &ctx, &n) or ctx != e.ctx or n != e.tokens) return scError(err, errlen, "state-cache materialization failed or chunk is corrupt");
    maybe_out.?.* = blob;
    ok = true;
    return 0;
}
/// Free the libc-owned materialization and clear its public C record.
export fn ds4_state_cache_blob_free(maybe_blob: ?*Blob) void {
    const blob = maybe_blob orelse return;
    freeOwned(blob.state);
    freeOwned(blob.prerope);
    @memset(std.mem.asBytes(blob), 0);
}
/// Release the handle, its token history, and its endpoint/chunk pins.
export fn ds4_state_cache_release(maybe_ref: ?*Ref) void {
    const ref = maybe_ref orelse return;
    const e = ref.endpoint;
    for (e.seg[0..e.nseg]) |s| findChunk(ref.cache, &s.key).?.pins -= 1;
    e.pins -= 1;
    ref.cache.lease_bytes -= @sizeOf(Ref) + (if (ref.ids != null) @as(u64, e.tokens) * @sizeOf(c_int) else @as(u64, 0));
    freeOwned(ref.ids);
    freeOwned(ref);
}
export fn ds4_state_cache_get_stats(maybe_cache: ?*const Cache, maybe_out: ?*Stats) void {
    const out = maybe_out orelse return;
    @memset(std.mem.asBytes(out), 0);
    const cache = maybe_cache orelse return;
    out.hot_bytes = cache.hot_bytes;
    out.ssd_bytes = cache.disk_bytes;
    out.metadata_bytes = @sizeOf(Cache) + cStringBytes(cache.dir).len + 1 + cache.lease_bytes;
    out.hot_reads = cache.hot_reads;
    out.disk_reads = cache.disk_reads;
    out.endpoint_evictions = cache.evictions;
    var chunks = cache.chunks;
    while (chunks) |p| : (chunks = p.next) {
        out.chunks += 1;
        out.metadata_bytes += @sizeOf(Chunk);
    }
    var endpoints = cache.endpoints;
    while (endpoints) |e| : (endpoints = e.next) {
        out.endpoints += 1;
        out.pinned_endpoints += @intFromBool(e.pins != 0);
        out.metadata_bytes += @sizeOf(Endpoint) + cStringBytes(e.profile).len + 1 + @as(u64, e.nseg) * @sizeOf(Segment);
    }
}
/// Refuse close with outstanding references; release them and retry.
export fn ds4_state_cache_close(maybe_cache: ?*Cache, err: [*c]u8, errlen: usize) c_int {
    const cache = maybe_cache orelse return 0;
    var it = cache.endpoints;
    while (it) |e| : (it = e.next) {
        if (e.pins != 0) return scError(err, errlen, "state-cache references remain pinned");
    }
    while (cache.endpoints) |e| {
        cache.endpoints = e.next;
        freeEndpoint(e);
    }
    while (cache.chunks) |chunk| removeChunk(cache, chunk, false);
    if (cache.lock_fd >= 0) {
        _ = c.flock(cache.lock_fd, c.LOCK.UN);
        _ = c.close(cache.lock_fd);
    }
    freeOwned(cache.dir);
    freeOwned(cache);
    return 0;
}
