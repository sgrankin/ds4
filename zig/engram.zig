//! Engram hashing and bounded sparse disk reads for the Metal/macOS host.
//! Public records and POSIX/dispatch declarations come from the real headers.
//! There is no model-table cache or mapped Metal view here: only fixed format
//! lookup tables, bounded read requests, and a reusable four-worker reader.
const std = @import("std");
const c = @import("c_api");
const Layout = c.ds4_engram_layout;
const History = c.ds4_engram_history;
const Table = c.ds4_engram_table;
const cols: usize = c.DS4_ENGRAM_COLS;
const dim: usize = c.DS4_ENGRAM_DIM;
const row_bytes: usize = c.DS4_ENGRAM_ROW_BYTES;
const layers: usize = c.DS4_ENGRAM_LAYERS;
const ngram: usize = c.DS4_ENGRAM_NGRAM;
const heads: usize = c.DS4_ENGRAM_HEADS;
const token_readers = 4;
const batch_readers = 16;
const max_batch_tokens: usize = 2048;

fn fail(error_code: c_int) bool {
    c.__error().* = error_code;
    return false;
}

export fn ds4_engram_layout_valid(maybe_layout: ?*const Layout) bool {
    const layout = maybe_layout orelse return false;
    if (layout.token_map == null or layout.vocab_size == 0 or layout.compressed_vocab_size == 0 or
        layout.compressed_vocab_size > std.math.maxInt(i32) or layout.pad_id >= layout.compressed_vocab_size) return false;
    for (layout.token_map[0..layout.vocab_size]) |id| if (id >= layout.compressed_vocab_size) return false;
    for (0..layers) |layer| {
        for (layout.multipliers[layer]) |m| if (m & 1 == 0 or m > @as(u64, std.math.maxInt(i64)) / layout.compressed_vocab_size) return false;
        var total: u64 = 0;
        for (layout.primes[layer]) |prime| {
            if (prime < 2) return false;
            total += prime;
        }
        if (total != layout.rows[layer]) return false;
    }
    return true;
}
// Newest first: copying History snapshots the entire hash state, with no hidden
// process-global tail. DEAD breaks every n-gram crossing a masked position.
export fn ds4_engram_history_reset(history: *History) void {
    history.tail = @splat(c.DS4_ENGRAM_DEAD);
}
// Validate the layout once at model load. This checks per-call IDs/history
// before any mutation. Output includes masked rows; the graph, not this reader,
// must suppress Engram at masked positions as required by the reference.
export fn ds4_engram_hash(maybe_layout: ?*const Layout, maybe_history: ?*History, tokens: [*c]const c_int, mask: [*c]const u8, count: usize, rows: [*c]u32) bool {
    const layout = maybe_layout orelse return false;
    const history = maybe_history orelse return false;
    if (layout.token_map == null or (count != 0 and (tokens == null or rows == null)) or count > std.math.maxInt(usize) / (layers * cols * @sizeOf(u32))) return false;
    for (history.tail) |id| if (id < c.DS4_ENGRAM_DEAD or (id >= 0 and @as(u32, @intCast(id)) >= layout.compressed_vocab_size)) return false;
    // C permits null inputs for an empty batch; a Zig slice still requires
    // a nonnull pointer, even at length zero.
    if (count != 0) for (tokens[0..count]) |id| {
        if (id < 0 or @as(u32, @intCast(id)) >= layout.vocab_size) return false;
    };
    var out = rows;
    for (0..count) |i| {
        const current: i32 = if (mask != null and mask[i] == 0) c.DS4_ENGRAM_DEAD else @intCast(layout.token_map[@intCast(tokens[i])]);
        var ids: [ngram]u32 = undefined;
        var blocked = false;
        for (&ids, 0..) |*id, j| {
            const value = if (j != 0) history.tail[j - 1] else current;
            blocked = blocked or value == c.DS4_ENGRAM_DEAD;
            id.* = if (blocked) layout.pad_id else @intCast(value);
        }
        for (0..layers) |layer| {
            var hash = @as(u64, ids[0]) *% layout.multipliers[layer][0];
            var offset: u32 = 0;
            for (1..ngram) |j| {
                hash ^= @as(u64, ids[j]) *% layout.multipliers[layer][j];
                for (0..heads) |head| {
                    const col = (j - 1) * heads + head;
                    const prime = layout.primes[layer][col];
                    out[0] = @as(u32, @intCast(hash % prime)) +% offset;
                    out += 1;
                    offset +%= prime;
                }
            }
        }
        history.tail[2] = history.tail[1];
        history.tail[1] = history.tail[0];
        history.tail[0] = current;
    }
    return true;
}
// A separate readonly FD owns no mapping. Each GGUF I8 row is 256 E4M3 codes
// followed by eight original E8M0 scales; validate the full table extent first.
export fn ds4_engram_table_open(maybe_table: ?*Table, path: [*c]const u8, offset: u64, rows: u32) bool {
    const table = maybe_table orelse return false;
    table.* = .{ .fd = -1, .offset = 0, .rows = 0 };
    const bytes = @as(u64, rows) * row_bytes;
    if (path == null or rows == 0 or offset > std.math.maxInt(i64) or bytes > @as(u64, std.math.maxInt(i64)) - offset) return fail(c.EINVAL);
    const fd = c.open(path, c.O_RDONLY | c.O_CLOEXEC);
    if (fd < 0) return false;
    var ok = false;
    defer if (!ok) {
        const saved = c.__error().*;
        _ = c.close(fd);
        c.__error().* = saved;
    };
    var st: c.struct_stat = undefined;
    if (c.fstat(fd, &st) != 0) return false;
    if ((st.st_mode & c.S_IFMT) != c.S_IFREG or st.st_size < 0 or offset + bytes > @as(u64, @intCast(st.st_size))) return fail(c.EINVAL);
    // Hashed rows are scattered: keep read-ahead off so unused pages do not
    // compete with model memory headroom. The diagnostic file-cache override
    // permits reclaimable OS pages, never a whole-table allocation here.
    if (c.fcntl(fd, c.F_NOCACHE, @as(c_int, if (c.getenv("DS4_ENGRAM_FILE_CACHE") != null) 0 else 1)) != 0 or
        c.fcntl(fd, c.F_RDAHEAD, @as(c_int, 0)) != 0) return false;
    table.* = .{ .fd = fd, .offset = offset, .rows = rows };
    ok = true;
    return true;
}
export fn ds4_engram_table_close(maybe_table: ?*Table) void {
    const table = maybe_table orelse return;
    if (table.fd >= 0) _ = c.close(table.fd);
    table.* = .{ .fd = -1, .offset = 0, .rows = 0 };
}
// EINTR retries and short pread progress retain the requested absolute offset;
// EOF before the whole 264-byte row is an I/O failure, not an all-zero row.
fn readRow(fd: c_int, offset: u64, row: *[row_bytes]u8) bool {
    var done: usize = 0;
    while (done < row_bytes) {
        const n = c.pread(fd, row[done..].ptr, row_bytes - done, @intCast(offset + done));
        if (n < 0 and c.__error().* == c.EINTR) continue;
        if (n <= 0) {
            if (n == 0) c.__error().* = c.EIO;
            return false;
        }
        done += @intCast(n);
    }
    return true;
}
fn e4m3(byte: u8) f32 {
    @setFloatMode(.strict);
    const exponent: c_int = (byte >> 3) & 15;
    const mantissa: c_int = byte & 7;
    const value = if (exponent != 0) c.ldexpf(@floatFromInt(8 + mantissa), exponent - 10) else c.ldexpf(@floatFromInt(mantissa), -9);
    return if (byte & 128 != 0) -value else value;
}
// All code/scale pairs fit in 128 KiB. This is a format lookup, not a cache of
// model rows. Strict F32 scaling plus BF16 round-to-nearest-even preserves the
// reference's signed zero, overflow and invalid-code/scale behavior.
var bf16: [256][256]u16 = undefined;
var bf16_once: c.pthread_once_t = c.ds4_zig_pthread_once_init;
fn initBf16() callconv(.c) void {
    @setFloatMode(.strict);
    for (0..256) |scale| for (0..256) |code| {
        var result: u16 = std.math.maxInt(u16);
        if (code & 127 != 127 and scale != 255) {
            const value = c.ldexpf(e4m3(@intCast(code)), @as(c_int, @intCast(scale)) - 127);
            var bits: u32 = @bitCast(value);
            bits = (bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000;
            if (bits & 0x7f800000 != 0x7f800000) result = @truncate(bits >> 16);
        }
        bf16[scale][code] = result;
    };
}
// Optional F32 format lookup for the same sixteen input bits (code + scale),
// including invalid values. It must keep the original scaling/rounding sequence.
var decode_lut: [256][256]f32 = undefined;
var decode_once: c.pthread_once_t = c.ds4_zig_pthread_once_init;
fn initDecode() callconv(.c) void {
    @setFloatMode(.strict);
    for (0..256) |scale| for (0..256) |code| {
        var value = std.math.nan(f32);
        if (code & 127 != 127 and scale != 255) {
            value = c.ldexpf(e4m3(@intCast(code)), @as(c_int, @intCast(scale)) - 127);
            var bits: u32 = @bitCast(value);
            bits = (bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000;
            value = @bitCast(bits);
        }
        decode_lut[scale][code] = value;
    };
}
// Caller owns count * DIM F32 outputs containing BF16-rounded values. A read
// or decode failure invalidates that output; no partial-success promise is made.
export fn ds4_engram_read(maybe_table: ?*const Table, rows: [*c]const u32, count: usize, out: [*c]f32) bool {
    @setFloatMode(.strict);
    const table = maybe_table orelse return fail(c.EINVAL);
    if (table.fd < 0 or (count != 0 and (rows == null or out == null)) or count > std.math.maxInt(usize) / (dim * 4)) return fail(c.EINVAL);
    if (count != 0) for (rows[0..count]) |row| {
        if (row >= table.rows) return fail(c.EINVAL);
    };
    _ = c.pthread_once(&bf16_once, initBf16);
    const lookup = c.getenv("DS4_ENGRAM_DECODE_LUT") != null;
    if (lookup) _ = c.pthread_once(&decode_once, initDecode);
    var raw: [row_bytes]u8 = undefined;
    for (0..count) |i| {
        if (!readRow(table.fd, table.offset + @as(u64, rows[i]) * row_bytes, &raw)) return false;
        for (0..dim) |j| {
            if (lookup) {
                const value = decode_lut[raw[dim + j / 32]][raw[j]];
                if (@as(u32, @bitCast(value)) & 0x7f800000 == 0x7f800000) return fail(c.EDOM);
                out[i * dim + j] = value;
            } else {
                const value = bf16[raw[dim + j / 32]][raw[j]];
                if (value == std.math.maxInt(u16)) return fail(c.EDOM);
                out[i * dim + j] = @bitCast(@as(u32, value) << 16);
            }
        }
    }
    return true;
}
const Request = struct { row: u32, output: u32 };
fn requestOrder(a: ?*const anyopaque, b: ?*const anyopaque) callconv(.c) c_int {
    const x: *const Request = @ptrCast(@alignCast(a.?));
    const y: *const Request = @ptrCast(@alignCast(b.?));
    return @as(c_int, @intFromBool(x.row > y.row)) - @as(c_int, @intFromBool(x.row < y.row));
}
const Batch = struct { table: *const Table, request: [*]const Request, out: [*]f32, count: usize, readers: usize, errors: [batch_readers]c_int };
fn readBatchPart(context: ?*anyopaque, part: usize) callconv(.c) void {
    const batch: *Batch = @ptrCast(@alignCast(context.?));
    const request = batch.request;
    const begin = batch.count * part / batch.readers;
    const end = batch.count * (part + 1) / batch.readers;
    var previous: [*c]const f32 = null;
    for (begin..end) |i| {
        const dst = batch.out + @as(usize, request[i].output) * dim;
        if (i > begin and request[i].row == request[i - 1].row) {
            @memcpy(dst[0..dim], previous[0..dim]);
        } else {
            if (!ds4_engram_read(batch.table, &request[i].row, 1, dst)) {
                batch.errors[part] = if (c.__error().* != 0) c.__error().* else c.EIO;
                return;
            }
            previous = dst;
        }
    }
}
export fn ds4_engram_read_batch(maybe_table: ?*const Table, rows: [*c]const u32, tokens: usize, stride: usize, out: [*c]f32) bool {
    const table = maybe_table orelse return fail(c.EINVAL);
    if (table.fd < 0 or (tokens != 0 and (rows == null or out == null or stride < cols)) or
        tokens > std.math.maxInt(usize) / (cols * dim * 4) or
        (tokens != 0 and tokens - 1 > (std.math.maxInt(usize) / 4 - cols) / stride)) return fail(c.EINVAL);
    for (0..tokens) |i| for (0..cols) |j| {
        if (rows[i * stride + j] >= table.rows) return fail(c.EINVAL);
    };
    if (tokens == 0) return true;
    // 2048 tokens * 24 requests * 8 bytes = 384 KiB, independent of table or
    // prefix size. Explicit usize counts avoid narrowing inferred size math.
    const cap: usize = @min(tokens, max_batch_tokens);
    const request: [*]Request = @ptrCast(@alignCast(c.malloc(cap * cols * @sizeOf(Request)) orelse return false));
    var start: usize = 0;
    var ok = true;
    while (ok and start < tokens) : (start += cap) {
        const n: usize = @min(tokens - start, cap);
        const count: usize = n * cols;
        for (0..count) |i| request[i] = .{ .row = rows[(start + i / cols) * stride + i % cols], .output = @intCast(i) };
        c.qsort(request, count, @sizeOf(Request), requestOrder);
        var batch = Batch{ .table = table, .request = request, .count = count, .out = out + start * cols * dim, .readers = 1, .errors = @splat(0) };
        // Fixed shared-pool concurrency hides random-read latency. Workers own
        // disjoint output rows, and synchronous dispatch_apply_f joins them all
        // before any GPU use. Duplicates at a part boundary may be read twice.
        const parallel_min: usize = if (c.getenv("DS4_ENGRAM_SMALL_BATCH_PARALLEL") != null) cols * 2 else 8;
        if (count >= parallel_min) {
            batch.readers = batch_readers;
            c.dispatch_apply_f(batch_readers, c.dispatch_get_global_queue(c.QOS_CLASS_USER_INITIATED, 0), &batch, readBatchPart);
        } else readBatchPart(&batch, 0);
        for (batch.errors[0..batch.readers]) |err| {
            if (err != 0) {
                c.__error().* = err;
                ok = false;
                break;
            }
        }
    }
    const saved = c.__error().*;
    c.free(request);
    c.__error().* = saved;
    return ok;
}

// Reader is private behind the public opaque handle. One caller owns it;
// pthread fields use SDK-generated layouts, while this container has no C ABI.
const Worker = struct { thread: c.pthread_t, reader: *Reader, index: c_uint };
const Reader = struct {
    workers: [token_readers]Worker,
    threads: c_uint,
    mutex: c.pthread_mutex_t,
    work: c.pthread_cond_t,
    done: c.pthread_cond_t,
    active: bool,
    shutdown: bool,
    generation: u64,
    completed: [2]c_uint,
    errors: [2]c_int,
    tables: [2]Table,
    ids: [2][cols]u32,
    out: [*c]f32,
};
fn readTokenThread(context: ?*anyopaque) callconv(.c) ?*anyopaque {
    const worker: *const Worker = @ptrCast(@alignCast(context.?));
    const reader = worker.reader;
    const index = worker.index;
    var seen: u64 = 0;
    _ = c.pthread_mutex_lock(&reader.mutex);
    while (true) {
        while (reader.generation == seen and !reader.shutdown) _ = c.pthread_cond_wait(&reader.work, &reader.mutex);
        if (reader.shutdown) break;
        seen = reader.generation;
        _ = c.pthread_mutex_unlock(&reader.mutex);
        for (0..2) |table| {
            const first = index * cols / token_readers;
            const end = (index + 1) * cols / token_readers;
            const ok = ds4_engram_read(&reader.tables[table], @as([*]u32, @ptrCast(&reader.ids[table])) + first, end - first, reader.out + (table * cols + first) * dim);
            const err = if (ok) 0 else if (c.__error().* != 0) c.__error().* else c.EIO;
            _ = c.pthread_mutex_lock(&reader.mutex);
            if (reader.errors[table] == 0) reader.errors[table] = err;
            reader.completed[table] += 1;
            if (reader.completed[table] == token_readers) {
                if (table == 1) reader.active = false;
                _ = c.pthread_cond_broadcast(&reader.done);
            }
            _ = c.pthread_mutex_unlock(&reader.mutex);
        }
        _ = c.pthread_mutex_lock(&reader.mutex);
    }
    _ = c.pthread_mutex_unlock(&reader.mutex);
    return null;
}
// Partial creation joins every successfully started worker before releasing
// synchronization objects. Failure permits the caller's synchronous fallback.
export fn ds4_engram_reader_create() ?*Reader {
    const reader: *Reader = @ptrCast(@alignCast(c.calloc(1, @sizeOf(Reader)) orelse return null));
    var err = c.pthread_mutex_init(&reader.mutex, null);
    if (err != 0) {
        c.free(reader);
        c.__error().* = err;
        return null;
    }
    err = c.pthread_cond_init(&reader.work, null);
    if (err != 0) {
        _ = c.pthread_mutex_destroy(&reader.mutex);
        c.free(reader);
        c.__error().* = err;
        return null;
    }
    err = c.pthread_cond_init(&reader.done, null);
    if (err != 0) {
        _ = c.pthread_cond_destroy(&reader.work);
        _ = c.pthread_mutex_destroy(&reader.mutex);
        c.free(reader);
        c.__error().* = err;
        return null;
    }
    for (0..token_readers) |i| {
        reader.workers[i].reader = reader;
        reader.workers[i].index = @intCast(i);
        err = c.pthread_create(&reader.workers[i].thread, null, readTokenThread, &reader.workers[i]);
        if (err != 0) break;
        reader.threads += 1;
    }
    if (err == 0) return reader;
    _ = c.pthread_mutex_lock(&reader.mutex);
    reader.shutdown = true;
    _ = c.pthread_cond_broadcast(&reader.work);
    _ = c.pthread_mutex_unlock(&reader.mutex);
    for (reader.workers[0..reader.threads]) |worker| if (c.pthread_join(worker.thread, null) != 0) {
        c.abort();
    };
    _ = c.pthread_cond_destroy(&reader.done);
    _ = c.pthread_cond_destroy(&reader.work);
    _ = c.pthread_mutex_destroy(&reader.mutex);
    c.free(reader);
    c.__error().* = err;
    return null;
}
// Copy table descriptors and IDs, but borrow their open FDs and caller output.
// Keep both alive until wait(table=1) or free, even if table=0 reports an error.
// The four workers persist across generations; no table data is retained.
export fn ds4_engram_reader_start(maybe_reader: ?*Reader, tables: [*c]const Table, ids: [*c]const [cols]u32, out: [*c]f32) bool {
    const reader = maybe_reader orelse return fail(c.EINVAL);
    if (tables == null or ids == null or out == null) return fail(c.EINVAL);
    for (0..2) |table| {
        if (tables[table].fd < 0) return fail(c.EINVAL);
        for (ids[table]) |id| if (id >= tables[table].rows) return fail(c.EINVAL);
    }
    _ = c.pthread_mutex_lock(&reader.mutex);
    if (reader.active or reader.shutdown) {
        _ = c.pthread_mutex_unlock(&reader.mutex);
        return fail(c.EBUSY);
    }
    reader.tables[0] = tables[0];
    reader.tables[1] = tables[1];
    reader.ids[0] = ids[0];
    reader.ids[1] = ids[1];
    reader.errors = @splat(0);
    reader.completed = @splat(0);
    reader.out = out;
    reader.generation +%= 1;
    reader.active = true;
    _ = c.pthread_cond_broadcast(&reader.work);
    _ = c.pthread_mutex_unlock(&reader.mutex);
    return true;
}
export fn ds4_engram_reader_wait(maybe_reader: ?*Reader, table: c_uint) bool {
    const reader = maybe_reader orelse return fail(c.EINVAL);
    if (table >= 2) return fail(c.EINVAL);
    _ = c.pthread_mutex_lock(&reader.mutex);
    if (!reader.active and reader.completed[1] == 0) {
        _ = c.pthread_mutex_unlock(&reader.mutex);
        return fail(c.EINVAL);
    }
    while (reader.completed[table] < token_readers) _ = c.pthread_cond_wait(&reader.done, &reader.mutex);
    const err = reader.errors[table];
    _ = c.pthread_mutex_unlock(&reader.mutex);
    if (err != 0) c.__error().* = err;
    return err == 0;
}
// Shutdown wakes idle workers, lets any borrowed-output work finish, then joins
// before destroying locks and freeing the owner. It never detaches live workers.
export fn ds4_engram_reader_free(maybe_reader: ?*Reader) void {
    const reader = maybe_reader orelse return;
    _ = c.pthread_mutex_lock(&reader.mutex);
    reader.shutdown = true;
    _ = c.pthread_cond_broadcast(&reader.work);
    _ = c.pthread_mutex_unlock(&reader.mutex);
    for (reader.workers[0..reader.threads]) |worker| if (c.pthread_join(worker.thread, null) != 0) {
        c.abort();
    };
    _ = c.pthread_cond_destroy(&reader.done);
    _ = c.pthread_cond_destroy(&reader.work);
    _ = c.pthread_mutex_destroy(&reader.mutex);
    c.free(reader);
}
