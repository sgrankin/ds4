//! Durable byte-prefix KV checkpoints shared by the server and agent.
//!
//! The file layout and graph-payload helpers are shared; cache policy is not.
//! The server owns automatic byte-prefix reuse, while the agent uses this format
//! for explicit sessions. Protocol-owned trailers (for example exact tool-ID to
//! DSML mappings) remain behind the caller's trailer hooks.
//!
//! Public layouts and engine/libc declarations come from the real C headers.
//! Only this module's implementation is handwritten; no translated C bodies are
//! linked. Returned strings and staged payloads keep their libc ownership rules.
const std = @import("std");
const c = @import("c_api");
const Str = [*c]const u8;
const String = [*c]u8;
const FILE = c.FILE;
const Engine = c.ds4_engine;
const Session = c.ds4_session;
const Tokens = c.ds4_tokens;
const Options = c.ds4_kvstore_options;
const Entry = c.ds4_kvstore_entry;
const Store = c.ds4_kvstore;
const EvictionContext = c.ds4_kvstore_eviction_context;
const TrailerHooks = c.ds4_kvstore_trailer_hooks;
const LoadResult = c.ds4_kvstore_load_result;
const PayloadFile = c.ds4_session_payload_file;
const LogFn = @TypeOf(@as(Store, undefined).log);
const fixed_header = c.DS4_KVSTORE_FIXED_HEADER;
const half_life = c.DS4_KVSTORE_HIT_HALF_LIFE_SECONDS;
const max_u64 = std.math.maxInt(u64);
const file_version = 1;
// Header byte 20 versions the graph payload separately from the outer KVC file:
// a stable envelope does not make old serialized session internals safe to load.
const payload_abi = 2;

// Tokenizers can merge across a prompt boundary. Trim a small tail for cheap
// token-prefix hits; byte-prefix lookup handles alternate tokenizations of the
// same rendered bytes. Aligning to 2048 also preserves compressor finalization
// at the backend's prefill chunk boundaries.
const default_boundary_trim = 32;
const default_boundary_align = 2048;

// Hits decay as the workload changes. Continued-prefix waypoints are cheaper
// victims than useful anchors, particularly when the incoming entry supersedes
// them. Cold/evict/shutdown entries receive a soft prior, not immunity to eviction.
const min_effective_hits = 0.01;
const continued_prefix_min_factor = 0.05;
const continued_prefix_hit_factor = 0.45;
const anchor_reason_score_factor = 2.0;

// fopen returns a nullable C pointer. Internally a successfully opened file is
// non-null, while ownership still ends with fclose in the same operation.
fn openFile(path: Str, mode: Str) ?*FILE {
    const fp = c.fopen(path, mode);
    return if (fp == null) null else @ptrCast(fp);
}

fn countAsUsize(n: c_int) usize {
    return @intCast(n);
}
fn entryAt(kc: *const Store, i: c_int) *Entry {
    return @ptrCast(kc.entry + countAsUsize(i));
}
// Preserve C's narrowing/unsigned conversions at the ABI and on-disk boundary.
// These are intentional bit conversions, not checks that values are positive.
fn truncateByte(n: c_int) u8 {
    return @truncate(@as(c_uint, @bitCast(n)));
}
fn unsignedBits(n: c_int) u32 {
    return @bitCast(n);
}
fn signedBits(n: u32) c_int {
    return @bitCast(n);
}
fn asFloat(n: anytype) f64 {
    return @floatFromInt(n);
}
fn unixSeconds() u64 {
    return @bitCast(@as(i64, c.time(null)));
}
fn errno() c_int {
    return c.__error().*;
}
fn die(msg: Str) noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-kvstore: %s\n", msg);
    c.exit(1);
}
fn xmalloc(n: usize) *anyopaque {
    return c.malloc(if (n != 0) n else 1) orelse die("out of memory");
}
fn xrealloc(p: ?*anyopaque, n: usize) *anyopaque {
    return c.realloc(p, if (n != 0) n else 1) orelse die("out of memory");
}
// Strings exposed to C callers and hooks remain malloc/free-owned.
fn xstrdup(s: Str) String {
    const n = c.strlen(s);
    const p: String = @ptrCast(xmalloc(n + 1));
    _ = c.memcpy(p, s, n + 1);
    return p;
}
// ArrayList owns construction storage. A trailing NUL is added only when take()
// transfers the result to C. For u8, c_allocator uses ordinary malloc/realloc,
// so every returned pointer remains valid for the caller's libc free().
const TextBuffer = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn append(b: *TextBuffer, ptr: ?*const anyopaque, len: usize) void {
        if (len > std.math.maxInt(usize) - b.bytes.items.len - 1) die("buffer overflow");
        b.bytes.appendSlice(std.heap.c_allocator, readableBytes(ptr, len)) catch die("out of memory");
    }
    fn puts(b: *TextBuffer, text: Str) void {
        b.append(text, c.strlen(text));
    }
    fn putc(b: *TextBuffer, value: u8) void {
        b.append(&value, 1);
    }
    fn print(b: *TextBuffer, comptime fmt: [*:0]const u8, args: anytype) void {
        const n = @call(.auto, c.snprintf, .{ @as(String, null), @as(usize, 0), fmt } ++ args);
        if (n < 0) die("vsnprintf failed");
        const len = countAsUsize(n);
        if (len > std.math.maxInt(usize) - b.bytes.items.len - 1) die("buffer overflow");
        const destination = b.bytes.addManyAsSlice(std.heap.c_allocator, len + 1) catch die("out of memory");
        _ = @call(.auto, c.snprintf, .{ destination.ptr, destination.len, fmt } ++ args);
        b.bytes.items.len -= 1; // snprintf's NUL is not part of the payload.
    }
    fn take(b: *TextBuffer) String {
        const owned = b.bytes.toOwnedSliceSentinel(std.heap.c_allocator, 0) catch die("out of memory");
        return owned.ptr;
    }
};
fn monotonicSeconds() f64 {
    var ts: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &ts);
    return asFloat(ts.tv_sec) + asFloat(ts.tv_nsec) / 1_000_000_000.0;
}
fn logName(kc: ?*const Store) Str {
    return if (kc != null and kc.?.log_name != null) kc.?.log_name else "ds4";
}
// The callback borrows msg only for this call; the module frees it afterward.
fn logMessage(kc: ?*Store, kind: c.ds4_kvstore_log_type, comptime fmt: [*:0]const u8, args: anytype) void {
    if (kc == null or kc.?.log == null) return;
    const n = @call(.auto, c.snprintf, .{ @as(String, null), @as(usize, 0), fmt } ++ args);
    if (n < 0) return;
    const msg: String = @ptrCast(xmalloc(countAsUsize(n) + 1));
    _ = @call(.auto, c.snprintf, .{ msg, countAsUsize(n) + 1, fmt } ++ args);
    kc.?.log.?(kc.?.log_ud, kind, msg);
    c.free(msg);
}

// A zero length permits a null C pointer. For nonempty spans the public API's
// pointer/length contract applies; turn that span into a slice once internally.
fn readableBytes(ptr: ?*const anyopaque, len: usize) []const u8 {
    return if (len == 0) "" else @as([*]const u8, @ptrCast(ptr.?))[0..len];
}

export fn ds4_kvstore_default_options() Options {
    return .{ .min_tokens = 512, .cold_max_tokens = 30000, .continued_interval_tokens = 10000, .boundary_trim_tokens = default_boundary_trim, .boundary_align_tokens = default_boundary_align };
}
export fn ds4_kvstore_reason_code(reason: Str) u8 {
    if (reason == null) return c.DS4_KVSTORE_REASON_UNKNOWN;
    const reasons = [_]struct { text: Str, code: u8 }{
        .{ .text = "cold", .code = c.DS4_KVSTORE_REASON_COLD },
        .{ .text = "continued", .code = c.DS4_KVSTORE_REASON_CONTINUED },
        .{ .text = "evict", .code = c.DS4_KVSTORE_REASON_EVICT },
        .{ .text = "shutdown", .code = c.DS4_KVSTORE_REASON_SHUTDOWN },
        .{ .text = "agent-system", .code = c.DS4_KVSTORE_REASON_AGENT_SYSTEM },
        .{ .text = "agent-session", .code = c.DS4_KVSTORE_REASON_AGENT_SESSION },
    };
    for (reasons) |known| if (c.strcmp(reason, known.text) == 0) return known.code;
    return c.DS4_KVSTORE_REASON_UNKNOWN;
}
export fn ds4_kvstore_key_kind(flags: u8) Str {
    if (flags & c.DS4_KVSTORE_EXT_RESPONSES_VISIBLE != 0) return "responses-visible";
    if (flags & c.DS4_KVSTORE_EXT_THINKING_VISIBLE != 0) return "thinking-visible";
    return "token-text";
}
export fn ds4_kvstore_le_put32(p: [*c]u8, v: u32) void {
    for (0..4) |i| p[i] = @truncate(v >> @as(u5, @intCast(i * 8)));
}
export fn ds4_kvstore_le_get32(p: [*c]const u8) u32 {
    var v: u32 = 0;
    for (0..4) |i| v |= @as(u32, p[i]) << @as(u5, @intCast(i * 8));
    return v;
}
fn putLe64(p: [*c]u8, v: u64) void {
    for (0..8) |i| p[i] = @truncate(v >> @as(u6, @intCast(i * 8)));
}
fn getLe64(p: [*c]const u8) u64 {
    var v: u64 = 0;
    for (0..8) |i| v |= @as(u64, p[i]) << @as(u6, @intCast(i * 8));
    return v;
}
export fn ds4_kvstore_sha1_bytes_hex(ptr: ?*const anyopaque, len: usize, out: [*c]u8) void {
    const input = readableBytes(ptr, len);
    var digest: [20]u8 = undefined;
    std.crypto.hash.Sha1.hash(input, &digest, .{});
    const hex = "0123456789abcdef";
    for (digest, 0..) |v, i| {
        out[2 * i] = hex[v >> 4];
        out[2 * i + 1] = hex[v & 15];
    }
    out[40] = 0;
}
export fn ds4_kvstore_sha_hex_name(name: Str, sha: String) bool {
    if (c.strlen(name) != 43) return false;
    const filename = name[0..43];
    if (!std.mem.eql(u8, filename[40..], ".kv")) return false;
    for (filename[0..40], 0..) |value, i| {
        if (c.isxdigit(value) == 0) return false;
        sha[i] = @intCast(c.tolower(value));
    }
    sha[40] = 0;
    return true;
}
export fn ds4_kvstore_path_join(dir: Str, name: Str) String {
    var b: TextBuffer = .{};
    b.puts(dir);
    if (!std.mem.endsWith(u8, b.bytes.items, "/")) b.putc('/');
    b.puts(name);
    return b.take();
}
export fn ds4_kvstore_path_for_sha(kc: *Store, sha: Str) String {
    var name: [44]u8 = undefined;
    _ = c.memcpy(&name, sha, 40);
    _ = c.memcpy(name[40..].ptr, ".kv", 4);
    return ds4_kvstore_path_join(kc.dir, &name);
}
fn mkdirP(path: Str) bool {
    if (path == null or path[0] == 0) return false;
    const tmp = xstrdup(path);
    defer c.free(tmp);
    var p = tmp + 1;
    while (p[0] != 0) : (p += 1) {
        if (p[0] != '/') continue;
        p[0] = 0;
        if (c.mkdir(tmp, 0o700) != 0 and errno() != c.EEXIST) return false;
        p[0] = '/';
    }
    return c.mkdir(tmp, 0o700) == 0 or errno() == c.EEXIST;
}
export fn ds4_kvstore_entry_free(e: *Entry) void {
    c.free(e.path);
    @memset(std.mem.asBytes(e), 0);
}
export fn ds4_kvstore_clear(kc: *Store) void {
    var i: c_int = 0;
    while (i < kc.len) : (i += 1) ds4_kvstore_entry_free(entryAt(kc, i));
    c.free(kc.entry);
    kc.entry = null;
    kc.len = 0;
    kc.cap = 0;
}
fn appendEntry(kc: *Store, e: Entry) void {
    if (kc.len == kc.cap) {
        kc.cap = if (kc.cap != 0) kc.cap * 2 else 16;
        kc.entry = @ptrCast(@alignCast(xrealloc(kc.entry, countAsUsize(kc.cap) * @sizeOf(Entry))));
    }
    kc.entry[countAsUsize(kc.len)] = e;
    kc.len += 1;
}
// Entry names hash rendered bytes, not token IDs. The graph payload still
// contains the exact tokens; a filename only identifies a rendered byte prefix.
// model_id is header byte 7; Flash stays zero for older-file compatibility.
export fn ds4_kvstore_fill_header(h: [*c]u8, model_id: u8, quant_bits: u8, reason: u8, ext_flags: u8, tokens: u32, hits: u32, ctx_size: u32, created_at: u64, last_used: u64, payload_bytes: u64) void {
    @memset(h[0..fixed_header], 0);
    h[0] = 'K';
    h[1] = 'V';
    h[2] = 'C';
    h[3] = file_version;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    h[20] = payload_abi;
    putLe64(h + 24, created_at);
    putLe64(h + 32, last_used);
    putLe64(h + 40, payload_bytes);
}
// Routed-expert quantizations represented by checkpoint identity.
export fn ds4_kvstore_quant_bits_supported(bits: c_int) bool {
    return bits == 2 or bits == 4 or bits == 5 or bits == 6 or bits == 8;
}
export fn ds4_kvstore_read_header(fp: *FILE, e: *Entry, text_bytes: *u32) bool {
    var h: [fixed_header]u8 = undefined;
    if (c.fread(&h, 1, h.len, fp) != h.len) return false;
    if (h[0] != 'K' or h[1] != 'V' or h[2] != 'C' or h[3] != file_version or h[20] != payload_abi) return false;
    e.quant_bits = h[4];
    e.reason = if (h[5] <= c.DS4_KVSTORE_REASON_AGENT_SESSION) h[5] else c.DS4_KVSTORE_REASON_UNKNOWN;
    e.ext_flags = h[6];
    e.model_id = h[7];
    e.tokens = ds4_kvstore_le_get32(h[8..].ptr);
    e.hits = ds4_kvstore_le_get32(h[12..].ptr);
    e.ctx_size = ds4_kvstore_le_get32(h[16..].ptr);
    e.created_at = getLe64(h[24..].ptr);
    e.last_used = getLe64(h[32..].ptr);
    e.payload_bytes = getLe64(h[40..].ptr);
    var tb: [4]u8 = undefined;
    if (c.fread(&tb, 1, tb.len, fp) != tb.len) return false;
    text_bytes.* = ds4_kvstore_le_get32(&tb);
    e.text_bytes = text_bytes.*;
    return e.tokens != 0 and ds4_kvstore_quant_bits_supported(e.quant_bits);
}
export fn ds4_kvstore_read_entry_file(path: Str, sha: Str, out: *Entry) bool {
    var st: c.struct_stat = undefined;
    if (c.stat(path, &st) != 0 or st.st_size < fixed_header + 4) return false;
    const fp = openFile(path, "rb") orelse return false;
    var e = std.mem.zeroes(Entry);
    var text_bytes: u32 = 0;
    const ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    _ = c.fclose(fp);
    if (!ok) return false;
    const fixed: u64 = fixed_header + 4;
    if (max_u64 - fixed < text_bytes or max_u64 - fixed - text_bytes < e.payload_bytes) return false;
    if (@as(u64, @intCast(st.st_size)) < fixed + text_bytes + e.payload_bytes) return false;
    _ = c.memcpy(&e.sha, sha, 41);
    e.path = xstrdup(path);
    e.file_size = @intCast(st.st_size);
    out.* = e;
    return true;
}
fn refreshEntries(kc: *Store) void {
    if (!kc.enabled) return;
    ds4_kvstore_clear(kc);
    const directory = c.opendir(kc.dir);
    if (directory == null) return;
    defer _ = c.closedir(directory);
    while (true) {
        const directory_entry = c.readdir(directory);
        if (directory_entry == null) break;
        const name: Str = @ptrCast(&directory_entry.*.d_name);
        var sha: [41]u8 = undefined;
        if (!ds4_kvstore_sha_hex_name(name, &sha)) continue;
        const path = ds4_kvstore_path_join(kc.dir, name);
        var e = std.mem.zeroes(Entry);
        if (ds4_kvstore_read_entry_file(path, &sha, &e)) appendEntry(kc, e);
        c.free(path);
    }
}
export fn ds4_kvstore_touch_file(path: Str, hits: u32) bool {
    const fp = openFile(path, "r+b") orelse return false;
    defer _ = c.fclose(fp);
    var e = std.mem.zeroes(Entry);
    var text_bytes: u32 = 0;
    if (!ds4_kvstore_read_header(fp, &e, &text_bytes)) return false;
    var h: [fixed_header]u8 = undefined;
    ds4_kvstore_fill_header(&h, e.model_id, e.quant_bits, e.reason, e.ext_flags, e.tokens, hits, e.ctx_size, e.created_at, unixSeconds(), e.payload_bytes);
    return c.fseek(fp, 0, 0) == 0 and c.fwrite(&h, 1, h.len, fp) == h.len;
}
fn incomingSupersedes(e: *const Entry, incoming: ?*const EvictionContext) bool {
    const inc = incoming orelse return false;
    if (inc.text == null or e.reason != c.DS4_KVSTORE_REASON_CONTINUED or e.text_bytes == 0 or e.text_bytes > std.math.maxInt(usize)) return false;
    if (e.text_bytes >= inc.text_len or e.model_id != inc.model_id) return false;
    if (inc.reject_different_quant and e.quant_bits != inc.quant_bits) return false;
    // A smaller-context checkpoint can load into a larger context, not the
    // reverse. An incoming entry dominates only if it is as widely reusable.
    if (inc.ctx_size > e.ctx_size) return false;
    var sha: [41]u8 = undefined;
    ds4_kvstore_sha1_bytes_hex(inc.text, @intCast(e.text_bytes), &sha);
    return c.strcmp(&sha, &e.sha) == 0;
}
export fn ds4_kvstore_entry_eviction_score(e: ?*const Entry, live: ?*const Tokens, timestamp: u64, incoming: ?*const EvictionContext) f64 {
    // This policy was compiled with -ffast-math in C. Keep that float policy
    // explicit; changing it can change which nearly tied entry is evicted.
    @setFloatMode(.optimized);
    if (e == null or e.?.file_size == 0) return 0.0;
    _ = live;
    const ent = e.?;
    var hits = asFloat(ent.hits);
    const used_at = if (ent.last_used != 0) ent.last_used else ent.created_at;
    if (used_at == 0) {
        hits = 0;
    } else if (timestamp > used_at) {
        hits *= c.exp2(-asFloat(timestamp - used_at) / asFloat(@as(u64, half_life)));
        if (hits < min_effective_hits) hits = 0;
    }
    var score = (hits + 1.0) * asFloat(ent.tokens) / asFloat(ent.file_size);
    if (ent.reason == c.DS4_KVSTORE_REASON_COLD or ent.reason == c.DS4_KVSTORE_REASON_EVICT or ent.reason == c.DS4_KVSTORE_REASON_SHUTDOWN) score *= anchor_reason_score_factor;
    if (incomingSupersedes(ent, incoming)) score *= continued_prefix_min_factor + continued_prefix_hit_factor * (if (hits > 0) hits / (hits + 1.0) else 0.0);
    return score;
}
export fn ds4_kvstore_evict(kc: *Store, live: ?*const Tokens, extra_bytes: u64, incoming: ?*const EvictionContext) void {
    if (!kc.enabled or kc.budget_bytes == 0 or extra_bytes > kc.budget_bytes) return;
    refreshEntries(kc);
    const timestamp = unixSeconds();
    var total: u64 = 0;
    var i: c_int = 0;
    while (i < kc.len) : (i += 1) total +%= kc.entry[countAsUsize(i)].file_size;
    const target = kc.budget_bytes - extra_bytes;
    while (total > target and kc.len > 0) {
        var victim: c_int = 0;
        var score = ds4_kvstore_entry_eviction_score(entryAt(kc, 0), live, timestamp, incoming);
        i = 1;
        while (i < kc.len) : (i += 1) {
            const candidate = ds4_kvstore_entry_eviction_score(entryAt(kc, i), live, timestamp, incoming);
            if (candidate < score or (candidate == score and kc.entry[countAsUsize(i)].last_used < kc.entry[countAsUsize(victim)].last_used)) {
                victim = i;
                score = candidate;
            }
        }
        var e = kc.entry[countAsUsize(victim)];
        if (c.unlink(e.path) == 0) {
            logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s", .{ logName(kc), e.tokens, e.hits, asFloat(e.file_size) / 1048576.0, if (e.path != null) @as(Str, e.path) else @as(Str, "?") });
            total = if (total >= e.file_size) total - e.file_size else 0;
        } else {
            total = 0;
        }
        ds4_kvstore_entry_free(&e);
        _ = c.memmove(kc.entry + countAsUsize(victim), kc.entry + countAsUsize(victim + 1), countAsUsize(kc.len - victim - 1) * @sizeOf(Entry));
        kc.len -= 1;
    }
}
export fn ds4_kvstore_open(kc: *Store, dir: Str, budget_mb: u64, reject_different_quant: bool, opt: Options, log_name: Str, logger: LogFn, log_ud: ?*anyopaque) bool {
    @memset(std.mem.asBytes(kc), 0);
    if (dir == null) return false;
    kc.log_name = log_name;
    kc.log = logger;
    kc.log_ud = log_ud;
    if (!mkdirP(dir)) {
        logMessage(kc, c.DS4_KVSTORE_LOG_DEFAULT, "%s: failed to create KV cache directory %s: %s", .{ logName(kc), dir, c.strerror(errno()) });
        return false;
    }
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.budget_bytes = (if (budget_mb == 0) @as(u64, c.DS4_KVSTORE_DEFAULT_MB) else budget_mb) *% 1048576;
    kc.reject_different_quant = reject_different_quant;
    kc.opt = opt;
    ds4_kvstore_evict(kc, null, 0, null);
    logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)", .{ logName(kc), kc.dir, @as(c_ulonglong, kc.budget_bytes / 1048576), @as(Str, if (reject_different_quant) "reject" else "accept"), opt.min_tokens, opt.cold_max_tokens, opt.continued_interval_tokens, opt.boundary_trim_tokens, opt.boundary_align_tokens, @as(c_ulonglong, half_life) });
    return true;
}
export fn ds4_kvstore_close(kc: *Store) void {
    ds4_kvstore_clear(kc);
    c.free(kc.dir);
    @memset(std.mem.asBytes(kc), 0);
}
export fn ds4_kvstore_render_tokens_text(engine: ?*Engine, tokens: *const Tokens, out_len: ?*usize) String {
    var b: TextBuffer = .{};
    var i: c_int = 0;
    while (i < tokens.len) : (i += 1) {
        var len: usize = 0;
        const piece = c.ds4_token_text(engine, tokens.v[countAsUsize(i)], &len);
        b.append(piece, len);
        c.free(piece);
    }
    if (out_len) |n| n.* = b.bytes.items.len;
    return b.take();
}
export fn ds4_kvstore_byte_prefix_match(text: Str, text_len: usize, prefix: Str, prefix_len: usize) bool {
    if (prefix_len > text_len) return false;
    return std.mem.eql(u8, readableBytes(text, prefix_len), readableBytes(prefix, prefix_len));
}
export fn ds4_kvstore_tokens_copy_prefix(dst: *Tokens, src: ?*const Tokens, n: c_int) void {
    dst.len = 0;
    const source = src orelse return;
    const count = @min(n, source.len);
    var i: c_int = 0;
    while (i < count) : (i += 1) c.ds4_tokens_push(dst, source.v[countAsUsize(i)]);
}
export fn ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(engine: ?*Engine, exact_prefix: ?*const Tokens, suffix_text: Str, out: *Tokens) void {
    c.ds4_tokens_copy(out, exact_prefix);
    var suffix = std.mem.zeroes(Tokens);
    // The suffix can begin with native chat controls such as <｜User｜> or
    // </think>. Tokenize rendered chat, not plain text BPE.
    c.ds4_tokenize_rendered_chat(engine, if (suffix_text != null) suffix_text else "", &suffix);
    var i: c_int = 0;
    while (i < suffix.len) : (i += 1) c.ds4_tokens_push(out, suffix.v[countAsUsize(i)]);
    c.ds4_tokens_free(&suffix);
}
export fn ds4_kvstore_store_len(kc: *const Store, tokens: c_int) c_int {
    const trim = kc.opt.boundary_trim_tokens;
    const alignment = kc.opt.boundary_align_tokens;
    if (tokens > kc.opt.min_tokens + trim) {
        var stable = tokens - trim;
        if (alignment > 0) stable -= @rem(stable, alignment);
        if (stable >= kc.opt.min_tokens) return stable;
    }
    return tokens;
}
export fn ds4_kvstore_chat_anchor_pos(kc: *const Store, prompt: ?*const Tokens, user: c_int, assistant: c_int) c_int {
    if (prompt == null or user < 0 or assistant < 0) return -1;
    // Cold anchors should reuse across independent tasks. Stable user-role
    // scaffolding may precede the task: stop at the first assistant marker and
    // choose the last user marker before it, rather than the last one overall.
    var last_user: c_int = -1;
    var i: c_int = 0;
    while (i < prompt.?.len) : (i += 1) {
        const token = prompt.?.v[countAsUsize(i)];
        if (token == assistant) break;
        if (token == user) last_user = i;
    }
    return if (last_user >= kc.opt.min_tokens) last_user else -1;
}
fn continuedStoreStep(kc: *const Store) c_int {
    if (!kc.enabled or kc.opt.continued_interval_tokens <= 0) return 0;
    var step = kc.opt.continued_interval_tokens;
    const alignment = kc.opt.boundary_align_tokens;
    if (alignment > 0) {
        step = @divTrunc(step + alignment - 1, alignment) * alignment;
        if (step <= 0) step = alignment;
    }
    return step;
}
export fn ds4_kvstore_continued_store_target(kc: *const Store, live_tokens: c_int) c_int {
    const step = continuedStoreStep(kc);
    if (step <= 0 or live_tokens < kc.opt.min_tokens or @rem(live_tokens, step) != 0 or live_tokens <= kc.continued_last_store_tokens) return 0;
    return live_tokens;
}
export fn ds4_kvstore_note_store(kc: *Store, tokens: c_int) void {
    if (tokens > kc.continued_last_store_tokens) {
        kc.continued_last_store_tokens = tokens;
    }
}
export fn ds4_kvstore_suppress_continued_store(kc: *Store, tokens: c_int) c_int {
    if (ds4_kvstore_continued_store_target(kc, tokens) != tokens) return -1;
    const old = kc.continued_last_store_tokens;
    ds4_kvstore_note_store(kc, tokens);
    return old;
}
export fn ds4_kvstore_restore_suppressed_continued(kc: *Store, old_tokens: c_int, suppressed_tokens: c_int) void {
    if (old_tokens >= 0 and kc.continued_last_store_tokens == suppressed_tokens) kc.continued_last_store_tokens = old_tokens;
}
export fn ds4_kvstore_file_size_fits(kc: ?*const Store, text_bytes: u64, payload_bytes: u64, trailer_bytes: u64, file_out: ?*u64, required_out: ?*u64) bool {
    const fixed: u64 = fixed_header + 4;
    if (max_u64 - fixed < text_bytes or max_u64 - fixed - text_bytes < payload_bytes or max_u64 - fixed - text_bytes - payload_bytes < trailer_bytes) return false;
    const bytes = fixed + text_bytes + payload_bytes + trailer_bytes;
    if (file_out) |p| p.* = bytes;
    if (kc == null or kc.?.budget_bytes == 0) return true;
    // One snapshot's serialized size includes its trailer. Keep 1% headroom so
    // accounting/filesystem surprises do not immediately evict the new file.
    var slack = bytes / 100;
    if (bytes % 100 != 0) slack += 1;
    if (max_u64 - bytes < slack) return false;
    const required = bytes + slack;
    if (required_out) |p| p.* = required;
    return required <= kc.?.budget_bytes;
}

fn storedTextMatches(path: Str, sha: Str, text: Str, text_len: usize) bool {
    if (text_len > std.math.maxInt(u32)) return false;
    const fp = openFile(path, "rb") orelse return false;
    var hdr = std.mem.zeroes(Entry);
    var text_bytes: u32 = 0;
    var ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) and text_bytes == text_len;
    var stored: String = null;
    if (ok) {
        stored = @ptrCast(xmalloc(@as(usize, text_bytes) + 1));
        ok = c.fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    _ = c.fclose(fp);
    defer c.free(stored);
    if (!ok) return false;
    var stored_sha: [41]u8 = undefined;
    ds4_kvstore_sha1_bytes_hex(stored, text_bytes, &stored_sha);
    return c.strcmp(&stored_sha, sha) == 0 and (text_len == 0 or c.memcmp(stored, text, text_len) == 0);
}
fn existingFileCompatible(kc: *Store, path: Str, sha: Str, text: Str, text_len: usize, model_id: c_int, quant_bits: c_int, ctx_size: c_int) bool {
    if (c.access(path, 0) != 0) return false;
    var e = std.mem.zeroes(Entry);
    if (!ds4_kvstore_read_entry_file(path, sha, &e)) return false;
    const compatible = e.model_id == truncateByte(model_id) and (!kc.reject_different_quant or e.quant_bits == truncateByte(quant_bits)) and e.ctx_size <= unsignedBits(ctx_size) and storedTextMatches(path, sha, text, text_len);
    ds4_kvstore_entry_free(&e);
    if (!compatible) {
        if (c.unlink(path) == 0) logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache replaced incompatible file %s", .{ logName(kc), path });
        return false;
    }
    return true;
}
fn trailerSize(hooks: ?*const TrailerHooks, text: Str, bytes: ?*u64) bool {
    if (bytes) |p| p.* = 0;
    if (hooks == null or hooks.?.serialized_size == null) return true;
    return hooks.?.serialized_size.?(hooks.?.ud, text, bytes);
}
fn trailerWrite(hooks: ?*const TrailerHooks, fp: *FILE, text: Str, bytes: ?*u64) bool {
    if (bytes) |p| p.* = 0;
    if (hooks == null or hooks.?.write == null) return true;
    return hooks.?.write.?(hooks.?.ud, fp, text, bytes);
}
fn rewriteTrailer(path: Str, text: Str, hooks: ?*const TrailerHooks) void {
    var estimate: u64 = 0;
    if (hooks == null or hooks.?.write == null or hooks.?.serialized_size == null or !trailerSize(hooks, text, &estimate) or estimate == 0) return;
    const fp = openFile(path, "r+b") orelse return;
    defer _ = c.fclose(fp);
    var hdr = std.mem.zeroes(Entry);
    var text_bytes: u32 = 0;
    var ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    const end: u64 = fixed_header + 4 +% @as(u64, text_bytes) +% hdr.payload_bytes;
    if (ok and end <= std.math.maxInt(i64) and c.fseeko(fp, @intCast(end), 0) == 0 and c.ftruncate(c.fileno(fp), @intCast(end)) == 0) {
        var ignored: u64 = 0;
        ok = trailerWrite(hooks, fp, text, &ignored) and c.fflush(fp) == 0;
        if (ok and ignored > 0) {
            var h: [fixed_header]u8 = undefined;
            ds4_kvstore_fill_header(&h, hdr.model_id, hdr.quant_bits, hdr.reason, hdr.ext_flags | hooks.?.ext_flag, hdr.tokens, hdr.hits, hdr.ctx_size, hdr.created_at, unixSeconds(), hdr.payload_bytes);
            _ = c.fseeko(fp, 0, 0) == 0 and c.fwrite(&h, 1, h.len, fp) == h.len and c.fflush(fp) == 0;
        }
    }
}
fn setError(err: String, len: usize, msg: Str) void {
    if (err != null and len != 0) _ = c.snprintf(err, len, "%s", msg);
}
fn saveError(saved_errno: c_int, save_err: Str) Str {
    return if (saved_errno != 0) c.strerror(saved_errno) else if (save_err[0] != 0) save_err else "unknown error";
}
export fn ds4_kvstore_store_live_prefix_text(
    kc: *Store,
    engine: ?*Engine,
    session: ?*Session,
    tokens: ?*const Tokens,
    store_len: c_int,
    reason: Str,
    cache_text_override: Str,
    cache_text_ext: u8,
    cache_text_key: Str,
    hooks: ?*const TrailerHooks,
    err: String,
    err_len: usize,
) bool {
    if (!kc.enabled or tokens == null or store_len < kc.opt.min_tokens) return false;
    const original_len = tokens.?.len;
    var store_tokens = std.mem.zeroes(Tokens);
    ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);
    defer c.ds4_tokens_free(&store_tokens);
    const quant_bits = c.ds4_engine_routed_quant_bits(engine);
    if (!ds4_kvstore_quant_bits_supported(quant_bits)) return false;
    const model_id = c.ds4_engine_model_id(engine);
    var save_err: [160]u8 = @splat(0);
    const live_tokens: ?*const Tokens = c.ds4_session_tokens(session);
    if (live_tokens == null or live_tokens.?.len != store_tokens.len or !c.ds4_tokens_starts_with(live_tokens.?, &store_tokens)) {
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d", .{ logName(kc), store_tokens.len, reason, if (live_tokens != null) live_tokens.?.len else @as(c_int, -1) });
        return false;
    }
    var text_len: usize = 0;
    const text_override = cache_text_override != null and cache_text_override[0] != 0;
    const text = if (text_override) xstrdup(cache_text_override) else ds4_kvstore_render_tokens_text(engine, &store_tokens, &text_len);
    defer c.free(text);
    if (text_override) text_len = c.strlen(text);
    if (text_len > std.math.maxInt(u32)) {
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d because rendered text is too large", .{ logName(kc), store_tokens.len });
        return false;
    }
    var trailer_est: u64 = 0;
    if (!trailerSize(hooks, text, &trailer_est)) {
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed", .{ logName(kc), store_tokens.len, reason });
        return false;
    }
    var sha: [41]u8 = undefined;
    ds4_kvstore_sha1_bytes_hex(text, text_len, &sha);
    const path = ds4_kvstore_path_for_sha(kc, &sha);
    defer c.free(path);
    if (existingFileCompatible(kc, path, &sha, text, text_len, model_id, quant_bits, c.ds4_session_ctx(session))) {
        rewriteTrailer(path, text, hooks);
        return true;
    }
    // Stage the already-evaluated graph before admission/eviction. Its private
    // temporary file belongs to this operation and is freed on every exit;
    // write_staged_payload copies it into the KVC envelope without re-evaluation.
    var staged = std.mem.zeroes(PayloadFile);
    if (c.ds4_session_stage_payload(session, &staged, &save_err, save_err.len) != 0) {
        const msg = saveError(0, &save_err);
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d reason=%s because KV payload staging failed: %s", .{ logName(kc), store_tokens.len, reason, msg });
        setError(err, err_len, msg);
        return false;
    }
    defer c.ds4_session_payload_file_free(&staged);
    const payload_bytes = staged.bytes;
    var est_file: u64 = 0;
    var est_required: u64 = 0;
    if (!ds4_kvstore_file_size_fits(kc, text_len, payload_bytes, trailer_est, &est_file, &est_required)) {
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB", .{ logName(kc), store_tokens.len, reason, asFloat(est_file) / 1048576.0, asFloat(est_required) / 1048576.0, asFloat(kc.budget_bytes) / 1048576.0 });
        return false;
    }
    const incoming: EvictionContext = .{
        .text = text,
        .text_len = text_len,
        .model_id = truncateByte(model_id),
        .quant_bits = truncateByte(quant_bits),
        .ctx_size = unsignedBits(c.ds4_session_ctx(session)),
        .reject_different_quant = kc.reject_different_quant,
    };
    ds4_kvstore_evict(kc, live_tokens, est_file, &incoming);
    var tmpb: TextBuffer = .{};
    tmpb.print("%s.tmp.%ld", .{ path, @as(c_long, c.getpid()) });
    const tmp = tmpb.take();
    defer c.free(tmp);
    const save_t0 = monotonicSeconds();
    const fp = openFile(tmp, "wb") orelse {
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache failed to create %s: %s save=%.1f ms", .{ logName(kc), tmp, c.strerror(errno()), (monotonicSeconds() - save_t0) * 1000.0 });
        return false;
    };
    const timestamp = unixSeconds();
    var h: [fixed_header]u8 = undefined;
    var ext_flags: u8 = if (trailer_est > 0 and hooks != null) hooks.?.ext_flag else 0;
    if (text_override) ext_flags |= cache_text_ext;
    ds4_kvstore_fill_header(&h, truncateByte(model_id), truncateByte(quant_bits), ds4_kvstore_reason_code(reason), ext_flags, unsignedBits(store_tokens.len), 0, unsignedBits(c.ds4_session_ctx(session)), timestamp, timestamp, payload_bytes);
    var tb: [4]u8 = undefined;
    ds4_kvstore_le_put32(&tb, @intCast(text_len));
    var trailer_bytes: u64 = 0;
    c.__error().* = 0;
    var ok = c.fwrite(&h, 1, h.len, fp) == h.len and
        c.fwrite(&tb, 1, tb.len, fp) == tb.len and
        c.fwrite(text, 1, text_len, fp) == text_len and
        c.ds4_session_write_staged_payload(&staged, fp, &save_err, save_err.len) == 0 and
        trailerWrite(hooks, fp, text, &trailer_bytes) and
        c.fflush(fp) == 0;
    var saved_errno = errno();
    if (c.fclose(fp) != 0) {
        if (saved_errno == 0) saved_errno = errno();
        ok = false;
    }
    var final_file: u64 = 0;
    var final_required: u64 = 0;
    var over_budget = false;
    if (ok and !ds4_kvstore_file_size_fits(kc, text_len, payload_bytes, trailer_bytes, &final_file, &final_required)) {
        over_budget = true;
        ok = false;
    }
    if (ok and c.rename(tmp, path) != 0) {
        saved_errno = errno();
        ok = false;
    }
    const save_ms = (monotonicSeconds() - save_t0) * 1000.0;
    if (!ok) {
        if (over_budget) {
            logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms", .{ logName(kc), store_tokens.len, reason, asFloat(final_file) / 1048576.0, asFloat(final_required) / 1048576.0, asFloat(kc.budget_bytes) / 1048576.0, save_ms });
        } else {
            logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache store failed (%s): %s save=%.1f ms", .{ logName(kc), reason, saveError(saved_errno, &save_err), save_ms });
        }
        setError(err, err_len, saveError(saved_errno, &save_err));
        _ = c.unlink(tmp);
    } else {
        const key: Str = if (text_override) (if (cache_text_key != null) cache_text_key else "visible-transcript") else "token-text";
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms", .{ logName(kc), store_tokens.len, original_len - store_tokens.len, reason, key, asFloat(fixed_header + 4 +% @as(u64, text_len) +% payload_bytes +% trailer_bytes) / 1048576.0, save_ms });
    }
    return ok;
}
export fn ds4_kvstore_store_live_prefix(kc: *Store, engine: ?*Engine, session: ?*Session, tokens: ?*const Tokens, store_len: c_int, reason: Str, hooks: ?*const TrailerHooks, err: String, err_len: usize) bool {
    return ds4_kvstore_store_live_prefix_text(kc, engine, session, tokens, store_len, reason, null, 0, null, hooks, err, err_len);
}
export fn ds4_kvstore_maybe_store_continued(kc: *Store, engine: ?*Engine, session: ?*Session, hooks: ?*const TrailerHooks, err: String, err_len: usize) bool {
    const session_tokens: ?*const Tokens = c.ds4_session_tokens(session);
    const tokens = session_tokens orelse return false;
    const target = ds4_kvstore_continued_store_target(kc, tokens.len);
    if (target == 0) return false;
    if (ds4_kvstore_store_live_prefix(kc, engine, session, tokens, target, "continued", hooks, err, err_len)) {
        ds4_kvstore_note_store(kc, target);
        return true;
    }
    return false;
}
export fn ds4_kvstore_find_text_prefix(kc: *Store, prompt_text: Str, model_id: c_int, quant_bits: c_int, ctx_size: c_int) c_int {
    if (prompt_text == null) return -1;
    const prompt_bytes = c.strlen(prompt_text);
    refreshEntries(kc);
    var best: c_int = -1;
    var i: c_int = 0;
    while (i < kc.len) : (i += 1) {
        const e = entryAt(kc, i);
        if (e.text_bytes > prompt_bytes or e.text_bytes > std.math.maxInt(usize)) continue;
        if (signedBits(e.tokens) < kc.opt.min_tokens or e.model_id != truncateByte(model_id) or unsignedBits(ctx_size) < e.ctx_size) continue;
        if (kc.reject_different_quant and e.quant_bits != truncateByte(quant_bits)) continue;
        if (best >= 0) {
            const b = entryAt(kc, best);
            if (e.text_bytes < b.text_bytes or (e.text_bytes == b.text_bytes and e.tokens <= b.tokens)) continue;
        }
        var sha: [41]u8 = undefined;
        ds4_kvstore_sha1_bytes_hex(prompt_text, @intCast(e.text_bytes), &sha);
        if (c.strcmp(&sha, &e.sha) == 0) best = i;
    }
    return best;
}
export fn ds4_kvstore_try_load_text(kc: *Store, engine: ?*Engine, session: ?*Session, prompt_text: Str, effective_prompt: ?*Tokens, result: ?*LoadResult, hooks: ?*const TrailerHooks, responses_protocol: bool) c_int {
    if (result) |r| @memset(std.mem.asBytes(r), 0);
    if (effective_prompt) |p| p.len = 0;
    if (!kc.enabled or prompt_text == null) return 0;
    const quant_bits = c.ds4_engine_routed_quant_bits(engine);
    if (!ds4_kvstore_quant_bits_supported(quant_bits)) return 0;
    const model_id = c.ds4_engine_model_id(engine);
    const prompt_bytes = c.strlen(prompt_text);
    const index = ds4_kvstore_find_text_prefix(kc, prompt_text, model_id, quant_bits, c.ds4_session_ctx(session));
    if (index < 0) return 0;
    const e = kc.entry[countAsUsize(index)];
    const path = xstrdup(e.path);
    defer c.free(path);
    const load_t0 = monotonicSeconds();
    const fp = openFile(path, "rb") orelse return 0;
    var text_bytes: u32 = 0;
    var hdr = std.mem.zeroes(Entry);
    var fail_reason: Str = "invalid header";
    var header_ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    var cached_text: String = null;
    defer c.free(cached_text);
    if (header_ok) {
        if (hdr.model_id != truncateByte(model_id)) {
            header_ok = false;
            fail_reason = "cached checkpoint was written for a different model";
        } else if (text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = @ptrCast(xmalloc(@as(usize, text_bytes) + 1));
            if (c.fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = 0;
                var text_sha: [41]u8 = undefined;
                ds4_kvstore_sha1_bytes_hex(cached_text, text_bytes, &text_sha);
                if (c.strcmp(&text_sha, &e.sha) != 0) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes, cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    var err: [160]u8 = @splat(0);
    var loaded: c_int = 0;
    const sep: Str = if (responses_protocol) " " else "";
    const proto: Str = if (responses_protocol) "RESPPROTO" else "";
    if (header_ok and c.ds4_session_load_payload(session, fp, hdr.payload_bytes, &err, err.len) == 0) {
        const loaded_tokens: ?*const Tokens = c.ds4_session_tokens(session);
        if (loaded_tokens != null and loaded_tokens.?.len == signedBits(hdr.tokens)) {
            loaded = signedBits(hdr.tokens);
            // Lookup was by rendered bytes, but restored state owns its exact
            // token history. Retain those IDs and tokenize only the text after
            // the matched byte prefix, avoiding boundary retokenization drift.
            if (effective_prompt) |p| ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(engine, loaded_tokens, prompt_text + text_bytes, p);
            if (hooks != null and hooks.?.load != null and hdr.ext_flags & hooks.?.ext_flag != 0) _ = hooks.?.load.?(hooks.?.ud, fp, hooks.?.load_wanted);
        } else {
            c.ds4_session_invalidate(session);
            _ = c.unlink(path);
            logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache discarded corrupt text-prefix payload%s%s %s", .{ logName(kc), sep, proto, path });
        }
    } else {
        if (header_ok) c.ds4_session_invalidate(session);
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache load failed%s%s %s: %s load=%.1f ms", .{ logName(kc), sep, proto, path, if (header_ok) @as(Str, &err) else fail_reason, (monotonicSeconds() - load_t0) * 1000.0 });
    }
    _ = c.fclose(fp);
    if (loaded > 0) {
        const load_ms = (monotonicSeconds() - load_t0) * 1000.0;
        kc.continued_last_store_tokens = loaded;
        _ = ds4_kvstore_touch_file(path, hdr.hits +% 1);
        logMessage(kc, c.DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s", .{ logName(kc), sep, proto, loaded, text_bytes, @as(c_uint, hdr.quant_bits), ds4_kvstore_key_kind(hdr.ext_flags), load_ms, path });
        if (result) |r| {
            r.tokens = loaded;
            r.text_bytes = text_bytes;
            r.quant_bits = hdr.quant_bits;
            r.ext_flags = hdr.ext_flags;
            r.load_ms = load_ms;
            r.path = xstrdup(path);
        }
    }
    return loaded;
}
// The result path is an independent libc-owned copy, not an entry borrow.
export fn ds4_kvstore_load_result_free(result: ?*LoadResult) void {
    const r = result orelse return;
    c.free(r.path);
    @memset(std.mem.asBytes(r), 0);
}
