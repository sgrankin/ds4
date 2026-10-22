//! SSD expert-cache sizing and the explicit simulated-memory diagnostic.
//! Public layouts and POSIX declarations come from the C headers; successful
//! MemoryLock acquisition owns one mapping until release, not a heap allocation.
const std = @import("std");
const c = @import("c_api");
const MemoryLock = c.ds4_ssd_memory_lock;
const CachePlan = c.ds4_ssd_cache_plan;
const gib: u64 = 1 << 30;

// Accept decimal positive digits, optionally followed by case-insensitive GB.
// No signs, spaces, fractions or other suffixes. Outputs are zeroed before any
// validation, and libc range/errno behavior remains the C boundary's authority.
export fn ds4_parse_gib_arg(maybe_s: ?[*:0]const u8, maybe_bytes: ?*u64) bool {
    if (maybe_bytes) |bytes| bytes.* = 0;
    const s = maybe_s orelse return false;
    if (s[0] == 0 or maybe_bytes == null) return false;
    const text = std.mem.span(s);
    var len = text.len;
    if (len > 2 and (text[len - 2] == 'g' or text[len - 2] == 'G') and
        (text[len - 1] == 'b' or text[len - 1] == 'B')) len -= 2;
    if (len == 0) return false;
    for (text[0..len]) |ch| if (ch < '0' or ch > '9') return false;
    var number: [32]u8 = undefined;
    if (len >= number.len) return false;
    @memcpy(number[0..len], text[0..len]);
    number[len] = 0;
    c.__error().* = 0;
    const value = c.strtoull(@ptrCast(&number), null, 10);
    if (c.__error().* != 0 or value == 0 or value > std.math.maxInt(u64) / gib) return false;
    maybe_bytes.?.* = value * gib;
    return true;
}

// A bare integer counts experts; a GB suffix requests a byte budget instead.
// Exactly one output is set on success, and both remain zero on parse failure.
export fn ds4_parse_streaming_cache_experts_arg(maybe_s: ?[*:0]const u8, maybe_experts: ?*u32, maybe_bytes: ?*u64) bool {
    if (maybe_experts) |experts| experts.* = 0;
    if (maybe_bytes) |bytes| bytes.* = 0;
    const s = maybe_s orelse return false;
    if (s[0] == 0 or maybe_experts == null or maybe_bytes == null) return false;
    const text = std.mem.span(s);
    if (text.len > 2 and (text[text.len - 2] == 'g' or text[text.len - 2] == 'G') and
        (text[text.len - 1] == 'b' or text[text.len - 1] == 'B'))
        return ds4_parse_gib_arg(s, maybe_bytes);
    for (text) |ch| if (ch < '0' or ch > '9') return false;
    c.__error().* = 0;
    const value = c.strtoul(s, null, 10);
    if (c.__error().* != 0 or value == 0 or value > std.math.maxInt(u32)) return false;
    maybe_experts.?.* = @intCast(value);
    return true;
}

export fn ds4_ssd_cache_experts_for_byte_budget(bytes: u64, per_expert_bytes: u64) u32 {
    if (bytes == 0 or per_expert_bytes == 0) return 0;
    const experts = bytes / per_expert_bytes;
    if (experts == 0 or experts > std.math.maxInt(u32)) return 0;
    return @intCast(experts);
}

fn autoCachePercent(default_percent: u32) u64 {
    const env = c.getenv("DS4_SSD_AUTO_CACHE_PCT");
    if (env != null) {
        if (env[0] != 0) {
            c.__error().* = 0;
            var end: [*c]u8 = null;
            const value = c.strtoul(env, &end, 10);
            if (end != null and @intFromPtr(end) != @intFromPtr(env) and
                end[0] == 0 and c.__error().* == 0 and value >= 50 and value <= 95)
                return value;
            _ = c.fprintf(c.__stderrp, "ds4: invalid DS4_SSD_AUTO_CACHE_PCT=%s (want 50..95); using default\n", env);
        }
    }
    // The original streaming policy trades random SSD misses against resident
    // experts. Its ROCm APU default was measured at 80%, not a promise that the
    // same split fits every unified-memory host: pinned upload staging, prefill
    // headroom and cache growth ride above steady-state residency. Preserve the
    // caller's default; only an explicit 50..95 environment value overrides it.
    return default_percent;
}

export fn ds4_ssd_auto_cache_plan(recommended_bytes: u64, default_percent: u32, model_limit_bytes: u64, non_routed_bytes: u64, per_expert_bytes: u64, max_model_experts: u64, maybe_out: ?*CachePlan) bool {
    const out = maybe_out orelse return false;
    @memset(std.mem.asBytes(out), 0);
    if (recommended_bytes == 0 or per_expert_bytes == 0 or default_percent < 50 or default_percent > 95) return false;
    const pct = autoCachePercent(default_percent);
    // Keep C's overflow policy: saturate the target before applying the model
    // limit, then plan at least one expert even when non-routed bytes consume it.
    out.model_target_bytes = if (recommended_bytes > std.math.maxInt(u64) / pct)
        std.math.maxInt(u64)
    else
        recommended_bytes * pct / 100;
    if (model_limit_bytes != 0 and out.model_target_bytes > model_limit_bytes) out.model_target_bytes = model_limit_bytes;
    if (out.model_target_bytes > non_routed_bytes) out.cache_bytes = out.model_target_bytes - non_routed_bytes;
    var cache_experts = out.cache_bytes / per_expert_bytes;
    if (cache_experts == 0) cache_experts = 1;
    if (max_model_experts != 0 and cache_experts > max_model_experts) cache_experts = max_model_experts;
    if (cache_experts > std.math.maxInt(u32)) cache_experts = std.math.maxInt(u32);
    out.cache_experts = @intCast(cache_experts);
    // This product deliberately keeps unsigned C wrapping, rather than adding
    // a new rejection rule for values outside the normal model-size domain.
    out.effective_cache_bytes = cache_experts *% per_expert_bytes;
    return out.cache_experts != 0;
}

export fn ds4_ssd_memory_lock_acquire(maybe_lock: ?*MemoryLock, bytes: u64) bool {
    @setFloatMode(.optimized); // Match the C TU's GiB diagnostic arithmetic.
    const lock = maybe_lock orelse return false;
    lock.* = .{ .ptr = null, .bytes = 0 };
    if (bytes == 0) return true;
    if (bytes > std.math.maxInt(usize)) {
        _ = c.fprintf(c.__stderrp, "ds4: --simulate-used-memory is too large for this process\n");
        return false;
    }
    const mapping = c.mmap(null, @intCast(bytes), c.PROT_READ | c.PROT_WRITE, c.MAP_PRIVATE | c.MAP_ANONYMOUS, -1, 0);
    if (mapping == c.MAP_FAILED) {
        _ = c.fprintf(c.__stderrp, "ds4: --simulate-used-memory mmap %.2f GiB failed: %s\n", @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gib)), c.strerror(c.__error().*));
        return false;
    }
    const ptr = mapping.?;
    const page_long = c.sysconf(c._SC_PAGESIZE);
    const page: u64 = if (page_long > 0) @intCast(page_long) else 4096;
    const chunk_bytes: u64 = 256 * 1024 * 1024;
    // Touch and lock bounded chunks. One giant mlock is harder to diagnose and
    // can cause long uninterruptible VM work on macOS. The volatile page writes
    // really commit memory; this opt-in diagnostic is intentionally not cheap.
    const p: [*]volatile u8 = @ptrCast(ptr);
    var locked: u64 = 0;
    var off: u64 = 0;
    while (off < bytes) : (off +%= chunk_bytes) {
        const len: u64 = @min(bytes - off, chunk_bytes);
        var pos = off;
        while (pos < off + len) : (pos += page) p[@intCast(pos)] = @truncate(pos / page);
        if (len != 0) p[@intCast(off + len - 1)] = 1;
        const chunk = @as([*]u8, @ptrCast(ptr)) + @as(usize, @intCast(off));
        if (c.mlock(chunk, @intCast(len)) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n", @as(f64, @floatFromInt(locked)) / @as(f64, @floatFromInt(gib)), @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gib)), c.strerror(c.__error().*));
            // Only the completed prefix is locked. Release it, then unmap the
            // entire allocation; publish no ownership in lock on this failure.
            if (locked != 0) _ = c.munlock(ptr, @intCast(locked));
            _ = c.munmap(ptr, @intCast(bytes));
            return false;
        }
        locked += len;
    }
    lock.* = .{ .ptr = ptr, .bytes = bytes };
    _ = c.fprintf(c.__stderrp, "ds4: simulated used memory: locked %.2f GiB before model load\n", @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gib)));
    return true;
}

// Release is idempotent for a zeroed/released handle. Acquisition expects an
// empty handle; it does not implicitly release an older mapping passed in it.
export fn ds4_ssd_memory_lock_release(maybe_lock: ?*MemoryLock) void {
    const lock = maybe_lock orelse return;
    if (lock.ptr == null or lock.bytes == 0) return;
    _ = c.munlock(lock.ptr, @intCast(lock.bytes));
    _ = c.munmap(lock.ptr, @intCast(lock.bytes));
    lock.* = .{ .ptr = null, .bytes = 0 };
}
