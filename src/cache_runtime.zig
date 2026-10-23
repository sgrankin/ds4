//! Shared Server/Agent cache admission and stable forward-policy identity.
//! Model/engine handles are borrowed; only the public cache owns stored bytes.
const std = @import("std");
const c = @import("cache_runtime_c");
const allocator = std.heap.c_allocator;

fn bytes(value: [*c]const u8) []const u8 {
    return std.mem.span(value);
}
fn failure(errors: []u8, message: [*:0]const u8) ?*c.ds4_state_cache {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message);
    return null;
}
fn megabytes(name: [*:0]const u8, fallback: u64) ?u64 {
    const value = c.getenv(name);
    if (value == null) return fallback << 20;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const amount = c.strtoull(value, &end, 10);
    // Preserve libc's accepted spelling, including whitespace and '+', while
    // rejecting the same explicit negative/empty/trailing/overflow cases.
    if (c.__error().* != 0 or value[0] == 0 or value[0] == '-' or end[0] != 0 or amount > (std.math.maxInt(u64) >> 20)) return null;
    return amount << 20;
}
fn unsignedLongLong(value: anytype) c_ulonglong {
    return switch (@typeInfo(@TypeOf(value)).int.signedness) {
        .signed => @bitCast(@as(i64, value)),
        .unsigned => @intCast(value),
    };
}
fn earlier(_: void, first: [*c]const u8, second: [*c]const u8) bool {
    return std.mem.order(u8, bytes(first), bytes(second)) == .lt;
}

pub fn open(engine: ?*c.ds4_engine, options: *const c.ds4_engine_options, context: c_int, profile: [*c]u8, errors: []u8) ?*c.ds4_state_cache {
    const directory = c.getenv("DS4_STATE_CACHE_DIR");
    if (directory == null or directory[0] == 0) return null;
    if (!c.ds4_engine_is_deepseek41(engine) or c.ds4_tp_enabled(&options.tp) or options.directional_steering_file != null or options.backend != c.DS4_BACKEND_METAL)
        return failure(errors, "state cache requires native text DS4.1 on Metal");

    const model_path = c.ds4_engine_model_path(engine);
    const hot = megabytes("DS4_STATE_CACHE_HOT_MB", 512);
    // C short-circuits: a malformed HOT value does not parse SSD or stat.
    if (hot == null) return failure(errors, "invalid state cache limits or model identity");
    const disk = megabytes("DS4_STATE_CACHE_SSD_MB", 8192);
    if (disk == null or disk.? == 0) return failure(errors, "invalid state cache limits or model identity");
    var metadata: c.struct_stat = undefined;
    if (model_path == null or c.stat(model_path, &metadata) != 0)
        return failure(errors, "invalid state cache limits or model identity");

    var count: usize = 0;
    var cursor = c.environ;
    while (cursor[0] != null) : (cursor += 1) {
        if (std.mem.startsWith(u8, bytes(cursor[0]), "DS4_METAL_")) count += 1;
    }
    const allocated = allocator.alloc([*c]const u8, @max(count, 1)) catch null;
    var identity: [*c]u8 = null;
    var length: usize = 0;
    const stream = c.open_memstream(&identity, &length);
    if (allocated == null or stream == null) {
        if (allocated) |policy| allocator.free(policy);
        if (stream != null) _ = c.fclose(stream);
        c.free(identity);
        return failure(errors, "cannot allocate state cache identity");
    }
    const policy = allocated.?;
    defer allocator.free(policy);
    defer c.free(identity);
    var stream_open = true;
    defer if (stream_open) {
        _ = c.fclose(stream);
    };
    var used: usize = 0;
    cursor = c.environ;
    while (cursor[0] != null) : (cursor += 1) {
        if (std.mem.startsWith(u8, bytes(cursor[0]), "DS4_METAL_")) {
            policy[used] = cursor[0];
            used += 1;
        }
    }
    std.mem.sort([*c]const u8, policy[0..count], {}, earlier);

    // Keep the existing spelling (DPR1) and field widths byte-for-byte. Cache
    // profiles include the file identity, policy and replay execution profile.
    const modified = if (@hasField(c.struct_stat, "st_mtimespec")) metadata.st_mtimespec else metadata.st_mtim;
    _ = c.fprintf(stream, "DSV4-v2-DPR1-20261003\n%s\n%llu:%llu:%llu:%lld:%ld:%d:%d\n", model_path, unsignedLongLong(metadata.st_dev), unsignedLongLong(metadata.st_ino), unsignedLongLong(metadata.st_size), @as(c_longlong, modified.tv_sec), @as(c_long, modified.tv_nsec), context, c.ds4_engine_routed_quant_bits(engine));
    for (policy[0..count]) |entry| _ = c.fprintf(stream, "%s\n", entry);
    _ = c.fprintf(stream, "prefill_chunk=%u quality=%d\n", c.ds4_engine_prefill_chunk(engine), @as(c_int, @intFromBool(options.quality)));
    // Replay checkpoints belong to their configured K/selected policy too.
    const replay = c.getenv("DS4_V41_DECODER_REPLAY_ROWS");
    if (replay != null and !std.mem.eql(u8, bytes(replay), "0")) {
        const selected = c.getenv("DS4_V41_DECODER_REPLAY_SELECTED");
        _ = c.fprintf(stream, "decoder_replay=DRP1 rows=%s selected=%d\n", replay, @as(c_int, @intFromBool(selected != null and std.mem.eql(u8, bytes(selected), "1"))));
    }
    const closed = c.fclose(stream);
    stream_open = false;
    if (closed != 0) return failure(errors, "cannot finish cache identity");
    c.ds4_kvstore_sha1_bytes_hex(identity, length, profile);
    return c.ds4_state_cache_open(directory, hot.?, disk.?, errors.ptr, errors.len);
}
