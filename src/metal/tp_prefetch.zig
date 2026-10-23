//! Gate-time bandwidth prefetch: prioritized ranges are cut to what fits in
//! this rank's observed exchange wait. Transport updates the actual EWMA; no
//! shadow metric or plan storage is introduced here.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("routed_batch.zig").Command;
const Policy = c.ds4_metal_tp_prefetch_policy;
const limit = c.DS4_TP_PREFETCH_MAX;
// C's function-local cached tuning survives GPU cleanup/reinitialization.
var margin_us: f64 = -1;
var bytes_per_us: f64 = -1;
fn state() *Policy {
    return @ptrCast(c.ds4_metal_tp_prefetch_policy_state());
}
pub fn plan(gate: u32, map: ?*const anyopaque, size: u64, offsets: [*c]const u64, lengths: [*c]const u64, count: u32) c_int {
    if (gate >= 2) return 0;
    const s = state();
    if (!s.enabled or !c.ds4_metal_tp_policy_state().*.poll_gates or map == null) {
        s.count[gate] = 0;
        return 1;
    }
    var used: u32 = 0;
    for (0..@min(count, limit)) |index| {
        const bytes = lengths[index];
        const offset = offsets[index];
        if (bytes == 0 or offset > size or bytes > size - offset) continue;
        s.plan[gate][used] = .{ .offset = offset, .bytes = bytes };
        used += 1;
    }
    s.map = map;
    s.map_size = size;
    s.count[gate] = used;
    return 1;
}
fn budget(wait_us: f64) u64 {
    // This exact host expression is compiled with C's optimized FP policy.
    @setFloatMode(.optimized);
    if (margin_us < 0) {
        margin_us = @floatFromInt(c.ds4_zig_metal_env_u64("DS4_TP_PREFETCH_MARGIN_US", 8, 0, 1000));
        bytes_per_us = @floatFromInt(c.ds4_zig_metal_env_u64("DS4_TP_PREFETCH_BYTES_PER_US", c.DS4_TP_PREFETCH_BYTES_PER_US, 1000, 10000000));
    }
    var bytes: u64 = if (wait_us > margin_us) @intFromFloat((wait_us - margin_us) * bytes_per_us) else 0;
    // Initial observations use a small default, so a slow rank does not delay
    // its own release pickup. Environment values remain process-cached.
    if (wait_us == 0) bytes = 4 << 20;
    return bytes;
}
pub fn encode(gate: u32) c_int {
    const s = state();
    const count = s.count[gate];
    s.count[gate] = 0;
    if (count == 0 or !c.ds4_metal_tp_get_facts().batch) return 1;
    const scratch = c.ds4_metal_tp_prefetch_storage();
    if (scratch.buffer == null) return 1;
    const pso = c.ds4_metal_routed_pipeline(0, "kernel_touch_u8_stride_table", 0);
    if (pso == null) return 1;
    var remaining = budget(s.exchange_ewma_us[gate]);
    if (remaining == 0) return 1;
    var launch = std.mem.zeroes(c.ds4_metal_tp_prefetch_launch);
    launch.pipeline = pso;
    launch.scratch = scratch;
    launch.threads = 256;
    var retained: [limit]Buffer = @splat(.{ .view = std.mem.zeroes(c.ds4_metal_attention_view) });
    defer for (retained) |buffer| buffer.deinit();
    var total_lines: u64 = 0;
    for (s.plan[gate][0..count]) |range| {
        if (remaining == 0) break;
        const bytes = @min(range.bytes, remaining);
        remaining -= bytes;
        if (bytes == 0) continue;
        var inner: u64 = 0;
        const pointer = api.retainModelRange(s.map, s.map_size, range.offset, bytes, &inner) orelse continue;
        const view = c.ds4_metal_attention_view{ .buffer = pointer, .offset = inner, .bytes = bytes };
        const lines = (bytes +% 127) / 128;
        // The original bound is tested after wrapping the actual model view.
        // Each rejected temporary retain is retired before trying another row.
        if (total_lines +% lines > 1 << 20) {
            (Buffer{ .view = view }).deinit();
            break;
        }
        const address = c.ds4_metal_tp_prefetch_address(view);
        if (address == 0) {
            (Buffer{ .view = view }).deinit();
            continue;
        }
        retained[launch.count] = .{ .view = view };
        launch.ranges[launch.count] = view;
        launch.addresses[launch.count] = address;
        launch.lines[launch.count] = @intCast(lines);
        launch.count += 1;
        total_lines +%= lines;
    }
    // As in C, a nonzero budget still opens/ends an encoder when no model view
    // was usable. The fixed leaf skips the dispatch for an empty touch table.
    launch.groups = (total_lines + 255) / 256;
    var command = Command{};
    defer command.drop();
    if (!command.acquire()) return 0;
    return c.ds4_metal_tp_prefetch_encode(command.token.?, &launch);
}
