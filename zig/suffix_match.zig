//! Shared immutable-endpoint suffix matching for the two actual native callers.
//! Plan is the caller's generated real ds4_suffix_match_plan, never a new record.
const std = @import("std");

/// Match one immutable endpoint. Token-zero matches belong to exact prefix
/// lookup, and all bytes at/after cutoff remain fresh. Highest reused count
/// wins; descending destination ends preserve the original tie order.
pub fn shift(comptime Plan: type, source_pointer: [*c]const c_int, source_end: u32, edited_pointer: [*c]const c_int, edited_length: u32, cutoff: u32, right: bool, plan: *Plan) bool {
    plan.* = std.mem.zeroes(Plan);
    if (source_pointer == null or edited_pointer == null or source_end < 64 or cutoff < 64 or cutoff >= edited_length) return false;
    const source = source_pointer[0..source_end];
    const edited = edited_pointer[0..edited_length];
    var end = if (right or cutoff < source_end) cutoff else source_end;
    while (end >= 64) : (end -= 1) {
        if ((source_end ^ end) & 1 != 0 or !std.mem.eql(c_int, edited[end - 64 .. end], source[source_end - 64 .. source_end])) continue;
        var src = source_end - 64;
        var dst = end - 64;
        while (src != 0 and dst != 0 and source[src - 1] == edited[dst - 1]) {
            src -= 1;
            dst -= 1;
        }
        if (src & 1 != 0) {
            src += 1;
            dst += 1;
        }
        const reused = source_end - src;
        if (dst == 0 or (src | dst) & 1 != 0 or (!right and dst > src) or reused < 64 or reused <= plan.reused_tokens) continue;
        plan.* = .{ .source_start = src, .destination_start = dst, .source_end = source_end, .destination_end = end, .reused_tokens = reused, .fresh_tokens = edited_length - end };
    }
    return plan.reused_tokens != 0;
}
