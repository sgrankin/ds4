//! Request-selected cache boundaries. Positions are exact prompt-token ends;
//! a breakpoint permits prefix reuse, never skipping changed earlier input.
const std = @import("std");

pub fn contains(positions: []const c_int, tokens: c_int) bool {
    if (tokens <= 0) return false;
    for (positions) |position| if (position == tokens) return true;
    return false;
}

/// Keep all explicit positions for reads, but create at most four marked
/// checkpoints per request. Materialization supplies sorted distinct positions.
pub fn writes(positions: []const c_int) []const c_int {
    return positions[positions.len - @min(positions.len, 4) ..];
}

pub fn allowsRead(explicit: bool, positions: []const c_int, tokens: c_int) bool {
    return !explicit or contains(positions, tokens);
}

pub fn allowsWrite(explicit: bool, positions: []const c_int, tokens: c_int) bool {
    return !explicit or contains(writes(positions), tokens);
}

test "explicit caching needs a marked endpoint and retains older lookup marks" {
    const marks = [_]c_int{ 128, 256, 512, 1024, 2048, 4096 };
    try std.testing.expect(!allowsRead(true, &.{}, 128));
    try std.testing.expect(!allowsWrite(true, &.{}, 128));
    try std.testing.expect(allowsRead(true, &marks, 128));
    try std.testing.expect(!allowsWrite(true, &marks, 128));
    try std.testing.expect(allowsWrite(true, &marks, 512));
    try std.testing.expect(!allowsRead(true, &marks, 300));
    try std.testing.expect(!allowsWrite(true, &marks, 300));
    try std.testing.expect(allowsRead(false, &.{}, 300));
    try std.testing.expect(allowsWrite(false, &.{}, 300));
}
