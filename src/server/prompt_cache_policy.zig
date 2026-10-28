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

/// Graft cache credit counts imported tokens, not their evaluated destination
/// position. Trust the live frontier only when all of its tokens match input.
pub fn automaticFrontier(cached: c_int, live: c_int, common: c_int, prompt: c_int) c_int {
    return if (common == live and live <= prompt) @max(cached, live) else cached;
}

pub const BoundaryPolicy = enum { recent, anchors };

/// Choose spaced semantic ends after the evaluated frontier. Anchors keeps the
/// earliest and latest of the same recent set: an exact-prefix restart point
/// and a suffix source, without stopping for all intervening candidates.
pub fn semanticBoundaries(ends: []const c_int, prompt: c_int, evaluated: c_int, spacing: c_int, limit: usize, policy: BoundaryPolicy, out: []c_int) usize {
    const maximum = @min(@min(limit, out.len), 16);
    var selected: [16]c_int = undefined;
    var count: usize = 0;
    var reverse = ends.len;
    var next = prompt;
    while (reverse != 0 and count < maximum) {
        reverse -= 1;
        const position = ends[reverse];
        if (position <= evaluated or position < 64 or position >= prompt or next - position < spacing) continue;
        selected[count] = position;
        count += 1;
        next = position;
    }
    if (policy == .anchors and count > 2) {
        out[0] = selected[count - 1];
        out[1] = selected[0];
        return 2;
    }
    for (0..count) |index| out[index] = selected[count - 1 - index];
    return count;
}

/// Call at the first decoded token that reaches the semantic byte end. Keep a
/// straddling token fresh so the source stays wholly inside the retained block.
pub fn semanticTokenEnd(token_index: usize, semantic_byte_end: usize, token_byte_end: usize) usize {
    return if (semantic_byte_end == token_byte_end) token_index + 1 else token_index;
}

/// An inserted token belongs to a half-open control prefix only when it is
/// strictly before that prefix's end. The caller applies this after graft success.
pub fn controlEndAfterPadding(control_end: c_int, pad_at: c_int) c_int {
    return if (pad_at < control_end) control_end + 1 else control_end;
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

test "semantic capture never rewinds an evaluated graft boundary" {
    const evaluated = automaticFrontier(4096, 12000, 12000, 14000);
    try std.testing.expectEqual(@as(c_int, 12000), evaluated);
    try std.testing.expect(8192 <= evaluated); // Existing C must be skipped.
    try std.testing.expect(13000 > evaluated); // A later end can be captured.
    // An unrelated or longer old graph cannot suppress fresh checkpoints.
    try std.testing.expectEqual(@as(c_int, 4096), automaticFrontier(4096, 12000, 64, 14000));
    try std.testing.expectEqual(@as(c_int, 4096), automaticFrontier(4096, 12000, 12000, 10000));
}

test "semantic anchors retain prefix and suffix from the same recent set" {
    const ends = [_]c_int{ 100, 150, 700, 1700, 1800, 2950 };
    var out: [16]c_int = undefined;
    const recent = semanticBoundaries(&ends, 3000, 0, 500, 4, .recent, &out);
    try std.testing.expectEqualSlices(c_int, &.{ 150, 700, 1800 }, out[0..recent]);
    const anchors = semanticBoundaries(&ends, 3000, 0, 500, 4, .anchors, &out);
    try std.testing.expectEqualSlices(c_int, &.{ 150, 1800 }, out[0..anchors]);
    // Existing limit1/2 behavior stays meaningful and needs no extra anchor.
    const one = semanticBoundaries(&ends, 3000, 0, 500, 1, .anchors, &out);
    try std.testing.expectEqualSlices(c_int, &.{1800}, out[0..one]);
    const two = semanticBoundaries(&ends, 3000, 0, 500, 2, .anchors, &out);
    try std.testing.expectEqualSlices(c_int, &.{ 700, 1800 }, out[0..two]);
}

test "semantic selection respects evaluated frontier, spacing and capacity" {
    const ends = [_]c_int{ 20, 80, 250, 800, 1200, 1600, 1990, 2000 };
    var out: [2]c_int = undefined;
    const count = semanticBoundaries(&ends, 2000, 800, 400, 16, .anchors, &out);
    try std.testing.expectEqualSlices(c_int, &.{ 1200, 1600 }, out[0..count]);
    try std.testing.expectEqual(@as(usize, 0), semanticBoundaries(&ends, 2000, 2000, 400, 4, .anchors, &out));
    try std.testing.expectEqual(@as(usize, 0), semanticBoundaries(&ends, 2000, 0, 400, 0, .anchors, &out));
    try std.testing.expectEqual(@as(usize, 0), semanticBoundaries(&ends, 2000, 0, 400, 4, .anchors, &.{}));
    const recent = semanticBoundaries(&ends, 2000, 0, 400, 16, .recent, &out);
    try std.testing.expectEqualSlices(c_int, &.{ 1200, 1600 }, out[0..recent]);
}

test "semantic end floors a BPE straddler and preserves aligned ends" {
    // Native token4697 is ">\n\n". A result ending after '>' cannot retain
    // that token without also retaining bytes from the following separator.
    try std.testing.expectEqual(@as(usize, 6), semanticTokenEnd(6, 17, 19));
    try std.testing.expectEqual(@as(usize, 7), semanticTokenEnd(6, 19, 19));
    try std.testing.expectEqual(@as(usize, 0), semanticTokenEnd(0, 1, 3));
}

test "successful padding keeps a not-yet-evaluated control out of source admission" {
    const original_end: c_int = 1000 - 63;
    const padded_graft_target: c_int = (1000 + 1) - 64;
    try std.testing.expect(padded_graft_target >= original_end); // Old gate admitted it.
    try std.testing.expect(padded_graft_target < controlEndAfterPadding(original_end, 410));
    try std.testing.expectEqual(original_end, controlEndAfterPadding(original_end, original_end));
    try std.testing.expectEqual(original_end, controlEndAfterPadding(original_end, original_end + 1));
}
