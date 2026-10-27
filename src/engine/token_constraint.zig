//! A borrowed matcher walk. Draft exploration rolls back to the already
//! accepted first token; only tokens committed by the backend survive return.
const std = @import("std");
const c = @import("gguf_c");

pub const Walk = struct {
    callbacks: *const c.ds4_token_constraint,
    advanced: usize = 0,

    /// Preserve raw model logits, including the saved verifier frontiers.
    /// An empty legal finite set is an error, never an argmax fallback to id0.
    pub fn mask(self: *const Walk, logits: []const f32, output: []f32, words: []u32) bool {
        @setFloatMode(.strict);
        if (output.len != logits.len or words.len != (logits.len + 31) / 32) return false;
        @memset(words, 0);
        if (self.callbacks.mask.?(self.callbacks.context, words.ptr, words.len) != 0) return false;
        var maximum = -std.math.inf(f32);
        for (logits, output, 0..) |value, *masked, index| {
            const allowed = words[index / 32] & (@as(u32, 1) << @as(u5, @intCast(index % 32))) != 0;
            masked.* = if (allowed) value else -std.math.inf(f32);
            if (allowed and std.math.isFinite(value)) maximum = @max(maximum, value);
        }
        if (!std.math.isFinite(maximum)) return false;
        // Legacy greedy and probability builders start from a finite -1e30
        // sentinel. Shift this exceptional row so both select legal values;
        // a common offset preserves its ordering and softmax distribution.
        if (maximum <= -1e30) {
            for (output) |*value| if (std.math.isFinite(value.*)) {
                value.* -= maximum;
            };
        }
        return true;
    }

    pub fn accept(self: *Walk, token: c_int) bool {
        if (self.callbacks.accept.?(self.callbacks.context, token) != 0) return false;
        self.advanced += 1;
        return true;
    }

    pub fn rewind(self: *Walk) bool {
        if (self.advanced == 0) return true;
        if (self.callbacks.rollback.?(self.callbacks.context, self.advanced) != 0) return false;
        self.advanced = 0;
        return true;
    }
};

const Toy = struct {
    position: usize = 1, // The caller accepted the first token already.
    mask_error: bool = false,
    empty: bool = false,
    accepts: usize = 0,
    rollbacks: usize = 0,

    fn get(context: ?*anyopaque) *Toy {
        return @ptrCast(@alignCast(context.?));
    }
    fn mask(context: ?*anyopaque, words: [*c]u32, count: usize) callconv(.c) c_int {
        const self = get(context);
        if (self.mask_error or count != 1) return -1;
        words[0] = if (self.empty) 0 else @as(u32, 1) << @intCast(self.position);
        return 0;
    }
    fn accept(context: ?*anyopaque, token: c_int) callconv(.c) c_int {
        const self = get(context);
        if (token != @as(c_int, @intCast(self.position))) return -1;
        self.position += 1;
        self.accepts += 1;
        return 0;
    }
    fn rollback(context: ?*anyopaque, count: usize) callconv(.c) c_int {
        const self = get(context);
        if (count >= self.position) return -1;
        self.position -= count;
        self.rollbacks += count;
        return 0;
    }
    fn callbacks(self: *Toy) c.ds4_token_constraint {
        return .{ .context = self, .mask = mask, .accept = accept, .rollback = rollback };
    }
};

test "draft masks follow tentative tokens and rollback excludes the already accepted seed" {
    const check = std.testing;
    var matcher: Toy = .{};
    const callbacks = matcher.callbacks();
    var walk: Walk = .{ .callbacks = &callbacks };
    const raw = [_]f32{ 100, 2, 3, 4 };
    var output: [4]f32 = undefined;
    var words: [1]u32 = undefined;
    try check.expect(walk.mask(&raw, &output, &words));
    try check.expectEqual(@as(f32, 2), output[1]);
    try check.expectEqual(-std.math.inf(f32), output[0]);
    try check.expect(walk.accept(1));
    try check.expect(walk.mask(&raw, &output, &words));
    try check.expectEqual(@as(f32, 3), output[2]);
    try check.expect(walk.accept(2));
    try check.expect(walk.rewind());
    try check.expectEqual(@as(usize, 1), matcher.position);
    try check.expectEqual(@as(usize, 2), matcher.rollbacks);
    try check.expectEqualSlices(f32, &.{ 100, 2, 3, 4 }, &raw);

    // Target verification accepts one draft, rejects the next, then aborts:
    // only our accepted draft is rolled back, leaving the caller's seed.
    try check.expect(walk.accept(1));
    try check.expect(!walk.accept(3));
    try check.expectEqual(@as(usize, 1), walk.advanced);
    try check.expect(walk.rewind());
    try check.expectEqual(@as(usize, 1), matcher.position);
    try check.expectEqual(@as(usize, 0), walk.advanced);
}

test "empty masks and callback errors cannot select an unpermitted fallback" {
    var matcher: Toy = .{ .empty = true };
    const callbacks = matcher.callbacks();
    const walk: Walk = .{ .callbacks = &callbacks };
    var output: [4]f32 = undefined;
    var words: [1]u32 = undefined;
    try std.testing.expect(!walk.mask(&.{ 100, 2, 3, 4 }, &output, &words));
    matcher.empty = false;
    try std.testing.expect(walk.mask(&.{ 100, -1e35, 3, 4 }, &output, &words));
    try std.testing.expectEqual(@as(f32, 0), output[1]);
    try std.testing.expectEqual(-std.math.inf(f32), output[0]);
    matcher.mask_error = true;
    try std.testing.expect(!walk.mask(&.{ 100, 2, 3, 4 }, &output, &words));
    try std.testing.expectEqual(@as(usize, 0), matcher.accepts);
}
