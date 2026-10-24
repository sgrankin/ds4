//! Model-free ngram adapters remain usable by CPU fixtures and the diagnostic
//! reference without linking a second production C/GPU owner.
const std = @import("std");
const c = @import("gguf_c");
const ngrams = @import("ngrams.zig");

export fn ds4_native_qwen4_ngram_read(optional_model: ?*const c.ds4_model, rows: [*c]const u32, count: usize, output: [*c]f32) bool {
    const model = optional_model orelse {
        c.__error().* = c.EINVAL;
        return false;
    };
    if (model.ngram_tensor == null or (count != 0 and (rows == null or output == null)) or count > std.math.maxInt(usize) / (160 * @sizeOf(f32))) {
        c.__error().* = c.EINVAL;
        return false;
    }
    const width = model.ngram_tensor.*.dim[0];
    if (width == 0 or width > 160) {
        c.__error().* = c.EINVAL;
        return false;
    }
    return ngrams.read(model, if (count == 0) &.{} else rows[0..count], if (count == 0) &.{} else output[0 .. count * @as(usize, @intCast(width))]);
}

export fn ds4_native_qwen4_ple_step(config: *const c.ds4_qwen4_config, token: c_int, previous: *[c.DS4_MAX_PLE_NGRAM]c_int, rows: [*c]u32) void {
    ngrams.step(config, token, previous, rows[0..config.ple.n_heads]);
}
