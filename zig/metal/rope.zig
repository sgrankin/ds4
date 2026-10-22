//! Complete ordinary RoPE scheduling, head normalization and V4.1 frequencies.
//! Every shader packet is an actual shared record. The fixed ARC encoder only
//! binds the chosen packet/resources and dispatches the native geometry.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
pub const Command = @import("routed_batch.zig").Command;
pub const View = c.ds4_metal_attention_view;
pub const TailArgs = c.ds4_metal_rope_tail_args;
const Launch = c.ds4_metal_rope_launch;
const Pipeline = c.ds4_metal_rope_pipeline;
const allocator = std.heap.c_allocator;
pub const Tune = struct {
    context: u32,
    inverse: bool = false,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
};
fn narrow(value: u32) i32 {
    return @bitCast(value);
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn report(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
fn plan(kind: u32, pipeline: Pipeline, input: View) Launch {
    return .{
        .kind = kind,
        .pipeline = pipeline.pipeline,
        .args = std.mem.zeroes(c.ds4_metal_rope_packet),
        .input = input,
        .positions = null,
        .position_bytes = 0,
        .position_buffer = false,
        .groups_x = 1,
        .groups_y = 1,
        .threads = 1,
        .shared_bytes = 0,
    };
}
fn encode(command: *const Command, launch: *const Launch) bool {
    return c.ds4_metal_rope_encode(command.token, launch) != 0;
}
fn threads(width: u32, maximum: u64) u64 {
    const lanes: u64 = width / 4;
    var count: u64 = 32;
    while (count < lanes and count < maximum) count *= 2;
    return @max(@min(@min(count, maximum), lanes), 1);
}
pub fn makeArgs(tokens: u32, heads: u32, width: u32, rotated: u32, tune: Tune) TailArgs {
    const row = @as(u64, width) * 4;
    const token = @as(u64, heads) *% row;
    return .{
        .ne00 = width,
        .ne01 = heads,
        .ne02 = tokens,
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = row,
        .nb02 = token,
        .nb03 = @as(u64, tokens) *% token,
        .nb0 = 4,
        .nb1 = row,
        .nb2 = token,
        .nb3 = @as(u64, tokens) *% token,
        .n_dims = narrow(rotated),
        .mode = 0,
        .n_ctx_orig = narrow(tune.context),
        .inverse = @intFromBool(tune.inverse),
        .freq_base = tune.base,
        .freq_scale = tune.scale,
        .ext_factor = tune.ext,
        .attn_factor = tune.attention,
        .beta_fast = tune.fast,
        .beta_slow = tune.slow,
        .src2 = false,
    };
}
/// The caller keeps the borrowed command and buffer alive. Positions really are
/// position+t*step: compressor publication uses its ratio, rather than step1.
pub fn apply(
    command: *const Command,
    input: View,
    args: *const TailArgs,
    tokens: u32,
    heads: u32,
    width: u32,
    position: u32,
    step: u32,
) bool {
    const facts = c.ds4_metal_rope_facts();
    const tail: u32 = if (args.n_dims > 0) @intCast(args.n_dims) else 0;
    const lanes_match = tail <= width and (width - tail) & 31 == 0;
    const pair = facts & c.DS4_ROPE_PAIR_AVAILABLE != 0 and
        !present("DS4_METAL_DISABLE_INPLACE_ROPE_PAIR") and
        args.mode == 0 and
        !args.src2 and
        lanes_match and
        (facts & c.DS4_ROPE_M3 != 0 or (facts & c.DS4_ROPE_M5 != 0 and
            tokens == 1));
    // The grouped256 schedule helps long prefill, but sacrifices the per-head
    // parallelism that short batches and decode need.
    const shared = pair and
        facts & c.DS4_ROPE_SHARED_AVAILABLE != 0 and
        !present("DS4_METAL_DISABLE_SHARED_ROPE_COEFF") and
        tail == 64 and
        heads >= 4 and
        tokens >= 32;
    // Keep long prefill on shared4. Reconstructing affine positions inside its
    // coefficient cohort perturbs YaRN fast-math codegen; decode pairs are exact.
    const affine = pair and
        !shared and
        facts & c.DS4_ROPE_AFFINE_AVAILABLE != 0 and
        !present("DS4_METAL_DISABLE_AFFINE_ROPE_PAIR") and
        tokens == 1 and
        facts & (c.DS4_ROPE_M3 | c.DS4_ROPE_M5) != 0;
    const kind: u32 = if (affine) c.DS4_ROPE_AFFINE else if (shared) c.DS4_ROPE_SHARED else if (pair) c.DS4_ROPE_PAIR else c.DS4_ROPE_REFERENCE;
    const pipeline = c.ds4_metal_rope_pipeline_get(kind);
    if (pipeline.pipeline == null) return false;
    var launch = plan(kind, pipeline, input);
    var stack: [256]i32 = undefined;
    var dynamic: ?[]i32 = null;
    defer if (dynamic) |values| allocator.free(values);
    if (affine) {
        launch.args.affine = .{
            .row_bytes = @as(u64, width) * 4,
            .token_bytes = @as(u64, heads) *% width *% 4,
            .head_dim = narrow(width),
            .n_dims = args.n_dims,
            .n_ctx_orig = args.n_ctx_orig,
            .inverse = args.inverse,
            .pos0 = position,
            .pos_step = step,
            .freq_base = args.freq_base,
            .freq_scale = args.freq_scale,
            .ext_factor = args.ext_factor,
            .attn_factor = args.attn_factor,
            .beta_fast = args.beta_fast,
            .beta_slow = args.beta_slow,
        };
    } else {
        const positions = if (tokens <= stack.len) stack[0..tokens] else allocated: {
            dynamic = allocator.alloc(i32, tokens) catch return report("ds4: failed to allocate Metal RoPE position buffer");
            break :allocated dynamic.?;
        };
        for (positions, 0..) |*value, index| value.* = narrow(position +% @as(u32, @intCast(index)) *% step);
        launch.args.tail = args.*;
        launch.positions = positions.ptr;
        launch.position_bytes = @as(u64, tokens) * 4;
        launch.position_buffer = launch.position_bytes > 4096;
    }
    launch.groups_x = if (shared) @as(u64, (heads +% 3) / 4) else heads;
    launch.groups_y = tokens;
    launch.threads = @max(if (shared) 256 else if (pair) @min(tail, 256) else @min(width, 256), 1);
    return encode(command, &launch);
}
pub fn plain(x: ?*api.Tensor, tokens: u32, heads: u32, width: u32, rotated: u32, position: u32, tune: Tune) bool {
    if (api.init() == 0 or x == null or tokens == 0 or heads == 0 or width == 0 or rotated > width or rotated & 1 != 0) return false;
    if (rotated == 0) return true;
    const buffer = Buffer.tensor(x);
    defer buffer.deinit();
    if (!buffer.fits(@as(u64, tokens) *% heads *% width *% 4)) return report("ds4: Metal RoPE received undersized activation buffer");
    const args = makeArgs(tokens, heads, width, rotated, tune);
    var command: Command = .{ .label = "RoPE tail" };
    defer command.drop();
    return command.acquire() and apply(&command, buffer.view, &args, tokens, heads, width, position, 1) and command.finish();
}
fn normArgs(tokens: u32, heads: u32, width: u32, eps: f32) c.ds4_metal_rms_args {
    const row = @as(u64, width) * 4;
    const plane = row *% heads;
    return .{
        .ne00 = narrow(width),
        .ne00_t = narrow(width / 4),
        .nb1 = row,
        .nb2 = plane,
        .nb3 = plane *% tokens,
        .eps = eps,
        .nef1 = .{ narrow(heads), 1, 1 },
        .nef2 = .{ narrow(tokens), 1, 1 },
        .nef3 = .{ 1, 1, 1 },
        .nbf1 = .{ row, row, row },
        .nbf2 = .{ plane, row, row },
        .nbf3 = .{ plane *% tokens, row, row },
    };
}
pub fn headNorm(x: ?*api.Tensor, tokens: u32, heads: u32, width: u32, eps: f32) bool {
    if (api.init() == 0 or x == null or tokens == 0 or heads == 0 or width == 0 or width & 3 != 0) return false;
    const buffer = Buffer.tensor(x);
    defer buffer.deinit();
    if (!buffer.fits(@as(u64, tokens) *% heads *% width *% 4)) return report("ds4: Metal head RMS norm received undersized activation buffer");
    const pipeline = c.ds4_metal_rope_pipeline_get(c.DS4_ROPE_PLAIN_HEAD);
    if (pipeline.pipeline == null) return false;
    var launch = plan(c.DS4_ROPE_PLAIN_HEAD, pipeline, buffer.view);
    launch.args.norm = normArgs(tokens, heads, width, eps);
    launch.groups_x = heads;
    launch.groups_y = tokens;
    launch.threads = threads(width, pipeline.max_threads);
    launch.shared_bytes = 128;
    var command: Command = .{ .label = "head RMS norm" };
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish();
}
pub fn fused(x: ?*api.Tensor, tokens: u32, heads: u32, width: u32, rotated: u32, position: u32, tune: Tune, eps: f32) bool {
    if (api.init() == 0 or
        x == null or
        tokens == 0 or
        heads == 0 or
        width == 0 or
        width & 3 != 0 or
        rotated > width or
        position > @as(u32, std.math.maxInt(i32)) -% tokens) return false;
    const buffer = Buffer.tensor(x);
    defer buffer.deinit();
    if (!buffer.fits(@as(u64, tokens) *% heads *% width *% 4)) return report("ds4: Metal fused head norm/RoPE received undersized activation buffer");
    const pipeline = c.ds4_metal_rope_pipeline_get(c.DS4_ROPE_FUSED_HEAD);
    if (pipeline.pipeline == null) return false;
    var launch = plan(c.DS4_ROPE_FUSED_HEAD, pipeline, buffer.view);
    launch.args.head = .{
        .n_head = narrow(heads),
        .head_dim = narrow(width),
        .head_dim4 = narrow(width / 4),
        .n_dims = narrow(rotated),
        .n_ctx_orig = narrow(tune.context),
        .pos0 = narrow(position),
        .inverse = @intFromBool(tune.inverse),
        .eps = eps,
        .freq_base = tune.base,
        .freq_scale = tune.scale,
        .ext_factor = tune.ext,
        .attn_factor = tune.attention,
        .beta_fast = tune.fast,
        .beta_slow = tune.slow,
    };
    launch.groups_x = heads;
    launch.groups_y = tokens;
    launch.threads = threads(width, pipeline.max_threads);
    launch.shared_bytes = 128;
    var command: Command = .{ .label = "fused head norm/RoPE" };
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish();
}
pub fn glm(
    x: ?*api.Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    offset: u32,
    position: u32,
    tune: Tune,
    label: [*:0]const u8,
) bool {
    @setFloatMode(.strict);
    if (api.init() == 0 or
        x == null or
        tokens == 0 or
        heads == 0 or
        width == 0 or
        rotated == 0 or
        offset > width or
        rotated > width - offset or
        rotated & 1 != 0 or
        position > std.math.maxInt(u32) - tokens or
        !std.math.isFinite(tune.base) or
        tune.base <= 0 or
        !std.math.isFinite(tune.scale) or
        tune.scale <= 0 or
        !std.math.isFinite(tune.ext) or
        !std.math.isFinite(tune.attention) or
        !std.math.isFinite(tune.fast) or
        !std.math.isFinite(tune.slow)) return false;
    const buffer = Buffer.tensor(x);
    defer buffer.deinit();
    if (!buffer.fits(@as(u64, tokens) *% heads *% width *% 4)) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal %s received undersized buffer\n", label);
        return false;
    }
    const pipeline = c.ds4_metal_rope_pipeline_get(c.DS4_ROPE_GLM);
    if (pipeline.pipeline == null) return false;
    var launch = plan(c.DS4_ROPE_GLM, pipeline, buffer.view);
    launch.args.glm = .{
        .n_tokens = tokens,
        .n_head = heads,
        .head_dim = width,
        .rot_dim = rotated,
        .rot_offset = offset,
        .pos0 = position,
        .n_ctx_orig = tune.context,
        .freq_base = tune.base,
        .freq_scale = tune.scale,
        .ext_factor = tune.ext,
        .attn_factor = tune.attention,
        .beta_fast = tune.fast,
        .beta_slow = tune.slow,
    };
    launch.groups_x = heads;
    launch.groups_y = tokens;
    launch.threads = threads(rotated, 1024);
    var command: Command = .{ .label = label };
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish();
}
// Frequency rounding errors become long-context phase errors. Preserve the
// reference's double low/high and f32 pow/reciprocal/YaRN operation order.
// Only rope_entry reaches v41. Compressor imports makeArgs/apply, whose
// reachable declarations do not instantiate this private lazy state.
var frequencies: [2][32]f32 = @splat(@splat(0));
var frequency_once: c.pthread_once_t = c.ds4_metal_rope_once_init;
fn initializeFrequencies() callconv(.c) void {
    @setFloatMode(.strict);
    for (0..2) |kind| {
        const base: f32 = if (kind != 0) 160000.0 else 10000.0;
        const low: f32 = @floatCast(c.floor(64.0 * c.log(65536.0 / (32.0 * 2.0 * c.M_PI)) / (2.0 * c.log(@as(f64, base)))));
        const high: f32 = @floatCast(c.ceil(64.0 * c.log(65536.0 / (2.0 * c.M_PI)) / (2.0 * c.log(@as(f64, base)))));
        for (0..32) |index| {
            const denominator = c.powf(base, @as(f32, @floatFromInt(index)) / 32.0);
            var value: f32 = 1.0 / denominator;
            if (kind != 0) {
                const ramp = c.fminf(1.0, c.fmaxf(0.0, (@as(f32, @floatFromInt(index)) - low) / (high - low)));
                const smooth: f32 = 1.0 - ramp;
                const interpolate = (value / 16.0) * (1.0 - smooth);
                const extrapolate = value * smooth;
                value = interpolate + extrapolate;
            }
            frequencies[kind][index] = value;
        }
    }
}
pub fn v41(x: ?*api.Tensor, width: u32, heads: u32, rows: u32, start: u32, stride: u32, compressed: bool, inverse: bool) bool {
    @setFloatMode(.strict);
    const count = @as(u64, width) *% heads *% rows;
    if (width < 64 or
        heads == 0 or
        rows == 0 or
        rows > 1048576 or
        stride == 0 or
        @as(u64, start) + @as(u64, rows - 1) * stride >= 1048576 or
        @as(u64, heads) * rows > @as(u64, std.math.maxInt(u64)) / width or
        x == null or
        count > std.math.maxInt(u64) / 4 or
        api.tensorBytes(x) < count * 4) return false;
    if (api.init() == 0) return false;
    _ = c.pthread_once(&frequency_once, initializeFrequencies);
    const pipeline = c.ds4_metal_rope_pipeline_get(c.DS4_ROPE_V41);
    if (pipeline.pipeline == null) return false;
    const buffer = Buffer.tensor(x);
    defer buffer.deinit();
    var launch = plan(c.DS4_ROPE_V41, pipeline, buffer.view);
    launch.args.v41 = .{
        .width = width,
        .heads = heads,
        .rows = rows,
        .start = start,
        .inverse = @intFromBool(inverse),
        .stride = stride,
        .frequencies = frequencies[@intFromBool(compressed)],
    };
    launch.groups_x = heads;
    launch.groups_y = rows;
    launch.threads = 32;
    var command: Command = .{ .label = "V4.1 unit-magnitude RoPE" };
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish();
}
