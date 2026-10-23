//! Rolling compressor state, full/replay prefill and emitted-row publication.
//! Ratio four carries an overlapping previous half; other ratios do not.
const std = @import("std");
const kernels = @import("compressor_ops.zig");
const c = kernels.c;
const api = kernels.api;
const Buffer = kernels.Buffer;
const Tensor = kernels.Tensor;
const View = kernels.View;
const Ops = kernels.Ops;
const Command = kernels.Command;
const Rope = @import("rope.zig");
const at = kernels.at;
const max32 = std.math.maxInt(u32);
pub const Settings = struct {
    map: ?*const anyopaque,
    size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64 = 0,
    norm_type: u32 = 0,
    width: u32,
    ratio: u32,
    pos: u32,
    n_rot: u32 = 0,
    original_context: u32 = 0,
    quantize: bool = false,
    base: f32 = 0,
    scale: f32 = 0,
    ext: f32 = 0,
    attention: f32 = 0,
    fast: f32 = 0,
    slow: f32 = 0,
    eps: f32 = 0,
    fn stateWidth(self: Settings) u32 {
        return (if (self.ratio == 4) @as(u32, 2) else 1) *% self.width;
    }
    fn stateRows(self: Settings) u32 {
        return (if (self.ratio == 4) @as(u32, 2) else 1) *% self.ratio;
    }
    fn apeBytes(self: Settings) u64 {
        return @as(u64, self.stateWidth()) *% self.ratio *% (if (self.ape_type == 1) @as(u64, 2) else 4);
    }
    fn range(self: Settings, offset: u64, bytes: u64) bool {
        return offset <= self.size and bytes <= self.size - offset;
    }
    fn ranges(self: Settings) bool {
        return self.range(self.ape_offset, self.apeBytes()) and self.range(self.norm_offset, @as(u64, self.width) * 4);
    }
    fn valid(self: Settings) bool {
        return self.map != null and self.width != 0 and self.ratio != 0 and self.n_rot <= self.width and self.n_rot & 1 == 0 and self.ape_type <= 1 and self.norm_type == 0;
    }
};
pub const Request = struct {
    cache: ?*Tensor,
    state_kv: ?*Tensor,
    state_score: ?*Tensor,
    kv: ?*const Tensor,
    score: ?*const Tensor,
    settings: Settings,
    count: u32 = 1,
    fn pointers(self: Request, cache: bool) bool {
        return self.kv != null and self.score != null and self.state_kv != null and self.state_score != null and (!cache or self.cache != null);
    }
};
const Buffers = struct {
    kv: Buffer,
    score: Buffer,
    state_kv: Buffer,
    state_score: Buffer,
    cache: Buffer,
    fn init(request: Request) Buffers {
        return .{
            .kv = Buffer.tensor(request.kv),
            .score = Buffer.tensor(request.score),
            .state_kv = Buffer.tensor(request.state_kv),
            .state_score = Buffer.tensor(request.state_score),
            .cache = Buffer.tensor(request.cache),
        };
    }
    fn deinit(self: Buffers) void {
        self.cache.deinit();
        self.state_score.deinit();
        self.state_kv.deinit();
        self.score.deinit();
        self.kv.deinit();
    }
    fn fits(self: Buffers, request: Request, input_count: u32, output_count: u32, check_cache: bool) bool {
        const settings = request.settings;
        const input_bytes = @as(u64, input_count) *% settings.stateWidth() *% 4;
        const state_bytes = @as(u64, settings.stateRows()) *% settings.stateWidth() *% 4;
        const output_bytes = @as(u64, output_count) *% settings.width *% 4;
        return self.kv.fits(input_bytes) and self.score.fits(input_bytes) and self.state_kv.fits(state_bytes) and self.state_score.fits(state_bytes) and (!check_cache or self.cache.fits(output_bytes));
    }
};
const Scope = struct {
    had_batch: bool,
    fn begin() ?Scope {
        const had_batch = c.ds4_gpu_commands_active() != 0;
        if (!had_batch and c.ds4_gpu_begin_commands() == 0) return null;
        return .{ .had_batch = had_batch };
    }
    fn end(self: Scope, command: *Command, ok: bool) bool {
        command.drop();
        if (self.had_batch) return ok;
        const ended = c.ds4_gpu_end_commands() != 0;
        return ended and ok;
    }
};
fn projectRows(ops: *Ops, request: Request, buffers: Buffers, ape: View, start: u32, indices: []const i32) bool {
    const settings = request.settings;
    const byte_offset = @as(u64, start) *% settings.stateWidth() *% 4;
    return ops.projected(at(buffers.kv.view, byte_offset), at(buffers.score.view, byte_offset), buffers.state_kv.view, buffers.state_score.view, ape, settings.ape_type, settings.stateWidth(), settings.ratio, settings.pos +% start, indices, settings.stateRows());
}
fn clearState(ops: *Ops, settings: Settings, buffers: Buffers) bool {
    return ops.fill(buffers.state_kv.view, settings.stateWidth(), settings.stateRows(), 0) and ops.fill(buffers.state_score.view, settings.stateWidth(), settings.stateRows(), kernels.infinity);
}
fn normalize(request: Request, command: *Command, buffers: Buffers, count: u32) bool {
    const settings = request.settings;
    if (c.ds4_gpu_rms_norm_weight_rows_tensor(request.cache, request.cache, settings.map, settings.size, settings.norm_offset, settings.width, count, settings.eps) == 0) return false;
    if (settings.n_rot != 0) {
        const args = Rope.makeArgs(count, 1, settings.width, settings.n_rot, .{
            .context = settings.original_context,
            .base = settings.base,
            .scale = settings.scale,
            .ext = settings.ext,
            .attention = settings.attention,
            .fast = settings.fast,
            .slow = settings.slow,
        });
        if (!command.acquire() or command.owned != 0) return false;
        const borrowed = Rope.Command{ .token = command.token, .owned = command.owned };
        if (!Rope.apply(&borrowed, buffers.cache.view, &args, count, 1, settings.width, settings.pos, settings.ratio)) return false;
    }
    return !settings.quantize or c.ds4_gpu_dsv4_fp8_kv_quantize_tensor(request.cache, count, settings.width, settings.n_rot) != 0;
}
fn storeOne(request: Request) bool {
    const settings = request.settings;
    const width = settings.stateWidth();
    if (!request.pointers(false) or settings.map == null or width == 0 or settings.ratio == 0 or settings.ape_type > 1) return false;
    const pso = c.ds4_metal_compressor_pipeline(c.DS4_COMP_STORE, false);
    if (pso.handle == null) return false;
    if (!settings.range(settings.ape_offset, settings.apeBytes())) return false;
    const buffers = Buffers.init(request);
    defer buffers.deinit();
    if (!buffers.fits(request, 1, 0, false)) return false;
    const ape = kernels.model(settings.map, settings.size, settings.ape_offset, settings.apeBytes()) orelse return false;
    defer ape.deinit();
    var command = Command{};
    defer command.drop();
    if (!command.acquire()) return false;
    var ops = Ops{ .command = &command };
    defer ops.deinit();
    var launch = std.mem.zeroes(c.ds4_metal_compressor_launch);
    launch.kind = c.DS4_COMP_STORE;
    launch.pipeline = pso;
    launch.args.store = .{ .width = width, .ratio = settings.ratio, .pos = settings.pos, .ape_type = settings.ape_type };
    launch.kv = buffers.kv.view;
    launch.score = buffers.score.view;
    launch.ape = ape.view;
    launch.state_kv = buffers.state_kv.view;
    launch.state_score = buffers.state_score.view;
    launch.gx = (@as(u64, width) + 255) / 256;
    launch.gy = 1;
    launch.gz = 1;
    launch.tx = 256;
    launch.ty = 1;
    return ops.encode(&launch) and command.finish("compressor one-row store");
}
pub fn store(request: Request) c_int {
    if (api.init() == 0) return 0;
    const settings = request.settings;
    if (!request.pointers(false) or settings.map == null or settings.width == 0 or settings.ratio == 0 or request.count == 0 or settings.ape_type > 1) return 0;
    if (!settings.range(settings.ape_offset, settings.apeBytes())) return @intFromBool(kernels.errorLine("ds4: Metal compressor batch APE range is outside the mapped model"));
    const buffers = Buffers.init(request);
    defer buffers.deinit();
    if (!buffers.fits(request, request.count, 0, false)) return @intFromBool(kernels.errorLine("ds4: Metal compressor batch store received undersized buffers"));
    const ape = kernels.model(settings.map, settings.size, settings.ape_offset, settings.apeBytes()) orelse return 0;
    defer ape.deinit();
    const total = @as(u64, request.count) * settings.stateWidth();
    if (total > max32 or settings.stateRows() > std.math.maxInt(i32)) return @intFromBool(kernels.errorLine("ds4: Metal compressor batch store received too many elements"));
    var command = Command{};
    defer command.drop();
    var ops = Ops{ .command = &command };
    defer ops.deinit();
    if (!ops.reserve(c.DS4_COMP_APE, total * 4) or !ops.reserve(c.DS4_COMP_ADDED_SCORE, total * 4)) return 0;
    var stack: [16]i32 = undefined;
    var allocated: ?[]i32 = null;
    defer if (allocated) |memory| std.heap.c_allocator.free(memory);
    const rows = if (request.count <= stack.len) stack[0..request.count] else blk: {
        const memory = std.heap.c_allocator.alloc(i32, request.count) catch return @intFromBool(kernels.errorLine("ds4: failed to allocate compressor set_rows index list"));
        allocated = memory;
        break :blk memory;
    };
    for (rows, 0..) |*row, index| {
        const remainder = (settings.pos +% @as(u32, @intCast(index))) % settings.ratio;
        row.* = kernels.narrow(if (settings.ratio == 4) settings.ratio +% remainder else remainder);
    }
    if (!command.acquire()) return 0;
    // Store-batch preserves the explicit tile + add graph; the projected-state
    // helper's M3 fused APE add is intentionally not used on this API path.
    if (!ops.tileApe(ape.view, ops.view(c.DS4_COMP_APE), settings.ape_type == 1, settings.stateWidth(), settings.ratio, settings.pos, request.count) or
        !ops.add(buffers.score.view, ops.view(c.DS4_COMP_APE), ops.view(c.DS4_COMP_ADDED_SCORE), @intCast(total)) or
        !ops.setRows(buffers.kv.view, buffers.state_kv.view, rows, settings.stateRows(), settings.stateWidth()) or
        !ops.setRows(ops.view(c.DS4_COMP_ADDED_SCORE), buffers.state_score.view, rows, settings.stateRows(), settings.stateWidth())) return 0;
    return @intFromBool(command.finish("compressor batch DS4 store"));
}
fn packRows(ops: *Ops, request: Request, buffers: Buffers, count: u32, replay: bool) bool {
    const settings = request.settings;
    const width = settings.width;
    const source_row = @as(u64, settings.stateWidth()) * 4;
    const source_plane = @as(u64, 4) * source_row;
    const target_row = @as(u64, width) * 4;
    const target_plane = 8 * target_row;
    const score = ops.view(c.DS4_COMP_ADDED_SCORE);
    const direct_output_bytes = @as(u64, count) *% width *% 4;
    const direct_input_bytes = @as(u64, count) *% 4 *% settings.stateWidth() *% 4;
    const previous_bytes = @as(u64, 4) * settings.stateWidth() * 4;
    const direct = Ops.directMode(width, count) and !kernels.overlap(buffers.cache.view, direct_output_bytes, buffers.kv.view, direct_input_bytes) and
        (!replay or (!kernels.overlap(buffers.cache.view, direct_output_bytes, buffers.state_kv.view, previous_bytes) and !kernels.overlap(buffers.cache.view, direct_output_bytes, buffers.state_score.view, previous_bytes)));
    if (direct) return ops.packRatio4(buffers.kv.view, score, buffers.state_kv.view, buffers.state_score.view, width, count, replay, buffers.cache.view);
    const bytes = @as(u64, count) *% 8 *% width *% 4;
    if (!ops.reserve(c.DS4_COMP_KV, bytes) or !ops.reserve(c.DS4_COMP_SCORE, bytes)) return false;
    if (Ops.packMode(width)) {
        if (!ops.packRatio4(buffers.kv.view, score, buffers.state_kv.view, buffers.state_score.view, width, count, replay, null)) return false;
    } else {
        if (!ops.fill(ops.view(c.DS4_COMP_KV), width, 8 *% count, 0) or !ops.fill(ops.view(c.DS4_COMP_SCORE), width, 8 *% count, kernels.infinity)) return false;
        if (replay and (!ops.copy(buffers.state_kv.view, ops.view(c.DS4_COMP_KV), width, 4, 1, 4, source_row, source_plane, target_row, target_plane) or
            !ops.copy(buffers.state_score.view, ops.view(c.DS4_COMP_SCORE), width, 4, 1, 4, source_row, source_plane, target_row, target_plane))) return false;
        if (!ops.copy(at(buffers.kv.view, target_row), at(ops.view(c.DS4_COMP_KV), 4 * target_row), width, 4, count, 4, source_row, source_plane, target_row, target_plane) or
            !ops.copy(at(score, target_row), at(ops.view(c.DS4_COMP_SCORE), 4 * target_row), width, 4, count, 4, source_row, source_plane, target_row, target_plane)) return false;
        if (count > 1 and (!ops.copy(buffers.kv.view, at(ops.view(c.DS4_COMP_KV), target_plane), width, 4, count - 1, 4, source_row, source_plane, target_row, target_plane) or
            !ops.copy(score, at(ops.view(c.DS4_COMP_SCORE), target_plane), width, 4, count - 1, 4, source_row, source_plane, target_row, target_plane))) return false;
    }
    return ops.pool(buffers.cache.view, ops.view(c.DS4_COMP_KV), ops.view(c.DS4_COMP_SCORE), .{ target_row, 4, target_plane }, .{ target_row, 4, target_plane }, 8, width, count);
}
fn runPrefill(request: Request, replay: bool) bool {
    const settings = request.settings;
    const count = request.count / settings.ratio;
    const cutoff = count * settings.ratio;
    const remainder = request.count - cutoff;
    const buffers = Buffers.init(request);
    defer buffers.deinit();
    if (!buffers.fits(request, request.count, count, count != 0) or !buffers.cache.valid()) return kernels.errorLine(if (replay) "ds4: Metal compressor replay received undersized buffers" else "ds4: Metal compressor prefill received undersized buffers");
    const ape = kernels.model(settings.map, settings.size, settings.ape_offset, settings.apeBytes()) orelse return false;
    defer ape.deinit();
    const scope = Scope.begin() orelse return false;
    var command = Command{};
    defer command.drop();
    var ops = Ops{ .command = &command };
    defer ops.deinit();
    const result = blk: {
        if (!command.acquire() or command.owned != 0) break :blk false;
        if (!replay) {
            if (!clearState(&ops, settings, buffers)) break :blk false;
            if (settings.ratio == 4) {
                const previous = [_]i32{ 0, 1, 2, 3 };
                if (cutoff >= 4 and !projectRows(&ops, request, buffers, ape.view, cutoff - 4, &previous)) break :blk false;
                if (remainder != 0) {
                    var rows: [4]i32 = undefined;
                    for (rows[0..remainder], 0..) |*row, index| row.* = @intCast(4 + index);
                    if (!projectRows(&ops, request, buffers, ape.view, cutoff, rows[0..remainder])) break :blk false;
                }
            } else if (remainder != 0) {
                var rows: [128]i32 = undefined;
                if (remainder > rows.len) break :blk kernels.errorLine("ds4: Metal compressor prefill remainder exceeds local row list");
                for (rows[0..remainder], 0..) |*row, index| row.* = @intCast(index);
                if (!projectRows(&ops, request, buffers, ape.view, cutoff, rows[0..remainder])) break :blk false;
            }
        }
        if (count != 0) {
            if (!ops.reserve(c.DS4_COMP_ADDED_SCORE, @as(u64, cutoff) *% settings.stateWidth() *% 4) or
                !ops.addApe(buffers.score.view, ops.view(c.DS4_COMP_ADDED_SCORE), ape.view, settings.ape_type, settings.stateWidth(), settings.ratio, settings.pos, cutoff)) break :blk false;
            if (settings.ratio == 4) {
                if (!packRows(&ops, request, buffers, count, replay)) break :blk false;
            } else {
                const row_stride = @as(u64, settings.stateWidth()) * 4;
                const stride = [3]u64{ row_stride, 4, @as(u64, settings.ratio) *% row_stride };
                if (!ops.pool(buffers.cache.view, buffers.kv.view, ops.view(c.DS4_COMP_ADDED_SCORE), stride, stride, settings.ratio, settings.width, count)) break :blk false;
            }
            if (!normalize(request, &command, buffers, count)) break :blk false;
        }
        if (replay) {
            // An aligned nonzero ubatch seeds its first compressed row with
            // the previous half, then publishes the last full input group as
            // rows 0..3 for the next call. This update follows finalization.
            if (!clearState(&ops, settings, buffers)) break :blk false;
            const previous = [_]i32{ 0, 1, 2, 3 };
            if (!projectRows(&ops, request, buffers, ape.view, request.count - 4, &previous)) break :blk false;
        }
        break :blk true;
    };
    return scope.end(&command, result);
}
pub fn prefill(request: Request, replay: bool) c_int {
    if (api.init() == 0) return 0;
    const settings = request.settings;
    if (!request.pointers(true) or !settings.valid() or request.count == 0 or (replay and (settings.ratio != 4 or request.count & 3 != 0 or settings.pos & 3 != 0))) return 0;
    if (!settings.ranges()) return @intFromBool(kernels.errorLine(if (replay) "ds4: Metal compressor replay tensor range is outside the mapped model" else "ds4: Metal compressor prefill tensor range is outside the mapped model"));
    return @intFromBool(runPrefill(request, replay));
}
pub fn statePrefill(request: Request) c_int {
    if (api.init() == 0) return 0;
    const settings = request.settings;
    if (!request.pointers(false) or settings.map == null or settings.width == 0 or settings.ape_type > 1) return 0;
    if (!settings.range(settings.ape_offset, settings.apeBytes())) return @intFromBool(kernels.errorLine("ds4: Metal compressor prefill-state APE range is outside the mapped model"));
    const buffers = Buffers.init(request);
    defer buffers.deinit();
    if (!buffers.fits(request, 4, 0, false)) return @intFromBool(kernels.errorLine("ds4: Metal compressor prefill-state received undersized buffers"));
    const ape = kernels.model(settings.map, settings.size, settings.ape_offset, settings.apeBytes()) orelse return 0;
    defer ape.deinit();
    const scope = Scope.begin() orelse return 0;
    var command = Command{};
    defer command.drop();
    var ops = Ops{ .command = &command };
    defer ops.deinit();
    const previous = [_]i32{ 0, 1, 2, 3 };
    const ok = command.acquire() and command.owned == 0 and clearState(&ops, settings, buffers) and projectRows(&ops, request, buffers, ape.view, 0, &previous);
    return @intFromBool(scope.end(&command, ok));
}
pub fn update(request: Request, comp_row: u32, already_stored: bool, decode: bool, defer_finalize: bool) c_int {
    if (api.init() == 0) return 0;
    const settings = request.settings;
    if (!request.pointers(true) or !settings.valid()) return 0;
    const emit = (settings.pos +% 1) % settings.ratio == 0;
    if (!settings.ranges()) return @intFromBool(kernels.errorLine("ds4: Metal compressor tensor range is outside the mapped model"));
    const buffers = Buffers.init(request);
    defer buffers.deinit();
    // C checks state byte extents here but defers backing-handle admission to
    // the store/pool operation (already_stored/non-emit is a valid no-op).
    const input_bytes = @as(u64, settings.stateWidth()) * 4;
    const state_bytes = @as(u64, settings.stateRows()) *% settings.stateWidth() *% 4;
    const output_bytes = @as(u64, comp_row +% @as(u32, @intFromBool(emit))) *% settings.width *% 4;
    if (!buffers.kv.fits(input_bytes) or !buffers.score.fits(input_bytes) or !buffers.cache.valid() or api.tensorBytes(request.state_kv) < state_bytes or api.tensorBytes(request.state_score) < state_bytes or (emit and buffers.cache.view.bytes < output_bytes)) return @intFromBool(kernels.errorLine("ds4: Metal compressor update received undersized buffers"));
    if (!already_stored) {
        const stored = if (!kernels.present("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE")) storeOne(request) else store(request) != 0;
        if (!stored) return 0;
    }
    if (!emit) return 1;
    const output = c.ds4_gpu_tensor_view(request.cache, @as(u64, comp_row) *% settings.width *% 4, @as(u64, settings.width) * 4) orelse return 0;
    defer c.ds4_gpu_tensor_free(output);
    const row = Buffer.tensor(output);
    defer row.deinit();
    var command = Command{};
    defer command.drop();
    var ops = Ops{ .command = &command };
    defer ops.deinit();
    if (!command.acquire() or !ops.statePool(row.view, buffers.state_kv.view, buffers.state_score.view, settings.width, settings.ratio, decode) or !command.finish("compressor DS4 softmax pool")) return 0;
    // Fused-finalize consumers need the un-normalized pooled row. State shift
    // is likewise left to that fused dispatch, exactly once.
    if (!defer_finalize) {
        if (c.ds4_gpu_rms_norm_weight_rows_tensor(output, output, settings.map, settings.size, settings.norm_offset, settings.width, 1, settings.eps) == 0 or
            c.ds4_gpu_rope_tail_tensor(output, 1, 1, settings.width, settings.n_rot, settings.pos +% 1 -% settings.ratio, settings.original_context, false, settings.base, settings.scale, settings.ext, settings.attention, settings.fast, settings.slow) == 0) return 0;
        if (settings.ratio == 4 and (!command.acquire() or !ops.shift(buffers.state_kv.view, buffers.state_score.view, settings.stateWidth()) or !command.finish("compressor ratio4 state shift"))) return 0;
    }
    return 1;
}
