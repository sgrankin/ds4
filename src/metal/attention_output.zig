//! Grouped attention-output projections and compact K-window expansion.
//! Numerical kernels are unchanged. Native owns admission, packets, per-row ID
//! layout, MPP/map fallback and exact BF16/projection/publication boundaries.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("routed_batch.zig").Command;
const Tensor = api.Tensor;
const View = c.ds4_metal_attention_view;
const Pipeline = c.ds4_metal_hc_pipeline;
const Mv = c.ds4_gpu_mul_mv_id_args;
const Mm = c.ds4_gpu_mul_mm_id_args;
const Map = c.ds4_gpu_mul_mm_id_map_args;
const Launch = c.ds4_metal_output_launch;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
const empty_view = std.mem.zeroes(View);
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn errorLine(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
fn narrow(value: u64) i32 {
    return @bitCast(@as(u32, @truncate(value)));
}
fn shifted(view: View, offset: u64) View {
    var result = view;
    result.offset +%= offset;
    return result;
}
fn mpp() bool {
    return c.ds4_metal_dense_facts(false) & c.DS4_DENSE_MPP != 0;
}
fn pipeline(name: [*:0]const u8, kind: u32, nsg: i16) ?*const Pipeline {
    return c.ds4_metal_routed_pipeline(kind, name, nsg);
}
fn weight(map: ?*const anyopaque, size: u64, offset: u64, count: u64) ?Buffer {
    var inner: u64 = 0;
    const retained = api.retainModelRange(map, size, offset, count, &inner) orelse return null;
    return .{ .view = .{ .buffer = retained, .offset = inner, .bytes = count } };
}
fn encode(command: *const Command, plan: Launch) bool {
    return c.ds4_metal_output_encode(command.token.?, &plan) != 0;
}
fn launch(kind: u32, pso: *const Pipeline, args: c.ds4_metal_output_args, w: View, x: View, out: View) Launch {
    return .{
        .kind = kind,
        .args = args,
        .pipeline = pso,
        .weights = w,
        .input = x,
        .output = out,
        .ids = empty_view,
        .workmap = empty_view,
        .work_ids_offset = 0,
        .work_list_offset = 0,
        .groups_x = 1,
        .groups_y = 1,
        .groups_z = 1,
        .threads_x = 32,
        .threads_y = 1,
        .shared_bytes = 0,
        .hint_weights = false,
        .fold = false,
        .fold_slot = 0,
        .fold_value = 0,
    };
}
fn lowArgs(width: u64, rank: u64, groups: u32, rows: u32, row_bytes: u64, q8: bool, ids: bool) Mv {
    return .{
        .nei0 = narrow(groups),
        .nei1 = narrow(rows),
        .nbi1 = if (ids) @as(u64, groups) * 4 else 0,
        .ne00 = narrow(width),
        .ne01 = narrow(rank),
        .ne02 = narrow(groups),
        .nb00 = if (q8) 34 else 1,
        .nb01 = row_bytes,
        .nb02 = rank *% row_bytes,
        .ne10 = narrow(width),
        .ne11 = narrow(groups),
        .ne12 = narrow(rows),
        .ne13 = 1,
        .nb10 = 4,
        .nb11 = width * 4,
        .nb12 = @as(u64, groups) *% width *% 4,
        .ne0 = narrow(rank),
        .ne1 = narrow(groups),
        .nb1 = rank * 4,
        .nr0 = 2,
        .tp_rank = 0,
        .tp_world = 0,
        .tp_expert_base = 0,
        .tp_addend = 0,
    };
}
fn mmArgs(width: u64, rank: u64, groups: u32, rows: u32, row_bytes: u64) Mm {
    const expert_bytes = rank *% row_bytes;
    const input_row = width * 4;
    return .{
        .ne00 = narrow(width),
        .ne02 = narrow(groups),
        .nb01 = row_bytes,
        .nb02 = expert_bytes,
        .nb03 = @as(u64, groups) *% expert_bytes,
        .ne11 = narrow(groups),
        .nb10 = 4,
        .nb11 = input_row,
        .nb12 = @as(u64, groups) *% input_row,
        .nb13 = @as(u64, rows) *% groups *% input_row,
        .ne20 = narrow(groups),
        .ne21 = narrow(rows),
        .ne0 = narrow(rank),
        .ne1 = narrow(groups),
        .r2 = 1,
        .r3 = 1,
        .tp_rank = 0,
        .tp_world = 0,
        .tp_expert_base = 0,
    };
}
fn mapArgs(width: u64, groups: u32, rows: u32) Map {
    return .{
        .ne02 = narrow(groups),
        .ne10 = narrow(width),
        .ne11 = narrow(groups),
        .nb11 = width * 4,
        .nb12 = @as(u64, groups) *% width *% 4,
        .ne21 = narrow(rows),
        .ne20 = narrow(groups),
        .nb21 = @as(u64, groups) * 4,
    };
}
fn mapName(groups: u32) ?[*:0]const u8 {
    return switch (groups) {
        1 => "kernel_mul_mm_id_map0_ne20_1",
        2 => "kernel_mul_mm_id_map0_ne20_2",
        4 => "kernel_mul_mm_id_map0_ne20_4",
        5 => "kernel_mul_mm_id_map0_ne20_5",
        6 => "kernel_mul_mm_id_map0_ne20_6",
        8 => "kernel_mul_mm_id_map0_ne20_8",
        10 => "kernel_mul_mm_id_map0_ne20_10",
        16 => "kernel_mul_mm_id_map0_ne20_16",
        22 => "kernel_mul_mm_id_map0_ne20_22",
        else => null,
    };
}
const Inputs = struct {
    heads: Buffer,
    low: Buffer,
    fn init(heads: ?*const Tensor, low: ?*Tensor) Inputs {
        return .{ .heads = Buffer.tensor(heads), .low = Buffer.tensor(low) };
    }
    fn deinit(self: Inputs) void {
        self.low.deinit();
        self.heads.deinit();
    }
};
const Scope = struct {
    had_batch: bool,
    command: Command = .{ .label = "attention output" },
    fn init() ?Scope {
        const had = c.ds4_gpu_commands_active() != 0;
        if (!had and c.ds4_gpu_begin_commands() == 0) return null;
        return .{ .had_batch = had };
    }
    fn acquire(self: *Scope) bool {
        return self.command.acquire() and self.command.owned == 0;
    }
    fn deinit(self: *Scope) void {
        self.command.drop();
    }
    fn end(self: *Scope, ok: bool) bool {
        self.command.drop();
        return if (self.had_batch) ok else c.ds4_gpu_end_commands() != 0 and ok;
    }
    fn restart(self: *Scope) bool {
        self.command.drop();
        return c.ds4_gpu_end_commands() != 0 and c.ds4_gpu_begin_commands() != 0 and self.acquire();
    }
};
const Profile = struct {
    enabled: bool,
    rows: u32,
    started: f64,
    fn init(scope: *Scope, rows: u32, ok: bool) ?Profile {
        const enabled = present("DS4_METAL_ATTN_OUT_STAGE_PROFILE") and c.ds4_gpu_commands_active() != 0;
        if (enabled and ok and !scope.restart()) return null;
        return .{ .enabled = enabled, .rows = rows, .started = if (enabled) c.ds4_metal_dense_now_ms() else 0 };
    }
    fn boundary(self: *Profile, scope: *Scope, name: [*:0]const u8, ok: bool) bool {
        @setFloatMode(.optimized);
        if (!ok or !self.enabled) return ok;
        scope.command.drop();
        if (c.ds4_gpu_end_commands() == 0) return false;
        const now = c.ds4_metal_dense_now_ms();
        _ = c.fprintf(c.__stderrp, "ds4: Metal attention output stage tokens=%u %s=%.3f ms\n", self.rows, name, now - self.started);
        self.started = now;
        return c.ds4_gpu_begin_commands() != 0 and scope.acquire();
    }
};
fn idsBuffer(rows: u32, groups: u32, q4: bool) ?Buffer {
    const bytes = @as(u64, rows) *% groups *% 4;
    const view = c.ds4_metal_output_scratch(c.DS4_OUT_IDS_SCRATCH, bytes, present("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE"), if (q4) "attention output Q4 group ids" else "attention output group ids");
    if (view.buffer == null) return null;
    const ids: [*]i32 = @ptrCast(@alignCast(c.ds4_metal_attention_contents(view).?));
    for (0..rows) |row| for (0..groups) |group| {
        ids[row * groups + group] = narrow(group);
    };
    return .{ .view = view };
}
fn direct(command: *const Command, args: Mv, w: View, input: Inputs, q8: bool, ids: View) bool {
    if (args.ne00 <= 0 or args.ne01 <= 0 or args.nei0 <= 0 or args.nei1 <= 0 or w.buffer == null or !input.heads.valid() or !input.low.valid()) return false;
    const nsg: u64 = if (q8) 4 else 2;
    const program = pipeline(if (ids.buffer != null) "kernel_mul_mv_id_q8_0_f32" else if (q8) "kernel_dsv4_attn_out_low_q8_0_f32" else "kernel_dsv4_attn_out_low_q4_K_f32", 1, @intCast(nsg)) orelse return false;
    var plan = launch(if (ids.buffer != null) c.DS4_OUT_LOW_IDS else c.DS4_OUT_LOW_DIRECT, program, .{ .low = args }, w, input.heads.view, input.low.view);
    // Q8 k-splits a nr0-row group across SIMD groups; Q4 gives each group
    // its own nr0 rows. Using Q4's stride for Q8 leaves output rows unwritten.
    const row_group = @as(u64, @intCast(args.nr0)) * (if (q8) @as(u64, 1) else nsg);
    plan.groups_x = (@as(u64, @intCast(args.ne01)) + row_group - 1) / row_group;
    plan.groups_z = @as(u64, @intCast(args.nei0)) * @as(u64, @intCast(args.nei1));
    plan.threads_y = nsg;
    plan.shared_bytes = if (q8) 256 else 32;
    plan.ids = ids;
    return encode(command, plan);
}
fn accelerated(command: *const Command, args: Mm, w: View, input: Inputs, program: ?*const Pipeline) bool {
    if (args.ne00 <= 0 or args.ne0 <= 0 or args.ne02 <= 0 or args.ne1 <= 0 or args.ne21 <= 0 or w.buffer == null or !input.heads.valid() or !input.low.valid()) return false;
    var plan = launch(c.DS4_OUT_MPP, program orelse return false, .{ .grouped = args }, w, input.heads.view, input.low.view);
    plan.groups_x = (@as(u64, @intCast(args.ne21)) + 63) / 64;
    plan.groups_y = (@as(u64, @intCast(args.ne0)) + 63) / 64;
    plan.groups_z = @intCast(args.ne02);
    plan.threads_x = 128;
    plan.shared_bytes = 8192;
    return encode(command, plan);
}
fn mapped(command: *const Command, args: Mm, map: Map, w: View, input: Inputs, ids: View, name: ?[*:0]const u8, q8: bool) bool {
    if (args.ne00 <= 0 or args.ne0 <= 0 or args.ne20 <= 0 or args.ne21 <= 0 or args.ne02 <= 0 or w.buffer == null or !input.heads.valid() or !input.low.valid() or ids.buffer == null) return false;
    const map_program = pipeline(name orelse return false, 0, 0) orelse return false;
    const matrix_program = pipeline(if (q8) "kernel_mul_mm_id_q8_0_f32" else "kernel_mul_mm_id_q4_K_f32", 2, 0) orelse return false;
    const groups: u64 = @intCast(args.ne02);
    const rows: u64 = @intCast(args.ne21);
    const used: u64 = @intCast(args.ne20);
    const tpe = groups * 4;
    const hids = groups *% rows *% 4;
    if (tpe > max64 - hids) return false;
    const work_offset = (tpe +% hids +% 7) & ~@as(u64, 7);
    const pairs = used * rows;
    const capacity = (pairs + 31 * groups + 31) / 32;
    if (capacity > (max64 - 8) / 8 or work_offset > max64 - 8 - capacity * 8) return false;
    const work = c.ds4_metal_output_scratch(c.DS4_OUT_WORKMAP_SCRATCH, work_offset + 8 + capacity * 8, false, "ds4_moe_id_map");
    if (work.buffer == null) return false;
    defer api.releaseModelBuffer(work.buffer.?);
    var plan = launch(c.DS4_OUT_MAP, map_program, .{ .map = map }, w, input.heads.view, input.low.view);
    plan.ids = ids;
    plan.workmap = work;
    plan.work_ids_offset = tpe;
    plan.work_list_offset = work_offset;
    plan.threads_x = groups;
    plan.shared_bytes = groups * used * 2;
    if (!encode(command, plan)) return false;
    plan = launch(c.DS4_OUT_MAPPED, matrix_program, .{ .grouped = args }, w, input.heads.view, input.low.view);
    plan.workmap = work;
    plan.work_ids_offset = tpe;
    plan.work_list_offset = work_offset;
    plan.groups_x = capacity;
    plan.groups_y = (@as(u64, @intCast(args.ne0)) + 63) / 64;
    plan.threads_x = 128;
    plan.shared_bytes = 8192;
    plan.hint_weights = present("DS4_METAL_MOE_MM_ID_USE_RESOURCES") and !present("DS4_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES");
    return encode(command, plan);
}

pub const Projection = struct {
    out: ?*Tensor,
    low: ?*Tensor,
    heads: ?*const Tensor,
    model: ?*const anyopaque,
    size: u64,
    a: u64,
    b: u64,
    width: u64,
    rank: u64,
    groups: u32,
    outputs: u64,
    rows: u32 = 1,
    round_low: bool = false,
    full_low: u64 = 0,
    low_offset: u64 = 0,
    group_tmp: ?*Tensor = null,
    low_tmp: ?*Tensor = null,

    fn valid(self: Projection, legacy_scratch: bool) bool {
        return self.out != null and self.low != null and self.heads != null and self.model != null and self.width != 0 and self.rank != 0 and self.groups != 0 and self.outputs != 0 and self.rows != 0 and self.width <= max32 and self.rank <= max32 and self.outputs <= max32 and (!legacy_scratch or (self.group_tmp != null and self.low_tmp != null));
    }
    pub fn batch(self: Projection) bool {
        if (api.init() == 0 or !self.valid(true)) return false;
        const low_dim = @as(u64, self.groups) *% self.rank;
        const full = if (self.full_low != 0) self.full_low else low_dim;
        if (self.width % 32 != 0 or low_dim % 32 != 0 or low_dim > max32 or full > max32 or full % 32 != 0 or self.low_offset % 32 != 0 or self.low_offset > full or low_dim > full - self.low_offset) return errorLine("ds4: Metal attention output batch received invalid q8 dimensions");
        const a_row = self.width / 32 * 34;
        const b_row = full / 32 * 34;
        const a_bytes = @as(u64, self.groups) *% self.rank *% a_row;
        const b_bytes = self.outputs *% b_row;
        if (self.a > self.size or a_bytes > self.size - self.a or self.b > self.size or b_bytes > self.size - self.b) return errorLine("ds4: Metal attention output batch weights are outside the mapped model");
        if (api.tensorBytes(self.heads) < @as(u64, self.rows) *% self.groups *% self.width *% 4 or api.tensorBytes(self.low) < @as(u64, self.rows) *% low_dim *% 4 or api.tensorBytes(self.out) < @as(u64, self.rows) *% self.outputs *% 4) return errorLine("ds4: Metal attention output batch received undersized buffers");
        const use_direct = self.rows < 32 and !present("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT");
        const use_mpp = self.rows >= 32 and self.rows % c.DS4_METAL_ATTN_OUT_MPP_TILE_N == 0 and mpp();
        var ids: ?Buffer = null;
        defer if (ids) |buffer| buffer.deinit();
        if (!use_direct and !use_mpp) ids = idsBuffer(self.rows, self.groups, false) orelse return false;
        const w = weight(self.model, self.size, self.a, a_bytes) orelse return false;
        defer w.deinit();
        const input = Inputs.init(self.heads, self.low);
        defer input.deinit();
        var scope = Scope.init() orelse return false;
        defer scope.deinit();
        var ok = scope.acquire();
        var profile = Profile.init(&scope, self.rows, ok) orelse return scope.end(false);
        const matrix = mmArgs(self.width, self.rank, self.groups, self.rows, a_row);
        if (ok) {
            if (use_mpp) {
                ok = accelerated(&scope.command, matrix, w.view, input, pipeline("kernel_attn_out_low_q8_0_mpp_direct_rhs_n64", 2, 0));
                if (!ok) {
                    c.ds4_metal_dense_warn_mpp();
                    if (mapName(self.groups) != null) {
                        ids = idsBuffer(self.rows, self.groups, false);
                        if (ids) |buffer| ok = mapped(&scope.command, matrix, mapArgs(self.width, self.groups, self.rows), w.view, input, buffer.view, mapName(self.groups), true);
                    }
                }
            } else if (self.rows >= 32 and mapName(self.groups) != null) ok = mapped(&scope.command, matrix, mapArgs(self.width, self.groups, self.rows), w.view, input, ids.?.view, mapName(self.groups), true) else ok = direct(&scope.command, lowArgs(self.width, self.rank, self.groups, self.rows, a_row, true, !use_direct), w.view, input, true, if (use_direct) empty_view else ids.?.view);
        }
        ok = profile.boundary(&scope, "low_proj", ok);
        if (ok and self.round_low) ok = c.ds4_gpu_dsv41_quantize(self.low, @intCast(low_dim), self.rows, c.DS4_V41_BF16) != 0;
        if (ok and full != low_dim) {
            const matrix_rows = if (mpp()) self.rows / 32 * 32 else 0;
            if (matrix_rows != 0) ok = (KSlice{ .out = self.out, .model = self.model, .size = self.size, .offset = self.b, .width = full, .outputs = self.outputs, .first = self.low_offset, .count = low_dim, .input = self.low }).rows(matrix_rows, low_dim, 0);
            var row = matrix_rows;
            while (ok and row < self.rows) : (row += 1) {
                const target = c.ds4_gpu_tensor_view(self.out, @as(u64, row) *% self.outputs *% 4, self.outputs * 4);
                ok = target != null and (KSlice{ .out = target, .model = self.model, .size = self.size, .offset = self.b, .width = full, .outputs = self.outputs, .first = self.low_offset, .count = low_dim, .input = self.low, .input_offset = @as(u64, row) * low_dim }).scalar();
                c.ds4_gpu_tensor_free(target);
            }
        } else if (ok and self.round_low and self.rows <= 8) ok = c.ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(self.out, self.model, self.size, self.b, low_dim, self.outputs, self.low, self.rows) != 0 else if (ok) ok = c.ds4_gpu_matmul_q8_0_tensor(self.out, self.model, self.size, self.b, low_dim, self.outputs, self.low, self.rows) != 0;
        ok = profile.boundary(&scope, "out_proj", ok);
        return scope.end(ok);
    }
    pub fn q4Batch(self: Projection, output_type: u32) bool {
        if (api.init() == 0 or !self.valid(false) or self.rows < 32) return false;
        const low_dim = @as(u64, self.groups) *% self.rank;
        if (self.width % 256 != 0 or low_dim % 256 != 0 or low_dim > max32) return false;
        const a_row = self.width / 256 * 144;
        const b_row = switch (output_type) {
            c.DS4_METAL_TENSOR_Q8_0 => low_dim / 32 * 34,
            c.DS4_METAL_TENSOR_Q4_K => low_dim / 256 * 144,
            c.DS4_METAL_TENSOR_Q4_0 => low_dim / 32 * 18,
            else => return false,
        };
        const a_bytes = @as(u64, self.groups) *% self.rank *% a_row;
        const b_bytes = self.outputs *% b_row;
        if (self.a > self.size or a_bytes > self.size - self.a or self.b > self.size or b_bytes > self.size - self.b) return errorLine("ds4: Metal Q4 attention output batch weights are outside the mapped model");
        const head_row = @as(u64, self.groups) *% self.width *% 4;
        if (api.tensorBytes(self.heads) < @as(u64, self.rows) *% head_row or api.tensorBytes(self.low) < @as(u64, self.rows) *% low_dim *% 4 or api.tensorBytes(self.out) < @as(u64, self.rows) *% self.outputs *% 4) return errorLine("ds4: Metal Q4 attention output batch received undersized buffers");
        const padded64 = (@as(u64, self.rows) + 63) / 64 * 64;
        const padded: u32 = if (padded64 <= max32) @intCast(padded64) else self.rows;
        // Only substantial tails borrow already-available workspace. Harmless
        // extra rows are computed; the following projection reads real rows.
        const padding = self.rows >= 256 and padded > self.rows and api.tensorBytes(self.heads) >= @as(u64, padded) *% head_row and api.tensorBytes(self.low) >= @as(u64, padded) *% low_dim *% 4;
        var use_mpp = (self.rows % 64 == 0 or padding) and mpp();
        var program: ?*const Pipeline = null;
        if (use_mpp) {
            program = pipeline("kernel_attn_out_low_q4_K_mpp_direct_rhs_n64", 2, 0);
            if (program == null) {
                c.ds4_metal_dense_warn_mpp();
                use_mpp = false;
            }
        }
        var ids: ?Buffer = null;
        defer if (ids) |buffer| buffer.deinit();
        if (!use_mpp) ids = idsBuffer(self.rows, self.groups, true) orelse return false;
        const w = weight(self.model, self.size, self.a, a_bytes) orelse return false;
        defer w.deinit();
        const input = Inputs.init(self.heads, self.low);
        defer input.deinit();
        var scope = Scope.init() orelse return false;
        defer scope.deinit();
        var ok = scope.acquire();
        if (ok) {
            const matrix = mmArgs(self.width, self.rank, self.groups, if (use_mpp and padding) padded else self.rows, a_row);
            ok = if (use_mpp) accelerated(&scope.command, matrix, w.view, input, program) else mapped(&scope.command, matrix, mapArgs(self.width, self.groups, self.rows), w.view, input, ids.?.view, mapName(self.groups), false);
        }
        // General quantized projection remains an explicit public library,
        // selected independently; no C attention-output controller is called.
        if (ok) ok = c.ds4_gpu_matmul_quant_tensor(self.out, self.model, self.size, self.b, output_type, low_dim, self.outputs, self.low, self.rows) != 0;
        return scope.end(ok);
    }
    pub fn lowOnly(self: Projection, q4: bool, group0: u32) bool {
        if (api.init() == 0 or self.low == null or self.heads == null or self.model == null or self.width == 0 or self.rank == 0 or self.groups == 0 or self.width > max32 or self.rank > max32) return false;
        const low_dim = @as(u64, self.groups) *% self.rank;
        if (q4) {
            if (self.width % 256 != 0) return errorLine("ds4: Metal attention output low received invalid Q4_K dimensions");
        } else if (self.width % 32 != 0 or low_dim > max32) return errorLine("ds4: Metal attention output low received invalid q8 dimensions");
        const row = if (q4) self.width / 256 * 144 else self.width / 32 * 34;
        const group_bytes = self.rank *% row;
        var offset = self.a;
        var count = @as(u64, self.groups) *% group_bytes;
        if (q4) {
            if (self.rank > max64 / row or group0 > max64 / group_bytes or self.groups > max64 / group_bytes) return false;
            const skip = @as(u64, group0) * group_bytes;
            count = @as(u64, self.groups) * group_bytes;
            if (offset > max64 - skip or offset + skip > self.size or count > self.size - (offset + skip)) return errorLine("ds4: Metal Q4 attention output low weights are outside the mapped model");
            offset += skip;
        } else if (offset > self.size or count > self.size - offset) return errorLine("ds4: Metal attention output low weights are outside the mapped model");
        if (api.tensorBytes(self.heads) < @as(u64, self.groups) *% self.width *% 4 or api.tensorBytes(self.low) < low_dim *% 4) return errorLine(if (q4) "ds4: Metal Q4 attention output low received undersized buffers" else "ds4: Metal attention output low received undersized buffers");
        const w = weight(self.model, self.size, offset, count) orelse return false;
        defer w.deinit();
        const input = Inputs.init(self.heads, self.low);
        defer input.deinit();
        var scope = Scope.init() orelse return false;
        defer scope.deinit();
        const ok = scope.acquire() and direct(&scope.command, lowArgs(self.width, self.rank, self.groups, 1, row, !q4, false), w.view, input, !q4, empty_view);
        return scope.end(ok);
    }
    pub fn tpScalar(self: Projection, total: u32, first: u32) bool {
        if (api.init() == 0 or self.out == null or self.low == null or self.heads == null or self.model == null or self.width == 0 or self.rank == 0 or self.groups == 0 or first +% self.groups > total or self.width % 32 != 0 or self.rank *% self.groups % 32 != 0 or self.width > max32 or self.rank > max32 or self.outputs > max32) return false;
        const row = self.width / 32 * 34;
        const group_bytes = self.rank *% row;
        const a_bytes = @as(u64, total) *% group_bytes;
        if (self.a > self.size or a_bytes > self.size - self.a) return errorLine("ds4: Metal TP attention output weights are outside the mapped model");
        // The owned head/low groups remain compact at offset zero, even when
        // their corresponding weights are a window in the complete matrix.
        if (api.tensorBytes(self.heads) < @as(u64, self.groups) *% self.width *% 4 or api.tensorBytes(self.low) < @as(u64, self.groups) *% self.rank *% 4 or api.tensorBytes(self.out) < self.outputs *% 4) return errorLine("ds4: Metal TP attention output received undersized buffers");
        const w = weight(self.model, self.size, self.a, a_bytes) orelse return false;
        defer w.deinit();
        const input = Inputs.init(self.heads, self.low);
        defer input.deinit();
        var command: Command = .{ .label = "TP attention output" };
        defer command.drop();
        if (!command.acquire() or command.owned != 0) return false;
        if (!direct(&command, lowArgs(self.width, self.rank, self.groups, 1, row, true, false), shifted(w.view, @as(u64, first) *% group_bytes), input, true, empty_view)) return false;
        return (KSlice{ .out = self.out, .model = self.model, .size = self.size, .offset = self.b, .width = @as(u64, total) *% self.rank, .outputs = self.outputs, .first = @as(u64, first) *% self.rank, .count = @as(u64, self.groups) *% self.rank, .input = self.low }).scalar();
    }
};

pub const KSlice = struct {
    out: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    width: u64,
    outputs: u64,
    first: u64,
    count: u64,
    input: ?*const Tensor,
    input_offset: u64 = 0,
    pub fn scalar(self: KSlice) bool {
        if (api.init() == 0 or self.width & 31 != 0 or self.first & 31 != 0 or self.count & 31 != 0 or self.count == 0 or self.first +% self.count > self.width or self.width > max32 or self.outputs > max32) return false;
        const x = Buffer.tensor(self.input);
        defer x.deinit();
        const y = Buffer.tensor(self.out);
        defer y.deinit();
        if (!x.fits((self.input_offset +% self.count) *% 4) or !y.fits(self.outputs * 4)) return errorLine("ds4: Metal Q8_0 kslice matmul received undersized buffers");
        const row = self.width / 32 * 34;
        const bytes = self.outputs *% row;
        if (self.offset > self.size or bytes > self.size - self.offset) return errorLine("ds4: Metal Q8_0 kslice weights are outside the mapped model");
        const w = weight(self.model, self.size, self.offset, bytes) orelse return false;
        defer w.deinit();
        var command: Command = .{ .label = "Q8_0 kslice matvec" };
        defer command.drop();
        if (!command.acquire()) return false;
        const facts = c.ds4_metal_dense_facts(false);
        const fallback: u64 = if (facts & c.DS4_DENSE_TP_TWO != 0) 2 else 4;
        const requested = c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", fallback, 1, 8);
        const nsg: i16 = if (self.outputs > 65536) 8 else @intCast(requested);
        const fold = if (command.owned == 0) c.ds4_metal_output_fold_take(self.out, self.outputs * 4) else std.mem.zeroes(c.ds4_metal_output_fold);
        const program = pipeline(if (fold.active) "kernel_dsv4_mul_mv_q8_0_f32_tp_flag_checked" else "kernel_mul_mv_q8_0_f32", 1, nsg) orelse return false;
        var a = std.mem.zeroes(c.ds4_metal_mv_args);
        a.ne00 = narrow(self.count);
        a.ne01 = narrow(self.outputs);
        a.ne02 = 1;
        a.nb00 = 34;
        a.nb01 = row;
        a.nb02 = row *% self.outputs;
        a.nb03 = a.nb02;
        a.ne10 = narrow(self.count);
        a.ne11 = 1;
        a.ne12 = 1;
        a.nb10 = 4;
        a.nb11 = self.width * 4;
        a.nb12 = a.nb11;
        a.nb13 = a.nb11;
        a.ne0 = narrow(self.outputs);
        a.ne1 = 1;
        a.nr0 = 2;
        a.r2 = 1;
        a.r3 = 1;
        var plan = launch(c.DS4_OUT_KSLICE_MV, program, .{ .vector = a }, shifted(w.view, self.first / 32 * 34), shifted(x.view, self.input_offset *% 4), y.view);
        plan.groups_x = (self.outputs + 1) / 2;
        plan.threads_y = @intCast(nsg);
        plan.shared_bytes = 256;
        plan.fold = fold.active;
        plan.fold_slot = fold.slot;
        plan.fold_value = fold.value;
        if (!encode(&command, plan)) return false;
        if (fold.active) c.ds4_metal_output_fold_publish();
        return command.finish();
    }
    pub fn rows(self: KSlice, count: u64, stride: u64, first_input: u64) bool {
        if (api.init() == 0 or !mpp() or self.out == null or self.input == null or self.model == null or self.width % 64 != 0 or self.outputs % 64 != 0 or self.first % 32 != 0 or self.count % 64 != 0 or self.count == 0 or self.first > self.width or self.count > self.width - self.first or stride > max32 or first_input > stride or self.count > stride - first_input or count % 32 != 0 or count == 0 or self.width > max32 or self.outputs > max32 or count > max32) return false;
        const row = self.width / 32 * 34;
        const wbytes = self.outputs *% row;
        const x = Buffer.tensor(self.input);
        defer x.deinit();
        const y = Buffer.tensor(self.out);
        defer y.deinit();
        if (!x.fits(count *% stride *% 4) or !y.fits(count *% self.outputs *% 4) or self.offset > self.size or wbytes > self.size - self.offset) return false;
        const w = weight(self.model, self.size, self.offset, wbytes) orelse return false;
        defer w.deinit();
        const tile: u64 = if (count % 128 == 0) 128 else if (count % 64 == 0) 64 else 32;
        const name: [*:0]const u8 = if (tile == 128) "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128" else if (tile == 64) "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64" else "kernel_mul_mm_q8_0_f32_nax_direct_rhs";
        const program = c.ds4_metal_dense_pipeline(c.DS4_DENSE_MM, name, 0, 0, false, false) orelse return false;
        var a = std.mem.zeroes(c.ds4_metal_mm_args);
        a.ne00 = narrow(self.count);
        a.ne02 = 1;
        a.nb01 = row;
        a.nb02 = row *% self.outputs;
        a.nb03 = a.nb02;
        a.ne12 = 1;
        a.nb10 = 4;
        a.nb11 = stride * 4;
        a.nb12 = count *% stride *% 4;
        a.nb13 = a.nb12;
        a.ne0 = narrow(self.outputs);
        a.ne1 = narrow(count);
        a.r2 = 1;
        a.r3 = 1;
        var command: Command = .{ .label = "Q8_0 NAX kslice rows matmul" };
        defer command.drop();
        if (!command.acquire()) return false;
        var plan = launch(c.DS4_OUT_KSLICE_MM, program, .{ .matrix = a }, shifted(w.view, self.first / 32 * 34), shifted(x.view, first_input * 4), y.view);
        plan.groups_x = count / tile;
        plan.groups_y = self.outputs / 64;
        plan.threads_x = 128;
        plan.shared_bytes = 8192;
        return encode(&command, plan) and command.finish();
    }
};
