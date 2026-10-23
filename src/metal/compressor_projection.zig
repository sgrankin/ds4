//! Decode projection/store fusions. Return 1 if dispatched, 0 for the original
//! separate path, and -1 on error where that public API defines the distinction.
const std = @import("std");
const kernels = @import("compressor_ops.zig");
const c = kernels.c;
const api = kernels.api;
const Buffer = kernels.Buffer;
const Tensor = kernels.Tensor;
const View = kernels.View;
const Command = kernels.Command;
const narrow = kernels.narrow;
const at = kernels.at;
const Fused = c.ds4_metal_compressor_fused;
const zero_view = std.mem.zeroes(View);
pub const Pair = struct {
    out_kv: ?*Tensor,
    out_score: ?*Tensor,
    state_kv: ?*Tensor,
    state_score: ?*Tensor,
    weight_kv: u64,
    weight_score: u64,
    ape: u64,
    ape_type: u32,
    width: u32,
    fn pointers(self: Pair) bool {
        return self.out_kv != null and self.out_score != null and self.state_kv != null and self.state_score != null;
    }
    fn range(self: Pair, size: u64, input: u64, ratio: u32) bool {
        const bytes = input *% 2 *% self.width;
        const ape_bytes = @as(u64, ratio) *% self.width *% (if (self.ape_type == 1) @as(u64, 2) else 4);
        return inRange(size, self.weight_kv, bytes) and inRange(size, self.weight_score, bytes) and inRange(size, self.ape, ape_bytes);
    }
    fn tensorsFit(self: Pair, ratio: u32) bool {
        const rows: u32 = if (ratio == 4) 2 *% ratio else ratio;
        const output = @as(u64, self.width) * 4;
        const state = @as(u64, rows) *% output;
        return api.tensorBytes(self.out_kv) >= output and api.tensorBytes(self.out_score) >= output and api.tensorBytes(self.state_kv) >= state and api.tensorBytes(self.state_score) >= state;
    }
};
pub const Request = struct {
    pairs: [2]Pair,
    map: ?*const anyopaque,
    size: u64,
    input: u64,
    x: ?*const Tensor,
    ratio: u32,
    pos: u32,
};
const OwnedPair = struct {
    views: c.ds4_metal_compressor_pair_views = std.mem.zeroes(c.ds4_metal_compressor_pair_views),
    fn activations(item: Pair) OwnedPair {
        return .{ .views = .{
            .weight_kv = zero_view,
            .weight_score = zero_view,
            .ape = zero_view,
            .out_kv = Buffer.tensor(item.out_kv).view,
            .out_score = Buffer.tensor(item.out_score).view,
            .state_kv = Buffer.tensor(item.state_kv).view,
            .state_score = Buffer.tensor(item.state_score).view,
        } };
    }
    fn valid(self: OwnedPair) bool {
        return self.views.out_kv.buffer != null and self.views.out_score.buffer != null and self.views.state_kv.buffer != null and self.views.state_score.buffer != null;
    }
    fn wrap(self: *OwnedPair, item: Pair, request: Request) bool {
        const bytes = request.input *% 2 *% item.width;
        const ape_bytes = @as(u64, request.ratio) *% item.width *% (if (item.ape_type == 1) @as(u64, 2) else 4);
        self.views.weight_kv = (kernels.model(request.map, request.size, item.weight_kv, bytes) orelse return false).view;
        self.views.weight_score = (kernels.model(request.map, request.size, item.weight_score, bytes) orelse return false).view;
        self.views.ape = (kernels.model(request.map, request.size, item.ape, ape_bytes) orelse return false).view;
        return true;
    }
    fn deinit(self: OwnedPair) void {
        inline for (comptime std.meta.fieldNames(c.ds4_metal_compressor_pair_views)) |name| (Buffer{ .view = @field(self.views, name) }).deinit();
    }
};
fn inRange(size: u64, offset: u64, bytes: u64) bool {
    return offset <= size and bytes <= size - offset;
}
fn mv(width: u64, outputs: u64, q8: bool, nr0: i32) c.ds4_metal_mv_args {
    const row = if (q8) width / 32 * 34 else width *% 2;
    return .{
        .ne00 = narrow(width),
        .ne01 = narrow(outputs),
        .ne02 = 1,
        .nb00 = if (q8) 34 else 2,
        .nb01 = row,
        .nb02 = row *% outputs,
        .nb03 = row *% outputs,
        .ne10 = narrow(width),
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = width *% 4,
        .nb12 = width *% 4,
        .nb13 = width *% 4,
        .ne0 = narrow(outputs),
        .ne1 = 1,
        .nr0 = nr0,
        .r2 = 1,
        .r3 = 1,
    };
}
fn store(item: Pair, request: Request) c.ds4_gpu_dsv4_compressor_store_one_args {
    return .{ .width = item.width, .ratio = request.ratio, .pos = request.pos, .ape_type = item.ape_type };
}
fn dispatch(plan: *const Fused, label: [*:0]const u8, error_result: c_int) c_int {
    var command = Command{};
    defer command.drop();
    if (!command.acquire()) return error_result;
    if (c.ds4_metal_compressor_fused_encode(command.token.?, plan) == 0 or !command.finish(label)) return -1;
    return 1;
}
pub fn pair(request: Request) c_int {
    if (api.init() == 0) return -1;
    const facts = c.ds4_metal_attention_get_facts().flags;
    if (facts & c.DS4_ATTN_QUALITY != 0 or facts & (c.DS4_ATTN_M3 | c.DS4_ATTN_M5) == 0 or kernels.present("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ") or kernels.present("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE")) return 0;
    const first = request.pairs[0];
    if (!first.pointers() or request.map == null or request.x == null or request.input == 0 or first.width == 0 or request.ratio == 0 or first.ape_type > 1) return -1;
    if (request.input != 4096 or (first.width != 256 and first.width != 512 and first.width != 1024) or (request.ratio != 4 and request.ratio != 128)) return 0;
    if (!first.range(request.size, request.input, request.ratio)) return -1;
    const input = Buffer.tensor(request.x);
    defer input.deinit();
    var output = OwnedPair.activations(first);
    defer output.deinit();
    if (!input.fits(request.input * 4) or !output.valid() or !first.tensorsFit(request.ratio)) return -1;
    if (!output.wrap(first, request)) return -1;
    const nsg = @min((request.input + 127) / 128, @as(u64, 8));
    const nr0: i32 = if (c.ds4_metal_compressor_pair_nr4() and (first.width == 512 or first.width == 1024) and request.input >= 4096) 4 else 2;
    var plan = std.mem.zeroes(Fused);
    plan.kind = c.DS4_COMP_FUSED_PAIR;
    plan.pipeline = c.ds4_metal_routed_pipeline(1, "kernel_mul_mv_f16_f32_pair_compressor_store_4", @intCast(nsg));
    if (plan.pipeline == null) return -1;
    plan.mv[0] = mv(request.input, first.width, false, nr0);
    plan.store[0] = store(first, request);
    plan.input = input.view;
    plan.pairs[0] = output.views;
    plan.groups = (@as(u64, first.width) + @as(u64, @intCast(nr0)) - 1) / @as(u64, @intCast(nr0));
    plan.nsg = nsg;
    plan.shared = 32 * @as(u64, @intCast(nr0)) * 4;
    return dispatch(&plan, "F16 paired matvec compressor state store", -1);
}
/// The attention/indexer pairs share activation and matvec shape. The
/// unchanged kernel preserves the separate paths' reduction trees bit for bit.
pub fn quad(request: Request) c_int {
    if (api.init() == 0) return -1;
    const first = request.pairs[0];
    const second = request.pairs[1];
    if (!first.pointers() or !second.pointers() or request.map == null or request.x == null or first.ape_type > 1 or second.ape_type > 1) return -1;
    if (request.input != 4096 or (first.width != 512 and first.width != 1024) or second.width != 256 or request.ratio != 4) return 0;
    if (!first.range(request.size, request.input, request.ratio) or !second.range(request.size, request.input, request.ratio)) return -1;
    const input = Buffer.tensor(request.x);
    defer input.deinit();
    var left = OwnedPair.activations(first);
    defer left.deinit();
    var right = OwnedPair.activations(second);
    defer right.deinit();
    if (!input.fits(request.input * 4) or !left.valid() or !right.valid() or !first.tensorsFit(request.ratio) or !second.tensorsFit(request.ratio)) return -1;
    if (!left.wrap(first, request) or !right.wrap(second, request)) return -1;
    var plan = std.mem.zeroes(Fused);
    plan.kind = c.DS4_COMP_FUSED_QUAD;
    plan.pipeline = c.ds4_metal_routed_pipeline(1, "kernel_mul_mv_f16_f32_quad_compressor_store_4", 8);
    if (plan.pipeline == null) return -1;
    plan.mv[0] = mv(request.input, first.width, false, 2);
    plan.store = .{ store(first, request), store(second, request) };
    plan.input = input.view;
    plan.pairs = .{ left.views, right.views };
    plan.groups = (@as(u64, first.width) + second.width + 1) / 2;
    plan.nsg = 8;
    plan.shared = 256;
    return dispatch(&plan, "F16 quad matvec compressor state store", -1);
}
pub const Qkv = struct { projection: Request, qr: ?*Tensor, raw: ?*Tensor, q_offset: u64, kv_offset: u64, rank: u32, kv_width: u32 };
/// Decode-only Q8 q_a/KV and F16 compressor projections read the same
/// normalized input; fusion preserves each standalone kernel's operation order.
pub fn qkv(request: Qkv) c_int {
    if (api.init() == 0) return 0;
    const projection = request.projection;
    const first = projection.pairs[0];
    const second = projection.pairs[1];
    if (request.qr == null or request.raw == null or !first.pointers() or !second.pointers() or projection.map == null or projection.x == null or projection.ratio == 0 or projection.input & 31 != 0 or first.width & 1 != 0 or second.width & 1 != 0) return 0;
    // The public fused-QKV width is u32: preserve its original row-byte
    // product before widening, independently of the u64 MV packet factory.
    const q8_row: u64 = (@as(u32, @intCast(projection.input)) / 32) *% 34;
    const q_bytes = @as(u64, request.rank) *% q8_row;
    const kv_bytes = @as(u64, request.kv_width) *% q8_row;
    if (!inRange(projection.size, request.q_offset, q_bytes) or !inRange(projection.size, request.kv_offset, kv_bytes) or !first.range(projection.size, projection.input, projection.ratio) or !second.range(projection.size, projection.input, projection.ratio)) return -1;
    if (!first.tensorsFit(projection.ratio) or !second.tensorsFit(projection.ratio) or api.tensorBytes(projection.x) < projection.input *% 4 or api.tensorBytes(request.qr) < @as(u64, request.rank) * 4 or api.tensorBytes(request.raw) < @as(u64, request.kv_width) * 4) return -1;
    var left = OwnedPair.activations(first);
    defer left.deinit();
    var right = OwnedPair.activations(second);
    defer right.deinit();
    const input = Buffer.tensor(projection.x);
    defer input.deinit();
    const qr = Buffer.tensor(request.qr);
    defer qr.deinit();
    const raw = Buffer.tensor(request.raw);
    defer raw.deinit();
    const q_weight = kernels.model(projection.map, projection.size, request.q_offset, q_bytes) orelse return -1;
    defer q_weight.deinit();
    const kv_weight = kernels.model(projection.map, projection.size, request.kv_offset, kv_bytes) orelse return -1;
    defer kv_weight.deinit();
    if (!left.wrap(first, projection)) return -1;
    if (second.width != 0) {
        if (!right.wrap(second, projection)) return -1;
    }
    // With no indexer pair, the original kernel still binds valid first-pair
    // weights/APE in its unused slots. Borrow those views without a second owner.
    var right_views = right.views;
    if (second.width == 0) {
        right_views.weight_kv = left.views.weight_kv;
        right_views.weight_score = left.views.weight_score;
        right_views.ape = left.views.ape;
    }
    var plan = std.mem.zeroes(Fused);
    plan.kind = c.DS4_COMP_FUSED_QKV;
    plan.pipeline = c.ds4_metal_routed_pipeline(1, "kernel_dsv4_qkv_pair_quad_compressor_store_q8_0", 8);
    if (plan.pipeline == null) return 0;
    plan.mv = .{ mv(projection.input, request.rank, true, 2), mv(projection.input, request.kv_width, true, 2), mv(projection.input, first.width, false, 2) };
    plan.store = .{ store(first, projection), store(second, projection) };
    plan.pair_groups = (@max(request.rank, request.kv_width) +% 1) / 2;
    plan.input = input.view;
    plan.q_weights = q_weight.view;
    plan.kv_weights = kv_weight.view;
    plan.qr = qr.view;
    plan.raw = raw.view;
    plan.pairs = .{ left.views, right_views };
    plan.groups = (@as(u64, plan.pair_groups) + 1) / 2 + (@as(u64, first.width) + second.width + 1) / 2;
    plan.nsg = 8;
    plan.shared = 1024;
    return dispatch(&plan, "qkv pair + quad compressor store", 0);
}
pub const Finalize = struct {
    stage: ?*Tensor,
    attn_cache: ?*Tensor,
    attn_row: u32,
    attn_norm: u64,
    index_cache: ?*Tensor,
    index_row: u32,
    index_norm: u64,
    states: [4]?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    rope: c.ds4_metal_attention_rope_args,
    eps: f32,
};
/// Emit-path fusion replaces norm/RoPE/FP8/commit/QAT dispatches with the
/// original combined kernel. Its state shift and finalization happen once.
pub fn finalize(request: Finalize) c_int {
    if (api.init() == 0) return 0;
    if (request.stage == null or request.attn_cache == null or request.index_cache == null or request.map == null) return 0;
    if (!inRange(request.size, request.attn_norm, 512 * 4) or !inRange(request.size, request.index_norm, 128 * 4)) return 0;
    for (request.states, 0..) |tensor, index| if (tensor == null or api.tensorBytes(tensor) < @as(u64, if (index < 2) 8 * 1024 * 4 else 8 * 256 * 4)) return 0;
    if (api.tensorBytes(request.stage) < 512 * 4 or api.tensorBytes(request.index_cache) < (@as(u64, request.index_row) + 1) * 128 * 4 or api.tensorBytes(request.attn_cache) < (@as(u64, request.attn_row) + 1) * 512 * 2) return 0;
    const an = kernels.model(request.map, request.size, request.attn_norm, 512 * 4) orelse return 0;
    defer an.deinit();
    const ix = kernels.model(request.map, request.size, request.index_norm, 128 * 4) orelse return 0;
    defer ix.deinit();
    var plan = std.mem.zeroes(Fused);
    plan.kind = c.DS4_COMP_FUSED_FINALIZE;
    plan.pipeline = c.ds4_metal_routed_pipeline(0, "kernel_dsv4_comp_row_finalize_f32", 0);
    if (plan.pipeline == null) return 0;
    const stage = Buffer.tensor(request.stage);
    defer stage.deinit();
    const ac = Buffer.tensor(request.attn_cache);
    defer ac.deinit();
    const ic = Buffer.tensor(request.index_cache);
    defer ic.deinit();
    var states: [4]Buffer = undefined;
    for (&states, request.states) |*owner, tensor| owner.* = Buffer.tensor(tensor);
    defer for (states) |owner| owner.deinit();
    plan.finalize = .{ .rope = request.rope, .rms_eps = request.eps, .pad0 = 0 };
    plan.attn_stage = stage.view;
    plan.attn_norm = an.view;
    plan.attn_cache = at(ac.view, @as(u64, request.attn_row) * 512 * 2);
    plan.index_cache = at(ic.view, @as(u64, request.index_row) * 128 * 4);
    plan.index_norm = ix.view;
    plan.pairs[0].state_kv = states[0].view;
    plan.pairs[0].state_score = states[1].view;
    plan.pairs[1].state_kv = states[2].view;
    plan.pairs[1].state_score = states[3].view;
    plan.groups = 2 + (4 * 1024 + 4 * 256 + 255) / 256;
    // This kernel uses a flat256-thread group, encoded as32×8 only for fused
    // projection kernels. Finalize's encoder selects its original256×1 shape.
    plan.nsg = 1;
    plan.shared = 1024;
    const result = dispatch(&plan, "compressor row finalize", 0);
    return if (result < 0) 0 else result;
}
