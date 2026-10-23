//! DeepSeek router selection: transformed probabilities, optional bias/hash or
//! visual selection, and normalization of the original un-biased probabilities.
//! Fusion choices keep their original environment/hardware/quality fallthrough.
const std = @import("std");
const kernels = @import("router_ops.zig");
const c = kernels.c;
const api = @import("bridge.zig");
const Buffer = kernels.Buffer;
const View = kernels.View;
const Ops = kernels.Ops;
const Pso = kernels.Pso;
const Tensor = api.Tensor;
const empty = std.mem.zeroes(View);
pub const Mode = enum { single, batch, visual };
pub const Request = struct {
    selected: ?*Tensor,
    weights: ?*Tensor,
    probs: ?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    experts: u32,
    used: u32,
    scale: f32,
    groups: u32 = 1,
    used_groups: u32 = 0,
    has_bias: bool,
    hash: bool,
    logits: ?*const Tensor,
    tokens: ?*const Tensor = null,
    token: u32 = 0,
    rows: u32 = 1,
    mode: Mode = .single,
    vision: ?*const anyopaque = null,
    vision_size: u64 = 0,
    visual_offset: u64 = 0,
    vocab: u32 = 0,
};
fn absent(name: [*:0]const u8) bool {
    return c.getenv(name) == null;
}
fn errorLine(line: [*:0]const u8) c_int {
    return api.reportError(line);
}
fn weight(map: ?*const anyopaque, size: u64, offset: u64, bytes: u64) Buffer {
    var inner: u64 = 0;
    const value = api.retainModelRange(map, size, offset, bytes, &inner);
    return .{
        .view = .{
            .buffer = value,
            .offset = inner,
            .bytes = bytes,
        },
    };
}
fn capable(p: Pso) bool {
    return p.handle != null and p.simd_width == 32 and p.threads >= 256;
}
const Views = struct {
    logits: Buffer,
    selected: Buffer,
    weights: Buffer,
    probs: Buffer,
    tokens: Buffer,
    fn init(r: Request) Views {
        return .{
            .logits = Buffer.tensor(r.logits),
            .selected = Buffer.tensor(r.selected),
            .weights = Buffer.tensor(r.weights),
            .probs = Buffer.tensor(r.probs),
            .tokens = if (r.mode == .single) Buffer{ .view = empty } else Buffer.tensor(r.tokens),
        };
    }
    fn deinit(v: Views) void {
        v.tokens.deinit();
        v.probs.deinit();
        v.weights.deinit();
        v.selected.deinit();
        v.logits.deinit();
    }
    fn fits(v: Views, r: Request) bool {
        const probabilities = @as(u64, r.rows) *% r.experts *% 4;
        const selected = @as(u64, r.rows) *% r.used *% 4;
        return v.logits.fits(probabilities) and v.probs.fits(probabilities) and
            v.selected.fits(selected) and v.weights.fits(selected) and
            (r.mode == .single or v.tokens.fits(@as(u64, r.rows) * 4));
    }
};
fn selectedPlan(r: Request, v: Views, bias: View, hash: View) c.ds4_metal_router_launch {
    var p = std.mem.zeroes(c.ds4_metal_router_launch);
    p.source = v.logits.view;
    p.probs = v.probs.view;
    p.bias = bias;
    p.hash = hash;
    p.tokens = v.tokens.view;
    p.selected = v.selected.view;
    p.weights = v.weights.view;
    p.args.one = .{
        .has_bias = @intFromBool(r.has_bias and !r.hash),
        .hash_mode = @intFromBool(r.hash),
        .use_token_buffer = @intFromBool(r.mode != .single),
        .token = if (r.mode == .single) r.token else 0,
        .hash_rows = r.hash_rows,
    };
    p.groups_x = 1;
    p.groups_y = 1;
    p.threads = 256;
    return p;
}
fn weightsOne(ops: Ops, v: Views, pipeline: Pso) bool {
    var p = kernels.plan(c.DS4_ROUTER_WEIGHT_ONE, pipeline);
    p.probs = v.probs.view;
    p.selected = v.selected.view;
    p.weights = v.weights.view;
    p.groups_x = 6;
    p.threads = 6;
    p.direct_threads = true;
    return ops.encode(&p);
}
fn select(ops: Ops, r: Request, v: Views, bias: View, hash: View, visual: View) bool {
    const facts = c.ds4_metal_router_facts();
    const quality = facts & c.DS4_ROUTER_QUALITY != 0;
    const pre = facts & c.DS4_ROUTER_PRE_M5 != 0;
    const apple = pre or facts & c.DS4_ROUTER_M5 != 0;
    const mixed = r.mode == .visual;
    const flash = r.experts == 256 and r.used == 6 and @abs(r.scale - 1.5) <= 1.0e-6;
    if (flash and !mixed and !quality and r.rows == 1 and absent("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION")) {
        const simd = !r.hash and capable(kernels.pso(c.DS4_ROUTER_SIMD, false)) and apple and
            (!pre or absent("DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_FINALIZE"));
        const fused_weights = simd and capable(kernels.pso(c.DS4_ROUTER_SIMD_WEIGHTS, false)) and apple and
            (!pre or absent("DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_WEIGHTS_FUSION"));
        const transform = fused_weights and !r.hash and facts & c.DS4_ROUTER_SSD == 0 and apple and
            absent("DS4_METAL_DISABLE_PRE_M5_ROUTER_TRANSFORM_FINALIZE_FUSION") and
            capable(kernels.pso(c.DS4_ROUTER_TRANSFORM_WEIGHTS, false));
        const final_kind: u32 = if (transform) c.DS4_ROUTER_TRANSFORM_WEIGHTS else if (fused_weights) c.DS4_ROUTER_SIMD_WEIGHTS else if (simd) c.DS4_ROUTER_SIMD else c.DS4_ROUTER_FINALIZE;
        const probability = if (transform) std.mem.zeroes(Pso) else kernels.pso(c.DS4_ROUTER_TRANSFORM, true);
        const final = kernels.pso(final_kind, true);
        const weights = if (fused_weights) std.mem.zeroes(Pso) else kernels.pso(c.DS4_ROUTER_WEIGHTS_ONE, true);
        if (final.handle == null or (!transform and probability.handle == null) or (!fused_weights and weights.handle == null)) return false;
        if (!transform and !ops.unary(probability, v.logits.view, v.probs.view, r.experts, 1, true, 0, 0)) return false;
        if ((r.has_bias and !r.hash and bias.buffer == null) or (r.hash and hash.buffer == null) or (r.mode != .single and v.tokens.view.buffer == null)) return false;
        var p = selectedPlan(r, v, bias, hash);
        p.kind = if (transform) c.DS4_ROUTER_SELECT_TRANSFORM else c.DS4_ROUTER_SELECT_ONE;
        p.pipeline = final;
        p.fused_weights = fused_weights;
        p.shared = if (simd) 4096 else 2048;
        if (!ops.encode(&p)) return false;
        return fused_weights or weightsOne(ops, v, weights);
    }
    if (flash and !mixed and !quality and r.rows == 1) {
        if (!ops.unary(kernels.pso(c.DS4_ROUTER_TRANSFORM, true), v.logits.view, v.probs.view, r.experts, 1, true, 0, 0)) return false;
    } else if (!ops.unary(kernels.pso(c.DS4_ROUTER_SOFTPLUS, false), v.logits.view, v.probs.view, r.experts, r.rows, true, 0, 0) or
        !ops.unary(kernels.pso(c.DS4_ROUTER_SQRT, false), v.probs.view, v.probs.view, r.experts, r.rows, true, 0, 0)) return false;

    if (mixed) {
        const pipeline = kernels.pso(c.DS4_ROUTER_VISUAL_PSO, false);
        if (pipeline.handle == null or r.experts != 256 or r.used != 6) return false;
        var p = selectedPlan(r, v, bias, hash);
        p.kind = c.DS4_ROUTER_SELECT_VISUAL;
        p.pipeline = pipeline;
        p.visual = visual;
        p.args.visual = .{
            .hash_rows = r.hash_rows,
            .vocab_size = r.vocab,
            .n_tokens = r.rows,
            .has_bias = @intFromBool(r.has_bias and !r.hash),
            .hash_mode = @intFromBool(r.hash),
        };
        p.groups_x = r.rows;
        p.shared = 2048;
        if (!ops.encode(&p)) return false;
    } else if (r.hash) {
        if (!ops.hashRows(hash, v.tokens.view, if (r.mode == .single) r.token else null, v.selected.view, r.hash_rows, r.used, r.rows)) return false;
    } else {
        var score = r.probs;
        var scratch: ?*Tensor = null;
        defer if (scratch) |value| c.ds4_metal_resource_dispose(value);
        if (r.has_bias) {
            if (bias.buffer == null) return false;
            scratch = c.ds4_metal_router_selection_scratch(@as(u64, r.rows) *% r.experts *% 4);
            if (scratch == null) return false;
            const view = Buffer.tensor(scratch);
            defer view.deinit();
            if (!ops.binary(kernels.pso(c.DS4_ROUTER_ADD, false), kernels.binaryArgs(r.experts, r.rows, r.experts, false), v.probs.view, bias, view.view, null)) return false;
            score = scratch;
        }
        if (c.ds4_gpu_indexer_topk_tensor(r.selected, score, r.experts, r.rows, r.used) == 0) return false;
    }
    const batch = flash and !quality and r.rows > 1 and facts & c.DS4_ROUTER_M3 != 0 and
        kernels.pso(c.DS4_ROUTER_WEIGHTS_BATCH, false).handle != null and
        absent("DS4_METAL_DISABLE_ROUTER_WEIGHTS_BATCH_FUSION") and absent("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION");
    if (batch) {
        var p = kernels.plan(c.DS4_ROUTER_WEIGHT_BATCH, kernels.pso(c.DS4_ROUTER_WEIGHTS_BATCH, false));
        p.probs = v.probs.view;
        p.selected = v.selected.view;
        p.weights = v.weights.view;
        p.scale = r.scale;
        p.shared = 160;
        p.groups_x = r.rows;
        p.threads = 6;
        return ops.encode(&p);
    }
    if (flash and !quality and r.rows == 1) return weightsOne(ops, v, kernels.pso(c.DS4_ROUTER_WEIGHTS_ONE, true));
    const sum = Buffer{ .view = c.ds4_metal_router_sum_scratch(@as(u64, r.rows) * 4) };
    defer sum.deinit();
    if (!sum.valid() or !ops.gatherWeights(v.probs.view, v.selected.view, v.weights.view, r.experts, r.used, r.rows) or
        !ops.sum(v.weights.view, sum.view, r.used, r.rows) or
        !ops.unary(kernels.pso(c.DS4_ROUTER_CLAMP, false), sum.view, sum.view, 1, r.rows, false, 6.103515625e-5, std.math.inf(f32))) return false;
    // Divide first, then scale; the 2^-14 floor and kernel ordering are part of
    // the numerical contract. Selection bias never changes these probabilities.
    return ops.binary(kernels.pso(c.DS4_ROUTER_DIVIDE, false), kernels.binaryArgs(r.used, r.rows, 1, true), v.weights.view, sum.view, v.weights.view, null) and
        ops.binary(kernels.pso(c.DS4_ROUTER_SCALE_PSO, false), kernels.binaryArgs(r.used, r.rows, 1, false), v.weights.view, empty, v.weights.view, r.scale);
}
pub fn run(r: Request) c_int {
    if (api.init() == 0) return 0;
    if (r.selected == null or r.weights == null or r.probs == null or r.logits == null or r.map == null or r.experts == 0 or r.used == 0) return 0;
    if (r.mode == .single) {
        if (r.hash and r.token >= r.hash_rows) return 0;
    } else if (r.tokens == null or r.rows == 0) return 0;
    if (r.mode == .visual) {
        if (r.vision == null or r.vocab == 0 or r.experts != 256 or r.used != 6 or @abs(r.scale - 1.5) > 1.0e-6) return 0;
    } else if (r.groups > 1 or r.used_groups > 0) return errorLine("ds4: Metal router group gating is not part of this DeepSeek V4 path");
    const views = Views.init(r);
    defer views.deinit();
    if (!views.fits(r)) return errorLine(switch (r.mode) {
        .single => "ds4: Metal router select received undersized buffers",
        .batch => "ds4: Metal router batch select received undersized buffers",
        .visual => "ds4: Metal visual router received undersized buffers",
    });
    const bias = if (r.has_bias and !r.hash) weight(r.map, r.size, r.bias_offset, @as(u64, r.experts) * 4) else Buffer{ .view = empty };
    defer bias.deinit();
    if (r.has_bias and !r.hash and !bias.valid()) return 0;
    const hash = if (r.hash) weight(r.map, r.size, r.hash_offset, @as(u64, r.hash_rows) *% r.used *% 4) else Buffer{ .view = empty };
    defer hash.deinit();
    if (r.hash and !hash.valid()) return 0;
    const visual = if (r.mode == .visual) weight(r.vision, r.vision_size, r.visual_offset, @as(u64, r.experts) * 4) else Buffer{ .view = empty };
    defer visual.deinit();
    if (r.mode == .visual and !visual.valid()) return 0;
    const had_batch = c.ds4_gpu_commands_active() != 0;
    if (!had_batch and c.ds4_gpu_begin_commands() == 0) return 0;
    var owned: c_int = 0;
    const token = api.acquire(&owned);
    defer if (token) |value| {
        _ = api.release(value, owned, 0, "router select");
    };
    const ok = if (token != null) select(.{
        .command = token,
    }, r, views, bias.view, hash.view, visual.view) else false;
    // Even a failed partial sequence closes its own batch. A borrowed caller's
    // batch stays active, and command-token release only drops its local retain.
    const ended = if (had_batch) true else c.ds4_gpu_end_commands() != 0;
    return @intFromBool(ended and ok);
}
