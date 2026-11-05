//! V4.1 fused dispatches. Native code owns geometry and admission; the Metal
//! leaf only binds borrowed packets and retained buffer views.
const std = @import("std");
const c = @import("metal_c");
const K = @import("compressor_ops.zig");
const Dense = @import("dense_matmul.zig");
const Rope = @import("rope.zig");
const Tensor = c.ds4_gpu_tensor;
const View = c.ds4_metal_attention_view;
const Buffer = K.Buffer;
const Binding = c.ds4_metal_v41_binding;
const zero_view = std.mem.zeroes(View);
fn buf(index: u32, view: View) Binding {
    return .{ .index = index, .view = view, .bytes = null, .length = 0 };
}
fn arg(index: u32, pointer: anytype) Binding {
    return .{ .index = index, .view = zero_view, .bytes = pointer, .length = @sizeOf(@typeInfo(@TypeOf(pointer)).pointer.child) };
}
fn fits(t: ?*const Tensor, bytes: u64) bool {
    return t != null and c.ds4_gpu_tensor_bytes(t) >= bytes;
}
const Grid = struct { x: u64 = 1, y: u64 = 1, z: u64 = 1, tx: u64 = 32, ty: u64 = 1, shared: u32 = 0, nsg: i16 = 0, barrier: bool = false };
fn dispatch(name: [*:0]const u8, bindings: []const Binding, grid: Grid) c_int {
    if (c.ds4_gpu_init() == 0) return 0;
    var command: K.Command = .{};
    defer command.drop();
    if (!command.acquire()) return 0;
    const launch = c.ds4_metal_v41_launch{ .kernel = name, .nsg = grid.nsg, .count = @intCast(bindings.len), .shared_bytes = grid.shared, .bindings = bindings.ptr, .gx = grid.x, .gy = grid.y, .gz = grid.z, .tx = grid.tx, .ty = grid.ty, .barrier = grid.barrier };
    return @intFromBool(c.ds4_metal_v41_encode(command.token, &launch) != 0 and command.finish(name));
}
fn threads(width: u32) u32 {
    var n: u32 = 32;
    while (n < width / 4 and n < 1024) n *= 2;
    return @max(@min(n, width / 4), 1);
}
export fn ds4_gpu_dsv41_router_rows(selected: ?*Tensor, weights: ?*Tensor, probs: ?*Tensor, logits: ?*const Tensor, map: ?*const anyopaque, size: u64, bias_offset: u64, experts: u32, top_k: u32, scale: f32, rows: u32) c_int {
    if (experts == 0 or experts > 1024 or top_k == 0 or top_k > experts or top_k > 32 or rows == 0 or rows > 65535 or
        !fits(logits, @as(u64, rows) * experts * 4) or !fits(probs, @as(u64, rows) * experts * 4) or
        !fits(selected, @as(u64, rows) * top_k * 4) or !fits(weights, @as(u64, rows) * top_k * 4) or c.ds4_gpu_init() == 0) return 0;
    const bias = K.model(map, size, bias_offset, @as(u64, experts) * 4) orelse return 0;
    defer bias.deinit();
    const l = Buffer.tensor(logits);
    defer l.deinit();
    const p = Buffer.tensor(probs);
    defer p.deinit();
    const s = Buffer.tensor(selected);
    defer s.deinit();
    const w = Buffer.tensor(weights);
    defer w.deinit();
    const args = extern struct { n_expert: u32, top_k: u32, has_bias: u32, scale: f32 }{ .n_expert = experts, .top_k = top_k, .has_bias = 1, .scale = scale };
    var nth: u32 = 1;
    while (nth < experts) nth *= 2;
    return dispatch("kernel_dsv41_router_one", &.{ arg(0, &args), buf(1, l.view), buf(2, bias.view), buf(3, p.view), buf(4, s.view), buf(5, w.view) }, .{ .y = rows, .tx = nth, .shared = (nth * 12 + 15 + 128) & ~@as(u32, 15) });
}
export fn ds4_gpu_dsv41_router_one(selected: ?*Tensor, weights: ?*Tensor, probs: ?*Tensor, logits: ?*const Tensor, map: ?*const anyopaque, size: u64, bias: u64, experts: u32, k: u32, scale: f32) c_int {
    return ds4_gpu_dsv41_router_rows(selected, weights, probs, logits, map, size, bias, experts, k, scale, 1);
}
export fn ds4_gpu_dsv41_resident_route_one(selected: ?*Tensor, probs: ?*const Tensor, map: ?*const anyopaque, size: u64, bias_offset: u64, experts: u32, top_k: u32, resident: [*c]const u8, trace: ?*Tensor) c_int {
    if (experts < 6 or experts > 1024 or top_k != 6 or resident == null or
        !fits(selected, 24) or !fits(probs, @as(u64, experts) * 4) or
        (trace != null and !fits(trace, 32))) return 0;
    var count: u32 = 0;
    for (resident[0..experts]) |present| count += @intFromBool(present != 0);
    if (count < top_k) return 0;
    const bias = K.model(map, size, bias_offset, @as(u64, experts) * 4) orelse return 0;
    defer bias.deinit();
    const p = Buffer.tensor(probs);
    defer p.deinit();
    const s = Buffer.tensor(selected);
    defer s.deinit();
    const t = Buffer.tensor(if (trace != null) trace else selected);
    defer t.deinit();
    const args = extern struct { n_expert: u32, trace: u32 }{ .n_expert = experts, .trace = @intFromBool(trace != null) };
    const mask = Binding{ .index = 4, .view = zero_view, .bytes = resident, .length = experts };
    const simd = if (c.getenv("DS4_METAL_V41_RESIDENT_GPU_SIMD")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
    const kernel: [*:0]const u8 = if (simd) "kernel_dsv41_resident_route_one_simd" else "kernel_dsv41_resident_route_one";
    return dispatch(kernel, &.{ arg(0, &args), buf(1, s.view), buf(2, p.view), buf(3, bias.view), mask, buf(5, t.view) }, .{ .tx = 32 });
}
const QuantRope = extern struct { width: u32, start: u32, inverse: u32 = 0, mode: u32, frequencies: [32]f32 };
const MvRope = extern struct { width: u32, start: u32, inverse: u32, frequencies: [32]f32 };
fn mvRope(width: u32, start: u32, compressed: bool, inverse: bool) MvRope {
    return .{ .width = width, .start = start, .inverse = @intFromBool(inverse), .frequencies = Rope.frequencyValues(compressed) };
}
export fn ds4_gpu_dsv41_rope_coeff_table(tab: ?*Tensor, position: u32) c_int {
    if (position >= 1048576 or !fits(tab, 512)) return 0;
    const out = Buffer.tensor(tab);
    defer out.deinit();
    const raw = mvRope(512, position, false, true);
    const comp = mvRope(512, position, true, true);
    return dispatch("kernel_dsv41_rope_coeff_table", &.{ arg(0, &raw), arg(1, &comp), buf(2, out.view) }, .{ .tx = 64, .barrier = true });
}
export fn ds4_gpu_dsv41_rope_quantize(x: ?*const Tensor, dst: ?*Tensor, offset: u64, width: u32, start: u32, compressed: bool, format: c.ds4_v41_activation_format) c_int {
    const block: u32 = if (format == c.DS4_V41_FP4_E4M3) 16 else 32;
    if (c.getenv("DS4_METAL_DISABLE_V41_ROPE_QUANTIZE") != null or width < 64 or width % block != 0 or format < 1 or format > 3 or start >= 1048576 or offset % 4 != 0 or !fits(x, @as(u64, width) * 4) or offset > c.ds4_gpu_tensor_bytes(dst) or !fits(dst, offset + @as(u64, width) * 4)) return 0;
    const input = Buffer.tensor(x);
    defer input.deinit();
    const out = Buffer.tensor(dst);
    defer out.deinit();
    const args = QuantRope{ .width = width, .start = start, .mode = @intCast(format), .frequencies = Rope.frequencyValues(compressed) };
    return dispatch("kernel_dsv41_rope_quantize", &.{ arg(0, &args), buf(1, input.view), buf(2, K.at(out.view, offset)) }, .{ .x = width / block });
}
export fn ds4_gpu_dsv41_rope_quantize_heads_tab(x: ?*Tensor, width: u32, heads: u32, start: u32, compressed: bool, format: c.ds4_v41_activation_format, tab: ?*const Tensor) c_int {
    const block: u32 = if (format == c.DS4_V41_FP4_E4M3) 16 else 32;
    if (width < 64 or width % block != 0 or heads == 0 or heads > 65535 or start >= 1048576 or format < 1 or format > 3 or !fits(x, @as(u64, width) * heads * 4) or !fits(tab, 512)) return 0;
    const input = Buffer.tensor(x);
    defer input.deinit();
    const table = Buffer.tensor(tab);
    defer table.deinit();
    const args = QuantRope{ .width = width, .start = start, .mode = @intCast(format), .frequencies = Rope.frequencyValues(compressed) };
    return dispatch("kernel_dsv41_rope_quantize_heads_tab", &.{ arg(0, &args), buf(1, input.view), buf(2, K.at(table.view, if (compressed) 256 else 0)) }, .{ .x = width / block, .y = heads });
}
export fn ds4_gpu_dsv41_norm_pair_rows(out0: ?*Tensor, x0: ?*const Tensor, offset0: u64, n0: u32, out1: ?*Tensor, x1: ?*const Tensor, offset1: u64, n1: u32, map: ?*const anyopaque, size: u64, eps: f32, rows: u32) c_int {
    const t0 = threads(n0);
    const t1 = threads(n1);
    if (n0 == 0 or n1 == 0 or n0 % 4 != 0 or n1 % 4 != 0 or rows == 0 or rows > 65535 or t0 % 32 != 0 or t1 % 32 != 0 or t0 + t1 > 1024 or !fits(x0, @as(u64, n0) * rows * 4) or !fits(out0, @as(u64, n0) * rows * 4) or !fits(x1, @as(u64, n1) * rows * 4) or !fits(out1, @as(u64, n1) * rows * 4) or c.ds4_gpu_init() == 0) return 0;
    const w0 = K.model(map, size, offset0, @as(u64, n0) * 4) orelse return 0;
    defer w0.deinit();
    const w1 = K.model(map, size, offset1, @as(u64, n1) * 4) orelse return 0;
    defer w1.deinit();
    const a = Buffer.tensor(x0);
    defer a.deinit();
    const b = Buffer.tensor(x1);
    defer b.deinit();
    const o0 = Buffer.tensor(out0);
    defer o0.deinit();
    const o1 = Buffer.tensor(out1);
    defer o1.deinit();
    const args = extern struct { n0: u32, n1: u32, ntg0: u32, ntg1: u32, eps: f32 }{ .n0 = n0, .n1 = n1, .ntg0 = t0, .ntg1 = t1, .eps = eps };
    return dispatch("kernel_dsv41_norm_pair", &.{ arg(0, &args), buf(1, a.view), buf(2, w0.view), buf(3, o0.view), buf(4, b.view), buf(5, w1.view), buf(6, o1.view) }, .{ .y = rows, .tx = t0 + t1, .shared = 256 });
}
export fn ds4_gpu_dsv41_index_keys_bf16(view: ?*Tensor, cache: ?*const Tensor, first: u32, rows: u32, flag: ?*Tensor) c_int {
    if (rows == 0) return 1;
    const end = @as(u64, first) + rows;
    if (end * 32 > std.math.maxInt(u32) or !fits(cache, end * 128 * 4) or !fits(view, end * 128 * 2) or (flag != null and !fits(flag, 4))) return 0;
    const src = Buffer.tensor(cache);
    defer src.deinit();
    const out = Buffer.tensor(view);
    defer out.deinit();
    const f = Buffer.tensor(if (flag != null) flag else view);
    defer f.deinit();
    const args = extern struct { n_vec: u32, check: u32 }{ .n_vec = rows * 32, .check = @intFromBool(flag != null) };
    return dispatch("kernel_dsv41_index_keys_bf16", &.{ arg(0, &args), buf(1, K.at(src.view, @as(u64, first) * 512)), buf(2, K.at(out.view, @as(u64, first) * 256)), buf(3, f.view) }, .{ .x = (@as(u64, rows) * 32 + 255) / 256, .tx = 256 });
}
export fn ds4_gpu_dsv41_indexer_score_one_tensor(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, heads: u32, width: u32, scale: f32, half: bool) c_int {
    const flags = c.ds4_metal_v41_test_flags();
    const wide = flags & c.DS4_GPU_TEST_INDEX_SCORE_WIDE != 0 or (flags & c.DS4_GPU_TEST_INDEX_SCORE_DIRECT == 0 and rows >= 8192 and c.getenv("DS4_METAL_DISABLE_V41_INDEX_SCORE_WIDE") == null);
    if (heads != 32 or width != 128 or !wide) return c.ds4_gpu_glm_indexer_score_one_tensor(scores, q, weights, keys, rows, heads, width, scale, half);
    if (!fits(scores, @as(u64, rows) * 4) or !fits(q, 4096 * 4) or !fits(weights, 128) or !fits(keys, @as(u64, rows) * 128 * @as(u64, if (half) 2 else 4))) return 0;
    const qs = Buffer.tensor(q);
    defer qs.deinit();
    const w = Buffer.tensor(weights);
    defer w.deinit();
    const k = Buffer.tensor(keys);
    defer k.deinit();
    const out = Buffer.tensor(scores);
    defer out.deinit();
    const args = c.ds4_metal_index_score_args{ .n_rows = rows, .n_head = 32, .head_dim = 128, .cache_f16 = @intFromBool(half), .scale = scale };
    return dispatch("kernel_glm_indexer_score_one_wide", &.{ arg(0, &args), buf(1, qs.view), buf(2, w.view), buf(3, k.view), buf(4, out.view) }, .{ .x = (@as(u64, rows) + 255) / 256, .tx = 256, .shared = (4096 + 32) * 4 });
}
export fn ds4_gpu_dsv41_indexer_score_row(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, cache: ?*const Tensor, view: ?*const Tensor, rows: u32) c_int {
    if ((rows < 8192 and c.ds4_metal_v41_test_flags() & c.DS4_GPU_TEST_INDEX_SCORE_WIDE == 0) or c.ds4_metal_v41_test_flags() & c.DS4_GPU_TEST_INDEX_SCORE_DIRECT != 0 or view == null or c.ds4_metal_v41_on(c.DS4_GPU_V41_INDEX_BF16) == 0) return ds4_gpu_dsv41_indexer_score_one_tensor(scores, q, weights, cache, rows, 32, 128, 1.0 / 64.0, false);
    if (!fits(scores, @as(u64, rows) * 4) or !fits(q, 4096 * 4) or !fits(weights, 128) or !fits(view, @as(u64, rows) * 256)) return 0;
    const qs = Buffer.tensor(q);
    defer qs.deinit();
    const w = Buffer.tensor(weights);
    defer w.deinit();
    const k = Buffer.tensor(view);
    defer k.deinit();
    const out = Buffer.tensor(scores);
    defer out.deinit();
    const args = c.ds4_metal_index_score_args{ .n_rows = rows, .n_head = 32, .head_dim = 128, .cache_f16 = 0, .scale = 1.0 / 64.0 };
    const name: [*:0]const u8 = if (rows < 10240) "kernel_dsv41_indexer_score_rows_split4_bf16" else if (rows < 65536) "kernel_dsv41_indexer_score_rows_split2_bf16" else "kernel_dsv41_indexer_score_bf16";
    const tg: u32 = if (rows < 65536) 256 else if (rows < 98304) 512 else 128;
    const keys_per_tg: u32 = if (rows < 10240) tg / 2 else if (rows < 65536) tg else 2 * tg;
    return dispatch(name, &.{ arg(0, &args), buf(1, qs.view), buf(2, w.view), buf(3, k.view), buf(4, out.view) }, .{ .x = (@as(u64, rows) + keys_per_tg - 1) / keys_per_tg, .tx = tg, .shared = (4096 + 32) * 4 });
}
export fn ds4_gpu_dsv41_indexer_all_batch(selected: ?*Tensor, rows: u32, start: u32, ratio: u32) c_int {
    if (ratio == 0 or rows == 0 or rows > std.math.maxInt(u32) - start or (start + rows) / ratio > 512 or !fits(selected, @as(u64, rows) * 512 * 4)) return 0;
    const out = Buffer.tensor(selected);
    defer out.deinit();
    const args = [4]u32{ rows, start, ratio, 512 };
    return dispatch("kernel_dsv41_indexer_all", &.{ arg(0, &args), buf(1, out.view) }, .{ .x = 2, .y = rows, .tx = 256 });
}
export fn ds4_gpu_dsv41_candidate_topk_batch(selected: ?*Tensor, blocks: ?*const Tensor, width: u32, rows: u32, start: u32) c_int {
    if (rows == 0 or start > std.math.maxInt(u32) - 8 or rows > std.math.maxInt(u32) - 8 - start or width > std.math.maxInt(i32) or rows > std.math.maxInt(i32) or (start + rows + 7) / 8 > width or (start + 8) / 8 <= 2048) return 0;
    return @import("index_selection.zig").topk(selected, blocks, width, rows, 2048, start + 7, 8);
}
export fn ds4_gpu_dsv41_candidate_mask_batch(mask: ?*Tensor, selected: ?*const Tensor, width: u32, rows: u32, stride: u32) c_int {
    if (width == 0 or rows == 0 or width < 2048 or stride < width or !fits(mask, (@as(u64, stride) * (rows - 1) + width) * 4) or !fits(selected, @as(u64, rows) * 2048 * 4)) return 0;
    // The public mask primitive infers a dense row stride. Views keep each
    // row's advertised width while the parent retains padded storage.
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        const out = c.ds4_gpu_tensor_view(mask, @as(u64, row) * stride * 4, @as(u64, width) * 4) orelse return 0;
        defer c.ds4_gpu_tensor_free(out);
        const ids = c.ds4_gpu_tensor_view(selected, @as(u64, row) * 2048 * 4, 2048 * 4) orelse return 0;
        defer c.ds4_gpu_tensor_free(ids);
        if (c.ds4_gpu_dsv4_topk_mask_tensor(out, ids, width, 1, 2048) == 0) return 0;
    }
    return 1;
}
export fn ds4_gpu_dsv41_hc_project(mix: ?*Tensor, residual: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, n: u32, mix_dim: u32, eps: f32) c_int {
    return c.ds4_gpu_hc_rms_norm_mix_f16_tensor(mix, residual, map, size, offset, n, mix_dim, eps);
}
fn splitArgs(width: u32, hc: u32, iters: u32, hc_eps: f32, norm_eps: f32) c.ds4_gpu_hc_split_weighted_sum_norm_args {
    const row = @as(u64, width) * 4;
    const mix = @as(u64, 2 * hc + hc * hc) * 4;
    return .{ .n_embd = width, .n_hc = @intCast(hc), .sinkhorn_iters = @intCast(iters), .n_rows = 1, .mix_hc = 2 * hc + hc * hc, .nb_mix1 = mix, .nb_split1 = mix, .nb_x0 = 4, .nb_x1 = row, .nb_x2 = @as(u64, hc) * row, .nb0 = 4, .nb1 = row, .nb_norm1 = row, .eps = hc_eps, .norm_eps = norm_eps };
}
export fn ds4_gpu_dsv41_hc_block_input_rows(mix: ?*Tensor, x: ?*Tensor, norm: ?*Tensor, split: ?*Tensor, stream: ?*const Tensor, pre: ?*const Tensor, map: ?*const anyopaque, size: u64, fn_offset: u64, scale_offset: u64, base_offset: u64, norm_offset: u64, n: u32, mix_dim: u32, width: u32, hc: u32, iters: u32, hc_eps: f32, norm_eps: f32, rows: u32, pre_stride: u32) c_int {
    if (n < 4096 or n % 4096 != 0 or mix_dim != 24 or hc != 4 or width % 4 != 0 or n != hc * width or rows == 0 or rows > 8 or pre_stride < hc or !fits(stream, @as(u64, n) * rows * 4) or !fits(mix, @as(u64, 24) * rows * 4) or !fits(split, @as(u64, 24) * rows * 4) or !fits(x, @as(u64, width) * rows * 4) or !fits(norm, @as(u64, width) * rows * 4) or !fits(pre, (@as(u64, rows - 1) * pre_stride + hc) * 4) or c.ds4_gpu_init() == 0) return 0;
    const fnw = K.model(map, size, fn_offset, @as(u64, n) * 2 * mix_dim) orelse return 0;
    defer fnw.deinit();
    const sw = K.model(map, size, scale_offset, 12) orelse return 0;
    defer sw.deinit();
    const bw = K.model(map, size, base_offset, 96) orelse return 0;
    defer bw.deinit();
    const nw = K.model(map, size, norm_offset, @as(u64, width) * 4) orelse return 0;
    defer nw.deinit();
    const sr = Buffer.tensor(stream);
    defer sr.deinit();
    const pr = Buffer.tensor(pre);
    defer pr.deinit();
    const mx = Buffer.tensor(mix);
    defer mx.deinit();
    const sp = Buffer.tensor(split);
    defer sp.deinit();
    const out = Buffer.tensor(x);
    defer out.deinit();
    const no = Buffer.tensor(norm);
    defer no.deinit();
    const completion = Buffer{ .view = c.ds4_metal_v41_completion("v41_block_input", 8 * 64 * 4) };
    defer completion.deinit();
    if (!completion.valid()) return 0;
    const args = c.ds4_gpu_hc_norm_mix_args{ .n = @intCast(n), .out_dim = @intCast(mix_dim), .eps = norm_eps };
    const split_args = splitArgs(width, hc, iters, hc_eps, norm_eps);
    return dispatch("kernel_dsv41_hc_block_input", &.{ arg(0, &args), arg(1, &split_args), buf(2, sr.view), buf(3, fnw.view), buf(4, mx.view), buf(5, sw.view), buf(6, bw.view), buf(7, sp.view), buf(8, out.view), buf(9, nw.view), buf(10, no.view), buf(11, pr.view), buf(12, completion.view), arg(13, &pre_stride) }, .{ .x = (mix_dim + 1) / 2 + 4, .y = rows, .tx = 32, .ty = 8, .shared = 96 * 4 });
}
export fn ds4_gpu_dsv41_hc_input(x: ?*Tensor, norm: ?*Tensor, split: ?*Tensor, mix: ?*const Tensor, residual: ?*const Tensor, pre: ?*const Tensor, map: ?*const anyopaque, size: u64, scale_offset: u64, base_offset: u64, norm_offset: u64, width: u32, hc: u32, iters: u32, hc_eps: f32, norm_eps: f32) c_int {
    if (width == 0 or width % 4 != 0 or hc != 4 or !fits(x, @as(u64, width) * 4) or !fits(norm, @as(u64, width) * 4) or !fits(split, 96) or !fits(mix, 96) or !fits(residual, @as(u64, width) * hc * 4) or !fits(pre, hc * 4) or c.ds4_gpu_init() == 0) return 0;
    const sw = K.model(map, size, scale_offset, 12) orelse return 0;
    defer sw.deinit();
    const bw = K.model(map, size, base_offset, 96) orelse return 0;
    defer bw.deinit();
    const nw = K.model(map, size, norm_offset, @as(u64, width) * 4) orelse return 0;
    defer nw.deinit();
    const mx = Buffer.tensor(mix);
    defer mx.deinit();
    const rs = Buffer.tensor(residual);
    defer rs.deinit();
    const pr = Buffer.tensor(pre);
    defer pr.deinit();
    const out = Buffer.tensor(x);
    defer out.deinit();
    const no = Buffer.tensor(norm);
    defer no.deinit();
    const sp = Buffer.tensor(split);
    defer sp.deinit();
    const args = splitArgs(width, hc, iters, hc_eps, norm_eps);
    return dispatch("kernel_dsv41_hc_split_sum_norm", &.{ arg(0, &args), buf(1, mx.view), buf(2, sw.view), buf(3, bw.view), buf(4, rs.view), buf(5, sp.view), buf(6, out.view), buf(7, nw.view), buf(8, no.view), buf(9, pr.view) }, .{ .tx = threads(width), .shared = (width + 4 + 32) * 4 });
}
const DeferredArgs = extern struct { n_embd: i32 = 5120, n_hc: i32 = 4, iters: i32 = 0, hc_eps: f32 = 0, norm_eps: f32 };
export fn ds4_gpu_dsv41_deferred_collapse(x: ?*Tensor, norm: ?*Tensor, residual: ?*const Tensor, pre: ?*const Tensor, map: ?*const anyopaque, size: u64, offset: u64, eps: f32) c_int {
    if (c.ds4_metal_v41_on(c.DS4_GPU_V41_DEFERRED_HC) == 0 or c.ds4_metal_v41_dense_decode_allowed() == 0 or !fits(x, 5120 * 4) or !fits(norm, 5120 * 4) or !fits(residual, 20480 * 4) or !fits(pre, 16) or c.ds4_gpu_init() == 0) return 0;
    const weight = K.model(map, size, offset, 5120 * 4) orelse return 0;
    defer weight.deinit();
    const out = Buffer.tensor(x);
    defer out.deinit();
    const no = Buffer.tensor(norm);
    defer no.deinit();
    const rs = Buffer.tensor(residual);
    defer rs.deinit();
    const pr = Buffer.tensor(pre);
    defer pr.deinit();
    const count = Buffer{ .view = c.ds4_metal_v41_completion("v41_deferred_hc", 256) };
    defer count.deinit();
    if (!count.valid()) return 0;
    const args = DeferredArgs{ .norm_eps = eps };
    if (c.ds4_metal_v41_on(c.DS4_GPU_V41_SPLIT_REDUCE) != 0) {
        c.ds4_metal_v41_note("kernel_dsv41_deferred_collapse");
        const slots = Buffer{ .view = c.ds4_metal_v41_completion("v41_collapse_slots", 128) };
        defer slots.deinit();
        if (!slots.valid()) return 0;
        // Both dispatches share the caller's command buffer. The barrier is
        // required when that buffer borrows a concurrent encoder.
        var command: K.Command = .{};
        defer command.drop();
        if (!command.acquire()) return 0;
        const part_bindings = [_]Binding{ arg(0, &args), buf(1, rs.view), buf(2, pr.view), buf(3, out.view), buf(4, slots.view) };
        const part = c.ds4_metal_v41_launch{ .kernel = "kernel_dsv41_deferred_collapse_part", .nsg = 0, .count = part_bindings.len, .shared_bytes = 0, .bindings = &part_bindings, .gx = 32, .gy = 1, .gz = 1, .tx = 32, .ty = 1, .barrier = true };
        if (c.ds4_metal_v41_encode(command.token, &part) == 0) return 0;
        const final_bindings = [_]Binding{ arg(0, &args), buf(1, out.view), buf(2, slots.view), buf(3, weight.view), buf(4, no.view), buf(5, count.view) };
        const final = c.ds4_metal_v41_launch{ .kernel = "kernel_dsv41_deferred_collapse_norm", .nsg = 0, .count = final_bindings.len, .shared_bytes = 0, .bindings = &final_bindings, .gx = 10, .gy = 1, .gz = 1, .tx = 128, .ty = 1, .barrier = false };
        return @intFromBool(c.ds4_metal_v41_encode(command.token, &final) != 0 and command.finish("V4.1 independent HC collapse"));
    }
    return dispatch("kernel_dsv41_deferred_collapse", &.{ arg(0, &args), buf(1, rs.view), buf(2, pr.view), buf(3, out.view), buf(4, weight.view), buf(5, no.view), buf(6, count.view) }, .{ .tx = 1024, .shared = (5120 + 32) * 4 });
}
export fn ds4_gpu_dsv41_query_deferred_hc(q: ?*Tensor, mix: ?*Tensor, split: ?*Tensor, qr: ?*const Tensor, residual: ?*const Tensor, map: ?*const anyopaque, size: u64, q_offset: u64, fn_offset: u64, scale_offset: u64, base_offset: u64, pos: u32, compressed: bool, iters: u32, hc_eps: f32, norm_eps: f32, x: ?*Tensor, norm: ?*Tensor, pre: ?*const Tensor, norm_offset: u64, kv: ?*const Tensor, cache: ?*Tensor, cache_offset: u64) c_int {
    if (pos >= 1048576 or !fits(x, 5120 * 4) or !fits(norm, 5120 * 4) or !fits(pre, 16) or !fits(q, 32768 * 4) or !fits(qr, 1280 * 4) or !fits(mix, 96) or !fits(split, 96) or !fits(residual, 20480 * 4) or !fits(kv, 512 * 4) or cache_offset % 4 != 0 or cache_offset > c.ds4_gpu_tensor_bytes(cache) or !fits(cache, cache_offset + 512 * 4) or c.ds4_gpu_init() == 0) return 0;
    const qw = K.model(map, size, q_offset, 32768 * 40 * 34) orelse return 0;
    defer qw.deinit();
    const fw = K.model(map, size, fn_offset, 20480 * 24 * 2) orelse return 0;
    defer fw.deinit();
    const sw = K.model(map, size, scale_offset, 12) orelse return 0;
    defer sw.deinit();
    const bw = K.model(map, size, base_offset, 96) orelse return 0;
    defer bw.deinit();
    const nw = K.model(map, size, norm_offset, 5120 * 4) orelse return 0;
    defer nw.deinit();
    const qbuf = Buffer.tensor(q);
    defer qbuf.deinit();
    const mx = Buffer.tensor(mix);
    defer mx.deinit();
    const sp = Buffer.tensor(split);
    defer sp.deinit();
    const qrbuf = Buffer.tensor(qr);
    defer qrbuf.deinit();
    const rs = Buffer.tensor(residual);
    defer rs.deinit();
    const out = Buffer.tensor(x);
    defer out.deinit();
    const no = Buffer.tensor(norm);
    defer no.deinit();
    const pr = Buffer.tensor(pre);
    defer pr.deinit();
    const kb = Buffer.tensor(kv);
    defer kb.deinit();
    const cb = Buffer.tensor(cache);
    defer cb.deinit();
    const count = Buffer{ .view = c.ds4_metal_v41_completion("v41_deferred_hc", 256) };
    defer count.deinit();
    if (!count.valid()) return 0;
    const hc = c.ds4_gpu_hc_norm_mix_args{ .n = 20480, .out_dim = 24, .eps = norm_eps };
    const split_args = splitArgs(5120, 4, iters, hc_eps, norm_eps);
    const mv = Dense.mvArgs(1280, 32768, 1, true, 8);
    const rope = mvRope(512, pos, compressed, false);
    const kvargs = QuantRope{ .width = 512, .start = pos, .mode = 1, .frequencies = rope.frequencies };
    return dispatch("kernel_dsv41_query_deferred_hc", &.{ arg(0, &hc), arg(1, &split_args), buf(2, rs.view), buf(3, fw.view), buf(4, mx.view), buf(5, sw.view), buf(6, bw.view), buf(7, sp.view), buf(8, count.view), arg(9, &mv), buf(10, qw.view), buf(11, qrbuf.view), buf(12, qbuf.view), arg(13, &rope), buf(14, out.view), buf(15, nw.view), buf(16, no.view), buf(17, pr.view), arg(18, &kvargs), buf(19, kb.view), buf(20, K.at(cb.view, cache_offset)) }, .{ .x = 16 + 2 + 4096, .tx = 32, .ty = 8, .shared = 96 * 4 });
}
export fn ds4_gpu_dsv41_project_q(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, input: ?*const Tensor, pos: u32, compressed: bool) c_int {
    return projectQ(out, map, size, offset, width, outputs, input, pos, compressed, null);
}
export fn ds4_gpu_dsv41_project_q_tab(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, input: ?*const Tensor, pos: u32, compressed: bool, tab: ?*const Tensor) c_int {
    return projectQ(out, map, size, offset, width, outputs, input, pos, compressed, tab);
}
fn projectQ(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, input: ?*const Tensor, pos: u32, compressed: bool, tab: ?*const Tensor) c_int {
    if (width == 0 or width % 32 != 0 or outputs == 0 or outputs % 512 != 0 or outputs > 65536 or pos >= 1048576 or !fits(input, @as(u64, width) * 4) or !fits(out, @as(u64, outputs) * 4) or c.ds4_gpu_init() == 0) return 0;
    const weight = K.model(map, size, offset, @as(u64, width / 32) * 34 * outputs) orelse return 0;
    defer weight.deinit();
    const src = Buffer.tensor(input);
    defer src.deinit();
    const dst = Buffer.tensor(out);
    defer dst.deinit();
    const mv = Dense.mvArgs(width, outputs, 1, true, 2);
    const rope = mvRope(512, pos, compressed, false);
    const short_q8 = if (c.getenv("DS4_METAL_V41_SHORT_Q8_QUERY")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
    // The helper preserves the default NSG=4 reduction. Any explicit NSG
    // override or non-single-rank execution keeps the generic dispatch.
    if (short_q8 and width == 1280 and outputs == 32768 and
        c.getenv("DS4_METAL_Q8_MV_NSG") == null and
        c.ds4_metal_attention_get_facts().flags & c.DS4_ATTN_TP_SINGLE != 0)
    {
        if (fits(tab, 512)) {
            const table = Buffer.tensor(tab);
            defer table.deinit();
            return dispatch("kernel_dsv41_query_short_q8_tab", &.{ arg(0, &mv), buf(1, weight.view), buf(2, src.view), buf(3, dst.view), arg(4, &rope), buf(5, K.at(table.view, if (compressed) 256 else 0)) }, .{ .x = 2048, .tx = 32, .ty = 8 });
        }
        return dispatch("kernel_dsv41_query_short_q8", &.{ arg(0, &mv), buf(1, weight.view), buf(2, src.view), buf(3, dst.view), arg(4, &rope) }, .{ .x = 2048, .tx = 32, .ty = 8 });
    }
    const nsg: u32 = @intCast(c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", if (c.ds4_metal_dense_facts(false) & c.DS4_DENSE_TP_TWO != 0) 2 else 4, 1, 8));
    return dispatch("kernel_mul_mv_q8_0_f32_bf16_rope", &.{ arg(0, &mv), buf(1, weight.view), buf(2, src.view), buf(3, dst.view), arg(4, &rope) }, .{ .x = (@as(u64, outputs) + 1) / 2, .tx = 32, .ty = nsg, .nsg = @intCast(nsg), .shared = 256 });
}
export fn ds4_gpu_dsv41_matmul_expand_rows(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, outputs: u32, input: ?*const Tensor, add: ?*const Tensor, residual: ?*const Tensor, split: ?*const Tensor, hc: u32, rows: u32) c_int {
    if (hc != 4 or width == 0 or width % 32 != 0 or outputs == 0 or outputs > 65536 or rows == 0 or rows > 8 or !fits(input, @as(u64, rows) * width * 4) or (add != null and !fits(add, @as(u64, rows) * outputs * 4)) or !fits(residual, @as(u64, rows) * hc * outputs * 4) or !fits(out, @as(u64, rows) * hc * outputs * 4) or !fits(split, @as(u64, rows) * 24 * 4) or c.ds4_gpu_init() == 0) return 0;
    const weight = K.model(map, size, offset, @as(u64, width / 32) * 34 * outputs) orelse return 0;
    defer weight.deinit();
    const src = Buffer.tensor(input);
    defer src.deinit();
    const dst = Buffer.tensor(out);
    defer dst.deinit();
    const ad = Buffer.tensor(if (add != null) add else input);
    defer ad.deinit();
    const rs = Buffer.tensor(residual);
    defer rs.deinit();
    const sp = Buffer.tensor(split);
    defer sp.deinit();
    const nr: u32 = if (rows > 1) 4 else 2;
    const mv = Dense.mvArgs(width, outputs, rows, true, @intCast(nr));
    const args = extern struct { n_embd: u32, has_add: u32, split_stride: u32 }{ .n_embd = outputs, .has_add = @intFromBool(add != null), .split_stride = 24 };
    const nsg: u32 = @intCast(c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", if (c.ds4_metal_dense_facts(false) & c.DS4_DENSE_TP_TWO != 0) 2 else 4, 1, 8));
    const literal = rows == 1 and width == 8192 and outputs == 5120 and add == null and c.ds4_metal_v41_on(c.DS4_GPU_V41_LITERAL_SHAPES) != 0;
    const name: [*:0]const u8 = if (rows > 1) "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_rows_seq4" else if (literal) "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_static" else "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4";
    return dispatch(name, &.{ arg(0, &mv), buf(1, weight.view), buf(2, src.view), buf(3, dst.view), arg(4, &args), buf(5, ad.view), buf(6, rs.view), buf(7, K.at(sp.view, 16)), buf(8, K.at(sp.view, 32)) }, .{ .x = (@as(u64, outputs) + nr - 1) / nr, .tx = 32, .ty = nsg, .nsg = @intCast(nsg), .shared = if (rows > 1) 512 else 256 });
}
export fn ds4_gpu_dsv41_shared_swiglu(mid: ?*Tensor, gate: ?*Tensor, up: ?*Tensor, map: ?*const anyopaque, size: u64, gate_offset: u64, up_offset: u64, width: u32, outputs: u32, x: ?*const Tensor, clamp: f32, rows: u32) c_int {
    return @import("router_shared.zig").shared(.{ .gate = gate, .up = up, .mid = mid, .map = map, .size = size, .gate_offset = gate_offset, .up_offset = up_offset, .width = width, .outputs = outputs, .x = x, .rows = rows, .clamp = clamp, .mode = .bf16_mid });
}
fn lowArgs(width: u32, rank: u32, groups: u32) c.ds4_gpu_mul_mv_id_args {
    var args = std.mem.zeroes(c.ds4_gpu_mul_mv_id_args);
    const row = @as(u64, width / 32) * 34;
    args.nei0 = @intCast(groups);
    args.nei1 = 1;
    args.ne00 = @intCast(width);
    args.ne01 = @intCast(rank);
    args.ne02 = @intCast(groups);
    args.nb00 = 34;
    args.nb01 = row;
    args.nb02 = @as(u64, rank) * row;
    args.ne10 = @intCast(width);
    args.ne11 = @intCast(groups);
    args.ne12 = 1;
    args.ne13 = 1;
    args.nb10 = 4;
    args.nb11 = @as(u64, width) * 4;
    args.nb12 = @as(u64, groups) * width * 4;
    args.ne0 = @intCast(rank);
    args.ne1 = @intCast(groups);
    args.nb1 = @as(u64, rank) * 4;
    args.nr0 = 2;
    return args;
}
fn lowProject(low: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rank: u32, groups: u32, heads: ?*const Tensor, pos: u32, compressed: bool, rotate: bool) c_int {
    if (width == 0 or width % 512 != 0 or rank == 0 or groups == 0 or @as(u64, groups) * rank > std.math.maxInt(u32) or !fits(low, @as(u64, groups) * rank * 4) or !fits(heads, @as(u64, groups) * width * 4) or pos >= 1048576 or c.ds4_gpu_init() == 0) return 0;
    const weight = K.model(map, size, offset, @as(u64, groups) * rank * (width / 32) * 34) orelse return 0;
    defer weight.deinit();
    const src = Buffer.tensor(heads);
    defer src.deinit();
    const dst = Buffer.tensor(low);
    defer dst.deinit();
    const args = lowArgs(width, rank, groups);
    const rope = mvRope(512, pos, compressed, true);
    const literal = !rotate and width == 4096 and rank == 1024 and groups == 8 and c.ds4_metal_v41_on(c.DS4_GPU_V41_LITERAL_SHAPES) != 0;
    const name: [*:0]const u8 = if (rotate) "kernel_dsv4_attn_out_low_q8_0_f32_rope" else if (literal) "kernel_dsv4_attn_out_low_q8_0_f32_static" else "kernel_dsv4_attn_out_low_q8_0_f32";
    var bindings = [_]Binding{ arg(0, &args), buf(1, weight.view), buf(2, src.view), buf(3, dst.view), arg(4, &rope) };
    return dispatch(name, bindings[0..if (rotate) 5 else 4], .{ .x = (@as(u64, rank) + 1) / 2, .z = groups, .tx = 32, .ty = 4, .nsg = 4, .shared = 256 });
}
export fn ds4_gpu_dsv41_attention_low(low: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rank: u32, groups: u32, heads: ?*Tensor, pos: u32, compressed: bool) c_int {
    return lowProject(low, map, size, offset, width, rank, groups, heads, pos, compressed, true);
}
export fn ds4_gpu_dsv41_attention_low_plain(low: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u32, rank: u32, groups: u32, heads: ?*const Tensor) c_int {
    return lowProject(low, map, size, offset, width, rank, groups, heads, 0, false, false);
}
const RadixArgs = extern struct { n: u32, rows: u32, k: u32, pass: u32, parity: u32, n_tg: u32 };
fn encodeOn(command: *K.Command, name: [*:0]const u8, bindings: []const Binding, grid: Grid) bool {
    const launch = c.ds4_metal_v41_launch{ .kernel = name, .nsg = grid.nsg, .count = @intCast(bindings.len), .shared_bytes = grid.shared, .bindings = bindings.ptr, .gx = grid.x, .gy = grid.y, .gz = grid.z, .tx = grid.tx, .ty = grid.ty, .barrier = grid.barrier };
    return c.ds4_metal_v41_encode(command.token, &launch) != 0;
}
fn radixTopk(selected: ?*Tensor, scores: ?*const Tensor, n: u32, rows: u32, k: u32, mapping: ?*const Tensor) c_int {
    if (c.ds4_gpu_init() == 0 or !fits(selected, @as(u64, k) * rows * 4) or !fits(scores, @as(u64, n) * rows * 4)) return 0;
    const out = Buffer.tensor(selected);
    defer out.deinit();
    const src = Buffer.tensor(scores);
    defer src.deinit();
    const map = Buffer.tensor(mapping);
    defer map.deinit();
    const state_offset = @as(u64, rows) * 4 * 2048 * 4;
    const counts_offset = state_offset + @as(u64, rows) * 4 * 4;
    const bytes = counts_offset + @as(u64, rows) * 4096 * 2 * 4;
    const scratch = Buffer{ .view = c.ds4_metal_v41_completion("v41_topk_select", bytes) };
    defer scratch.deinit();
    if (!scratch.valid()) return 0;
    const state: *c.ds4_metal_v41_topk_state = c.ds4_metal_v41_topk_state_get();
    var args = RadixArgs{ .n = n, .rows = rows, .k = k, .pass = 0, .parity = state.parity, .n_tg = (n + 2047) / 2048 };
    state.parity ^= 1;
    var command: K.Command = .{};
    defer command.drop();
    if (!command.acquire()) return 0;
    if (state.rows != rows) {
        const words: u32 = @intCast(scratch.view.bytes / 4);
        if (!encodeOn(&command, "kernel_topk_select_zero", &.{ buf(0, scratch.view), arg(1, &words) }, .{ .x = (@as(u64, words) + 255) / 256, .tx = 256 })) return 0;
        state.rows = rows;
    }
    while (args.pass < 3) : (args.pass += 1) {
        if (!encodeOn(&command, "kernel_topk_select_hist", &.{ arg(0, &args), buf(1, src.view), buf(2, scratch.view) }, .{ .x = args.n_tg, .y = rows, .tx = 1024, .shared = 2048 * 4 })) return 0;
    }
    if (!encodeOn(&command, "kernel_topk_select_count", &.{ arg(0, &args), buf(1, src.view), buf(2, scratch.view), buf(3, K.at(scratch.view, state_offset)), buf(4, K.at(scratch.view, counts_offset)) }, .{ .x = args.n_tg, .y = rows, .tx = 1024 })) return 0;
    if (!encodeOn(&command, "kernel_topk_select_write", &.{ arg(0, &args), buf(1, src.view), buf(2, K.at(scratch.view, state_offset)), buf(3, K.at(scratch.view, counts_offset)), buf(4, out.view) }, .{ .x = args.n_tg, .y = rows, .tx = 1024 })) return 0;
    if (k <= 512 and c.ds4_metal_v41_on(c.DS4_GPU_V41_DECODE_TOPK) != 0) {
        const mapped: u32 = @intFromBool(mapping != null);
        if (!encodeOn(&command, "kernel_topk_select_sort512", &.{ arg(0, &args), buf(1, src.view), buf(2, out.view), buf(3, if (mapping != null) map.view else src.view), arg(4, &mapped) }, .{ .x = rows, .tx = 512 })) return 0;
    } else {
        if (!encodeOn(&command, "kernel_topk_select_sort", &.{ arg(0, &args), buf(1, src.view), buf(2, out.view) }, .{ .x = rows, .tx = 1024, .shared = 2 * 2048 * 4 })) return 0;
    }
    return @intFromBool(command.finish("V4.1 radix indexer top-k"));
}
export fn ds4_gpu_dsv41_indexer_topk_tensor(selected: ?*Tensor, scores: ?*const Tensor, width: u32, rows: u32, k: u32) c_int {
    if (width >= 16384 and width <= (1 << 23) and k > 0 and k <= 2048 and c.getenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == null) return radixTopk(selected, scores, width, rows, k, null);
    return @import("index_selection.zig").topkDecode(selected, scores, width, rows, k, 0, 0, true);
}
export fn ds4_gpu_dsv41_indexer_candidates_tensor(selected: ?*Tensor, scores: ?*Tensor, list: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, mask: ?*const Tensor, n: u32, scale: f32, k: u32, list_ready: c_int) c_int {
    if (n > (1 << 23)) return 0;
    const blocks = (n + 7) / 8;
    const count = 16384 - (blocks * 8 - n);
    if (c.getenv("DS4_METAL_DISABLE_V41_INDEX_CANDIDATES") != null or blocks <= 2048 or k == 0 or k > 2048 or c.getenv("DS4_METAL_DISABLE_V41_INDEX_SCORE_WIDE") != null or c.getenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") != null or c.ds4_metal_v41_test_flags() & c.DS4_GPU_TEST_INDEX_SCORE_DIRECT != 0 or !fits(selected, @as(u64, k) * 4) or !fits(scores, @as(u64, count) * 4) or !fits(list, 2048 * 4) or !fits(mask, @as(u64, blocks) * 4) or !fits(q, 4096 * 4) or !fits(weights, 128) or !fits(keys, @as(u64, n) * 128 * 4)) return 0;
    const lb = Buffer.tensor(list);
    defer lb.deinit();
    const mb = Buffer.tensor(mask);
    defer mb.deinit();
    const qs = Buffer.tensor(q);
    defer qs.deinit();
    const wb = Buffer.tensor(weights);
    defer wb.deinit();
    const kb = Buffer.tensor(keys);
    defer kb.deinit();
    const out = Buffer.tensor(scores);
    defer out.deinit();
    const ids = Buffer.tensor(selected);
    defer ids.deinit();
    const once = c.ds4_metal_v41_on(c.DS4_GPU_V41_DECODE_TOPK) != 0;
    if (once and list_ready != 0) c.ds4_metal_v41_note("candidate list reused") else if (dispatch("kernel_dsv41_candidate_list", &.{ arg(0, &blocks), buf(1, mb.view), buf(2, lb.view) }, .{ .tx = 1024 }) == 0) return 0;
    const args = c.ds4_metal_index_score_args{ .n_rows = count, .n_head = 32, .head_dim = 128, .cache_f16 = 0, .scale = scale };
    if (dispatch("kernel_dsv41_indexer_score_candidates", &.{ arg(0, &args), buf(1, qs.view), buf(2, wb.view), buf(3, kb.view), buf(4, out.view), buf(5, lb.view) }, .{ .x = (@as(u64, count) + 255) / 256, .tx = 256, .shared = (4096 + 32) * 4 }) == 0) return 0;
    const mapped = once and k <= 512;
    if (radixTopk(selected, scores, count, 1, k, if (mapped) list else null) == 0) return 0;
    if (mapped) return 1;
    return dispatch("kernel_dsv41_candidate_ids", &.{ arg(0, &k), buf(1, lb.view), buf(2, ids.view) }, .{ .x = (@as(u64, k) + 255) / 256, .tx = 256 });
}
