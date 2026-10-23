//! Compressed-attention index scoring and selection. Native code owns shapes,
//! causal bounds, packet construction and the top-k scratch ping-pong plan.
//! Cached PSOs, Metal scratch buffers and command/ARC messages stay in leaves.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Tensor = c.ds4_gpu_tensor;
const max_u32 = std.math.maxInt(u32);
const max_i32 = std.math.maxInt(i32);

fn fits(tensor: ?*const Tensor, bytes: u64) bool {
    return api.tensorHasBuffer(tensor) != 0 and api.tensorBytes(tensor) >= bytes;
}
fn rmsThreads(width: u32) u64 {
    const float4_width = width / 4;
    var count: u64 = 32;
    while (count < float4_width and count < 1024) count *= 2;
    return @max(@min(count, float4_width), 1);
}
fn bits(value: u32) i32 {
    return @bitCast(value);
}
fn unsigned(value: i32) u64 {
    return @bitCast(@as(i64, value));
}
fn scoreDispatch(launch: *const c.ds4_metal_index_score_launch, label: [*:0]const u8) c_int {
    if (launch.pipeline.handle == null) return 0;
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    return api.release(command, owned, c.ds4_metal_index_score_encode(command, launch), label);
}

pub fn scoreOne(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, heads: u32, width: u32, scale: f32, half: bool) c_int {
    if (api.init() == 0) return 0;
    if (scores == null or q == null or weights == null or keys == null or rows == 0 or
        heads == 0 or width == 0 or !std.math.isFinite(scale) or scale <= 0) return 0;
    const element: u64 = if (half) 2 else 4;
    if (!fits(scores, @as(u64, rows) * 4) or !fits(q, @as(u64, heads) *% width *% 4) or
        !fits(weights, @as(u64, heads) * 4) or !fits(keys, @as(u64, rows) *% width *% element))
        return api.reportError("ds4: Metal GLM indexer score received undersized buffers");
    const direct = heads == 32 and width == 128;
    const threads = if (direct) 32 else rmsThreads(width);
    var args = std.mem.zeroes(c.ds4_metal_index_score_packet);
    args.one = .{ .n_rows = rows, .n_head = heads, .head_dim = width, .cache_f16 = @intFromBool(half), .scale = scale };
    const launch = c.ds4_metal_index_score_launch{
        .pipeline = c.ds4_metal_index_get_pipeline(if (direct) c.DS4_METAL_INDEX_DIRECT else c.DS4_METAL_INDEX_SCORE),
        .batch = false,
        .args = args,
        .q = q,
        .weights = weights,
        .keys = keys,
        .scores = scores,
        .groups_x = rows,
        .groups_y = 1,
        .threads_x = threads,
        .threads_y = if (direct) 4 else 1,
        .shared_bytes = if (direct) (128 + 4) * 4 else threads * 4,
    };
    return scoreDispatch(&launch, if (direct) "GLM indexer direct score" else "GLM indexer score");
}

pub fn scoreBatch(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, rows: u32, tokens: u32, start: u32, ratio: u32, heads: u32, width: u32, scale: f32, half: bool, tiled_f32: bool) c_int {
    if (api.init() == 0) return 0;
    if (scores == null or q == null or weights == null or keys == null or rows == 0 or
        tokens == 0 or ratio == 0 or heads == 0 or width != 128 or start > max_u32 - tokens or
        (start + tokens) / ratio > rows or !std.math.isFinite(scale) or scale <= 0) return 0;
    const element: u64 = if (half) 2 else 4;
    if (!fits(scores, @as(u64, rows) *% tokens *% 4) or !fits(q, @as(u64, tokens) *% heads *% width *% 4) or
        !fits(weights, @as(u64, tokens) *% heads *% 4) or !fits(keys, @as(u64, rows) *% width *% element))
        return api.reportError("ds4: Metal GLM indexer batch scores received undersized buffers");
    const tiled = !c.ds4_metal_index_get_facts().quality and tokens >= 8 and heads == 32;
    const pipeline: u32 = if (tiled) (if (tiled_f32) c.DS4_METAL_INDEX_TILED_F32 else c.DS4_METAL_INDEX_TILED) else c.DS4_METAL_INDEX_BATCH;
    var args = std.mem.zeroes(c.ds4_metal_index_score_packet);
    args.batch = .{
        .n_rows = rows,
        .n_tokens = tokens,
        .n_head = heads,
        .head_dim = width,
        .pos0 = start,
        .cache_f16 = @intFromBool(half),
        .row_group_size = ratio,
        .pad0 = 0,
        .q_token_stride = @as(u64, heads) * width * 4,
        .q_head_stride = @as(u64, width) * 4,
        .weights_token_stride = @as(u64, heads) * 4,
        .score_token_stride = @as(u64, rows) * 4,
        .scale = scale,
    };
    const threads = rmsThreads(width);
    // The tiled path stages8 queries,32 keys and their8x32 dot products.
    const q_shared = 8 * 128;
    const k_shared = 32 * 128;
    const dots = 8 * 32;
    const tiled_shared: u64 = if (tiled_f32) (q_shared + k_shared + dots) * 4 else (q_shared + k_shared) * 2 + dots * 4;
    const launch = c.ds4_metal_index_score_launch{
        .pipeline = c.ds4_metal_index_get_pipeline(pipeline),
        .batch = true,
        .args = args,
        .q = q,
        .weights = weights,
        .keys = keys,
        .scores = scores,
        .groups_x = if (tiled) (@as(u64, rows) + 31) / 32 else rows,
        .groups_y = if (tiled) (@as(u64, tokens) + 7) / 8 else tokens,
        .threads_x = if (tiled) 32 else threads,
        .threads_y = if (tiled) 4 else 1,
        .shared_bytes = if (tiled) tiled_shared else threads * 4,
    };
    return scoreDispatch(&launch, "GLM indexer batch scores");
}

fn flagsBytes(source_rows: u32, rows: u32) u64 {
    return ((@as(u64, rows) + (@as(u64, source_rows) + 63) / 64) * 4 + 255) & ~@as(u64, 255);
}
pub fn packedBytes(source_rows: u32, rows: u32) u64 {
    return flagsBytes(source_rows, rows) + @as(u64, rows) * 32 * 128 * 2 +
        ((@as(u64, source_rows) + 63) / 64) * 64 * 128 * 2;
}
fn packDispatch(launch: *const c.ds4_metal_index_pack_launch, label: [*:0]const u8) c_int {
    if (launch.pipeline.handle == null) return 0;
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    return api.release(command, owned, c.ds4_metal_index_pack_encode(command, launch), label);
}
pub fn pack(packed_tensor: ?*Tensor, q: ?*const Tensor, keys: ?*const Tensor, source_rows: u32, rows: u32) c_int {
    if (source_rows == 0 or source_rows > max_i32 or rows == 0 or rows > max_i32 or
        c.ds4_gpu_dsv41_tensor_ops_available() == 0) return 0;
    if (!fits(q, @as(u64, rows) * 32 * 128 * 4) or !fits(keys, @as(u64, source_rows) * 128 * 4) or
        !fits(packed_tensor, packedBytes(source_rows, rows))) return 0;
    const flags = flagsBytes(source_rows, rows);
    const launch = c.ds4_metal_index_pack_launch{
        .pipeline = c.ds4_metal_index_get_pipeline(c.DS4_METAL_INDEX_PACK),
        .scores = false,
        .args = .{ source_rows, rows, 0, 0 },
        .range = .{ 0, 0 },
        .q = q,
        .keys = keys,
        .weights = null,
        .output = packed_tensor,
        .packed_source = packed_tensor,
        .flags_offset = flags,
        .keys_offset = flags + @as(u64, rows) * 8192,
        .groups_x = @as(u64, rows) + (@as(u64, source_rows) + 63) / 64,
        .groups_y = 1,
        .shared_bytes = 16,
    };
    return packDispatch(&launch, "V4.1 index input packing");
}
pub fn scorePacked(scores: ?*Tensor, q: ?*const Tensor, weights: ?*const Tensor, keys: ?*const Tensor, packed_tensor: ?*const Tensor, source_rows: u32, rows: u32, start: u32, ratio: u32, packed_rows: u32, offset: u32) c_int {
    if ((ratio != 1 and ratio != 2) or source_rows == 0 or source_rows > max_i32 or rows == 0 or
        packed_rows == 0 or packed_rows > max_i32 or rows > max_u32 - start or
        (start + rows) / ratio > source_rows or offset > packed_rows or rows > packed_rows - offset or
        c.ds4_gpu_dsv41_tensor_ops_available() == 0) return 0;
    if (!fits(q, @as(u64, rows) * 32 * 128 * 4) or !fits(weights, @as(u64, rows) * 32 * 4) or
        !fits(keys, @as(u64, source_rows) * 128 * 4) or !fits(scores, @as(u64, rows) * source_rows * 4) or
        !fits(packed_tensor, packedBytes(source_rows, packed_rows))) return 0;
    const flags = flagsBytes(source_rows, packed_rows);
    const launch = c.ds4_metal_index_pack_launch{
        .pipeline = c.ds4_metal_index_get_pipeline(c.DS4_METAL_INDEX_PACKED_SCORE),
        .scores = true,
        .args = .{ source_rows, rows, start, ratio },
        .range = .{ packed_rows, offset },
        .q = q,
        .weights = weights,
        .keys = keys,
        .output = scores,
        .packed_source = packed_tensor,
        .flags_offset = flags,
        .keys_offset = flags + @as(u64, packed_rows) * 8192,
        .groups_x = (@as(u64, source_rows) + 63) / 64,
        .groups_y = rows,
        .shared_bytes = 32 * 64 * 4,
    };
    return packDispatch(&launch, "V4.1 packed index scores");
}

pub fn topk(selected: ?*Tensor, scores: ?*const Tensor, width: u32, rows: u32, k: u32, start: u32, ratio: u32) c_int {
    if (api.init() == 0 or selected == null or scores == null or width == 0 or rows == 0 or k == 0 or k > width) return 0;
    const sort_kind: u32 = if (ratio == 0) c.DS4_METAL_INDEX_SORT else if (api.envPresent("DS4_METAL_DISABLE_V41_TOPK_SHUFFLE") != 0) c.DS4_METAL_INDEX_SORT_CAUSAL else c.DS4_METAL_INDEX_SORT_SHUFFLE;
    const merge_kind: u32 = if (ratio == 0) c.DS4_METAL_INDEX_MERGE else if (k == 512 and api.envPresent("DS4_METAL_DISABLE_V41_TOPK_PREFIX") == 0) c.DS4_METAL_INDEX_MERGE_PREFIX else c.DS4_METAL_INDEX_MERGE_CAUSAL;
    const sort = c.ds4_metal_index_get_pipeline(sort_kind);
    const merge = c.ds4_metal_index_get_pipeline(merge_kind);
    if (sort.handle == null or merge.handle == null) return 0;
    if (!fits(scores, @as(u64, width) *% rows *% 4) or !fits(selected, @as(u64, k) *% rows *% 4))
        return api.reportError("ds4: Metal graph indexer top-k received undersized buffers");
    const facts = c.ds4_metal_index_get_facts();
    const maximum = if (facts.sort_threads == 0) 256 else facts.sort_threads;
    var threads: u32 = 1;
    while (threads < width and @as(u64, threads) * 2 <= maximum) threads *= 2;
    if (ratio != 0 and (sort.max_threads < threads or (start +% 1) / ratio <= threads / 2)) return 0;
    // C shader widths are signed32-bit fields, with the original u32 bit
    // narrowing and unsigned size products retained at this ABI boundary.
    const partitions = bits((width +% threads -% 1) / threads);
    const block_k = bits(@min(k, threads));
    var work = bits(k);
    if (partitions > 1) {
        const last = bits(width) -% (partitions -% 1) *% bits(threads);
        work = (partitions -% 1) *% block_k +% @min(last, block_k);
    }
    const scratch_row = unsigned(work) *% 4;
    const one_pass = partitions <= 1;
    const scratch_bytes = (if (one_pass) @as(u64, 1) else 2) *% scratch_row *% rows;
    if (c.ds4_metal_index_reserve_scratch(scratch_bytes) == 0) return 0;
    var args = std.mem.zeroes(c.ds4_metal_index_topk_packet);
    args.sort = .{
        .ne00 = bits(width),
        .ne01 = bits(rows),
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = @as(u64, width) * 4,
        .nb02 = @as(u64, width) *% rows *% 4,
        .nb03 = @as(u64, width) *% rows *% 4,
        .ne0 = work,
        .ne1 = bits(rows),
        .ne2 = 1,
        .ne3 = 1,
        .top_k = block_k,
        .causal_start = start,
        .causal_ratio = ratio,
    };
    // Each block stages indices followed by their scores, rounded to16B.
    const shared = (@as(u64, threads) * 8 + 15) & ~@as(u64, 15);
    var current: u64 = 0;
    var next = scratch_row *% rows;
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    var launch = c.ds4_metal_index_topk_launch{
        .pipeline = sort,
        .merge = false,
        .to_selected = one_pass,
        .args = args,
        .scores = scores,
        .selected = selected,
        .source_offset = 0,
        .target_offset = current,
        .groups = unsigned(partitions) *% rows,
        .threads = threads,
        .shared_bytes = shared,
    };
    if (c.ds4_metal_index_topk_encode(command, &launch) == 0) return api.release(command, owned, 0, "indexer top-k");
    var length = block_k;
    while (length < work) {
        const merges = @divTrunc(work +% 2 *% length -% 1, 2 *% length);
        var merge_threads = merge.max_threads;
        if (merge_threads == 0 or merge_threads > 512) merge_threads = 512;
        merge_threads = @max(@min(merge_threads, unsigned(length)), 1);
        args.merge = .{
            .ne00 = width,
            .ne01 = rows,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = 4,
            .nb01 = @as(u64, width) * 4,
            .nb02 = @as(u64, width) *% rows *% 4,
            .nb03 = @as(u64, width) *% rows *% 4,
            .ne0 = work,
            .ne1 = bits(rows),
            .ne2 = 1,
            .ne3 = 1,
            .top_k = if (merges == 1) bits(k) else work,
            .len = length,
            .causal_start = start,
            .causal_ratio = ratio,
            .block_width = threads,
            .block_top_k = @bitCast(block_k),
        };
        launch = .{
            .pipeline = merge,
            .merge = true,
            .to_selected = merges == 1,
            .args = args,
            .scores = scores,
            .selected = selected,
            .source_offset = current,
            .target_offset = next,
            .groups = unsigned(merges) *% rows,
            .threads = merge_threads,
            .shared_bytes = 0,
        };
        if (c.ds4_metal_index_topk_encode(command, &launch) == 0) return api.release(command, owned, 0, "indexer top-k");
        std.mem.swap(u64, &current, &next);
        length *%= 2;
    }
    return api.release(command, owned, 1, "indexer top-k");
}

pub fn causalTopk(selected: ?*Tensor, scores: ?*const Tensor, width: u32, rows: u32, start: u32, ratio: u32) c_int {
    if ((ratio != 1 and ratio != 2) or rows == 0 or rows > max_u32 - start or width > max_i32 or rows > max_i32 or
        (start + rows) / ratio > width or (start +% 1) / ratio < 1024) return 0;
    return topk(selected, scores, width, rows, 512, start, ratio);
}
fn rowArgs(source_width: u32, target_width: u32, rows: u32) c.ds4_metal_index_rows_args {
    return .{
        .ne00 = source_width,
        .ne01 = rows,
        .nb00 = 4,
        .nb01 = @as(u64, source_width) * 4,
        .ne0 = target_width,
        .ne1 = rows,
        .nb0 = 4,
        .nb1 = @as(u64, target_width) * 4,
    };
}
pub fn mask(out: ?*Tensor, selected: ?*const Tensor, width: u32, rows: u32, k: u32) c_int {
    if (api.init() == 0 or out == null or selected == null or width == 0 or rows == 0 or k == 0) return 0;
    if (!fits(selected, @as(u64, k) *% rows *% 4) or !fits(out, @as(u64, width) *% rows *% 4))
        return api.reportError("ds4: Metal dsv4 top-k mask received undersized buffers");
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    var launch = c.ds4_metal_index_rows_launch{
        .pipeline = c.ds4_metal_index_get_pipeline(c.DS4_METAL_INDEX_MASK_FILL),
        .args = rowArgs(k, width, rows),
        .source = selected,
        .destination = out,
        .groups = (@as(u64, width) * rows + 255) / 256,
        .threads = 256,
        .shared_bytes = 0,
    };
    if (c.ds4_metal_index_rows_encode(command, &launch) == 0) return api.release(command, owned, 0, "dsv4 top-k mask");
    launch.pipeline = c.ds4_metal_index_get_pipeline(c.DS4_METAL_INDEX_MASK_SCATTER);
    launch.groups = (@as(u64, k) * rows + 255) / 256;
    return api.release(command, owned, c.ds4_metal_index_rows_encode(command, &launch), "dsv4 top-k mask");
}
pub fn sortRows(out: ?*Tensor, source: ?*const Tensor, width: u32, rows: u32) c_int {
    if (api.init() == 0 or out == null or source == null or width == 0 or rows == 0 or (width & (width - 1)) != 0) return 0;
    const pipeline = c.ds4_metal_index_get_pipeline(c.DS4_METAL_INDEX_ROW_SORT);
    if (pipeline.handle == null) return 0;
    const bytes = @as(u64, width) *% rows *% 4;
    if (!fits(source, bytes) or !fits(out, bytes)) return api.reportError("ds4: Metal row sort received undersized buffers");
    var threads: u64 = width;
    if (pipeline.max_threads != 0) threads = @min(threads, pipeline.max_threads);
    if (threads == 0) return 0;
    const shared = @as(u64, width) * 4;
    const maximum = c.ds4_metal_index_get_facts().max_shared_bytes;
    if (maximum != 0 and shared > maximum) return api.reportError("ds4: Metal row sort scratch exceeds threadgroup memory limit");
    var owned: c_int = 0;
    const command = api.acquire(&owned) orelse return 0;
    const launch = c.ds4_metal_index_rows_launch{
        .pipeline = pipeline,
        .args = rowArgs(width, width, rows),
        .source = source,
        .destination = out,
        .groups = rows,
        .threads = threads,
        .shared_bytes = shared,
    };
    return api.release(command, owned, c.ds4_metal_index_rows_encode(command, &launch), "sort i32 rows asc");
}
