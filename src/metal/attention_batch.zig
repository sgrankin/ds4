//! Rectangular prefill and batched raw/mixed/indexed attention. Kernel source
//! and packets stay unchanged; this owner chooses masks, staging, tiles and
//! publication order. The zero-prefix cache and stage profiler remain explicit
//! shared resource/diagnostic services, not hidden attention controllers.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Scalar = @import("attention_decode.zig");
const Buffer = Scalar.Buffer;
const Command = Scalar.Command;
const View = c.ds4_metal_attention_view;
const Pipeline = c.ds4_metal_attention_pipeline;
const Tensor = api.Tensor;
const allocator = std.heap.c_allocator;
const empty_view = std.mem.zeroes(View);
fn narrow(value: u32) i32 {
    return @bitCast(value);
}
fn alignUp(value: u64, alignment: u64) u64 {
    return (value +% alignment -% 1) & ~(alignment - 1);
}
fn inverseSqrt(width: u32) f32 {
    @setFloatMode(.optimized);
    return 1.0 / @sqrt(@as(f32, @floatFromInt(width)));
}

pub const Kind = enum { prefill_raw, prefill_mixed, prefill_masked, batch_raw, batch_mixed, batch_visual, noncausal, indexed, glm_compact };
/// Query rows are local in q/heads; q_row0 selects their position in a full
/// static prefix. Batch raw keys instead use raw_cap/raw_start ring storage.
pub const Problem = struct {
    heads: ?*Tensor,
    query: ?*const Tensor,
    raw: ?*const Tensor,
    comp: ?*const Tensor = null,
    mask: ?*const Tensor = null,
    topk: ?*const Tensor = null,
    model: ?*const anyopaque = null,
    model_size: u64 = 0,
    sinks_offset: u64 = 0,
    q_row0: u32 = 0,
    rows: u32,
    raw_rows: u32,
    comp_rows: u32 = 0,
    raw_cap: u32 = 0,
    raw_start: u32 = 0,
    pos0: u32 = 0,
    window: u32 = 0,
    ratio: u32 = 0,
    heads_count: u32,
    width: u32,
    comp_half: bool = false,
    use_mask: bool = false,
    visual_tokens: [*c]const i32 = null,
    vocab_size: u32 = 0,
    top_k: u32 = 0,
    // Only GLM compact reuses this kernel with half input and no sink buffer.
    raw_half: bool = false,
    scale_width: u32 = 0,

    fn prefill(kind: Kind) bool {
        return switch (kind) {
            .prefill_raw, .prefill_mixed, .prefill_masked, .glm_compact => true,
            else => false,
        };
    }
    fn mixed(kind: Kind) bool {
        return kind == .prefill_mixed or kind == .prefill_masked or kind == .batch_mixed or kind == .batch_visual;
    }
    fn publicValid(self: Problem, kind: Kind) bool {
        if (self.heads == null or self.query == null or self.raw == null or self.rows == 0 or self.raw_rows == 0) return false;
        if (kind == .glm_compact) return self.raw_rows <= self.raw_cap and self.q_row0 < self.raw_rows and self.rows <= self.raw_rows - self.q_row0 and self.heads_count != 0 and self.width == 512 and self.scale_width != 0;
        if (self.model == null) return false;
        if (!prefill(kind) and (self.raw_cap < self.raw_rows or self.raw_start >= self.raw_cap)) return false;
        if ((kind == .prefill_mixed or kind == .batch_mixed) and self.ratio == 0) return false;
        if (mixed(kind) and self.comp_rows != 0 and self.comp == null) return false;
        if (self.use_mask and self.mask == null) return false;
        if (kind == .prefill_masked and (self.comp == null or self.mask == null or self.comp_rows == 0 or self.ratio == 0)) return false;
        if (kind == .batch_visual and (self.visual_tokens == null or self.vocab_size == 0 or (self.comp_rows != 0 and self.ratio == 0))) return false;
        if (kind == .noncausal and (self.heads_count == 0 or self.width == 0)) return false;
        if (kind == .indexed and (self.comp == null or self.topk == null or self.comp_rows == 0 or self.top_k == 0 or self.top_k > self.comp_rows or self.top_k & (self.top_k - 1) != 0 or self.ratio == 0 or self.heads_count == 0 or self.width != 512)) return false;
        return true;
    }
    fn keyRows(self: Problem, kind: Kind) u32 {
        return self.raw_rows +% (if (mixed(kind) or kind == .indexed) self.comp_rows else 0);
    }
    fn queryBytes(self: Problem) u64 {
        return @as(u64, self.rows) *% self.heads_count *% self.width *% 4;
    }
};
const Inputs = struct {
    query: Buffer,
    raw: Buffer,
    comp: Buffer,
    mask: Buffer,
    heads: Buffer,
    fn init(problem: Problem, kind: Kind) Inputs {
        const needs_comp = (Problem.mixed(kind) or kind == .indexed) and problem.comp_rows != 0;
        return .{
            .query = Buffer.tensor(problem.query),
            .raw = Buffer.tensor(problem.raw),
            .comp = Buffer.tensor(if (needs_comp) problem.comp else problem.raw),
            .mask = Buffer.tensor(if (problem.use_mask and problem.comp_rows != 0) problem.mask else problem.raw),
            .heads = Buffer.tensor(problem.heads),
        };
    }
    fn deinit(self: Inputs) void {
        self.heads.deinit();
        self.mask.deinit();
        self.comp.deinit();
        self.raw.deinit();
        self.query.deinit();
    }
    fn fits(self: Inputs, problem: Problem, kind: Kind) bool {
        const q_bytes = problem.queryBytes();
        const raw_count = if (Problem.prefill(kind)) problem.raw_rows else problem.raw_cap;
        const raw_element: u64 = if (kind == .glm_compact and problem.raw_half) 2 else 4;
        const raw_bytes = @as(u64, raw_count) *% problem.width *% raw_element;
        const comp_bytes = @as(u64, problem.comp_rows) *% problem.width *% @as(u64, if (problem.comp_half) 2 else 4);
        const mask_rows: u32 = if (Problem.prefill(kind)) problem.q_row0 +% problem.rows else problem.rows;
        const mask_bytes = @as(u64, problem.comp_rows) *% mask_rows *% 4;
        return self.query.fits(q_bytes) and self.raw.fits(raw_bytes) and self.comp.valid() and self.mask.valid() and self.heads.fits(q_bytes) and
            (!(Problem.mixed(kind) or kind == .indexed) or problem.comp_rows == 0 or self.comp.fits(comp_bytes)) and
            (!problem.use_mask or problem.comp_rows == 0 or self.mask.fits(mask_bytes));
    }
};
fn sinkRange(problem: Problem, kind: Kind) ?Buffer {
    const bytes = @as(u64, problem.heads_count) * 4;
    if (problem.sinks_offset > problem.model_size or bytes > problem.model_size - problem.sinks_offset) {
        _ = api.reportError(switch (kind) {
            .noncausal => "ds4: Metal noncausal attention sinks range is outside the mapped model",
            .batch_visual => "ds4: Metal visual attention sinks are outside the mapped model",
            .indexed => "ds4: Metal indexed attention sinks range is outside the mapped model",
            else => "ds4: Metal attention sinks range is outside the mapped model",
        });
        return null;
    }
    var offset: u64 = 0;
    const retained = api.retainModelRange(problem.model, problem.model_size, problem.sinks_offset, bytes, &offset) orelse return null;
    return .{ .view = .{ .buffer = retained, .offset = offset, .bytes = bytes } };
}
fn pipeline(kind: u32, nsg: u32, nwg: u32, pad: bool, boundary: bool, sinks: bool) Pipeline {
    return c.ds4_metal_attention_batch_pipeline(kind, nsg, nwg, pad, boundary, sinks);
}
fn batchScratch(kind: u32, bytes: u64, label: [*:0]const u8) Buffer {
    return .{ .view = c.ds4_metal_attention_batch_scratch(kind, bytes, label) };
}

/// Rectangular causal/window masks keep logical key positions, even when the
/// raw ring was copied in two pieces. Compressed selection never replaces
/// causal visibility: its GPU intersection runs only over compressed columns.
fn fillMask(view: View, problem: Problem, kind: Kind) bool {
    const keys = problem.keyRows(kind);
    const pointer = c.ds4_metal_attention_contents(view) orelse return false;
    const mask: [*]u16 = @ptrCast(@alignCast(pointer));
    if (kind == .noncausal) {
        @memset(mask[0 .. @as(u64, problem.rows) *% keys], 0);
        return true;
    }
    var bounds: []u32 = &.{};
    defer if (bounds.len != 0) allocator.free(bounds);
    if (kind == .batch_visual) {
        bounds = allocator.alloc(u32, @as(usize, problem.rows) * 2) catch return false;
        if (c.ds4_deepseek4_attention_bounds(problem.visual_tokens, problem.rows, problem.vocab_size, problem.pos0, problem.raw_rows, problem.window, bounds.ptr) == 0) return false;
    }
    const is_prefill = Problem.prefill(kind);
    const first_raw = problem.pos0 +% problem.rows -% problem.raw_rows;
    for (0..problem.rows) |query_index| {
        const row_index: u32 = @intCast(query_index);
        const position = (if (is_prefill) problem.q_row0 else problem.pos0) +% row_index;
        const row = mask[@as(u64, row_index) *% keys ..][0..keys];
        for (0..problem.raw_rows) |key_index| {
            const key: u32 = @intCast(key_index);
            const key_position = if (is_prefill) key else first_raw +% key;
            const visible = if (kind == .batch_visual)
                key_position >= bounds[2 * query_index] and key_position <= bounds[2 * query_index + 1]
            else
                key_position <= position and (problem.window == 0 or position -% key_position < problem.window);
            row[key_index] = if (visible) 0 else 0xfc00;
        }
        const visible_comp = if (problem.ratio != 0) (position +% 1) / problem.ratio else 0;
        for (problem.raw_rows..keys) |key_index| row[key_index] = if (key_index - problem.raw_rows < visible_comp) 0 else 0xfc00;
    }
    return true;
}
const Geometry = struct {
    keys: u32,
    rows: u32,
    heads: u32,
    width: u32,
    key_block: u32,
    query_block: u32,
    nsg: u32,
    nwg: u32,
    pad: bool,
    boundary: bool,
    mask_bytes: u64,
    kv_bytes: u64,
    pad_bytes: u64,
    block_bytes: u64,
    tmp_bytes: u64,
    blocks_x: u64,
    blocks_y: u64,
    fn init(problem: Problem, kind: Kind, vector: bool) Geometry {
        const keys = problem.keyRows(kind);
        const key_block: u32 = if (vector) 32 else 64;
        const query_block: u32 = if (vector) 1 else 8;
        const nwg: u32 = if (vector) 32 else 1;
        var nsg: u32 = if (vector) 1 else 8;
        if (vector) while (2 * nwg * nsg * key_block < keys and nsg < 4) {
            nsg *= 2;
        };
        const pad = keys % key_block != 0;
        const row_bytes = @as(u64, problem.width) * 2;
        const pad_bytes = if (pad) @as(u64, key_block) *% (2 *% row_bytes +% @as(u64, problem.rows) * 2) else 1;
        const mask_bytes = @as(u64, keys) *% problem.rows *% 2;
        const nrows = @as(u64, problem.rows) * problem.heads_count;
        const blocks_x = (@as(u64, keys) + key_block - 1) / key_block;
        const blocks_y = (@as(u64, problem.rows) + query_block - 1) / query_block;
        return .{
            .keys = keys,
            .rows = problem.rows,
            .heads = problem.heads_count,
            .width = problem.width,
            .key_block = key_block,
            .query_block = query_block,
            .nsg = nsg,
            .nwg = nwg,
            .pad = pad,
            .boundary = problem.rows % query_block != 0,
            .mask_bytes = mask_bytes,
            .kv_bytes = @as(u64, keys) *% row_bytes,
            .pad_bytes = pad_bytes,
            .block_bytes = alignUp(blocks_x *% blocks_y, 32),
            .tmp_bytes = nrows *% problem.width *% nwg *% 4 +% nrows *% (2 * nwg) *% 4,
            .blocks_x = blocks_x,
            .blocks_y = blocks_y,
        };
    }
    fn padArgs(self: Geometry) c.ds4_metal_attention_pad_args {
        const row = @as(u64, self.width) * 2;
        return .{
            .ne11 = narrow(self.keys),
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row,
            .nb12 = @as(u64, self.keys) * row,
            .nb13 = @as(u64, self.keys) * row,
            .nb21 = row,
            .nb22 = @as(u64, self.keys) * row,
            .nb23 = @as(u64, self.keys) * row,
            .ne31 = narrow(self.rows),
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = @as(u64, self.keys) * 2,
            .nb32 = self.mask_bytes,
            .nb33 = self.mask_bytes,
        };
    }
    fn arguments(self: Geometry, scale: f32) c.ds4_metal_attention_vec_args {
        const row = @as(u64, self.width) * 4;
        const half_row = @as(u64, self.width) * 2;
        return .{
            .ne01 = narrow(self.rows),
            .ne02 = narrow(self.heads),
            .ne03 = 1,
            .nb01 = @as(u64, self.heads) * row,
            .nb02 = row,
            .nb03 = @as(u64, self.rows) *% self.heads *% row,
            .ne11 = narrow(self.keys),
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .ns10 = narrow(self.width),
            .nb11 = half_row,
            .nb12 = @as(u64, self.keys) * half_row,
            .nb13 = @as(u64, self.keys) * half_row,
            .ns20 = narrow(self.width),
            .nb21 = half_row,
            .nb22 = @as(u64, self.keys) * half_row,
            .nb23 = @as(u64, self.keys) * half_row,
            .ne31 = narrow(self.rows),
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = @as(u64, self.keys) * 2,
            .nb32 = self.mask_bytes,
            .nb33 = self.mask_bytes,
            .ne1 = narrow(self.heads),
            .ne2 = narrow(self.rows),
            .ne3 = 1,
            .scale = scale,
            .max_bias = 0,
            .m0 = 0,
            .m1 = 0,
            .n_head_log2 = 0,
            .logit_softcap = 0,
        };
    }
};
const Profile = struct {
    enabled: bool,
    mode: [*:0]const u8,
    started: f64 = 0,
    problem: Problem,
    geometry: Geometry,
    fn begin(self: *Profile, command: *Command) bool {
        return !self.enabled or c.ds4_metal_attention_profile_begin(&command.token, &self.started) != 0;
    }
    fn stage(self: *Profile, command: *Command, label: [*:0]const u8) bool {
        const p = self.problem;
        return !self.enabled or c.ds4_metal_attention_profile_stage(&command.token, self.mode, label, p.rows, p.comp_rows, self.geometry.keys, p.heads_count, p.width, p.window, p.ratio, &self.started) != 0;
    }
};
fn copyMask(command: *Command, input: View, target: View, problem: Problem, kind: Kind, geometry: Geometry, facts: c.ds4_metal_attention_facts) bool {
    if (!problem.use_mask or problem.comp_rows == 0) return true;
    const source = Scalar.at(input, @as(u64, problem.q_row0) *% problem.comp_rows *% 4);
    const destination = Scalar.at(target, @as(u64, problem.raw_rows) * 2);
    if (!Problem.prefill(kind)) {
        const pso = pipeline(c.DS4_ATTN_BATCH_CAUSAL, 0, 0, false, false, false);
        if (pso.handle == null) return false;
        const args = c.ds4_metal_attention_causal_args{
            .n_comp = problem.comp_rows,
            .n_tokens = problem.rows,
            .pos0 = problem.pos0,
            .ratio = problem.ratio,
            .n_keys = geometry.keys,
        };
        return c.ds4_metal_attention_causal_mask(command.token, pso, &args, input, destination, @min(@as(u64, 256), pso.threads)) != 0;
    }
    const columns = problem.comp_rows;
    const source_stride = @as(u64, columns) * 4;
    const destination_stride = @as(u64, geometry.keys) * 2;
    const args = api.CopyArgs{
        .nk0 = columns,
        .ne00 = columns,
        .ne01 = problem.rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = source_stride,
        .nb02 = @as(u64, problem.rows) *% source_stride,
        .nb03 = @as(u64, problem.rows) *% source_stride,
        .ne0 = columns,
        .ne1 = problem.rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = 2,
        .nb1 = destination_stride,
        .nb2 = @as(u64, problem.rows) *% destination_stride,
        .nb3 = @as(u64, problem.rows) *% destination_stride,
    };
    var threads: u64 = 32;
    while (threads < columns and threads < facts.copy_threads[1]) threads *= 2;
    threads = @min(threads, facts.copy_threads[1], @as(u64, columns));
    if (threads == 0) threads = 1;
    const groups = ((@as(u64, columns) + threads - 1) / threads) * problem.rows;
    return c.ds4_metal_attention_copy(command.token, c.DS4_ATTN_COPY_F32_GENERIC, &args, 0, source, destination, groups, threads) != 0;
}
const Pipelines = struct {
    pad: Pipeline = std.mem.zeroes(Pipeline),
    block: Pipeline = std.mem.zeroes(Pipeline),
    attention: Pipeline = std.mem.zeroes(Pipeline),
    reduce: Pipeline = std.mem.zeroes(Pipeline),
    fn init(geometry: Geometry, vector: bool, sinks: bool) ?Pipelines {
        var result = Pipelines{};
        if (geometry.pad) {
            result.pad = if (vector) c.ds4_metal_attention_get_pipeline(c.DS4_ATTN_PAD_PSO, geometry.width, geometry.nsg, geometry.nwg, true, false) else pipeline(c.DS4_ATTN_BATCH_PAD, geometry.nsg, geometry.nwg, true, false, sinks);
            if (result.pad.handle == null) return null;
        }
        if (!vector) result.block = pipeline(c.DS4_ATTN_BATCH_BLOCK_MAP, geometry.nsg, geometry.nwg, geometry.pad, geometry.boundary, sinks);
        result.attention = pipeline(if (vector) c.DS4_ATTN_BATCH_VECTOR else c.DS4_ATTN_BATCH_NONVECTOR, geometry.nsg, geometry.nwg, geometry.pad, geometry.boundary, sinks);
        if (vector) result.reduce = c.ds4_metal_attention_get_pipeline(c.DS4_ATTN_REDUCE_PSO, geometry.width, geometry.nsg, geometry.nwg, geometry.pad, false);
        if (result.attention.handle == null or (vector and result.reduce.handle == null) or (!vector and result.block.handle == null)) return null;
        return result;
    }
};
fn dispatch(command: *Command, inputs: Inputs, sinks: View, geometry: Geometry, mask: View, kv: View, pad: View, block: View, tmp: View, vector: bool, pipelines: Pipelines, cache: c.ds4_metal_attention_cache, profile: *Profile, scale: f32) bool {
    if (geometry.pad) {
        const args = geometry.padArgs();
        const ok = if (vector) c.ds4_metal_attention_pad(command.token, pipelines.pad, &args, kv, mask, pad) else c.ds4_metal_attention_batch_pad(command.token, pipelines.pad, &args, kv, mask, pad, geometry.key_block);
        if (ok == 0 or !profile.stage(command, "pad")) return false;
    }
    if (!vector) {
        if (cache.entry == null or !cache.block_ready) {
            const args = c.ds4_metal_attention_block_args{
                .ne01 = narrow(geometry.rows),
                .ne30 = narrow(geometry.keys),
                .ne31 = narrow(geometry.rows),
                .ne32 = 1,
                .ne33 = 1,
                .nb31 = @as(u64, geometry.keys) * 2,
                .nb32 = geometry.mask_bytes,
                .nb33 = geometry.mask_bytes,
            };
            if (c.ds4_metal_attention_block_map(command.token, pipelines.block, &args, mask, block, geometry.blocks_x, geometry.blocks_y) == 0) return false;
            c.ds4_metal_attention_cache_publish(cache.entry, false, true);
        }
        if (!profile.stage(command, "block_map")) return false;
    }
    const args = geometry.arguments(scale);
    if (vector) {
        const shared_elems = (alignUp(geometry.width, 128) + 4 * geometry.key_block + 2 * alignUp(geometry.width, 128)) * geometry.nsg;
        const shared = alignUp(shared_elems * 2, 16);
        if (c.ds4_metal_attention_vec(command.token, pipelines.attention, &args, inputs.query.view, kv, mask, sinks, pad, tmp, null, geometry.rows, geometry.heads, geometry.nwg, 32, geometry.nsg, shared) == 0 or !profile.stage(command, "attention_vec")) return false;
        const rows = @as(u64, geometry.rows) * geometry.heads;
        const reduce_args = c.ds4_metal_attention_reduce_args{ .nrows = @bitCast(@as(u32, @truncate(rows))) };
        return c.ds4_metal_attention_reduce(command.token, pipelines.reduce, &reduce_args, tmp, inputs.heads.view, null, rows, 32 * geometry.nwg) != 0 and profile.stage(command, "attention_reduce");
    }
    const padded_value = alignUp(geometry.width, 64);
    const shared_elems = @as(u64, geometry.query_block) * (geometry.width + 2 * padded_value + 2 * (2 * geometry.key_block));
    const shared = alignUp(shared_elems * 2, 16);
    return c.ds4_metal_attention_nonvector(command.token, pipelines.attention, &args, inputs.query.view, kv, mask, sinks, pad, block, inputs.heads.view, geometry.blocks_y, geometry.heads, geometry.nsg, shared) != 0 and profile.stage(command, "attention");
}
fn encode(command: *Command, problem: Problem, kind: Kind, inputs: Inputs, sinks: View) bool {
    if (problem.width != 512 or problem.heads_count == 0) return false;
    const prefill = Problem.prefill(kind);
    const mixed = Problem.mixed(kind) and problem.comp_rows != 0;
    const vector = prefill and kind != .glm_compact and problem.raw_rows < 20 and problem.q_row0 == 0 and problem.rows == problem.raw_rows;
    const geometry = Geometry.init(problem, kind, vector);
    if (!inputs.fits(problem, kind)) {
        _ = api.reportError(if (!prefill) (if (mixed) "ds4: Metal decode mixed batch FlashAttention received undersized buffers" else "ds4: Metal decode raw batch FlashAttention received undersized buffers") else if (Problem.mixed(kind)) (if (vector) "ds4: Metal prefill static mixed DS4 FlashAttention received undersized buffers" else "ds4: Metal prefill static mixed DS4 non-vector FlashAttention received undersized buffers") else if (vector) "ds4: Metal prefill raw DS4 FlashAttention received undersized buffers" else "ds4: Metal prefill shared-KV FlashAttention received undersized buffers");
        return false;
    }
    var cache = std.mem.zeroes(c.ds4_metal_attention_cache);
    // Only full square zero-prefix masks can be reused. Selected masks,
    // rectangles and decode windows always materialize their own mask.
    if (prefill and !vector and !problem.use_mask and problem.q_row0 == 0 and problem.rows == problem.raw_rows) {
        const cache_kind: u32 = if (!Problem.mixed(kind)) c.DS4_GPU_PREFILL_MASK_CACHE_RAW else if (problem.ratio == 4) c.DS4_GPU_PREFILL_MASK_CACHE_RATIO4 else if (problem.ratio == 128) c.DS4_GPU_PREFILL_MASK_CACHE_RATIO128 else 0;
        if (cache_kind != 0) {
            const key = c.ds4_metal_attention_cache_key{
                .kind = cache_kind,
                .n_tokens = problem.raw_rows,
                .n_comp = problem.comp_rows,
                .n_keys = geometry.keys,
                .window = problem.window,
                .ratio = problem.ratio,
                .nqptg = geometry.query_block,
                .ncpsg = geometry.key_block,
                .has_kvpad = geometry.pad,
                .bc_mask = geometry.boundary,
                .mask_bytes = geometry.mask_bytes,
                .blk_bytes = geometry.block_bytes,
            };
            cache = c.ds4_metal_attention_cache_acquire(&key);
        }
    }
    const mask = if (cache.entry != null) Buffer{ .view = cache.mask } else Buffer.scratch(c.DS4_ATTN_TRANSIENT_MASK, geometry.mask_bytes, "ds4_flash_attn_mask");
    defer mask.deinit();
    var block = Buffer{ .view = cache.block };
    defer block.deinit();
    var kv = Buffer{ .view = empty_view };
    defer kv.deinit();
    var pad = Buffer{ .view = empty_view };
    defer pad.deinit();
    var tmp = Buffer{ .view = empty_view };
    defer tmp.deinit();
    if (cache.entry != null and cache.created) {
        if (!fillMask(mask.view, problem, kind)) return false;
        c.ds4_metal_attention_cache_publish(cache.entry, true, false);
    }
    if (!mask.valid()) return false;
    // Original raw-vector path reserves pad before KV; all other paths reserve
    // KV first. A failed reservation must leave the same prior scratch state.
    if (vector and !Problem.mixed(kind)) {
        pad = Buffer.scratch(c.DS4_ATTN_PAD, geometry.pad_bytes, "ds4_flash_attn_pad");
        if (!pad.valid()) return false;
    }
    kv = Buffer.scratch(c.DS4_ATTN_KV, geometry.kv_bytes, if (vector and Problem.mixed(kind)) "ds4_flash_attn_kv" else "ds4_flash_attn_kv_f16");
    if (!kv.valid()) return false;
    if (!pad.valid()) {
        pad = Buffer.scratch(c.DS4_ATTN_PAD, geometry.pad_bytes, "ds4_flash_attn_pad");
        if (!pad.valid()) return false;
    }
    if (vector) {
        tmp = Buffer.scratch(c.DS4_ATTN_TMP, geometry.tmp_bytes, "ds4_flash_attn_tmp");
        if (!tmp.valid()) return false;
    } else if (cache.entry == null) {
        block = batchScratch(c.DS4_ATTN_BATCH_BLOCK, geometry.block_bytes, "ds4_flash_attn_blk");
        if (!block.valid()) return false;
    }
    var profile = Profile{
        .enabled = prefill and c.ds4_metal_attention_profile_enabled(),
        .mode = if (Problem.mixed(kind)) (if (vector) "static_mixed_vec" else "static_mixed_nonvec") else if (vector) "raw_vec" else "raw_nonvec",
        .problem = problem,
        .geometry = geometry,
    };
    if (!profile.begin(command)) return false;
    const facts = c.ds4_metal_attention_get_facts();
    const has_sinks = kind != .glm_compact;
    var pipelines: ?Pipelines = null;
    // Raw prefill looks up kernels before staging; mixed prefill and decode
    // batches do so after staging/mask construction, exactly as the C paths.
    if (prefill and !Problem.mixed(kind)) {
        if (cache.entry == null and !fillMask(mask.view, problem, kind)) return false;
        if (!profile.stage(command, "mask_fill")) return false;
        pipelines = Pipelines.init(geometry, vector, has_sinks) orelse return false;
    }
    if (prefill) {
        if (!Scalar.copyHalf(command, inputs.raw.view, kv.view, problem.raw_half, problem.raw_rows *% problem.width, facts) or !profile.stage(command, "copy_raw")) return false;
    } else if (!Scalar.ringCopy(command, inputs.raw.view, problem.raw_cap, problem.raw_start, problem.raw_rows, problem.width, kv.view, facts)) return false;
    if (mixed) {
        const target = Scalar.at(kv.view, @as(u64, problem.raw_rows) * problem.width * 2);
        if (!Scalar.copyHalf(command, inputs.comp.view, target, problem.comp_half, problem.comp_rows *% problem.width, facts) or !profile.stage(command, "copy_comp")) return false;
    }
    if (!prefill or Problem.mixed(kind)) {
        if (cache.entry == null and !fillMask(mask.view, problem, kind)) return false;
        if (!profile.stage(command, "mask_fill") or !copyMask(command, inputs.mask.view, mask.view, problem, kind, geometry, facts)) return false;
        if (problem.use_mask and problem.comp_rows != 0 and !profile.stage(command, "mask_comp_copy")) return false;
        pipelines = Pipelines.init(geometry, vector, has_sinks) orelse return false;
    }
    const scale = inverseSqrt(if (kind == .glm_compact) problem.scale_width else problem.width);
    return dispatch(command, inputs, if (has_sinks) sinks else inputs.query.view, geometry, mask.view, kv.view, pad.view, block.view, tmp.view, vector, pipelines.?, cache, &profile, scale);
}
fn indexed(problem: Problem) c_int {
    // Preserve the public range check before buffer validation, but do not wrap
    // the mapped sink range until all tensor extents have passed.
    const sink_bytes = @as(u64, problem.heads_count) * 4;
    if (problem.sinks_offset > problem.model_size or sink_bytes > problem.model_size - problem.sinks_offset) return api.reportError("ds4: Metal indexed attention sinks range is outside the mapped model");
    const inputs = Inputs.init(problem, .indexed);
    defer inputs.deinit();
    const topk = Buffer.tensor(problem.topk);
    defer topk.deinit();
    const topk_bytes = @as(u64, problem.top_k) *% problem.rows *% 4;
    if (!inputs.fits(problem, .indexed) or !topk.fits(topk_bytes)) return api.reportError("ds4: Metal indexed mixed attention received undersized buffers");
    const sinks = sinkRange(problem, .indexed) orelse return 0;
    defer sinks.deinit();
    const facts = c.ds4_metal_attention_get_facts();
    const quality = facts.flags & c.DS4_ATTN_QUALITY != 0;
    const sort = pipeline(c.DS4_ATTN_BATCH_SORT, 0, 0, false, false, true);
    const decode = problem.rows == 1;
    const dual_heads = !decode and !quality and c.ds4_metal_attention_batch_mpp() and (problem.heads_count == 64 or problem.heads_count == 32) and problem.top_k == 512 and problem.window == 128 and problem.width == 512;
    const splits: u32 = if (decode and !quality) 12 else 1;
    const split = splits > 1;
    const attention = pipeline(if (split) c.DS4_ATTN_BATCH_INDEX_SPLIT else if (decode) c.DS4_ATTN_BATCH_INDEX_DECODE else if (dual_heads) c.DS4_ATTN_BATCH_INDEX_DUAL else c.DS4_ATTN_BATCH_INDEX, 0, 0, false, false, true);
    const reduce = if (split) pipeline(c.DS4_ATTN_BATCH_INDEX_REDUCE, 0, 0, false, false, true) else std.mem.zeroes(Pipeline);
    if (sort.handle == null or attention.handle == null or (split and reduce.handle == null)) return 0;
    if (problem.top_k > sort.threads) return api.reportError("ds4: Metal indexed attention top-k exceeds sort threadgroup limit");
    // Fast decode uses the same full selected set in score order and avoids a
    // chronological sort. Quality restores sorted order for reproducibility.
    const skip_sort = !quality and decode;
    const sorted = if (skip_sort) Buffer{ .view = empty_view } else batchScratch(c.DS4_ATTN_BATCH_TOPK, topk_bytes, "ds4_indexed_topk_sorted");
    defer sorted.deinit();
    if (!skip_sort and !sorted.valid()) return 0;
    const row_bytes = @as(u64, problem.width) * 4;
    const half_row = @as(u64, problem.width) * 2;
    const sort_args = c.ds4_metal_index_rows_args{
        .ne00 = problem.top_k,
        .ne01 = problem.rows,
        .nb00 = 4,
        .nb01 = @as(u64, problem.top_k) * 4,
        .ne0 = problem.top_k,
        .ne1 = problem.rows,
        .nb0 = 4,
        .nb1 = @as(u64, problem.top_k) * 4,
    };
    const args = c.ds4_metal_attention_indexed_args{
        .n_tokens = problem.rows,
        .n_head = problem.heads_count,
        .n_raw = problem.raw_rows,
        .raw_cap = problem.raw_cap,
        .raw_start = problem.raw_start,
        .n_comp = problem.comp_rows,
        .top_k = problem.top_k,
        .pos0 = problem.pos0,
        .window = problem.window,
        .ratio = problem.ratio,
        .comp_kv_f16 = @intFromBool(problem.comp_half),
        .n_splits = splits,
        .q_token_stride = @as(u64, problem.heads_count) * row_bytes,
        .q_head_stride = row_bytes,
        .raw_row_stride = row_bytes,
        .comp_row_stride = if (problem.comp_half) half_row else row_bytes,
        .topk_token_stride = @as(u64, problem.top_k) * 4,
        .dst_token_stride = @as(u64, problem.heads_count) * row_bytes,
        .dst_head_stride = row_bytes,
        .scale = inverseSqrt(problem.width),
    };
    var command = Command.init() orelse return 0;
    defer command.deinit();
    if (!skip_sort and c.ds4_metal_attention_index_sort(command.token, sort, &sort_args, topk.view, sorted.view, problem.rows, problem.top_k, @as(u64, problem.top_k) * 4) == 0) return 0;
    const selected = if (skip_sort) topk.view else sorted.view;
    const rows = @as(u64, problem.rows) * problem.heads_count;
    if (split) {
        const partial_bytes = rows *% problem.width *% splits *% 4;
        const stats_bytes = rows *% splits *% 2 *% 4;
        if (partial_bytes > std.math.maxInt(u64) - stats_bytes) return 0;
        const tmp = Buffer.scratch(c.DS4_ATTN_TMP, partial_bytes + stats_bytes, "ds4_dsv4_indexed_attention_split_tmp");
        defer tmp.deinit();
        if (!tmp.valid()) return 0;
        if (c.ds4_metal_attention_index_run(command.token, attention, &args, inputs.query.view, inputs.raw.view, inputs.comp.view, selected, empty_view, tmp.view, true, problem.rows, (@as(u64, problem.heads_count) + 7) / 8, splits, 16 * 128 * 2 * 4) == 0 or
            c.ds4_metal_attention_index_reduce(command.token, reduce, &args, tmp.view, sinks.view, inputs.heads.view, rows) == 0) return 0;
    } else {
        const group_heads: u64 = if (dual_heads) 16 else 8;
        const shared: u64 = @as(u64, if (decode) 16 else 1) * 128 * 4 * 2;
        if (c.ds4_metal_attention_index_run(command.token, attention, &args, inputs.query.view, inputs.raw.view, inputs.comp.view, selected, sinks.view, inputs.heads.view, false, problem.rows, (@as(u64, problem.heads_count) + group_heads - 1) / group_heads, 1, shared) == 0) return 0;
    }
    return command.finish("graph indexed mixed attention heads");
}

pub fn run(kind: Kind, problem: Problem) c_int {
    if (api.init() == 0 or !problem.publicValid(kind)) return 0;
    if (kind == .indexed) return indexed(problem);
    const sinks = if (kind == .glm_compact) Buffer{ .view = empty_view } else sinkRange(problem, kind) orelse return 0;
    defer sinks.deinit();
    var command = Command.init() orelse return 0;
    defer command.deinit();
    // The inner fixed-width admission follows sink wrapping and command
    // acquisition, as in the original public functions.
    if (problem.width != 512 or problem.heads_count == 0) return 0;
    const inputs = Inputs.init(problem, kind);
    defer inputs.deinit();
    if (!encode(&command, problem, kind, inputs, sinks.view)) return 0;
    return command.finish(switch (kind) {
        .prefill_raw => "graph prefill raw attention heads",
        .prefill_mixed => "graph prefill static mixed attention heads",
        .prefill_masked => "graph prefill masked mixed attention heads",
        .batch_raw => "graph decode raw batch attention heads",
        .batch_mixed => "graph decode mixed batch attention heads",
        .batch_visual => "graph visual mixed batch attention heads",
        .noncausal => "graph noncausal raw batch attention heads",
        .glm_compact => "GLM dense compact prefill attention",
        .indexed => unreachable,
    });
}
