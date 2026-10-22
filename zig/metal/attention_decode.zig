//! Single-query raw/gathered FlashAttention and gathered half-KV publication.
//! Kernels are unchanged. Native owns scratch geometry, copy/pad plans and the
//! deferred inverse-RoPE state; ARC/PSO/encoder leaves borrow fixed packets.
//! Backend serialization is the existing contract for this Metal-local state.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Tensor = api.Tensor;
const View = c.ds4_metal_attention_view;
const Facts = c.ds4_metal_attention_facts;
const Pipeline = c.ds4_metal_attention_pipeline;
const RopeArgs = c.ds4_metal_attention_rope_args;
const PadArgs = c.ds4_metal_attention_pad_args;
const VecArgs = c.ds4_metal_attention_vec_args;
const max_u32 = std.math.maxInt(u32);
const max_u64 = std.math.maxInt(u64);

const DeferredRope = struct { args: RopeArgs, pending: bool = false, used: bool = false };
var deferred_rope: DeferredRope = .{ .args = std.mem.zeroes(RopeArgs) };
var nwg_environment: c_int = -1;
var warned_unconsumed = false;
fn narrow(value: u32) i32 {
    return @bitCast(value);
}
fn present(name: [*:0]const u8) bool {
    return api.envPresent(name) != 0;
}
fn fact(facts: Facts, mask: u32) bool {
    return facts.flags & mask != 0;
}
fn alignUp(value: u64, alignment: u64) u64 {
    return (value +% alignment -% 1) & ~(alignment - 1);
}

pub fn setRope(width: u32, dimensions: u32, position: u32, original_context: u32, inverse: bool, base: f32, scale: f32, ext: f32, attention: f32, fast: f32, slow: f32) void {
    const row_bytes = @as(u64, width) * 4;
    deferred_rope = .{ .args = .{
        .row_bytes = row_bytes,
        .token_bytes = row_bytes,
        .head_dim = narrow(width),
        .n_dims = narrow(dimensions),
        .n_ctx_orig = narrow(original_context),
        .inverse = @intFromBool(inverse),
        .pos0 = position,
        .pos_step = 1,
        .freq_base = base,
        .freq_scale = scale,
        .ext_factor = ext,
        .attn_factor = attention,
        .beta_fast = fast,
        .beta_slow = slow,
    }, .pending = true, .used = false };
}
pub fn ropeUsed() c_int {
    return @intFromBool(deferred_rope.used);
}

pub const Buffer = struct {
    view: View,
    pub fn tensor(value: ?*const Tensor) Buffer {
        return .{ .view = c.ds4_metal_attention_tensor_view(value) };
    }
    pub fn scratch(kind: u32, bytes: u64, label: [*:0]const u8) Buffer {
        return .{ .view = c.ds4_metal_attention_scratch(kind, bytes, label) };
    }
    pub fn valid(self: Buffer) bool {
        return self.view.buffer != null;
    }
    pub fn fits(self: Buffer, bytes: u64) bool {
        return self.valid() and self.view.bytes >= bytes;
    }
    pub fn deinit(self: Buffer) void {
        if (self.view.buffer) |pointer| api.releaseModelBuffer(pointer);
    }
};
pub fn at(view: View, offset: u64) View {
    var shifted = view;
    shifted.offset += offset;
    return shifted;
}
pub const Command = struct {
    token: ?*anyopaque,
    owned: c_int,
    pub fn init() ?Command {
        var owned: c_int = 0;
        const token = api.acquire(&owned) orelse return null;
        return .{ .token = token, .owned = owned };
    }
    pub fn deinit(self: *Command) void {
        if (self.token) |token| _ = api.release(token, self.owned, 0, "attention heads");
    }
    pub fn finish(self: *Command, label: [*:0]const u8) c_int {
        const token = self.token.?;
        self.token = null;
        return api.release(token, self.owned, 1, label);
    }
};
fn copyArgs(count: u32) api.CopyArgs {
    const src = @as(u64, count) * 4;
    const dst = @as(u64, count) * 2;
    return .{
        .nk0 = count,
        .ne00 = count,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = 4,
        .nb01 = src,
        .nb02 = src,
        .nb03 = src,
        .ne0 = count,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = 2,
        .nb1 = dst,
        .nb2 = dst,
        .nb3 = dst,
    };
}
pub fn copyHalf(command: *const Command, source: View, target: View, source_half: bool, count: u32, facts: Facts) bool {
    if (source.buffer == null or target.buffer == null) return false;
    if (source_half and count == 0) return true;
    if (count == 0) return false;
    var kind: u32 = undefined;
    var slot: usize = undefined;
    if (source_half) {
        if (api.envBool("DS4_METAL_DISABLE_CONTIG_F16_F16_COPY") > 0) {
            return c.ds4_metal_attention_copy(command.token.?, c.DS4_ATTN_COPY_F16_BLIT, null, count, source, target, 0, 0) != 0;
        }
        kind = c.DS4_ATTN_COPY_F16_CONTIG;
        slot = 2;
    } else if (api.envBool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0) {
        kind = c.DS4_ATTN_COPY_F32_CONTIG;
        slot = 0;
    } else {
        kind = c.DS4_ATTN_COPY_F32_GENERIC;
        slot = 1;
    }
    const work: u64 = if (slot == 1) count else (@as(u64, count) + 3) / 4;
    var threads: u64 = 32;
    while (threads < work and threads < facts.copy_threads[slot]) threads *= 2;
    threads = @min(threads, facts.copy_threads[slot], work);
    if (threads == 0) threads = 1;
    const args = copyArgs(count);
    return c.ds4_metal_attention_copy(command.token.?, kind, &args, count, source, target, (work + threads - 1) / threads, threads) != 0;
}
pub fn ringCopy(command: *const Command, raw: View, capacity: u32, start: u32, rows: u32, width: u32, target: View, facts: Facts) bool {
    if (raw.buffer == null or target.buffer == null or capacity == 0 or start >= capacity or rows == 0 or rows > capacity or width == 0) return false;
    const tail: u32 = @min(capacity - start, rows);
    const head = rows - tail;
    const tail_count = @as(u64, tail) * width;
    const head_count = @as(u64, head) * width;
    const raw_offset = @as(u64, start) * width * 4;
    const target_offset = @as(u64, tail) * width * 2;
    if (tail_count > max_u32 or head_count > max_u32 or raw_offset > max_u64 - raw.offset or target_offset > max_u64 - target.offset) return false;
    if (tail != 0 and !copyHalf(command, at(raw, raw_offset), target, false, @intCast(tail_count), facts)) return false;
    return head == 0 or copyHalf(command, raw, at(target, target_offset), false, @intCast(head_count), facts);
}
const Stage = struct { raw_cap: u32, raw_start: u32, raw_rows: u32, comp_rows: u32, width: u32, comp_half: bool };
fn stage(command: *const Command, request: Stage, raw: View, comp: View, target: View, mask: View, pad: View, fuse_pad: bool, shared_pad: bool, facts: Facts, did_fuse: *bool) bool {
    did_fuse.* = false;
    if (raw.buffer == null or comp.buffer == null or target.buffer == null or request.raw_cap == 0 or request.raw_start >= request.raw_cap or request.raw_rows == 0 or request.raw_rows > request.raw_cap or request.comp_rows == 0 or request.width == 0) return false;
    const supported = request.comp_half and request.width == 512 and request.raw_cap <= max_u32 / 128;
    const comp_count = @as(u64, request.comp_rows) * request.width;
    const comp_offset = @as(u64, request.raw_rows) * request.width * 2;
    if (comp_count > max_u32 or comp_offset > max_u64 - target.offset) return false;
    const vectors = (@as(u64, request.raw_rows) + request.comp_rows) * 128;
    const eligible = supported and vectors != 0 and vectors <= max_u32 and !fact(facts, c.DS4_ATTN_QUALITY) and !present("DS4_METAL_DISABLE_GATHERED_KV_STAGE") and fact(facts, c.DS4_ATTN_STAGE_AVAILABLE) and (fact(facts, c.DS4_ATTN_M3) or fact(facts, c.DS4_ATTN_M5));
    const component_disabled = eligible and (api.envBool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") > 0 or api.envBool("DS4_METAL_DISABLE_CONTIG_F16_F16_COPY") > 0);
    const fusion = eligible and !component_disabled;
    const pad_fusion = fusion and fuse_pad and mask.buffer != null and pad.buffer != null and !present("DS4_METAL_DISABLE_GATHERED_KV_PAD_FUSION");
    if (present("DS4_METAL_REQUIRE_GATHERED_KV_STAGE") and supported and !fusion) {
        _ = api.reportError("ds4: required Metal gathered KV staging kernel was not selected");
        return false;
    }
    if (fusion) {
        const args = c.ds4_metal_attention_stage_args{ .raw_cap = request.raw_cap, .raw_start = request.raw_start, .n_raw = request.raw_rows, .n_comp = request.comp_rows, .pad_rows = if (pad_fusion) 32 else 0, .shared_pad = @intFromBool(pad_fusion and shared_pad) };
        const work = vectors + (if (pad_fusion) @as(u64, 32 * 128 + 32) else 0);
        const threads: u64 = @min(facts.stage_threads, @as(u64, 256), work);
        if (threads == 0) return false;
        // Same safety net as C: retire the current encoder before flushing a
        // deferred norm/store task, then acquire the staging encoder afresh.
        if (c.ds4_gpu_kv_norm_task_pending() != 0 and
            (c.ds4_metal_attention_break_encoder(command.token.?) == 0 or c.ds4_gpu_kv_norm_task_flush() == 0)) return false;
        if (c.ds4_metal_attention_stage(command.token.?, &args, raw, comp, target, if (pad_fusion) mask else target, if (pad_fusion) pad else target, (work + threads - 1) / threads, threads) == 0) return false;
        did_fuse.* = pad_fusion;
        return true;
    }
    if (!ringCopy(command, raw, request.raw_cap, request.raw_start, request.raw_rows, request.width, target, facts)) return false;
    return copyHalf(command, comp, at(target, comp_offset), request.comp_half, @intCast(comp_count), facts);
}

pub fn stageTensor(dst: ?*Tensor, raw_tensor: ?*const Tensor, raw_cap: u32, raw_start: u32, raw_rows: u32, comp_tensor: ?*const Tensor, comp_half: u32, comp_rows: u32, width: u32) c_int {
    if (api.init() == 0) return 0;
    if (dst == null or raw_tensor == null or comp_tensor == null or raw_cap == 0 or raw_start >= raw_cap or raw_rows == 0 or raw_rows > raw_cap or comp_rows == 0 or comp_half == 0 or width != 512) return 0;
    const raw = Buffer.tensor(raw_tensor);
    defer raw.deinit();
    const comp = Buffer.tensor(comp_tensor);
    defer comp.deinit();
    const target = Buffer.tensor(dst);
    defer target.deinit();
    if (!raw.fits(@as(u64, raw_cap) * width * 4) or !comp.fits(@as(u64, comp_rows) * width * 2) or !target.fits((@as(u64, raw_rows) + comp_rows) * width * 2)) return api.reportError("ds4: Metal gathered KV staging received undersized buffers");
    var command = Command.init() orelse return 0;
    defer command.deinit();
    var fused = false;
    const empty = std.mem.zeroes(View);
    if (!stage(&command, .{ .raw_cap = raw_cap, .raw_start = raw_start, .raw_rows = raw_rows, .comp_rows = comp_rows, .width = width, .comp_half = comp_half != 0 }, raw.view, comp.view, target.view, empty, empty, false, false, c.ds4_metal_attention_get_facts(), &fused)) return 0;
    return command.finish("gathered KV staging");
}

fn fillZero(command: *const Command, mask: View, count: u32, facts: Facts) bool {
    if (mask.buffer == null or count == 0) return false;
    var args = std.mem.zeroes(c.ds4_metal_attention_unary_args);
    args.ne00 = narrow(count);
    args.ne01 = 1;
    args.ne02 = 1;
    args.ne03 = 1;
    args.ne0 = narrow(count);
    args.ne1 = 1;
    args.ne2 = 1;
    args.ne3 = 1;
    args.nb00 = 4;
    args.nb0 = 4;
    args.nb01 = @as(u64, count) * 4;
    args.nb02 = args.nb01;
    args.nb03 = args.nb01;
    args.nb1 = args.nb01;
    args.nb2 = args.nb01;
    args.nb3 = args.nb01;
    var threads: u64 = @min(@as(u64, count), facts.fill_threads, @as(u64, 256));
    if (threads == 0) threads = 1;
    return c.ds4_metal_attention_fill(command.token.?, &args, mask, (@as(u64, count) + threads - 1) / threads, threads) != 0;
}
fn padArgs(keys: u32, width: u32) PadArgs {
    const half_row = @as(u64, width) * 2;
    const cache = @as(u64, keys) * half_row;
    const mask = @as(u64, keys) * 2;
    return .{ .ne11 = narrow(keys), .ne_12_2 = 1, .ne_12_3 = 1, .nb11 = half_row, .nb12 = cache, .nb13 = cache, .nb21 = half_row, .nb22 = cache, .nb23 = cache, .ne31 = 1, .ne32 = 1, .ne33 = 1, .nb31 = mask, .nb32 = mask, .nb33 = mask };
}
fn vecArgs(keys: u32, heads: u32, width: u32) VecArgs {
    @setFloatMode(.optimized); // Same F32 inverse-sqrt scale as the C TU.
    const row = @as(u64, width) * 4;
    const half_row = @as(u64, width) * 2;
    const cache = @as(u64, keys) * half_row;
    const mask = @as(u64, keys) * 2;
    return .{
        .ne01 = 1,
        .ne02 = narrow(heads),
        .ne03 = 1,
        .nb01 = @as(u64, heads) * row,
        .nb02 = row,
        .nb03 = @as(u64, heads) * row,
        .ne11 = narrow(keys),
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = narrow(width),
        .nb11 = half_row,
        .nb12 = cache,
        .nb13 = cache,
        .ns20 = narrow(width),
        .nb21 = half_row,
        .nb22 = cache,
        .nb23 = cache,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask,
        .nb32 = mask,
        .nb33 = mask,
        .ne1 = narrow(heads),
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(width))),
        .max_bias = 0,
        .m0 = 0,
        .m1 = 0,
        .n_head_log2 = 0,
        .logit_softcap = 0,
    };
}
fn simdgroups(keys: u32, groups: u32) u32 {
    var count: u32 = 1;
    while (2 * groups * count * 32 < keys and count < 4) count *= 2;
    return count;
}
fn packedFeature(facts: Facts) bool {
    const old = fact(facts, c.DS4_ATTN_PRE_M5);
    if (old and (present("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") or present("DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PACKED32_REDUCE"))) return false;
    return old or fact(facts, c.DS4_ATTN_M5);
}
fn pipeline(kind: u32, width: u32, nsg: u32, nwg: u32, padded: bool, shared: bool) Pipeline {
    return c.ds4_metal_attention_get_pipeline(kind, width, nsg, nwg, padded, shared);
}

pub const Request = struct {
    heads: ?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    sinks_offset: u64,
    q: ?*const Tensor,
    raw: ?*const Tensor,
    raw_rows: u32,
    raw_cap: u32,
    raw_start: u32,
    comp: ?*const Tensor,
    comp_half: u32,
    comp_rows: u32,
    mask: ?*const Tensor,
    use_mask: u32,
    n_head: u32,
    width: u32,
};
fn attend(command: *const Command, request: Request, q: View, raw: View, comp: View, heads: View, mask_input: View, sinks: View, facts: Facts) bool {
    // The public wrapper validates all byte extents first; this inner kernel
    // family additionally accepts only the fixed 512-element head dimension.
    if (request.width != 512) return false;
    const gathered = request.comp_rows != 0;
    const keys = request.raw_rows + request.comp_rows;
    var nwg: u32 = 32;
    if (gathered) {
        if (nwg_environment < 0) {
            const value = c.getenv("DS4_METAL_FLASH_NWG");
            nwg_environment = if (value == null) 0 else c.atoi(value);
        }
        // Keep 32 by default: reducing short-context groups saved 1 us here but
        // made the following concurrent FFN block about 50 us slower in testing.
        if (nwg_environment > 0) nwg = @intCast(nwg_environment);
        nwg = @min(@max(nwg, @as(u32, 1)), @as(u32, 32));
    }
    const nsg = simdgroups(keys, nwg);
    const rope = &deferred_rope.args;
    const packed_cache = request.use_mask == 0 and request.comp_half != 0 and request.comp_rows != 0 and keys <= 1024;
    const packed_geometry = request.n_head == 64 and request.width == 512 and nsg == 1 and nwg == 32;
    const compatible_rope = deferred_rope.pending and rope.head_dim == 512 and rope.n_dims == 64 and rope.row_bytes == 2048 and rope.inverse != 0;
    const packed_shape = gathered and packedFeature(facts) and !fact(facts, c.DS4_ATTN_QUALITY) and packed_cache and packed_geometry and compatible_rope;
    var persistent = false;
    if (gathered) {
        const m5 = fact(facts, c.DS4_ATTN_M5) and !present("DS4_METAL_DISABLE_M5_PERSISTENT_ZERO_ATTN_MASK") and (!packed_shape or !present("DS4_METAL_DISABLE_M5_PACKED_ZERO_MASK"));
        persistent = request.use_mask == 0 and (fact(facts, c.DS4_ATTN_M3) or m5) and !present("DS4_METAL_DISABLE_PERSISTENT_ZERO_ATTN_MASK");
    } else persistent = fact(facts, c.DS4_ATTN_PRE_M5) and !present("DS4_METAL_DISABLE_PRE_M5_DECODE_RAW_ZERO_ATTN_MASK") and !fact(facts, c.DS4_ATTN_QUALITY) and !fact(facts, c.DS4_ATTN_SSD) and fact(facts, c.DS4_ATTN_TP_SINGLE);
    const mask_bytes = @as(u64, keys) * 2;
    var mask = if (persistent) Buffer.scratch(c.DS4_ATTN_ZERO_MASK, mask_bytes, "ds4_flash_attn_zero_mask") else Buffer{ .view = std.mem.zeroes(View) };
    defer mask.deinit();
    // Raw-only zero-mask allocation failure falls back to a transient mask;
    // gathered's persistent-mask failure is fatal to this encode, as in C.
    if (!gathered and !mask.valid()) {
        mask = Buffer.scratch(c.DS4_ATTN_TRANSIENT_MASK, mask_bytes, "ds4_flash_attn_mask");
        if (mask.valid()) {
            const contents = c.ds4_metal_attention_contents(mask.view).?;
            @memset(@as([*]u8, @ptrCast(contents))[0..@intCast(mask_bytes)], 0);
        }
    } else if (gathered and !persistent) mask = Buffer.scratch(c.DS4_ATTN_MASK, mask_bytes, "ds4_flash_attn_mask");
    if (!mask.valid()) return false;
    const half_row = @as(u64, request.width) * 2;
    const kv = Buffer.scratch(c.DS4_ATTN_KV, @as(u64, keys) * half_row, if (gathered) "ds4_flash_attn_kv" else "ds4_flash_attn_kv_f16");
    defer kv.deinit();
    if (!kv.valid()) return false;
    const pad = Buffer.scratch(c.DS4_ATTN_PAD, 2 * 32 * half_row + 32 * 2, "ds4_flash_attn_pad");
    defer pad.deinit();
    if (!pad.valid()) return false;
    const tmp_bytes = @as(u64, request.n_head) * request.width * nwg * 4 + @as(u64, request.n_head) * (2 * nwg) * 4;
    const tmp = Buffer.scratch(c.DS4_ATTN_TMP, tmp_bytes, "ds4_flash_attn_tmp");
    defer tmp.deinit();
    if (!tmp.valid()) return false;
    const has_pad = keys % 32 != 0;
    const shared_pad = gathered and has_pad and fact(facts, c.DS4_ATTN_M3) and !present("DS4_METAL_DISABLE_SHARED_KV_PAD");
    const empty_pipeline = std.mem.zeroes(Pipeline);
    var pad_pso = if (!gathered and has_pad) pipeline(c.DS4_ATTN_PAD_PSO, request.width, nsg, nwg, has_pad, false) else empty_pipeline;
    if (!gathered and has_pad and pad_pso.handle == null) return false;
    const packed_threads: u64 = 8 * 32;
    const packed_shared: u64 = 512 * 2 + 8 * 128 * 2 + (32 * 32 + 2 * 32 + 32) * 4 + 32 * 33 * 4 * 4;
    const packed_requested = packed_shape and !present("DS4_METAL_DISABLE_M5_FLASH_ATTN_PACKED32_REDUCE");
    const packed_pso = if (packed_requested) pipeline(c.DS4_ATTN_PACKED_PSO, request.width, 1, 32, has_pad, shared_pad) else empty_pipeline;
    const use_packed = packed_pso.handle != null and packed_pso.simd_width == 32 and packed_pso.threads >= packed_threads and (facts.max_shared_bytes == 0 or facts.max_shared_bytes >= packed_shared);
    if (packed_requested and present("DS4_METAL_TRACE_M5_FLASH_ATTN_PACKED32_REDUCE")) _ = c.fprintf(c.__stderrp, "ds4: packed FA use=%d max_threads=%lu tew=%lu tgmem=%lu need=%lu\n", @as(c_int, @intFromBool(use_packed)), @as(c_ulong, packed_pso.threads), @as(c_ulong, packed_pso.simd_width), @as(c_ulong, facts.max_shared_bytes), @as(c_ulong, packed_shared));
    const vec_pso = if (!use_packed) pipeline(c.DS4_ATTN_VEC_PSO, request.width, nsg, nwg, has_pad, shared_pad) else empty_pipeline;
    var reduce_pso = if (!use_packed) pipeline(c.DS4_ATTN_REDUCE_PSO, request.width, nsg, nwg, has_pad, shared_pad) else empty_pipeline;
    if (!use_packed and (vec_pso.handle == null or reduce_pso.handle == null)) return false;
    var pad_fused = false;
    if (gathered) {
        if (!persistent and !fillZero(command, mask.view, keys, facts)) return false;
        if (request.use_mask != 0 and request.comp_rows != 0 and !copyHalf(command, mask_input, at(mask.view, @as(u64, request.raw_rows) * 2), false, request.comp_rows, facts)) return false;
        if (!stage(command, .{ .raw_cap = request.raw_cap, .raw_start = request.raw_start, .raw_rows = request.raw_rows, .comp_rows = request.comp_rows, .width = request.width, .comp_half = request.comp_half != 0 }, raw, comp, kv.view, mask.view, pad.view, has_pad, shared_pad, facts, &pad_fused)) return false;
        if (has_pad and !pad_fused) {
            pad_pso = pipeline(c.DS4_ATTN_PAD_PSO, request.width, nsg, nwg, has_pad, shared_pad);
            if (pad_pso.handle == null) return false;
        }
    } else if (!ringCopy(command, raw, request.raw_cap, request.raw_start, request.raw_rows, request.width, kv.view, facts)) return false;
    if (has_pad and !pad_fused) {
        const args = padArgs(keys, request.width);
        if (c.ds4_metal_attention_pad(command.token.?, pad_pso, &args, kv.view, mask.view, pad.view) == 0) return false;
    }
    const args = vecArgs(keys, request.n_head, request.width);
    if (use_packed) {
        if (c.ds4_metal_attention_vec(command.token.?, packed_pso, &args, q, kv.view, mask.view, sinks, pad.view, heads, rope, request.n_head, 1, 1, packed_threads, 1, packed_shared) == 0) return false;
        deferred_rope.pending = false;
        deferred_rope.used = true;
        return true;
    }
    const shared_elems = (alignUp(request.width, 128) + 4 * 32 + 2 * alignUp(request.width, 128)) * nsg;
    const shared_bytes = alignUp(shared_elems * 2, 16);
    if (c.ds4_metal_attention_vec(command.token.?, vec_pso, &args, q, kv.view, mask.view, sinks, pad.view, tmp.view, null, 1, request.n_head, nwg, 32, nsg, shared_bytes) == 0) return false;
    // Raw-only never consumes a deferred inverse tail. Gathered consumes at
    // the reduction boundary even if the fused PSO subsequently fails lookup.
    const fuse_rope = gathered and deferred_rope.pending;
    if (gathered) deferred_rope.pending = false;
    if (fuse_rope) {
        deferred_rope.used = true;
        reduce_pso = pipeline(c.DS4_ATTN_ROPE_PSO, request.width, nsg, nwg, has_pad, shared_pad);
        if (reduce_pso.handle == null) return false;
    }
    const reduction = c.ds4_metal_attention_reduce_args{ .nrows = narrow(request.n_head) };
    return c.ds4_metal_attention_reduce(command.token.?, reduce_pso, &reduction, tmp.view, heads, if (fuse_rope) rope else null, request.n_head, 32 * nwg) != 0;
}

pub fn run(request: Request) c_int {
    if (api.init() == 0) return 0;
    if (request.heads == null or request.map == null or request.q == null or request.raw == null) return 0;
    if (request.raw_rows == 0 or request.n_head == 0 or request.width == 0 or request.raw_cap < request.raw_rows or request.raw_start >= request.raw_cap) return 0;
    if (request.raw_rows > max_u32 - request.comp_rows or request.raw_rows + request.comp_rows > 8192) return 0;
    if ((request.comp_rows != 0 and request.comp == null) or (request.use_mask != 0 and request.mask == null)) return 0;
    const sink_bytes = @as(u64, request.n_head) * 4;
    if (request.sinks_offset > request.size or sink_bytes > request.size - request.sinks_offset) return api.reportError("ds4: Metal graph attention heads sink range is outside the mapped model");
    const q = Buffer.tensor(request.q);
    defer q.deinit();
    const raw = Buffer.tensor(request.raw);
    defer raw.deinit();
    const comp = if (request.comp_rows != 0) Buffer.tensor(request.comp) else Buffer{ .view = std.mem.zeroes(View) };
    defer comp.deinit();
    const mask = if (request.use_mask != 0) Buffer.tensor(request.mask) else Buffer{ .view = std.mem.zeroes(View) };
    defer mask.deinit();
    const heads = Buffer.tensor(request.heads);
    defer heads.deinit();
    const query_bytes = @as(u64, request.n_head) *% request.width *% 4;
    const compressed_bytes = @as(u64, request.comp_rows) *% request.width *% (if (request.comp_half != 0) @as(u64, 2) else 4);
    if (!q.fits(query_bytes) or !raw.fits(@as(u64, request.raw_cap) *% request.width *% 4) or !heads.fits(query_bytes) or (request.comp_rows != 0 and !comp.fits(compressed_bytes)) or (request.use_mask != 0 and !mask.fits(@as(u64, request.comp_rows) * 4))) return api.reportError("ds4: Metal graph attention heads received undersized buffers");
    var inner: u64 = 0;
    const sinks = api.retainModelRange(request.map, request.size, request.sinks_offset, sink_bytes, &inner) orelse return 0;
    defer api.releaseModelBuffer(sinks);
    if (request.comp_rows == 0 and c.ds4_gpu_kv_norm_task_pending() != 0 and c.ds4_gpu_kv_norm_task_flush() == 0) return 0;
    var command = Command.init() orelse return 0;
    defer command.deinit();
    const sinks_view = View{ .buffer = sinks, .offset = inner, .bytes = sink_bytes };
    if (!attend(&command, request, q.view, raw.view, comp.view, heads.view, mask.view, sinks_view, c.ds4_metal_attention_get_facts())) return 0;
    if (request.comp_rows != 0 and c.ds4_gpu_kv_norm_task_pending() != 0) {
        if (!warned_unconsumed) {
            warned_unconsumed = true;
            _ = api.reportError("ds4: deferred kv norm task was not consumed by the attention path");
        }
        if (c.ds4_gpu_kv_norm_task_flush() == 0) return 0;
    }
    return command.finish(if (request.comp_rows == 0) "graph raw attention heads" else "graph attention heads");
}
