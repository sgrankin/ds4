//! Sole C facade; the pure owner owns deferred-RoPE state and scalar attention.
const c = @import("metal_c");
const owner = @import("attention_decode.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_set_decode_attn_rope_fuse(width: u32, dimensions: u32, position: u32, original_context: u32, inverse: bool, base: f32, scale: f32, ext: f32, attention: f32, fast: f32, slow: f32) void {
    owner.setRope(width, dimensions, position, original_context, inverse, base, scale, ext, attention, fast, slow);
}
export fn ds4_gpu_decode_attn_rope_fuse_used() c_int {
    return owner.ropeUsed();
}
export fn ds4_gpu_flash_kv_stage_f16_tensor(dst: ?*Tensor, raw: ?*const Tensor, raw_cap: u32, raw_start: u32, raw_rows: u32, comp: ?*const Tensor, comp_half: u32, comp_rows: u32, width: u32) c_int {
    return owner.stageTensor(dst, raw, raw_cap, raw_start, raw_rows, comp, comp_half, comp_rows, width);
}
export fn ds4_gpu_attention_decode_heads_tensor(heads: ?*Tensor, map: ?*const anyopaque, size: u64, sinks_offset: u64, q: ?*const Tensor, raw: ?*const Tensor, raw_rows: u32, raw_cap: u32, raw_start: u32, comp: ?*const Tensor, comp_half: u32, comp_rows: u32, mask: ?*const Tensor, use_mask: u32, n_head: u32, width: u32) c_int {
    return owner.run(.{
        .heads = heads,
        .map = map,
        .size = size,
        .sinks_offset = sinks_offset,
        .q = q,
        .raw = raw,
        .raw_rows = raw_rows,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .comp = comp,
        .comp_half = comp_half,
        .comp_rows = comp_rows,
        .mask = mask,
        .use_mask = use_mask,
        .n_head = n_head,
        .width = width,
    });
}

export fn ds4_gpu_dsv41_heads_once_arm(table: ?*const Tensor, compressed: bool) void {
    owner.headsArm(table, compressed);
}
export fn ds4_gpu_dsv41_heads_once_take() c_int {
    return owner.headsTake();
}

export fn ds4_gpu_dsv41_attention_decode(heads: ?*Tensor, map: ?*const anyopaque, size: u64, sinks: u64, q: ?*const Tensor, raw: ?*const Tensor, raw_rows: u32, raw_cap: u32, raw_start: u32, cache: ?*const Tensor, ids: ?*const Tensor, selected: ?*Tensor, cache_rows: u32, attended: u32, n_head: u32, width: u32) c_int {
    _ = selected;
    if (width != 512 or attended == 0 or attended > cache_rows or c.ds4_gpu_tensor_bytes(cache) < @as(u64, cache_rows) * width * 4 or c.ds4_gpu_tensor_bytes(ids) < @as(u64, attended) * 4 or c.ds4_metal_attention_get_facts().flags & c.DS4_ATTN_QUALITY != 0 or c.getenv("DS4_METAL_DISABLE_GATHERED_KV_STAGE") != null) return 0;
    return owner.run(.{ .heads = heads, .map = map, .size = size, .sinks_offset = sinks, .q = q, .raw = raw, .raw_rows = raw_rows, .raw_cap = raw_cap, .raw_start = raw_start, .comp = cache, .comp_half = 0, .comp_rows = attended, .mask = null, .use_mask = 0, .n_head = n_head, .width = width, .ids = ids });
}
export fn ds4_gpu_dsv41_flash_attn_reduce_check(heads: ?*Tensor, tmp: ?*const Tensor, rows: u32, transform: c_int, tab: ?*const Tensor, compressed: bool) c_int {
    return owner.reduceCheck(heads, tmp, rows, transform != 0, tab, compressed);
}
