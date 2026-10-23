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
