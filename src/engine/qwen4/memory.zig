//! Conservative placement estimate, preserving the released Qwen formula.
//! This prices live recurrence plus both MTP snapshots even for tiny contexts;
//! it is admission accounting, not a report of lazily allocated tensor bytes.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");

pub fn prefillChunk(context: u32) u32 {
    const raw = c.getenv("DS4_QWEN4_PREFILL_CHUNK");
    const requested = if (raw != null and raw[0] != 0) c.strtoul(raw, null, 10) else 8192;
    const chunk: u32 = if (requested == 0 or requested > 65536) 8192 else @intCast(requested);
    return @min(context, chunk);
}

pub fn estimate(config: *const dimensions.Config, context: u32, prefill_chunk: u32) c.ds4_context_memory {
    const shape = &config.shape;
    const rows: u64 = if (prefill_chunk != 0) prefill_chunk else prefillChunk(context);
    const blocks: u64 = context / 4 + 1;
    const width: u64 = shape.n_embd;
    const hc_width = width * shape.n_hc;
    const kv_row = @as(u64, 2) * shape.n_head_kv * shape.n_head_dim * 2;
    var attention_layers: u32 = 0;
    for (0..shape.n_layer) |layer| attention_layers += @intFromBool(!dimensions.isLinear(config, @intCast(layer)));
    var result = std.mem.zeroes(c.ds4_context_memory);
    result.prefill_cap = @intCast(rows);
    result.raw_cap = context;
    result.comp_cap = @intCast(blocks);
    // F16 K/V, raw F32 indexer keys, pooled F16 block keys and mRoPE positions.
    result.raw_bytes = @as(u64, attention_layers) * context * kv_row + @as(u64, context) * 16;
    result.compressed_bytes = @as(u64, attention_layers) * (context *% shape.n_indexer_head_dim *% 4 + blocks * shape.n_indexer_head_dim * 2);
    result.scratch_bytes = rows * (12 * hc_width + 24 * width + @as(u64, shape.n_expert_used + 1) * (width + 2 * shape.n_ff_exp) + 2 * blocks) * 4;
    result.scratch_bytes += @as(u64, 3) * (shape.n_layer - attention_layers) *
        (dimensions.stateFloats(shape) + dimensions.historyFloats(shape)) * @sizeOf(f32);
    result.total_bytes = result.raw_bytes + result.compressed_bytes + result.scratch_bytes;
    return result;
}
