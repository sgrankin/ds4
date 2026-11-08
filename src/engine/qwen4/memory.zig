//! Conservative placement estimate, preserving the released Qwen formula.
//! This prices live recurrence plus both MTP snapshots even for tiny contexts;
//! it is admission accounting, not a report of lazily allocated tensor bytes.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");

pub fn prefillChunk(context: u32) u32 {
    return resolveChunk(context, 0);
}

pub fn resolveChunk(context: u32, explicit: u32) u32 {
    if (explicit != 0) return @min(context, explicit);
    const raw = c.getenv("DS4_QWEN4_PREFILL_CHUNK");
    const requested = if (raw != null and raw[0] != 0) c.strtoul(raw, null, 10) else 8192;
    const chunk: u32 = if (requested == 0 or requested > 65536) 8192 else @intCast(requested);
    return @min(context, chunk);
}

pub fn estimate(config: *const dimensions.Config, context: u32, prefill_chunk: u32) c.ds4_context_memory {
    const shape = &config.shape;
    const rows: u64 = resolveChunk(context, prefill_chunk);
    const blocks: u64 = context / 4 + 1;
    const width: u64 = shape.n_embd;
    const hc: u64 = shape.n_hc;
    const hc_width = width * hc;
    const kv_row = @as(u64, 2) * shape.n_head_kv * shape.n_head_dim * 2;
    const conv = dimensions.convWidth(shape);
    const v_dim = @as(u64, shape.n_lin_v_head) * shape.n_lin_head_dim;
    const q_dim = @as(u64, shape.n_head) * shape.n_head_dim;
    const kv_dim = @as(u64, shape.n_head_kv) * shape.n_head_dim;
    const iq_dim = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
    const k_blocks: u64 = shape.n_indexer_top_k / 4;
    const f: u64 = shape.n_ff_exp;
    const used: u64 = shape.n_expert_used;
    const mtp = shape.n_nextn_predict != 0;
    var n_attn: u64 = 0;
    for (0..shape.n_layer) |layer| n_attn += @intFromBool(!dimensions.isLinear(config, @intCast(layer)));
    const n_linear = shape.n_layer - n_attn;
    var result = std.mem.zeroes(c.ds4_context_memory);
    result.prefill_cap = @intCast(rows);
    result.raw_cap = context;
    result.comp_cap = @intCast(blocks);
    result.raw_bytes = n_attn * context * kv_row + @as(u64, context) * 16;
    result.compressed_bytes = n_attn * (@as(u64, context) * shape.n_indexer_head_dim * 4 + blocks * shape.n_indexer_head_dim * 2);
    const old_row = 12 * hc_width + 24 * width + (used + 1) * (width + 2 * f) + 2 * blocks;
    const gpu_row = 6 * hc_width + 2 * shape.n_hc_lowrank + 16 * hc * hc +
        5 * width + conv + 2 * v_dim + 2 * shape.n_lin_v_head + 5 * q_dim +
        2 * kv_dim + 2 * iq_dim + shape.n_indexer_head_dim + blocks + blocks / 8 +
        5 * k_blocks + 7 + 2 * shape.n_expert + 2 * used + (used + 1) * (f + width) + 3 * f;
    const state = n_linear * (v_dim * shape.n_lin_head_dim + (shape.n_lin_conv - 1) * conv) + dimensions.pleHistoryFloats(shape);
    const attn_part = 3 * @as(u64, shape.n_head) * 64 * (2 + shape.n_head_dim);
    var fixed = @as(u64, if (mtp) 4 else 1) * state + attn_part + shape.n_expert + @as(u64, if (mtp) 12 else 4) * shape.n_vocab;
    if (mtp) fixed += 12 * (hc + 1) * width + 4 * hc_width + 1 + 2 * ((@as(u64, shape.n_vocab) + 4095) / 4096);
    // Half copies reserve float-word equivalents. CPU staging and all three
    // rollback snapshots are counted even before lazy allocation.
    const reserved = gpu_row + (width + used * f + 1) / 2;
    result.scratch_bytes = (rows * (@max(reserved, old_row) + hc_width + 4) + fixed) * 4;
    result.total_bytes = result.raw_bytes + result.compressed_bytes + result.scratch_bytes;
    return result;
}
