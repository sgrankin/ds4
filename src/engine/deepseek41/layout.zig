//! V4.1 resource dimensions and admission estimate. All widths come from this
//! graph's validated config; environment switches choose the existing layout.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
pub const Row = c.ds41_prefill_row;
pub const RowField = std.meta.FieldEnum(Row);
pub const storage_fields = [_]RowField{
    .residual,      .pre,        .selected_comp, .selected,      .after_attn,   .ffn_split,
    .attn_split,    .qr,         .q,             .kv,            .latent,       .index_k,
    .pool_kv,       .pool_score, .index_q,       .index_weights, .heads,        .low,
    .block,         .mix,        .x,             .norm,          .route_logits, .route_probs,
    .route_weights, .mid,        .block_mask,    .engram_rows,
};
pub const alias_fields = [_]RowField{
    .engram_kv, .flat_norm,  .experts, .gate,   .up, .shared_gate,
    .shared_up, .shared_mid, .shared,  .routed,
};
pub const carry_fields = [_]RowField{ .residual, .pre, .ffn_split, .selected_comp, .block_mask };
pub const Scratch = enum {
    image_text_mask,
    residual,
    after_attn,
    flat_norm,
    mix,
    attn_split,
    ffn_split,
    pre,
    x,
    norm,
    block,
    qr,
    q,
    kv,
    latent,
    raw_prefill,
    pool_kv,
    pool_score,
    index_q,
    index_k,
    index_weights,
    index_scores,
    selected_comp,
    index_packed,
    selected_kv,
    block_scores,
    block_selected,
    block_mask,
    heads,
    low,
    route_logits,
    route_probs,
    selected,
    route_weights,
    gate,
    up,
    mid,
    experts,
    routed,
    shared_gate,
    shared_up,
    shared_mid,
    shared,
    engram_rows,
    engram_prefetch,
    engram_kv,
    logits,
};

pub fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn compressedF16() bool {
    const value = c.getenv("DS4_V41_COMPRESSED_KV_F16");
    return value != null and c.strcmp(value, "1") == 0;
}
pub fn ceiling(context: u32, chunk: u32) u32 {
    // An explicit chunk bounds scratch independently of context. Preserve the
    // default 2K/4K/8K choices and the supported 8192-row ceiling.
    if (chunk != 0) return @min(context, @min(chunk, 8192));
    const limit: u32 = if (context < 8192 or present("DS4_METAL_DISABLE_V41_WIDE_CHUNK")) 2048 else if (context < 16384 or present("DS4_METAL_DISABLE_V41_8K_CHUNK")) 4096 else 8192;
    return @min(context, limit);
}
pub fn prefillLimit(context: u32, chunk: u32) u32 {
    const cap = ceiling(context, chunk);
    return if (chunk == 0 and present("DS4_METAL_V41_DEMAND_WORKSPACE") and cap > 2048) 2048 else cap;
}

pub const Layout = struct {
    shape: c.ds4_shape,
    context: u32,
    prefill: u32,
    maximum: u32,
    carry: u32,
    alias: bool,
    compact: bool,
    compressed_f16: bool,

    pub fn init(config: *const Config, context: u32, chunk: u32) Layout {
        var plan = Layout{
            .shape = config.shape,
            .context = context,
            .prefill = prefillLimit(context, chunk),
            .maximum = ceiling(context, chunk),
            .carry = 0,
            .alias = !present("DS4_METAL_DISABLE_V41_PREFILL_ALIAS"),
            .compact = !present("DS4_METAL_DISABLE_V41_COMPACT_CARRY"),
            .compressed_f16 = compressedF16(),
        };
        var words: u64 = 0;
        inline for (carry_fields) |field| words += plan.carryWidth(field);
        var cap = (@as(u64, 3) << 30) / (words * 4);
        cap = @min(cap, 32768);
        cap = @min(cap, context);
        // Causal sweep boundaries are independent of encoder tile size.
        cap -= cap % 2048;
        plan.carry = if (plan.prefill != 0 and cap > plan.prefill) @intCast(cap) else 0;
        return plan;
    }

    pub fn fromRecord(config: *const Config, graph: *const c.ds41_gpu_graph) Layout {
        return .{
            .shape = config.shape,
            .context = graph.ctx,
            .prefill = graph.prefill_cap,
            .maximum = graph.prefill_max,
            .carry = graph.carry_cap,
            .alias = graph.prefill_alias,
            .compact = graph.compact_carry,
            .compressed_f16 = graph.compressed_kv_f16,
        };
    }

    pub fn rowWidth(self: *const Layout, field: RowField) u64 {
        const s = &self.shape;
        return switch (field) {
            .residual, .after_attn, .flat_norm => @as(u64, s.n_hc) * s.n_embd,
            .pre => s.n_hc,
            .selected_comp => s.n_indexer_top_k,
            .selected, .route_weights => s.n_expert_used,
            .ffn_split, .attn_split, .mix => 24,
            .qr => s.n_lora_q,
            .q, .heads => @as(u64, s.n_head) * s.n_head_dim,
            .kv, .latent, .pool_kv, .pool_score => s.n_head_dim,
            .index_k => s.n_indexer_head_dim,
            .index_q => @as(u64, s.n_indexer_head) * s.n_indexer_head_dim,
            .index_weights => s.n_indexer_head,
            .low => @as(u64, s.n_out_group) * s.n_lora_o,
            .block, .x, .norm, .shared, .routed => s.n_embd,
            .route_logits, .route_probs => s.n_expert,
            .mid, .gate, .up => @as(u64, s.n_expert_used) * s.n_ff_exp,
            .block_mask => (@as(u64, self.context) + 7) / 8,
            .engram_rows => c.DS4_ENGRAM_COLS * c.DS4_ENGRAM_DIM,
            .engram_kv => (@as(u64, s.n_hc) + 1) * s.n_embd,
            .experts => @as(u64, s.n_expert_used) * s.n_embd,
            .shared_gate, .shared_up, .shared_mid => s.n_ff_exp,
        };
    }

    pub fn scratchWidth(self: *const Layout, comptime field: Scratch) u64 {
        return switch (field) {
            .image_text_mask => (@as(u64, self.prefill) + 3) / 4,
            .pre => 4,
            .raw_prefill => (@as(u64, self.prefill) + 128) * self.shape.n_head_dim,
            .index_scores => @as(u64, 32) * self.context,
            .index_packed => c.ds4_gpu_dsv41_indexer_packed_bytes(self.context, self.prefill) / 4,
            .selected_kv => @as(u64, self.shape.n_indexer_top_k) * self.shape.n_head_dim,
            .block_scores => (@as(u64, self.context) + 7) / 8,
            .block_selected => 2048,
            .engram_prefetch => @as(u64, if (self.carry != 0) self.carry else self.prefill) * c.DS4_ENGRAM_COLS * c.DS4_ENGRAM_DIM,
            .logits => self.shape.n_vocab,
            else => self.rowWidth(@field(RowField, @tagName(field))),
        };
    }

    pub fn carryWidth(self: *const Layout, comptime field: RowField) u64 {
        const width = self.rowWidth(field);
        if (!self.compact) return width;
        return switch (field) {
            .residual => (width + 1) / 2, // Two BF16 values per F32-sized word.
            .block_mask => (width + 31) / 32,
            else => width,
        };
    }

    pub fn bytes(self: *const Layout) u64 {
        var floats: u64 = 40 * 128 * 512;
        for (0..4) |owner| {
            const rows = @as(u64, self.context) / @as(u64, if (owner < 3) 2 else 1) + 1;
            floats += rows * (@as(u64, if (self.compressed_f16) 256 else 512) + 128) + 2 * 512;
        }
        floats += @as(u64, 2) * 2 * self.shape.n_embd * self.shape.n_hc;
        inline for (comptime std.meta.fieldNames(Scratch)) |name| floats += self.scratchWidth(@field(Scratch, name));
        inline for (storage_fields) |field| floats += self.rowWidth(field) * self.prefill;
        if (!self.alias) {
            inline for (alias_fields) |field| floats += self.rowWidth(field) * self.prefill;
        }
        inline for (carry_fields) |field| floats += self.carryWidth(field) * self.carry;
        // Reserve view objects and the bounded host hash-ID array as before.
        floats += @as(u64, self.prefill) * 512;
        if (self.carry > self.prefill) floats += @as(u64, self.carry - self.prefill) * 2 * c.DS4_ENGRAM_COLS;
        // Expert-major activation packing and index sort buffers are backend
        // allocations outside this record, but still belong in admission.
        const packed_workspace: u64 = if (self.prefill >= 512) @as(u64, if (self.prefill > 4096) 512 else 256) * 1024 * 1024 else 0;
        const sort = @as(u64, 32) * self.context * 2 * @sizeOf(u32);
        return floats * 4 + @sizeOf(c.ds41_gpu_graph) + @as(u64, self.shape.n_vocab) * 4 + packed_workspace + sort;
    }

    pub fn memory(self: *const Layout) c.ds4_context_memory {
        var result = std.mem.zeroes(c.ds4_context_memory);
        result.prefill_cap = self.prefill;
        result.raw_cap = 128;
        result.comp_cap = self.context + 1;
        result.raw_bytes = 40 * 128 * 512 * 4;
        for (0..4) |owner| {
            const rows = @as(u64, self.context) / @as(u64, if (owner < 3) 2 else 1) + 1;
            result.compressed_bytes += rows * (@as(u64, 512) * @as(u64, if (self.compressed_f16) 2 else 4) + 128 * 4);
        }
        result.total_bytes = self.bytes();
        result.scratch_bytes = result.total_bytes - result.raw_bytes - result.compressed_bytes;
        return result;
    }
};
